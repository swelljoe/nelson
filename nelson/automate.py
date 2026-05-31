"""P6 automation loop: one idempotent, unattended benchmark pass.

The phases P1-P5 each do one job — build the corpus, run a competitor, score a
run, rank the field. P6 is the *driver* that ties them into a single pass that a
cron entry (or ``--interval``) can fire repeatedly: optionally refresh the
corpus, age out vulns that are no longer 0-day to anyone, sync the competitor
roster, plan the missing cells of the (competitor x case x file) matrix, run
them, score what completed, regenerate the leaderboard, and surface anything
that needs a human (auth expiry, infra failures).

It owns **no scoring or detection logic** — that all lives in the phase engines,
which are injected so the whole pass runs under test with fakes (no podman, no
network, no judge). What lives here is *policy*: which cells to run, when a case
has aged out, and the unattended-safety rails (budget/run caps + an auth circuit
breaker so a dead host session can't burn an hour failing identically).

Integrity rule carried through: ``auth_failed`` / ``infra_error`` runs are
counted and surfaced as alerts, never as misses; age-out only retires a case
when the data *proves* it is stale (never on missing dates), and only flips
status — it never deletes ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

import yaml

from .corpus import Case
from .runner import Competitor, vulnerable_files

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    from .corpus import CorpusPipeline, SeedSource
    from .db import Database
    from .runner import RunResult
    from .score import Scorer


class Runner(Protocol):
    """The slice of :class:`~nelson.runner.BenchRunner` the loop drives.

    Typed structurally (not as the concrete class) so any runtime that can point
    a competitor at a file plugs in — the pluggable boundary P7 grows into — and
    so the pass is testable with a fake runner.
    """

    def run_case(
        self, case: Case, competitor: Competitor, target_file: str
    ) -> RunResult: ...


# Run statuses that mean a cell already has a settled or in-flight result, so the
# matrix planner should not schedule it again. auth_failed / infra_error are
# *not* here: those got no fair look, so a cell that only ever failed that way is
# re-runnable (with --retry-failed) without violating the integrity rule.
_SETTLED_STATUSES = frozenset({"complete", "pending", "running"})
_RETRYABLE_STATUSES = frozenset({"auth_failed", "infra_error"})


# -- Dates / recency ---------------------------------------------------------


def parse_date_prefix(value: str | None) -> date | None:
    """Parse a loose date into a ``date``, or None if it isn't one.

    Tolerates the shapes these fields actually carry: a full ISO timestamp
    (``2026-01-15T09:00:00Z``), a plain date (``2026-01-15``), a month
    (``2025-01`` -> first of the month), or a year (``2025`` -> Jan 1). Month/
    year are padded to the first day so a disclosure mid-month reads as *after* a
    month-granular cutoff.
    """
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    head = text[:10]
    parts = head.split("-")
    try:
        year = int(parts[0])
        month = int(parts[1]) if len(parts) > 1 and parts[1] else 1
        day = int(parts[2]) if len(parts) > 2 and parts[2] else 1
        return date(year, month, day)
    except (ValueError, IndexError):
        return None


def case_is_fresh_for(case: Case, competitor: Competitor) -> bool:
    """Is ``case`` plausibly 0-day to ``competitor`` (disclosed after its cutoff)?

    A case is fresh when its disclosure is strictly after the competitor's
    knowledge cutoff. If either date is missing or unparseable we default to
    **fresh** (True): the recency filter only *excludes* a cell when we can prove
    the model was likely trained on the bug — never silently on missing data.
    """
    disclosure = parse_date_prefix(case.disclosure_date)
    cutoff = parse_date_prefix(competitor.knowledge_cutoff)
    if disclosure is None or cutoff is None:
        return True
    return disclosure > cutoff


# -- Matrix planning ---------------------------------------------------------


@dataclass(frozen=True)
class MatrixCell:
    """One unit of work: point ``competitor`` at ``target_file`` of ``case``."""

    competitor: Competitor
    case: Case
    target_file: str


RunStatusMap = dict[tuple[str, str, str | None], set[str]]


def existing_run_status_map(runs: Iterable[Any]) -> RunStatusMap:
    """Index ``db.list_runs()`` rows by (competitor, case, file) -> {statuses}."""
    out: RunStatusMap = {}
    for r in runs:
        key = (r["competitor_name"], r["case_ext_id"], r["target_file"])
        out.setdefault(key, set()).add(r["status"])
    return out


def plan_matrix(
    competitors: Sequence[Competitor],
    cases: Sequence[Case],
    existing: RunStatusMap,
    *,
    recency_filter: bool = True,
    retry_failed: bool = False,
) -> list[MatrixCell]:
    """The (competitor x case x file) cells that still need a run.

    Idempotent by construction: a cell with a complete/pending/running run is
    skipped, so re-running the loop only fills gaps. A cell whose only runs were
    auth_failed / infra_error is skipped too unless ``retry_failed`` (those never
    got a fair look). With ``recency_filter`` on, a cell is dropped when the case
    is provably not 0-day to that competitor. Output is sorted (competitor name,
    case ext_id, file order) for a deterministic, resumable pass.
    """
    cells: list[MatrixCell] = []
    for comp in sorted(competitors, key=lambda c: c.name):
        for case in sorted(cases, key=lambda c: c.ext_id):
            if recency_filter and not case_is_fresh_for(case, comp):
                continue
            for target in vulnerable_files(case):
                statuses = existing.get((comp.name, case.ext_id, target), set())
                if statuses & _SETTLED_STATUSES:
                    continue
                if statuses and statuses <= _RETRYABLE_STATUSES and not retry_failed:
                    continue
                cells.append(MatrixCell(comp, case, target))
    return cells


# -- Age-out -----------------------------------------------------------------


def select_aged_out(
    cases: Sequence[Case], competitors: Sequence[Competitor]
) -> tuple[list[Case], date | None]:
    """Vetted cases no active competitor would find 0-day, and the cutoff used.

    A case is aged out when its disclosure is on or before the *minimum* known
    cutoff among active competitors — i.e. every active competitor was likely
    trained after the bug was disclosed, so it is no longer a fair 0-day test for
    any of them. Guards keep this conservative and never destructive: if any
    active competitor lacks a parseable cutoff we cannot prove staleness, so
    nothing ages out (returns ``([], None)``); a case with no disclosure date is
    never aged out.
    """
    active = [c for c in competitors if c.status == "active"]
    cutoffs = [parse_date_prefix(c.knowledge_cutoff) for c in active]
    if not active or any(c is None for c in cutoffs):
        return [], None
    min_cutoff = min(c for c in cutoffs if c is not None)
    aged = [
        case
        for case in cases
        if (d := parse_date_prefix(case.disclosure_date)) is not None
        and d <= min_cutoff
    ]
    return aged, min_cutoff


# -- Competitor roster (config-driven) ---------------------------------------


def load_competitors(path: str | Path) -> list[Competitor]:
    """Load a competitor roster from a YAML list of dicts.

    Each entry mirrors the :class:`Competitor` fields (``name`` + ``model``
    required); unknown keys are ignored so the file can carry comments/extras.
    """
    data: Any = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a YAML list of competitors")
    known = {
        "name",
        "model",
        "runtime",
        "tool_profile",
        "auth_profile",
        "cost_model",
        "size_class",
        "knowledge_cutoff",
        "status",
    }
    out: list[Competitor] = []
    for i, entry in enumerate(data):
        if not isinstance(entry, dict):
            raise ValueError(f"{path}: competitor #{i + 1} must be a mapping")
        fields = cast("dict[str, Any]", entry)
        if not fields.get("name"):
            raise ValueError(f"{path}: competitor #{i + 1} needs a name")
        if not fields.get("model"):
            raise ValueError(f"{path}: competitor {fields['name']!r} needs a model")
        out.append(Competitor(**{k: v for k, v in fields.items() if k in known}))
    return out


def sync_competitors(
    db: Database, declared: Sequence[Competitor]
) -> tuple[int, list[str]]:
    """Reconcile the DB roster with the declared one; returns (upserted, retired).

    Declared competitors are upserted; any *active* competitor in the DB not in
    the declared set is flipped to ``retired`` (never deleted — its historical
    runs stay scoreable). Adding or removing a competitor is thus pure config.
    """
    declared_names = {c.name for c in declared}
    for comp in declared:
        db.upsert_competitor(comp.to_db_fields())
    retired: list[str] = []
    for row in db.list_competitors():
        if row["name"] not in declared_names and row["status"] == "active":
            db.upsert_competitor({"name": row["name"], "status": "retired"})
            retired.append(row["name"])
    return len(declared), retired


# -- Loop report -------------------------------------------------------------


@dataclass
class LoopReport:
    """What one pass did. ``needs_attention`` is the cron alert signal."""

    corpus: dict[str, int] | None = None
    aged_out: list[str] = field(default_factory=list)
    competitors_synced: int = 0
    competitors_retired: list[str] = field(default_factory=list)
    planned: int = 0
    ran: int = 0
    completed: int = 0
    auth_failed: int = 0
    infra_error: int = 0
    skipped_caps: int = 0
    circuit_broken: bool = False
    spend_usd: float = 0.0
    scored: int = 0
    judge_errors: int = 0
    refused: int = 0  # runs the refusal judge confirmed declined the task
    report_path: str | None = None

    @property
    def needs_attention(self) -> bool:
        """True when a human should look: an auth failure (likely an expired
        host session) or the auth circuit breaker tripped. Infra errors are
        surfaced in the summary but don't, on their own, raise this flag."""
        return self.circuit_broken or self.auth_failed > 0


# -- The pass ----------------------------------------------------------------


def run_once(
    db: Database,
    *,
    runner: Runner | None,
    scorer: Scorer | None = None,
    pipeline: CorpusPipeline | None = None,
    seed_source: SeedSource | None = None,
    corpus_limit: int | None = None,
    declared_competitors: Sequence[Competitor] | None = None,
    age_out: bool = True,
    recency_filter: bool = True,
    retry_failed: bool = False,
    max_runs: int | None = None,
    max_spend_usd: float | None = None,
    auth_fail_abort: int = 3,
    concurrency: int = 1,
    report_html_path: str | Path | None = None,
    on_event: Callable[[str], None] | None = None,
) -> LoopReport:
    """Run one full, idempotent benchmark pass and return what it did.

    Stages (each skipped if its engine is absent): refresh corpus (``pipeline``)
    -> sync roster (``declared_competitors``) -> age out stale cases -> plan +
    run the missing matrix cells (``runner``, bounded by ``max_runs`` /
    ``max_spend_usd`` and an auth circuit breaker) -> score new completions
    (``scorer``) -> regenerate the leaderboard. Reading vetted cases / active
    competitors from the DB *after* the corpus and roster stages means newly
    vetted cases and newly synced competitors enter this same pass.
    """
    report = LoopReport()
    emit = on_event or (lambda _msg: None)

    # 1. Refresh corpus (opt-in: only when a pipeline is wired). Idempotent; a
    #    re-run advances candidates as OSV/NVD catch up, vetting new cases.
    if pipeline is not None:
        emit("refreshing corpus")
        summary = pipeline.build(seed_source, limit=corpus_limit)
        report.corpus = summary.as_dict()

    # 2. Sync the competitor roster from config (adds/retires are pure config).
    if declared_competitors is not None:
        synced, retired = sync_competitors(db, declared_competitors)
        report.competitors_synced = synced
        report.competitors_retired = retired
        if retired:
            emit(f"retired {len(retired)} competitor(s) absent from roster")

    competitors = [
        Competitor.from_row(r) for r in db.list_competitors() if r["status"] == "active"
    ]
    cases = [Case.from_row(r) for r in db.list_cases(status="vetted")]

    # 3. Age out cases that are no longer 0-day to any active competitor.
    if age_out:
        aged, cutoff = select_aged_out(cases, competitors)
        for case in aged:
            db.update_case_fields(
                case.ext_id,
                {
                    "status": "retired",
                    "vet_notes": f"aged out: disclosed on/before all active "
                    f"competitor cutoffs (<= {cutoff})",
                },
            )
            report.aged_out.append(case.ext_id)
        if aged:
            emit(f"aged out {len(aged)} case(s) (cutoff <= {cutoff})")
        aged_ids = {c.ext_id for c in aged}
        cases = [c for c in cases if c.ext_id not in aged_ids]

    # 4. Plan + run the missing matrix cells.
    if runner is not None and competitors and cases:
        existing = existing_run_status_map(db.list_runs())
        plan = plan_matrix(
            competitors,
            cases,
            existing,
            recency_filter=recency_filter,
            retry_failed=retry_failed,
        )
        report.planned = len(plan)
        emit(f"planned {len(plan)} run(s)")
        _execute_runs(
            plan,
            runner,
            report,
            max_runs=max_runs,
            max_spend_usd=max_spend_usd,
            auth_fail_abort=auth_fail_abort,
            emit=emit,
            concurrency=concurrency,
        )

    # 5. Score new completions (truth + optional FP judge; bounded by what ran).
    if scorer is not None:
        for r in db.list_runs():
            # needs_scoring defers to the scorer's own precision setting, so a
            # precision-less scorer won't keep re-scoring for absent FP verdicts.
            if r["status"] == "complete" and scorer.needs_scoring(r["id"]):
                rs = scorer.score_run(r["id"])
                report.scored += 1
                if rs.outcome == "judge_error":
                    report.judge_errors += 1
                elif rs.outcome == "refused":
                    report.refused += 1
        if report.scored:
            emit(f"scored {report.scored} run(s)")

    # 6. Regenerate the leaderboard report (a pure function of persisted scores).
    if scorer is not None and report_html_path is not None:
        _write_leaderboard(db, scorer, report_html_path)
        report.report_path = str(report_html_path)
        emit(f"wrote leaderboard report to {report_html_path}")

    return report


def _execute_runs(
    plan: Sequence[MatrixCell],
    runner: Runner,
    report: LoopReport,
    *,
    max_runs: int | None,
    max_spend_usd: float | None,
    auth_fail_abort: int,
    emit: Callable[[str], None],
    concurrency: int = 1,
) -> None:
    """Run planned cells under unattended-safety rails, mutating ``report``.

    ``concurrency == 1`` (default) runs cells strictly sequentially. ``> 1``
    fans out across worker threads (see :func:`_execute_runs_concurrent`); the
    runs are I/O-bound on the model API, so threads — not processes — recover
    the idle wait.
    """
    if concurrency <= 1:
        _execute_runs_sequential(
            plan,
            runner,
            report,
            max_runs=max_runs,
            max_spend_usd=max_spend_usd,
            auth_fail_abort=auth_fail_abort,
            emit=emit,
        )
    else:
        _execute_runs_concurrent(
            plan,
            runner,
            report,
            max_runs=max_runs,
            max_spend_usd=max_spend_usd,
            auth_fail_abort=auth_fail_abort,
            emit=emit,
            concurrency=concurrency,
        )


def _execute_runs_sequential(
    plan: Sequence[MatrixCell],
    runner: Runner,
    report: LoopReport,
    *,
    max_runs: int | None,
    max_spend_usd: float | None,
    auth_fail_abort: int,
    emit: Callable[[str], None],
) -> None:
    """Run planned cells one at a time.

    Stops launching new runs once a run cap or spend cap is hit, or once
    ``auth_fail_abort`` *consecutive* auth failures trip the circuit breaker —
    the signature of an expired host session, where continuing would only burn
    budget failing identically. A non-auth outcome resets the consecutive count,
    so one mis-authed competitor among healthy ones doesn't trip it.
    """
    consecutive_auth = 0
    for cell in plan:
        if report.circuit_broken:
            report.skipped_caps += 1
            continue
        if max_runs is not None and report.ran >= max_runs:
            report.skipped_caps += 1
            continue
        if max_spend_usd is not None and report.spend_usd >= max_spend_usd:
            report.skipped_caps += 1
            continue

        result = runner.run_case(cell.case, cell.competitor, cell.target_file)
        report.ran += 1
        if result.cost_usd:
            report.spend_usd += result.cost_usd

        if result.status == "complete":
            report.completed += 1
            consecutive_auth = 0
        elif result.status == "auth_failed":
            report.auth_failed += 1
            consecutive_auth += 1
            if consecutive_auth >= auth_fail_abort:
                report.circuit_broken = True
                emit(
                    f"auth circuit breaker: {consecutive_auth} consecutive auth "
                    "failures — aborting remaining runs (check host login)"
                )
        else:  # infra_error (or any other non-complete status)
            report.infra_error += 1
            consecutive_auth = 0


def _execute_runs_concurrent(
    plan: Sequence[MatrixCell],
    runner: Runner,
    report: LoopReport,
    *,
    max_runs: int | None,
    max_spend_usd: float | None,
    auth_fail_abort: int,
    emit: Callable[[str], None],
    concurrency: int,
) -> None:
    """Run planned cells across ``concurrency`` worker threads.

    One worker owns one competitor at a time and runs that competitor's cells
    sequentially, so **no model ever has two runs in flight at once** — this is
    the rate-limit guard (a free OpenRouter endpoint would 429 under parallel
    same-model load). With more competitors than workers, ``concurrency`` of
    them run concurrently; load naturally spreads across distinct models.

    Checkouts are pre-warmed once per case before the pool starts, so workers
    only read the shared ``:ro`` source tree (``prepare_checkout`` would
    otherwise race two threads into the same rmtree+refetch).

    Safety rails are shared state under one lock. The auth breaker uses the
    concurrency-safe rule **total auth-failures ≥ K while nothing has completed**
    — "consecutive" is undefined when runs overlap, and this still catches the
    dead-host-login case (everything failing from the start) without tripping
    once any run has succeeded.
    """
    import contextlib
    import queue as _queue
    import threading

    # Pre-warm each distinct case's checkout sequentially (idempotent fast-path
    # afterwards). A case that can't be checked out is left to fail per-run as
    # infra_error inside run_case, exactly as in the sequential path.
    prewarm = getattr(runner, "prewarm_checkout", None)
    if callable(prewarm):
        seen: set[str] = set()
        for cell in plan:
            if cell.case.ext_id in seen:
                continue
            seen.add(cell.case.ext_id)
            # A case that can't be checked out is left to fail per-run as
            # infra_error inside run_case, exactly as in the sequential path.
            with contextlib.suppress(Exception):
                prewarm(cell.case)

    # Group cells by competitor (order-preserving): one queue entry per model.
    groups: dict[str, list[MatrixCell]] = {}
    for cell in plan:
        groups.setdefault(cell.competitor.name, []).append(cell)
    work: _queue.Queue[list[MatrixCell]] = _queue.Queue()
    for cells in groups.values():
        work.put(cells)

    lock = threading.Lock()
    stop = threading.Event()

    def worker() -> None:
        while not stop.is_set():
            try:
                cells = work.get_nowait()
            except _queue.Empty:
                return
            for cell in cells:
                with lock:
                    if stop.is_set():
                        break
                    if max_runs is not None and report.ran >= max_runs:
                        stop.set()
                        break
                    if max_spend_usd is not None and report.spend_usd >= max_spend_usd:
                        stop.set()
                        break
                result = runner.run_case(cell.case, cell.competitor, cell.target_file)
                with lock:
                    report.ran += 1
                    if result.cost_usd:
                        report.spend_usd += result.cost_usd
                    if result.status == "complete":
                        report.completed += 1
                    elif result.status == "auth_failed":
                        report.auth_failed += 1
                        if (
                            auth_fail_abort > 0
                            and report.completed == 0
                            and report.auth_failed >= auth_fail_abort
                        ):
                            report.circuit_broken = True
                            stop.set()
                            emit(
                                f"auth circuit breaker: {report.auth_failed} auth "
                                "failures with no completions — aborting remaining "
                                "runs (check host login)"
                            )
                    else:  # infra_error (or any other non-complete status)
                        report.infra_error += 1

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Cells never attempted (caps/breaker stopped the pool) count as skipped.
    report.skipped_caps += max(0, len(plan) - report.ran)


def _write_leaderboard(db: Database, scorer: Scorer, html_path: str | Path) -> None:
    """Render the leaderboard HTML from already-persisted scores (no judge spend)."""
    from .html_report import generate_leaderboard_report
    from .score import case_scores, drop_competitors, leaderboard

    retired = {c["name"] for c in db.list_competitors() if c["status"] == "retired"}
    run_scores = drop_competitors(
        [scorer.load_run_score(r["id"]) for r in db.list_runs()], retired
    )
    entries = leaderboard(run_scores)
    html = generate_leaderboard_report(entries, case_scores(run_scores))
    Path(html_path).write_text(html)
