"""P6 automation loop.

The planning core (recency gate, matrix dedup, age-out, roster sync) is pure, so
it is tested directly. The pass itself (`run_once`) is driven with a fake runner
that persists coherent DB rows and a *real* Scorer wired to fake judges, so the
budget rails, the auth circuit breaker, age-out, and the scoring/report stages
are all exercised without podman, a network, or a real judge.
"""

import threading
import time
from datetime import date

import pytest
import yaml

from nelson.automate import (
    LoopReport,
    case_is_fresh_for,
    existing_run_status_map,
    load_competitors,
    parse_date_prefix,
    plan_matrix,
    run_once,
    select_aged_out,
    sync_competitors,
)
from nelson.corpus import Case
from nelson.db import Database
from nelson.runner import Competitor, RunResult
from nelson.score import FPVerdict, Scorer, TruthVerdict

# -- Builders ----------------------------------------------------------------


def _case(ext_id, *, files=("Foo.java",), disclosure=None, status="vetted"):
    return Case(
        source="cvd",
        ext_id=ext_id,
        status=status,
        repo_url="https://example.invalid/r",
        vuln_commit="deadbeef",
        fix_commit="cafef00d",
        disclosure_date=disclosure,
        gt_files=list(files),
        gt_hunks=[{"file": f, "start": 73, "end": 79} for f in files],
    )


def _comp(name, model="m", *, cutoff=None, status="active"):
    return Competitor(name=name, model=model, knowledge_cutoff=cutoff, status=status)


# -- Dates / recency ---------------------------------------------------------


def test_parse_date_prefix_handles_iso_date_month_and_year():
    assert parse_date_prefix("2026-01-15T09:00:00Z") == date(2026, 1, 15)
    assert parse_date_prefix("2026-01-15") == date(2026, 1, 15)
    assert parse_date_prefix("2025-03") == date(2025, 3, 1)  # month -> first
    assert parse_date_prefix("2025") == date(2025, 1, 1)  # year -> Jan 1
    assert parse_date_prefix("") is None
    assert parse_date_prefix(None) is None
    assert parse_date_prefix("not-a-date") is None


def test_case_is_fresh_when_disclosed_after_cutoff():
    comp = _comp("c", cutoff="2025-01")
    assert case_is_fresh_for(_case("A", disclosure="2026-01-15"), comp)
    assert not case_is_fresh_for(_case("B", disclosure="2024-06-01"), comp)


def test_case_is_fresh_defaults_true_on_missing_dates():
    # Never silently exclude a cell for lack of data — only on proof of staleness.
    assert case_is_fresh_for(_case("A", disclosure=None), _comp("c", cutoff="2025-01"))
    assert case_is_fresh_for(_case("A", disclosure="2026-01-01"), _comp("c"))


# -- Matrix planning ---------------------------------------------------------


def test_plan_matrix_one_cell_per_competitor_case_file():
    comps = [_comp("alpha"), _comp("beta")]
    cases = [_case("A", files=("Foo.java", "Bar.java"))]
    cells = plan_matrix(comps, cases, {}, recency_filter=False)
    assert len(cells) == 4  # 2 competitors x 1 case x 2 files
    # Deterministic order: competitor name, then case, then file order.
    assert [(c.competitor.name, c.target_file) for c in cells] == [
        ("alpha", "Foo.java"),
        ("alpha", "Bar.java"),
        ("beta", "Foo.java"),
        ("beta", "Bar.java"),
    ]


def _run_row(status, *, name="alpha", ext_id="A", target="Foo.java"):
    return {
        "competitor_name": name,
        "case_ext_id": ext_id,
        "target_file": target,
        "status": status,
    }


def test_plan_matrix_skips_settled_and_inflight_cells():
    comps = [_comp("alpha")]
    cases = [_case("A")]
    for status in ("complete", "pending", "running"):
        existing = existing_run_status_map([_run_row(status)])
        assert plan_matrix(comps, cases, existing, recency_filter=False) == []


def test_plan_matrix_retries_failed_only_when_asked():
    comps = [_comp("alpha")]
    cases = [_case("A")]
    existing = existing_run_status_map(
        [_run_row("auth_failed"), _run_row("infra_error")]
    )
    assert plan_matrix(comps, cases, existing, recency_filter=False) == []
    retried = plan_matrix(
        comps, cases, existing, recency_filter=False, retry_failed=True
    )
    assert len(retried) == 1


def test_plan_matrix_applies_recency_filter():
    comps = [_comp("alpha", cutoff="2025-01")]
    cases = [
        _case("FRESH", disclosure="2026-01-01"),
        _case("STALE", disclosure="2024-01-01"),
    ]
    cells = plan_matrix(comps, cases, {}, recency_filter=True)
    assert [c.case.ext_id for c in cells] == ["FRESH"]
    # Disabling the filter brings the stale case back.
    assert len(plan_matrix(comps, cases, {}, recency_filter=False)) == 2


def test_existing_run_status_map_groups_by_cell():
    rows = [
        _run_row("complete", name="a", target="F"),
        _run_row("auth_failed", name="a", target="F"),
        _run_row("running", name="b", target="F"),
    ]
    m = existing_run_status_map(rows)
    assert m[("a", "A", "F")] == {"complete", "auth_failed"}
    assert m[("b", "A", "F")] == {"running"}


# -- Age-out -----------------------------------------------------------------


def test_select_aged_out_retires_cases_before_min_cutoff():
    comps = [_comp("a", cutoff="2025-06"), _comp("b", cutoff="2025-01")]
    cases = [
        _case("OLD", disclosure="2024-12-01"),  # <= min cutoff (2025-01) -> aged out
        _case("NEW", disclosure="2026-01-01"),
    ]
    aged, cutoff = select_aged_out(cases, comps)
    assert [c.ext_id for c in aged] == ["OLD"]
    assert cutoff == date(2025, 1, 1)


def test_select_aged_out_is_inert_without_full_cutoff_data():
    # If any active competitor lacks a cutoff, staleness can't be proven.
    comps = [_comp("a", cutoff="2025-06"), _comp("b", cutoff=None)]
    cases = [_case("OLD", disclosure="2020-01-01")]
    assert select_aged_out(cases, comps) == ([], None)


def test_select_aged_out_never_ages_a_case_without_a_disclosure_date():
    comps = [_comp("a", cutoff="2025-06")]
    cases = [_case("NODATE", disclosure=None)]
    assert select_aged_out(cases, comps)[0] == []


def test_select_aged_out_ignores_retired_competitors():
    # Only *active* competitors define the freshness frontier.
    comps = [_comp("live", cutoff="2025-06", status="active")]
    cases = [_case("OLD", disclosure="2024-01-01")]
    aged, _ = select_aged_out(cases, comps)
    assert [c.ext_id for c in aged] == ["OLD"]


# -- Roster (config) ---------------------------------------------------------


def test_load_competitors_parses_known_fields_and_ignores_extras(tmp_path):
    path = tmp_path / "roster.yaml"
    path.write_text(
        yaml.safe_dump(
            [
                {
                    "name": "claude-code/sonnet",
                    "model": "sonnet",
                    "size_class": "medium",
                    "knowledge_cutoff": "2025-01",
                    "note": "ignored extra key",
                }
            ]
        )
    )
    comps = load_competitors(path)
    assert len(comps) == 1
    assert comps[0].name == "claude-code/sonnet"
    assert comps[0].size_class == "medium"


def test_load_competitors_requires_name_and_model(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump([{"name": "x"}]))  # no model
    with pytest.raises(ValueError, match="needs a model"):
        load_competitors(path)


def test_sync_competitors_upserts_declared_and_retires_absent(tmp_path):
    db = Database(tmp_path / "t.db")
    db.upsert_competitor({"name": "old/model", "model": "old", "status": "active"})
    declared = [_comp("new/model", "new")]
    upserted, retired = sync_competitors(db, declared)
    assert upserted == 1
    assert retired == ["old/model"]
    rows = {r["name"]: r["status"] for r in db.list_competitors()}
    assert rows == {"new/model": "active", "old/model": "retired"}


# -- LoopReport --------------------------------------------------------------


def test_loop_report_needs_attention_on_auth_or_breaker():
    assert not LoopReport().needs_attention
    assert LoopReport(auth_failed=1).needs_attention
    assert LoopReport(circuit_broken=True).needs_attention
    assert not LoopReport(infra_error=3).needs_attention  # infra alone doesn't


# -- run_once (fake runner + real Scorer with fake judges) -------------------


class FakeRunner:
    """A BenchRunner stand-in. Persists a coherent run row + findings per call and
    returns a canned RunResult, so planning/scoring stages see real DB state."""

    def __init__(self, db, *, statuses=None, cost=0.1, findings=None):
        self.db = db
        self._statuses = list(statuses) if statuses else []
        self.cost = cost
        self.findings = findings or []
        self.calls: list[tuple[str, str, str]] = []
        self._i = 0

    def _next_status(self) -> str:
        if not self._statuses:
            return "complete"
        status = self._statuses[min(self._i, len(self._statuses) - 1)]
        self._i += 1
        return status

    def run_case(self, case, competitor, target_file) -> RunResult:
        self.calls.append((competitor.name, case.ext_id, target_file))
        status = self._next_status()
        comp_id = self.db.upsert_competitor(competitor.to_db_fields())
        row = self.db.get_case(case.ext_id)
        case_id = row["id"] if row else self.db.upsert_case(case.to_db_fields())
        run_id = self.db.create_run(case_id, comp_id, target_file)
        self.db.start_run(run_id)
        if status == "complete":
            self.db.complete_run(
                run_id, cost_usd=self.cost, wall_clock_s=1.0, tokens_in=1, tokens_out=1
            )
            for f in self.findings:
                self.db.add_run_finding(
                    run_id,
                    file=f.get("file"),
                    line_start=f.get("line"),
                    line_end=f.get("line"),
                    description=f.get("explanation", "x"),
                    confidence=f.get("confidence"),
                    cwe=f.get("cwe"),
                )
            return RunResult(
                status="complete", cost_usd=self.cost, findings=self.findings
            )
        if status == "auth_failed":
            self.db.mark_run_auth_failed(run_id, "fake auth")
            return RunResult(status="auth_failed", error="fake auth")
        self.db.mark_run_infra_error(run_id, "fake infra")
        return RunResult(status="infra_error", error="fake infra")


class _TruthJudge:
    """Canned TruthJudge (annotated so its return type matches the Protocol)."""

    name = "fake-truth"

    def __init__(self, verdict: TruthVerdict):
        self.verdict = verdict

    def judge(self, case, finding):
        return self.verdict


class _FPJudge:
    """Canned FPJudge (annotated return matches the Protocol)."""

    name = "fake-fp"

    def __init__(self, verdict: FPVerdict):
        self.verdict = verdict

    def judge(self, finding, source):
        return self.verdict


class _Code:
    def source(self, case, file) -> str | None:
        return "void f(){}"


def _seed_db(tmp_path, *, cases, competitors):
    db = Database(tmp_path / "t.db")
    for c in cases:
        db.upsert_case(c.to_db_fields())
    for comp in competitors:
        db.upsert_competitor(comp.to_db_fields())
    return db


def test_run_once_plans_and_runs_missing_cells(tmp_path):
    db = _seed_db(
        tmp_path,
        cases=[_case("A", disclosure="2026-01-01")],
        competitors=[_comp("alpha", cutoff="2025-01")],
    )
    runner = FakeRunner(db)
    report = run_once(db, runner=runner, age_out=False)
    assert report.planned == 1
    assert report.ran == 1
    assert report.completed == 1
    assert runner.calls == [("alpha", "A", "Foo.java")]


def test_run_once_is_idempotent(tmp_path):
    db = _seed_db(
        tmp_path,
        cases=[_case("A", disclosure="2026-01-01")],
        competitors=[_comp("alpha", cutoff="2025-01")],
    )
    runner = FakeRunner(db)
    run_once(db, runner=runner, age_out=False)
    report2 = run_once(db, runner=runner, age_out=False)
    # The first pass's complete run settles the cell; the second plans nothing.
    assert report2.planned == 0
    assert report2.ran == 0
    assert len(runner.calls) == 1


def test_run_once_honors_max_runs_cap(tmp_path):
    db = _seed_db(
        tmp_path,
        cases=[_case("A"), _case("B"), _case("C")],
        competitors=[_comp("alpha")],
    )
    runner = FakeRunner(db)
    report = run_once(
        db, runner=runner, age_out=False, recency_filter=False, max_runs=1
    )
    assert report.planned == 3
    assert report.ran == 1
    assert report.skipped_caps == 2


def test_run_once_honors_spend_cap(tmp_path):
    db = _seed_db(
        tmp_path,
        cases=[_case("A"), _case("B"), _case("C")],
        competitors=[_comp("alpha")],
    )
    runner = FakeRunner(db, cost=0.10)
    report = run_once(
        db, runner=runner, age_out=False, recency_filter=False, max_spend_usd=0.15
    )
    # Stops once spend reaches the cap: run 1 (0.10) under, run 2 takes it to 0.20,
    # then the third is skipped.
    assert report.ran == 2
    assert report.skipped_caps == 1
    assert report.spend_usd == pytest.approx(0.20)


def test_run_once_auth_circuit_breaker_aborts_remaining(tmp_path):
    db = _seed_db(
        tmp_path,
        cases=[_case(x) for x in ("A", "B", "C", "D", "E")],
        competitors=[_comp("alpha")],
    )
    runner = FakeRunner(db, statuses=["auth_failed"])  # every run auth-fails
    report = run_once(
        db, runner=runner, age_out=False, recency_filter=False, auth_fail_abort=2
    )
    assert report.auth_failed == 2  # tripped on the 2nd consecutive failure
    assert report.circuit_broken is True
    assert report.skipped_caps == 3  # remaining cells aborted
    assert report.needs_attention is True


# -- Concurrency (bench loop --concurrency N) --------------------------------


class _ProbeRunner:
    """Records in-flight competitors per call to prove the executor runs ≤1 cell
    per model at once while genuinely overlapping *distinct* models. Writes a
    coherent complete-run row from each worker thread (exercises the thread-local
    DB connections)."""

    def __init__(self, db, *, hold=0.03):
        self.db = db
        self.hold = hold
        self._lock = threading.Lock()
        self._inflight: dict[str, int] = {}
        self.max_inflight = 0
        self.same_model_overlap = False
        self.calls: list[tuple[str, str]] = []

    def run_case(self, case, competitor, target_file) -> RunResult:
        with self._lock:
            self.calls.append((competitor.name, case.ext_id))
            self._inflight[competitor.name] = self._inflight.get(competitor.name, 0) + 1
            if self._inflight[competitor.name] > 1:
                self.same_model_overlap = True
            self.max_inflight = max(self.max_inflight, sum(self._inflight.values()))
        time.sleep(self.hold)  # widen the overlap window so parallelism is observable
        with self._lock:
            self._inflight[competitor.name] -= 1
        comp_id = self.db.upsert_competitor(competitor.to_db_fields())
        row = self.db.get_case(case.ext_id)
        case_id = row["id"] if row else self.db.upsert_case(case.to_db_fields())
        run_id = self.db.create_run(case_id, comp_id, target_file)
        self.db.start_run(run_id)
        self.db.complete_run(
            run_id, cost_usd=0.0, wall_clock_s=1.0, tokens_in=1, tokens_out=1
        )
        return RunResult(status="complete", cost_usd=0.0)


def _four_comps():
    return [_comp(n) for n in ("alpha", "beta", "gamma", "delta")]


def test_run_once_concurrency_runs_every_cell_exactly_once(tmp_path):
    db = _seed_db(
        tmp_path, cases=[_case("A"), _case("B"), _case("C")], competitors=_four_comps()
    )
    runner = _ProbeRunner(db)
    report = run_once(
        db, runner=runner, age_out=False, recency_filter=False, concurrency=4
    )
    assert (report.planned, report.ran, report.completed) == (12, 12, 12)
    assert sorted(runner.calls) == sorted(
        (c, k) for c in ("alpha", "beta", "gamma", "delta") for k in ("A", "B", "C")
    )


def test_run_once_concurrency_never_overlaps_one_model_but_overlaps_distinct(tmp_path):
    db = _seed_db(
        tmp_path, cases=[_case("A"), _case("B"), _case("C")], competitors=_four_comps()
    )
    runner = _ProbeRunner(db, hold=0.04)
    run_once(db, runner=runner, age_out=False, recency_filter=False, concurrency=4)
    assert runner.same_model_overlap is False  # rate-limit guard: ≤1 run per model
    assert runner.max_inflight > 1  # but distinct models did run in parallel


def test_run_once_concurrency_is_idempotent(tmp_path):
    db = _seed_db(
        tmp_path, cases=[_case("A"), _case("B")], competitors=[_comp("a"), _comp("b")]
    )
    run_once(
        db, runner=_ProbeRunner(db), age_out=False, recency_filter=False, concurrency=3
    )
    runner2 = _ProbeRunner(db)
    report2 = run_once(
        db, runner=runner2, age_out=False, recency_filter=False, concurrency=3
    )
    assert report2.planned == 0 and report2.ran == 0 and runner2.calls == []


def test_run_once_concurrency_auth_breaker_trips_with_zero_completes(tmp_path):
    db = _seed_db(tmp_path, cases=[_case("A"), _case("B")], competitors=_four_comps())
    runner = FakeRunner(db, statuses=["auth_failed"])  # every run auth-fails
    report = run_once(
        db,
        runner=runner,
        age_out=False,
        recency_filter=False,
        concurrency=4,
        auth_fail_abort=2,
    )
    assert report.circuit_broken is True
    assert report.completed == 0
    assert report.auth_failed >= 2  # exact count is nondeterministic under overlap
    assert report.ran + report.skipped_caps == report.planned == 8


class _CoordinatedRunner:
    """alpha completes and signals an event; the others block on that event
    before auth-failing — so a completion is *guaranteed* to land first. This
    pins down the breaker's concurrency-safe rule: once anything completes, the
    'total auth-fails with zero completes' trip can never fire."""

    def __init__(self, db):
        self.db = db
        self._completed = threading.Event()

    def run_case(self, case, competitor, target_file) -> RunResult:
        comp_id = self.db.upsert_competitor(competitor.to_db_fields())
        row = self.db.get_case(case.ext_id)
        case_id = row["id"] if row else self.db.upsert_case(case.to_db_fields())
        run_id = self.db.create_run(case_id, comp_id, target_file)
        self.db.start_run(run_id)
        if competitor.name == "alpha":
            self.db.complete_run(
                run_id, cost_usd=0.0, wall_clock_s=1.0, tokens_in=1, tokens_out=1
            )
            self._completed.set()
            return RunResult(status="complete", cost_usd=0.0)
        self._completed.wait(timeout=5.0)  # never fail before the completion lands
        self.db.mark_run_auth_failed(run_id, "fake auth")
        return RunResult(status="auth_failed", error="fake auth")


def test_run_once_concurrency_breaker_silent_once_something_completes(tmp_path):
    db = _seed_db(
        tmp_path,
        cases=[_case("A"), _case("B")],
        competitors=[_comp("alpha"), _comp("beta")],
    )
    report = run_once(
        db,
        runner=_CoordinatedRunner(db),
        age_out=False,
        recency_filter=False,
        concurrency=2,
        auth_fail_abort=1,  # would trip instantly on the first fail if not gated
    )
    assert report.circuit_broken is False  # a completion landed → never trips
    assert report.completed == 2  # alpha's two cells both completed
    assert report.auth_failed == 2  # beta's two cells failed, but never tripped


def test_run_once_ages_out_stale_case_and_excludes_it(tmp_path):
    db = _seed_db(
        tmp_path,
        cases=[
            _case("OLD", disclosure="2024-01-01"),
            _case("NEW", disclosure="2026-01-01"),
        ],
        competitors=[_comp("alpha", cutoff="2025-06")],
    )
    runner = FakeRunner(db)
    report = run_once(db, runner=runner, age_out=True)
    assert report.aged_out == ["OLD"]
    assert db.get_case("OLD")["status"] == "retired"
    # Only the fresh case is run.
    assert runner.calls == [("alpha", "NEW", "Foo.java")]


def test_run_once_scores_completions_and_counts_judge_errors(tmp_path):
    db = _seed_db(
        tmp_path,
        cases=[_case("A", disclosure="2026-01-01")],
        competitors=[_comp("alpha", cutoff="2025-01")],
    )
    # Finding lands inside Foo.java:73-79 so it localizes; the truth judge then
    # errors -> the run is undetermined (judge_error), never a silent miss.
    runner = FakeRunner(db, findings=[{"file": "Foo.java", "line": 76}])
    scorer = Scorer(
        db,
        _TruthJudge(TruthVerdict(None, error="boom")),
        fp_judge=_FPJudge(FPVerdict(False)),
        code=_Code(),
    )
    report = run_once(db, runner=runner, scorer=scorer, age_out=False)
    assert report.scored == 1
    assert report.judge_errors == 1


def test_run_once_writes_leaderboard_report(tmp_path):
    db = _seed_db(
        tmp_path,
        cases=[_case("A", disclosure="2026-01-01")],
        competitors=[_comp("alpha", cutoff="2025-01")],
    )
    runner = FakeRunner(db, findings=[{"file": "Foo.java", "line": 76}])
    scorer = Scorer(
        db,
        _TruthJudge(TruthVerdict(True)),
        fp_judge=_FPJudge(FPVerdict(False)),
        code=_Code(),
    )
    out = tmp_path / "leaderboard.html"
    report = run_once(
        db, runner=runner, scorer=scorer, age_out=False, report_html_path=out
    )
    assert report.report_path == str(out)
    assert out.is_file()
    assert "<!DOCTYPE html>" in out.read_text()


def test_run_once_syncs_competitor_roster(tmp_path):
    db = _seed_db(
        tmp_path,
        cases=[_case("A", disclosure="2026-01-01")],
        competitors=[_comp("stale/model", "old")],  # absent from the roster below
    )
    runner = FakeRunner(db)
    declared = [_comp("claude-code/sonnet", "sonnet", cutoff="2025-01")]
    report = run_once(db, runner=runner, declared_competitors=declared, age_out=False)
    assert report.competitors_retired == ["stale/model"]
    # Only the newly-declared, active competitor runs the matrix.
    assert runner.calls == [("claude-code/sonnet", "A", "Foo.java")]
