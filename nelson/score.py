"""P3 scoring: localization gate + Opus truth judge -> hit / miss per run.

Per ``(competitor, case)``, only a run that reached ``complete`` is scored. An
``auth_failed`` / ``infra_error`` run never got a fair look at the code and is
excluded from the denominator (the integrity rule) — never counted as a miss.

A reported finding is a *hit* only if it clears two gates:

  1. **localization** (deterministic, cheap): the finding points into a
     ground-truth file within N lines of a patched hunk. The pre-patch code the
     competitor read is the OLD side of the fix diff, so ``gt_hunks`` line
     numbers are the competitor's own line numbers — the gate is near-exact, and
     N only forgives a few lines of reporting drift.
  2. **truth judge** (Opus 4.7): shown the advisory (the truth side, exactly as
     pre-vet sees it), the judge rules whether the reported bug is the *same*
     root-cause bug the fix addressed — not merely the same file or function.

Only localized findings reach the judge (the gate is the inexpensive filter; the
rest are false-positive candidates for the P4 FP-judge). A judge *failure*
(timeout / auth / unparseable reply) is never read as "different bug": like a
pre-vet failure it leaves the run's outcome **undetermined** (``judge_error``),
so a genuine hit is never silently demoted to a miss. Judge cost is logged to the
``judgments`` ledger so it is tracked separately from competitor cost and never
distorts the Pareto view.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from .agents import _run_cli, classify_failure

# The judge-reply JSON repair/unwrap logic is shared with pre-vet; reuse the one
# tested implementation rather than forking it.
from .prevet import _loads_object, _unwrap_claude_json

if TYPE_CHECKING:
    from .corpus import Case
    from .db import Database

# How many lines a reported line may drift from a patched hunk and still localize.
# The competitor reads the pre-patch tree, so gt_hunks are its own line numbers;
# this only absorbs off-by-a-few reporting, not whole-function slack.
DEFAULT_LINE_TOLERANCE = 10

# Run outcomes that count toward the detection-rate denominator. judge_error and
# excluded are deliberately *not* here — an undetermined or never-run case can
# never be a "miss".
ELIGIBLE_OUTCOMES = frozenset({"hit", "miss"})


# -- Localization gate -------------------------------------------------------


def _normalize_path(path: str) -> str:
    """Canonicalize a path for matching: forward slashes, no mount/`./` prefix."""
    p = path.strip().replace("\\", "/")
    for prefix in ("/src/", "src/", "./"):
        if p.startswith(prefix):
            p = p[len(prefix) :]
            break
    return p.lstrip("/")


def paths_match(finding_path: str | None, gt_path: str | None) -> bool:
    """True if the finding's path denotes the ground-truth file.

    Equal after normalization, or one is a path-component suffix of the other —
    so a model reporting a fuller (mount-rooted) or shorter (module-relative)
    path than the diff's pre-patch path still matches. Suffix matching is on
    whole path components (``.../foo.c``), never a bare substring.
    """
    if not finding_path or not gt_path:
        return False
    f, g = _normalize_path(finding_path), _normalize_path(gt_path)
    if not f or not g:
        return False
    return f == g or f.endswith("/" + g) or g.endswith("/" + f)


@dataclass
class Localization:
    """Result of the localization gate for one finding."""

    matched: bool
    hunk: dict[str, Any] | None = None  # the gt_hunk the finding landed in


def localize(
    file: str | None,
    line: int | None,
    gt_hunks: list[dict[str, Any]],
    tolerance: int = DEFAULT_LINE_TOLERANCE,
) -> Localization:
    """Decide whether ``(file, line)`` lands within ``tolerance`` of a gt hunk.

    A finding with no line cannot localize (the gate is line-based); a hunk with
    no usable range is skipped. Returns the matched hunk for auditability.
    """
    if line is None:
        return Localization(False)
    for hunk in gt_hunks:
        if not paths_match(file, hunk.get("file")):
            continue
        start = hunk.get("start")
        if start is None:
            continue
        end = hunk.get("end", start)
        if start - tolerance <= line <= end + tolerance:
            return Localization(True, hunk)
    return Localization(False)


# -- Truth judge -------------------------------------------------------------

TRUTH_INSTRUCTIONS = """\
You are scoring a security benchmark. A model audited code that contained a \
known vulnerability and reported a finding. Decide whether the model's finding \
is the SAME vulnerability that a later fix addressed — the same root-cause bug, \
not merely a finding in the same file or function.

You are given the advisory for the real vulnerability (the ground truth) and the \
model's reported finding. Answer same_bug=true only if a security engineer would \
agree the model identified THIS specific vulnerability. A finding that flags a \
different bug, or only vaguely gestures at the right area without naming the \
actual flaw, is NOT the same.

Reply with ONLY a JSON object:
{"same_bug": <true|false>, "reasoning": "<one or two sentences>"}"""


@dataclass
class ReportedFinding:
    """The competitor-reported finding, as the truth judge sees it."""

    file: str | None
    line: int | None
    description: str | None = None
    cwe: str | None = None
    confidence: str | None = None

    @classmethod
    def from_row(cls, row: Any) -> ReportedFinding:
        return cls(
            file=row["file"],
            line=row["line_start"],
            description=row["description"],
            cwe=row["cwe"],
            confidence=row["confidence"],
        )


@dataclass
class TruthVerdict:
    """Outcome of truth-judging one finding.

    ``same_bug`` is meaningful only when ``error`` is None; when the judge could
    not decide, ``same_bug`` is None and ``error`` explains why. A None verdict
    must never be treated as "different bug" (that would corrupt a real hit into
    a miss) — the run's outcome stays undetermined instead.
    """

    same_bug: bool | None
    reasoning: str = ""
    tokens_in: int | None = None
    tokens_out: int | None = None
    cost_usd: float | None = None
    error: str | None = None

    @property
    def label(self) -> str:
        """Compact verdict string persisted to ``judge_truth_verdict``."""
        if self.error is not None:
            return f"error: {self.error}"
        return "same_bug" if self.same_bug else "different_bug"


@runtime_checkable
class TruthJudge(Protocol):
    name: str

    def judge(self, case: Case, finding: ReportedFinding) -> TruthVerdict: ...


def build_truth_prompt(case: Case, finding: ReportedFinding) -> str:
    """Assemble the advisory (truth) + the reported finding into a judge prompt."""
    advisory = "\n".join(
        f"{label}: {value}"
        for label, value in (
            ("Identifier", case.ext_id),
            ("CVE", case.cve_id),
            ("GHSA", case.ghsa_id),
            ("Bug class", case.bug_class),
            ("CWE", case.cwe),
            ("Severity", case.severity),
            ("Description", case.description),
        )
        if value
    )
    line_no = finding.line if finding.line is not None else "?"
    location = f"{finding.file or '?'}:{line_no}"
    reported = "\n".join(
        f"{label}: {value}"
        for label, value in (
            ("Location", location),
            ("Reported CWE", finding.cwe),
            ("Confidence", finding.confidence),
            ("Explanation", finding.description),
        )
        if value
    )
    return (
        f"{TRUTH_INSTRUCTIONS}\n\n"
        f"## Advisory (ground truth)\n{advisory}\n\n"
        f"## Model's reported finding\n{reported}"
    )


_BOOL_TRUE = {"true", "yes", "y", "same", "same_bug", "1"}
_BOOL_FALSE = {"false", "no", "n", "different", "different_bug", "0"}


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in _BOOL_TRUE:
            return True
        if v in _BOOL_FALSE:
            return False
    return None


def parse_truth_verdict(text: str) -> tuple[bool, str] | None:
    """Extract (same_bug, reasoning) from the judge's reply, or None if absent.

    Returns None (not a False verdict) when nothing parseable is found, so the
    caller can distinguish "judge said different" from "judge reply unusable".
    """
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        data = _loads_object(match.group(0))
        if data is not None and "same_bug" in data:
            same = _coerce_bool(data["same_bug"])
            if same is not None:
                return same, str(data.get("reasoning", ""))
    return None


class ClaudeTruthJudge:
    """Truth judge backed by ``claude -p`` (Opus by default).

    Mirrors :class:`nelson.prevet.ClaudeCLIJudge`: runs on the host against the
    already-signed-in subscription (no API key, no container), and a failure is
    surfaced via ``TruthVerdict.error`` rather than guessed.
    """

    name = "claude-cli"

    def __init__(self, model: str = "opus", timeout: int = 180):
        self.model = model
        self.timeout = timeout

    def judge(self, case: Case, finding: ReportedFinding) -> TruthVerdict:
        prompt = build_truth_prompt(case, finding)
        cmd = [
            "claude",
            "-p",
            "--model",
            self.model,
            "--output-format",
            "json",
            "--no-session-persistence",
        ]
        try:
            # env=None: inherit the host's authenticated claude CLI session.
            result = _run_cli(cmd, self.timeout, input_text=prompt)
        except subprocess.TimeoutExpired:
            return TruthVerdict(None, error="timeout")

        raw, stderr = result.stdout, result.stderr
        if result.returncode != 0:
            kind = classify_failure(raw + stderr, failed=True)
            return TruthVerdict(
                None,
                error=f"claude exit {result.returncode} ({kind}): {stderr[:200]}",
            )

        text, tin, tout, cost = _unwrap_claude_json(raw)
        parsed = parse_truth_verdict(text)
        if parsed is None:
            return TruthVerdict(
                None,
                reasoning=text[:300],
                tokens_in=tin,
                tokens_out=tout,
                cost_usd=cost,
                error="unparseable judge reply",
            )
        same, reasoning = parsed
        return TruthVerdict(
            same,
            reasoning=reasoning,
            tokens_in=tin,
            tokens_out=tout,
            cost_usd=cost,
        )


# -- Scorer ------------------------------------------------------------------


@dataclass
class FindingScore:
    """Per-finding scoring detail."""

    finding_id: int
    file: str | None
    line: int | None
    localized: bool
    truth: TruthVerdict | None = None  # None unless the finding reached the judge


@dataclass
class RunScore:
    """Scoring outcome for one run.

    ``outcome`` is one of:
      hit         — a localized finding the judge confirmed is the same bug;
      miss        — complete run, no confirmed finding (and none undetermined);
      judge_error — complete run, no confirmed finding but >=1 localized finding
                    the judge could not decide (undetermined, NOT a miss);
      excluded    — run never reached ``complete`` (auth_failed/infra_error/…).
    """

    run_id: int
    case_ext_id: str
    competitor_name: str
    status: str
    outcome: str
    findings: list[FindingScore] = field(default_factory=list)
    judge_cost: float = 0.0

    @property
    def eligible(self) -> bool:
        """Counts toward the detection-rate denominator (hit or genuine miss)."""
        return self.outcome in ELIGIBLE_OUTCOMES

    @property
    def is_hit(self) -> bool:
        return self.outcome == "hit"


def _outcome_from_findings(findings: list[FindingScore]) -> str:
    """Reduce per-finding results to a run outcome (for a complete run)."""
    localized = [f for f in findings if f.localized]
    if not localized:
        return "miss"  # complete run, nothing even near ground truth
    if any(f.truth is not None and f.truth.same_bug is True for f in localized):
        return "hit"
    # No confirmed hit. If any localized finding is undetermined (judge failed or
    # was never run), the outcome can't be a miss — it's undetermined.
    if any(f.truth is None or f.truth.error is not None for f in localized):
        return "judge_error"
    return "miss"


class Scorer:
    """Localizes + truth-judges run findings and persists/aggregates the result."""

    def __init__(
        self,
        db: Database,
        judge: TruthJudge,
        *,
        tolerance: int = DEFAULT_LINE_TOLERANCE,
    ):
        self.db = db
        self.judge = judge
        self.tolerance = tolerance

    def _labels(self, run: Any) -> tuple[Case, str]:
        """Load the run's case (with ground truth) and competitor name."""
        from .corpus import Case

        case_row = self.db.get_case_by_id(run["case_id"])
        if case_row is None:
            raise ValueError(f"run {run['id']} references missing case")
        comp_row = self.db.get_competitor_by_id(run["competitor_id"])
        comp_name = comp_row["name"] if comp_row is not None else "?"
        return Case.from_row(case_row), comp_name

    def score_run(self, run_id: int) -> RunScore:
        """Score one run: localize every finding, judge the localized ones, persist.

        Calls the truth judge — use :meth:`load_run_score` to rebuild a RunScore
        from already-persisted columns without re-spending judge budget.
        """
        run = self.db.get_run(run_id)
        if run is None:
            raise ValueError(f"run {run_id} not found")
        case, comp_name = self._labels(run)

        if run["status"] != "complete":
            # Integrity rule: a run that never got a fair look is excluded, not a
            # miss. No findings are scored.
            return RunScore(
                run_id, case.ext_id, comp_name, run["status"], outcome="excluded"
            )

        scores: list[FindingScore] = []
        judge_cost = 0.0
        for row in self.db.run_findings(run_id):
            loc = localize(
                row["file"], row["line_start"], case.gt_hunks, self.tolerance
            )
            truth: TruthVerdict | None = None
            if loc.matched:
                truth = self.judge.judge(case, ReportedFinding.from_row(row))
                judge_cost += truth.cost_usd or 0.0
                self.db.add_judgment(
                    target_kind="truth",
                    target_id=row["id"],
                    judge_model=self.judge.name,
                    verdict=truth.label,
                    reasoning=truth.reasoning,
                    tokens_in=truth.tokens_in,
                    tokens_out=truth.tokens_out,
                    cost_usd=truth.cost_usd,
                )
            self.db.record_finding_score(
                row["id"],
                matches_ground_truth=loc.matched,
                judge_truth_verdict=truth.label if truth is not None else None,
                judge_reasoning=truth.reasoning if truth is not None else None,
            )
            scores.append(
                FindingScore(
                    finding_id=row["id"],
                    file=row["file"],
                    line=row["line_start"],
                    localized=loc.matched,
                    truth=truth,
                )
            )

        return RunScore(
            run_id,
            case.ext_id,
            comp_name,
            run["status"],
            outcome=_outcome_from_findings(scores),
            findings=scores,
            judge_cost=judge_cost,
        )

    def load_run_score(self, run_id: int) -> RunScore:
        """Rebuild a RunScore from persisted columns, without calling the judge."""
        run = self.db.get_run(run_id)
        if run is None:
            raise ValueError(f"run {run_id} not found")
        case, comp_name = self._labels(run)
        if run["status"] != "complete":
            return RunScore(
                run_id, case.ext_id, comp_name, run["status"], outcome="excluded"
            )

        scores: list[FindingScore] = []
        for row in self.db.run_findings(run_id):
            localized = bool(row["matches_ground_truth"])
            truth: TruthVerdict | None = None
            verdict = row["judge_truth_verdict"]
            if localized and verdict is not None:
                truth = _verdict_from_label(verdict, row["judge_reasoning"] or "")
            elif localized:
                truth = None  # localized but never judged -> undetermined
            scores.append(
                FindingScore(
                    finding_id=row["id"],
                    file=row["file"],
                    line=row["line_start"],
                    localized=localized,
                    truth=truth,
                )
            )

        judge_cost = sum(
            j["cost_usd"] or 0.0
            for f in scores
            for j in self.db.judgments(target_kind="truth", target_id=f.finding_id)
        )
        return RunScore(
            run_id,
            case.ext_id,
            comp_name,
            run["status"],
            outcome=_outcome_from_findings(scores),
            findings=scores,
            judge_cost=judge_cost,
        )

    def needs_scoring(self, run_id: int) -> bool:
        """True if scoring this complete run would do new work (localize/judge).

        A run needs scoring if any finding is not yet localized, or any localized
        finding has not yet been judged. A zero-finding complete run reports as
        not-needing (its outcome is a settled miss, recomputable for free).
        """
        run = self.db.get_run(run_id)
        if run is None or run["status"] != "complete":
            return False
        for row in self.db.run_findings(run_id):
            if row["matches_ground_truth"] is None:
                return True
            if row["matches_ground_truth"] and row["judge_truth_verdict"] is None:
                return True
        return False


def _verdict_from_label(label: str, reasoning: str) -> TruthVerdict:
    """Inverse of TruthVerdict.label for verdicts read back from the DB."""
    if label.startswith("error:"):
        return TruthVerdict(
            None, reasoning=reasoning, error=label[len("error:") :].strip()
        )
    return TruthVerdict(label == "same_bug", reasoning=reasoning)


# -- Detection report --------------------------------------------------------


@dataclass
class CompetitorDetection:
    """Aggregated detection metrics for one competitor."""

    competitor_name: str
    hits: int = 0
    misses: int = 0
    judge_error: int = 0  # undetermined — excluded from the denominator
    excluded: int = 0  # auth_failed / infra_error — never a miss
    judge_cost: float = 0.0

    @property
    def eligible(self) -> int:
        """Cases that counted: confirmed hits + genuine misses."""
        return self.hits + self.misses

    @property
    def detection_rate(self) -> float:
        """Recall over eligible cases (0.0 when nothing was eligible)."""
        return self.hits / self.eligible if self.eligible else 0.0


def detection_report(run_scores: list[RunScore]) -> list[CompetitorDetection]:
    """Roll RunScores up per competitor, sorted alphabetically by competitor name."""
    by_name: dict[str, CompetitorDetection] = {}
    for rs in run_scores:
        d = by_name.setdefault(
            rs.competitor_name, CompetitorDetection(rs.competitor_name)
        )
        d.judge_cost += rs.judge_cost
        if rs.outcome == "hit":
            d.hits += 1
        elif rs.outcome == "miss":
            d.misses += 1
        elif rs.outcome == "judge_error":
            d.judge_error += 1
        else:  # excluded
            d.excluded += 1
    return [by_name[name] for name in sorted(by_name)]
