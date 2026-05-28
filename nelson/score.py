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
from pathlib import Path
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


def _repo_relative(path: str) -> str:
    """The repo-root-relative path for a finding, for ``git show vuln_commit:…``.

    Findings report paths *relative to the /src mount* (which is the repo root),
    so they are already repo-relative; we only peel a redundant absolute ``/src/``
    mount prefix or a leading ``./`` a model might prepend. Unlike
    :func:`_normalize_path` (used for suffix *matching*) this never strips a real
    top-level ``src/`` directory — ``git show`` needs the exact path.
    """
    p = path.strip().replace("\\", "/")
    if p.startswith("/src/"):
        p = p[len("/src/") :]
    elif p.startswith("./"):
        p = p[2:]
    elif p.startswith("/"):
        return ""

    parts = [part for part in p.split("/") if part and part != "."]
    if parts and parts[0].endswith(":"):
        parts = parts[1:]
    if not parts:
        return ""
    if any(part == ".." or ":" in part for part in parts):
        return ""
    return "/".join(parts)


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


# -- FP judge (precision) ----------------------------------------------------
#
# Every reported finding that is NOT the confirmed target bug — a finding that
# didn't localize, OR one that localized but the truth judge ruled a *different*
# bug — is a precision candidate: a real bug the model usefully found, or noise?
# The FP judge reads the actual pre-patch source and rules confirmed /
# false_positive / needs_review. Unlike the truth judge it is NEVER shown the
# advisory (that would invite circularity / over-trust); its judge() takes only
# the finding + source, so the advisory cannot leak into precision by construction.

# Bound the source sent to the judge. A finding deep in a large file gets a window
# centred on its line (so the flagged code and nearby callers survive) rather than
# a blind head-truncation.
_MAX_CODE_CHARS = 16000
_FP_CONTEXT_LINES = 200

FP_INSTRUCTIONS = """\
You are a senior security engineer triaging a single finding from an automated \
vulnerability scanner. You are NOT told whether the code contains a known bug — \
do not assume the scanner is right. Read the actual source and decide whether the \
reported finding is a REAL, exploitable security vulnerability or a FALSE POSITIVE.

Trace whether attacker-controlled input can reach the flagged code, account for \
validation or mitigations already present, and weigh realistic exploitability over \
theoretical possibility.

Reply with ONLY a JSON object:
{"verdict": "<confirmed|false_positive|needs_review>", "reasoning": "<why>"}
where confirmed = a real, exploitable bug, false_positive = not a real/exploitable \
security bug, and needs_review = you genuinely cannot tell from the code shown."""


@dataclass
class FPVerdict:
    """Outcome of FP-judging one non-target finding.

    ``is_real`` is True (a real bug), False (a false positive), or None when the
    judge could not decide. None splits two ways via ``error``: a clean
    ``needs_review`` (error is None) versus a judge *failure* (error set). Either
    way a None verdict is *undetermined* — never counted as a false positive
    (that would penalize the model for the judge's indecision) nor as a real bug.
    """

    is_real: bool | None
    reasoning: str = ""
    tokens_in: int | None = None
    tokens_out: int | None = None
    cost_usd: float | None = None
    error: str | None = None

    @property
    def label(self) -> str:
        """Compact verdict string persisted to ``judge_fp_verdict``."""
        if self.error is not None:
            return f"error: {self.error}"
        if self.is_real is None:
            return "needs_review"
        return "real_bug" if self.is_real else "false_positive"


@runtime_checkable
class CodeProvider(Protocol):
    def source(self, case: Case, file: str | None) -> str | None: ...


@runtime_checkable
class FPJudge(Protocol):
    name: str

    def judge(self, finding: ReportedFinding, source: str | None) -> FPVerdict: ...


class _GitShow(Protocol):
    """The slice of a GitRunner the code provider needs (prepare + show)."""

    def prepare(self, repo_url: str, commit: str, dest: Path) -> None: ...
    def show(self, dest: Path, rev: str, path: str) -> str: ...


class GitCodeProvider:
    """Reads a finding's pre-patch source via ``git show vuln_commit:path``.

    The competitor audited the tree at ``case.vuln_commit``, so that revision is
    the code the FP judge must reason about. A per-(repo, commit) shallow fetch
    and resolved file contents are both cached, so scoring many findings in one
    file costs one fetch + one show. A file absent at that revision (a path the
    model mis-reported, or a generated file) yields None, which the judge surfaces
    as undetermined — never a false positive.
    """

    def __init__(
        self,
        git: _GitShow | None = None,
        *,
        root: Path | str | None = None,
        timeout: float = 180.0,
    ):
        from .derive import SubprocessGitRunner

        self.git: _GitShow = git or SubprocessGitRunner(timeout=timeout)
        self.root = Path(root) if root else Path(".nelson_cache/fpjudge")
        self._prepared: set[tuple[str, str]] = set()
        self._cache: dict[tuple[str, str, str], str | None] = {}

    def source(self, case: Case, file: str | None) -> str | None:
        if not file or not case.repo_url or not case.vuln_commit:
            return None
        from .derive import GitError, _repo_slug

        norm = _repo_relative(file)
        if not norm:
            return None
        key = (case.repo_url, case.vuln_commit, norm)
        if key in self._cache:
            return self._cache[key]
        dest = self.root / _repo_slug(case.repo_url)
        text: str | None
        try:
            prep_key = (case.repo_url, case.vuln_commit)
            if prep_key not in self._prepared:
                self.git.prepare(case.repo_url, case.vuln_commit, dest)
                self._prepared.add(prep_key)
            text = self.git.show(dest, case.vuln_commit, norm)
        except GitError:
            text = None
        self._cache[key] = text
        return text


def _clip_source(source: str, line: int | None) -> tuple[str, str]:
    """Bound the source for the prompt; returns (text, header note).

    Small files are sent whole. A large file is windowed to ±_FP_CONTEXT_LINES
    around the finding's line, and the note records the absolute line range so
    the judge can map the reported line number onto the excerpt.
    """
    if len(source) <= _MAX_CODE_CHARS:
        return source, ""
    lines = source.splitlines()
    if line is not None and 1 <= line <= len(lines):
        lo = max(0, line - 1 - _FP_CONTEXT_LINES)
        hi = min(len(lines), line + _FP_CONTEXT_LINES)
        window = "\n".join(lines[lo:hi])
        excerpt = window[:_MAX_CODE_CHARS]
        if len(window) > _MAX_CODE_CHARS:
            return (
                excerpt,
                f" (lines {lo + 1}-{hi} of {len(lines)}, clipped by char limit; "
                "excerpt may end mid-window)",
            )
        return excerpt, f" (lines {lo + 1}-{hi} of {len(lines)}, clipped)"
    return source[:_MAX_CODE_CHARS], " (clipped to first portion)"


def build_fp_prompt(finding: ReportedFinding, source: str) -> str:
    """Assemble the finding + its source into an FP-judge prompt.

    Deliberately takes no Case: the advisory must never reach the FP judge.
    """
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
    clipped, note = _clip_source(source, finding.line)
    return (
        f"{FP_INSTRUCTIONS}\n\n"
        f"## Reported finding\n{reported}\n\n"
        f"## Source: {finding.file or '?'}{note}\n"
        f"```\n{clipped}\n```"
    )


_FP_REAL = {"confirmed", "real", "real_bug", "true_positive", "yes", "vulnerable"}
_FP_FALSE = {"false_positive", "false", "fp", "not_exploitable", "no", "safe"}
_FP_REVIEW = {"needs_review", "unsure", "unknown", "uncertain", "maybe", "review"}


def parse_fp_verdict(text: str) -> tuple[bool | None, str] | None:
    """Extract (is_real, reasoning) from the judge's reply, or None if unusable.

    The inner ``is_real`` is None for an explicit ``needs_review``; the outer
    None means nothing parseable was found, so the caller records a judge failure
    (the two are different: a deliberate "can't tell" vs a broken reply).
    """
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        data = _loads_object(match.group(0))
        if data is not None and "verdict" in data:
            verdict = str(data["verdict"]).strip().lower()
            reasoning = str(data.get("reasoning", ""))
            if verdict in _FP_REAL:
                return True, reasoning
            if verdict in _FP_FALSE:
                return False, reasoning
            if verdict in _FP_REVIEW:
                return None, reasoning
    return None


class ClaudeFPJudge:
    """FP judge backed by ``claude -p`` (Opus by default).

    Mirrors the truth / pre-vet judges: host CLI, subscription auth, no container,
    failures surfaced not guessed. Its ``judge`` signature carries no Case, so the
    advisory cannot leak into the precision side of scoring.
    """

    name = "claude-cli"

    def __init__(self, model: str = "opus", timeout: int = 180):
        self.model = model
        self.timeout = timeout

    def judge(self, finding: ReportedFinding, source: str | None) -> FPVerdict:
        if source is None:
            # No code to ground the judgment -> undetermined, never an FP.
            return FPVerdict(None, error="source unavailable")
        prompt = build_fp_prompt(finding, source)
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
            return FPVerdict(None, error="timeout")

        raw, stderr = result.stdout, result.stderr
        if result.returncode != 0:
            kind = classify_failure(raw + stderr, failed=True)
            return FPVerdict(
                None,
                error=f"claude exit {result.returncode} ({kind}): {stderr[:200]}",
            )

        text, tin, tout, cost = _unwrap_claude_json(raw)
        parsed = parse_fp_verdict(text)
        if parsed is None:
            return FPVerdict(
                None,
                reasoning=text[:300],
                tokens_in=tin,
                tokens_out=tout,
                cost_usd=cost,
                error="unparseable judge reply",
            )
        is_real, reasoning = parsed
        return FPVerdict(
            is_real,
            reasoning=reasoning,
            tokens_in=tin,
            tokens_out=tout,
            cost_usd=cost,
        )


# -- Scorer ------------------------------------------------------------------


def _fp_eligible(localized: bool, truth: TruthVerdict | None) -> bool:
    """Is this finding a precision candidate (i.e. should it be FP-judged)?

    Yes for any finding that is neither the confirmed target hit nor a localized
    finding the truth judge left undetermined. So: a non-localized finding, or a
    localized finding the judge ruled a *different* bug.
    """
    if not localized:
        return True  # a non-localized finding is always a precision candidate
    # Localized: a precision item only if the truth judge *decided* a different
    # bug. A same_bug hit, an undetermined verdict, or no verdict are all skipped.
    return truth is not None and truth.same_bug is False


@dataclass
class FindingScore:
    """Per-finding scoring detail."""

    finding_id: int
    file: str | None
    line: int | None
    localized: bool
    truth: TruthVerdict | None = None  # None unless the finding reached the judge
    fp: FPVerdict | None = None  # None unless the finding reached the FP judge

    @property
    def is_target_hit(self) -> bool:
        """The localized finding the truth judge confirmed is the same bug."""
        return self.localized and self.truth is not None and self.truth.same_bug is True

    @property
    def fp_category(self) -> str | None:
        """Precision bucket, or None if this finding is not a precision item.

        None = the confirmed target hit, or a localized finding the truth judge
        left undetermined (a candidate target, not noise). Otherwise:
        ``real_other`` (a genuine *different* bug — credited, not penalized),
        ``false_positive``, or ``undetermined`` (needs_review, FP-judge failure,
        or FP judge not yet run).
        """
        if self.is_target_hit:
            return None
        if self.localized and (self.truth is None or self.truth.error is not None):
            return None
        if self.fp is None:
            return "undetermined"
        if self.fp.is_real is True:
            return "real_other"
        if self.fp.is_real is False:
            return "false_positive"
        return "undetermined"


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
    judge_cost: float = 0.0  # truth-judge spend (detection)
    fp_cost: float = 0.0  # FP-judge spend (precision)

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
    """Localizes + truth-judges run findings, then (if an FP judge is wired)
    FP-judges the non-target findings, persisting/aggregating both.

    Detection scoring needs only the truth judge. Precision scoring is opt-in:
    pass both ``fp_judge`` and ``code`` (a CodeProvider) and ``score_run`` will
    also FP-judge every non-target finding. Without them, scoring is detection
    only, exactly as in P3."""

    def __init__(
        self,
        db: Database,
        judge: TruthJudge,
        *,
        tolerance: int = DEFAULT_LINE_TOLERANCE,
        fp_judge: FPJudge | None = None,
        code: CodeProvider | None = None,
    ):
        self.db = db
        self.judge = judge
        self.tolerance = tolerance
        self.fp_judge = fp_judge
        self.code = code

    @property
    def _scores_precision(self) -> bool:
        return self.fp_judge is not None and self.code is not None

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
        """Score one run: localize, truth-judge the localized findings, and (if a
        FP judge is wired) FP-judge the non-target findings; persist all of it.

        Calls the judge(s) — use :meth:`load_run_score` to rebuild a RunScore
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
        fp_cost = 0.0
        for row in self.db.run_findings(run_id):
            reported = ReportedFinding.from_row(row)
            loc = localize(
                row["file"], row["line_start"], case.gt_hunks, self.tolerance
            )
            truth: TruthVerdict | None = None
            if loc.matched:
                truth = self.judge.judge(case, reported)
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

            # Precision: every finding that isn't the target hit (and isn't a
            # localized-but-undetermined candidate) is FP-judged against the code.
            fp: FPVerdict | None = None
            if (
                self.fp_judge is not None
                and self.code is not None
                and _fp_eligible(loc.matched, truth)
            ):
                source = self.code.source(case, row["file"])
                fp = self.fp_judge.judge(reported, source)
                fp_cost += fp.cost_usd or 0.0
                self.db.add_judgment(
                    target_kind="fp",
                    target_id=row["id"],
                    judge_model=self.fp_judge.name,
                    verdict=fp.label,
                    reasoning=fp.reasoning,
                    tokens_in=fp.tokens_in,
                    tokens_out=fp.tokens_out,
                    cost_usd=fp.cost_usd,
                )
                self.db.record_fp_verdict(row["id"], verdict=fp.label)

            scores.append(
                FindingScore(
                    finding_id=row["id"],
                    file=row["file"],
                    line=row["line_start"],
                    localized=loc.matched,
                    truth=truth,
                    fp=fp,
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
            fp_cost=fp_cost,
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
            fp: FPVerdict | None = None
            fp_label = row["judge_fp_verdict"]
            if fp_label is not None:
                ledger = self.db.judgments(target_kind="fp", target_id=row["id"])
                reasoning = ledger[-1]["reasoning"] if ledger else ""
                fp = _fp_verdict_from_label(fp_label, reasoning or "")
            scores.append(
                FindingScore(
                    finding_id=row["id"],
                    file=row["file"],
                    line=row["line_start"],
                    localized=localized,
                    truth=truth,
                    fp=fp,
                )
            )

        judge_cost = sum(
            j["cost_usd"] or 0.0
            for f in scores
            for j in self.db.judgments(target_kind="truth", target_id=f.finding_id)
        )
        fp_cost = sum(
            j["cost_usd"] or 0.0
            for f in scores
            for j in self.db.judgments(target_kind="fp", target_id=f.finding_id)
        )
        return RunScore(
            run_id,
            case.ext_id,
            comp_name,
            run["status"],
            outcome=_outcome_from_findings(scores),
            findings=scores,
            judge_cost=judge_cost,
            fp_cost=fp_cost,
        )

    def needs_scoring(self, run_id: int) -> bool:
        """True if scoring this complete run would do new work (localize/judge).

        A run needs scoring if any finding is not yet localized, any localized
        finding has not yet been truth-judged, or — when this Scorer does
        precision — any FP-eligible finding has not yet been FP-judged. A
        zero-finding complete run reports as not-needing (its outcome is a settled
        miss, recomputable for free).
        """
        run = self.db.get_run(run_id)
        if run is None or run["status"] != "complete":
            return False
        for row in self.db.run_findings(run_id):
            if row["matches_ground_truth"] is None:
                return True
            if row["matches_ground_truth"] and row["judge_truth_verdict"] is None:
                return True
            if (
                self._scores_precision
                and row["judge_fp_verdict"] is None
                and _persisted_fp_eligible(
                    row["matches_ground_truth"], row["judge_truth_verdict"]
                )
            ):
                return True
        return False


def _persisted_fp_eligible(matches_ground_truth: Any, truth_verdict: Any) -> bool:
    """``_fp_eligible`` reconstructed from the persisted truth columns.

    Reached only after localization is recorded (matches_ground_truth set), so a
    localized finding's truth verdict is already settled to same_bug /
    different_bug / error: … here.
    """
    if not bool(matches_ground_truth):
        return True
    # Localized: a precision item only if the truth judge decided a *different*
    # bug (a settled non-same_bug, non-error verdict).
    return truth_verdict == "different_bug"


def _verdict_from_label(label: str, reasoning: str) -> TruthVerdict:
    """Inverse of TruthVerdict.label for verdicts read back from the DB."""
    if label.startswith("error:"):
        return TruthVerdict(
            None, reasoning=reasoning, error=label[len("error:") :].strip()
        )
    return TruthVerdict(label == "same_bug", reasoning=reasoning)


def _fp_verdict_from_label(label: str, reasoning: str = "") -> FPVerdict:
    """Inverse of FPVerdict.label for verdicts read back from the DB."""
    if label.startswith("error:"):
        return FPVerdict(
            None, reasoning=reasoning, error=label[len("error:") :].strip()
        )
    if label == "needs_review":
        return FPVerdict(None, reasoning=reasoning)
    return FPVerdict(label == "real_bug", reasoning=reasoning)


# -- Case rollup -------------------------------------------------------------


@dataclass
class CaseScore:
    """A (competitor, case) outcome, rolled up from its per-file runs.

    The file-scoped harness runs one container per (competitor, case, file), so a
    case has one RunScore per vulnerable file. The case is detected if *any* of
    those file-runs is a hit.
    """

    competitor_name: str
    case_ext_id: str
    outcome: str
    runs: list[RunScore] = field(default_factory=list)
    judge_cost: float = 0.0
    fp_cost: float = 0.0

    @property
    def eligible(self) -> bool:
        return self.outcome in ELIGIBLE_OUTCOMES

    @property
    def is_hit(self) -> bool:
        return self.outcome == "hit"


# Best-to-worst precedence for rolling file-run outcomes into a case outcome:
# any hit wins; failing that, an undetermined file (judge_error) keeps the case
# out of the denominator rather than calling it a clean miss; a genuine miss
# beats excluded. (Integrity: never count a case as missed when a file that
# carries the bug went unjudged.)
_CASE_OUTCOME_PRECEDENCE = ("hit", "judge_error", "miss", "excluded")


def _rollup_case_outcome(outcomes: Any) -> str:
    seen = set(outcomes)
    for outcome in _CASE_OUTCOME_PRECEDENCE:
        if outcome in seen:
            return outcome
    return "excluded"


def case_scores(run_scores: list[RunScore]) -> list[CaseScore]:
    """Group file-runs into per-(competitor, case) outcomes, first-seen order."""
    groups: dict[tuple[str, str], list[RunScore]] = {}
    for rs in run_scores:
        groups.setdefault((rs.competitor_name, rs.case_ext_id), []).append(rs)
    out: list[CaseScore] = []
    for (comp, case), runs in groups.items():
        out.append(
            CaseScore(
                competitor_name=comp,
                case_ext_id=case,
                outcome=_rollup_case_outcome(r.outcome for r in runs),
                runs=runs,
                judge_cost=sum(r.judge_cost for r in runs),
                fp_cost=sum(r.fp_cost for r in runs),
            )
        )
    return out


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
    """Per-competitor detection, counting **cases** (file-runs rolled up first).

    Detection is a per-case question, so multiple file-runs of one case collapse
    to a single hit/miss before counting. Sorted alphabetically by competitor.
    """
    by_name: dict[str, CompetitorDetection] = {}
    for cs in case_scores(run_scores):
        d = by_name.setdefault(
            cs.competitor_name, CompetitorDetection(cs.competitor_name)
        )
        d.judge_cost += cs.judge_cost
        if cs.outcome == "hit":
            d.hits += 1
        elif cs.outcome == "miss":
            d.misses += 1
        elif cs.outcome == "judge_error":
            d.judge_error += 1
        else:  # excluded
            d.excluded += 1
    return [by_name[name] for name in sorted(by_name)]


# -- Precision report --------------------------------------------------------


@dataclass
class CompetitorPrecision:
    """Aggregated precision metrics for one competitor (over all its findings).

    A finding is a *true finding* if it is the confirmed target bug
    (``target_hits``) or a confirmed real but different bug (``real_others`` —
    credited, never penalized). ``false_positives`` are findings the FP judge
    rejected against the code. ``undetermined`` (needs_review, FP-judge failure,
    or not yet judged) is excluded from precision — the integrity rule again: the
    judge's indecision never counts against the model.
    """

    competitor_name: str
    target_hits: int = 0
    real_others: int = 0
    false_positives: int = 0
    undetermined: int = 0
    cases: int = 0  # distinct cases with a complete run (the FP/case denominator)
    fp_cost: float = 0.0

    @property
    def true_findings(self) -> int:
        return self.target_hits + self.real_others

    @property
    def precision(self) -> float | None:
        """True findings / (true findings + false positives); None if neither."""
        decided = self.true_findings + self.false_positives
        return self.true_findings / decided if decided else None

    @property
    def fp_per_case(self) -> float | None:
        """Mean confirmed false positives per audited case; None if no cases."""
        return self.false_positives / self.cases if self.cases else None


def precision_report(run_scores: list[RunScore]) -> list[CompetitorPrecision]:
    """Per-competitor precision over every reported finding, alphabetical.

    Counts findings (not cases): a noisy model is penalized per false finding,
    while ``fp_per_case`` normalizes the FP count by the cases it audited.
    """
    by_name: dict[str, CompetitorPrecision] = {}
    cases_seen: dict[str, set[str]] = {}
    for rs in run_scores:
        p = by_name.setdefault(
            rs.competitor_name, CompetitorPrecision(rs.competitor_name)
        )
        p.fp_cost += rs.fp_cost
        if rs.status == "complete":
            cases_seen.setdefault(rs.competitor_name, set()).add(rs.case_ext_id)
        for f in rs.findings:
            if f.is_target_hit:
                p.target_hits += 1
                continue
            category = f.fp_category
            if category == "real_other":
                p.real_others += 1
            elif category == "false_positive":
                p.false_positives += 1
            elif category == "undetermined":
                p.undetermined += 1
            # category is None -> candidate-undetermined target; not a precision item
    for name, p in by_name.items():
        p.cases = len(cases_seen.get(name, set()))
    return [by_name[name] for name in sorted(by_name)]
