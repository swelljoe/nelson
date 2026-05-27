"""P3 scoring: localization gate, truth-verdict parsing, and the orchestrated
score with a fake judge.

The truth judge is injectable, so the full score_run flow — localize, judge,
persist, reduce to an outcome — is exercised without calling ``claude``.
"""

import pytest

from nelson.corpus import Case
from nelson.db import Database
from nelson.score import (
    ClaudeTruthJudge,
    CompetitorDetection,
    ReportedFinding,
    Scorer,
    TruthVerdict,
    build_truth_prompt,
    detection_report,
    localize,
    parse_truth_verdict,
    paths_match,
)

# -- Localization gate -------------------------------------------------------

HUNKS = [
    {"file": "src/main/java/Foo.java", "start": 55, "end": 61},
    {"file": "src/main/java/Foo.java", "start": 73, "end": 79},
]


def test_localize_inside_hunk():
    assert localize("src/main/java/Foo.java", 76, HUNKS).matched
    assert localize("src/main/java/Foo.java", 55, HUNKS).matched


def test_localize_within_tolerance_but_outside_hunk():
    # line 65 is between the two hunks; within 10 of hunk-end 61.
    assert localize("src/main/java/Foo.java", 65, HUNKS, tolerance=10).matched
    # 30 lines off-target with a tight tolerance does not localize.
    assert not localize("src/main/java/Foo.java", 30, HUNKS, tolerance=5).matched


def test_localize_returns_matched_hunk():
    loc = localize("src/main/java/Foo.java", 76, HUNKS)
    assert loc.hunk == {"file": "src/main/java/Foo.java", "start": 73, "end": 79}


def test_localize_wrong_file_never_matches():
    assert not localize("src/main/java/Bar.java", 76, HUNKS).matched


def test_localize_requires_a_line():
    assert not localize("src/main/java/Foo.java", None, HUNKS).matched


def test_paths_match_suffix_and_mount_prefix():
    # Model reports a mount-rooted or shorter path than the diff's old-side path.
    assert paths_match("/src/src/main/java/Foo.java", "src/main/java/Foo.java")
    assert paths_match("Foo.java", "src/main/java/Foo.java")
    assert paths_match("a/b/Foo.java", "Foo.java")


def test_paths_match_rejects_different_files():
    assert not paths_match("src/main/java/Other.java", "src/main/java/Foo.java")
    assert not paths_match(None, "x")
    # A bare-name suffix must align on a path component, not a substring.
    assert not paths_match("xFoo.java", "Foo.java")


# -- Truth-verdict parsing ---------------------------------------------------


def test_parse_truth_verdict_true_false():
    assert parse_truth_verdict('{"same_bug": true, "reasoning": "y"}') == (True, "y")
    assert parse_truth_verdict('{"same_bug": false, "reasoning": "n"}') == (False, "n")


def test_parse_truth_verdict_tolerates_prose_and_string_bools():
    text = 'Here is my call:\n```json\n{"same_bug": "yes", "reasoning": "matches"}\n```'
    assert parse_truth_verdict(text) == (True, "matches")


def test_parse_truth_verdict_none_when_absent():
    assert parse_truth_verdict("I cannot decide.") is None
    # Present key but uncoercible value -> unusable, not a False verdict.
    assert parse_truth_verdict('{"same_bug": "maybe"}') is None


def test_truth_verdict_label():
    assert TruthVerdict(True).label == "same_bug"
    assert TruthVerdict(False).label == "different_bug"
    assert TruthVerdict(None, error="timeout").label == "error: timeout"


def test_build_truth_prompt_includes_advisory_and_finding():
    case = Case(
        source="cvd", ext_id="GHSA-x", cwe="CWE-22", description="path traversal"
    )
    finding = ReportedFinding(file="Foo.java", line=76, description="zip slip")
    prompt = build_truth_prompt(case, finding)
    assert "CWE-22" in prompt
    assert "path traversal" in prompt
    assert "Foo.java:76" in prompt
    assert "zip slip" in prompt


# -- Orchestrated scoring (fake judge) ---------------------------------------


class FakeJudge:
    """Returns a canned verdict; records how many times it was asked."""

    name = "fake-judge"

    def __init__(self, verdict: TruthVerdict):
        self.verdict = verdict
        self.calls = 0

    def judge(self, case, finding):
        self.calls += 1
        return self.verdict


def _case(db: Database) -> int:
    return db.upsert_case(
        {
            "source": "cvd",
            "ext_id": "GHSA-test",
            "cwe": "CWE-22",
            "description": "path traversal",
            "gt_files": ["Foo.java"],
            "gt_hunks": [{"file": "Foo.java", "start": 73, "end": 79}],
        }
    )


def _complete_run(db: Database, case_id: int, *, findings: list[tuple[str, int]]):
    comp_id = db.upsert_competitor({"name": "claude-code/sonnet", "model": "sonnet"})
    run_id = db.create_run(case_id, comp_id)
    db.start_run(run_id, container_id="c1")
    db.complete_run(run_id, tokens_in=1, tokens_out=1, cost_usd=0.01, wall_clock_s=1.0)
    for file, line in findings:
        db.add_run_finding(
            run_id, file=file, line_start=line, line_end=line, description="x"
        )
    return run_id


def test_score_run_hit_persists_localization_and_verdict(tmp_path):
    db = Database(tmp_path / "t.db")
    case_id = _case(db)
    run_id = _complete_run(db, case_id, findings=[("Foo.java", 76)])
    judge = FakeJudge(TruthVerdict(True, reasoning="same zip-slip", cost_usd=0.05))

    rs = Scorer(db, judge).score_run(run_id)

    assert rs.outcome == "hit"
    assert rs.is_hit and rs.eligible
    assert judge.calls == 1
    assert rs.judge_cost == 0.05
    # Persisted: localization + truth verdict on the finding.
    f = db.run_findings(run_id)[0]
    assert f["matches_ground_truth"] == 1
    assert f["judge_truth_verdict"] == "same_bug"
    assert f["judge_reasoning"] == "same zip-slip"
    # Logged to the judgments ledger for separate cost accounting.
    js = db.judgments(target_kind="truth")
    assert len(js) == 1 and js[0]["cost_usd"] == 0.05


def test_score_run_off_target_finding_is_a_miss_without_judging(tmp_path):
    db = Database(tmp_path / "t.db")
    case_id = _case(db)
    # Line 10 is far from the hunk (73-79) -> not localized, judge never called.
    run_id = _complete_run(db, case_id, findings=[("Foo.java", 10)])
    judge = FakeJudge(TruthVerdict(True))

    rs = Scorer(db, judge).score_run(run_id)

    assert rs.outcome == "miss"
    assert judge.calls == 0
    f = db.run_findings(run_id)[0]
    assert f["matches_ground_truth"] == 0
    assert f["judge_truth_verdict"] is None


def test_score_run_localized_but_different_bug_is_a_miss(tmp_path):
    db = Database(tmp_path / "t.db")
    case_id = _case(db)
    run_id = _complete_run(db, case_id, findings=[("Foo.java", 76)])
    judge = FakeJudge(TruthVerdict(False, reasoning="different flaw"))

    rs = Scorer(db, judge).score_run(run_id)

    assert rs.outcome == "miss"
    assert rs.eligible  # a genuine miss still counts in the denominator
    assert judge.calls == 1


def test_score_run_judge_failure_is_undetermined_not_a_miss(tmp_path):
    db = Database(tmp_path / "t.db")
    case_id = _case(db)
    run_id = _complete_run(db, case_id, findings=[("Foo.java", 76)])
    judge = FakeJudge(TruthVerdict(None, error="timeout"))

    rs = Scorer(db, judge).score_run(run_id)

    # Integrity: a judge failure must never turn a localized finding into a miss.
    assert rs.outcome == "judge_error"
    assert not rs.eligible
    f = db.run_findings(run_id)[0]
    assert f["matches_ground_truth"] == 1
    assert f["judge_truth_verdict"] == "error: timeout"


def test_score_run_excludes_infra_error_run(tmp_path):
    db = Database(tmp_path / "t.db")
    case_id = _case(db)
    comp_id = db.upsert_competitor({"name": "claude-code/sonnet", "model": "sonnet"})
    run_id = db.create_run(case_id, comp_id)
    db.start_run(run_id, container_id="c1")
    db.mark_run_infra_error(run_id, "container died")
    judge = FakeJudge(TruthVerdict(True))

    rs = Scorer(db, judge).score_run(run_id)

    assert rs.outcome == "excluded"
    assert not rs.eligible
    assert judge.calls == 0


def test_score_run_hit_when_any_localized_finding_matches(tmp_path):
    # Two localized findings: one different, one same -> overall hit.
    db = Database(tmp_path / "t.db")
    case_id = _case(db)
    run_id = _complete_run(db, case_id, findings=[("Foo.java", 74), ("Foo.java", 78)])

    class SeqJudge:
        name = "seq"

        def __init__(self):
            self.verdicts = [TruthVerdict(False), TruthVerdict(True)]
            self.calls = 0

        def judge(self, case, finding):
            v = self.verdicts[self.calls]
            self.calls += 1
            return v

    rs = Scorer(db, SeqJudge()).score_run(run_id)
    assert rs.outcome == "hit"


# -- load_run_score / needs_scoring ------------------------------------------


def test_load_run_score_rebuilds_without_judging(tmp_path):
    db = Database(tmp_path / "t.db")
    case_id = _case(db)
    run_id = _complete_run(db, case_id, findings=[("Foo.java", 76)])
    judge = FakeJudge(TruthVerdict(True, reasoning="r", cost_usd=0.05))
    scorer = Scorer(db, judge)

    scorer.score_run(run_id)  # first pass judges + persists
    assert judge.calls == 1
    assert not scorer.needs_scoring(run_id)  # now settled

    reloaded = scorer.load_run_score(run_id)
    assert judge.calls == 1  # no new judge calls
    assert reloaded.outcome == "hit"
    assert reloaded.judge_cost == 0.05  # summed from the ledger


def test_needs_scoring_true_until_judged(tmp_path):
    db = Database(tmp_path / "t.db")
    case_id = _case(db)
    run_id = _complete_run(db, case_id, findings=[("Foo.java", 76)])
    scorer = Scorer(db, FakeJudge(TruthVerdict(True)))
    assert scorer.needs_scoring(run_id)
    scorer.score_run(run_id)
    assert not scorer.needs_scoring(run_id)


# -- Detection report --------------------------------------------------------


def test_detection_report_rates_and_exclusions():
    from nelson.score import RunScore

    scores = [
        RunScore(1, "c1", "alpha", "complete", "hit", judge_cost=0.1),
        RunScore(2, "c2", "alpha", "complete", "miss"),
        RunScore(3, "c3", "alpha", "auth_failed", "excluded"),
        RunScore(4, "c4", "alpha", "complete", "judge_error"),
        RunScore(5, "c1", "beta", "complete", "hit"),
    ]
    report = {d.competitor_name: d for d in detection_report(scores)}

    alpha = report["alpha"]
    assert alpha.hits == 1 and alpha.misses == 1
    assert alpha.excluded == 1 and alpha.judge_error == 1
    assert alpha.eligible == 2  # excluded + judge_error not in the denominator
    assert alpha.detection_rate == 0.5
    assert alpha.judge_cost == pytest.approx(0.1)
    assert report["beta"].detection_rate == 1.0


def test_competitor_detection_zero_eligible_is_zero_rate():
    d = CompetitorDetection("x", hits=0, misses=0, excluded=3)
    assert d.eligible == 0
    assert d.detection_rate == 0.0


# -- Case rollup (one run per file) ------------------------------------------


def test_case_scores_rolls_files_up_any_hit_wins():
    from nelson.score import RunScore, case_scores

    # Same case, two file-runs: a miss and a hit -> the case is a hit.
    runs = [
        RunScore(1, "GHSA-a", "alpha", "complete", "miss"),
        RunScore(2, "GHSA-a", "alpha", "complete", "hit", judge_cost=0.03),
    ]
    cases = case_scores(runs)
    assert len(cases) == 1
    assert cases[0].outcome == "hit"
    assert cases[0].judge_cost == pytest.approx(0.03)


def test_case_scores_undetermined_beats_miss():
    from nelson.score import RunScore, case_scores

    # No hit; one file undetermined -> the case is judge_error, never a miss.
    runs = [
        RunScore(1, "GHSA-b", "alpha", "complete", "miss"),
        RunScore(2, "GHSA-b", "alpha", "complete", "judge_error"),
    ]
    assert case_scores(runs)[0].outcome == "judge_error"


def test_detection_report_counts_cases_not_runs():
    from nelson.score import RunScore

    # One case, three file-runs (2 miss + 1 hit). Detection is per *case*: 1/1.
    runs = [
        RunScore(1, "GHSA-c", "alpha", "complete", "miss"),
        RunScore(2, "GHSA-c", "alpha", "complete", "hit"),
        RunScore(3, "GHSA-c", "alpha", "complete", "miss"),
    ]
    report = detection_report(runs)
    assert len(report) == 1
    assert report[0].hits == 1
    assert report[0].misses == 0
    assert report[0].eligible == 1
    assert report[0].detection_rate == 1.0


# -- Real judge wiring (no network) ------------------------------------------


def test_claude_truth_judge_is_a_truthjudge():
    from nelson.score import TruthJudge

    assert isinstance(ClaudeTruthJudge(), TruthJudge)
