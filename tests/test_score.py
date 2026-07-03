"""P3 scoring: localization gate, truth-verdict parsing, and the orchestrated
score with a fake judge.

The truth judge is injectable, so the full score_run flow — localize, judge,
persist, reduce to an outcome — is exercised without calling ``claude``.
"""

import pytest

from nelson.corpus import Case
from nelson.db import Database
from nelson.score import (
    ClaudeFPJudge,
    ClaudeRefusalJudge,
    ClaudeTruthJudge,
    CompetitorDetection,
    CompetitorPrecision,
    FindingScore,
    FPVerdict,
    GitCodeProvider,
    RefusalVerdict,
    ReportedFinding,
    RunScore,
    Scorer,
    TruthVerdict,
    build_fp_prompt,
    build_refusal_prompt,
    build_truth_prompt,
    detection_report,
    localize,
    parse_fp_verdict,
    parse_refusal_verdict,
    parse_truth_verdict,
    paths_match,
    precision_report,
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


def test_score_run_localized_but_different_bug_is_a_partial(tmp_path):
    # Right place (within tolerance), wrong bug: half credit, not a miss.
    db = Database(tmp_path / "t.db")
    case_id = _case(db)
    run_id = _complete_run(db, case_id, findings=[("Foo.java", 76)])
    judge = FakeJudge(TruthVerdict(False, reasoning="different flaw"))

    rs = Scorer(db, judge).score_run(run_id)

    assert rs.outcome == "partial"
    assert rs.is_partial
    assert not rs.is_hit
    assert rs.eligible  # a partial still counts in the denominator (where a miss did)
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


def test_detection_report_partial_is_eligible_non_hit():
    # A partial sits in the denominator exactly where a miss did, so it never
    # lifts detection_rate; half_credit_rate surfaces its half credit.
    from nelson.score import RunScore

    scores = [
        RunScore(1, "c1", "alpha", "complete", "hit"),
        RunScore(2, "c2", "alpha", "complete", "partial"),
        RunScore(3, "c3", "alpha", "complete", "miss"),
        RunScore(4, "c4", "alpha", "complete", "partial"),
    ]
    d = {r.competitor_name: r for r in detection_report(scores)}["alpha"]
    assert d.hits == 1 and d.partials == 2 and d.misses == 1
    assert d.eligible == 4  # hits + partials + misses
    assert d.detection_rate == 0.25  # 1/4 — unchanged by the partials
    # (1 + 0.5*2) / 4 = 0.5
    assert d.half_credit_rate == pytest.approx(0.5)


def test_competitor_detection_half_credit_rate_zero_eligible():
    d = CompetitorDetection("x", hits=0, partials=0, misses=0, excluded=2)
    assert d.eligible == 0
    assert d.half_credit_rate == 0.0


def test_drop_competitors_filters_named_competitors():
    from nelson.score import RunScore, drop_competitors

    scores = [
        RunScore(1, "c1", "alpha", "complete", "hit"),
        RunScore(2, "c1", "agy", "complete", "refused"),
        RunScore(3, "c2", "beta", "complete", "miss"),
    ]
    kept = drop_competitors(scores, {"agy"})
    assert [rs.competitor_name for rs in kept] == ["alpha", "beta"]
    # empty exclusion set is a no-op; unknown names are ignored
    assert drop_competitors(scores, set()) == scores
    assert len(drop_competitors(scores, {"nope"})) == 3


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


def test_case_scores_partial_beats_miss():
    from nelson.score import RunScore, case_scores

    # No hit; one file right-place/wrong-bug -> the case is a partial, not a miss.
    runs = [
        RunScore(1, "GHSA-c", "alpha", "complete", "miss"),
        RunScore(2, "GHSA-c", "alpha", "complete", "partial"),
    ]
    cs = case_scores(runs)[0]
    assert cs.outcome == "partial"
    assert cs.is_partial and cs.eligible and not cs.is_hit


def test_case_scores_undetermined_beats_partial():
    from nelson.score import RunScore, case_scores

    # Integrity: an unjudged file may hide a hit, so judge_error must outrank a
    # confirmed partial — a possible hit is never downgraded to half credit.
    runs = [
        RunScore(1, "GHSA-d", "alpha", "complete", "partial"),
        RunScore(2, "GHSA-d", "alpha", "complete", "judge_error"),
    ]
    assert case_scores(runs)[0].outcome == "judge_error"


def test_case_scores_hit_beats_partial():
    from nelson.score import RunScore, case_scores

    runs = [
        RunScore(1, "GHSA-e", "alpha", "complete", "partial"),
        RunScore(2, "GHSA-e", "alpha", "complete", "hit"),
    ]
    assert case_scores(runs)[0].outcome == "hit"


def test_noise_report_per_trial_detection_and_flaky_cases():
    from nelson.score import RunScore, noise_report

    # alpha over 3 trials: case A always hit; case B hit only in trial 0 (flaky).
    runs = []
    rid = 0
    for t in range(3):
        rid += 1
        runs.append(RunScore(rid, "A", "alpha", "complete", "hit", trial=t))
        rid += 1
        b = "hit" if t == 0 else "miss"
        runs.append(RunScore(rid, "B", "alpha", "complete", b, trial=t))

    r = noise_report(runs)[0]
    assert r.competitor_name == "alpha"
    assert r.n_trials == 3
    # trial 0: A,B both hit = 2/2; trials 1,2: A hit, B miss = 1/2
    assert r.per_trial == {0: (2, 2), 1: (1, 2), 2: (1, 2)}
    assert r.mean_rate == pytest.approx((1.0 + 0.5 + 0.5) / 3)
    assert (r.min_rate, r.max_rate) == (0.5, 1.0)
    assert r.spread == pytest.approx(0.5)
    assert r.per_case == {"A": (3, 3), "B": (1, 3)}
    assert r.flaky_cases == ["B"]  # A all-hit; B hit-some/miss-some


def test_noise_report_rolls_files_up_within_a_trial():
    from nelson.score import RunScore, noise_report

    # Case A is multi-file: trial 0 has a hit on one file (=> case hit); trial 1
    # all-miss (=> case miss). Rollup must happen per trial before counting.
    runs = [
        RunScore(1, "A", "alpha", "complete", "miss", trial=0),
        RunScore(2, "A", "alpha", "complete", "hit", trial=0),
        RunScore(3, "A", "alpha", "complete", "miss", trial=1),
        RunScore(4, "A", "alpha", "complete", "miss", trial=1),
    ]
    r = noise_report(runs)[0]
    assert r.per_trial == {0: (1, 1), 1: (0, 1)}
    assert r.per_case["A"] == (1, 2)
    assert r.flaky_cases == ["A"]


def test_noise_report_single_trial_has_zero_spread():
    from nelson.score import RunScore, noise_report

    runs = [
        RunScore(1, "A", "alpha", "complete", "hit"),
        RunScore(2, "B", "alpha", "complete", "miss"),
    ]
    r = noise_report(runs)[0]
    assert r.n_trials == 1
    assert r.spread == 0.0
    assert r.flaky_cases == []  # a case seen in only one trial can't be flaky


def test_leaderboard_detection_rate_is_deduped_union_across_trials():
    from nelson.score import RunScore, leaderboard

    # alpha over 3 trials, 2 cases. Case A hit every trial; case B hit only in
    # trial 0. Multipass de-dup: a case is a hit if found in ANY pass, so both A
    # and B are hits -> 2/2 = 100%. (The old mean-of-trials figure was 66.7%.)
    runs = []
    rid = 0
    for t in range(3):
        rid += 1
        runs.append(RunScore(rid, "A", "alpha", "complete", "hit", trial=t))
        rid += 1
        b = "hit" if t == 0 else "miss"
        runs.append(RunScore(rid, "B", "alpha", "complete", b, trial=t))

    e = leaderboard(runs)[0]
    assert e.n_trials == 3
    assert e.detection_rate == pytest.approx(1.0)  # union, not the 0.667 mean
    assert (e.hits, e.eligible) == (2, 2)
    # Per-trial variance is still available for the spread tooltip.
    assert (e.trial_min_rate, e.trial_max_rate) == (0.5, 1.0)
    assert e.trial_spread == pytest.approx(0.5)


def test_leaderboard_single_trial_rate_unchanged():
    from nelson.score import RunScore, leaderboard

    # Without --repeat (all trial 0), detection_rate is the plain hits/eligible.
    runs = [
        RunScore(1, "A", "alpha", "complete", "hit"),
        RunScore(2, "B", "alpha", "complete", "miss"),
        RunScore(3, "C", "alpha", "complete", "hit"),
    ]
    e = leaderboard(runs)[0]
    assert e.n_trials == 1
    assert e.detection_rate == pytest.approx(2 / 3)
    assert e.trial_spread == 0.0


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


def test_deduped_finding_counts_collapses_repeated_fp_across_trials():
    from nelson.score import (
        FindingScore,
        FPVerdict,
        RunScore,
        deduped_finding_counts,
    )

    def fp(fid, file, line, cwe):
        return FindingScore(
            fid, file, line, localized=False, fp=FPVerdict(is_real=False), cwe=cwe
        )

    # Same false positive (a.c ~line 100, CWE-125) restated in all 3 passes, plus
    # one genuinely distinct FP (b.c) in a single pass. De-duped: 2 distinct FPs.
    runs = [
        RunScore(
            t + 1,
            "alpha",
            "M",
            "complete",
            "miss",
            trial=t,
            findings=[fp(t + 1, "a.c", 100 + t, "CWE-125")],
        )
        for t in range(3)
    ]
    runs[0].findings.append(fp(99, "b.c", 5, "CWE-476"))

    counts = deduped_finding_counts(runs)["M"]
    assert counts.false_positives == 2  # 3 restatements collapse to 1, plus 1 distinct
    assert counts.true_findings == 0


def test_deduped_finding_counts_target_hit_beats_fp_in_cluster():
    from nelson.score import (
        FindingScore,
        FPVerdict,
        RunScore,
        TruthVerdict,
        deduped_finding_counts,
    )

    # Same spot: pass 0 confirmed the target bug, pass 1 the FP judge (wrongly)
    # called a restatement a false positive. The favorable verdict wins -> the
    # cluster is one target hit, not an FP.
    hit = FindingScore(
        1, "a.c", 42, localized=True, truth=TruthVerdict(True), cwe="CWE-787"
    )
    noise = FindingScore(
        2, "a.c", 43, localized=False, fp=FPVerdict(is_real=False), cwe="CWE-787"
    )
    runs = [
        RunScore(1, "alpha", "M", "complete", "hit", trial=0, findings=[hit]),
        RunScore(2, "alpha", "M", "complete", "miss", trial=1, findings=[noise]),
    ]
    counts = deduped_finding_counts(runs)["M"]
    assert counts.target_hits == 1
    assert counts.false_positives == 0


# -- Real judge wiring (no network) ------------------------------------------


def test_claude_truth_judge_is_a_truthjudge():
    from nelson.score import TruthJudge

    assert isinstance(ClaudeTruthJudge(), TruthJudge)


# == P4: FP judge (precision) ================================================

# -- FP-verdict parsing ------------------------------------------------------


def test_parse_fp_verdict_confirmed_false_review():
    assert parse_fp_verdict('{"verdict": "confirmed", "reasoning": "r"}') == (True, "r")
    assert parse_fp_verdict('{"verdict": "false_positive", "reasoning": "n"}') == (
        False,
        "n",
    )
    # needs_review -> inner None (a deliberate "can't tell"), with reasoning.
    assert parse_fp_verdict('{"verdict": "needs_review", "reasoning": "u"}') == (
        None,
        "u",
    )


def test_parse_fp_verdict_tolerates_prose_and_synonyms():
    text = 'My call:\n```json\n{"verdict": "FALSE", "reasoning": "safe"}\n```'
    assert parse_fp_verdict(text) == (False, "safe")


def test_parse_fp_verdict_none_when_unusable():
    # Outer None = nothing parseable -> the caller records a judge failure.
    assert parse_fp_verdict("I really cannot say") is None
    assert parse_fp_verdict('{"verdict": "banana"}') is None  # unknown verdict word


def test_fp_verdict_label():
    assert FPVerdict(True).label == "real_bug"
    assert FPVerdict(False).label == "false_positive"
    assert FPVerdict(None).label == "needs_review"  # clean undetermined
    assert FPVerdict(None, error="timeout").label == "error: timeout"  # failure


def test_build_fp_prompt_has_finding_and_source_only():
    # build_fp_prompt takes no Case, so the advisory cannot reach the FP judge.
    finding = ReportedFinding(
        file="Foo.java", line=42, description="cmd injection", cwe="CWE-78"
    )
    prompt = build_fp_prompt(finding, "void run(String s){ exec(s); }")
    assert "Foo.java:42" in prompt
    assert "CWE-78" in prompt
    assert "cmd injection" in prompt
    assert "exec(s)" in prompt


# -- Orchestrated precision scoring (fake FP judge + code) -------------------


class FakeFPJudge:
    """Returns a canned FP verdict; records the (finding, source) it was given."""

    name = "fake-fp"

    def __init__(self, verdict: FPVerdict):
        self.verdict = verdict
        self.calls = 0
        self.seen: list[tuple] = []

    def judge(self, finding, source):
        self.calls += 1
        self.seen.append((finding, source))
        return self.verdict


class FakeCode:
    """Canned CodeProvider; ``None`` text simulates an unavailable source."""

    def __init__(self, text: str | None = "void f(){}"):
        self.text = text
        self.calls = 0

    def source(self, case, file):
        self.calls += 1
        return self.text


def test_score_run_fp_judges_off_target_finding(tmp_path):
    db = Database(tmp_path / "t.db")
    case_id = _case(db)
    run_id = _complete_run(db, case_id, findings=[("Other.java", 10)])
    truth = FakeJudge(TruthVerdict(True))  # never reached (not localized)
    fp = FakeFPJudge(FPVerdict(False, reasoning="not exploitable", cost_usd=0.02))

    rs = Scorer(db, truth, fp_judge=fp, code=FakeCode()).score_run(run_id)

    assert truth.calls == 0 and fp.calls == 1
    assert rs.outcome == "miss"  # off-target -> detection miss
    assert rs.fp_cost == 0.02
    assert rs.findings[0].fp_category == "false_positive"
    row = db.run_findings(run_id)[0]
    assert row["judge_fp_verdict"] == "false_positive"
    js = db.judgments(target_kind="fp")
    assert len(js) == 1 and js[0]["cost_usd"] == 0.02


def test_score_run_fp_judges_localized_different_bug(tmp_path):
    # Q2: a finding that localized but the truth judge ruled a *different* bug is
    # still FP-judged — a genuine extra bug should be credited, not ignored.
    db = Database(tmp_path / "t.db")
    case_id = _case(db)
    run_id = _complete_run(db, case_id, findings=[("Foo.java", 76)])
    truth = FakeJudge(TruthVerdict(False, reasoning="different flaw"))
    fp = FakeFPJudge(FPVerdict(True, reasoning="real but other bug"))

    rs = Scorer(db, truth, fp_judge=fp, code=FakeCode()).score_run(run_id)

    assert truth.calls == 1  # localized -> truth-judged
    assert fp.calls == 1  # different_bug -> also FP-judged
    assert rs.outcome == "partial"  # detection: right place, not the target bug
    assert rs.findings[0].fp_category == "real_other"
    assert db.run_findings(run_id)[0]["judge_fp_verdict"] == "real_bug"


def test_score_run_does_not_fp_judge_the_target_hit(tmp_path):
    db = Database(tmp_path / "t.db")
    case_id = _case(db)
    run_id = _complete_run(db, case_id, findings=[("Foo.java", 76)])
    fp = FakeFPJudge(FPVerdict(False))

    rs = Scorer(
        db, FakeJudge(TruthVerdict(True)), fp_judge=fp, code=FakeCode()
    ).score_run(run_id)

    assert rs.outcome == "hit"
    assert fp.calls == 0  # the confirmed target bug is not a precision candidate
    assert rs.findings[0].fp_category is None
    assert db.run_findings(run_id)[0]["judge_fp_verdict"] is None


def test_score_run_does_not_fp_judge_undetermined_candidate(tmp_path):
    # A localized finding the truth judge could not decide might BE the target;
    # it is not handed to the FP judge (and stays out of precision).
    db = Database(tmp_path / "t.db")
    case_id = _case(db)
    run_id = _complete_run(db, case_id, findings=[("Foo.java", 76)])
    fp = FakeFPJudge(FPVerdict(False))

    rs = Scorer(
        db, FakeJudge(TruthVerdict(None, error="timeout")), fp_judge=fp, code=FakeCode()
    ).score_run(run_id)

    assert rs.outcome == "judge_error"
    assert fp.calls == 0
    assert rs.findings[0].fp_category is None


def test_fp_needs_review_is_undetermined_not_a_false_positive(tmp_path):
    # Integrity: the FP judge's indecision never counts as a false positive.
    db = Database(tmp_path / "t.db")
    case_id = _case(db)
    run_id = _complete_run(db, case_id, findings=[("Other.java", 10)])
    fp = FakeFPJudge(FPVerdict(None, reasoning="cannot tell"))

    rs = Scorer(
        db, FakeJudge(TruthVerdict(True)), fp_judge=fp, code=FakeCode()
    ).score_run(run_id)

    assert rs.findings[0].fp_category == "undetermined"
    assert db.run_findings(run_id)[0]["judge_fp_verdict"] == "needs_review"


def test_score_run_source_unavailable_never_an_fp(tmp_path):
    # No code to ground the verdict -> undetermined, never a false positive. Uses
    # the real ClaudeFPJudge, which short-circuits on a None source (no network).
    db = Database(tmp_path / "t.db")
    case_id = _case(db)
    run_id = _complete_run(db, case_id, findings=[("Other.java", 10)])

    rs = Scorer(
        db, FakeJudge(TruthVerdict(True)), fp_judge=ClaudeFPJudge(), code=FakeCode(None)
    ).score_run(run_id)

    f = rs.findings[0]
    assert f.fp is not None and f.fp.error == "source unavailable"
    assert f.fp_category == "undetermined"
    assert db.run_findings(run_id)[0]["judge_fp_verdict"] == "error: source unavailable"


def test_fp_judge_never_receives_the_advisory(tmp_path):
    db = Database(tmp_path / "t.db")
    case_id = db.upsert_case(
        {
            "source": "cvd",
            "ext_id": "GHSA-secret",
            "description": "SECRET-ADVISORY-TEXT",
            "gt_files": ["Foo.java"],
            "gt_hunks": [{"file": "Foo.java", "start": 73, "end": 79}],
        }
    )
    run_id = _complete_run(db, case_id, findings=[("Other.java", 5)])
    fp = FakeFPJudge(FPVerdict(False))
    code = FakeCode("clean source")

    Scorer(db, FakeJudge(TruthVerdict(True)), fp_judge=fp, code=code).score_run(run_id)

    finding, source = fp.seen[0]
    # The FP judge sees only the finding + its source — never the advisory.
    assert "SECRET-ADVISORY-TEXT" not in (finding.description or "")
    assert "SECRET-ADVISORY-TEXT" not in (source or "")


def test_load_run_score_rebuilds_fp_verdicts(tmp_path):
    db = Database(tmp_path / "t.db")
    case_id = _case(db)
    run_id = _complete_run(db, case_id, findings=[("Other.java", 10)])
    fp = FakeFPJudge(FPVerdict(False, reasoning="safe", cost_usd=0.02))
    scorer = Scorer(db, FakeJudge(TruthVerdict(True)), fp_judge=fp, code=FakeCode())

    scorer.score_run(run_id)
    assert fp.calls == 1

    reloaded = scorer.load_run_score(run_id)
    assert fp.calls == 1  # no new FP-judge calls
    reloaded_fp = reloaded.findings[0].fp
    assert reloaded_fp is not None and reloaded_fp.reasoning == "safe"
    assert reloaded.findings[0].fp_category == "false_positive"
    assert reloaded.fp_cost == pytest.approx(0.02)


def test_needs_scoring_accounts_for_fp(tmp_path):
    db = Database(tmp_path / "t.db")
    case_id = _case(db)
    run_id = _complete_run(db, case_id, findings=[("Other.java", 10)])

    detection_only = Scorer(db, FakeJudge(TruthVerdict(True)))
    detection_only.score_run(run_id)
    assert not detection_only.needs_scoring(run_id)  # detection settled
    assert detection_only.needs_scoring(
        run_id, include_precision=True
    )  # FP still missing in persisted score

    precision = Scorer(
        db,
        FakeJudge(TruthVerdict(True)),
        fp_judge=FakeFPJudge(FPVerdict(False)),
        code=FakeCode(),
    )
    assert precision.needs_scoring(run_id)  # FP not yet done
    precision.score_run(run_id)
    assert not precision.needs_scoring(run_id)


# -- GitCodeProvider ---------------------------------------------------------


class FakeGit:
    """A _GitShow whose tree is a {repo_path: contents} dict."""

    def __init__(self, contents: dict[str, str]):
        self.contents = contents
        self.prepared: list[tuple[str, str]] = []
        self.shown: list[str] = []

    def prepare(self, repo_url, commit, dest):
        self.prepared.append((repo_url, commit))

    def show(self, dest, rev, path):
        from nelson.derive import GitError

        self.shown.append(path)
        if path not in self.contents:
            raise GitError(f"missing {path} at {rev}")
        return self.contents[path]


def test_git_code_provider_resolves_repo_relative_path(tmp_path):
    case = Case(source="cvd", ext_id="x", repo_url="https://r", vuln_commit="abc")
    git = FakeGit({"src/main/Foo.java": "the code"})
    cp = GitCodeProvider(git, root=tmp_path)
    # A mount-absolute path is peeled to repo-relative; a real top-level src/ is
    # preserved (unlike the matching normalizer).
    assert cp.source(case, "/src/src/main/Foo.java") == "the code"
    assert cp.source(case, "src/main/Foo.java") == "the code"
    assert cp.source(case, "C:\\src\\main\\Foo.java") == "the code"


def test_git_code_provider_rejects_unsafe_repo_path_inputs(tmp_path):
    case = Case(source="cvd", ext_id="x", repo_url="https://r", vuln_commit="abc")
    git = FakeGit({"src/main/Foo.java": "the code"})
    cp = GitCodeProvider(git, root=tmp_path)

    assert cp.source(case, "/etc/passwd") is None
    assert cp.source(case, "src/../main/Foo.java") is None
    assert cp.source(case, "src/main/Foo:java") is None
    assert git.shown == []


def test_git_code_provider_caches_one_fetch_per_repo_commit(tmp_path):
    case = Case(source="cvd", ext_id="x", repo_url="https://r", vuln_commit="abc")
    git = FakeGit({"a.py": "A", "b.py": "B"})
    cp = GitCodeProvider(git, root=tmp_path)
    cp.source(case, "a.py")
    cp.source(case, "b.py")
    cp.source(case, "a.py")
    assert git.prepared == [("https://r", "abc")]  # prepared once for the repo@commit


def test_git_code_provider_missing_file_is_none(tmp_path):
    case = Case(source="cvd", ext_id="x", repo_url="https://r", vuln_commit="abc")
    cp = GitCodeProvider(FakeGit({}), root=tmp_path)
    # A path absent at that revision -> None -> the judge reports undetermined.
    assert cp.source(case, "Nope.java") is None


def test_git_code_provider_without_repo_is_none(tmp_path):
    case = Case(source="cvd", ext_id="x")  # no repo_url / vuln_commit
    cp = GitCodeProvider(FakeGit({"x.py": "y"}), root=tmp_path)
    assert cp.source(case, "x.py") is None


# -- Precision report --------------------------------------------------------


def test_precision_report_counts_and_rate():
    findings = [
        FindingScore(1, "Foo.java", 76, True, truth=TruthVerdict(True)),  # target hit
        FindingScore(2, "G.java", 5, False, fp=FPVerdict(True)),  # real other bug
        FindingScore(3, "H.java", 6, False, fp=FPVerdict(False)),  # false positive
        FindingScore(4, "I.java", 7, False, fp=FPVerdict(None)),  # undetermined
    ]
    rs = RunScore(
        1, "GHSA-a", "alpha", "complete", "hit", findings=findings, fp_cost=0.06
    )
    p = {r.competitor_name: r for r in precision_report([rs])}["alpha"]

    assert (p.target_hits, p.real_others, p.false_positives, p.undetermined) == (
        1,
        1,
        1,
        1,
    )
    assert p.true_findings == 2
    assert p.precision == pytest.approx(2 / 3)  # 2 true / (2 true + 1 fp)
    assert p.cases == 1
    assert p.fp_per_case == pytest.approx(1.0)
    assert p.fp_cost == pytest.approx(0.06)


def test_precision_is_none_when_nothing_decided():
    # Only an undetermined finding -> precision is None (not a misleading 0%).
    findings = [FindingScore(1, "F.java", 5, False, fp=FPVerdict(None))]
    rs = RunScore(1, "GHSA-x", "alpha", "complete", "miss", findings=findings)
    p = precision_report([rs])[0]
    assert p.precision is None
    assert p.false_positives == 0
    assert p.undetermined == 1


def test_competitor_precision_guards_empty_denominators():
    p = CompetitorPrecision("x", false_positives=2, cases=0)
    assert p.fp_per_case is None  # no audited cases
    p2 = CompetitorPrecision("y")  # nothing reported
    assert p2.precision is None


def test_claude_fp_judge_is_an_fpjudge():
    from nelson.score import FPJudge

    assert isinstance(ClaudeFPJudge(), FPJudge)


def test_claude_fp_judge_source_unavailable_short_circuits():
    # No network: a None source returns an undetermined verdict immediately.
    verdict = ClaudeFPJudge().judge(ReportedFinding(file="x", line=1), None)
    assert verdict.is_real is None
    assert verdict.error == "source unavailable"


# == Refusal judge (detection integrity) =====================================

# The verbatim refusal captured from Gemini-behind-agy on the junrar case.
_AGY_REFUSAL = (
    "Sorry, I cannot fulfill your request to analyze the specified code file for "
    "exploitable security vulnerabilities. For information on securing Java "
    "applications you can search for the OWASP Top Ten."
)


class FakeRefusalJudge:
    """Returns a canned refusal verdict; records how many times it was asked."""

    name = "fake-refusal"

    def __init__(self, verdict: RefusalVerdict):
        self.verdict = verdict
        self.calls = 0

    def judge(self, final_text):
        self.calls += 1
        return self.verdict


def _no_finding_run(db, case_id, *, raw_output, name="agy/antigravity"):
    comp_id = db.upsert_competitor(
        {"name": name, "model": "antigravity", "runtime": "agy"}
    )
    run_id = db.create_run(case_id, comp_id)
    db.start_run(run_id, container_id="c1")
    db.complete_run(
        run_id,
        tokens_in=1,
        tokens_out=1,
        cost_usd=0.0,
        wall_clock_s=8.0,
        raw_output=raw_output,
    )
    return run_id


def test_parse_refusal_verdict_true_false_and_absent():
    assert parse_refusal_verdict('{"refused": true, "reasoning": "declined"}') == (
        True,
        "declined",
    )
    assert parse_refusal_verdict('prose {"refused": "no"} more') == (False, "")
    assert parse_refusal_verdict("no json here") is None


def test_refusal_verdict_label():
    assert RefusalVerdict(True).label == "refused"
    assert RefusalVerdict(False).label == "attempted"
    assert RefusalVerdict(None, error="timeout").label == "error: timeout"


def test_build_refusal_prompt_is_output_only_no_advisory():
    # Integrity: the refusal judge sees the model's output, never the advisory.
    prompt = build_refusal_prompt(_AGY_REFUSAL)
    assert _AGY_REFUSAL in prompt
    assert "refused" in prompt.lower()
    for leak in ("CWE-22", "GHSA", "path traversal", "Foo.java", "ground truth"):
        assert leak not in prompt


def test_score_run_confirmed_refusal_is_refused_not_miss(tmp_path):
    db = Database(tmp_path / "t.db")
    case_id = _case(db)
    run_id = _no_finding_run(db, case_id, raw_output=_AGY_REFUSAL)
    refusal = FakeRefusalJudge(
        RefusalVerdict(True, reasoning="declined", cost_usd=0.02)
    )

    rs = Scorer(db, FakeJudge(TruthVerdict(True)), refusal_judge=refusal).score_run(
        run_id
    )

    assert rs.outcome == "refused"
    assert not rs.eligible  # carved out of the detection denominator, not a miss
    assert refusal.calls == 1
    assert rs.judge_cost == pytest.approx(0.02)  # folded into detection-judge spend
    # Persisted to the ledger keyed by the run, so reload needs no re-judging.
    js = db.judgments(target_kind="refusal", target_id=run_id)
    assert len(js) == 1 and js[0]["verdict"] == "refused"


def test_score_run_attempted_verdict_stays_a_miss(tmp_path):
    db = Database(tmp_path / "t.db")
    case_id = _case(db)
    # Output with no JSON array but the judge says the model engaged -> a miss.
    run_id = _no_finding_run(db, case_id, raw_output="I reviewed it; looks fine.")
    refusal = FakeRefusalJudge(RefusalVerdict(False, reasoning="analyzed, found none"))

    rs = Scorer(db, FakeJudge(TruthVerdict(True)), refusal_judge=refusal).score_run(
        run_id
    )

    assert rs.outcome == "miss" and rs.eligible
    assert refusal.calls == 1


def test_score_run_refusal_judge_error_stays_a_miss(tmp_path):
    # Conservative: an undecidable refusal verdict never excludes a genuine miss.
    db = Database(tmp_path / "t.db")
    case_id = _case(db)
    run_id = _no_finding_run(db, case_id, raw_output=_AGY_REFUSAL)
    refusal = FakeRefusalJudge(RefusalVerdict(None, error="timeout"))

    rs = Scorer(db, FakeJudge(TruthVerdict(True)), refusal_judge=refusal).score_run(
        run_id
    )

    assert rs.outcome == "miss"
    assert refusal.calls == 1


def test_score_run_compliant_empty_array_is_never_refusal_judged(tmp_path):
    # A clean-file "[]" answer emitted the contract -> not a candidate, no judge call.
    db = Database(tmp_path / "t.db")
    case_id = _case(db)
    run_id = _no_finding_run(db, case_id, raw_output="No issues found.\n[]")
    refusal = FakeRefusalJudge(RefusalVerdict(True))

    rs = Scorer(db, FakeJudge(TruthVerdict(True)), refusal_judge=refusal).score_run(
        run_id
    )

    assert rs.outcome == "miss"
    assert refusal.calls == 0


def test_score_run_refusal_without_judge_is_back_compat_miss(tmp_path):
    # No refusal judge wired: behaviour is identical to before (a plain miss).
    db = Database(tmp_path / "t.db")
    case_id = _case(db)
    run_id = _no_finding_run(db, case_id, raw_output=_AGY_REFUSAL)

    rs = Scorer(db, FakeJudge(TruthVerdict(True))).score_run(run_id)

    assert rs.outcome == "miss"
    assert not db.judgments(target_kind="refusal", target_id=run_id)


def test_needs_scoring_refusal_candidate_until_judged(tmp_path):
    db = Database(tmp_path / "t.db")
    case_id = _case(db)
    run_id = _no_finding_run(db, case_id, raw_output=_AGY_REFUSAL)

    # Without a refusal judge, a zero-finding run is a settled miss (no work).
    assert not Scorer(db, FakeJudge(TruthVerdict(True))).needs_scoring(run_id)

    # With one wired, the un-judged candidate needs scoring; not after judging.
    scorer = Scorer(
        db,
        FakeJudge(TruthVerdict(True)),
        refusal_judge=FakeRefusalJudge(RefusalVerdict(True, cost_usd=0.01)),
    )
    assert scorer.needs_scoring(run_id)
    scorer.score_run(run_id)
    assert not scorer.needs_scoring(run_id)


def test_load_run_score_reflects_persisted_refusal_without_rejudging(tmp_path):
    db = Database(tmp_path / "t.db")
    case_id = _case(db)
    run_id = _no_finding_run(db, case_id, raw_output=_AGY_REFUSAL)
    refusal = FakeRefusalJudge(RefusalVerdict(True, cost_usd=0.02))
    scorer = Scorer(db, FakeJudge(TruthVerdict(True)), refusal_judge=refusal)
    scorer.score_run(run_id)

    reloaded = scorer.load_run_score(run_id)
    assert reloaded.outcome == "refused"
    assert reloaded.judge_cost == pytest.approx(0.02)
    assert refusal.calls == 1  # load_run_score did not re-judge


def test_detection_report_counts_refused_apart_from_miss():
    # refused is its own column and stays out of the detection denominator.
    scores = [
        RunScore(1, "c1", "alpha", "complete", "hit"),
        RunScore(2, "c2", "alpha", "complete", "miss"),
        RunScore(3, "c3", "alpha", "complete", "refused"),
    ]
    d = {r.competitor_name: r for r in detection_report(scores)}["alpha"]
    assert d.hits == 1 and d.misses == 1 and d.refused == 1
    assert d.excluded == 0
    assert d.eligible == 2  # refused not in the denominator
    assert d.detection_rate == 0.5


def test_case_scores_miss_beats_refused_refused_beats_excluded():
    from nelson.score import case_scores

    # Engaged on one file (miss) but refused another -> the case counts as a miss.
    mixed = [
        RunScore(1, "GHSA-a", "alpha", "complete", "refused"),
        RunScore(2, "GHSA-a", "alpha", "complete", "miss"),
    ]
    assert case_scores(mixed)[0].outcome == "miss"
    # Refused everywhere it ran -> the case is refused (not a miss, not excluded).
    all_refused = [
        RunScore(3, "GHSA-b", "alpha", "complete", "refused"),
        RunScore(4, "GHSA-b", "alpha", "auth_failed", "excluded"),
    ]
    assert case_scores(all_refused)[0].outcome == "refused"


def test_claude_refusal_judge_is_a_refusaljudge():
    from nelson.score import RefusalJudge

    assert isinstance(ClaudeRefusalJudge(), RefusalJudge)


# -- Judge de-duplication (same bug class, same place, same model = one bug) ----


def test_score_run_dedups_truth_judge_for_same_bug_cluster(tmp_path):
    db = Database(tmp_path / "t.db")
    case_id = _case(db)
    # Three restatements of one bug: same file, adjacent lines inside the hunk.
    run_id = _complete_run(
        db, case_id, findings=[("Foo.java", 75), ("Foo.java", 76), ("Foo.java", 77)]
    )
    judge = FakeJudge(TruthVerdict(True, reasoning="zip slip", cost_usd=0.05))

    rs = Scorer(db, judge).score_run(run_id)

    assert rs.outcome == "hit"
    assert judge.calls == 1  # one cluster -> one judge call (was 3)
    assert rs.judge_cost == 0.05  # counted once, not x3
    # Every member still carries the derived verdict.
    for f in db.run_findings(run_id):
        assert f["matches_ground_truth"] == 1
        assert f["judge_truth_verdict"] == "same_bug"
    # The ledger records exactly the one real call.
    assert len(db.judgments(target_kind="truth")) == 1


def test_score_run_does_not_dedup_distant_findings(tmp_path):
    db = Database(tmp_path / "t.db")
    case_id = _case(db)
    # Both localize (hunk 73-79) but are >cluster_tolerance apart -> two clusters.
    run_id = _complete_run(db, case_id, findings=[("Foo.java", 73), ("Foo.java", 79)])
    judge = FakeJudge(TruthVerdict(True, cost_usd=0.05))

    Scorer(db, judge).score_run(run_id)

    assert judge.calls == 2


def test_score_run_cluster_tolerance_zero_judges_each_distinct_line(tmp_path):
    db = Database(tmp_path / "t.db")
    case_id = _case(db)
    run_id = _complete_run(db, case_id, findings=[("Foo.java", 75), ("Foo.java", 76)])
    judge = FakeJudge(TruthVerdict(True, cost_usd=0.05))

    Scorer(db, judge, cluster_tolerance=0).score_run(run_id)

    assert judge.calls == 2  # de-dup off: adjacent-but-distinct lines judged apart


def test_score_run_dedups_fp_judge_for_same_bug_cluster(tmp_path):
    db = Database(tmp_path / "t.db")
    case_id = _case(db)
    # Two restatements of one off-target finding (far from the hunk, adjacent).
    run_id = _complete_run(
        db, case_id, findings=[("Other.java", 10), ("Other.java", 11)]
    )
    truth = FakeJudge(TruthVerdict(True))  # never reached (not localized)
    fp = FakeFPJudge(FPVerdict(False, reasoning="safe", cost_usd=0.02))
    code = FakeCode()

    rs = Scorer(db, truth, fp_judge=fp, code=code).score_run(run_id)

    assert truth.calls == 0
    assert fp.calls == 1  # one cluster -> one FP call (was 2)
    assert code.calls == 1  # one source read for the cluster
    assert rs.fp_cost == 0.02
    for f in db.run_findings(run_id):
        assert f["judge_fp_verdict"] == "false_positive"
