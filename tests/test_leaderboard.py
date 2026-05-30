"""P5 leaderboard + Pareto reporting.

Aggregation and ranking are pure functions of RunScores, so they are exercised
with synthetic multi-competitor data (no judge, no container). A DB round-trip
covers the new plumbing — that competitor cost/latency/size reach a RunScore —
and an HTML smoke test covers the renderer.
"""

import pytest

from nelson.db import Database
from nelson.html_report import generate_leaderboard_report
from nelson.score import (
    ClaudeTruthJudge,
    FindingScore,
    FPVerdict,
    LeaderboardEntry,
    RunScore,
    Scorer,
    TruthVerdict,
    case_scores,
    leaderboard,
    pareto_frontier,
)

# -- Synthetic run scores ----------------------------------------------------


def _hit_finding(fid: int, file: str = "Foo.java", line: int = 76) -> FindingScore:
    return FindingScore(fid, file, line, localized=True, truth=TruthVerdict(True))


def _fp_finding(fid: int) -> FindingScore:
    return FindingScore(fid, "Noise.java", 1, localized=False, fp=FPVerdict(False))


def _real_other(fid: int) -> FindingScore:
    return FindingScore(fid, "Other.java", 9, localized=False, fp=FPVerdict(True))


def _scores() -> list[RunScore]:
    """alpha: 1 hit + 1 FP over 2 cases (50% det, 50% prec); beta: 2 clean hits."""
    return [
        RunScore(
            1,
            "caseA",
            "alpha",
            "complete",
            "hit",
            findings=[_hit_finding(1), _fp_finding(2)],
            judge_cost=0.02,
            fp_cost=0.03,
            competitor_cost=0.20,
            wall_clock_s=100.0,
            tokens_in=1000,
            tokens_out=200,
            size_class="large",
        ),
        RunScore(
            2,
            "caseB",
            "alpha",
            "complete",
            "miss",
            findings=[],
            competitor_cost=0.10,
            wall_clock_s=50.0,
            tokens_in=500,
            tokens_out=100,
            size_class="large",
        ),
        RunScore(
            3,
            "caseA",
            "beta",
            "complete",
            "hit",
            findings=[_hit_finding(3)],
            competitor_cost=0.05,
            wall_clock_s=20.0,
            size_class="small",
        ),
        RunScore(
            4,
            "caseB",
            "beta",
            "complete",
            "hit",
            findings=[_hit_finding(4, line=80)],
            competitor_cost=0.05,
            wall_clock_s=20.0,
            size_class="small",
        ),
    ]


def test_leaderboard_aggregates_detection_precision_and_economics():
    entries = {e.competitor_name: e for e in leaderboard(_scores())}

    alpha = entries["alpha"]
    assert alpha.detection_rate == 0.5  # 1 hit / (1 hit + 1 miss)
    assert alpha.precision == 0.5  # 1 target hit / (1 + 1 FP)
    assert alpha.fp_per_case == 0.5  # 1 FP over 2 cases
    assert alpha.cost_per_case == pytest.approx(0.15)  # (0.20 + 0.10) / 2
    assert alpha.latency_per_case == 75.0  # (100 + 50) / 2
    assert alpha.tokens_per_case == 900.0  # (1000+200 + 500+100) / 2
    assert alpha.size_class == "large"

    beta = entries["beta"]
    assert beta.detection_rate == 1.0
    assert beta.precision == 1.0
    assert beta.cost_per_case == 0.05
    assert beta.latency_per_case == 20.0
    assert beta.size_class == "small"


def test_leaderboard_ranks_best_first():
    ranked = leaderboard(_scores())
    # beta (100% detection) sorts above alpha (50%).
    assert [e.competitor_name for e in ranked] == ["beta", "alpha"]


def test_leaderboard_keeps_judge_spend_out_of_competitor_cost():
    alpha = {e.competitor_name: e for e in leaderboard(_scores())}["alpha"]
    # competitor_cost is the model's own spend; judge/fp spend is carried apart.
    assert alpha.competitor_cost == pytest.approx(0.30)
    assert round(alpha.judge_cost, 4) == 0.02
    assert round(alpha.fp_cost, 4) == 0.03
    # The Pareto cost axis uses only competitor spend.
    assert alpha.cost_per_case == pytest.approx(0.15)


def test_leaderboard_real_other_is_credited_not_penalized():
    scores = [
        RunScore(
            1,
            "caseA",
            "gamma",
            "complete",
            "hit",
            findings=[_hit_finding(1), _real_other(2)],
            competitor_cost=0.10,
            wall_clock_s=10.0,
            size_class="medium",
        ),
    ]
    gamma = leaderboard(scores)[0]
    # A confirmed *different* real bug counts as a true finding -> precision 1.0.
    assert gamma.real_others == 1
    assert gamma.precision == 1.0
    assert gamma.false_positives == 0


def test_leaderboard_excludes_failed_run_from_cost_and_cases():
    scores = [
        *_scores(),
        RunScore(
            5,
            "caseC",
            "beta",
            "auth_failed",
            "excluded",
            competitor_cost=99.0,
            wall_clock_s=999.0,
            size_class="small",
        ),
    ]
    beta = {e.competitor_name: e for e in leaderboard(scores)}["beta"]
    # The auth_failed run contributes neither cost, latency, nor a case.
    assert beta.cases == 2
    assert beta.competitor_cost == pytest.approx(0.10)
    assert beta.latency_per_case == 20.0


def test_leaderboard_zero_cases_yields_none_economics():
    scores = [
        RunScore(1, "caseA", "solo", "auth_failed", "excluded", competitor_cost=5.0),
    ]
    solo = leaderboard(scores)[0]
    assert solo.cases == 0
    assert solo.cost_per_case is None
    assert solo.latency_per_case is None
    assert solo.fp_per_case is None


# -- Quality + Pareto --------------------------------------------------------


def test_quality_is_detection_times_precision():
    e = LeaderboardEntry("x", hits=1, misses=1, target_hits=1, false_positives=1)
    assert e.detection_rate == 0.5
    assert e.precision == 0.5
    assert e.quality == 0.25


def test_quality_treats_unknown_precision_as_one():
    # No findings at all -> precision None; detection then drives quality.
    e = LeaderboardEntry("x", hits=0, misses=1)
    assert e.precision is None
    assert e.quality == 0.0


def _entry(name, hits, elig, th, fp, cost, cases):
    return LeaderboardEntry(
        name,
        hits=hits,
        misses=elig - hits,
        target_hits=th,
        false_positives=fp,
        cases=cases,
        competitor_cost=cost,
    )


def test_pareto_frontier_drops_dominated_points():
    a = _entry("a", 1, 1, 1, 0, cost=0.05, cases=1)  # q 1.0, cost 0.05
    b = _entry("b", 1, 10, 1, 0, cost=0.001, cases=1)  # q 0.1, cost 0.001
    c = _entry("c", 1, 2, 1, 1, cost=0.5, cases=1)  # q 0.25, cost 0.5 (dominated)
    front = pareto_frontier(
        [a, b, c], x=lambda e: e.cost_per_case, y=lambda e: e.quality
    )
    # c is dominated by a (cheaper AND higher quality); a and b are a real trade.
    assert [e.competitor_name for e in front] == ["b", "a"]  # sorted by cost asc


def test_pareto_frontier_keeps_ties():
    a = _entry("a", 1, 1, 1, 0, cost=0.10, cases=1)
    b = _entry("b", 1, 1, 1, 0, cost=0.10, cases=1)  # identical point
    front = pareto_frontier([a, b], x=lambda e: e.cost_per_case, y=lambda e: e.quality)
    assert {e.competitor_name for e in front} == {"a", "b"}


def test_pareto_frontier_skips_entries_missing_a_coordinate():
    a = _entry("a", 1, 1, 1, 0, cost=0.05, cases=1)
    no_cases = _entry("z", 0, 0, 0, 0, cost=0.0, cases=0)  # cost_per_case None
    front = pareto_frontier(
        [a, no_cases], x=lambda e: e.cost_per_case, y=lambda e: e.quality
    )
    assert [e.competitor_name for e in front] == ["a"]


# -- DB round-trip (the cost/latency/size plumbing) --------------------------


def _db_run(tmp_path, *, cost=0.31, wall=223.0, size="medium"):
    db = Database(tmp_path / "n.db")
    case_id = db.upsert_case(
        {
            "source": "cvd",
            "ext_id": "GHSA-test",
            "cwe": "CWE-22",
            "gt_files": ["Foo.java"],
            "gt_hunks": [{"file": "Foo.java", "start": 73, "end": 79}],
        }
    )
    comp_id = db.upsert_competitor(
        {
            "name": "claude-code/sonnet",
            "model": "sonnet",
            "size_class": size,
            "knowledge_cutoff": "2025-01",
        }
    )
    run_id = db.create_run(case_id, comp_id)
    db.start_run(run_id, container_id="c1")
    db.complete_run(
        run_id, tokens_in=1000, tokens_out=50, cost_usd=cost, wall_clock_s=wall
    )
    db.add_run_finding(
        run_id, file="Foo.java", line_start=76, line_end=76, description="x"
    )
    return db, run_id


def test_load_run_score_carries_competitor_cost_latency_and_size(tmp_path):
    db, run_id = _db_run(tmp_path)
    scorer = Scorer(db, ClaudeTruthJudge())  # judge is never invoked by load
    rs = scorer.load_run_score(run_id)
    assert rs.competitor_cost == 0.31
    assert rs.wall_clock_s == 223.0
    assert rs.tokens_in == 1000
    assert rs.tokens_out == 50
    assert rs.size_class == "medium"
    assert rs.knowledge_cutoff == "2025-01"


def test_excluded_run_still_carries_metadata(tmp_path):
    db = Database(tmp_path / "n.db")
    case_id = db.upsert_case({"source": "cvd", "ext_id": "GHSA-e"})
    comp_id = db.upsert_competitor({"name": "c/m", "size_class": "small"})
    run_id = db.create_run(case_id, comp_id)
    db.mark_run_auth_failed(run_id, "Not logged in")
    rs = Scorer(db, ClaudeTruthJudge()).load_run_score(run_id)
    assert rs.outcome == "excluded"
    assert rs.size_class == "small"


# -- HTML report -------------------------------------------------------------


def test_generate_leaderboard_report_renders_table_pareto_and_matrix():
    scores = _scores()
    html = generate_leaderboard_report(leaderboard(scores), case_scores(scores))
    assert "<!DOCTYPE html>" in html
    assert "Leaderboard" in html
    assert "alpha" in html and "beta" in html
    assert "<svg" in html  # the Pareto scatter
    assert "Per-case results" in html
    assert "HIT" in html  # a matrix cell
    assert "scatter-pt-frontier" in html  # beta is on the cost frontier
    assert "Other real" in html  # off-target real-bug column is shown
    assert "Judge $" not in html  # judge spend dropped from the model table
    assert "Tokens/case" in html  # token usage column
    assert "Tokens &amp; time per case" in html  # token chart section
    assert "token-bar" in html  # at least one bar rendered


def test_generate_leaderboard_report_handles_no_runs():
    html = generate_leaderboard_report([], [])
    assert "<!DOCTYPE html>" in html
    assert "No plottable data" in html  # empty scatter, no crash
