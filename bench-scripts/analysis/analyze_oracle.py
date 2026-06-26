#!/usr/bin/env python3
"""Oracle-CWE experiment analysis: per-competitor detection, oracle vs control.

Loads both DBs (no judge spend — reloads persisted scores) and prints, per
competitor, the open-mode (control) vs oracle-mode detection rate and the delta.
Run after run_oracle_experiment.sh finishes (both passes scored).

    .venv/bin/python analyze_oracle.py
"""

from nelson.db import Database
from nelson.score import (
    ELIGIBLE_OUTCOMES,
    ClaudeTruthJudge,
    Scorer,
    case_scores,
    drop_competitors,
)

ORACLE_DB = "nelson-oracle.db"
CONTROL_DB = "nelson-oracle-control.db"


def detection(db_path: str) -> dict[str, tuple[int, int]]:
    """competitor -> (hits, eligible) over rolled-up case outcomes."""
    db = Database(db_path)
    scorer = Scorer(db, ClaudeTruthJudge(), tolerance=10)
    retired = {c["name"] for c in db.list_competitors() if c["status"] == "retired"}
    rss = drop_competitors(
        [scorer.load_run_score(r["id"]) for r in db.list_runs()], retired
    )
    out: dict[str, list[int]] = {}
    for cs in case_scores(rss):
        if cs.outcome in ELIGIBLE_OUTCOMES:
            h, e = out.setdefault(cs.competitor_name, [0, 0])
            out[cs.competitor_name] = [h + (1 if cs.is_hit else 0), e + 1]
    return {k: (v[0], v[1]) for k, v in out.items()}


def rate(he: tuple[int, int]) -> float:
    h, e = he
    return h / e if e else 0.0


def main() -> None:
    control = detection(CONTROL_DB)
    oracle = detection(ORACLE_DB)
    names = sorted(set(control) | set(oracle))
    print(f"{'competitor':40s} {'open':>10s} {'oracle':>10s} {'delta':>8s}")
    print("-" * 72)
    for n in names:
        c = control.get(n, (0, 0))
        o = oracle.get(n, (0, 0))
        cr, orr = rate(c), rate(o)
        delta = orr - cr
        print(
            f"{n:40s} {c[0]:>2}/{c[1]:<2} {cr:>4.0%} "
            f"{o[0]:>2}/{o[1]:<2} {orr:>4.0%} {delta:>+7.0%}"
        )
    # Cohort totals (sum of hits / sum of eligible).
    ch = sum(v[0] for v in control.values())
    ce = sum(v[1] for v in control.values())
    oh = sum(v[0] for v in oracle.values())
    oe = sum(v[1] for v in oracle.values())
    print("-" * 72)
    print(
        f"{'COHORT':40s} {ch:>2}/{ce:<2} {ch / ce if ce else 0:>4.0%} "
        f"{oh:>2}/{oe:<2} {oh / oe if oe else 0:>4.0%} "
        f"{(oh / oe if oe else 0) - (ch / ce if ce else 0):>+7.0%}"
    )


if __name__ == "__main__":
    main()
