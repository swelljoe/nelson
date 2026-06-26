#!/usr/bin/env python3
"""Analyze the open-prompt strategy probe (untracked experiment).

Decodes the prompting arm from each run's ``trial`` index
(``arm = trial // repeat``) and reports, per (model, case, arm):

  - localized detection per trial (does ANY finding land within the gate
    tolerance of a ground-truth hunk) — the same mechanical gate the benchmark
    uses, and the bar every baseline miss fails;
  - the mean across repeats (the leaderboard's --repeat metric) and the UNION
    (found in at least one trial — the brute-force "best-of-N" view).

This is localization-only (free, no judge). A localized hit is "right place";
confirming it is the SAME bug needs the truth judge — run that as a second pass
(like fpjudge_libyang.py) only on the localized hits worth confirming.

    .venv/bin/python analyze_promptlab.py [nelson-promptlab.db] [--repeat N]
"""

import argparse
import sqlite3

from nelson.corpus import load_manifest_dir
from nelson.score import localize

ARMS = ["open", "plan", "checklist"]
TOLERANCE = 10  # mirror the benchmark's localization gate
CASES_DIR = "cases/"

# Baseline detection (nelson.db, from the per-case audit) for the difficulty
# column — sorted easy->hard so the matrix reads top (solved) to bottom (missed).
BASELINE = {
    "GHSA-w52v-v783-gw97": "18/20",
    "GHSA-j273-m5qq-6825": "17/20",
    "GHSA-f26g-jm89-4g65": "11/20",
    "GHSA-x9h5-r9v2-vcww": "1/21",
    "GHSA-9f49-8x56-jmjc": "1/21",
    "CVE-2026-5199": "0/21",
}
ORDER = list(BASELINE)


def load(db_path: str, repeat: int | None):
    cases = {c.ext_id: c for c in load_manifest_dir(CASES_DIR)}
    db = sqlite3.connect(db_path)
    db.row_factory = sqlite3.Row
    runs = db.execute(
        """
        SELECT r.id AS rid, cp.name AS model, ca.ext_id AS ext, r.trial AS trial,
               r.target_file AS file, r.status AS status
        FROM runs r
        JOIN competitors cp ON cp.id = r.competitor_id
        JOIN cases ca ON ca.id = r.case_id
        ORDER BY cp.name, ca.ext_id, r.trial, r.id
        """
    ).fetchall()
    # Dedup to one best run per (model, ext, trial): prefer complete, else latest id.
    best: dict[tuple, sqlite3.Row] = {}
    for r in runs:
        k = (r["model"], r["ext"], r["trial"])
        cur = best.get(k)
        if (
            cur is None
            or (r["status"] == "complete" and cur["status"] != "complete")
            or (r["status"] == cur["status"] and r["rid"] > cur["rid"])
        ):
            best[k] = r

    # repeat must be known to decode arm = trial // repeat. Inferring it from the
    # max trial present is wrong mid-run (a partial DB under-counts trials and
    # mislabels arms), so prefer the explicit value; infer only as a fallback.
    if repeat is None:
        trials = {t for (_m, _e, t) in best}
        repeat = max(1, (max(trials) + 1) // len(ARMS)) if trials else 1

    # localized-hit per run
    findings = db.execute(
        "SELECT run_id, file, line_start FROM run_findings"
    ).fetchall()
    by_run: dict[int, list[sqlite3.Row]] = {}
    for f in findings:
        by_run.setdefault(f["run_id"], []).append(f)

    def hit(row) -> bool:
        case = cases[row["ext"]]
        for f in by_run.get(row["rid"], []):
            if localize(f["file"], f["line_start"], case.gt_hunks, TOLERANCE).matched:
                return True
        return False

    return best, repeat, hit


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("db", nargs="?", default="nelson-promptlab.db")
    ap.add_argument(
        "--repeat",
        type=int,
        default=2,
        help="trials per arm (must match the run; 0 = infer from data)",
    )
    args = ap.parse_args()
    best, repeat, hit = load(args.db, args.repeat or None)
    models = sorted({m for (m, _e, _t) in best})
    print(f"db={args.db}  repeat={repeat}  arms={ARMS}\n")

    def cell(model, ext, trial):
        # Only a *complete* run is data; an infra_error/auth_failed run never got
        # a fair look (e.g. a dead server) and must read as no-data, not a miss.
        row = best.get((model, ext, trial))
        if row is None or row["status"] != "complete":
            return None
        return hit(row)

    for model in models:
        short = model.rsplit("/", 1)[-1]
        print(f"{'=' * 74}\n## {short}\n{'=' * 74}")
        hdr = f"{'case':22} {'base':>6}  " + "  ".join(f"{a:^11}" for a in ARMS)
        print(hdr)
        print(f"{'':22} {'':>6}  " + "  ".join(f"{'mean / uni':^11}" for _ in ARMS))
        print("-" * len(hdr))
        for ext in ORDER:
            cells = []
            for arm_idx, _arm in enumerate(ARMS):
                hits = [cell(model, ext, arm_idx * repeat + r) for r in range(repeat)]
                done = [h for h in hits if h is not None]
                if not done:
                    cells.append(f"{'--':^11}")
                else:
                    n = sum(1 for h in done if h)
                    uni = "Y" if n else "n"
                    cells.append(f"{n}/{len(done)} {uni:>5}")
            print(f"{ext:22} {BASELINE.get(ext, '?'):>6}  " + "  ".join(cells))
        print()

    # Headline: per arm, how many cases the model UNION-detected, split by whether
    # the baseline solved it (regression guard) or missed it (the real question).
    # Denominators count only cases with data (a complete run), so a dead-server
    # gap reads as "n/COVERED", never as a silent miss.
    print(
        f"{'=' * 74}\n## ARM SUMMARY (union detection across {repeat} trials)\n{'=' * 74}"
    )
    missed = {e for e in ORDER if BASELINE[e].split("/")[0] in ("0", "1")}
    for model in models:
        short = model.rsplit("/", 1)[-1]
        print(f"\n{short}")
        for arm_idx, arm in enumerate(ARMS):
            solved_hit = solved_cov = miss_hit = miss_cov = 0
            for ext in ORDER:
                vals = [cell(model, ext, arm_idx * repeat + r) for r in range(repeat)]
                if all(v is None for v in vals):
                    continue  # no data for this (case, arm) — exclude from coverage
                any_hit = any(v for v in vals)
                if ext in missed:
                    miss_cov += 1
                    miss_hit += int(any_hit)
                else:
                    solved_cov += 1
                    solved_hit += int(any_hit)
            print(
                f"  {arm:9}: solved-cases {solved_hit}/{solved_cov}"
                f"   hard-miss-cases {miss_hit}/{miss_cov}"
            )


if __name__ == "__main__":
    main()
