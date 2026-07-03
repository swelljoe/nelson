#!/usr/bin/env python3
"""Merge the two Gemma-4-12B-think benchmark halves into baseline nelson.db.

The think run was split across two Strix Halo boxes into two scratch DBs:
  nelson-think-a.db  competitor_id 40 (gemma4-12b-think),   box .1, 11 cases
  nelson-think-b.db  competitor_id 41 (gemma4-12b-think-b), box .2, 11 cases

Both halves belong to ONE logical competitor. This imports every run (+ its
run_findings) from both scratch DBs into baseline nelson.db under a single
competitor — gemma4-12b-think (id 40) — remapping case_id by ext_id (never by
raw id) and minting fresh run ids, then deletes the temporary -b competitor row.

Runs link to competitors by competitor_id and to cases by case_id, so folding
-b's runs under id 40 is a plain competitor reassignment (the proven "deepseek
split" move). Judge verdict columns come over NULL (the loop ran --no-score);
scoring happens afterwards.

Idempotent guard: refuses to run if comp 40 already has runs in baseline (so a
double-invoke can't duplicate the import).

Usage:
    .venv/bin/python bench-scripts/merge_think_runs.py --commit
    (without --commit: dry-run summary only)
"""

from __future__ import annotations

import argparse
import sqlite3
import sys

BASELINE = "nelson.db"
SOURCES = [
    # (scratch db, source competitor_id, dest competitor_id in baseline)
    ("nelson-think-a.db", 40, 40),
    ("nelson-think-b.db", 41, 40),
]
DEST_COMP = 40
DROP_COMP = 41  # temp -b row, removed after fold

RUN_COLS = [
    "case_id", "competitor_id", "container_id", "status", "started_at",
    "completed_at", "tokens_in", "tokens_out", "cost_usd", "wall_clock_s",
    "transcript_path", "raw_output", "error_msg", "target_file", "trial",
]
FIND_COLS = [
    "run_id", "file", "line_start", "line_end", "description", "confidence",
    "cwe", "matches_ground_truth", "judge_truth_verdict", "judge_fp_verdict",
    "judge_reasoning",
]


def ext_map(conn: sqlite3.Connection) -> dict[str, int]:
    return {r[1]: r[0] for r in conn.execute("SELECT id, ext_id FROM cases")}


def id_ext(conn: sqlite3.Connection) -> dict[int, str]:
    return {r[0]: r[1] for r in conn.execute("SELECT id, ext_id FROM cases")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", action="store_true", help="write; else dry-run")
    args = ap.parse_args()

    base = sqlite3.connect(BASELINE)
    base.row_factory = sqlite3.Row
    base.execute("PRAGMA foreign_keys=ON")

    existing = base.execute(
        "SELECT COUNT(*) FROM runs WHERE competitor_id=?", (DEST_COMP,)
    ).fetchone()[0]
    if existing:
        print(
            f"ABORT: competitor {DEST_COMP} already has {existing} runs in "
            f"{BASELINE}; import already done. Refusing to duplicate.",
            file=sys.stderr,
        )
        return 1

    base_ext = ext_map(base)  # ext_id -> baseline case id
    total_runs = total_finds = 0
    per_source = []

    for db_path, src_comp, dest_comp in SOURCES:
        src = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        src.row_factory = sqlite3.Row
        src_id_ext = id_ext(src)  # src case id -> ext_id

        runs = src.execute(
            "SELECT * FROM runs WHERE competitor_id=? ORDER BY id", (src_comp,)
        ).fetchall()

        n_runs = n_finds = 0
        status_counts: dict[str, int] = {}
        for run in runs:
            ext = src_id_ext.get(run["case_id"])
            if ext is None or ext not in base_ext:
                print(
                    f"  SKIP run {run['id']} ({db_path}): case_id "
                    f"{run['case_id']} ext={ext} not in baseline",
                    file=sys.stderr,
                )
                continue
            new_case_id = base_ext[ext]
            status_counts[run["status"]] = status_counts.get(run["status"], 0) + 1
            if args.commit:
                vals = [
                    new_case_id if c == "case_id"
                    else dest_comp if c == "competitor_id"
                    else run[c]
                    for c in RUN_COLS
                ]
                # S608 suppressed below: interpolated names are the trusted module-level
                # RUN_COLS constant (never user input); all values are bound via ?.
                cur = base.execute(
                    f"INSERT INTO runs ({','.join(RUN_COLS)}) "  # noqa: S608
                    f"VALUES ({','.join('?' * len(RUN_COLS))})",
                    vals,
                )
                new_run_id = cur.lastrowid
                finds = src.execute(
                    "SELECT * FROM run_findings WHERE run_id=?", (run["id"],)
                ).fetchall()
                for f in finds:
                    fvals = [new_run_id if c == "run_id" else f[c] for c in FIND_COLS]
                    # S608 suppressed below: names are the trusted FIND_COLS constant; ? binds.
                    base.execute(
                        f"INSERT INTO run_findings ({','.join(FIND_COLS)}) "  # noqa: S608
                        f"VALUES ({','.join('?' * len(FIND_COLS))})",
                        fvals,
                    )
                n_finds += len(finds)
            else:
                n_finds += src.execute(
                    "SELECT COUNT(*) FROM run_findings WHERE run_id=?", (run["id"],)
                ).fetchone()[0]
            n_runs += 1
        per_source.append((db_path, src_comp, n_runs, n_finds, status_counts))
        total_runs += n_runs
        total_finds += n_finds
        src.close()

    print(f"{'COMMIT' if args.commit else 'DRY-RUN'} — fold into competitor {DEST_COMP}")
    for db_path, src_comp, n_runs, n_finds, sc in per_source:
        print(f"  {db_path} (comp {src_comp}): {n_runs} runs, {n_finds} findings  {sc}")
    print(f"  TOTAL: {total_runs} runs, {total_finds} findings")

    if args.commit:
        dropped = base.execute(
            "DELETE FROM competitors WHERE id=?", (DROP_COMP,)
        ).rowcount
        base.commit()
        print(f"  dropped temp competitor row id {DROP_COMP}: {dropped}")
        final = base.execute(
            "SELECT status, COUNT(*) FROM runs WHERE competitor_id=? GROUP BY status",
            (DEST_COMP,),
        ).fetchall()
        print(f"  baseline comp {DEST_COMP} now: {[tuple(r) for r in final]}")
    base.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
