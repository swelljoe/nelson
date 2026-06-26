#!/usr/bin/env python3
"""Delete a competitor's runs (and their findings + judgments) so the matrix
re-plans those cells fresh on the next `bench loop`.

Use this when a competitor's recorded runs are not what you want to keep and a
plain idempotent loop would skip them:
  * the model was re-pointed to a new endpoint (e.g. a crippled local run is
    being replaced by a hosted one) — clear ALL its runs;
  * a batch of cells failed for an infrastructure reason (server OOM, timeout)
    and you want them re-planned as empty — clear only --status infra_error.

A plain `bench loop` (NO --retry-failed) then re-plans exactly the now-empty
cells: empty cells are planned, while other models' historical retryable cells
stay skipped. (`--retry-failed` is roster-wide and would re-run everyone.)

    python clear_competitor_runs.py <db> <competitor-name> [--status S [S ...]]

--status filters by run status (e.g. infra_error auth_failed running). Omit to
clear every run for the competitor. Prints counts; commits only after printing.
"""

from __future__ import annotations

import argparse
import sqlite3


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("db")
    ap.add_argument("competitor", help="exact competitor name")
    ap.add_argument(
        "--status",
        nargs="*",
        default=None,
        help="only clear runs in these statuses (default: all)",
    )
    args = ap.parse_args()

    con = sqlite3.connect(args.db)
    where = "comp.name = ?"
    params: list = [args.competitor]
    if args.status:
        where += " AND r.status IN (%s)" % ",".join("?" * len(args.status))
        params += args.status

    rids = [
        r[0]
        for r in con.execute(
            f"SELECT r.id FROM runs r JOIN competitors comp ON comp.id = r.competitor_id "
            f"WHERE {where}",
            params,
        ).fetchall()
    ]
    if not rids:
        print(f"no matching runs for {args.competitor!r} (status={args.status})")
        return 0

    rph = ",".join("?" * len(rids))
    fids = [
        r[0]
        for r in con.execute(
            f"SELECT id FROM run_findings WHERE run_id IN ({rph})", rids
        ).fetchall()
    ]

    jn = 0
    if fids:
        fph = ",".join("?" * len(fids))
        jn = con.execute(
            f"DELETE FROM judgments WHERE target_kind IN ('fp','truth','refusal') "
            f"AND target_id IN ({fph})",
            fids,
        ).rowcount
    fn = con.execute(f"DELETE FROM run_findings WHERE run_id IN ({rph})", rids).rowcount
    rn = con.execute(f"DELETE FROM runs WHERE id IN ({rph})", rids).rowcount
    con.commit()
    print(f"cleared {rn} runs, {fn} findings, {jn} judgments for {args.competitor!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
