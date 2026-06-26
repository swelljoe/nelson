#!/usr/bin/env python3
"""Surgically add/update competitors in a DB WITHOUT syncing the roster.

This is the heart of the "add a model to the baseline" workflow. The obvious
path — `bench loop --competitors roster.yaml` — also *syncs* the roster: every
active competitor absent from the file is RETIRED, and the leaderboard then drops
retired models from the report. That is correct when the YAML is the source of
truth, but wrong when you only want to ADD a model and keep every existing result
visible.

So instead we upsert just the competitors in the given roster (idempotent), leave
everyone else active, and run `bench loop` with NO --competitors. The matrix
planner sees the existing models as already fully covered (0 cells) and fills only
the new (model x case x file) cells.

    python surgical_upsert.py <db> <roster.yaml> [roster2.yaml ...]
"""

from __future__ import annotations

import sys

from nelson.automate import load_competitors
from nelson.db import Database


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    db_path, rosters = argv[0], argv[1:]
    db = Database(db_path)
    n = 0
    for roster in rosters:
        for c in load_competitors(roster):
            db.upsert_competitor(c.to_db_fields())
            print(f"upserted {c.name}")
            n += 1
    print(f"done: {n} competitor(s) upserted into {db_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
