#!/usr/bin/env python3
"""Verify the corrected run is now a HIT and regenerate the baseline report.

No judge spend: load_run_score / _write_leaderboard read persisted columns only.
"""

from __future__ import annotations

import sys

from nelson.automate import _write_leaderboard
from nelson.db import Database
from nelson.score import ClaudeTruthJudge, Scorer, drop_competitors, leaderboard

DB = "nelson.db"
HTML = "bench-report-gemma-minimax.html"

db = Database(DB)
scorer = Scorer(db, ClaudeTruthJudge(model="opus"))

# Verify run 416 (hy3-preview / GHSA-f26g) is now a hit.
rs = scorer.load_run_score(416)
print(f"run 416  {rs.competitor_name}  {rs.case_ext_id}  outcome={rs.outcome}")

# Leaderboard line for hy3-preview for sanity.
retired = {c["name"] for c in db.list_competitors() if c["status"] == "retired"}
run_scores = drop_competitors(
    [scorer.load_run_score(r["id"]) for r in db.list_runs()], retired
)
for e in leaderboard(run_scores):
    if "hy3" in e.competitor_name:
        print(f"hy3 leaderboard: hits={e.hits} misses={e.misses} cases={e.cases}")

if "--write" in sys.argv:
    _write_leaderboard(db, scorer, HTML)
    print(f"wrote {HTML}")
else:
    print("(dry run; pass --write to regenerate the report)")
