#!/usr/bin/env python3
"""Run the real Opus truth judge on the misnumbering flips found offline.

For each flip in ``misnumber_flips.json``, ask the same ``ClaudeTruthJudge`` the
benchmark uses whether the model's finding — judged at its RELOCATED (true) line,
i.e. what it would have cited had read_file numbered its output — is the same bug
as the advisory. This is the certainty step: a flip only counts as a genuinely
suppressed detection if the judge rules same_bug. Read-only on the DB.
"""

from __future__ import annotations

import json
import sqlite3

from nelson.corpus import Case
from nelson.score import ClaudeTruthJudge, ReportedFinding

conn = sqlite3.connect("file:nelson.db?mode=ro", uri=True)
conn.row_factory = sqlite3.Row
flips = json.loads(open("misnumber_flips.json").read())
judge = ClaudeTruthJudge(model="opus")

for fl in flips:
    row = conn.execute("SELECT * FROM cases WHERE ext_id = ?", (fl["case"],)).fetchone()
    case = Case.from_row(row)
    finding = ReportedFinding(
        file=fl["file"],
        line=fl["true_line"],
        description=fl["explanation"],
        cwe=fl["reported_cwe"],
    )
    v = judge.judge(case, finding)
    print(f"=== {fl['case']} / {fl['comp']} (run {fl['run_id']}) ===")
    print(
        f"  claimed L{fl['claimed_line']} -> true L{fl['true_line']} (drift {fl['drift']})"
    )
    print(f"  code: {fl['code'][:120]}")
    verdict = "ERROR" if v.error else ("SAME_BUG" if v.same_bug else "different_bug")
    print(f"  VERDICT: {verdict}")
    print(f"  reasoning: {v.reasoning or v.error}")
