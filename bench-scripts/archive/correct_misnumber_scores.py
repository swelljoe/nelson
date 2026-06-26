#!/usr/bin/env python3
"""Correct baseline scores that the missing-line-number tooling bug suppressed.

Background: ``read_file`` used to return source with no line numbers, so a model
that genuinely found a bug had to HAND-COUNT to cite a location and sometimes
drifted — landing tens of lines off and failing the near-exact localization
gate. ``analyze_misnumber.py`` swept the frozen baseline, relocated every quoted
snippet in the real source, and found the runs where the snippet actually sits on
a ground-truth hunk despite a wrong claimed line (``misnumber_flips.json``).

The harness, not the model, caused those misses, so we correct them. For each
flip we re-run the SAME Opus truth judge the benchmark uses, judged at the
RELOCATED line (what the model would have cited had the tool numbered its
output). Only a ``same_bug`` ruling counts. On a confirmed flip we:
  * set ``line_start``/``line_end`` to the relocated true line, so the gate is
    self-consistent and a future live re-score stays a hit (score_run recomputes
    localize() from line_start);
  * set matches_ground_truth=1 + the truth verdict + reasoning (so the report,
    built from persisted columns, reflects the hit with no judge re-spend);
  * append a truth judgment to the audit ledger.
The original model-claimed line is preserved verbatim in ``runs.raw_output``
(untouched) and restated in the reasoning note, so nothing is lost.

Idempotent: a flip whose finding is already corrected (matches_ground_truth=1) is
skipped. Writes to nelson.db — back it up first.
"""

from __future__ import annotations

import json

from nelson.corpus import Case
from nelson.db import Database
from nelson.score import ClaudeTruthJudge, ReportedFinding

DB = "nelson.db"


def find_finding(db: Database, run_id: int, claimed_line: int, base: str):
    for row in db.run_findings(run_id):
        f = row["file"] or ""
        if row["line_start"] == claimed_line and f.split("/")[-1] == base:
            return row
    return None


def main() -> None:
    flips = json.loads(open("misnumber_flips.json").read())
    db = Database(DB)
    judge = ClaudeTruthJudge(model="opus")
    corrected = skipped = rejected = 0

    for fl in flips:
        base = fl["file"].split("/")[-1]
        row = find_finding(db, fl["run_id"], fl["claimed_line"], base)
        if row is None:
            print(
                f"[!] no finding row for run {fl['run_id']} {base} L{fl['claimed_line']}"
            )
            continue
        if row["matches_ground_truth"] == 1:
            print(
                f"[=] already corrected: run {fl['run_id']} {base} (finding {row['id']})"
            )
            skipped += 1
            continue

        case = Case.from_row(db.get_case(fl["case"]))
        reported = ReportedFinding(
            file=fl["file"],
            line=fl["true_line"],
            description=fl["explanation"],
            cwe=fl["reported_cwe"],
        )
        v = judge.judge(case, reported)
        if v.error or not v.same_bug:
            verdict = "ERROR" if v.error else "different_bug"
            print(
                f"[x] {fl['case']} {fl['comp']}: judge ruled {verdict} -> NOT corrected"
            )
            print(f"    {v.reasoning or v.error}")
            rejected += 1
            continue

        note = (
            f"[line-number-tooling correction] Model cited L{fl['claimed_line']} but its "
            f"quoted code sits at L{fl['true_line']} (drift {fl['drift']}); read_file gave "
            f"no line numbers so the line was hand-counted. Relocated and re-judged at the "
            f"true line. Truth judge: {v.reasoning}"
        )
        db.conn.execute(
            "UPDATE run_findings SET line_start = ?, line_end = ? WHERE id = ?",
            (fl["true_line"], fl["true_line"], row["id"]),
        )
        db.record_finding_score(
            row["id"],
            matches_ground_truth=True,
            judge_truth_verdict=v.label,
            judge_reasoning=note,
        )
        db.add_judgment(
            target_kind="truth",
            target_id=row["id"],
            judge_model=judge.name,
            verdict=v.label,
            reasoning=note,
            tokens_in=v.tokens_in,
            tokens_out=v.tokens_out,
            cost_usd=v.cost_usd,
        )
        print(
            f"[OK] {fl['case']} {fl['comp']} (run {fl['run_id']}, finding {row['id']}): "
            f"L{fl['claimed_line']}->L{fl['true_line']} HIT ({v.label})"
        )
        corrected += 1

    print(f"\ncorrected={corrected} skipped(already)={skipped} rejected={rejected}")


if __name__ == "__main__":
    main()
