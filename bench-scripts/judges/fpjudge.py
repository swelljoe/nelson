#!/usr/bin/env python3
"""Code-grounded FP-judge over off-target findings — the generalized form of
fpjudge_promptlab.py, fpjudge_gemma_promptlab.py and fpjudge_libyang.py.

An experiment run records every finding a model reported, but only the ones that
localize to the planted CVE are "the known bug". The rest are off-target and
ambiguous: a genuine secondary bug the project never fixed, or a hallucination.
This pass runs the production Opus FP-judge (reads the pre-patch source via
`git show vuln_commit:path`, NEVER the advisory) over the off-target findings so a
report can show real-secondary-bugs vs false-positives.

What it does (identical across the three originals):
  * fetch every finding from completed runs across all given DBs;
  * on-target findings (localize within --tolerance of a ground-truth hunk) are
    the known CVE — left untouched, not judged;
  * off-target findings are deduped GLOBALLY by distinct site
    (case, file, line, cwe) and judged once — the verdict is a property of the
    CODE, so the same bug found across tiers/trials/arms/models is one judge call,
    its verdict reused everywhere and persisted to every row at that site;
  * idempotent: a site already carrying a persisted CLEAN verdict is reused
    (a persisted ERROR is a transient failure and gets retried).

Pass one --db per tier to get the per-tier "does a smaller quant hallucinate
more?" breakdown (the four-DB Qwen quant sweep); pass one for a single-DB run
(gemma). Restrict to one case with --case (the libyang-only pass).

    python fpjudge.py --db nelson-promptlab-bf16.db --db nelson-promptlab.db \
        --db nelson-promptlab-6bit.db --db nelson-promptlab-4bit.db \
        --label BF16 --label Q8 --label Q6 --label Q4
    python fpjudge.py --db nelson-gemma-promptlab.db
    python fpjudge.py --db nelson-fp-sweep.db --case GHSA-9f49-8x56-jmjc
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

from nelson.corpus import load_manifest_dir
from nelson.score import ClaudeFPJudge, GitCodeProvider, ReportedFinding, localize

_FETCH = """
    SELECT f.id fid, ca.ext_id ext, f.file file, f.line_start line_start,
           f.line_end line_end, f.description description, f.cwe cwe,
           f.confidence confidence, f.judge_fp_verdict jv, f.judge_reasoning jr
    FROM run_findings f
    JOIN runs r ON r.id = f.run_id
    JOIN cases ca ON ca.id = r.case_id
    WHERE r.status = 'complete'
"""


def to_label(v) -> str:
    if v.error:
        # "source unavailable" is STABLE (the model cited a path git can't
        # resolve — mangled/hallucinated), so persist UNDETERMINED and never
        # retry; any OTHER error (CLI timeout/hiccup) stays ERROR for retry.
        return "UNDETERMINED" if v.error == "source unavailable" else "ERROR"
    if v.is_real is True:
        return "REAL"
    if v.is_real is False:
        return "FALSE-POS"
    return "UNDETERMINED"


def site_key(r: dict) -> tuple:
    return (r["ext"], r["file"], r["line_start"], r["cwe"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", action="append", required=True, help="experiment DB (repeatable)")
    ap.add_argument("--label", action="append", default=[], help="tier label per --db (optional)")
    ap.add_argument("--case", default=None, help="restrict to one case ext_id")
    ap.add_argument("--cases-dir", default="cases/")
    ap.add_argument("--tolerance", type=int, default=10)
    ap.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="judge CLI timeout (300 vs 180 default: big C clips run long under load)",
    )
    args = ap.parse_args()

    dbs = args.db
    labels = args.label or [Path(d).stem for d in dbs]
    if len(labels) != len(dbs):
        raise SystemExit(f"got {len(labels)} --label for {len(dbs)} --db")

    cases = {c.ext_id: c for c in load_manifest_dir(args.cases_dir)}
    code = GitCodeProvider()
    judge = ClaudeFPJudge(model="opus", timeout=args.timeout)

    # Gather off-target findings across every DB, tagged with their tier + db so
    # the verdict can be written back to the exact row it came from.
    offtarget: list[tuple[str, str, dict]] = []  # (tier, db, row)
    ontarget_n = 0
    for tier, db in zip(labels, dbs):
        c = sqlite3.connect(db)
        c.row_factory = sqlite3.Row
        sql = _FETCH + (" AND ca.ext_id = ?" if args.case else "")
        rows = c.execute(sql, (args.case,) if args.case else ()).fetchall()
        c.close()
        for r in rows:
            case = cases[r["ext"]]
            if localize(r["file"], r["line_start"], case.gt_hunks, args.tolerance).matched:
                ontarget_n += 1
                continue
            offtarget.append((tier, db, dict(r)))

    # Distinct site -> verdict. Seed from any already-persisted CLEAN verdict.
    verdicts: dict[tuple, tuple[str, str]] = {}
    for _tier, _db, r in offtarget:
        key = site_key(r)
        if key not in verdicts and r["jv"] and r["jv"] != "ERROR":
            verdicts[key] = (r["jv"], r["jr"] or "")

    # Judge each not-yet-known distinct site once.
    total_cost = 0.0
    judged = 0
    sites: list[tuple] = []
    for _tier, _db, r in offtarget:
        key = site_key(r)
        if key not in sites:
            sites.append(key)
        if key in verdicts:
            continue
        case = cases[r["ext"]]
        src = code.source(case, r["file"])
        v = judge.judge(ReportedFinding.from_row(r), src)
        total_cost += v.cost_usd or 0.0
        verdicts[key] = (to_label(v), v.reasoning or "")
        judged += 1
        print(
            f"  judged [{judged}] {r['ext']} "
            f"{(r['file'] or '?').rsplit('/', 1)[-1]}:{r['line_start']} "
            f"{r['cwe']} -> {verdicts[key][0]}",
            flush=True,
        )

    # Persist (idempotent) the site verdict onto every off-target row, per DB.
    for _tier, db in zip(labels, dbs):
        c = sqlite3.connect(db)
        for _t2, d2, r in offtarget:
            if d2 != db:
                continue
            lbl, reason = verdicts[site_key(r)]
            c.execute(
                "UPDATE run_findings SET judge_fp_verdict=?, judge_reasoning=? WHERE id=?",
                (lbl, reason, r["fid"]),
            )
        c.commit()
        c.close()

    # ---- report -------------------------------------------------------------
    print("\n" + "=" * 72)
    print(
        f"on-target rows (known CVE, not judged): {ontarget_n}\n"
        f"off-target rows: {len(offtarget)} over {len(sites)} distinct sites\n"
        f"judged this run: {judged}  (rest reused persisted verdicts)\n"
        f"FP-judge Opus spend this run: ${total_cost:.2f}"
    )

    print("\n" + "=" * 72 + "\n## DISTINCT-SITE VERDICTS BY CASE\n" + "=" * 72)
    by_case: dict[str, dict[str, int]] = {}
    seen: set[tuple] = set()
    for _tier, _db, r in offtarget:
        key = site_key(r)
        if key in seen:
            continue
        seen.add(key)
        lbl = verdicts[key][0]
        by_case.setdefault(r["ext"], {}).setdefault(lbl, 0)
        by_case[r["ext"]][lbl] += 1
    for ext in sorted(by_case):
        line = ", ".join(f"{k}={v}" for k, v in sorted(by_case[ext].items()))
        print(f"  {ext:24s} {line}")

    # Per-tier row-level breakdown (only meaningful with >1 DB).
    if len(dbs) > 1:
        print("\n" + "=" * 72 + "\n## OFF-TARGET ROWS BY TIER x VERDICT\n" + "=" * 72)
        by_tier: dict[str, dict[str, int]] = {}
        for tier, _db, r in offtarget:
            lbl = verdicts[site_key(r)][0]
            by_tier.setdefault(tier, {}).setdefault(lbl, 0)
            by_tier[tier][lbl] += 1
        for tier in labels:
            t = by_tier.get(tier, {})
            line = ", ".join(f"{k}={v}" for k, v in sorted(t.items())) or "(none)"
            print(f"  {tier:5s} {line}")


if __name__ == "__main__":
    main()
