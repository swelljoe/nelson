#!/usr/bin/env python3
"""CWE-sweep false-positive analysis.

Reads the per-model experiment DBs written by run_cwe_fp_experiment.py and, for
each (model, case, file), maps every run back to the CWE it was hinted with
(trial index -> applicable_cwes(language) order; the last trial is the open
baseline) and counts the findings the model reported under that prompt
(``run_findings`` rows).

The headline question: across the *wrong*-CWE prompts (every applicable CWE that
is NOT the case's ground truth), how often does the model report a finding it was
explicitly told not to invent? That's the false-positive surface of the "many
individual CWE prompts" method.

    .venv/bin/python analyze_cwe_fp.py [db_path]
"""

import sqlite3
import sys
from pathlib import Path

from nelson.cwe import applicable_cwes
from nelson.inventory import LANGUAGE_MAP

# Both models write to one shared DB (run_ids must be globally unique so the
# nelson-run-<id> containers don't collide); model is recovered per run via the
# competitors join.
DB = sys.argv[1] if len(sys.argv) > 1 else "nelson-fp-sweep.db"

# A run only counts toward FP rates if the model got a fair look.
SCORED = "complete"


def language(file_path: str) -> str:
    return LANGUAGE_MAP[Path(file_path).suffix.lower()]


def hinted_cwe(file_path: str, trial: int) -> str:
    """Recover which CWE a run was hinted with from its trial index."""
    cwes = [c.id for c in applicable_cwes(language(file_path))]
    return cwes[trial] if 0 <= trial < len(cwes) else "OPEN"


def load(db_path: str):
    """One best run per (model, case, file, trial), joined to case + a finding count.

    A crashed-server resume leaves a stale ``infra_error`` row alongside the fresh
    retry, so collapse duplicates to a single row per cell — preferring a
    ``complete`` run, else the latest attempt (highest id)."""
    c = sqlite3.connect(db_path)
    c.row_factory = sqlite3.Row
    rows = c.execute(
        """
        SELECT cp.name AS model, ca.ext_id AS case_ext, ca.cwe AS gt_cwe,
               r.target_file AS tf, r.trial AS trial, r.status AS status, r.id AS id,
               (SELECT COUNT(*) FROM run_findings f WHERE f.run_id = r.id) AS nf
        FROM runs r
        JOIN cases ca ON ca.id = r.case_id
        JOIN competitors cp ON cp.id = r.competitor_id
        ORDER BY cp.name, ca.ext_id, r.target_file, r.trial
        """
    ).fetchall()
    c.close()
    best: dict[tuple, sqlite3.Row] = {}
    for r in rows:
        key = (r["model"], r["case_ext"], r["tf"], r["trial"])
        cur = best.get(key)
        if cur is None:
            best[key] = r
            continue
        # prefer complete; otherwise the later attempt
        cur_rank = (cur["status"] == SCORED, cur["id"])
        new_rank = (r["status"] == SCORED, r["id"])
        if new_rank > cur_rank:
            best[key] = r
    return list(best.values())


def main() -> None:
    if not Path(DB).exists():
        raise SystemExit(f"{DB} not found")
    overall = []  # (model, case_ext, gt_cwe, hit_gt, n_wrong, fp_prompts, fp_total, open_nf)
    rows = load(DB)
    # group by model, then (case, file)
    by_model: dict[str, list[sqlite3.Row]] = {}
    for r in rows:
        by_model.setdefault(r["model"], []).append(r)

    for model, mrows in by_model.items():
        short = model.rsplit("/", 1)[-1]
        print(f"{'=' * 72}\n## {short}  ({model})\n{'=' * 72}")
        groups: dict[tuple[str, str], list[sqlite3.Row]] = {}
        for r in mrows:
            groups.setdefault((r["case_ext"], r["tf"]), []).append(r)

        for (case_ext, tf), grp in groups.items():
            gt = grp[0]["gt_cwe"]
            fname = tf.rsplit("/", 1)[-1]
            print(f"\n### {case_ext}  GT={gt}  file={fname}")
            print(f"{'trial':>5}  {'CWE':<9} {'GT?':<3} {'status':<12} {'findings':>8}")
            print("-" * 44)
            hit_gt = False
            open_nf = 0
            wrong = []  # (cwe, nf) over complete, non-GT, non-OPEN prompts
            for r in grp:
                cwe = hinted_cwe(tf, r["trial"])
                is_gt = cwe == gt
                flag = "*" if is_gt else ("o" if cwe == "OPEN" else "")
                print(
                    f"{r['trial']:>5}  {cwe:<9} {flag:<3} "
                    f"{r['status']:<12} {r['nf']:>8}"
                )
                if r["status"] != SCORED:
                    continue
                if cwe == "OPEN":
                    open_nf = r["nf"]
                elif is_gt:
                    hit_gt = r["nf"] > 0
                else:
                    wrong.append((cwe, r["nf"]))
            fp_prompts = sum(1 for _, nf in wrong if nf > 0)
            fp_total = sum(nf for _, nf in wrong)
            n_wrong = len(wrong)
            rate = fp_prompts / n_wrong if n_wrong else 0.0
            print(
                f"  GT-CWE detection: {'FINDING' if hit_gt else 'none'}   "
                f"open-baseline findings: {open_nf}"
            )
            print(
                f"  WRONG-CWE prompts firing: {fp_prompts}/{n_wrong} ({rate:.0%})   "
                f"spurious findings total: {fp_total}"
            )
            if wrong:
                fired = [f"{cwe}({nf})" for cwe, nf in wrong if nf > 0]
                if fired:
                    print(f"  fired on: {', '.join(fired)}")
            overall.append(
                (short, case_ext, gt, hit_gt, n_wrong, fp_prompts, fp_total, open_nf)
            )
        print()

    # Cross-model summary.
    print(f"{'=' * 72}\n## SUMMARY — wrong-CWE false-positive surface\n{'=' * 72}")
    print(
        f"{'model':<16} {'case':<22} {'GT':<8} {'GT-hit':<7} "
        f"{'FP prompts':<11} {'spurious':<9} {'open':<5}"
    )
    print("-" * 80)
    for m, ce, gt, hit, nw, fpp, fpt, opn in overall:
        print(
            f"{m:<16} {ce:<22} {gt:<8} {('yes' if hit else 'no'):<7} "
            f"{f'{fpp}/{nw}':<11} {fpt:<9} {opn:<5}"
        )


if __name__ == "__main__":
    main()
