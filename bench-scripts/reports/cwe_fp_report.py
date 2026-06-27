#!/usr/bin/env python3
"""Self-contained HTML report for the CWE-sweep false-positive probe.

Reads nelson-fp-sweep.db (per-CWE finding counts) plus the persisted libyang
FP-judge verdicts (run_findings.judge_fp_verdict, written by fpjudge_libyang.py)
and renders one standalone HTML file — the per-(model, case) CWE matrix, the
false-positive summary, and the libyang real-vs-hallucination verdicts.

    .venv/bin/python cwe_fp_report.py [--db nelson-fp-sweep.db] [--out cwe-fp-report.html]
"""

import argparse
import html
import sqlite3
from pathlib import Path

from nelson.cwe import applicable_cwes, cwe_name
from nelson.inventory import LANGUAGE_MAP
from nelson.report_style import BASE_CSS, THEME_HEAD, THEME_TOGGLE, THEME_VARS

CASES = {  # ext_id -> (label, found/missed)
    "GHSA-w52v-v783-gw97": ("Ghost — SQL injection", "consistently FOUND"),
    "GHSA-9f49-8x56-jmjc": ("libyang — use-after-free", "consistently MISSED"),
}


def cwe_of(tf: str, trial: int) -> str:
    cwes = [c.id for c in applicable_cwes(LANGUAGE_MAP[Path(tf).suffix.lower()])]
    return cwes[trial] if 0 <= trial < len(cwes) else "OPEN"


def esc(s) -> str:
    return html.escape(str(s)) if s is not None else ""


def load(db: str):
    c = sqlite3.connect(db)
    c.row_factory = sqlite3.Row
    runs = c.execute(
        """
        SELECT r.id rid, cp.name model, ca.ext_id case_ext, ca.cwe gt_cwe,
               r.target_file tf, r.trial trial, r.status status
        FROM runs r JOIN competitors cp ON cp.id = r.competitor_id
        JOIN cases ca ON ca.id = r.case_id
        ORDER BY cp.name, ca.ext_id, r.target_file, r.trial, r.id
        """
    ).fetchall()
    finds = c.execute(
        "SELECT run_id, line_start, description, judge_fp_verdict, judge_reasoning FROM run_findings"
    ).fetchall()
    c.close()
    by_run: dict[int, list] = {}
    for f in finds:
        by_run.setdefault(f["run_id"], []).append(f)
    # collapse to best run per (model,case,file,trial): prefer complete, else latest id
    best: dict[tuple, sqlite3.Row] = {}
    for r in runs:
        k = (r["model"], r["case_ext"], r["tf"], r["trial"])
        cur = best.get(k)
        rank = (r["status"] == "complete", r["rid"])
        if cur is None or rank > (cur["status"] == "complete", cur["rid"]):
            best[k] = r
    return list(best.values()), by_run


VCLASS = {
    "REAL": "v-real",
    "FALSE-POS": "v-fp",
    "ERROR": "v-err",
    "UNDETERMINED": "v-und",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="nelson-fp-sweep.db")
    ap.add_argument("--out", default="cwe-fp-report.html")
    args = ap.parse_args()

    runs, finds_by_run = load(args.db)

    # index: (model, case_ext, tf) -> {trial: run}
    grouped: dict[tuple, dict] = {}
    for r in runs:
        grouped.setdefault((r["model"], r["case_ext"], r["tf"]), {})[r["trial"]] = r

    models = sorted({r["model"] for r in runs})
    summary_rows = []
    detail_html = []
    site_verdicts: dict[tuple, tuple] = {}  # (file,line) -> (label, reasoning)

    for model in models:
        short = model.rsplit("/", 1)[-1]
        for case_ext, (clabel, cstatus) in CASES.items():
            tfs = sorted({k[2] for k in grouped if k[0] == model and k[1] == case_ext})
            for tf in tfs:
                trials = grouped[(model, case_ext, tf)]
                gt = next(iter(trials.values()))["gt_cwe"]
                fname = tf.rsplit("/", 1)[-1]
                rows_html = []
                hit_gt = False
                open_nf = 0
                wrong_fired = wrong_total = wrong_n = 0
                for trial in sorted(trials):
                    r = trials[trial]
                    cwe = cwe_of(tf, trial)
                    fs = finds_by_run.get(r["rid"], [])
                    nf = len(fs)
                    is_gt = cwe == gt
                    is_open = cwe == "OPEN"
                    if r["status"] == "complete":
                        if is_open:
                            open_nf = nf
                        elif is_gt:
                            hit_gt = nf > 0
                        else:
                            wrong_n += 1
                            if nf:
                                wrong_fired += 1
                                wrong_total += nf
                    # finding cells
                    fcells = []
                    for f in fs:
                        v = f["judge_fp_verdict"]
                        vbadge = (
                            f'<span class="badge {VCLASS.get(v, "")}">{esc(v)}</span>'
                            if v
                            else ""
                        )
                        if v:
                            site_verdicts[(fname, f["line_start"])] = (
                                v,
                                f["judge_reasoning"],
                            )
                        fcells.append(
                            f'<div class="finding">L{esc(f["line_start"])} {vbadge}'
                            f'<span class="fdesc">{esc((f["description"] or "")[:150])}</span></div>'
                        )
                    rowcls = "gt" if is_gt else ("open" if is_open else "")
                    nm = cwe_name(cwe) if cwe != "OPEN" else "open scan (no hint)"
                    tag = " ★GT" if is_gt else ""
                    rows_html.append(
                        f'<tr class="{rowcls}"><td>{esc(cwe)}{tag}</td>'
                        f'<td class="nm">{esc(nm or "")}</td>'
                        f'<td class="ct {"hot" if nf else "zero"}">{nf}</td>'
                        f"<td>{esc(r['status']) if r['status'] != 'complete' else ''}</td>"
                        f'<td class="fnd">{"".join(fcells)}</td></tr>'
                    )
                rate = wrong_fired / wrong_n if wrong_n else 0
                summary_rows.append(
                    (
                        short,
                        clabel,
                        cstatus,
                        gt,
                        hit_gt,
                        wrong_fired,
                        wrong_n,
                        wrong_total,
                        open_nf,
                    )
                )
                detail_html.append(f"""
<h3>{esc(short)} &middot; {esc(clabel)} <span class="sub">({esc(fname)}, GT={esc(gt)})</span></h3>
<p class="metaline">Wrong-CWE prompts that fired: <b>{wrong_fired}/{wrong_n}</b> ({rate:.0%}) &middot;
spurious findings: <b>{wrong_total}</b> &middot; ground-truth-CWE prompt: <b>{"reported a finding" if hit_gt else "empty"}</b> &middot;
open baseline: <b>{open_nf}</b> finding(s)</p>
<table class="matrix"><thead><tr><th>CWE hint</th><th>name</th><th>#</th><th>status</th><th>reported finding(s)</th></tr></thead>
<tbody>{"".join(rows_html)}</tbody></table>""")

    # summary table
    srows = []
    for short, clabel, cstatus, gt, hit, fired, nw, total, opn in summary_rows:
        cs = "found" if "FOUND" in cstatus else "missed"
        srows.append(
            f'<tr><td>{esc(short)}</td><td>{esc(clabel)}<span class="cs {cs}">{esc(cstatus)}</span></td>'
            f'<td>{esc(gt)}</td><td class="{"yes" if hit else "no"}">{"yes" if hit else "no"}</td>'
            f"<td><b>{fired}/{nw}</b> ({(fired / nw if nw else 0):.0%})</td>"
            f"<td>{total}</td><td>{opn}</td></tr>"
        )

    # libyang verdict tally + site table
    tally = {"REAL": 0, "FALSE-POS": 0, "ERROR": 0, "UNDETERMINED": 0}
    vrows = []
    for (fname, line), (lbl, reason) in sorted(
        site_verdicts.items(), key=lambda kv: kv[0][1]
    ):
        tally[lbl] = tally.get(lbl, 0) + 1
        vrows.append(
            f"<tr><td>{esc(fname)}:{esc(line)}</td>"
            f'<td><span class="badge {VCLASS.get(lbl, "")}">{esc(lbl)}</span></td>'
            f'<td class="reason">{esc(reason)}</td></tr>'
        )
    tally_html = " &middot; ".join(
        f'<span class="badge {VCLASS.get(k, "")}">{k}={v}</span>'
        for k, v in tally.items()
        if v
    )

    extra = """
.matrix th,.matrix td{border:1px solid var(--border)}
.matrix td.ct{text-align:center; font-weight:700; width:2.2rem}
td.hot{color:var(--bad)} td.zero{color:var(--text-muted)}
tr.gt{background:var(--amber-bg)} tr.gt td:first-child{font-weight:700}
tr.open{background:var(--hover)}
td.nm{color:var(--text-muted); font-size:.85em} td.fnd{font-size:.82em}
.finding{border:0; background:transparent; padding:0; margin:2px 0; border-radius:0}
.fdesc{color:var(--text-muted); margin-left:.4rem}
.badge{color:#fff}
.v-real{background:var(--bad)} .v-fp{background:var(--text-muted)}
.v-err{background:var(--orange)} .v-und{background:var(--text-muted)}
.cs{display:inline-block; margin-left:.5rem; font-size:.72rem; padding:1px 7px;
  border-radius:10px; font-weight:700}
.cs.found{background:var(--good-bg); color:var(--good)}
.cs.missed{background:var(--bad-bg); color:var(--bad)}
td.yes{color:var(--good); font-weight:700} td.no{color:var(--bad); font-weight:700}
.metaline{color:var(--text-muted); font-size:.9rem; margin:.3rem 0 .2rem}
.reason{color:var(--text-muted); font-size:.84em}
.keyrow{display:flex; gap:1.4rem; flex-wrap:wrap; margin:.5rem 0}
.keyrow div{background:var(--surface); border:1px solid var(--border); border-radius:8px;
  padding:.7rem 1rem; flex:1; min-width:230px}
.big{font-size:1.5rem; font-weight:800}
"""
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CWE-sweep false-positive probe</title>{THEME_HEAD}
<style>{THEME_VARS}{BASE_CSS}{extra}</style></head><body>{THEME_TOGGLE}
<h1>CWE-sweep false-positive probe</h1>
<p class="lede">Does the &ldquo;many individual CWE prompts&rdquo; method (one oracle-style hint per
applicable CWE, union the findings) spike false positives? Two local Qwen&nbsp;3.6 models
(<code>27b</code> dense, <code>35b-A3b</code> MoE) over two single-file cases &mdash; one the
models find, one they miss. Each applicable CWE is hinted in its own run; the
ground-truth-CWE prompt and an open (no-hint) baseline are included for reference. Finding
counts are raw model output; the libyang non-target findings are then adjudicated by an
Opus FP-judge that reads the pre-patch source.</p>

<div class="callout"><b>Headline:</b> the raw &ldquo;% of wrong CWEs that fired&rdquo; overstates the
danger &mdash; behavior is <b>bimodal by case difficulty</b>.
<div class="keyrow">
<div><span class="cs found">FOUND case (Ghost SQLi)</span><br><span class="big">relabeling</span><br>
nearly every wrong-CWE prompt re-reports the <i>same</i> real bug at the <i>same line</i>.
Dedup-by-location collapses them to the one correct finding. Redundancy, not FPs.</div>
<div><span class="cs missed">MISSED case (libyang UAF)</span><br><span class="big">{tally.get("REAL", 0)} real / {tally.get("FALSE-POS", 0)} false</span><br>
diverse, distinctly-located findings. The FP-judge finds a genuine <i>second</i> bug (a
revision-date <code>strcpy</code> overflow) mixed with an equal volume of hallucinations.</div>
</div></div>

<h2>False-positive summary</h2>
<table><thead><tr><th>model</th><th>case</th><th>GT CWE</th><th>GT-CWE prompt fired</th>
<th>wrong-CWE prompts fired</th><th>spurious findings</th><th>open baseline</th></tr></thead>
<tbody>{"".join(srows)}</tbody></table>
<p class="metaline">&ldquo;GT-CWE prompt fired&rdquo; = the ground-truth-CWE hint produced &ge;1 raw finding
(not judge-verified as the actual bug). &ldquo;wrong-CWE prompts fired&rdquo; = how many of the
remaining applicable CWEs produced any finding despite the hint&rsquo;s explicit
&ldquo;do not invent an issue to fit the category&rdquo; guard.</p>

<h2>libyang non-target findings — Opus FP-judge verdicts</h2>
<p class="metaline">Distinct code sites (deduped across CWE prompts and both models), each judged
against the pre-patch source. The planted UAF region is excluded (it is the candidate true
positive, not a false positive). Tally: {tally_html}</p>
<table><thead><tr><th>site</th><th>verdict</th><th>judge reasoning</th></tr></thead>
<tbody>{"".join(vrows)}</tbody></table>

<h2>Per-model CWE matrix</h2>
<p class="metaline">★GT marks the ground-truth-CWE row (highlighted); the blue row is the open
baseline. <span class="badge v-real">REAL</span>/<span class="badge v-fp">FALSE-POS</span> badges
on a finding are the libyang FP-judge verdict.</p>
{"".join(detail_html)}

<p style="color:#999;font-size:.8rem;margin-top:2rem">Generated from <code>{esc(args.db)}</code>.
Sweep run on local Qwen (free); FP-judge on Opus.</p>
</body></html>"""

    Path(args.out).write_text(doc)
    print(f"wrote {args.out} ({len(doc)} bytes)")


if __name__ == "__main__":
    main()
