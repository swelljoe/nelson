#!/usr/bin/env python3
"""Self-contained HTML report for the Gemma QAT prompt-lab over the FULL 9-case
corpus (untracked experiment).

Unlike promptlab_report.py (6 single-file Qwen cases) this aggregates the 3
MULTI-FILE cases correctly: a case is detected in a trial if ANY of its audited
files localizes a finding to the planted bug — the same notion of case detection
the 9-case baseline uses. Detection here is localization-only (the free gate); a
localized hit is "right place", and confirming SAME-bug needs the truth judge as
a second pass (run that only on the localized hits worth confirming).

    .venv/bin/python gemma_promptlab_report.py \
        [--db nelson-gemma-promptlab.db] [--out gemma-promptlab-report.html] [--repeat 2]
"""

import argparse
import html
import sqlite3

from nelson.corpus import load_manifest_dir
from nelson.report_style import BASE_CSS, THEME_HEAD, THEME_TOGGLE, THEME_VARS
from nelson.score import localize

ARMS = ["open", "plan", "checklist"]
TOLERANCE = 10  # mirror the benchmark's localization gate
CASES_DIR = "cases/"

# 9-case difficulty reference: baseline case-detection from nelson.db
# (hits / eligible competitors, computed 2026-06-07, incl. the hy3 line-number
# correction), ordered easy -> hard. (CWE, language, role) label the matrix.
META = [
    ("GHSA-w52v-v783-gw97", "CWE-89", "JS", "17/20", "hit anchor"),
    ("GHSA-j273-m5qq-6825", "CWE-22", "Java", "12/20", "guard"),
    ("GHSA-f26g-jm89-4g65", "CWE-77", "Rust", "11/19", "medium"),
    ("CVE-2026-7474", "CWE-22", "Go", "7/21", "medium-hard"),
    ("GHSA-9f49-8x56-jmjc", "CWE-416", "C", "2/21", "hard"),
    ("GHSA-mpxh-8fq3-x8mh", "CWE-787", "C", "1/17", "hard"),
    ("GHSA-x9h5-r9v2-vcww", "CWE-122", "C", "1/20", "hard"),
    ("GHSA-cc7p-2j3x-x7xf", "CWE-863", "PHP", "0/20", "hardest"),
    ("CVE-2026-5199", "CWE-639", "Go", "0/20", "hardest"),
]
ORDER = [m[0] for m in META]
BASELINE = {m[0]: m[3] for m in META}
CASE_META = {m[0]: (m[1], m[2], m[4]) for m in META}
# "hard" = baseline detection <= 1 competitor (the cases worth watching).
HARD = {e for e in ORDER if BASELINE[e].split("/")[0] in ("0", "1")}

# Shared themeable stylesheet (light/dark) + this report's bespoke styles (hard
# rows + the precision-section verdict bar, copied from the Qwen compare report).
CSS = (
    THEME_VARS
    + BASE_CSS
    + """
.hardrow td.l{font-weight:600}
tr.tot td{border-top:2px solid var(--border);background:var(--surface-2)}
.legend{font-size:13px;color:var(--text-muted);margin:.4rem 0 1.2rem}
.vbar{display:inline-flex;width:120px;height:13px;border:1px solid var(--border);
border-radius:2px;overflow:hidden;vertical-align:middle}
.vbar span{display:block;height:100%}
.vreal{background:var(--good)}.vfp{background:var(--bad)}.vrev{background:var(--text-muted)}
"""
)

# Map the persisted judge_fp_verdict labels to display buckets.
_VLBL = {"REAL": "real", "FALSE-POS": "fp", "UNDETERMINED": "review", "ERROR": "review"}


def short(m):
    return m.rsplit("/", 1)[-1]


def load(db_path, repeat):
    """best[(model, ext, file, trial)] -> row; plus a per-run localized-hit fn.

    Keyed by FILE (not just case) so a multi-file case keeps every file's run.
    """
    cases = {c.ext_id: c for c in load_manifest_dir(CASES_DIR)}
    db = sqlite3.connect(db_path)
    db.row_factory = sqlite3.Row
    runs = db.execute(
        """SELECT r.id rid, cp.name model, ca.ext_id ext, r.trial trial,
                  r.target_file file, r.status status
           FROM runs r JOIN competitors cp ON cp.id=r.competitor_id
           JOIN cases ca ON ca.id=r.case_id
           ORDER BY cp.name, ca.ext_id, r.target_file, r.trial, r.id"""
    ).fetchall()
    best = {}
    for r in runs:
        k = (r["model"], r["ext"], r["file"], r["trial"])
        cur = best.get(k)
        if (
            cur is None
            or (r["status"] == "complete" and cur["status"] != "complete")
            or (r["status"] == cur["status"] and r["rid"] > cur["rid"])
        ):
            best[k] = r

    if repeat is None:
        trials = {t for (_m, _e, _f, t) in best}
        repeat = max(1, (max(trials) + 1) // len(ARMS)) if trials else 1

    findings = db.execute(
        "SELECT run_id, file, line_start FROM run_findings"
    ).fetchall()
    by_run = {}
    for f in findings:
        by_run.setdefault(f["run_id"], []).append(f)

    def hit_run(rid, ext):
        case = cases[ext]
        for f in by_run.get(rid, []):
            if localize(f["file"], f["line_start"], case.gt_hunks, TOLERANCE).matched:
                return True
        return False

    return best, repeat, hit_run


def case_trial(best, hit_run, model, ext, trial):
    """Aggregate a case across its files at one trial: None=no data (no complete
    file-run), else True if ANY complete file-run localized the planted bug."""
    rows = [
        r for (m, e, _f, t), r in best.items() if m == model and e == ext and t == trial
    ]
    complete = [r for r in rows if r["status"] == "complete"]
    if not complete:
        return None
    return any(hit_run(r["rid"], ext) for r in complete)


def matrix_html(best, repeat, hit_run, model):
    out = ["<table><tr><th class=l>case</th><th>baseline</th>"]
    out += [f"<th>{a}</th>" for a in ARMS]
    out.append("</tr>")
    for ext in ORDER:
        cwe, lang, role = CASE_META[ext]
        tagcls = "tag hard" if ext in HARD else "tag"
        rowcls = " class=hardrow" if ext in HARD else ""
        out.append(
            f"<tr{rowcls}><td class=l>{ext}<br><small>{cwe} · {lang}</small>"
            f"<span class='{tagcls}'>{role}</span></td><td>{BASELINE[ext]}</td>"
        )
        for ai, _arm in enumerate(ARMS):
            vals = [
                case_trial(best, hit_run, model, ext, ai * repeat + r)
                for r in range(repeat)
            ]
            done = [v for v in vals if v is not None]
            if not done:
                out.append("<td class=nd>—</td>")
                continue
            n = sum(1 for v in done if v)
            cls = "miss" if n == 0 else ("hit" if n == len(done) else "part")
            out.append(f"<td class={cls}>{n}/{len(done)}</td>")
        out.append("</tr>")
    out.append("</table>")
    return "".join(out)


def arm_summary(best, repeat, hit_run, model):
    """Per arm: union (any-trial) detection over solved vs hard-miss cases."""
    rows = {}
    for ai, arm in enumerate(ARMS):
        sh = sc = mh = mc = 0
        for ext in ORDER:
            vals = [
                case_trial(best, hit_run, model, ext, ai * repeat + r)
                for r in range(repeat)
            ]
            if all(v is None for v in vals):
                continue
            anyhit = any(vals)
            if ext in HARD:
                mc += 1
                mh += int(anyhit)
            else:
                sc += 1
                sh += int(anyhit)
        rows[arm] = (sh, sc, mh, mc)
    return rows


def summary_html(per_model_rows, models):
    out = [
        "<table><tr><th class=l>model</th><th class=l>arm</th>"
        "<th>solved-cases</th><th>hard-miss-cases</th></tr>"
    ]
    for model in models:
        for arm in ARMS:
            sh, sc, mh, mc = per_model_rows[model][arm]
            out.append(
                f"<tr><td class=l>{short(model)}</td><td class=l>{arm}</td>"
                f"<td class={'hit' if sc and sh == sc else 'part'}>{sh}/{sc}</td>"
                f"<td class={'miss' if mh == 0 else 'hit'}>{mh}/{mc}</td></tr>"
            )
    out.append("</table>")
    return "".join(out)


def verdict(per_model_rows, models):
    """Data-driven headline: best per-model union detection, hard cases first."""
    lines = []
    for model in models:
        # union across ALL arms (best-of) on hard and solved
        rows = per_model_rows[model]
        # recompute union-of-arms from the arm tuples is lossy; approximate with
        # the max over arms (a case union-detected by any arm counts).
        best_hard = max(mh for (_sh, _sc, mh, _mc) in rows.values())
        best_hard_cov = max(mc for (_sh, _sc, _mh, mc) in rows.values())
        best_solv = max(sh for (sh, _sc, _mh, _mc) in rows.values())
        best_solv_cov = max(sc for (_sh, sc, _mh, _mc) in rows.values())
        lines.append(
            f"<b>{short(model)}</b>: best arm solves {best_solv}/{best_solv_cov} "
            f"solved-tier cases and cracks {best_hard}/{best_hard_cov} hard-miss cases."
        )
    return "<br>".join(lines)


def precision_data(db_path):
    """Read the code-grounded FP-judge verdicts (judge_fp_verdict) off the DB.

    Only off-target findings carry a verdict (the on-target CVE hits were never
    judged). Returns (per_case, per_model, on_n):
      - per_case tallies DISTINCT off-target sites (case,file,line,cwe), deduped
        so it answers "how many real secondary bugs live in this file vs
        hallucinations." The verdict is a property of the code, not the model.
      - per_model tallies off-target ROWS per model — does one model report
        proportionally more hallucinations?
      - on_n = on-target (planted-CVE) findings, not FP-judged, for the caveat.
    """
    per_case, per_model, seen = {}, {}, set()
    on_n = 0
    db = sqlite3.connect(db_path)
    db.row_factory = sqlite3.Row
    rows = db.execute(
        """SELECT ca.ext_id e, cp.name model, f.file file, f.line_start ls,
                  f.cwe cwe, f.judge_fp_verdict v
           FROM run_findings f JOIN runs r ON r.id=f.run_id
           JOIN cases ca ON ca.id=r.case_id
           JOIN competitors cp ON cp.id=r.competitor_id
           WHERE r.status='complete'"""
    ).fetchall()
    db.close()
    for r in rows:
        if not r["v"]:
            on_n += 1  # un-judged == on-target CVE hit (the only rows we skip)
            continue
        lbl = _VLBL.get(r["v"], "review")
        per_model.setdefault(r["model"], {"real": 0, "fp": 0, "review": 0})[lbl] += 1
        key = (r["e"], r["file"], r["ls"], r["cwe"])
        if key in seen:
            continue
        seen.add(key)
        per_case.setdefault(r["e"], {"real": 0, "fp": 0, "review": 0})[lbl] += 1
    return per_case, per_model, on_n


def _verdict_bar(real, fp, review):
    """A compact stacked proportion bar (green real / red fp / grey review)."""
    tot = real + fp + review
    if not tot:
        return "<span class=nd>&mdash;</span>"
    seg = []
    for n, cls in ((real, "vreal"), (fp, "vfp"), (review, "vrev")):
        if n:
            seg.append(f"<span class={cls} style='width:{n / tot * 100:.0f}%'></span>")
    return f"<span class=vbar>{''.join(seg)}</span>"


def precision_case_html(per_case):
    """Per-case: distinct off-target sites split real / FP / undetermined."""
    out = [
        "<table><tr><th class=l>case</th><th>distinct off-target sites</th>"
        "<th>real secondary bug</th><th>false positive</th>"
        "<th>undetermined</th><th>mix</th></tr>"
    ]
    treal = tfp = trev = 0
    for ext in ORDER:
        v = per_case.get(ext)
        cwe, lang, _role = CASE_META[ext]
        if not v:
            out.append(
                f"<tr><td class=l>{ext}<br><small>{cwe} &middot; {lang}</small></td>"
                f"<td class=nd>0</td><td class=nd>&mdash;</td><td class=nd>&mdash;</td>"
                f"<td class=nd>&mdash;</td><td class=nd>&mdash;</td></tr>"
            )
            continue
        real, fp, rev = v["real"], v["fp"], v["review"]
        treal += real
        tfp += fp
        trev += rev
        tot = real + fp + rev
        rowcls = " class=hardrow" if ext in HARD else ""
        out.append(
            f"<tr{rowcls}><td class=l>{ext}<br><small>{cwe} &middot; {lang}</small></td>"
            f"<td>{tot}</td>"
            f"<td class={'hit' if real else 'nd'}>{real or '&mdash;'}</td>"
            f"<td class={'miss' if fp else 'nd'}>{fp or '&mdash;'}</td>"
            f"<td>{rev or '&mdash;'}</td><td>{_verdict_bar(real, fp, rev)}</td></tr>"
        )
    tot = treal + tfp + trev
    out.append(
        f"<tr class=tot><td class=l><b>all cases</b></td><td><b>{tot}</b></td>"
        f"<td class=hit><b>{treal}</b></td><td class=miss><b>{tfp}</b></td>"
        f"<td><b>{trev}</b></td><td>{_verdict_bar(treal, tfp, trev)}</td></tr>"
    )
    out.append("</table>")
    return "".join(out)


def precision_model_html(per_model, models):
    """Per-model: off-target ROW counts by verdict — hallucination propensity."""
    out = [
        "<table><tr><th class=l>model</th><th>off-target rows</th><th>real</th>"
        "<th>false positive</th><th>undetermined</th><th>% real</th><th>mix</th></tr>"
    ]
    for model in models:
        v = per_model.get(model, {"real": 0, "fp": 0, "review": 0})
        real, fp, rev = v["real"], v["fp"], v["review"]
        tot = real + fp + rev
        pct = f"{real / tot * 100:.0f}%" if tot else "&mdash;"
        out.append(
            f"<tr><td class=l>{short(model)}</td><td>{tot}</td>"
            f"<td class=hit>{real}</td><td class=miss>{fp}</td><td>{rev}</td>"
            f"<td>{pct}</td><td>{_verdict_bar(real, fp, rev)}</td></tr>"
        )
    out.append("</table>")
    return "".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="nelson-gemma-promptlab.db")
    ap.add_argument("--out", default="gemma-promptlab-report.html")
    ap.add_argument("--repeat", type=int, default=2)
    args = ap.parse_args()

    best, repeat, hit_run = load(args.db, args.repeat or None)
    models = sorted({m for (m, _e, _f, _t) in best})
    per_model_rows = {m: arm_summary(best, repeat, hit_run, m) for m in models}

    matrices = "".join(
        f"<h3>{short(m)}</h3>{matrix_html(best, repeat, hit_run, m)}" for m in models
    )

    pc, pm, on_n = precision_data(args.db)

    doc = f"""<!doctype html><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Nelson — Gemma 4 QAT prompt-lab (9-case)</title>{THEME_HEAD}<style>{CSS}</style>
{THEME_TOGGLE}
<h1>Gemma 4 (QAT-4bit) prompt-lab — full 9-case corpus</h1>
<p class=sub>The two new Quantization-Aware-Training 4-bit Gemma 4 models
(<code>gemma4-31b-qat</code> dense, <code>gemma4-26b-a4b-qat</code> MoE), self-hosted
and free, over all 9 corpus cases. Three non-leaking prompting arms
(<b>open</b> / <b>plan-first</b> / <b>CWE-checklist</b>) &times; {repeat} trials/arm
at temperature 0.5. Multi-file cases (nomad/craft/ghost) audit every baseline file;
a case is detected if ANY file localizes within {TOLERANCE} lines of the planted
hunk (the benchmark's gate). Motivation: at 8-bit the 26B-A4B MoE tied the
field-leading 4/9 detection, so we feel out the QAT-4bit edges as we did Qwen 3.6.</p>

<div class=callout>
<div class=big>Headline</div>
{verdict(per_model_rows, models)}
<br><small>(union = detected by at least one trial of that arm; localization-only,
no truth-judge yet — confirm the hard-case hits with the judge before claiming them.)</small>
</div>

<h2>Detection matrix</h2>
<p>Cell = case-detected trials / completed trials for that arm (a case counts if
any of its files hit). <span class=hit>green</span> = every trial, <span class=part>
amber</span> = some, <span class=miss>red</span> = none, <span class=nd>—</span> =
no data. Hard-miss cases (baseline ≤1) bold.</p>
{matrices}

<h2>Arm summary (union over {repeat} trials)</h2>
<p>Cases each arm detected in at least one trial, split by whether the baseline
solved the case (regression guard) or missed it (the real question). Denominators
count only cases with a completed run.</p>
{summary_html(per_model_rows, models)}

<h2>Off-target findings: real secondary bugs vs false positives</h2>
<p>Detection above is only half the story. Every finding a model reports that
<i>isn't</i> the planted CVE still lands in front of a human (or a downstream
model) who burns time and tokens triaging it — so a model that floods the queue
with hallucinations is expensive even when it finds the real bug. But these are
real in-the-wild OSS files, not synthetic single-bug fixtures, so an off-target
finding is <b>not automatically</b> a false positive: it may be a genuine,
never-fixed secondary bug. We judged every off-target finding with the
code-grounded Opus FP-judge (it reads the <b>pre-patch source</b> via
<code>git show</code>, never the advisory) and classified each as a real bug, a
false positive, or undetermined. Distinct sites are deduped (the verdict is a
property of the code, not which model/arm/trial surfaced it), so this counts how
many real secondary bugs actually live in each file.</p>
{precision_case_html(pc)}
<p class=legend><span class=hit>green</span> = judge-confirmed real bug,
<span class=miss>red</span> = false positive, grey = undetermined. Hard-miss cases
bold. The on-target CVE hits ({on_n} findings) are not shown here — those are the
planted bug, already known real.</p>

<h3>Does one model hallucinate more? Off-target rows by model</h3>
<p>Per-model off-target finding volume split by verdict (row-level, so repeated
reports of the same site count each time). A lower <b>% real</b> means more of
that model's noise is hallucination — the cost a human pays per report.</p>
{precision_model_html(pm, models)}

<h2>Method &amp; caveats</h2>
<ul>
<li>Models: <code>gemma4-31b-qat</code> (dense, 10.20.30.1) and
<code>gemma4-26b-a4b-qat</code> (MoE, 10.20.30.2), self-hosted llama-server, free,
QAT-4bit weights.</li>
<li>15 target files over 9 cases (nomad 2, craft 5, ghost 2, rest 1); a case is a
hit if any file localizes. 90 runs/model = 15 files &times; 3 arms &times; {repeat} trials.</li>
<li>Temperature 0.5 so repeats explore different reasoning paths; llama-server
<code>seed=-1</code> randomizes per request. A timeout/infra_error reads as
no-data, never a miss.</li>
<li>Detection is localization-only (free). Hard-case hits still need the truth
judge to confirm SAME-bug; this report is the cheap first pass.</li>
<li>Off-target findings are FP-judged once per distinct site by the code-grounded
Opus judge (<code>fpjudge_gemma_promptlab.py</code>), which reads the pre-patch
source and never the advisory; verdicts persist on <code>judge_fp_verdict</code>
so the report is reproducible from the DB. On-target CVE hits are not judged.</li>
</ul>
<p><small>Generated from {html.escape(args.db)} · {len(models)} models ·
{len(ORDER)} cases &times; {len(ARMS)} arms &times; {repeat} trials.</small></p>
"""
    with open(args.out, "w") as f:
        f.write(doc)
    print(f"wrote {args.out} ({len(doc)} bytes), {len(models)} models")


if __name__ == "__main__":
    main()
