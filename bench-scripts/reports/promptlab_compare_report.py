#!/usr/bin/env python3
"""Side-by-side HTML report: Qwen 3.6 quant sweep (BF16 / 8-bit / 6-bit / 4-bit).

Renders the same localized-detection matrix as promptlab_report.py, but with each
arm split into one column per quant — full-precision BF16 (reference) / UD-Q8_K_XL
/ UD-Q6_K_XL / UD-Q4_K_XL — so the precision tiers can be compared at a glance. A
cell group where the tiers DISAGREE on hit-status (one finds the bug, another
doesn't) is flagged with a Δ. Adds a token-usage section: mean generation tokens,
context tokens, and wall-clock per run for each tier — a smaller quant that needs
more tokens to reach the same answer is a false economy for anyone who can run
either. Reuses analyze_promptlab.load for the data layer and promptlab_report's
constants/helpers, so detection is computed identically to the single-DB reports.

    .venv/bin/python promptlab_compare_report.py \
        [--repeat 2] [--out promptlab-compare-report.html]
"""

import argparse
import html
import sqlite3
from statistics import mean

from analyze_promptlab import ARMS, BASELINE, ORDER, load
from nelson.report_style import THEME_HEAD, THEME_TOGGLE
from promptlab_report import CASE_META, CSS, HARD, finding_counts

# Precision columns, reference (full precision) first. (label, db path,
# competitor-name suffix). The 8-bit roster has no suffix; the BF16/6/4-bit
# rosters append theirs to the same base names.
QUANTS = [
    ("BF16", "nelson-promptlab-bf16.db", "-bf16"),
    ("Q8", "nelson-promptlab.db", ""),
    ("Q6", "nelson-promptlab-6bit.db", "-q6-k-xl"),
    ("Q4", "nelson-promptlab-4bit.db", "-q4-k-xl"),
]
REF = QUANTS[0][0]  # full-precision tier the deltas are measured against
REF_SUFFIX = QUANTS[0][2]  # its competitor-name suffix (for the token baseline)

# (display label, base competitor name). Per quant the run name is base+suffix.
FAMILIES = [
    ("Qwen 3.6 27B (dense)", "raw-api-loop/qwen3.6-27b"),
    ("Qwen 3.6 35B-A3B (MoE)", "raw-api-loop/qwen3.6-35b-A3b"),
]

# Extra styles layered on top of promptlab_report.CSS for the paired columns.
EXTRA_CSS = """
th.armgrp{border-bottom:1px solid var(--border)}
.quant{font-size:11px;color:var(--text-muted);font-weight:400}
td.qsep,th.qsep{border-left:2px solid var(--border)}
td.diff{outline:2px solid var(--callout-edge);outline-offset:-2px;position:relative}
td.diff::after{content:'\\0394';position:absolute;top:0;right:2px;font-size:9px;
color:var(--accent);font-weight:700}
.legend{font-size:13px;color:var(--text-muted);margin:.4rem 0 1.2rem}
.same{color:var(--good);font-weight:600}
.warn{color:var(--accent);font-weight:600}
td.up{color:var(--bad);font-weight:600}
td.down{color:var(--good);font-weight:600}
tr.tot td{border-top:2px solid var(--border);background:var(--surface-2)}
.vbar{display:inline-flex;width:120px;height:13px;border:1px solid var(--border);
border-radius:2px;overflow:hidden;vertical-align:middle}
.vbar span{display:block;height:100%}
.vreal{background:var(--good)}.vfp{background:var(--bad)}.vrev{background:var(--text-muted)}
"""


# ---------------------------------------------------------------- data layer


def comp_name(base, suffix):
    return base + suffix


def quant_data(repeat):
    """Per quant: (best, repeat, hit, agg) from its DB; tolerate a missing DB."""
    out = {}
    for qlabel, db, _suf in QUANTS:
        try:
            best, rep, hit = load(db, repeat or None)
            agg = finding_counts(db, rep)
        except sqlite3.OperationalError:
            best, rep, hit, agg = {}, repeat, (lambda _r: False), {}
        out[qlabel] = {"best": best, "repeat": rep, "hit": hit, "agg": agg}
    return out


def token_stats(repeat):
    """Per (quant, competitor): (n, mean_tok_out, mean_tok_in, mean_wall) over
    completed runs. Returns {qlabel: {competitor_name: tuple}}."""
    out = {}
    for qlabel, db, _suf in QUANTS:
        stats = {}
        try:
            conn = sqlite3.connect(db)
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """SELECT cp.name m, r.tokens_out o, r.tokens_in i, r.wall_clock_s w
                   FROM runs r JOIN competitors cp ON cp.id=r.competitor_id
                   WHERE r.status='complete'"""
            ).fetchall()
            conn.close()
        except sqlite3.OperationalError:
            rows = []
        by = {}
        for r in rows:
            by.setdefault(r["m"], []).append((r["o"], r["i"], r["w"]))
        for name, vals in by.items():
            stats[name] = (
                len(vals),
                mean(v[0] for v in vals),
                mean(v[1] for v in vals),
                mean(v[2] for v in vals if v[2] is not None) if vals else 0.0,
            )
        out[qlabel] = stats
    return out


def arm_cell(qd, model, ext, ai):
    """(n_hits, n_completed) for one (quant, model, case, arm), or None."""
    best, repeat, hit = qd["best"], qd["repeat"], qd["hit"]

    def one(trial):
        row = best.get((model, ext, trial))
        if row is None or row["status"] != "complete":
            return None
        return hit(row)

    vals = [one(ai * repeat + r) for r in range(repeat)]
    done = [v for v in vals if v is not None]
    if not done:
        return None
    return sum(1 for v in done if v), len(done)


# ---------------------------------------------------------------- cells/render


def _cls(cell):
    if cell is None:
        return "nd"
    n, done = cell
    if n == 0:
        return "miss"
    return "hit" if n == done else "part"


def _txt(cell):
    return "&mdash;" if cell is None else f"{cell[0]}/{cell[1]}"


def _anyhit(cell):
    return None if cell is None else cell[0] > 0


def matrix_html(data, base):
    out = ["<table><tr><th class=l rowspan=2>case</th><th rowspan=2>baseline</th>"]
    for arm in ARMS:
        out.append(f"<th class='armgrp qsep' colspan={len(QUANTS)}>{arm}</th>")
    out.append("</tr><tr>")
    for _arm in ARMS:
        for qi, (qlabel, _db, _suf) in enumerate(QUANTS):
            sep = " qsep" if qi == 0 else ""
            out.append(f"<th class='quant{sep}'>{qlabel}</th>")
    out.append("</tr>")

    for ext in ORDER:
        cwe, lang, role = CASE_META[ext]
        tagcls = "tag hard" if ext in HARD else "tag"
        rowcls = " class=hardrow" if ext in HARD else ""
        out.append(
            f"<tr{rowcls}><td class=l>{ext}<br><small>{cwe} &middot; {lang}</small>"
            f"<span class='{tagcls}'>{role}</span></td>"
            f"<td>{BASELINE[ext]}</td>"
        )
        for ai, _arm in enumerate(ARMS):
            cells = []
            for qlabel, _db, suf in QUANTS:
                cells.append(arm_cell(data[qlabel], comp_name(base, suf), ext, ai))
            present = {_anyhit(c) for c in cells if c is not None}
            disagree = len(present) > 1
            for qi, c in enumerate(cells):
                sep = " qsep" if qi == 0 else ""
                d = " diff" if (disagree and c is not None) else ""
                out.append(f"<td class='{_cls(c)}{sep}{d}'>{_txt(c)}</td>")
        out.append("</tr>")
    out.append("</table>")
    return "".join(out)


def summary_html(data):
    """Per (family, quant, arm): solved-cases and hard-miss-cases union counts."""
    out = [
        "<table><tr><th class=l>model</th><th>quant</th><th class=l>arm</th>"
        "<th>solved-cases</th><th>hard-miss-cases</th></tr>"
    ]
    for label, base in FAMILIES:
        for qlabel, _db, suf in QUANTS:
            qd = data[qlabel]
            model = comp_name(base, suf)
            for ai, arm in enumerate(ARMS):
                sh = sc = mh = mc = 0
                for ext in ORDER:
                    cell = arm_cell(qd, model, ext, ai)
                    if cell is None:
                        continue
                    anyhit = cell[0] > 0
                    if ext in HARD:
                        mc += 1
                        mh += int(anyhit)
                    else:
                        sc += 1
                        sh += int(anyhit)
                mcls = "miss" if mh == 0 else "hit"
                scls = "hit" if sh == sc else "part"
                sep = " qsep" if ai == 0 else ""
                out.append(
                    f"<tr><td class=l>{label}</td><td class='{sep.strip()}'>"
                    f"<b>{qlabel}</b></td><td class=l>{arm}</td>"
                    f"<td class={scls}>{sh}/{sc}</td>"
                    f"<td class={mcls}>{mh}/{mc}</td></tr>"
                )
    out.append("</table>")
    return "".join(out)


def _delta_cell(val, ref):
    """A <td> showing val with its % delta vs ref, colored by direction."""
    if not ref:
        return f"<td>{val:.0f}</td>"
    pct = (val - ref) / ref * 100
    cls = "up" if pct > 1 else "down" if pct < -1 else ""
    sign = f"{pct:+.1f}%"
    return f"<td class={cls or 'l'} style='text-align:center'>{val:.0f} <small>({sign})</small></td>"


def tokens_html(tok):
    """Mean generation/context tokens and wall-clock per run, Q8 vs Q6 vs Q4,
    with deltas against the reference quant — the cost-to-answer view."""
    out = [
        "<table><tr><th class=l>model</th><th>quant</th><th>completed runs</th>"
        "<th>gen tokens/run</th><th>context tokens/run</th>"
        "<th>wall-clock/run</th></tr>"
    ]
    for label, base in FAMILIES:
        ref = tok[REF].get(comp_name(base, REF_SUFFIX))  # full-precision baseline
        for qlabel, _db, suf in QUANTS:
            s = tok[qlabel].get(comp_name(base, suf))
            if s is None:
                out.append(
                    f"<tr><td class=l>{label}</td><td><b>{qlabel}</b></td>"
                    f"<td class=nd>&mdash;</td><td class=nd>&mdash;</td>"
                    f"<td class=nd>&mdash;</td><td class=nd>&mdash;</td></tr>"
                )
                continue
            n, to, ti, w = s
            ro, ri, rw = (ref[1], ref[2], ref[3]) if ref else (0, 0, 0)
            wcell = (
                f"<td>{w:.0f}s</td>"
                if not rw
                else (
                    lambda p: (
                        f"<td class={'up' if p > 1 else 'down' if p < -1 else 'l'}"
                        f" style='text-align:center'>{w:.0f}s <small>({p:+.1f}%)</small></td>"
                    )
                )((w - rw) / rw * 100)
            )
            out.append(
                f"<tr><td class=l>{label}</td><td><b>{qlabel}</b></td>"
                f"<td>{n}</td>{_delta_cell(to, ro)}{_delta_cell(ti, ri)}{wcell}</tr>"
            )
    out.append("</table>")
    return "".join(out)


# Map the persisted judge_fp_verdict labels to display buckets.
_VLBL = {"REAL": "real", "FALSE-POS": "fp", "UNDETERMINED": "review", "ERROR": "review"}


def precision_data():
    """Read the code-grounded FP-judge verdicts (judge_fp_verdict) off every DB.

    Only off-target findings carry a verdict (the on-target CVE hits were never
    judged). Returns (per_case, per_tier): per_case tallies DISTINCT off-target
    sites (case,file,line,cwe) deduped GLOBALLY across tiers — the verdict is a
    property of the code, not the quant — so it answers "how many real secondary
    bugs live in this file vs hallucinations." per_tier tallies off-target ROWS
    per tier — does a smaller quant report proportionally more hallucinations?"""
    per_case, per_tier, seen = {}, {}, set()
    for qlabel, db, _suf in QUANTS:
        try:
            conn = sqlite3.connect(db)
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """SELECT ca.ext_id e, f.file file, f.line_start ls, f.cwe cwe,
                   f.judge_fp_verdict v
                   FROM run_findings f JOIN runs r ON r.id=f.run_id
                   JOIN cases ca ON ca.id=r.case_id
                   WHERE r.status='complete' AND f.judge_fp_verdict IS NOT NULL"""
            ).fetchall()
            conn.close()
        except sqlite3.OperationalError:
            rows = []
        for r in rows:
            lbl = _VLBL.get(r["v"], "review")
            per_tier.setdefault(qlabel, {"real": 0, "fp": 0, "review": 0})[lbl] += 1
            key = (r["e"], r["file"], r["ls"], r["cwe"])
            if key in seen:
                continue
            seen.add(key)
            per_case.setdefault(r["e"], {"real": 0, "fp": 0, "review": 0})[lbl] += 1
    return per_case, per_tier


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


def precision_tier_html(per_tier):
    """Per-tier: off-target ROW counts by verdict — hallucination propensity."""
    out = [
        "<table><tr><th>tier</th><th>off-target rows</th><th>real</th>"
        "<th>false positive</th><th>undetermined</th><th>% real</th><th>mix</th></tr>"
    ]
    for qlabel, _db, _suf in QUANTS:
        v = per_tier.get(qlabel)
        if not v:
            out.append(f"<tr><td><b>{qlabel}</b></td><td class=nd>&mdash;</td></tr>")
            continue
        real, fp, rev = v["real"], v["fp"], v["review"]
        tot = real + fp + rev
        pct = f"{real / tot * 100:.0f}%" if tot else "&mdash;"
        out.append(
            f"<tr><td><b>{qlabel}</b></td><td>{tot}</td>"
            f"<td class=hit>{real}</td><td class=miss>{fp}</td><td>{rev}</td>"
            f"<td>{pct}</td><td>{_verdict_bar(real, fp, rev)}</td></tr>"
        )
    out.append("</table>")
    return "".join(out)


# ---------------------------------------------------------------- verdict


def verdict_facts(data):
    """Aggregate, per quant across both families, the union detection on hard vs
    solved cases — so the headline reports the numbers instead of asserting."""
    suffix_of = {q: s for q, _d, s in QUANTS}
    facts = {}
    for qlabel, _db, _suf in QUANTS:
        qd = data[qlabel]
        hard_hit = hard_tot = solv_hit = solv_tot = ncells = 0
        for _label, base in FAMILIES:
            model = comp_name(base, suffix_of[qlabel])
            for ext in ORDER:
                # union over arms: detected in any arm/trial
                got = anyrun = False
                for ai in range(len(ARMS)):
                    cell = arm_cell(qd, model, ext, ai)
                    if cell is not None:
                        anyrun = True
                        ncells += 1
                        if cell[0] > 0:
                            got = True
                if not anyrun:
                    continue
                if ext in HARD:
                    hard_tot += 1
                    hard_hit += int(got)
                else:
                    solv_tot += 1
                    solv_hit += int(got)
        facts[qlabel] = (hard_hit, hard_tot, solv_hit, solv_tot, ncells)
    return facts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="promptlab-compare-report.html")
    ap.add_argument("--repeat", type=int, default=2)
    args = ap.parse_args()

    data = quant_data(args.repeat)
    tok = token_stats(args.repeat)
    pc, pt = precision_data()
    # On-target (planted-CVE) findings, not FP-judged — for the caveat count.
    onN = 0
    for _q, db, _s in QUANTS:
        try:
            conn = sqlite3.connect(db)
            onN += conn.execute(
                """SELECT count(*) FROM run_findings f JOIN runs r ON r.id=f.run_id
                   WHERE r.status='complete' AND f.judge_fp_verdict IS NULL"""
            ).fetchone()[0]
            conn.close()
        except sqlite3.OperationalError:
            pass
    rep = data[REF]["repeat"]

    matrices = "".join(
        f"<h3>{label}</h3>{matrix_html(data, base)}" for label, base in FAMILIES
    )

    # Data-derived verdict line for the hard cases (the whole question).
    facts = verdict_facts(data)
    hard_line = " &middot; ".join(
        f"<b>{q}</b> {facts[q][0]}/{facts[q][1]}" for q, _d, _s in QUANTS if facts[q][1]
    )
    # cells completed per quant, for the coverage note
    completed = {q: facts[q][4] for q, _d, _s in QUANTS}
    quant_labels = ", ".join(q for q, _d, _s in QUANTS)

    doc = f"""<!doctype html><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Nelson — Qwen 3.6 precision sweep (BF16/8/6/4-bit)</title>{THEME_HEAD}
<style>{CSS}{EXTRA_CSS}</style>
{THEME_TOGGLE}
<h1>Qwen 3.6 precision sweep: full-precision BF16 vs 8-bit vs 6-bit vs 4-bit</h1>
<p class=sub>How far can the self-hosted Qwen 3.6 models be quantized before
vulnerability detection — or the tokens spent reaching it — degrades? The same
open-prompt probe across four precision tiers (full-precision <b>BF16</b> /
<b>UD-Q8_K_XL</b> / <b>UD-Q6_K_XL</b> / <b>UD-Q4_K_XL</b>): three non-leaking arms
(<b>open</b> / <b>plan-first</b> / <b>CWE-checklist</b>) &times; two models, {rep}
trials/arm at temperature 0.5, over six single-file cases spanning an 18&rarr;0
baseline and five languages. Detection = a finding localized within 10 lines of
the planted ground-truth hunk.</p>

<div class=callout>
<div class=big>Hard-miss detection by quant: {hard_line}</div>
The capability ceiling is the headline — if every tier posts the same hard-case
count, the bottleneck is model capability, not precision. The <b>token-usage</b>
section below answers the second question: does a smaller quant spend <i>more</i>
tokens to reach the same answer (a false economy for anyone who can run either)?
Deltas there are measured against <b>{REF}</b>.
</div>

<h2>Detection matrix &mdash; {quant_labels} side by side</h2>
<p class=legend>Cell = localized hits / completed trials, one column per quant.
<span class=hit>green</span> = both trials hit, <span class=part>amber</span> =
split, <span class=miss>red</span> = neither, <span class=nd>&mdash;</span> = no
data. A <b>Δ</b> marks an arm where the quants <i>disagree</i> on whether the bug
was found. Hard-miss cases (baseline ≤1) bold.</p>
{matrices}

<h2>Token usage &mdash; cost to answer per quant</h2>
<p>Mean per completed run. <b>Gen tokens</b> = what the model actually generates
(the real compute-to-answer); <b>context tokens</b> = input re-read across the
ReAct loop (proxy for tool-call turns); <b>wall-clock</b> = latency (also moved by
host load). Percentages are vs <b>{REF}</b>; <span class=warn>+</span> means the
smaller quant works harder. A smaller quant that needs materially more gen tokens
for the same detection is a false economy where both quants are runnable.</p>
{tokens_html(tok)}

<h2>Arm summary (union over {rep} trials)</h2>
<p>How many solved/medium cases and hard-miss cases each (model, quant, arm)
detected in at least one trial. Denominators count only cases with a completed
run.</p>
{summary_html(data)}

<h2>Off-target findings: real secondary bugs vs false positives</h2>
<p>These are real in-the-wild OSS files, not synthetic single-bug fixtures, so a
finding that <i>isn't</i> the planted CVE is not automatically a false positive —
it may be a genuine, never-fixed secondary bug. We judged every off-target finding
with the code-grounded Opus FP-judge (it reads the <b>pre-patch source</b>, never
the advisory) and classified each as a real bug, a false positive, or
undetermined. Distinct sites are deduped <b>globally</b> across all four tiers
(the verdict is a property of the code, not the precision), so this counts how
many real secondary bugs actually live in each file.</p>
{precision_case_html(pc)}
<p class=legend><span class=hit>green</span> = judge-confirmed real bug,
<span class=miss>red</span> = false positive, grey = undetermined. Hard-miss cases
bold. The on-target CVE hits ({onN} findings) are not shown here — those are the
planted bug, already known real.</p>

<h3>Does a smaller quant hallucinate more? Off-target rows by tier</h3>
<p>Per-tier off-target finding volume split by verdict (row-level, so repeated
reports of the same bug count each time). If the smaller quants don't post a
materially lower <b>% real</b>, precision is precision-invariant too — the smaller
build is no noisier, consistent with the detection result.</p>
{precision_tier_html(pt)}

<h2>Method &amp; caveats</h2>
<ul>
<li>Models: <code>qwen3.6-27b</code> (dense, 10.20.30.1) and
<code>qwen3.6-35b-A3b</code> (MoE, 10.20.30.2), self-hosted llama-server, free.
Identical endpoints &mdash; only the loaded precision differs (BF16 / UD-Q8/Q6/Q4_K_XL).</li>
<li>Per tier: {len(ORDER)} cases &times; {len(ARMS)} arms &times; {rep} trials
&times; 2 models = {len(ORDER) * len(ARMS) * rep * 2} cells. Completed runs found: {", ".join(f"{q} {completed[q]}" for q, _d, _s in QUANTS)}.</li>
<li>Temperature 0.5 so repeats explore; llama-server's default <code>seed=-1</code>
randomizes per request. Detection is localization-only (free); on the hard cases
zero findings landed near the planted bug, so there is nothing for a truth-judge
to confirm.</li>
<li>Token counts are llama-server's reported usage. Context tokens are large
because the ReAct loop re-sends the transcript each turn (cache-blind accounting);
the cross-quant <i>comparison</i> is still apples-to-apples since the harness is
identical.</li>
<li>Separate DBs per quant ({", ".join(f"<code>{html.escape(d)}</code>" for _q, d, _s in QUANTS)});
the baseline benchmark DB is untouched.</li>
</ul>
<p><small>Generated from {", ".join(html.escape(d) for _q, d, _s in QUANTS)} ·
{len(QUANTS)} quants &times; {len(FAMILIES)} models &times; {len(ORDER)} cases
&times; {len(ARMS)} arms &times; {rep} trials.</small></p>
"""
    with open(args.out, "w") as f:
        f.write(doc)
    print(f"wrote {args.out} ({len(doc)} bytes)")


if __name__ == "__main__":
    main()
