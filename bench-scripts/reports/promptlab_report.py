#!/usr/bin/env python3
"""Self-contained HTML report for the open-prompt strategy probe.

Renders the per-(model, case, arm) localized-detection matrix, the arm summary
(solved vs hard-miss), and the off-target finding-volume table (the FP angle),
with the headline: did plan-first or the CWE-checklist lift hard-bug detection
over the plain open prompt? Reuses analyze_promptlab.load for the data layer.

    .venv/bin/python promptlab_report.py [--db nelson-promptlab.db] [--out promptlab-report.html]
"""

import argparse
import html
import sqlite3

from analyze_promptlab import ARMS, BASELINE, ORDER, load

from nelson.report_style import BASE_CSS, THEME_HEAD, THEME_TOGGLE, THEME_VARS

# (ext_id -> (CWE, language, role)) for the matrix labels.
CASE_META = {
    "GHSA-w52v-v783-gw97": ("CWE-89", "JS", "hit anchor"),
    "GHSA-j273-m5qq-6825": ("CWE-22", "Java", "guard"),
    "GHSA-f26g-jm89-4g65": ("CWE-77", "Rust", "medium"),
    "GHSA-x9h5-r9v2-vcww": ("CWE-122", "C", "hard miss"),
    "GHSA-9f49-8x56-jmjc": ("CWE-416", "C", "hard miss"),
    "CVE-2026-5199": ("CWE-639", "Go", "hardest miss"),
}
HARD = {e for e in ORDER if BASELINE[e].split("/")[0] in ("0", "1")}

# Shared themeable stylesheet (light/dark) + this report's one bespoke row style.
# `CSS` is re-exported and consumed by promptlab_compare_report.py too.
CSS = THEME_VARS + BASE_CSS + "\n.hardrow td.l{font-weight:600}\n"


def short(m):
    return m.rsplit("/", 1)[-1]


def finding_counts(db_path, repeat):
    """findings-per-run on the HARD cases per (model, arm) -> (total, trials)."""
    db = sqlite3.connect(db_path)
    db.row_factory = sqlite3.Row
    rows = db.execute(
        """SELECT cp.name m, ca.ext_id e, r.trial t,
           (SELECT count(*) FROM run_findings f WHERE f.run_id=r.id) nf
           FROM runs r JOIN competitors cp ON cp.id=r.competitor_id
           JOIN cases ca ON ca.id=r.case_id WHERE r.status='complete'"""
    ).fetchall()
    agg = {}
    for r in rows:
        if r["e"] not in HARD:
            continue
        key = (short(r["m"]), ARMS[r["t"] // repeat])
        a = agg.setdefault(key, [0, 0])
        a[0] += r["nf"]
        a[1] += 1
    return agg


def matrix_html(best, repeat, hit, model):
    def cell(ext, trial):
        row = best.get((model, ext, trial))
        if row is None or row["status"] != "complete":
            return None
        return hit(row)

    out = ["<table><tr><th class=l>case</th><th>baseline</th>"]
    out += [f"<th>{a}</th>" for a in ARMS]
    out.append("</tr>")
    for ext in ORDER:
        cwe, lang, role = CASE_META[ext]
        tagcls = "tag hard" if ext in HARD else "tag"
        rowcls = " class=hardrow" if ext in HARD else ""
        out.append(
            f"<tr{rowcls}><td class=l>{ext}<br><small>{cwe} · {lang}</small>"
            f"<span class='{tagcls}'>{role}</span></td>"
            f"<td>{BASELINE[ext]}</td>"
        )
        for ai, _arm in enumerate(ARMS):
            vals = [cell(ext, ai * repeat + r) for r in range(repeat)]
            done = [v for v in vals if v is not None]
            if not done:
                out.append("<td class=nd>—</td>")
                continue
            n = sum(1 for v in done if v)
            if n == 0:
                cls = "miss"
            elif n == len(done):
                cls = "hit"
            else:
                cls = "part"
            out.append(f"<td class={cls}>{n}/{len(done)}</td>")
        out.append("</tr>")
    out.append("</table>")
    return "".join(out)


def summary_html(best, repeat, hit, models):
    def cell(model, ext, trial):
        row = best.get((model, ext, trial))
        if row is None or row["status"] != "complete":
            return None
        return hit(row)

    out = [
        "<table><tr><th class=l>model</th><th class=l>arm</th>"
        "<th>solved-cases</th><th>hard-miss-cases</th></tr>"
    ]
    for model in models:
        for ai, arm in enumerate(ARMS):
            sh = sc = mh = mc = 0
            for ext in ORDER:
                vals = [cell(model, ext, ai * repeat + r) for r in range(repeat)]
                if all(v is None for v in vals):
                    continue
                anyhit = any(vals)
                if ext in HARD:
                    mc += 1
                    mh += int(anyhit)
                else:
                    sc += 1
                    sh += int(anyhit)
            mcls = "miss" if mh == 0 else "hit"
            out.append(
                f"<tr><td class=l>{short(model)}</td><td class=l>{arm}</td>"
                f"<td class={'hit' if sh == sc else 'part'}>{sh}/{sc}</td>"
                f"<td class={mcls}>{mh}/{mc}</td></tr>"
            )
    out.append("</table>")
    return "".join(out)


def fp_html(agg, models):
    out = [
        "<table><tr><th class=l>model</th><th class=l>arm</th>"
        "<th>off-target findings</th><th>trials</th><th>per run</th></tr>"
    ]
    for model in models:
        for arm in ARMS:
            nf, tr = agg.get((short(model), arm), [0, 0])
            per = nf / tr if tr else 0
            out.append(
                f"<tr><td class=l>{short(model)}</td><td class=l>{arm}</td>"
                f"<td>{nf}</td><td>{tr}</td><td>{per:.2f}</td></tr>"
            )
    out.append("</table>")
    return "".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="nelson-promptlab.db")
    ap.add_argument("--out", default="promptlab-report.html")
    ap.add_argument("--repeat", type=int, default=2)
    args = ap.parse_args()

    best, repeat, hit = load(args.db, args.repeat or None)
    models = sorted({m for (m, _e, _t) in best})
    agg = finding_counts(args.db, repeat)

    matrices = "".join(
        f"<h3>{short(m)}</h3>{matrix_html(best, repeat, hit, m)}" for m in models
    )

    doc = f"""<!doctype html><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Nelson — open-prompt strategy probe</title>{THEME_HEAD}<style>{CSS}</style>
{THEME_TOGGLE}
<h1>Open-prompt strategy probe</h1>
<p class=sub>Can prompting alone — without leaking the CWE — lift hard-bug
detection? Three non-leaking arms (<b>open</b> / <b>plan-first</b> /
<b>CWE-checklist</b>) &times; two local Qwen 3.6 models, {repeat} trials/arm at
temperature 0.5, over six single-file cases spanning an 18→0 baseline and five
languages. Detection = a finding localized within {10} lines of the planted
ground-truth hunk (the benchmark's gate).</p>

<div class=callout>
<div class=big>Verdict: no arm cracked a single hard-miss case.</div>
Both models, all three arms: <b>0/3</b> on the hard cases. Plan-first and the
CWE-checklist detect <b>exactly what the plain open prompt detects</b> — they help
on neither the hard misses nor (beyond trial-noise) the solved cases. Combined
with the CWE-sweep probe (many narrow prompts also failed on hard cases), the
lever for hard bugs is <b>not prompt scaffolding or breadth</b> — only the oracle
CWE leak moved cheap-model detection, and that is unavailable in real use.
</div>

<h2>Detection matrix</h2>
<p>Cell = localized hits / completed trials. <span class=hit>green</span> = both
trials hit, <span class=part>amber</span> = split, <span class=miss>red</span> =
neither, <span class=nd>—</span> = no data. Hard-miss cases (baseline ≤1) bold.</p>
{matrices}

<h2>Arm summary (union over {repeat} trials)</h2>
<p>Of the solved/medium cases and the hard-miss cases, how many each arm detected
in at least one trial. Denominators count only cases with a completed run.</p>
{summary_html(best, repeat, hit, models)}

<h2>False-positive angle: off-target finding volume on the hard cases</h2>
<p>Every finding on the three hard cases is off-target (none localized to the
planted bug). Does the checklist's breadth spike noise? <b>No</b> — the structured
arms are if anything <i>more</i> conservative than the open prompt, the opposite of
the CWE-sweep's hard-case FP blowup. The "<code>output [] if a class doesn't
apply</code>" guard holds.</p>
{fp_html(agg, models)}

<h2>Method &amp; caveats</h2>
<ul>
<li>Models: <code>qwen3.6-27b</code> (dense, 10.20.30.1) and
<code>qwen3.6-35b-A3b</code> (MoE, 10.20.30.2), self-hosted llama-server, free.</li>
<li>Temperature 0.5 (raised from the hardcoded 0.1 via the new
<code>NELSON_TEMPERATURE</code> knob) so repeats explore; llama-server's default
<code>seed=-1</code> randomizes per request.</li>
<li>Detection is localization-only (free). On the hard cases this is the whole
story — zero findings landed near the planted bug under any arm, so there is
nothing for a truth-judge to confirm; no judge spend.</li>
<li>The 27b server crashed mid-run after the gix case; the three hard cases were
re-run after a restart (resume-safe). 35b completed in one pass.</li>
</ul>
<p><small>Generated from {html.escape(args.db)} · {len(models)} models ·
{len(ORDER)} cases &times; {len(ARMS)} arms &times; {repeat} trials.</small></p>
"""
    with open(args.out, "w") as f:
        f.write(doc)
    print(f"wrote {args.out} ({len(doc)} bytes)")


if __name__ == "__main__":
    main()
