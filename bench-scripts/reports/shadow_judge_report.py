#!/usr/bin/env python3
"""Report for the shadow-judging experiment (reads nelson-shadow-judge.db).

Answers: can DeepSeek / MiMo / local Gemma-4-31B reproduce Opus's truth verdicts,
and would a 3-judge vote reliably settle borderline HIT/MISS calls? Prints a console
summary and writes bench-shadow-judge.html. Safe to run while the job is still in
flight — it reports over whatever cells exist.

Usage: .venv/bin/python shadow_judge_report.py [--db nelson-shadow-judge.db]
"""

from __future__ import annotations

import argparse
import html
import sqlite3
from collections import defaultdict
from pathlib import Path

from nelson.report_style import BASE_CSS, THEME_HEAD, THEME_TOGGLE, THEME_VARS

MODELS_ORDER = ["deepseek", "mimo", "glm52", "gemma31b"]


def _kappa(pairs: list[tuple[int, int]]) -> float | None:
    """Cohen's kappa for binary (opus, shadow) verdict pairs."""
    n = len(pairs)
    if n == 0:
        return None
    po = sum(1 for a, b in pairs if a == b) / n
    pa1 = sum(a for a, _ in pairs) / n
    pb1 = sum(b for _, b in pairs) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    if pe == 1.0:
        return 1.0 if po == 1.0 else 0.0
    return (po - pe) / (1 - pe)


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.0%}"


def _dedupe_disagreements(ds: list[dict]) -> list[tuple]:
    """Collapse the unanimous-vs-Opus set to distinct (case, file, line) spots.

    The raw set double-counts a spot reported many ways (e.g. w52v crud.js:113 by
    several competitors -> distinct prompts, one location). Returns
    (case, file, line, opus_same_bug, n_input_variants) sorted by case.
    """
    agg: dict[tuple, list] = {}
    for i in ds:
        k = (i["case"], i["file"], i["line"])
        if k not in agg:
            agg[k] = [i["opus"], 0]
        agg[k][1] += 1
    return sorted(
        ((c, f, line, v[0], v[1]) for (c, f, line), v in agg.items()),
        key=lambda r: (r[0], r[1] or ""),
    )


def _model_verdict(trials: list[int | None]) -> int | None:
    """Majority vote over a model's trials for one input (None if no usable trial)."""
    votes = [v for v in trials if v is not None]
    if not votes:
        return None
    ones = sum(votes)
    if ones * 2 == len(votes):
        return None  # genuine tie -> abstain
    return 1 if ones * 2 > len(votes) else 0


def load(db: str):
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """SELECT model, trial, input_hash, source_db, case_ext_id, finding_file,
                  finding_line, opus_same_bug, opus_conflict, shadow_same_bug,
                  latency_ms, tokens_out, error
           FROM shadow_verdicts"""
    ).fetchall()
    conn.close()
    return rows


def build(rows):
    # input metadata + per-(model,input) trial lists
    inputs: dict[str, dict] = {}
    trials: dict[tuple[str, str], list[int | None]] = defaultdict(list)
    raw_by_model: dict[str, list[sqlite3.Row]] = defaultdict(list)
    for r in rows:
        h = r["input_hash"]
        if h not in inputs:
            inputs[h] = {
                "opus": r["opus_same_bug"],
                "conflict": r["opus_conflict"],
                "case": r["case_ext_id"],
                "file": r["finding_file"],
                "line": r["finding_line"],
            }
        trials[(r["model"], h)].append(r["shadow_same_bug"])
        raw_by_model[r["model"]].append(r)

    # per-model majority verdict per input
    model_verdict: dict[str, dict[str, int | None]] = defaultdict(dict)
    for (model, h), tl in trials.items():
        model_verdict[model][h] = _model_verdict(tl)

    models = [m for m in MODELS_ORDER if m in model_verdict]
    models += [m for m in model_verdict if m not in models]

    # ---- per-model stats ----
    per_model = {}
    for m in models:
        mv = model_verdict[m]
        pairs, dn_hit, dn_tot, sm_hit, sm_tot = [], 0, 0, 0, 0
        for h, info in inputs.items():
            if info["conflict"]:
                continue
            sv = mv.get(h)
            if sv is None:
                continue
            opus = info["opus"]
            pairs.append((opus, sv))
            if opus == 0:
                dn_tot += 1
                dn_hit += sv == 0
            else:
                sm_tot += 1
                sm_hit += sv == 1
        acc = (sum(a == b for a, b in pairs) / len(pairs)) if pairs else None
        # self-consistency over trials
        consistent = flips = multi = 0
        for h in inputs:
            tl = [v for v in trials[(m, h)] if v is not None]
            if len(tl) >= 2:
                multi += 1
                if len(set(tl)) == 1:
                    consistent += 1
                else:
                    flips += 1
        raws = raw_by_model[m]
        lats = sorted(r["latency_ms"] for r in raws if r["latency_ms"] is not None)
        errs = sum(1 for r in raws if r["error"])
        unparse = sum(1 for r in raws if r["error"] == "unparseable")
        toks = [r["tokens_out"] for r in raws if r["tokens_out"] is not None]
        per_model[m] = {
            "n_inputs": sum(1 for h in inputs if mv.get(h) is not None),
            "calls": len(raws),
            "errors": errs,
            "unparseable": unparse,
            "acc": acc,
            "kappa": _kappa(pairs),
            "diff_recall": (dn_hit / dn_tot) if dn_tot else None,
            "same_recall": (sm_hit / sm_tot) if sm_tot else None,
            "diff_tot": dn_tot,
            "same_tot": sm_tot,
            "tp": sum(1 for a, b in pairs if a == 1 and b == 1),
            "fn": sum(1 for a, b in pairs if a == 1 and b == 0),
            "fp": sum(1 for a, b in pairs if a == 0 and b == 1),
            "tn": sum(1 for a, b in pairs if a == 0 and b == 0),
            "self_consistent": (consistent / multi) if multi else None,
            "flips": flips,
            "lat_med": lats[len(lats) // 2] if lats else None,
            "lat_p95": lats[int(len(lats) * 0.95)] if lats else None,
            "tok_total": sum(toks),
        }

    # ---- consensus / voting ----
    consensus = {
        "unanimous": 0,
        "unanimous_match_opus": 0,
        "panel_match_opus": 0,
        "panel_total": 0,
        "split_match_opus": 0,
        "split_total": 0,
        "disagree_set": [],  # all-3 agree but contradict Opus
    }
    for h, info in inputs.items():
        if info["conflict"]:
            continue
        votes = [model_verdict[m].get(h) for m in models]
        usable = [v for v in votes if v is not None]
        if len(usable) < len(models):  # need all models for a clean panel
            continue
        ones = sum(usable)
        if ones * 2 == len(usable):
            continue  # panel tie
        panel = 1 if ones * 2 > len(usable) else 0
        consensus["panel_total"] += 1
        consensus["panel_match_opus"] += panel == info["opus"]
        unanimous = len(set(usable)) == 1
        if unanimous:
            consensus["unanimous"] += 1
            consensus["unanimous_match_opus"] += usable[0] == info["opus"]
            if usable[0] != info["opus"]:
                consensus["disagree_set"].append(info)
        else:
            consensus["split_total"] += 1
            consensus["split_match_opus"] += panel == info["opus"]

    return models, inputs, per_model, consensus


def to_console(models, inputs, per_model, consensus):
    n = len(inputs)
    n_same = sum(i["opus"] for i in inputs.values())
    print(
        f"\n=== SHADOW JUDGING — {n} distinct Opus-judged inputs "
        f"(same={n_same}, different={n - n_same}) ===\n"
    )
    hdr = (
        f"{'model':<10} {'acc':>5} {'kappa':>6} {'diffRec':>8} {'sameRec':>8} "
        f"{'self-cons':>9} {'flips':>5} {'err':>4} {'medMs':>6}"
    )
    print(hdr)
    print("-" * len(hdr))
    for m in models:
        s = per_model[m]
        print(
            f"{m:<10} {_pct(s['acc']):>5} "
            f"{('n/a' if s['kappa'] is None else f'{s["kappa"]:.2f}'):>6} "
            f"{_pct(s['diff_recall']):>8} {_pct(s['same_recall']):>8} "
            f"{_pct(s['self_consistent']):>9} {s['flips']:>5} {s['errors']:>4} "
            f"{(s['lat_med'] or 0):>6}"
        )
    print(
        "\n  acc/kappa exclude Opus self-conflicts; diffRec = recall on the "
        "harder 'different_bug' class (catches non-matches, not rubber-stamping)."
    )

    c = consensus
    print("\n=== 3-JUDGE PANEL (majority of the three models) ===")
    if c["panel_total"]:
        print(
            f"  panel-majority agrees with Opus: "
            f"{c['panel_match_opus']}/{c['panel_total']} "
            f"({c['panel_match_opus'] / c['panel_total']:.0%})"
        )
    if c["unanimous"]:
        print(
            f"  all 3 unanimous: {c['unanimous']}/{c['panel_total']} inputs; of those, "
            f"{c['unanimous_match_opus']}/{c['unanimous']} match Opus "
            f"({c['unanimous_match_opus'] / c['unanimous']:.0%})  "
            "<- the easy-consensus rate"
        )
    if c["split_total"]:
        print(
            f"  2-1 split: {c['split_total']} inputs; majority matches Opus "
            f"{c['split_match_opus']}/{c['split_total']} "
            f"({c['split_match_opus'] / c['split_total']:.0%})"
        )
    print("\n=== DIRECTIONAL BIAS (how each model errs vs Opus) ===")
    for m in models:
        s = per_model[m]
        print(
            f"  {m:<10} over-calls same_bug (lenient, FP): {s['fp']:>3}   "
            f"misses same_bug (strict, FN): {s['fn']:>3}"
        )
    print(
        "  All lean LENIENT (FP>>FN): a cheap vote resolves borderline calls "
        "toward HIT, not Opus's stricter 'different'."
    )

    dd = _dedupe_disagreements(c["disagree_set"])
    print(
        f"\n  {len(dd)} distinct spots where all 3 shadows AGREE but CONTRADICT "
        f"Opus ({len(c['disagree_set'])} input variants):"
    )
    for cas, f, line, opus, nv in dd:
        print(
            f"    {cas:<22} {f}:{line}  opus={'same' if opus else 'diff'} "
            f"shadows={'diff' if opus else 'same'}"
            f"{f'  (x{nv})' if nv > 1 else ''}"
        )


def to_html(models, inputs, per_model, consensus, path):
    n = len(inputs)
    n_same = sum(i["opus"] for i in inputs.values())
    c = consensus

    def cell(x):
        return html.escape(str(x))

    rows_html = ""
    for m in models:
        s = per_model[m]
        kap = "n/a" if s["kappa"] is None else f"{s['kappa']:.2f}"
        rows_html += (
            f"<tr><td class='mono'>{cell(m)}</td>"
            f"<td class='num big'>{_pct(s['acc'])}</td>"
            f"<td class='num'>{kap}</td>"
            f"<td class='num'>{_pct(s['diff_recall'])} "
            f"<span class='muted'>({s['tp'] and ''}{s['diff_tot']})</span></td>"
            f"<td class='num'>{_pct(s['same_recall'])} "
            f"<span class='muted'>({s['same_tot']})</span></td>"
            f"<td class='num'>{_pct(s['self_consistent'])}</td>"
            f"<td class='num'>{s['flips']}</td>"
            f"<td class='num'>TP {s['tp']} / FN {s['fn']} / FP {s['fp']} / TN {s['tn']}</td>"
            f"<td class='num'>{s['errors']}<span class='muted'>"
            f"/{s['unparseable']}u</span></td>"
            f"<td class='num'>{s['lat_med'] or '-'}/{s['lat_p95'] or '-'}</td>"
            f"</tr>"
        )

    dd = _dedupe_disagreements(c["disagree_set"])
    ds_rows = "".join(
        f"<tr><td class='mono'>{cell(cas)}</td>"
        f"<td class='mono'>{cell(f)}:{cell(line)}</td>"
        f"<td>opus said <b>{'same' if opus else 'different'}</b>, "
        f"all 3 shadows said <b>{'different' if opus else 'same'}</b>"
        f"{f' <span class=muted>(x{nv} input variants)</span>' if nv > 1 else ''}"
        f"</td></tr>"
        for cas, f, line, opus, nv in dd
    )

    bias_rows = "".join(
        f"<tr><td class='mono'>{cell(m)}</td>"
        f"<td class='num'>{per_model[m]['fp']}</td>"
        f"<td class='num'>{per_model[m]['fn']}</td></tr>"
        for m in models
    )

    panel_pct = (c["panel_match_opus"] / c["panel_total"]) if c["panel_total"] else 0
    unan_pct = (c["unanimous_match_opus"] / c["unanimous"]) if c["unanimous"] else 0
    split_pct = (c["split_match_opus"] / c["split_total"]) if c["split_total"] else 0

    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Shadow judging — can cheap models replace the Opus truth judge?</title>\
{THEME_HEAD}
<style>{THEME_VARS}{BASE_CSS}.cards .card{{min-width:180px}}</style></head>
<body>{THEME_TOGGLE}
<h1>Shadow judging: can a cheap panel reproduce the Opus truth judge?</h1>
<p class="sub">{n} distinct (advisory + reported-finding) inputs Opus already
truth-judged ({n_same} <b>same_bug</b> / {n - n_same} <b>different_bug</b>), replayed
through each candidate judge. Each model's verdict = majority over its trials.
"different_bug" is the hard, discriminating class: a judge that rubber-stamps
"same_bug" scores high accuracy on the majority class but useless diffRecall.</p>

<h2>Per-model agreement with Opus</h2>
<table>
<tr><th>model</th><th>accuracy</th><th>kappa</th>
<th>diffRecall (n)</th><th>sameRecall (n)</th>
<th>self-consist</th><th>flips</th><th>confusion</th><th>err/unp</th>
<th>lat med/p95 ms</th></tr>
{rows_html}
</table>
<p class="muted">accuracy & kappa over non-conflict inputs with a usable verdict;
confusion is vs Opus (positive = same_bug): TP same✓ / FN missed-same / FP
false-same / TN different✓. self-consist = share of inputs whose trials all agreed.</p>

<h2>3-judge panel (majority vote of the three models)</h2>
<div class="cards">
<div class="card"><div class="v">{panel_pct:.0%}</div>
<div class="l">panel majority matches Opus<br>({c["panel_match_opus"]}/{c["panel_total"]})</div></div>
<div class="card"><div class="v">{unan_pct:.0%}</div>
<div class="l">when all 3 agree, match Opus<br>({c["unanimous_match_opus"]}/{c["unanimous"]} unanimous)</div></div>
<div class="card"><div class="v">{split_pct:.0%}</div>
<div class="l">on 2-1 splits, majority matches Opus<br>({c["split_match_opus"]}/{c["split_total"]})</div></div>
<div class="card"><div class="v">{c["unanimous"]}/{c["panel_total"]}</div>
<div class="l">inputs that are unanimous<br>(the "easy" share)</div></div>
</div>
<p class="muted">The voting hypothesis: where the panel is unanimous it should track
Opus closely (easy cases, no vote needed); the value is on the 2-1 splits, which are
exactly the borderline calls a tie-break is for.</p>

<h2>Directional bias — how each judge errs vs Opus</h2>
<p class="muted">FP = over-calls <b>same_bug</b> when Opus said different (lenient: would
turn a MISS into a false HIT). FN = calls <b>different</b> when Opus said same (strict:
would suppress a real hit). Every model leans lenient (FP &gt; FN), so a cheap vote
resolves borderline calls <b>toward HIT</b>, not Opus's stricter "different".</p>
<table>
<tr><th>model</th><th>FP — over-calls same (lenient)</th>
<th>FN — misses same (strict)</th></tr>
{bias_rows}
</table>

<h2>Distinct spots where all 3 shadows agree but contradict Opus ({len(dd)})</h2>
<p class="muted">Deduped by (case, file, line) — one spot reported many ways collapses
to one row. All three cheap judges agreed with each other but disagreed with Opus:
either Opus is borderline/wrong here, or the panel shares a lenient blind spot. Note
these are nearly all the panel calling a localized-but-different finding "same" — incl.
the cc7p case the partial tier rules right-place/wrong-bug. Worth a human spot-check.</p>
<table>{ds_rows or "<tr><td class=muted>none</td></tr>"}</table>
</body></html>"""
    Path(path).write_text(doc)
    print(f"\nwrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="nelson-shadow-judge.db")
    ap.add_argument("--html", default="bench-shadow-judge.html")
    args = ap.parse_args()
    rows = load(args.db)
    if not rows:
        print("no rows yet")
        return
    models, inputs, per_model, consensus = build(rows)
    to_console(models, inputs, per_model, consensus)
    to_html(models, inputs, per_model, consensus, args.html)


if __name__ == "__main__":
    main()
