#!/usr/bin/env python3
"""Report for the shadow FP-judging experiment (reads nelson-shadow-fp-judge.db).

Answers: can DeepSeek / MiMo / local Gemma-4-31B reproduce Opus's *precision* (FP)
verdicts — "is this finding a REAL bug or a FALSE POSITIVE?" — and would a 3-judge
vote settle the borderline calls? Prints a console summary and writes
bench-shadow-fp-judge.html. Safe to run while the job is in flight.

Framing differs from the truth report on purpose. The FP judge's *job* is to filter
noise, so the discriminating skill is **fpRecall** = recall on the false_positive class
(of the findings Opus called noise, how many did the shadow also reject). A judge that
rubber-stamps everything "real" scores high realRecall but useless fpRecall — and that
LENIENT failure (shadow says real where Opus says false_positive) is exactly the
direction that does NOT reduce false positives. We report bias direction per model.

Usage: .venv/bin/python shadow_fp_judge_report.py [--db nelson-shadow-fp-judge.db]
"""

from __future__ import annotations

import argparse
import html
import sqlite3
from collections import defaultdict
from pathlib import Path

from nelson.report_style import BASE_CSS, THEME_HEAD, THEME_TOGGLE, THEME_VARS

MODELS_ORDER = ["deepseek", "mimo", "gemma31b"]


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

    One spot reported many ways (distinct prompts, one location) collapses to one row.
    Returns (case, file, line, opus_real, n_input_variants) sorted by case.
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
                  finding_line, opus_real, opus_conflict, shadow_real, shadow_label,
                  latency_ms, tokens_out, error
           FROM shadow_fp_verdicts"""
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
                "opus": r["opus_real"],
                "conflict": r["opus_conflict"],
                "case": r["case_ext_id"],
                "file": r["finding_file"],
                "line": r["finding_line"],
            }
        trials[(r["model"], h)].append(r["shadow_real"])
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
        pairs, fp_hit, fp_tot, real_hit, real_tot = [], 0, 0, 0, 0
        for h, info in inputs.items():
            if info["conflict"]:
                continue
            sv = mv.get(h)
            if sv is None:
                continue
            opus = info["opus"]
            pairs.append((opus, sv))
            if opus == 0:  # Opus called it a false positive (the noise class)
                fp_tot += 1
                fp_hit += sv == 0
            else:  # Opus called it a real bug
                real_tot += 1
                real_hit += sv == 1
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
        abstain = sum(1 for r in raws if r["shadow_label"] == "needs_review")
        toks = [r["tokens_out"] for r in raws if r["tokens_out"] is not None]
        fp_conf = sum(1 for a, b in pairs if a == 0 and b == 1)  # lenient
        fn_conf = sum(1 for a, b in pairs if a == 1 and b == 0)  # strict
        per_model[m] = {
            "n_inputs": sum(1 for h in inputs if mv.get(h) is not None),
            "calls": len(raws),
            "errors": errs,
            "unparseable": unparse,
            "abstain": abstain,
            "acc": acc,
            "kappa": _kappa(pairs),
            "fp_recall": (fp_hit / fp_tot) if fp_tot else None,
            "real_recall": (real_hit / real_tot) if real_tot else None,
            "fp_tot": fp_tot,
            "real_tot": real_tot,
            "tp": sum(1 for a, b in pairs if a == 1 and b == 1),
            "fn": fn_conf,
            "fp": fp_conf,
            "tn": sum(1 for a, b in pairs if a == 0 and b == 0),
            "lean": (
                "lenient"
                if fp_conf > fn_conf
                else "strict"
                if fn_conf > fp_conf
                else "balanced"
            ),
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
    n_real = sum(i["opus"] for i in inputs.values())
    print(
        f"\n=== SHADOW FP-JUDGING — {n} distinct Opus FP-judged inputs "
        f"(real={n_real}, false_positive={n - n_real}) ===\n"
    )
    hdr = (
        f"{'model':<10} {'acc':>5} {'kappa':>6} {'fpRec':>6} {'realRec':>8} "
        f"{'lean':>9} {'self-cons':>9} {'abst':>5} {'err':>4} {'medMs':>6}"
    )
    print(hdr)
    print("-" * len(hdr))
    for m in models:
        s = per_model[m]
        print(
            f"{m:<10} {_pct(s['acc']):>5} "
            f"{('n/a' if s['kappa'] is None else f'{s["kappa"]:.2f}'):>6} "
            f"{_pct(s['fp_recall']):>6} {_pct(s['real_recall']):>8} "
            f"{s['lean']:>9} {_pct(s['self_consistent']):>9} {s['abstain']:>5} "
            f"{s['errors']:>4} {(s['lat_med'] or 0):>6}"
        )
    print(
        "\n  fpRec = recall on the false_positive class (the FP judge's actual job: "
        "catching noise). A lenient judge has high realRec but low fpRec."
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
            f"  {m:<10} lenient (calls REAL, Opus said FP): {s['fp']:>3}   "
            f"strict (calls FP, Opus said REAL): {s['fn']:>3}   -> {s['lean']}"
        )
    print(
        "  Lenient errors wave noise THROUGH (the failure mode that does not reduce "
        "false positives); strict errors discard a real bug."
    )

    dd = _dedupe_disagreements(c["disagree_set"])
    print(
        f"\n  {len(dd)} distinct spots where all 3 shadows AGREE but CONTRADICT "
        f"Opus ({len(c['disagree_set'])} input variants):"
    )
    for cas, f, line, opus, nv in dd:
        print(
            f"    {cas:<22} {f}:{line}  opus={'real' if opus else 'fp'} "
            f"shadows={'fp' if opus else 'real'}"
            f"{f'  (x{nv})' if nv > 1 else ''}"
        )


def to_html(models, inputs, per_model, consensus, path):
    n = len(inputs)
    n_real = sum(i["opus"] for i in inputs.values())
    c = consensus

    def cell(x):
        return html.escape(str(x))

    rows_html = ""
    for m in models:
        s = per_model[m]
        kap = "n/a" if s["kappa"] is None else f"{s['kappa']:.2f}"
        lean_cls = {"lenient": "bad", "strict": "amber", "balanced": "good"}[s["lean"]]
        rows_html += (
            f"<tr><td class='mono'>{cell(m)}</td>"
            f"<td class='num big'>{_pct(s['acc'])}</td>"
            f"<td class='num'>{kap}</td>"
            f"<td class='num'>{_pct(s['fp_recall'])} "
            f"<span class='muted'>({s['fp_tot']})</span></td>"
            f"<td class='num'>{_pct(s['real_recall'])} "
            f"<span class='muted'>({s['real_tot']})</span></td>"
            f"<td class='num {lean_cls}'>{s['lean']}</td>"
            f"<td class='num'>{_pct(s['self_consistent'])}</td>"
            f"<td class='num'>TP {s['tp']} / FN {s['fn']} / FP {s['fp']} / TN {s['tn']}</td>"
            f"<td class='num'>{s['abstain']}</td>"
            f"<td class='num'>{s['errors']}<span class='muted'>"
            f"/{s['unparseable']}u</span></td>"
            f"<td class='num'>{s['lat_med'] or '-'}/{s['lat_p95'] or '-'}</td>"
            f"</tr>"
        )

    dd = _dedupe_disagreements(c["disagree_set"])
    ds_rows = "".join(
        f"<tr><td class='mono'>{cell(cas)}</td>"
        f"<td class='mono'>{cell(f)}:{cell(line)}</td>"
        f"<td>opus said <b>{'real bug' if opus else 'false positive'}</b>, "
        f"all 3 shadows said <b>{'false positive' if opus else 'real bug'}</b>"
        f"{f' <span class=muted>(x{nv} input variants)</span>' if nv > 1 else ''}"
        f"</td></tr>"
        for cas, f, line, opus, nv in dd
    )

    bias_rows = "".join(
        f"<tr><td class='mono'>{cell(m)}</td>"
        f"<td class='num'>{per_model[m]['fp']}</td>"
        f"<td class='num'>{per_model[m]['fn']}</td>"
        f"<td class='num'>{per_model[m]['lean']}</td></tr>"
        for m in models
    )

    panel_pct = (c["panel_match_opus"] / c["panel_total"]) if c["panel_total"] else 0
    unan_pct = (c["unanimous_match_opus"] / c["unanimous"]) if c["unanimous"] else 0
    split_pct = (c["split_match_opus"] / c["split_total"]) if c["split_total"] else 0

    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Shadow FP-judging — can cheap models replace the Opus precision judge?</title>\
{THEME_HEAD}
<style>{THEME_VARS}{BASE_CSS}.cards .card{{min-width:180px}}</style></head>
<body>{THEME_TOGGLE}
<h1>Shadow FP-judging: can a cheap panel reproduce the Opus precision judge?</h1>
<p class="sub">{n} distinct (finding + pre-patch source) inputs Opus already
FP-judged ({n_real} <b>real_bug</b> / {n - n_real} <b>false_positive</b>), replayed
through each candidate judge on the IDENTICAL advisory-blind prompt. Each model's
verdict = majority over its trials. The FP judge's job is to filter noise, so
<b>fpRecall</b> (recall on the false_positive class) is the discriminating metric: a
judge that rubber-stamps "real" scores high accuracy on the majority class but useless
fpRecall — and that lenient failure is the one that does <em>not</em> reduce false
positives. Compare against the truth-judge run (bench-shadow-judge.html), where all
three leaned lenient.</p>

<h2>Per-model agreement with Opus</h2>
<table>
<tr><th>model</th><th>accuracy</th><th>kappa</th>
<th>fpRecall (n)</th><th>realRecall (n)</th><th>lean</th>
<th>self-consist</th><th>confusion</th><th>abstain</th><th>err/unp</th>
<th>lat med/p95 ms</th></tr>
{rows_html}
</table>
<p class="muted">accuracy & kappa over non-conflict inputs with a usable verdict;
confusion is vs Opus (positive = real_bug): TP real✓ / FN missed-real (strict) / FP
false-real (lenient: noise waved through) / TN false-positive✓. abstain = explicit
needs_review (a valid no-call, excluded from accuracy). self-consist = share of inputs
whose trials all agreed.</p>

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
<p class="muted">Voting hypothesis: where the panel is unanimous it should track Opus
closely (easy calls, no vote needed); the value is on the 2-1 splits, exactly the
borderline calls a tie-break is for.</p>

<h2>Directional bias — how each judge errs vs Opus</h2>
<p class="muted">Lenient = calls a finding <b>real</b> when Opus called it a
<b>false positive</b> (waves noise through — the failure that does NOT reduce false
positives). Strict = calls it a <b>false positive</b> when Opus said <b>real</b>
(discards a genuine bug). For the FP-reduction use-case, lenient is the costly
direction.</p>
<table>
<tr><th>model</th><th>lenient — calls real, Opus said FP</th>
<th>strict — calls FP, Opus said real</th><th>net lean</th></tr>
{bias_rows}
</table>

<h2>Distinct spots where all 3 shadows agree but contradict Opus ({len(dd)})</h2>
<p class="muted">Deduped by (case, file, line). All three cheap judges agreed with each
other but disagreed with Opus — either Opus is borderline here, or the panel shares a
blind spot. Pair with the truth-judge disagreement report for human refereeing.</p>
<table>{ds_rows or "<tr><td class=muted>none</td></tr>"}</table>
</body></html>"""
    Path(path).write_text(doc)
    print(f"\nwrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="nelson-shadow-fp-judge.db")
    ap.add_argument("--html", default="bench-shadow-fp-judge.html")
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
