#!/usr/bin/env python3
"""Adjudication report: the cases where all 3 shadow judges contradict Opus.

For each (case, file, line) where DeepSeek + MiMo + the local Gemma unanimously
disagreed with the Opus truth judge, lay out everything a human needs to referee the
call: the advisory (ground truth), the pre-patch SOURCE around the flagged line, what
the reporting model said the bug was, Opus's verdict + reasoning, and each shadow's
verdict + reasoning. The point is to reality-check the "Opus is right" assumption.

Reads nelson-shadow-judge.db (shadow verdicts) + the source DBs (findings + Opus
reasoning, READ-ONLY) and fetches source via `git show vuln_commit:path`.

Usage: .venv/bin/python shadow_disagreement_report.py [--html bench-shadow-disagree.html]
"""

from __future__ import annotations

import argparse
import hashlib
import html
import sqlite3
from collections import defaultdict
from pathlib import Path

from nelson.corpus import Case
from nelson.report_style import BASE_CSS, THEME_HEAD, THEME_TOGGLE, THEME_VARS
from nelson.score import (
    GitCodeProvider,
    ReportedFinding,
    build_truth_prompt,
    parse_truth_verdict,
)

SOURCE_DBS = [
    "nelson.db",
    "nelson-exp.db",
    "nelson-gemma-promptlab.db",
    "nelson-oracle.db",
    "nelson-oracle-control.db",
    "nelson-repeat.db",
]
MODELS_ORDER = ["deepseek", "mimo", "gemma31b"]
CODE_CONTEXT = 22  # lines either side of the flagged line


def load_eval_meta() -> tuple[dict, dict]:
    """hash -> finding/advisory metadata, and ext_id -> Case (for source fetch)."""
    by_hash: dict[str, dict] = {}
    cases: dict[str, Case] = {}
    for db_path in SOURCE_DBS:
        if not Path(db_path).exists():
            continue
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        case_cache: dict[int, Case] = {}
        rows = conn.execute(
            """SELECT rf.*, r.case_id AS _case_id
               FROM run_findings rf JOIN runs r ON rf.run_id = r.id
               WHERE rf.judge_truth_verdict IN ('same_bug', 'different_bug')"""
        ).fetchall()
        for row in rows:
            cid = row["_case_id"]
            if cid not in case_cache:
                crow = conn.execute(
                    "SELECT * FROM cases WHERE id = ?", (cid,)
                ).fetchone()
                if crow is None:
                    continue
                case_cache[cid] = Case.from_row(crow)
            case = case_cache[cid]
            cases.setdefault(case.ext_id, case)
            finding = ReportedFinding.from_row(row)
            h = hashlib.sha256(build_truth_prompt(case, finding).encode()).hexdigest()[
                :16
            ]
            if h in by_hash:
                continue
            by_hash[h] = {
                "ext_id": case.ext_id,
                "file": finding.file,
                "line": finding.line,
                "reported_desc": finding.description,
                "reported_cwe": finding.cwe,
                "confidence": finding.confidence,
                "opus_verdict": row["judge_truth_verdict"],
                "opus_reasoning": row["judge_reasoning"],
                "source_db": db_path,
            }
        conn.close()
    return by_hash, cases


def load_shadow(db: str):
    """hash -> {model: (majority_verdict, representative_reasoning)} + opus refs."""
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """SELECT model, input_hash, opus_same_bug, opus_conflict, shadow_same_bug,
                  raw_reply FROM shadow_verdicts"""
    ).fetchall()
    conn.close()
    # collect per (hash, model): votes + reasonings
    agg: dict[str, dict] = defaultdict(
        lambda: {"opus_same": None, "conflict": 0, "models": defaultdict(list)}
    )
    for r in rows:
        h = r["input_hash"]
        agg[h]["opus_same"] = r["opus_same_bug"]
        agg[h]["conflict"] = r["opus_conflict"]
        reasoning = ""
        if r["raw_reply"]:
            parsed = parse_truth_verdict(r["raw_reply"])
            if parsed:
                reasoning = parsed[1]
        agg[h]["models"][r["model"]].append((r["shadow_same_bug"], reasoning))

    out: dict[str, dict] = {}
    for h, d in agg.items():
        models: dict[str, tuple] = {}
        for m, votes in d["models"].items():
            usable = [(v, why) for v, why in votes if v is not None]
            if not usable:
                models[m] = (None, "")
                continue
            ones = sum(v for v, _ in usable)
            maj = (
                1 if ones * 2 > len(usable) else (0 if ones * 2 < len(usable) else None)
            )
            # representative reasoning: first usable vote matching the majority
            why = next((w for v, w in usable if v == maj), usable[0][1])
            models[m] = (maj, why)
        out[h] = {
            "opus_same": d["opus_same"],
            "conflict": d["conflict"],
            "models": models,
        }
    return out


def find_disagreements(shadow: dict) -> list[str]:
    """Hashes where all 3 shadows have a verdict, are unanimous, and != Opus."""
    hits = []
    for h, d in shadow.items():
        if d["conflict"]:
            continue
        verds = [d["models"].get(m, (None, ""))[0] for m in MODELS_ORDER]
        if any(v is None for v in verds):
            continue
        if len(set(verds)) == 1 and verds[0] != d["opus_same"]:
            hits.append(h)
    return hits


def code_window(provider: GitCodeProvider, case: Case, file: str, line: int) -> str:
    src = provider.source(case, file)
    if not src:
        return "(source unavailable for this revision/path)"
    lines = src.splitlines()
    if not line or line < 1 or line > len(lines):
        return "\n".join(lines[:CODE_CONTEXT]) + "\n…(line out of range)"
    lo = max(0, line - 1 - CODE_CONTEXT)
    hi = min(len(lines), line + CODE_CONTEXT)
    out = []
    for i in range(lo, hi):
        mark = ">>" if (i + 1) == line else "  "
        out.append(f"{mark}{i + 1:>5} {lines[i]}")
    return "\n".join(out)


def render(disagree_hashes, meta, shadow, cases, html_path):
    esc = html.escape

    # group by (ext_id, file, line)
    groups: dict[tuple, list[str]] = defaultdict(list)
    for h in disagree_hashes:
        m = meta.get(h)
        if not m:
            continue
        groups[(m["ext_id"], m["file"], m["line"])].append(h)

    provider = GitCodeProvider()
    sections = []
    for (ext_id, file, line), hashes in sorted(groups.items()):
        case = cases[ext_id]
        d0 = shadow[hashes[0]]
        opus_label = "same_bug" if d0["opus_same"] else "different_bug"
        shadow_label = "different_bug" if d0["opus_same"] else "same_bug"
        advisory = (
            f"<b>{esc(case.ext_id)}</b> &nbsp; {esc(case.cve_id or '')} &nbsp; "
            f"CWE: {esc(case.cwe or '?')} &nbsp; class: {esc(case.bug_class or '?')}"
            f"<br><span class='muted'>{esc(case.description or '')}</span>"
        )
        code = esc(code_window(provider, case, file, line))

        # one block per distinct reported finding (input variant) at this spot
        variants = []
        for h in hashes:
            m = meta[h]
            sm = shadow[h]["models"]
            shadow_rows = "".join(
                f"<tr><td class='mono'>{mk}</td>"
                f"<td class='v {'vsame' if sm.get(mk, (None,))[0] == 1 else 'vdiff'}'>"
                f"{'same_bug' if sm.get(mk, (None,))[0] == 1 else 'different_bug'}</td>"
                f"<td>{esc(sm.get(mk, (None, ''))[1] or '—')}</td></tr>"
                for mk in MODELS_ORDER
            )
            variants.append(
                f"<div class='variant'>"
                f"<div class='rep'><span class='tag'>reported bug</span> "
                f"<span class='muted'>[{esc(m['reported_cwe'] or '?')}, "
                f"conf {esc(str(m['confidence']) or '?')}]</span><br>"
                f"{esc(m['reported_desc'] or '(no description)')}</div>"
                f"<table class='judges'>"
                f"<tr><td class='mono'>OPUS</td>"
                f"<td class='v {'vsame' if m['opus_verdict'] == 'same_bug' else 'vdiff'}'>"
                f"{esc(m['opus_verdict'])}</td>"
                f"<td>{esc(m['opus_reasoning'] or '—')}</td></tr>"
                f"{shadow_rows}</table></div>"
            )

        sections.append(
            f"<section><h2>{esc(ext_id)} — <span class='mono'>{esc(file)}:{line}</span>"
            f" <span class='badge'>Opus: {opus_label} &nbsp;·&nbsp; all 3 shadows: "
            f"{shadow_label}</span> <span class='muted'>({len(hashes)} reporting "
            f"variant{'s' if len(hashes) > 1 else ''})</span></h2>"
            f"<div class='adv'>{advisory}</div>"
            f"<pre class='code'>{code}</pre>"
            f"{''.join(variants)}</section>"
        )

    extra = """
h2{border-bottom:0; font-size:1.05rem; margin:0 0 8px}
section{border:1px solid var(--border); border-radius:10px; padding:16px 18px;
        margin:0 0 22px; background:var(--surface)}
.adv{background:var(--surface-2); border-left:3px solid var(--accent); padding:8px 12px;
     border-radius:4px; margin:6px 0 10px}
pre.code{background:var(--surface-2); border:1px solid var(--border); border-radius:6px;
         padding:10px 12px; overflow-x:auto; font:12.5px/1.45 ui-monospace,
         SFMono-Regular,Menlo,monospace}
.variant{border-top:1px dashed var(--border); padding-top:10px; margin-top:10px}
.rep{margin:0 0 8px}
.tag{font-size:10px; text-transform:uppercase; letter-spacing:.05em;
     background:var(--surface-2); padding:2px 6px; border-radius:4px;
     color:var(--text-muted); margin-left:0}
table.judges{background:transparent; border:0; border-radius:0}
table.judges td{vertical-align:top; padding:4px 8px; border-bottom:1px solid var(--border)}
.judges .mono{white-space:nowrap}
.v{font-weight:700; font-size:12px; white-space:nowrap}
.vsame{color:var(--good)} .vdiff{color:var(--bad)}
"""
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Shadow vs Opus — disagreement adjudication</title>{THEME_HEAD}
<style>{THEME_VARS}{BASE_CSS}{extra}</style></head>
<body>{THEME_TOGGLE}
<h1>Shadow judges vs Opus — disagreement adjudication</h1>
<p class="intro">{len(groups)} distinct (case, file, line) spots where DeepSeek, MiMo and
the local Gemma-4-31B <b>all three agreed with each other but contradicted Opus</b>.
For each: the advisory ground truth, the pre-patch source around the flagged line
(<span class='mono'>&gt;&gt;</span> marks it), and — per reporting variant — what the
model said it found, then every judge's verdict and reasoning. Your call: is Opus right,
or is the cheap panel?</p>
{"".join(sections)}
</body></html>"""
    Path(html_path).write_text(doc)
    print(
        f"wrote {html_path} — {len(groups)} disagreement spots, "
        f"{len(disagree_hashes)} reporting variants"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shadow-db", default="nelson-shadow-judge.db")
    ap.add_argument("--html", default="bench-shadow-disagree.html")
    args = ap.parse_args()
    meta, cases = load_eval_meta()
    shadow = load_shadow(args.shadow_db)
    dh = find_disagreements(shadow)
    render(dh, meta, shadow, cases, args.html)


if __name__ == "__main__":
    main()
