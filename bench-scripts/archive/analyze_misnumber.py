#!/usr/bin/env python3
"""Offline test: did read_file's missing line numbers unfairly suppress detections?

Read-only over the frozen baseline ``nelson.db``. For every COMPLETE run's
finding that lands in a ground-truth FILE but currently fails the localization
gate, we relocate the model's *quoted code snippet* in the real pre-patch source
(at the case's vuln_commit), recover the TRUE line the snippet sits on, and
recompute the gate against that true line.

A finding that was a gate-MISS on its claimed line but a gate-HIT on its
relocated line is a detection the missing-line-number bug HID: the model quoted
the right code but hand-counted the wrong line. Those flips — and only those —
are the "unfairly limited" cases. Everything else (snippet genuinely sits far
from any hunk) is a different bug in the same file, which the gate+judge are
*supposed* to reject.

No DB writes. No model calls. Pure relocation + the real ``score.localize``.
Flips are emitted as JSON for a downstream Opus same-bug confirmation pass.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from nelson.runner import parse_competitor_findings
from nelson.score import DEFAULT_LINE_TOLERANCE, localize

DB = "nelson.db"
CACHE = Path("bench-cache")


def norm_base(p: str | None) -> str | None:
    return p.split("/")[-1] if p else p


def load_cases(conn: sqlite3.Connection) -> dict[int, sqlite3.Row]:
    return {r["id"]: r for r in conn.execute("SELECT * FROM cases")}


def gt_hunks_of(case: sqlite3.Row) -> list[dict]:
    raw = case["gt_hunks"]
    return json.loads(raw) if raw else []


def resolve_source(ext_id: str, reported_file: str) -> Path | None:
    """Map a model-reported repo-relative path to the checkout file on disk."""
    root = CACHE / ext_id / "src"
    direct = root / reported_file
    if direct.is_file():
        return direct
    # Fall back to basename search (some models drop the leading dir).
    base = norm_base(reported_file)
    hits = [p for p in root.rglob(base or "") if p.is_file()]
    return hits[0] if len(hits) == 1 else (hits[0] if hits else None)


def relocate(snippet: str | None, src: Path, claimed: int | None) -> list[int]:
    """Return source line numbers (1-based) where ``snippet`` actually appears.

    The model's ``code`` is usually one line but can be several. We anchor on the
    longest non-trivial stripped line (most distinctive), match it as a substring
    of each source line's stripped form, and return every hit. The caller picks
    the occurrence nearest the claimed line (the charitable reading).
    """
    if not snippet or not snippet.strip():
        return []
    anchors = [ln.strip() for ln in snippet.splitlines() if len(ln.strip()) >= 6]
    if not anchors:
        return []
    anchor = max(anchors, key=len)
    try:
        lines = src.read_text(errors="replace").splitlines()
    except OSError:
        return []
    hits = [i + 1 for i, ln in enumerate(lines) if anchor in ln.strip()]
    return hits


def pick(hits: list[int], claimed: int | None) -> int | None:
    if not hits:
        return None
    if claimed is None:
        return hits[0]
    return min(hits, key=lambda h: abs(h - claimed))


def main() -> None:
    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    cases = load_cases(conn)

    runs = conn.execute(
        """SELECT r.id, r.case_id, r.raw_output, comp.name AS comp
           FROM runs r JOIN competitors comp ON comp.id = r.competitor_id
           WHERE r.status = 'complete' AND r.raw_output IS NOT NULL"""
    ).fetchall()

    flips: list[dict] = []
    drift_hits: list[int] = []  # drift for snippets that land on a hunk
    n_findings = 0
    n_right_file = 0
    n_snippet_found = 0
    n_elsewhere = 0  # snippet found, but true line still off-hunk (different bug)
    n_no_snippet = 0
    n_stable_hit = 0
    n_optimistic_only = 0  # would flip only under any-occurrence relocation

    for r in runs:
        case = cases[r["case_id"]]
        hunks = gt_hunks_of(case)
        gt_files = {norm_base(h.get("file")) for h in hunks}
        try:
            findings = parse_competitor_findings(r["raw_output"])
        except Exception:
            continue
        for f in findings:
            n_findings += 1
            rfile = f.get("file")
            if norm_base(rfile) not in gt_files:
                continue
            n_right_file += 1
            claimed = f.get("line")
            try:
                claimed = int(claimed) if claimed is not None else None
            except (TypeError, ValueError):
                claimed = None
            claimed_hit = localize(
                rfile, claimed, hunks, DEFAULT_LINE_TOLERANCE
            ).matched

            src = resolve_source(case["ext_id"], rfile)
            code = f.get("code")
            hits = relocate(code, src, claimed) if src else []
            true_line = pick(hits, claimed)

            if true_line is None:
                if claimed_hit:
                    n_stable_hit += 1
                else:
                    n_no_snippet += 1
                continue
            n_snippet_found += 1
            reloc_hit = localize(
                rfile, true_line, hunks, DEFAULT_LINE_TOLERANCE
            ).matched
            # Optimistic upper bound: does ANY occurrence of the snippet land on a
            # hunk? (charitable relocation picks nearest-claimed and can miss a
            # recurrence that sits in the fix region). If even this is ~0, the
            # "no broad suppression" conclusion is robust to relocation choice.
            any_on_hunk = any(
                localize(rfile, h, hunks, DEFAULT_LINE_TOLERANCE).matched for h in hits
            )
            if (not claimed_hit) and any_on_hunk and not reloc_hit:
                n_optimistic_only += 1
            drift = abs(true_line - claimed) if claimed is not None else None

            if claimed_hit and reloc_hit:
                n_stable_hit += 1
            elif (not claimed_hit) and reloc_hit:
                # The flip: missing line numbers turned a real localization into a miss.
                if drift is not None:
                    drift_hits.append(drift)
                flips.append(
                    {
                        "run_id": r["id"],
                        "case": case["ext_id"],
                        "cwe": case["cwe"],
                        "comp": r["comp"],
                        "file": rfile,
                        "claimed_line": claimed,
                        "true_line": true_line,
                        "drift": drift,
                        "code": (code or "").strip()[:200],
                        "explanation": (f.get("explanation") or "")[:400],
                        "reported_cwe": f.get("cwe"),
                    }
                )
            else:
                n_elsewhere += 1

    print("=== Offline misnumbering audit (frozen nelson.db, read-only) ===")
    print(f"complete runs scanned         : {len(runs)}")
    print(f"total findings                : {n_findings}")
    print(f"  in a ground-truth file      : {n_right_file}")
    print(f"    snippet relocated in src  : {n_snippet_found}")
    print(f"    snippet not found         : {n_no_snippet}")
    print(f"  stable hits (gate already ok): {n_stable_hit}")
    print(f"  different bug (snippet far from hunk): {n_elsewhere}")
    print(f"  >>> FLIPS miss->hit (nearest-claimed): {len(flips)} <<<")
    print(f"      + flips only under any-occurrence : {n_optimistic_only}")
    if drift_hits:
        drift_hits.sort()
        mid = drift_hits[len(drift_hits) // 2]
        print(
            f"      flip drift lines: min={drift_hits[0]} "
            f"median={mid} max={drift_hits[-1]}"
        )
    print()
    for x in flips:
        print(
            f"[FLIP] {x['case']} {x['comp']}  {norm_base(x['file'])} "
            f"claimed L{x['claimed_line']} -> true L{x['true_line']} "
            f"(drift {x['drift']})  GT={x['cwe']} reported={x['reported_cwe']}"
        )
        print(f"        code: {x['code'][:110]}")

    Path("misnumber_flips.json").write_text(json.dumps(flips, indent=2))
    print(f"\nwrote {len(flips)} flip(s) -> misnumber_flips.json")


if __name__ == "__main__":
    main()
