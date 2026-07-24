#!/usr/bin/env python3
"""Backfill per-run token counts into the DB by parsing existing transcripts.

    python bench-scripts/backfill_tokens.py [DB]

The native-agent CLIs don't all report usage in a way the runner captured live
(only qwen's JSON was parsed). But two of them print recoverable token accounting
into their stdout, which we save in full to the transcript files:

  reasonix  -- one "· <total> tok · in <A> (... ) out <B> ..." line PER TURN.
               Summing A and B across turns reproduces its --metrics totals exactly
               (verified: a 2-turn run's per-turn ins 11869+11952 == prompt_tokens
               23821). Cache-blind (A counts re-read context), which is the right
               "tokens the model actually processed" number for an efficiency view.
  qwen      -- the final type==result object carries cumulative usage.

mimo (default format) and kimi (text) emit no token counts, so they stay null.
Only COMPLETED runs with a transcript are touched, and only their token columns —
safe to run against the live DB (busy_timeout) and idempotent, so re-run it at the
end to sweep everything.
"""

import json
import re
import sqlite3
import sys
from pathlib import Path

DB = sys.argv[1] if len(sys.argv) > 1 else "nelson.db"
# reasonix per-turn line: "· 60237 tok · in 59335 (58752 cached / 583 new) · out 902 ..."
_REASONIX_TURN = re.compile(r"·\s*\d+\s*tok\s*·\s*in\s+(\d+).*?·\s*out\s+(\d+)")


def parse_reasonix(text):
    turns = _REASONIX_TURN.findall(text)
    if not turns:
        return None, None
    tin = sum(int(a) for a, _ in turns)
    tout = sum(int(b) for _, b in turns)
    return tin, tout


def parse_qwen(text):
    try:
        arr = json.loads(text[text.index("[") :])
    except (ValueError, json.JSONDecodeError):
        return None, None
    results = [o for o in arr if isinstance(o, dict) and o.get("type") == "result"]
    if not results:
        return None, None
    u = results[-1].get("usage") or {}
    return u.get("input_tokens"), u.get("output_tokens")


PARSERS = {"reasonix": parse_reasonix, "qwen": parse_qwen}

c = sqlite3.connect(DB, timeout=30)
c.execute("PRAGMA busy_timeout=30000")
c.row_factory = sqlite3.Row
comp = {r["id"]: r["name"] for r in c.execute("SELECT id, name FROM competitors")}

updated = {n: 0 for n in ("reasonix", "qwen")}
skipped = 0
for r in c.execute(
    "SELECT id, competitor_id, transcript_path, tokens_in FROM runs "
    "WHERE status='complete' AND transcript_path IS NOT NULL"
):
    name = comp.get(r["competitor_id"], "")
    runtime = name.split("/", 1)[0]
    parser = PARSERS.get(runtime)
    if parser is None:
        continue
    p = Path(r["transcript_path"])
    if not p.exists():
        skipped += 1
        continue
    tin, tout = parser(p.read_text(errors="replace"))
    if tin is None:
        continue
    c.execute(
        "UPDATE runs SET tokens_in=?, tokens_out=? WHERE id=?", (tin, tout, r["id"])
    )
    updated[runtime] += 1
c.commit()

print(f"backfilled tokens: {updated}   (missing transcript: {skipped})")
# quick sanity: mean tokens/run now captured, per competitor
print("\n  per-competitor token capture after backfill:")
for cid, name in sorted(comp.items(), key=lambda kv: kv[1]):
    row = c.execute(
        "SELECT COUNT(*) n, COUNT(tokens_in) tok, "
        "CAST(AVG(tokens_in) AS INT) ai, CAST(AVG(tokens_out) AS INT) ao "
        "FROM runs WHERE competitor_id=? AND status='complete'",
        (cid,),
    ).fetchone()
    if row["n"] and name.split("/")[0] in ("reasonix", "qwen", "mimo", "kimi"):
        print(
            f"    {name:32s} complete={row['n']:3d} with_tokens={row['tok']:3d} "
            f"mean_in={row['ai']} mean_out={row['ao']}"
        )
