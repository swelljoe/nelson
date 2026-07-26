#!/usr/bin/env bash
# Track B ("with preferred agent" / product leaderboard): add four vendor-native
# agent rows — reasonix/deepseek-v4-pro, qwen/qwen3.8-max-preview,
# mimo/mimo-v2.5-pro, kimi/k3 — alongside the existing native rows (claude-code/opus,
# codex/gpt-5.6-sol). Each drives its own model through its own agent CLI, restricted
# to a read-only/no-web-tool posture (contamination guard; the per-run stdout is the
# audit trail). NEVER cross-ranked against the uniform raw-api-loop "model" track.
#
# Retire the rest of the field so only these four plan this pass; reactivate on exit.
#
# COST: kimi/mimo/qwen run on prepaid subscription/token plans -> null cost on the
# board. reasonix meters DeepSeek V4 Pro upstream but its per-token counts land in a
# torn-down staging mount, so it too shows null here (documented in the roster).
# --max-spend-usd is a belt-and-suspenders backstop; real spend is bounded by the
# plans + --timeout. Concurrency 4 = one worker per competitor (distinct vendor
# endpoints, so no shared rate limit); kimi may 429 at Chinese peak and get pulled.
source "$(dirname "$0")/lib/common.sh"

DB=nelson.db
ROSTER=bench-scripts/rosters/competitors-native-agents-2026-07-23.yaml
HTML=bench-report-baseline-2026-07-21.html
SNAP=/home/joe/src/nelson/.native-agents-retired-snapshot.txt
NEW="reasonix/deepseek-v4-pro qwen/qwen3.8-max-preview mimo/mimo-v2.5-pro kimi/k3"

backup_db "$DB" "pre-native-agents-2026-07-23"
"$PY" "$_COMMON_DIR/surgical_upsert.py" "$DB" "$ROSTER"

"$PY" - "$DB" "$SNAP" $NEW <<'PY'
import sqlite3, sys
db, snap, keep = sys.argv[1], sys.argv[2], sys.argv[3:]
c = sqlite3.connect(db)
rows = [r[0] for r in c.execute("SELECT name FROM competitors WHERE status='active'")]
retire = [n for n in rows if n not in keep]
open(snap, "w").write("\n".join(retire) + ("\n" if retire else ""))
c.executemany("UPDATE competitors SET status='retired' WHERE name=?", [(n,) for n in retire])
c.commit()
print(f"retired {len(retire)} field competitors (snapshot -> {snap})")
print("active now:", [r[0] for r in c.execute("SELECT name FROM competitors WHERE status='active'")])
PY

reactivate() {
  [ -s "$SNAP" ] || { echo "no snapshot to reactivate"; return; }
  "$PY" - "$DB" "$SNAP" <<'PY'
import sqlite3, sys
db, snap = sys.argv[1], sys.argv[2]
names = [l.strip() for l in open(snap) if l.strip()]
c = sqlite3.connect(db)
c.executemany("UPDATE competitors SET status='active' WHERE name=?", [(n,) for n in names])
c.commit()
print(f"reactivated {len(names)} field competitors")
PY
}
trap reactivate EXIT

echo "=== bench loop (fill 4 native-agent rows) -> $DB ==="
"$PY" -m nelson bench loop \
  --db "$DB" \
  --no-age-out \
  --concurrency 4 \
  --timeout 1800 \
  --repeat 3 \
  --max-spend-usd 60 \
  --runs-dir runs-native-agents \
  --cache-dir .cache-native-agents \
  --fp-cache-dir .cache-native-agents-fp

reactivate
trap - EXIT
echo "=== leaderboard -> $HTML ==="
"$PY" -m nelson bench leaderboard --db "$DB" --html "$HTML"
echo "=== DONE. Report: $HTML ==="
