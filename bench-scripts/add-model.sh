#!/usr/bin/env bash
# Add one or more models to the baseline benchmark WITHOUT disturbing existing
# results, then score and regenerate the report. This is the generalized form of
# every run_newbatch_*.sh / run_gemma_minimax_bench.sh / run_qwen_agentworld_bench.sh.
#
# What it does, in order:
#   0. back up the DB (cp -n, so an existing backup is preserved);
#   1. (optional) clear a competitor's stale runs so they re-plan fresh;
#   2. surgically upsert the roster's competitors as active — NO roster sync,
#      so every existing active model stays active and visible in the report;
#   3. `bench loop` with NO --competitors and --no-age-out, so only the new
#      (model x case x file) cells run; everyone else is already covered.
#
# Usage:
#   bench-scripts/add-model.sh <roster.yaml> [more-rosters.yaml ...] [options]
#
# Options:
#   --db PATH            target DB (default: nelson.db, the sacrosanct baseline)
#   --html FILE          report path (default: bench-report-<roster-stem>.html)
#   --concurrency N      worker count (default: 3; use 1 for a single local server)
#   --timeout S          per-run wall-clock cap in seconds (default: 1800)
#   --clear NAME         delete this competitor's runs before the loop (re-point)
#   --clear-status S...  with --clear, only clear runs in these statuses
#   --backup-suffix S    label for the .bak file (default: pre-<roster-stem>-<date>)
#   --                   everything after is passed through to `bench loop`
#
# Examples:
#   bench-scripts/add-model.sh bench-scripts/rosters/competitors-qwen-agentworld.yaml --concurrency 1
#   bench-scripts/add-model.sh bench-scripts/rosters/competitors-north-mini-code-openrouter.yaml \
#       --clear raw-api-loop/north-mini-code
source "$(dirname "$0")/lib/common.sh"

DB=nelson.db
HTML=""
CONCURRENCY=3
TIMEOUT=1800
CLEAR=""
CLEAR_STATUS=()
BACKUP_SUFFIX=""
ROSTERS=()
PASSTHROUGH=()

while [ $# -gt 0 ]; do
  case "$1" in
    --db)            DB="$2"; shift 2 ;;
    --html)          HTML="$2"; shift 2 ;;
    --concurrency)   CONCURRENCY="$2"; shift 2 ;;
    --timeout)       TIMEOUT="$2"; shift 2 ;;
    --clear)         CLEAR="$2"; shift 2 ;;
    --clear-status)  shift; while [ $# -gt 0 ] && [[ "$1" != --* ]]; do CLEAR_STATUS+=("$1"); shift; done ;;
    --backup-suffix) BACKUP_SUFFIX="$2"; shift 2 ;;
    --)              shift; PASSTHROUGH=("$@"); break ;;
    -*)              echo "unknown option: $1" >&2; exit 2 ;;
    *)               ROSTERS+=("$1"); shift ;;
  esac
done

[ ${#ROSTERS[@]} -gt 0 ] || { echo "error: at least one roster yaml is required" >&2; exit 2; }

# Default report name from the first roster: competitors-X.yaml -> bench-report-X.html
if [ -z "$HTML" ]; then
  stem="$(basename "${ROSTERS[0]}" .yaml)"; stem="${stem#competitors-}"
  HTML="bench-report-$stem.html"
fi
if [ -z "$BACKUP_SUFFIX" ]; then
  stem="$(basename "${ROSTERS[0]}" .yaml)"; stem="${stem#competitors-}"
  BACKUP_SUFFIX="pre-$stem-$(date -u +%Y-%m-%d)"
fi

# 0. Back up before any mutation.
backup_db "$DB" "$BACKUP_SUFFIX"

# 1. Optionally clear a re-pointed/crippled competitor's runs.
if [ -n "$CLEAR" ]; then
  "$PY" "$_COMMON_DIR/clear_competitor_runs.py" "$DB" "$CLEAR" \
    ${CLEAR_STATUS:+--status "${CLEAR_STATUS[@]}"}
fi

# 2. Surgically upsert the new competitors (no roster sync).
"$PY" "$_COMMON_DIR/surgical_upsert.py" "$DB" "${ROSTERS[@]}"

# 3. Fill only the new cells, score, regenerate the report.
echo "=== bench loop -> $DB (html: $HTML, concurrency: $CONCURRENCY) ==="
"$PY" -m nelson bench loop \
  --db "$DB" \
  --concurrency "$CONCURRENCY" \
  --no-age-out \
  --timeout "$TIMEOUT" \
  --html "$HTML" \
  "${PASSTHROUGH[@]}"
