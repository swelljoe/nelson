#!/usr/bin/env bash
# Launch the config-driven trial-sweep runner (runners/arm_sweep.py) for one or
# more models against a SHARED experiment DB. This is the generalized form of the
# run_promptlab_experiment*.sh / run_gemma_promptlab_experiment.sh launchers.
#
# With one --model it runs in the foreground. With several it launches one nohup
# process per model (each typically a different self-hosted server) in parallel,
# staggered so their startup case-upserts don't collide on the SQLite lock, then
# waits for all. A shared DB keeps run_ids and `nelson-run-<id>` container names
# globally unique across the parallel processes.
#
# Scoring is NOT run here — these experiments read detection straight from
# run_findings. Run a report generator (reports/) afterwards.
#
# Usage:
#   bench-scripts/run-arm-sweep.sh --config CONF --db DB --model M1 [--model M2 ...] [opts]
#
# Options:
#   --config FILE   experiment YAML (runners/configs/*.yaml)  [required]
#   --db PATH       shared experiment DB                       [required]
#   --model NAME    competitor to run (repeatable)             [>=1 required]
#   --repeat N      override config repeat (arms mode)
#   --timeout S     per-run wall-clock cap (default: 1800)
#   --logdir DIR    where per-model logs go (default: .)
#
# Example (two self-hosted Qwen quants, one per box, shared DB):
#   bench-scripts/run-arm-sweep.sh --config bench-scripts/runners/configs/promptlab-qwen.yaml \
#       --db nelson-promptlab.db \
#       --model raw-api-loop/qwen3.6-27b --model raw-api-loop/qwen3.6-35b-A3b
source "$(dirname "$0")/lib/common.sh"

CONFIG=""
DB=""
MODELS=()
LOGDIR="."
EXTRA=()

while [ $# -gt 0 ]; do
  case "$1" in
    --config)  CONFIG="$2"; shift 2 ;;
    --db)      DB="$2"; shift 2 ;;
    --model)   MODELS+=("$2"); shift 2 ;;
    --repeat)  EXTRA+=(--repeat "$2"); shift 2 ;;
    --timeout) EXTRA+=(--timeout "$2"); shift 2 ;;
    --logdir)  LOGDIR="$2"; shift 2 ;;
    *)         echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

[ -n "$CONFIG" ] || { echo "error: --config is required" >&2; exit 2; }
[ -n "$DB" ]     || { echo "error: --db is required" >&2; exit 2; }
[ ${#MODELS[@]} -ge 1 ] || { echo "error: at least one --model is required" >&2; exit 2; }

RUNNER="$_COMMON_DIR/../runners/arm_sweep.py"
mkdir -p "$LOGDIR" bench-runs

# Single model: run in the foreground so the caller sees the trajectory live.
if [ ${#MODELS[@]} -eq 1 ]; then
  exec "$PY" "$RUNNER" --config "$CONFIG" --model "${MODELS[0]}" --db "$DB" "${EXTRA[@]}"
fi

# Multiple models: one nohup process each, staggered, then wait for all.
pids=()
for m in "${MODELS[@]}"; do
  stem="$(echo "$m" | tr '/ ' '__')"
  log="$LOGDIR/arm-sweep-$stem.log"
  echo "=== [$(date -u +%H:%M:%S)] launching $m -> $DB (log $log) ==="
  nohup "$PY" "$RUNNER" --config "$CONFIG" --model "$m" --db "$DB" "${EXTRA[@]}" \
    >"$log" 2>&1 &
  pids+=($!)
  sleep 5   # stagger so the startup case-upserts don't collide on the lock
done

echo "waiting for ${#pids[@]} process(es) ..."
rc=0
for p in "${pids[@]}"; do
  wait "$p" || rc=$?
done
echo "=== [$(date -u +%H:%M:%S)] DONE (exit $rc) ==="
exit $rc
