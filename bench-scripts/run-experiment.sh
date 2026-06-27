#!/usr/bin/env bash
# Run a benchmark experiment into an ISOLATED DB, leaving the sacrosanct baseline
# nelson.db untouched. This is the generalized form of run_semgrep_ab.sh,
# run_treesitter_*.sh, run_reposcope*.sh and the oracle/control passes: a fresh
# or resumable DB seeded from a roster + a cases directory, with the experiment
# knobs (repeat, oracle-cwe, repo-scope) exposed as flags.
#
# Unlike add-model.sh this DOES pass --competitors: an experiment DB's roster is
# its source of truth (there is nothing else to retire), so the normal seeding
# path is correct. `bench loop` is idempotent, so a re-run resumes by default and
# only fills missing cells. Pass --fresh to start clean (rm the DB first).
#
# Usage:
#   bench-scripts/run-experiment.sh --db EXP.db --roster R.yaml --cases-dir DIR [options]
#
# Options:
#   --db PATH         experiment DB (REQUIRED; must not be nelson.db)
#   --roster R.yaml   competitor roster (REQUIRED)
#   --cases-dir DIR   case manifest directory (REQUIRED, e.g. cases/ or a subset)
#   --html FILE       report path (default: bench-<db-stem>.html)
#   --concurrency N   worker count (default: 2)
#   --timeout S       per-run wall-clock cap (default: 1800)
#   --repeat N        trials per cell (default: 1)
#   --fresh           rm the DB first for a clean, reproducible run
#   --oracle-cwe      leak each case's ground-truth CWE into the prompt
#   --repo-scope      audit the whole tree instead of per-file (if roster doesn't set it)
#   --                everything after is passed through to `bench loop`
#
# Examples:
#   bench-scripts/run-experiment.sh --db nelson-semgrep-exp.db \
#       --roster bench-scripts/rosters/competitors-semgrep-ab.yaml --cases-dir cases/ --fresh
#   bench-scripts/run-experiment.sh --db nelson-reposcope-exp.db \
#       --roster bench-scripts/rosters/competitors-reposcope.yaml \
#       --cases-dir bench-scripts/case-subsets/cases-reposcope/ --repeat 3 --concurrency 3
source "$(dirname "$0")/lib/common.sh"

DB=""
ROSTER=""
CASES_DIR=""
HTML=""
CONCURRENCY=2
TIMEOUT=1800
REPEAT=1
FRESH=0
EXTRA=()
PASSTHROUGH=()

while [ $# -gt 0 ]; do
  case "$1" in
    --db)          DB="$2"; shift 2 ;;
    --roster)      ROSTER="$2"; shift 2 ;;
    --cases-dir)   CASES_DIR="$2"; shift 2 ;;
    --html)        HTML="$2"; shift 2 ;;
    --concurrency) CONCURRENCY="$2"; shift 2 ;;
    --timeout)     TIMEOUT="$2"; shift 2 ;;
    --repeat)      REPEAT="$2"; shift 2 ;;
    --fresh)       FRESH=1; shift ;;
    --oracle-cwe)  EXTRA+=(--oracle-cwe); shift ;;
    --repo-scope)  EXTRA+=(--repo-scope); shift ;;
    --)            shift; PASSTHROUGH=("$@"); break ;;
    -*)            echo "unknown option: $1" >&2; exit 2 ;;
    *)             echo "unexpected argument: $1" >&2; exit 2 ;;
  esac
done

[ -n "$DB" ]        || { echo "error: --db is required" >&2; exit 2; }
[ -n "$ROSTER" ]    || { echo "error: --roster is required" >&2; exit 2; }
[ -n "$CASES_DIR" ] || { echo "error: --cases-dir is required" >&2; exit 2; }
[ "$DB" != "nelson.db" ] || { echo "error: refusing to run an experiment into the baseline nelson.db" >&2; exit 2; }

[ -z "$HTML" ] && HTML="bench-$(basename "$DB" .db).html"

if [ "$FRESH" -eq 1 ]; then
  rm -f "$DB"
  echo "fresh start: removed $DB"
fi

echo "=== experiment loop -> $DB (roster: $ROSTER, cases: $CASES_DIR, repeat: $REPEAT) ==="
"$PY" -m nelson bench loop \
  --db "$DB" \
  --competitors "$ROSTER" \
  --cases-dir "$CASES_DIR" \
  --no-age-out \
  --repeat "$REPEAT" \
  --concurrency "$CONCURRENCY" \
  --timeout "$TIMEOUT" \
  --html "$HTML" \
  "${EXTRA[@]}" "${PASSTHROUGH[@]}"
