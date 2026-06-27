#!/usr/bin/env bash
# Shared preamble for every bench-scripts runner. Source it, don't execute it:
#
#   source "$(dirname "$0")/lib/common.sh"
#
# It does three things:
#   1. cd to the repo root (one level above bench-scripts/), so every relative
#      path (rosters/, cases/, the .venv) resolves the same way no matter where
#      the script was launched from.
#   2. Loads API keys from a secrets directory into the env vars the harness
#      expects, but ONLY for files that exist — a replicator who only has an
#      OpenRouter key still gets a working environment for OpenRouter models.
#   3. Exposes $PY (the project venv interpreter) and a backup_db helper.
#
# Secrets layout (override the directory with NELSON_SECRETS_DIR):
#   $NELSON_SECRETS_DIR/openrouter -> OPENROUTER_API_KEY
#   $NELSON_SECRETS_DIR/deepseek   -> DEEPSEEK_API_KEY
#   $NELSON_SECRETS_DIR/mimo       -> MIMO_API_KEY
#   $NELSON_SECRETS_DIR/gemini     -> GEMINI_API_KEY
# Self-hosted servers (llama-server / LM Studio) ignore the value, so a literal
# LMSTUDIO_API_KEY=lm-studio is exported unconditionally.

set -euo pipefail

# --- repo root -------------------------------------------------------------
# This file lives at <repo>/bench-scripts/lib/common.sh.
_COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$_COMMON_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# --- interpreter -----------------------------------------------------------
PY="${NELSON_PY:-$REPO_ROOT/.venv/bin/python}"
[ -x "$PY" ] || { echo "error: python interpreter not found at $PY (set NELSON_PY)"; exit 1; }

# --- secrets ---------------------------------------------------------------
NELSON_SECRETS_DIR="${NELSON_SECRETS_DIR:-/home/joe/secrets}"

# load_secret <file-basename> <ENV_VAR>: export ENV_VAR from the file if it
# exists and is non-empty; otherwise leave any pre-set value untouched.
load_secret() {
  local f="$NELSON_SECRETS_DIR/$1" var="$2"
  if [ -s "$f" ]; then
    export "$var"="$(cat "$f")"
  fi
}

load_secret openrouter OPENROUTER_API_KEY
load_secret deepseek   DEEPSEEK_API_KEY
load_secret mimo       MIMO_API_KEY
load_secret gemini     GEMINI_API_KEY
export LMSTUDIO_API_KEY="${LMSTUDIO_API_KEY:-lm-studio}"

# --- helpers ---------------------------------------------------------------
# backup_db <db-path> [suffix]: copy the DB to <db>.bak-<suffix> if it isn't
# already there (cp -n). Use before any mutation of the sacrosanct baseline.
backup_db() {
  local db="$1" suffix="${2:-pre-run-$(date -u +%Y-%m-%d)}"
  local dst="$db.bak-$suffix"
  if [ -f "$db" ]; then
    cp -n "$db" "$dst"
    echo "backed up $db -> $dst"
  fi
}
