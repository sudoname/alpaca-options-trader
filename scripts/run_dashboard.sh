#!/usr/bin/env bash
# =============================================================================
# Oracle dashboard launcher — run it yourself, no Claude needed.
#
# Usage (from repo root, Git Bash on Windows or bash on Linux):
#   bash scripts/run_dashboard.sh                # serve on 127.0.0.1:8787
#   bash scripts/run_dashboard.sh --selftest     # config + route self-test, then exit
#   DASHBOARD_PORT=9000 bash scripts/run_dashboard.sh
#
# Configuration comes from .env / environment (see oracle_dashboard.py):
#   DASHBOARD_HOST              default 127.0.0.1
#   DASHBOARD_PORT              default 8787
#   DASHBOARD_CACHE_TTL         default 60 (seconds)
#   DASHBOARD_BASIC_AUTH_USER / DASHBOARD_BASIC_AUTH_PASS   optional basic auth
#   DASHBOARD_ENABLE_TRAINING   "1" to expose the one-click training button
#   DASHBOARD_TRAINING_LEDGER   required when training is enabled
#
# On the production server this is managed by systemd (alps-dashboard.service);
# this script is for running it locally against local/snapshot data.
# =============================================================================
set -euo pipefail

cd "$(dirname "$0")/.."

PY="${PY:-python}"

# Load .env without overwriting variables already set in the shell
# (same convention as run.sh).
if [[ -f .env ]]; then
  while IFS= read -r line; do
    [[ "$line" =~ ^[[:space:]]*# ]] && continue
    [[ "$line" =~ ^[[:space:]]*$ ]] && continue
    key="${line%%=*}"
    [[ "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
    if [[ -z "${!key:-}" ]]; then
      value="${line#*=}"
      value="${value%$'\r'}"
      # strip one layer of surrounding quotes
      value="${value#\"}"; value="${value%\"}"
      value="${value#\'}"; value="${value%\'}"
      export "$key=$value"
    fi
  done < .env
fi

echo "Dashboard: http://${DASHBOARD_HOST:-127.0.0.1}:${DASHBOARD_PORT:-8787}  (Ctrl+C to stop)"
exec "$PY" oracle_dashboard.py "$@"
