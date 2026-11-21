#!/usr/bin/env bash
# =============================================================================
# RL Q-table training — run it yourself, no Claude needed.
#
# Trains the rule/RL gate's Q-table (rl_qtable.json) by replaying recorded
# decision episodes. This is the lightweight learner that sits in the live
# decision path; the heavier Oracle4 ML pipeline is scripts/train_oracle4.sh.
#
# Usage (from repo root, Git Bash on Windows or bash on Linux):
#   bash scripts/train_rl.sh                     # fetch episodes.db, train, report
#   bash scripts/train_rl.sh --skip-fetch        # reuse an existing local episodes.db
#   bash scripts/train_rl.sh --epochs 5          # more replay passes
#   bash scripts/train_rl.sh --reset             # start from a fresh Q-table
#   EPISODES_DB=server_data/2026-11-19/episodes.db bash scripts/train_rl.sh --skip-fetch
#
# What it does:
#   1. fetch  — scp the live episodes.db from the server (unless --skip-fetch)
#   2. train  — python train_rl.py --source episodes --replay  (+ your flags)
#   3. report — python train_rl.py --report  (Q-table summary)
#
# The updated Q-table lands in rl_qtable.json (repo root). Deploying it to the
# server is a deliberate, separate step — see RUNBOOK.md.
# =============================================================================
set -euo pipefail

cd "$(dirname "$0")/.."

# ---- configuration (override via env) ---------------------------------------
PY="${PY:-python}"
SERVER="${SERVER:-root@147.182.242.177}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519_1}"
REMOTE_DB="${REMOTE_DB:-/var/www/alps/episodes.db}"
QTABLE="${QTABLE:-rl_qtable.json}"

STAMP="$(date +%F)"
EPISODES_DB="${EPISODES_DB:-server_data/${STAMP}/episodes.db}"

SKIP_FETCH=0
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-fetch) SKIP_FETCH=1; shift ;;
    *) EXTRA_ARGS+=("$1"); shift ;;   # pass through: --epochs N, --reset, ...
  esac
done

step() { printf '\n==== %s ====\n' "$*"; }

# ---- 1. fetch episodes.db ----------------------------------------------------
if [[ "$SKIP_FETCH" -eq 0 ]]; then
  step "1/3 fetch episodes.db from ${SERVER}"
  mkdir -p "$(dirname "$EPISODES_DB")"
  scp -i "$SSH_KEY" "${SERVER}:${REMOTE_DB}" "$EPISODES_DB"
else
  step "1/3 fetch skipped — using ${EPISODES_DB}"
  [[ -f "$EPISODES_DB" ]] || { echo "episodes db not found: ${EPISODES_DB}" >&2; exit 1; }
fi

# ---- 2. train ------------------------------------------------------------------
step "2/3 replay-train Q-table (${QTABLE})"
"$PY" train_rl.py --replay \
  --source episodes \
  --db "$EPISODES_DB" \
  --qtable "$QTABLE" \
  "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"

# ---- 3. report -------------------------------------------------------------------
step "3/3 Q-table report"
"$PY" train_rl.py --report --qtable "$QTABLE"

step "DONE"
cat <<EOF

Updated Q-table: ${QTABLE}

To deploy it to the server (deliberate manual step):
  scp -i ${SSH_KEY} ${QTABLE} ${SERVER}:/var/www/alps/rl_qtable.json
  ssh -i ${SSH_KEY} ${SERVER} 'systemctl restart alps-scheduler alps-bot'
EOF
