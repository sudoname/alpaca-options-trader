#!/usr/bin/env bash
#
# deploy.sh — pull-based deploy for the alpaca-options-trader bot.
#
# Fetches origin/master, fast-forwards the working tree, installs deps, then
# GATES on the no-cred / no-network self-tests: if any integrity, risk, or
# model check fails the deploy aborts BEFORE touching a running process.
#
# Scope: shadow + paper only. This script ships code; it never changes what the
# bot trades and never starts an autonomous live loop. The optional restart is
# whatever you put in $RESTART_CMD — nothing is assumed about your supervisor.
#
# Usage:
#   ./deploy.sh                       # pull + verify (no restart)
#   ./deploy.sh --restart             # pull + verify + run $RESTART_CMD
#   RESTART_CMD="systemctl --user restart trader" ./deploy.sh --restart
#   BRANCH=master REMOTE=origin PYTHON=python3 ./deploy.sh
#
set -euo pipefail

# --- config (override via env) --------------------------------------------
BRANCH="${BRANCH:-master}"
REMOTE="${REMOTE:-origin}"
PYTHON="${PYTHON:-python3}"

RESTART=0
if [ "${1:-}" = "--restart" ]; then
    RESTART=1
fi

# Run from the repo root regardless of the caller's cwd.
cd "$(dirname "$0")"

# Fall back to `python` if `python3` is not on PATH.
if ! command -v "$PYTHON" >/dev/null 2>&1; then
    PYTHON=python
fi

echo "==> Deploying $(basename "$PWD") (branch=$BRANCH, python=$PYTHON)"

# --- 1. pull (fast-forward only; never merge a diverged tree) -------------
git fetch "$REMOTE" --quiet
LOCAL_REF="$(git rev-parse HEAD)"
TARGET_REF="$(git rev-parse "$REMOTE/$BRANCH")"
if [ "$LOCAL_REF" = "$TARGET_REF" ]; then
    echo "==> Already up to date ($(git rev-parse --short HEAD))."
else
    echo "==> Updating $(git rev-parse --short HEAD) -> $(git rev-parse --short "$REMOTE/$BRANCH")"
    git pull --ff-only "$REMOTE" "$BRANCH"
fi

# --- 2. dependencies (no new deps today, but keep the env honest) ---------
if [ -f requirements.txt ]; then
    echo "==> Installing dependencies"
    "$PYTHON" -m pip install --quiet -r requirements.txt
fi

# --- 3. self-test gate (no creds, no network) -----------------------------
# Any non-zero exit here aborts the deploy via `set -e`, so a broken build
# can never reach the restart step.
echo "==> Running self-test gate"
"$PYTHON" tests_integrity.py
"$PYTHON" risk_engine.py --selftest
"$PYTHON" episode_store.py --selftest
"$PYTHON" cost_model.py --selftest
"$PYTHON" features.py --selftest
"$PYTHON" shadow_recorder.py --selftest
"$PYTHON" model.py --selftest
"$PYTHON" walk_forward.py --selftest
"$PYTHON" regime.py --selftest
"$PYTHON" backtest_rl_gate.py --selftest
echo "==> All self-tests passed."

# --- 4. optional restart (you own the supervisor) -------------------------
if [ "$RESTART" = "1" ]; then
    if [ -n "${RESTART_CMD:-}" ]; then
        echo "==> Restarting: $RESTART_CMD"
        eval "$RESTART_CMD"
    else
        echo "==> --restart given but RESTART_CMD is unset; skipping restart."
        echo "    e.g. RESTART_CMD=\"systemctl --user restart trader\" ./deploy.sh --restart"
    fi
else
    echo "==> Code deployed. No restart requested (pass --restart to restart)."
fi

echo "==> Done: now at $(git rev-parse --short HEAD)."
