#!/usr/bin/env bash
#
# run.sh — process launcher for the alpaca-options-trader bot.
#
# Starts the long-lived processes in the background (nohup), tracks them by
# PID file, and tees their output to ./logs. This is a thin supervisor: it
# does NOT change what the bot trades. Scope stays shadow + paper.
#
# Daemon managed by default:
#   1. telegram_bot.py            — interactive control plane. This is the
#                                   working Alpaca-paper auto-trade surface and
#                                   it OWNS the RL advisor, the shadow recorder,
#                                   and the risk engine (all env-driven), so
#                                   "RL" is not a separate process to start.
#
# Auto-trade scheduler (opt-in, OFF by default):
#   run_alpaca_intraday.py        — Alpaca-native intraday auto-entry scheduler
#                                   for SPY/QQQ. It reuses smart_trader's full
#                                   pipeline (risk engine, PDT, shadow/RL) and is
#                                   DRY-RUN BY DEFAULT (logs trades it would place,
#                                   places nothing). Enable the daemon with
#                                   ENABLE_SCHEDULER=1; actually place paper orders
#                                   only with SCHEDULER_ARMED=1.
#   run_spy_*_daily.py            — LEGACY Schwab schedulers (dead OAuth stack,
#                                   trade via Schwab not Alpaca). Selectable via
#                                   SCHEDULER=run_spy_1dte_daily.py but unsupported.
#
# The screener (stock_screener.py) is NOT a daemon; run it on demand with
# `./run.sh screen`. It is standalone and is not auto-wired into live trades.
#
# Usage:
#   ./run.sh start            # start the Telegram bot (default if no arg)
#   ./run.sh stop             # stop managed daemons
#   ./run.sh restart          # stop then start
#   ./run.sh status           # show PID / running state
#   ./run.sh screen           # run the screener once (writes tickers file)
#
# Env overrides:
#   PYTHON=python3            # interpreter (a ./venv or ./.venv is preferred)
#   VENV=venv                 # explicit virtualenv dir
#   NO_BOT=1                  # do not start the Telegram bot
#   ENABLE_SCHEDULER=1        # also start the Alpaca intraday scheduler (dry-run)
#   SCHEDULER_ARMED=1         # let the scheduler place paper orders (else dry-run)
#   SCHEDULER=run_alpaca_intraday.py  # which scheduler script to start
#
set -euo pipefail

# --- config (override via env) --------------------------------------------
PYTHON="${PYTHON:-python3}"
SCHEDULER="${SCHEDULER:-run_alpaca_intraday.py}"

# Run from the repo root regardless of the caller's cwd.
cd "$(dirname "$0")"
ROOT="$PWD"
LOG_DIR="$ROOT/logs"
PID_DIR="$ROOT/run"
mkdir -p "$LOG_DIR" "$PID_DIR"

# Fall back to `python` if `python3` is not on PATH.
if ! command -v "$PYTHON" >/dev/null 2>&1; then
    PYTHON=python
fi

# Prefer a project virtualenv if one exists (matches deploy.sh).
for _venv in "${VENV:-}" .venv venv; do
    if [ -n "$_venv" ] && [ -x "$_venv/bin/python" ]; then
        PYTHON="$ROOT/$_venv/bin/python"
        break
    fi
done

# --- helpers ---------------------------------------------------------------
# A "service" is named; we keep <name>.pid and <name>.log alongside it.
pid_file() { echo "$PID_DIR/$1.pid"; }
log_file() { echo "$LOG_DIR/$1.log"; }

is_running() {
    # is_running <name> -> 0 if a live PID is recorded, else 1.
    local pf; pf="$(pid_file "$1")"
    [ -f "$pf" ] || return 1
    local pid; pid="$(cat "$pf" 2>/dev/null || true)"
    [ -n "$pid" ] || return 1
    kill -0 "$pid" 2>/dev/null
}

start_one() {
    # start_one <name> <script.py>
    local name="$1" script="$2"
    if [ ! -f "$script" ]; then
        echo "==> SKIP $name: $script not found"
        return 0
    fi
    if is_running "$name"; then
        echo "==> $name already running (PID $(cat "$(pid_file "$name")"))"
        return 0
    fi
    local lf; lf="$(log_file "$name")"
    echo "==> Starting $name ($script) -> $lf"
    # setsid so the child survives this shell / SSH disconnect; nohup as a
    # fallback if setsid is unavailable.
    if command -v setsid >/dev/null 2>&1; then
        setsid "$PYTHON" -u "$script" >>"$lf" 2>&1 < /dev/null &
    else
        nohup "$PYTHON" -u "$script" >>"$lf" 2>&1 < /dev/null &
    fi
    echo "$!" > "$(pid_file "$name")"
    # Brief liveness check so an instant crash is reported now, not silently.
    sleep 1
    if is_running "$name"; then
        echo "    started (PID $(cat "$(pid_file "$name")"))"
    else
        echo "    FAILED to stay up — last log lines:"
        tail -n 15 "$lf" 2>/dev/null | sed 's/^/      /' || true
        return 1
    fi
}

stop_one() {
    # stop_one <name>
    local name="$1" pf; pf="$(pid_file "$name")"
    if ! is_running "$name"; then
        echo "==> $name not running"
        rm -f "$pf"
        return 0
    fi
    local pid; pid="$(cat "$pf")"
    echo "==> Stopping $name (PID $pid)"
    kill "$pid" 2>/dev/null || true
    # Wait up to 10s for a graceful exit, then SIGKILL.
    for _ in $(seq 1 10); do
        is_running "$name" || break
        sleep 1
    done
    if is_running "$name"; then
        echo "    still alive; sending SIGKILL"
        kill -9 "$pid" 2>/dev/null || true
    fi
    rm -f "$pf"
}

status_one() {
    # status_one <name>
    local name="$1"
    if is_running "$name"; then
        echo "  $name: RUNNING (PID $(cat "$(pid_file "$name")"))"
    else
        echo "  $name: stopped"
    fi
}

# --- which services to manage ---------------------------------------------
# Names are stable handles for PID/log files; scripts may be swapped via env.
BOT_NAME="telegram_bot"
SCHED_NAME="scheduler"

require_env() {
    if [ ! -f "$ROOT/.env" ]; then
        echo "==> ERROR: .env not found in $ROOT"
        echo "    The bot needs credentials (ALPACA_API_KEY, TELEGRAM_BOT_TOKEN, ...)."
        echo "    Copy .env.example to .env and fill it in before starting."
        exit 1
    fi
    # Load .env into the environment so launch settings (ENABLE_SCHEDULER,
    # SCHEDULER_ARMED, NO_BOT, ...) can live there instead of being passed
    # inline. Variables already set in the shell take precedence (set -a only
    # exports; existing values passed like `ENABLE_SCHEDULER=1 ./run.sh` are
    # not overwritten because we skip keys already present in the environment).
    set -a
    while IFS= read -r _line || [ -n "$_line" ]; do
        case "$_line" in
            ''|\#*) continue ;;            # skip blanks and comments
            *=*)
                _key=${_line%%=*}
                # Only set if not already defined in the environment.
                if [ -z "${!_key+x}" ]; then
                    eval "$_line"
                fi
                ;;
        esac
    done < "$ROOT/.env"
    set +a
}

cmd_start() {
    require_env
    echo "==> Launcher: python=$PYTHON"
    if [ "${NO_BOT:-0}" != "1" ]; then
        start_one "$BOT_NAME" "telegram_bot.py"
    fi
    # Auto-trade scheduler is OFF unless explicitly enabled.
    if [ "${ENABLE_SCHEDULER:-0}" = "1" ]; then
        if [ "${SCHEDULER_ARMED:-0}" = "1" ]; then
            echo "==> ENABLE_SCHEDULER=1 SCHEDULER_ARMED=1: scheduler will PLACE paper orders ($SCHEDULER)"
        else
            echo "==> ENABLE_SCHEDULER=1: starting scheduler in DRY-RUN ($SCHEDULER)"
            echo "    NOTE: set SCHEDULER_ARMED=1 to actually place paper orders."
        fi
        start_one "$SCHED_NAME" "$SCHEDULER"
    fi
    echo "==> Done. Use './run.sh status' to check, './run.sh stop' to stop."
}

cmd_stop() {
    stop_one "$BOT_NAME"
    stop_one "$SCHED_NAME"
}

cmd_status() {
    echo "==> Services:"
    status_one "$BOT_NAME"
    if [ "${ENABLE_SCHEDULER:-0}" = "1" ] || is_running "$SCHED_NAME"; then
        status_one "$SCHED_NAME"
    fi
    echo "==> Logs in: $LOG_DIR"
}

cmd_screen() {
    require_env
    # One-shot screener run: refresh the moved-stock tickers + EV scores.
    # Not a daemon — returns when the scan completes.
    echo "==> Running screener once (python=$PYTHON)"
    exec "$PYTHON" -u stock_screener.py -s moved -n 10 --score ev --write-tickers
}

# --- dispatch --------------------------------------------------------------
case "${1:-start}" in
    start)   cmd_start ;;
    stop)    cmd_stop ;;
    restart) cmd_stop; cmd_start ;;
    status)  cmd_status ;;
    screen)  cmd_screen ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|screen}" >&2
        exit 2
        ;;
esac
