# Oracle Runbook — run, train, execute (no Claude required)

Everything you need to operate this system by hand. All commands assume
**repo root** and **Git Bash** on Windows (or plain bash on Linux).

---

## 0. One-time setup

```bash
python -m venv venv && source venv/Scripts/activate   # (Linux: venv/bin/activate)
pip install -r requirements.txt
cp .env.example .env   # if present — otherwise create .env by hand
```

Required `.env` keys (minimum):

| Key | Purpose |
| --- | --- |
| `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` | Alpaca **paper** account credentials |
| `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` | Telegram alerts + command bot |
| `SCHEDULER_SYMBOLS` | e.g. `SPY,QQQ` (default) |
| `SCHEDULER_ARMED` | `1` = scheduler may actually place paper orders |
| `DASHBOARD_BASIC_AUTH_USER` / `..._PASS` | dashboard login (optional locally) |

Sanity checks (each prints PASS/FAIL and exits):

```bash
python oracle_dashboard.py --selftest
python run_alpaca_intraday.py --selftest
```

---

## 1. RUN — the services

### Local (your PC)

| What | Command |
| --- | --- |
| Everything (bot + scheduler), supervised | `bash run.sh start` — then `status` / `stop` / `restart` |
| Telegram bot only (Windows double-click) | `start_bot.bat` |
| Dashboard | `bash scripts/run_dashboard.sh` → http://127.0.0.1:8787 |
| Stock screener (writes tickers.txt) | `bash run.sh screen` |

`run.sh` writes PID files to `run/` and logs to `logs/`. It automatically
defers to systemd if the server units are active, and sweeps stray Telegram
pollers so you never get HTTP 409 double-polling.

### Server (147.182.242.177, `/var/www/alps`)

Three systemd units run 24/7:

| Unit | Runs | Log |
| --- | --- | --- |
| `alps-bot` | `telegram_bot.py` (alerts, commands, EOD summary) | `/var/log/alps-bot.log` |
| `alps-scheduler` | `run_alpaca_intraday.py` (intraday scan/trade loop) | `journalctl -u alps-scheduler` |
| `alps-dashboard` | `oracle_dashboard.py` → https://oracle.lasgidi.ng | `journalctl -u alps-dashboard` |

```bash
ssh -i ~/.ssh/id_ed25519_1 root@147.182.242.177
systemctl status alps-bot alps-scheduler alps-dashboard
```

**Deploy a new version** (server is a git clone — never edit files on it):

```bash
# locally: commit + push, note the sha
ssh -i ~/.ssh/id_ed25519_1 root@147.182.242.177
cd /var/www/alps && git fetch && git checkout <sha>
systemctl restart alps-dashboard      # restart ONLY the unit(s) you changed
```

---

## 2. EXECUTE — trading

The scheduler is the trade executor. It is **dry-run by default**; nothing is
sent to Alpaca unless armed.

```bash
python run_alpaca_intraday.py --once       # one scan cycle, dry-run, prints decisions
python run_alpaca_intraday.py --dry-run    # continuous loop, no orders
python run_alpaca_intraday.py --armed      # continuous loop, PLACES PAPER ORDERS
```

Arming can also be done via env: `SCHEDULER_ARMED=1`. Key knobs (env):
`SCAN_INTERVAL_MIN` (15), `MAX_NEW_TRADES_PER_DAY` (4), `SCHEDULER_QTY`,
`ENTRY_START_ET`.

From Telegram (while `alps-bot` runs): the bot announces picks, fills, exits,
and the daily Oracle report; it accepts the status/report commands registered
in `telegram_bot.py`.

Every decision (traded or blocked) is logged to `episodes.db` and visible at
https://oracle.lasgidi.ng/decisions (funnel, agent contributions, decision log).

---

## 3. TRAIN — the two learners

### 3a. RL Q-table (lightweight, in the live decision path)

```bash
bash scripts/train_rl.sh                 # fetch live episodes.db → train → report
bash scripts/train_rl.sh --skip-fetch --epochs 5
EPISODES_DB=server_data/2026-11-19/episodes.db bash scripts/train_rl.sh --skip-fetch
```

Output: `rl_qtable.json`. Deploying it is deliberate and manual:

```bash
scp -i ~/.ssh/id_ed25519_1 rl_qtable.json root@147.182.242.177:/var/www/alps/rl_qtable.json
ssh -i ~/.ssh/id_ed25519_1 root@147.182.242.177 'systemctl restart alps-scheduler alps-bot'
```

### 3b. Oracle4 ML pipeline (heavy, research-side, never auto-promotes)

```bash
bash scripts/train_oracle4.sh            # fetch ledger → dataset → train → value → replay → report
bash scripts/train_oracle4.sh --skip-fetch --tag v3
```

Artifacts land in `artifacts/oracle4/` (git-ignored). The pipeline ends with a
read-only promotion report; expect `INSUFFICIENT_EVIDENCE` until forward shadow
evidence accumulates. Promotion always requires a human:

```bash
python -m oracle.ml.registry register --model-id mlp_<tag> --kind neural --artifact artifacts/oracle4/mlp_<tag>.json
python -m oracle.ml.registry approve  --model-id mlp_<tag> --approved-by <you>
```

### 3c. One-click from the dashboard

Set `DASHBOARD_ENABLE_TRAINING=1` and `DASHBOARD_TRAINING_LEDGER=<path>` in the
dashboard environment; a Train button appears in the ML section. It runs the
same pipeline — promotion still requires the manual registry steps above.

GPU note: training auto-detects CUDA (`--device auto`). Train on your local
GPU box; the server is CPU-only and should not train.

---

## 4. VERIFY — tests and live checks

```bash
# full suite (~95s). 4 known failures are expected and pre-existing:
#   TestTelegramDispatch::test_eight_commands_dispatch, TestEVEntryGate x2,
#   TestPhase4GreekFallback::test_portfolio_check_blocks_via_trader
python -m pytest --ignore=tests/test_safety.py -q
```

Live smoke (replace `USER:PASS` with the dashboard basic-auth pair):

```bash
curl -su USER:PASS https://oracle.lasgidi.ng/api/health
curl -su USER:PASS https://oracle.lasgidi.ng/api/decision-funnel
curl -su USER:PASS "https://oracle.lasgidi.ng/api/decisions?limit=3&filter=traded"
```

Telegram smoke: check `/var/log/alps-bot.log` for recent `[TELEGRAM]` lines, or
send `getMe` with the bot token.

---

## 5. File map (what not to touch)

| File/dir | Role |
| --- | --- |
| `episodes.db` | every decision + outcome (server-side source of truth) |
| `research_ledger.db` | Oracle4 research ledger (server-side) |
| `rl_qtable.json` | live RL policy — replaced only via §3a |
| `artifacts/`, `server_data/` | git-ignored, regenerable — never commit |
| `candidate_resolution.jsonl`, `orderbook_shadow.jsonl`, `DEPLOYMENT_STATUS.md` | runtime/local state — never commit |
| `docs/ORACLE_PAPER.md` | 1-page system/profitability paper |
| `docs/oracle4_runbook.md` | deep-dive on the ML pipeline flags |
