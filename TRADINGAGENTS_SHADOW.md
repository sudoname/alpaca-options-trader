# TradingAgents Shadow Experiment (pre-registered)

Status: **SHADOW ONLY — zero trade influence.** Registered before any data
was collected. Results are UNTESTED until the gate below passes.

## Hypothesis

Daily BUY/SELL/HOLD decisions from the TradingAgents multi-agent LLM
framework (Tauric Research, arXiv 2412.20138) carry directional information
about forward returns of the underlying.

## Why forward-only (no backtest)

LLMs plausibly memorized historical price paths during pretraining, so any
historical "backtest" of LLM decisions is contaminated by construction. The
only honest evaluation is decisions recorded **before** outcomes exist. The
paper's reported cumulative-return/Sharpe improvements cannot be separated
from this contamination; we treat them as unverified marketing.

## Mechanics

- `tradingagents_shadow.py --scan` (cron, pre-market trading days): one
  decision per universe symbol via `TradingAgentsGraph.propagate(sym, today)`,
  appended to `tradingagents_shadow.jsonl` — one row per scanned symbol,
  including errors (no survivorship). Cheapest viable models
  (`TA_DEEP_MODEL`/`TA_QUICK_MODEL`, 1 debate round) to cap cost.
- `--eval` (cron, weekly): joins decisions with official 'all'-adjusted daily
  closes (Alpaca) STRICTLY AFTER the decision date. Base price = close of the
  first trading day at/after the decision date; forward horizons 1d and 5d
  (trading days). Writes `tradingagents_eval.json`.
- `--status`: ledger counts and gate progress.
- Telegram `RESEARCH <ticker>` (flag `ENABLE_TRADINGAGENTS_RESEARCH`): sends
  the agent debate to the human. Decision support only; never a gate input.
- TradingAgents runs in a **separate venv** (`TA_PYTHON`) so its heavy
  dependency tree cannot disturb the trading venv.

## Pre-registered evidence gate (fixed before data collection)

- **Primary metric:** mean signed 5-trading-day forward return over non-HOLD
  decisions, signed BUY = +1, SELL = −1.
- **Gate:** n ≥ 40 non-HOLD decisions AND mean > 0 AND t ≥ 2
  (t = mean / (sd/√n), ddof=1).
- **Label rule (fixed):** first occurrence of BUY/SELL/HOLD in the decision
  text; anything else is UNKNOWN and excluded.
- Secondary (descriptive only, no gate): per-label 1d/5d table.
- No threshold tuning mid-sample. If the metric is changed, the sample
  restarts.

## What passing would and would not establish

Passing: the decisions correlated with 5d forward returns in this sample at
conventional significance. It would justify ONLY a next, separately gated
step (e.g. agreement analysis against the live profitability gate) — never a
direct wire into sizing or orders.

NOT established even if passed: profitability after options spreads/fees,
stability across regimes, causality, or that the LLM adds information beyond
momentum already visible in its inputs.

## Ops

```bash
# one-time install (separate venv; needs an LLM API key, e.g. OPENAI_API_KEY)
python3 -m venv venv-ta && venv-ta/bin/pip install git+https://github.com/TauricResearch/TradingAgents

# cron (server, UTC): scan pre-market, eval weekly
# 30 13 * * 1-5  cd /var/www/alps && ./venv-ta/bin/python tradingagents_shadow.py --scan  >> /var/log/alps-ta.log 2>&1
# 0  22 * * 5    cd /var/www/alps && ./venv/bin/python   tradingagents_shadow.py --eval  >> /var/log/alps-ta.log 2>&1

./venv/bin/python tradingagents_shadow.py --status   # progress any time
```

Flags default OFF. Without `ENABLE_TRADINGAGENTS_SHADOW=1`, `--scan` writes
nothing. Without the provider API key, `--scan` logs the reason and no-ops.
