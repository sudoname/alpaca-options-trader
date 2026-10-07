# IV-RV Credit-Spread Paper Experiment — Protocol & Evidence Gate

Status: **PAPER EXPERIMENT, UNTESTED**. Nothing here touches the live order
path. Every position is simulated by `spread_paper_trader.SpreadPaperTrader`
(no broker client, no order submission exists anywhere in the spread code).

## 1. Hypothesis (pre-registered before any results exist)

Defined-risk CREDIT spreads (bull put / bear call / iron condor), opened only
when ATM implied volatility is RICH versus trailing realized volatility
(IV/RV >= 1.25), have positive expected net P&L per dollar of max loss —
net of paying the FULL quoted spread at entry.

Why this and not something else — the literature triangulates here:
- Goyal & Saretto (JFE 2009): the RV−IV cross-section predicts option
  returns; selling expensive vol is the documented premium.
- Bakshi & Kapadia (RFS 2003), Coval & Shumway (JF 2001): the variance risk
  premium makes long premium structurally negative-EV.
- Bryzgalova, Pavlova & Sikorskaya (JF 2023): retail losses concentrate in
  exactly what this bot currently does live (single-leg buys, short-dated);
  Beckmeyer, Branger & Gayda (0DTE): multi-leg premium-collecting trades are
  the profitable cell.
- Our own ledger: live single-leg mean net P&L ≈ −15%/trade, win rate 37.5%.

## 2. Mechanics (`iv_rv_experiment.py`, flag `ENABLE_IVRV_CREDIT_PAPER`, default OFF)

- **RV**: trailing annualized vol of log returns from 'all'-adjusted daily
  closes (window 252, min 60 obs) — same bar machinery as the session work.
- **IV**: ATM call snapshot IV on the nearest expiration — the same IV source
  `smart_trader.propose_spread` uses, so the gate and the structure agree.
- **Gate**: IV/RV >= `IVRV_MIN_RATIO` (1.25). This is the ONLY experiment
  filter (`IVRV_MIN_ORACLE_SCORE` defaults to 0 so the oracle score cannot
  confound the tested variable; it is still recorded for later analysis).
- **Structure**: whatever `propose_spread` builds, accepted ONLY if it is a
  credit strategy; debit structures are skipped and the skip is recorded.
- **Entry fills (conservative)**: SELL legs at BID, BUY legs at ASK — the
  simulation pays the full quoted spread. Muravyev & Pearson show real
  effective spreads are < 40% of quoted with timing, so this is a lower
  bound on realizable edge. If any crossed-side quote is missing, no trade.
- **Exit**: held to expiration; every leg settles at INTRINSIC value from the
  official raw daily close of the underlying on expiration day. No early
  management (a management overlay would be a separate, later hypothesis).
- **Recording**: one JSONL row per SCANNED symbol per run
  (`iv_rv_scan_ledger.jsonl`), opened or not, so evaluation runs over the
  full candidate set — no survivorship in the ledger. Positions/trades live
  in dedicated files (`iv_rv_paper_positions.json` / `iv_rv_paper_trades.json`),
  isolated from the legacy spread paper flow.

## 3. Evidence gate (decided in advance; `--status` computes it)

The hypothesis is SUPPORTED only when ALL of:
1. **n >= 40 settled trades** (not open MTM — settled at intrinsic only);
2. **mean net P&L per $ max-loss > 0**;
3. **t-stat >= 2** on that mean (se from the settled-trade sample).

Until then every number this experiment produces is labeled UNTESTED.
If the gate fails after ~40–60 settled trades, the conclusion is recorded and
the experiment stops — no threshold-tuning on the same sample (that would be
in-sample fitting; any revised rule needs a fresh sample).

What passing the gate does NOT establish:
- that live fills match the crossed-quote assumption (it is conservative at
  entry but assumes exact intrinsic settlement and zero early-assignment
  cost on American-style single names — pin/assignment risk is unmodeled);
- regime robustness (the sample will span only the collection window);
- capacity beyond 1-contract structures.

A pass therefore graduates the idea to a LIVE paper-account stage with real
Alpaca paper fills, not to real-money execution.

## 4. Operations

```
# scan (weekday mid-morning ET; after the open settles)
python iv_rv_experiment.py --scan
# settle (weekday after the close, >= 16:30 ET so the official close exists)
python iv_rv_experiment.py --settle
# evidence-gate summary
python iv_rv_experiment.py --status
```

Env (all optional; defaults shown):
`ENABLE_IVRV_CREDIT_PAPER=0`, `IVRV_MIN_RATIO=1.25`, `IVRV_RV_WINDOW=252`,
`IVRV_MIN_RV_OBS=60`, `IVRV_MAX_OPENS_PER_RUN=3`, `IVRV_MIN_ORACLE_SCORE=0`,
`IVRV_UNIVERSE=SPY,QQQ,AAPL,MSFT,NVDA,AMZN,META,GOOGL,TSLA,AMD`
(liquid names with tight chains, per the execution literature).

Safety: flag OFF → `--scan`/`--settle` do nothing and write nothing; the
module is never imported by `run_alpaca_intraday.py` or any live path.
