# Oracle 4.0.2 — Gap Audit and Closure Report

**Historical snapshot:** The GBM/ensemble replay numbers below predate the
strict missing-prediction gate. The corrected variants require a fresh
backtest and promotion report; no profitability claim carries forward.

Branch: `oracle4.0.2` (base: `oracle4.0.1` @ `a373a36`)
Scope: close the eight architectural/profitability gaps identified after the 4.0.1
forward-evidence period, then prove (or honestly refuse to claim) positive
out-of-sample net expectancy after realistic execution costs.

## Starting baselines (measured on 4.0.1, research path)

| Metric | Value |
|---|---|
| Production win rate | 35.82% |
| Production profit factor | 0.51 |
| Expectancy per trade | −$54.69 |
| 4.0.1 classical challenger PF | 0.81 |
| Stamped EV ↔ realized P&L correlation | −0.07 |
| EV calibration slope | −0.266 |

The system was losing money and its own EV stamp was *anti-correlated* with
realized P&L. Every gap below traces back to one of those two facts.

## Gap-by-gap closure

### P0-1 — Trained models were never connected to valuation (`e807100`)
**Gap:** `ORACLE4_MODEL_ID` stamped registry metadata onto shadow records but the
valuation pipeline still bootstrapped scenarios from raw history; trained model
output never influenced a single ranked candidate.
**Fix:** `oracle/ml/valuation.py` gains forecast-conditioned scenario generation
(mode `forecast_conditioned`) next to the frozen `baseline_bootstrap_ewma` mode.
Challenger models auto-register; the mode is stamped into every valuation
artifact so no result can be mistaken for the other path.
**Files:** `oracle/ml/valuation.py`, `scripts/train_oracle4.sh`, `tests/test_oracle402_p01.py`.

### P0-2 — Synthetic market history (`fc2583d`)
**Gap:** when the bars DB was missing, shadow/forecast code fabricated history
(interpolated/synthetic bars), contaminating every downstream label.
**Fix:** real bars are the only history source. A missing/unset bars DB produces a
structured abstention (`BARS_DB_UNSET`), never fabricated data. All synthetic
fallbacks deleted (net −167 lines of fabrication paths).
**Files:** `oracle/ml/forecast.py`, `oracle/ml/shadow.py`, `run_alpaca_intraday.py`, 4 test files.

### P0-3 — Fuzzy trade/outcome attribution (`ff8c565`)
**Gap:** decisions, orders, fills and realized trades were joined by symbol+time
heuristics; re-ingested history could double-count P&L.
**Fix:** `oracle/ml/attribution.py` — deterministic
`make_client_order_id` = `o4-{sha256(decision_id)[:32]}-a{attempt}` (≤48 chars,
retry-distinct), stable `trade_identity`/`dedupe_trades`, and
`ExclusiveTradeMatcher` (one outcome row claims at most one trade, exactly once).
**Files:** `oracle/ml/attribution.py`, `oracle/ml/forward.py`, `tests/test_oracle402_p03.py`.

### P0-4 — Broken EV calibration (`dd8e169`)
**Gap:** stamped EV had slope −0.266 and correlation −0.07 against realized P&L —
worse than useless as a ranking signal.
**Fix:** `oracle/ml/ev_models.py` — a competing-model suite (base rate, logistic
recalibration, LightGBM PoP/profit/quantile, MLP profit, cohort shrinkage) fit on
prior sessions only, scored strictly OOS, with an explicit `challenger_policy`
("abstain" unless evidence suffices). Measured result: the stamped EV scored
Spearman **−0.661** OOS while `lgbm_quantile_q50` scored **+0.564** — but on only
10 eval opportunities, so the module verdict is `INSUFFICIENT_EVIDENCE` and the
policy abstains (see experiments doc §3).
**Files:** `oracle/ml/ev_models.py`, `tests/test_oracle402_p04.py`.

### P1-7 — Fragmented forecast interfaces (`3f7cd6d`)
**Gap:** classical, neural and baseline forecasters each had bespoke call shapes;
consumers special-cased every one.
**Fix:** unified `ForecastProvider` registry in `oracle/ml/forecast.py` with
checksum-chained model identity (id/kind/checksums/git SHA), structured
`ForecastUnavailable` abstention, classical/GBM/baseline/ensemble providers
behind one interface.
**Files:** `oracle/ml/forecast.py`, `oracle/ml/train.py`, `scripts/train_oracle4.sh`, 2 test files.

### P1-5 — Training data starvation (`2ce3aa4`)
**Gap:** models trained on a handful of live ledger sessions.
**Fix:** `oracle/ml/bars_backfill.py` (multi-year Alpaca bars backfill, read-only
market-data GETs, forming bars never inserted) + `oracle/ml/bars_dataset.py`
(bars → point-in-time synthesized decision records → the SAME dataset store
schema, so purged walk-forward loaders/training/registry work unchanged).
Local bar store now carries SPY 1-min history 2008–2021 plus 2024–2026.
**Files:** `oracle/ml/bars_backfill.py`, `oracle/ml/bars_dataset.py`, `tests/test_oracle402_p15.py`.

### P1-6 — No external information source (`df87d84`)
**Gap:** no prediction-market signal; no way to test whether one helps.
**Fix:** `oracle/ml/kalshi/` package — client, snapshot collector, market mapping,
strictly point-in-time `feature_builder` (observed_at ≤ as_of enforced in SQL,
missing = None, never fabricated), and a matched ablation harness (experiments
A–E incl. a permuted placebo and lagged-staleness arm). Live snapshot
collection has not yet accumulated data, so the ablation has no real-data
result yet — see experiments doc §4.
**Files:** `oracle/ml/kalshi/{__init__,client,collector,feature_builder,market_mapping,validation}.py`, `tests/test_oracle402_p16.py`.

### P1-8 — Research work blocking the trading loop (`e185805`)
**Gap:** shadow scans and bars feeding ran inline in the scheduler; a slow
research task could delay exits/order placement.
**Fix:** `oracle/ml/research_worker.py` — bounded queue (maxsize 8, drop-newest
backpressure), deepcopy snapshots at submit, failure isolation, 120 s soft
timeout flagging, atomic status JSON; opt-in via `ENABLE_ORACLE4_ASYNC_RESEARCH`
(default off — production behavior unchanged).
**Files:** `oracle/ml/research_worker.py`, `oracle_dashboard.py`, `run_alpaca_intraday.py`, `tests/test_oracle402_p18.py`.

## Validation gates (P10/P11)

* **P10 (`8d6a416`)** — `oracle/ml/backtest402.py`: comprehensive replay of 9
  variants (frozen baseline, quality, valuation-ranked, classical/neural/GBM/
  ensemble gated, combined) with friction sweep (1.0×/1.5×/2.0×), stop-overshoot
  stress, paired session bootstrap vs the frozen baseline, dual matched cohorts,
  and a parity check against the frozen 4.0.1 replayer (ok, n=690,
  max_abs_diff=0.0). Artifact fingerprint `bc9647af…`.
* **P11 (`88d23c2`)** — `oracle/ml/promotion402.py`: frozen promotion criteria
  (positive OOS expectancy, positive paired incremental with CI LB > 0,
  PF > 1.3, Sharpe > 1.0, survives 1.5× friction, ≥60 sessions, ≥200 resolved
  opportunities, ≥100 fills), closed verdict set, **never auto-promotes** —
  best possible outcome is `ELIGIBLE_PENDING_HUMAN_REVIEW`.

## Honest bottom line

**Overall promotion verdict: `INSUFFICIENT_EVIDENCE`** (report fingerprint
`6394a806…`). The best challenger (`gbm_gated`) cut the frozen baseline's loss
by ~89% (net −$333 vs −$3,138 over the window) with a paired per-session delta
of **+$400.63, 95% CI [1.29, 905.29]** and a matched-cohort delta of
**+$751.19, 95% CI [118.69, 1536.59]** — a real positive incremental signal —
but the sample is 10 replay sessions / 35 fills against required 60 / 100, no
variant has positive absolute expectancy, and `baseline_quality` is
`DO_NOT_PROMOTE` (confidently worse than baseline). No production flag was or
will be flipped by this work; promotion requires the forward-evidence period
described in the runbook plus a human decision.

## Commit chain

| Phase | Commit | Summary |
|---|---|---|
| P0-1 | `e807100` | wire trained models into valuation |
| P0-2 | `fc2583d` | eliminate synthetic market history |
| P0-3 | `ff8c565` | exact trade/outcome attribution |
| P0-4 | `dd8e169` | competing EV models + calibrated executable EV |
| P1-7 | `3f7cd6d` | unified forecast provider interface |
| P1-5 | `2ce3aa4` | multi-year bars backfill + bars-based dataset |
| P1-6 | `df87d84` | Kalshi integration + matched ablation harness |
| P1-8 | `e185805` | asynchronous research worker |
| P10 | `8d6a416` | comprehensive backtest harness |
| P11 | `88d23c2` | promotion gate |

Test status after every phase: full suite green except the 4 known pre-existing
failures unrelated to this work (Telegram dispatch ×1, EV entry gate ×2,
Phase-4 greek fallback ×1); final count 2,738 passed.
