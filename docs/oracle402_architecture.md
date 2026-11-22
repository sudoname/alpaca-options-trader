# Oracle 4.0.2 — Architecture

All new 4.0.2 code is **research-path only**: nothing in the production trading
path imports it unless an explicit opt-in flag is set, and no module in this
release can flip a production flag.

## Data flow (end to end)

```
Alpaca market data (bars, completed only)
        │
        ▼
bars_backfill.py ──► BarStore sqlite (artifacts/oracle4/bars/bars.sqlite)
  (historical)            ▲
bars_feed.py ─────────────┘ (live, forming bars never inserted)
        │
        ▼
bars_dataset.py ──► dataset store sqlite (same schema as ledger datasets)
        │                              e.g. artifacts/oracle4/dataset_bt1121.sqlite
        ▼
train.py (purged walk-forward) ──► model registry (registry.json, checksummed)
        │
        ▼
forecast.py  — unified ForecastProvider registry
  classical / gbm / neural / baseline / ensemble, structured abstention
        │
        ▼
valuation.py — scenario valuation + ranking
  modes: baseline_bootstrap_ewma (frozen) | forecast_conditioned (challenger)
        │
        ▼
backtest402.py — 9-variant replay, friction sweep, stress, paired bootstrap
        │              artifacts/oracle4/replay402.json (fingerprinted)
        ▼
promotion402.py — frozen criteria, closed verdicts, NEVER auto-promotes
                       artifacts/oracle4/promotion402.json (fingerprinted)
```

Sidecars feeding the same research path:

```
attribution.py   decisions ↔ orders ↔ fills ↔ realized trades (exclusive joins)
ev_models.py     competing EV models, OOS-scored, abstain-by-default policy
kalshi/          PIT snapshot store → kalshi_* features → matched ablation A–E
research_worker.py  bounded async queue isolating research from the scheduler
```

## Module reference

### `oracle/ml/attribution.py` (P0-3)
Deterministic identity layer. `make_client_order_id(decision_id, attempt)` →
`o4-{sha256[:32]}-a{n}` (≤48 chars, broker-safe, retry-distinct).
`trade_identity` prefers broker `order_id`, falls back to a content fingerprint.
`ExclusiveTradeMatcher` enforces one-outcome-one-trade claims; `dedupe_trades`
makes re-ingested history idempotent.

### `oracle/ml/forecast.py` (P0-2 + P1-7)
`ForecastProvider` registry with typed `Forecast` results (model id/kind/
checksums/git SHA, horizon, direction probabilities, return quantiles, RV,
barrier CIF, calibration provenance, missingness accounting) and
`ForecastUnavailable` structured abstention. No bars DB ⇒ `BARS_DB_UNSET`
abstention — fabrication paths were deleted, not guarded.

### `oracle/ml/valuation.py` (P0-1)
Per-scan past-only circular block bootstrap of underlying log returns; one
scenario matrix shared by all candidates in a scan (common random numbers).
`forecast_conditioned` mode tilts scenario generation by provider output; the
mode is stamped in every artifact.

### `oracle/ml/ev_models.py` (P0-4)
Chronological train/eval split with hard leak guard. Models: `base_rate`,
`logistic_recal`, `lgbm_pop`, `lgbm_profit`, `lgbm_quantile` (q10/q50/q90),
`neural_profit`, `cohort_pop`/`cohort_profit` (regime|side|dte shrinkage).
Emits `verdict` + `challenger_policy` ("abstain" below evidence floor of 50
eval opportunities). CLI: `python -m oracle.ml.ev_models --history … --out …`.

### `oracle/ml/bars_backfill.py` / `bars_dataset.py` (P1-5)
Backfill: strictly read-only market-data GETs, forming bars never inserted,
per-ingest manifests. Dataset builder: bars → `UnderlyingObs` (event_at =
available_at = bar end) → synthesized `DecisionRecord`s (`kind="bars_synthetic"`)
at sampled RTH times → identical feature/label/outcome pipeline →
`dataset._write_store` + manifest with `dataset_hash`.

### `oracle/ml/kalshi/` (P1-6)
`client` (REST, read-only), `collector` (snapshot store; no settlement column —
settled outcomes structurally cannot leak), `market_mapping` (SPY/QQQ ↔ Kalshi
tickers), `feature_builder` (PIT in SQL: `observed_at <= as_of` within
`max_staleness_min`; missing ⇒ None, fail-open), `validation` (matched ablation
A base / B base+kalshi / C kalshi_only / D base+placebo-permuted /
E base+lagged; identical samples, folds, learner, hyperparameters).

### `oracle/ml/research_worker.py` (P1-8)
`queue.Queue(maxsize=8)`, drop-newest on full (counted, `submit` never blocks),
`copy.deepcopy` payload snapshots, per-task failure isolation, 120 s soft
timeout flag, atomic status JSON for the dashboard. Opt-in:
`ENABLE_ORACLE4_ASYNC_RESEARCH` (default off).

### `oracle/ml/backtest402.py` (P10)
Builds on the **frozen** 4.0.1 replayer (`oracle/ml/replay.py` is imported,
never modified; parity is asserted every run). Adds:
* `compute_oof_gbm` — purged walk-forward LightGBM (native-NaN handling,
  last-train-session validation, fixed-budget fallback, honest stub when
  lightgbm is unavailable);
* `ensemble_preds` — intersection of model predictions, renormalized mean;
* `build_variants_402` — the replay's 6 variants + `gbm_gated`,
  `ensemble_gated`, `gbm_combined`. Frozen replay semantics are preserved:
  `valuation_ranked`/`combined` keys are **omitted** (not stubbed) when no
  valuation artifact is supplied;
* friction sweep (1.0×/1.5×/2.0×), stop-overshoot stress, paired session
  bootstrap vs `baseline_frozen`, dual matched cohorts, per-DTE/moneyness
  breakdowns (unconstrained cells; regime honestly reported unavailable);
* input crosschecks that **refuse to mix artifacts** (raises on fingerprint
  mismatch) and a sha256 artifact fingerprint.

### `oracle/ml/promotion402.py` (P11)
Reads `replay402.json`; scores each scoreable challenger against frozen
criteria:

| ID | Criterion | Threshold |
|---|---|---|
| C1 | expectancy/trade @1.0× | > 0 |
| C2 | paired bootstrap mean vs frozen baseline | > 0 |
| C3 | paired 95% CI lower bound | > 0 |
| C4 | profit factor @1.0× | > 1.3 |
| C5 | annualized daily Sharpe @1.0× | > 1.0 |
| C6 | net P&L @1.5× friction | > 0 |
| E1 | replay sessions | ≥ 60 |
| E2 | resolved opportunities | ≥ 200 |
| E3 | executed fills @1.0× | ≥ 100 |

Verdict ladder (first match wins): `DO_NOT_PROMOTE` (paired CI UB ≤ 0) →
`INSUFFICIENT_EVIDENCE` (any E unmet) → `NO_VALIDATED_EDGE` (any C fails) →
`ELIGIBLE_PENDING_HUMAN_REVIEW`. Missing metrics fail their criterion —
absence of evidence is never a pass. Unscoreable variants are reported
`NOT_SCOREABLE` and never ranked. The module cannot emit any promote-style
verdict; a human reads the fingerprinted report and decides.

## Invariants (enforced by tests)

1. **No fabrication** — missing data abstains with a machine-readable reason
   (forecast, Kalshi features, GBM stub, promotion criteria).
2. **Point-in-time everywhere** — PIT joins in SQL, purged walk-forward with
   embargo, holdout sessions (2026-11-12/13) excluded from every fit.
3. **Frozen baselines** — `replay.py` untouched; parity asserted; crosschecks
   refuse mixed artifacts.
4. **Exclusive attribution** — a trade's P&L is counted at most once.
5. **Never auto-promote** — promotion output is a report, not an action.
6. **Production isolation** — async research opt-in default-off; research
   modules unimported by the trading path.
