# Oracle 4.0 — Phase 0 gap audit

- **Audit date:** 2026-11-16 (local)
- **HEAD at audit:** `8af6d5bd6d91bfc473ade43a21f1e413c863603e`
  ("Fills-based close booking for bot-owned exits (fail-open)") — identical to
  the master-prompt reference commit.
- **Branch:** `oracle4-research` (created from master at the SHA above).
- **Working tree:** clean except runtime data (`candidate_resolution.jsonl`
  modified; untracked `DEPLOYMENT_STATUS.md`, `trading_history_server_2026-11-13.json`)
  — never committed by convention.
- **No trading API was called and no service was activated for this audit.**
  All facts below come from static file inspection and read-only SQLite/JSONL
  reads of files already in the working tree.

## 1. Gap classification

Legend: **implemented** (reusable as-is), **incomplete** (partial, needs
extension), **unavailable** (data/feed does not exist in this environment),
**verified missing** (no code exists).

| # | Oracle 4.0 capability | Status | Evidence / source paths |
|---|---|---|---|
| 1 | Frozen baseline decision kernel (snapshot → decision, deterministic, fail-open) | implemented | `oracle/decision/{direction,kernel,schema,snapshot_builder}.py`; parity test `tests/test_decision_parity.py` |
| 2 | Bayesian call/put/no-trade benchmark head | implemented | `oracle_intelligence_reports.py` (`compute_oracle_explain`), 14 agents in `oracle_agents.py`, tally in `oracle_voting.py`. Vote tally, **not a trained predictive model** |
| 3 | Gross EV / cost accounting / executable EV | implemented | `entry_ev_stamp.py`, `ev_engine.py`, `cost_model.py`, `oracle/execution/executable_ev.py`. Costs charged once at shadow-recorder netting; executable EV overlays `fill_model` |
| 4 | Fill priors, order simulation, lifecycle, limit policy | implemented | `oracle/execution/{fill_model,fill_simulator,calibration,lifecycle,limit_policy}.py`; conservative defaults; calibration fit exists but thin real-fill sample (see §3) |
| 5 | Event simulation engine | implemented | `oracle/engine/{backtest_driver,sim_broker,strategy,events,bus,clock}.py` — event-driven, deterministic |
| 6 | Walk-forward research harness | **incomplete** | `oracle/lab/walk_forward.py` splits by dataset fraction/fold count, **not whole exchange sessions**; no label-end/availability purge, no embargo parameter, no grouped decision IDs, no negative controls (future-feature / shuffled-label) |
| 7 | Purged chronological PoP calibration | implemented | `pop_recalibration.py` (isotonic/PAV, prior-sessions-only, per-cohort fallback). Scope limited to PoP; not a general validation harness |
| 8 | Quote-path exit replay with honest outcome states | implemented | `exit_replay.py` + `outcome_states.py` (`FILLED_RESOLVED / CONFIRMED_NO_FILL / POLICY_SKIP / UNRESOLVED / MISSING_COVERAGE`; `net_dollars_from_fills`) |
| 9 | Candidate-group + quote capture + provenance ledger | implemented | `research_capture.py` (flag `ENABLE_DECISION_LEDGER`), `research_ledger.py` (INSERT-only `snapshots`, idempotent append-only `events`), `quote_capture.py` (cron on server since 2026-11-15), `quote_coverage_probe.py`, `research_export.py` |
| 10 | Versioned feature/label store (`oracle/ml/{schema,features,labels,dataset}.py`) | **verified missing** | No `oracle/ml/` package exists |
| 11 | Multi-horizon underlying labels (log return, direction w/ neutral band, future RV, IV change, first-hit w/ censoring) | **verified missing** | No label pipeline anywhere; `oracle/lab/dataset.py` loads trade-level episodes only |
| 12 | Classical challengers (logistic, GBM, quantile regressors, EWMA/HAR-RV, discrete-time competing-risk hazards) | **verified missing** | No sklearn/lightgbm imports anywhere; `pop_recalibration.py` implements PAV locally; RL module is a custom Q-table |
| 13 | Multitask neural challenger (PyTorch MLP, multi-head) | **verified missing** | `torch` not imported anywhere, not in `requirements.txt` |
| 14 | Model registry (champion/challenger, immutable artifacts, approval state) | **verified missing** | No registry; `oracle/promotion*.py` is trade-promotion logic, not model registry |
| 15 | Calibrated joint scenario generator (return × RV × timing × liquidity) | **verified missing** | `ev_engine.py` is closed-form geometric path EV; no Monte-Carlo scenario module |
| 16 | Contract repricing along simulated paths with actual exit policy | **incomplete** | `exit_replay.py` replays **observed** quote paths; no simulated-path repricing; no CRR/American reference pricer (`pricing_diagnostics.py` is diagnostics only) |
| 17 | All-candidate valuation + constrained ranking with abstention | **incomplete** | Production is **score-first**: `smart_trader.py:2233 select_best_option()` scores candidates via `calculate_option_score()` (`smart_trader.py:1860`) and returns argmax — EV is stamped on the chosen contract afterward, so a lower-score candidate with higher executable EV is never selected. Group capture of all candidates exists (ledger `kind='candidate'/'shadow_candidate'`), enabling offline re-ranking. Abstention exists only as gates/vetoes, not as an EV=0 competitor in ranking |
| 18 | Side consistency / weak-data CALL fallback | implemented (defect confirmed present by design) | `smart_trader.py:1943` — `len(prices) < 5` ⇒ returns `'call'`; mirrored at `oracle/decision/direction.py:128-133` (`insufficient_data=True`, confidence 0). With `USE_SKIP_ON_WEAK_SIGNAL` off (production default) the CALL fallback is reachable. Kernel gates demote to no-trade but never flip side |
| 19 | Execution calibration incl. nonfills/cancels, marketability, slippage | **incomplete** | `oracle/execution/calibration.py` exists; ledger events contain `submitted/fill/exit/outcome` only — **zero `cancel`/`reject` rows** in the frozen extract, so confirmed-nonfill evidence is absent so far |
| 20 | Portfolio-level paired replay (overlap, cash constraints, mark-to-market) | **incomplete** | `oracle/engine/` can simulate orders; existing studies (`exit_replay.py`, `oracle/lab/spy_*`) sum independent trades; no portfolio cash/exposure-constrained paired replay |
| 21 | Forward shadow flags for Oracle 4 (`ENABLE_ORACLE4_SHADOW`, `ORACLE4_MODEL_ID`, `ORACLE4_MAX_CANDIDATES`, `ORACLE4_MAX_SCENARIOS`) | **verified missing** | No such flags; existing shadow layers: `SHADOW_RECORDER_ENABLED`, `ENABLE_DECISION_LEDGER`, `ENABLE_PUT_TIME_STOP_SHADOW`, `EXECUTABLE_EV_SHADOW_MODE` |
| 22 | Promotion report with pre-registered gates | **verified missing** | `oracle/promotion_audit.py` audits trade promotions, not model promotion |
| 23 | Historical option bid/ask backfill | **unavailable** | Documented: Alpaca config provides latest/indicative quotes only; no licensed point-in-time option NBBO. Forward capture (`quote_capture.py` cron) is the fallback. Indicative quotes are labeled as such |
| 24 | Historical underlying bars | implemented (feed available) | Alpaca stock bars used throughout (`smart_trader.py`, `backtest.py`); usable for initial model research without touching trading paths |

## 2. Score-first selection — defect detail (gap #17)

`select_best_option()` (`smart_trader.py:2233`):

1. Filters by side (from `determine_option_strategy()`), moneyness, quote
   presence, spread, budget.
2. Scores each survivor with `calculate_option_score()` (Greeks + IV +
   moneyness + learned-pattern boosts, clamped [0,100]).
3. Returns the max-score contract. EV/PoP stamping (`entry_ev_stamp.py`)
   happens on the winner only; profitability gates then veto or admit it.

Consequences verified: (a) viable same-side alternates with better executable
EV are excluded before valuation; (b) a vetoed winner is not replaced by the
runner-up — the scan abstains; (c) alternate-direction candidates are never
valued. Candidate-group capture records all eligible contracts, so the fix can
be built and evaluated **offline in the challenger only** (Phase 5), keeping
the baseline untouched.

## 3. Environment constraints (recorded, no values logged)

- **Credentials present (names only):** `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`,
  `ALPACA_PAPER` (paper account both locally and on the deployment),
  `TELEGRAM_BOT_TOKEN/CHAT_ID`, `SCHWAB_*`, `RH_USERNAME/PASSWORD`. No values
  were read or logged.
- **Quote provenance:** decision-time quotes in ledger snapshots
  (`quote_bid/ask/…`, `quote_ts_utc`, `quote_age_s`); forward quote ticks from
  `quote_capture.py` (server cron, weekdays 14:30–20:59 UTC, started
  2026-11-15). Historical option quote backfill unavailable (gap #23).
  Latest-quote reads are indicative and labeled.
- **Candidate-group coverage (frozen extract `research_ledger_server_2026-11-13.db`):**
  14,264 snapshots (2,526 `candidate`, 11,738 `shadow_candidate`),
  88 distinct underlyings, 2026-11-04T14:53Z → 2026-11-13T20:05Z
  (~8 trading sessions).
- **Event coverage (same extract):** 3,654 events — 779 `submitted`,
  779 `fill`, 754 `exit`, 1,342 `outcome`; **0 `cancel`/`reject`** rows.
  Confirmed-nonfill evidence therefore does not exist yet in data.
- **Fill/fee completeness:** production trading history is overwhelmingly
  mark-derived (server, 2026-11-16: 3,565 mark vs 14 fills-true of 3,579
  resolved). Fills-true close booking only became live at `8af6d5b`
  (2026-11-15); fee fields exist in schemas but are effectively 0 on this
  paper feed. Mark-derived outcomes must stay labeled `pnl_source='mark'` and
  may not be presented as actual fills.
- **Exports:** `export_decisions.jsonl` 14,264 lines (sha256 in manifest);
  `export_groups.jsonl` **empty (0 lines)** — group export has produced no
  rows yet (defect/coverage note); `candidate_resolution.jsonl` 70 rows.
- **Sample-size reality:** ~8 sessions of groups + days of quote ticks is far
  below the pre-registered gates (≥60 forward sessions, ≥200 resolved
  opportunities). All empirical gates in later phases start as
  **PENDING / INSUFFICIENT_EVIDENCE**; engineering proceeds regardless.
- **Compute:** CPU-only Python 3.10 locally (no torch/sklearn/lightgbm/duckdb/
  pyarrow installed or declared). New ML dependencies must be optional with
  stdlib/numpy fallbacks; CPU is the default device.

## 4. Test baseline at the frozen SHA

- Convention: full suite runs `python -m pytest -q --ignore=tests/test_safety.py`
  (that file hangs in this environment; root-level `test_safety.py` is a
  separate, collected module).
- Baseline at `8af6d5b` (2026-11-15 run, re-verified this audit — result in
  manifest): **2082 passed, 3 failed**, all pre-existing and unrelated to
  Oracle 4.0 scope:
  1. `tests/test_oracle_intelligence_reports.py::TestTelegramDispatch::test_eight_commands_dispatch`
     (live-data footer assertion),
  2. `test_safety.py::TestEVEntryGate::test_disabled_skips_gate` and
  3. `test_safety.py::TestEVEntryGate::test_passes_when_above_floor`
     (environment-dependent gate defaults).
- 45 test files under `tests/`, hermetic conventions (no network, traders via
  `__new__`, in-memory ledgers, patched `requests`).

## 5. Baseline freeze and parity statement

- Immutable manifest: `docs/oracle4_baseline_manifest.json` (code SHA, flag
  defaults, data-asset SHA-256 hashes, test baseline).
- All Oracle 4.0 additions are **new modules + disabled-by-default flags**.
  With every new flag off, no production code path changes: order path,
  sizing, exit management, hard risk checks, and the profitability gate are
  untouched. Parity requirement: baseline replay outputs at this SHA must be
  reproducible bit-for-bit through any new harness (Phase 2 acceptance).

## 6. Known defects carried into Oracle 4.0 work

1. Score-first selection excludes alternates (gap #17) — fix in challenger
   ranking only (Phase 5).
2. Weak-data CALL fallback reachable in production defaults (gap #18) —
   baseline behavior preserved; challenger must abstain instead.
3. `export_groups.jsonl` empty — group export path produced no rows from the
   frozen extract; Phase 1 ingestion reads the ledger directly instead.
4. No cancel/reject events captured — fill-probability calibration (Phase 6)
   will pool/shrink toward conservative priors until nonfill evidence exists.
5. Mark-derived P&L dominates history — any profitability claim from it is a
   sensitivity study, never an actual-fill result.
