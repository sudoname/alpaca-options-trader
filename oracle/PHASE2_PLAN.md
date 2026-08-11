# Oracle 3.0 — Phase 2 Plan: Simulation Engine, Validation-Gated Promotions, Memory

Phase 1 shipped five flag-gated, default-OFF, fail-open upgrades (Lab, Temporal
Integrity, Unified Decision/Execution, Adversarial Thesis, Executable EV) and is
now running **shadow-live** in production (`ENABLE_ADVERSARIAL_THESIS=true`,
`ENABLE_EXECUTABLE_EV=true`, `EXECUTABLE_EV_SHADOW_MODE=true`). Phase 2 does three
things:

1. **Build the one deferred Phase 1 module** — the event-driven simulation engine
   (`oracle/engine/`) — so backtest/shadow/paper/live share one strategy body and
   replays are deterministic.
2. **Turn shadow layers into *validation-gated* live behavior** — executable-EV
   veto and adversarial `p_no_trade` gating are promoted ONLY after the Lab +
   accumulated shadow telemetry prove they help out-of-sample.
3. **Close the learning loop** — calibrate the fill model from realized fills and
   enable semantic trade memory as context.

## Non-negotiables (carried verbatim from Phase 1)
- Every new behavior is **flag-gated, default-OFF, fail-open**. The live decision
  path in `smart_trader.py` stays **byte-identical when all Phase 2 flags are OFF**.
- Direction stays an OUTPUT. No indicator/catalyst/LLM/RL becomes a standalone
  trigger. The system may always abstain.
- New layers NEVER bypass hard risk (`risk_engine`, PDT, buying power,
  duplicate/quote-freshness). Promotions are **veto-only** (may reduce/abstain,
  never open/size/flip a trade).
- Each new module ships a module-level `_self_test() -> int` (0=PASS), is
  `__main__`-guarded, offline (no creds/network), and is added to
  `run_selftests.py`.
- **No live-money enablement** is introduced by writing this plan; promotions are
  paper-first and require the acceptance gates below.

## Current state (verified on disk)
- Present: `oracle/lab/{dataset,experiment,metrics,parameter_sweep,parameter_stability,walk_forward}.py`,
  `oracle/temporal.py`, `oracle/thesis_debate.py`, `oracle/trade_memory.py`,
  `oracle/execution/{fill_model,fill_simulator,executable_ev,calibration,client,alpaca}.py`,
  `oracle/decision/*`.
- **Absent: `oracle/engine/`** (event-driven sim — the deferred Phase 1 item).
- Live wires in `smart_trader.py`: flags (~L381-387), adversarial-thesis shadow
  block (~L2363), executable-EV block (~L2432) with shadow guard (~L2460),
  semantic-memory block in `record_trade_outcome` (gated OFF).
- Gate: `oracle_trade_gate.evaluate_oracle_gate(oracle_payload, config)`.
- Calibration ledger already exists:
  `oracle/execution/calibration.py::{record_execution_estimate, record_execution_realization, load_records, compute_calibration, format_calibration_report}`.
- Shadow telemetry live in `/var/log/alps-scheduler.log`: `[THESIS]`, `[EXEC EV]`,
  `would-reject`. Captured by `.oracle3_capture.sh` + `.oracle3_open_monitor.sh`.

---

## Upgrade A — Event-Driven Simulation Engine  [new pkg `oracle/engine/`]
Adapter-over-existing: do NOT rewrite `smart_trader`. Wrap the existing pure
decision functions behind a shared Strategy interface so backtest/shadow/paper/
live differ only by feed + execution adapter + clock. Reuse
`HistoricalMarketView`, `oracle_agents`/`oracle_voting`,
`oracle_intelligence_reports.compute_oracle_explain`, `ev_engine`, `cost_model`,
`exit_manager`, `risk_engine`, and the existing `oracle/execution/fill_simulator.py`.

- `oracle/engine/__init__.py`
- `oracle/engine/events.py` — frozen dataclasses for the event set: `MARKET_DATA,
  BAR, QUOTE, OPTION_QUOTE, NEWS, CATALYST, SIGNAL, TRADE_INTENT, ORDER_SUBMITTED,
  ORDER_ACCEPTED, ORDER_REJECTED, ORDER_PARTIAL_FILL, ORDER_FILLED,
  ORDER_CANCELLED, ORDER_REPLACED, POSITION_UPDATED, RISK_EVENT, EXIT_SIGNAL,
  SESSION_OPEN, SESSION_CLOSE`. Common fields: `event_id, event_type,
  event_timestamp, received_timestamp, source, symbol, strategy_mode,
  correlation_id`.
- `oracle/engine/clock.py` — `Clock` protocol; `LiveClock` (wall-clock) +
  `SimulationClock` (advances deterministically from the event stream). All
  time-based logic in the sim path reads the clock, never `datetime.now()`.
- `oracle/engine/bus.py` — deterministic in-memory `EventBus`, ordered by
  `(event_timestamp, seq)`, synchronous dispatch, no threads → replay determinism.
- `oracle/engine/strategy.py` — `Strategy` Protocol (`on_bar, on_quote,
  on_option_quote, on_news, on_timer, on_fill, on_position_update,
  on_session_open, on_session_close`) + `OracleStrategyAdapter` that calls the
  EXISTING pure functions and emits `TRADE_INTENT` (never touches the broker).
- `oracle/engine/backtest_driver.py` — feeds `HistoricalMarketView` +
  `SimulationClock` through the bus into `OracleStrategyAdapter` → `TRADE_INTENT`
  → `SimBroker` (built on `oracle/execution/fill_simulator.py`) → outcome; reuses
  `exit_manager`, `risk_engine`, `cost_model`. Deterministic given
  `(dataset, config, seed)`.
- LLM determinism: `backtest_driver` forbids live LLM calls — asserts a
  cached/mocked provider.
- Flag: `ENABLE_EVENT_DRIVEN_SIMULATION` (enables the driver; live `smart_trader`
  untouched).
- Tests (`tests/test_event_engine.py`): same adapter runs under sim + a
  paper-mock feed; same seed ⇒ identical signals/orders/fills/PnL; sim clock
  controls session open/close; event ordering preserved; injected look-ahead
  rejected via `oracle/temporal.py`.
- **Acceptance:** run `backtest_driver` twice with the same seed → byte-identical
  PnL and order log.

## Upgrade B — Run the Lab to produce OOS evidence  [uses existing `oracle/lab/`]
The Lab exists but is import-only. Phase 2 actually runs it to generate the
out-of-sample metrics that gate every promotion below. No new decision behavior.

- `oracle/lab/reports.py` (NEW, matches `calibration_reports.py` idiom):
  `compute_/format_/generate_lab_report_text` with regime / CALL / PUT / NO_TRADE
  / catalyst / strategy_mode / EV-bucket / PoP-bucket breakdowns + an
  IS-vs-OOS section.
- `oracle/lab/run_phase2_study.py` (NEW, offline harness): builds a point-in-time
  dataset via `dataset.build_dataset(...)`, then runs `walk_forward.walk_forward`
  (TRAIN→VALIDATE→FREEZE→TEST→ROLL) and `parameter_stability`, writing results to
  `oracle/lab/results/`. Deterministic (seeded). No creds/network — replays cached
  historical `MarketView` data + the shadow JSONL ledgers.
- Data sources it consumes: `oracle_prob_recorder` JSONL, `episode_store`
  (`episodes.db`), `candidate_resolution.jsonl`, and the calibration ledger.
- Flag: `ENABLE_ORACLE_LAB` (documents intent; Lab remains offline).
- **Acceptance:** `walk_forward` raises `oos_collapse` when TEST expectancy
  collapses vs VALIDATE; sweep never selects on the full sample
  (`in_sample_only=True`); a report is produced with an IS-vs-OOS section.

## Upgrade C — Executable-EV: shadow → paper veto  [modify existing wires]
Promote the executable-EV layer from "log would-reject" to an actual veto — but
only in the paper account, and only after the gate below.

- No new decision math; reuse `oracle/execution/executable_ev.compute_executable_ev`.
- Ensure the estimate is persisted every evaluation via
  `calibration.record_execution_estimate(...)` and the realized fill via
  `calibration.record_execution_realization(trade_id, ...)` on entry fill (wire
  in `place_order_with_stops` / `record_trade_outcome` if not already firing on
  every path).
- Promotion mechanics: keep `EXECUTABLE_EV_SHADOW_MODE=true` (log-only) until the
  gate passes; then set it `false` **on the paper account only** so a negative
  `executable_EV` returns None (veto-only, after the existing EV gate, before the
  risk engine — never overrides risk).
- **Promotion gate (must ALL hold before flipping shadow OFF, paper):**
  1. ≥ 200 shadow evaluations across ≥ 10 trading sessions.
  2. `compute_calibration` shows `execution_capture_ratio` stable and the
     would-reject bucket's realized expectancy is worse than the would-pass
     bucket by a margin exceeding its bootstrap CI.
  3. Lab walk-forward (Upgrade B) confirms the veto improves OOS expectancy
     (no `oos_collapse`).
  - NOTE: the single-session read on 2026-08-11 showed exec-EV rejects on
    *tradability* (wide spread), not *direction* — the whole book was red that
    day. The gate exists precisely to require multi-session directional evidence.
- Flags: `ENABLE_EXECUTABLE_EV` (already ON, shadow), `EXECUTABLE_EV_SHADOW_MODE`
  (flip per-account after gate), `ENABLE_FILL_MODEL`.
- Tests (`tests/test_executable_ev_promotion.py`): wide spread lowers
  executable_EV; +theoretical/−executable vetoes when shadow OFF; veto returns
  before the risk engine; `max_entry_price` never exceeded; shadow ON is
  byte-identical to pre-Phase-2.

## Upgrade D — Fill-model calibration  [uses existing `calibration.py`]
Replace the conservative heuristic `fill_probability`/slippage constants with
values learned from realized fills.

- `oracle/execution/fill_model.py`: add a `calibrated_params_from(records)`
  loader that reads `calibration.load_records()` + `compute_calibration()` and
  adjusts `fill_probability`/`slippage_estimate`/`liquidity_penalty` per
  spread/OI/quote-age bucket. Fail-open: if insufficient data, fall back to the
  current conservative constants (byte-identical).
- Flag: `ENABLE_FILL_MODEL` (OFF → conservative constants; ON → calibrated).
- **Acceptance:** with < N calibration records the model is byte-identical to
  today; with sufficient records, `model_capture_ratio` → 1.0 on the calibration
  set; deterministic for identical input.

## Upgrade E — Adversarial thesis: log-only → gated p_no_trade  [modify existing wire]
Promote the thesis skeptic from log-only to feeding adjusted `p_no_trade` into
`oracle_trade_gate.evaluate_oracle_gate`.

- No new math; reuse `thesis_debate.build_theses` + `adversarial_review`
  (bounded by `THESIS_MAX_NO_TRADE_BOOST=0.10`, renormalized, never flips
  direction).
- When promoted, pass the review's `adjusted_probability` into the gate instead
  of the raw probability; still veto-only.
- **Promotion gate (must ALL hold):**
  1. Over the shadow window, thesis flags fire on a non-trivial fraction of
     candidates (today it fired **0** flags — insufficient to promote).
  2. Flagged candidates show worse realized expectancy than un-flagged ones
     (evidence the skeptic is selective, not noise).
  3. Lab confirms no OOS degradation from the added abstention.
- Flag: `ENABLE_ADVERSARIAL_THESIS` (already ON shadow) + a new
  `ADVERSARIAL_THESIS_GATING` (default OFF) that switches log-only → gate-feeding.
- Tests (`tests/test_adversarial_thesis_gating.py`): gate receives adjusted prob
  only when gating flag ON; boost capped; direction preserved; negative-EV stays
  NO-TRADE; gating OFF is byte-identical.

## Upgrade F — Semantic trade memory: enable + wire retrieval  [uses existing `trade_memory.py`]
Turn on the postmortem loop and feed lessons back as **context only**.

- On close (`record_trade_outcome`), when `ENABLE_SEMANTIC_TRADE_MEMORY`, call
  `trade_memory.record_reflection({expected, actual, failure_mode, lesson,
  symbol, sector, regime, catalyst, strategy_mode, confidence})`.
- In the thesis path, when the flag is ON, call
  `trade_memory.retrieve_lessons({symbol, sector, regime, catalyst,
  strategy_mode, failure_mode})` and attach newest-first lessons to the thesis
  `notes` (natural-language context). **Never** moves numbers or overrides rules.
- Flag: `ENABLE_SEMANTIC_TRADE_MEMORY` (default OFF).
- Tests: reflection persisted append-only on close; retrieval filters correctly;
  lessons never alter `probability`/gate output; flag OFF is byte-identical.

## Upgrade G — Automated promotion gates (Stage 4)  [new `oracle/promotion.py`]
Codify the manual gates in C/D/E so a flag flips only when OOS thresholds are met.

- `oracle/promotion.py`: `evaluate_promotion(layer, calibration_stats,
  lab_result, thresholds) -> {promote: bool, reasons[], metrics}`. Pure,
  deterministic, fail-open (returns `promote=False` on any missing input).
- Consumed by an offline `run_promotion_check.py` that prints a report; it does
  NOT auto-edit `.env` — a human flips the flag after reading the report
  (auditable). Optionally emits a Telegram summary.
- Flag: none live-affecting (offline advisory). Thresholds live in `.env`.
- Tests: promote=False when data insufficient / `oos_collapse` / margin inside CI;
  promote=True only when all conditions pass.

---

## Storage / schema (additive, nullable — backward compatible)
Extend episode/trade + lab rows (prefer stamping into `episodes.features_json` /
trade dicts / JSONL ledgers over destructive SQLite migrations) with any of these
not already present: `decision_timestamp, event_timestamp, strategy_mode,
theoretical_EV, executable_EV, realized_EV, expected/actual entry/exit price,
fill_probability, fill_delay, slippage, spread_at_entry/exit, bull/bear/no_trade
thesis, thesis_flags, adjusted_p_no_trade, trade_reflection, experiment_id`.

## Configuration flags (all default OFF / shadow; document in `.env.example`)
Existing: `ENABLE_ORACLE_LAB, ENABLE_TEMPORAL_INTEGRITY,
ENABLE_EVENT_DRIVEN_SIMULATION, ENABLE_ADVERSARIAL_THESIS,
ENABLE_SEMANTIC_TRADE_MEMORY, ENABLE_EXECUTABLE_EV, ENABLE_FILL_MODEL,
EXECUTABLE_EV_SHADOW_MODE(ON), THESIS_MAX_NO_TRADE_BOOST(0.10)`.
New: `ADVERSARIAL_THESIS_GATING(OFF)` + promotion thresholds
(`PROMO_MIN_EVALS, PROMO_MIN_SESSIONS, PROMO_MIN_MARGIN`, etc.).

## Rollout (per spec, unchanged posture)
- **Stage 1 (offline):** Upgrade A (sim engine) + Upgrade B (run Lab). Research
  only, no live effect.
- **Stage 2 (shadow, already live):** keep accumulating `[THESIS]`/`[EXEC EV]`
  telemetry + calibration ledger; add fill-model calibration record/realization
  on every path (Upgrade D record-side).
- **Stage 3 (paper):** after Upgrade G reports PASS, flip
  `EXECUTABLE_EV_SHADOW_MODE=false` (C) and/or `ADVERSARIAL_THESIS_GATING=true`
  (E) and `ENABLE_FILL_MODEL=true` (D) **on the paper account only**; enable
  `ENABLE_SEMANTIC_TRADE_MEMORY` (F).
- **Stage 4 (gated live):** only after paper validation + a human reading the
  `run_promotion_check.py` report. **Not in scope to enable here.**

## Testing / CI
- Each new module: offline `_self_test() -> int`, `__main__`-guarded, added to
  `run_selftests.py`.
- New pytest suites: `test_event_engine.py`, `test_oracle_lab_study.py`,
  `test_executable_ev_promotion.py`, `test_adversarial_thesis_gating.py`,
  `test_fill_calibration.py`, `test_trade_memory_wire.py`, `test_promotion_gate.py`.
- Regression: with ALL Phase 2 flags OFF, `python -c "import py_compile,
  smart_trader"` OK and a smoke decision equals the Phase 1 output
  (shadow-identical).

## Verification (end-to-end, offline)
1. `python run_selftests.py` → all PASS, exit 0.
2. `python -m pytest tests/ -q` → all Phase 2 suites green.
3. Determinism: `backtest_driver` twice, same seed → identical PnL/order log.
4. Regression: all Phase 2 flags OFF → live path byte-identical to Phase 1.
5. Lab: `run_phase2_study.py` emits an IS-vs-OOS report; `oos_collapse` fires on a
   seeded degrading fold.
6. Promotion: `run_promotion_check.py` returns `promote=False` on today's
   insufficient shadow sample and only PASSes when the gates are met.

## Critical path
`A (event engine)` + `B (run Lab)` produce the OOS evidence → gates the `C/D/E`
promotions (paper) → `G` automates the promotion decision → Stage 4 (live) remains
manual + out of scope. The live shadow monitor is the ongoing feed for B/C/E.

## Files
NEW: `oracle/engine/{__init__,events,clock,bus,strategy,backtest_driver}.py`,
`oracle/engine/sim_broker.py` (or extend `oracle/execution/fill_simulator.py`),
`oracle/lab/reports.py`, `oracle/lab/run_phase2_study.py`, `oracle/promotion.py`,
`run_promotion_check.py`, `tests/test_*.py` (per above).
MODIFIED (flag-gated, byte-identical when OFF): `smart_trader.py` (promotion wires
for C/E/F + new flags in the `_flag/_f2` block), `oracle/execution/fill_model.py`
(calibrated params, Upgrade D), `oracle_prob_recorder.py`/`episode_store.py`
(additive fields), `.env.example` (new flags + thresholds), `run_selftests.py`
(register new modules).
