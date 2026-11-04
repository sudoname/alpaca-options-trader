# Research & Validation Layer

Evidence-driven research layer for Oracle: every trading decision becomes
explainable (what was known at decision time), replayable (deterministic
re-pricing under explicit cost/execution models), and measurable against
reliable outcomes — with component-level attribution (direction, contract
selection, entry execution, exit rules) and leakage-safe chronological
validation. New strategy ideas are evaluated in shadow, against the same
immutable evidence, before they are ever allowed to influence trading.

**Default state: OFF.** With `ENABLE_DECISION_LEDGER` unset (or `0`), the bot
behaves byte-for-byte as before — no files are created, no database is opened,
no code path changes. All research tooling is read-only against the ledger;
nothing in this layer can place, modify, or cancel an order.

---

## Components

| File | Purpose |
|---|---|
| `research_ledger.py` | SQLite ledger: immutable decision snapshots + append-only, idempotent lifecycle events |
| `research_capture.py` | Fail-open, flag-gated hooks wiring `smart_trader` to the ledger |
| `research_outcomes.py` | Reconcile ledger decisions against broker truth (orders, fill activities, expiry) + explicit skipped-candidate counterfactual policy |
| `research_replay.py` | Deterministic replay with execution/cost models, adverse scenarios, and a structurally order-proof `ReplayBroker` |
| `research_experiments.py` | Experiment registry (one controlled change), ablations, component attribution, `ShadowComparator` harness |
| `research_validation.py` | Chronological splits with purge/embargo, walk-forward, metrics, Brier/calibration, EV-vs-realized, block bootstrap |
| `research_cli.py` | Operator CLI: inspect / reconcile / replay / compare / report |
| `tests/test_research_layer.py` | Pytest coverage (32 tests, all offline) |

Integration points in `smart_trader.py` (all inside `try/except`, all no-ops
when the flag is off):

* constructor — builds `self.research_capture` (flag-gated)
* `_record_blocked_decision` — rejected candidates with `risk_blocked:<reason>` codes
* order submission success — accepted snapshot + `submitted`/`fill` events;
  `decision_id` attached to `trade_info` so exits link back
* non-2xx order HTTP response — `reject` event
* broker returned no fill — terminal no-fill event
* position close — `exit` + realized `outcome` events

## Setup

Add to `.env` (or export in the shell; shell wins):

```ini
ENABLE_DECISION_LEDGER=1                 # default 0 = fully disabled
DECISION_LEDGER_PATH=research_ledger.db  # optional; this is the default
```

Restart the bot. From then on every evaluated candidate (accepted or rejected)
is snapshotted with: option identifiers, bid/ask + sizes, quote freshness,
underlying price, greeks, signals, regime, direction, score, p_win, EV, cost
assumptions, sizing, risk decision, frozen prediction horizon + exit policy,
strategy/schema versions, git revision, and a secret-redacted config hash.
Field provenance is recorded (observed / estimated / stale / missing) — the
ledger never upgrades a guess to a fact.

## Commands

All commands are read-only unless stated. JSON to stdout.

```bash
# Inspect one decision end-to-end (snapshot + full lifecycle), or ledger stats
python research_cli.py inspect --id <decision_id>
python research_cli.py inspect --stats

# Reconcile outcomes against broker truth. DRY-RUN by default: prints the
# planned events; --apply writes them (idempotent — re-runs write nothing new).
python research_cli.py reconcile --no-network            # expiry/skips only, offline
python research_cli.py reconcile --apply                 # with Alpaca order/activity sync
python research_cli.py reconcile --apply --activities-after 2026-10-01T00:00:00Z

# Deterministic replay of the recorded cohort under cost/execution scenarios
python research_cli.py replay --scenario base
python research_cli.py replay --scenario all             # base, wide_spread, heavy_slippage, stale_quotes, thin_book
python research_cli.py replay --scenario base --full     # include per-decision results

# Compare: experiments, a shadow variant vs baseline, or per-component report
python research_cli.py compare experiments
python research_cli.py compare shadow --variant <name>
python research_cli.py compare components

# Leakage-safe validation report (walk-forward, metrics, calibration, CIs)
python research_cli.py report
python research_cli.py report --embargo-min 390 --folds 5 --experiments 3
```

Every module also has an offline self-test:

```bash
python research_ledger.py; python research_capture.py; python research_outcomes.py --selftest
python research_replay.py; python research_experiments.py; python research_validation.py
python research_cli.py --selftest
python -m pytest tests/test_research_layer.py -v
```

## Design rules (what you can rely on)

* **Fail-open capture.** A research write failure can never raise into the
  trading path; it is counted in `ResearchCapture.dropped` so missing evidence
  is measurable, not silent. The shadow harness uses a bounded queue —
  overflow increments `dropped` and returns `False`, never blocks.
* **Idempotent events.** Every lifecycle event has a deterministic
  `event_key` (hash of decision_id, type, order id, timestamp, qty, price,
  dedupe tag) with `INSERT OR IGNORE`. Restarts and reconciler re-runs cannot
  double-count. Snapshots are immutable inserts; corrections are new events.
* **Outcomes come from the broker, never from quotes.** A fill exists only if
  the broker order record or FILL activity stream says so. Expired-without-exit
  contracts get an `expiry` event whose payload states the settlement value was
  not observed. Skipped candidates are resolved only under the explicit
  `underlying_counterfactual_v1` policy and are always labeled
  `class: simulated` — never mixed with realized results unless
  `include_simulated=True` is passed deliberately.
* **Replay never fabricates.** No recorded two-sided quote → `unreplayable`.
  Quote older than the model's `max_quote_age_s` → `unfilled`. Displayed size
  caps fills (`partial`/`unfilled`); missing size is reported as
  `liquidity: unobserved`, not assumed infinite. Fills price at bid/ask ±
  slippage with per-side fees and the actual contract multiplier; a buy+sell
  round trip reconciles exactly with `cost_model.round_trip_cost` (spread paid
  once per side, nothing double-counted). `ReplayBroker` structurally exposes
  only read methods — any other attribute access or mutation raises
  `ReplayIsolationError`, so research commands cannot submit orders even by bug.
* **One controlled change.** `ExperimentRegistry.register` rejects candidates
  differing from baseline in more than one parameter (unless explicitly
  overridden); rejection criteria are frozen at registration; failures are
  logged as results too.
* **No leakage.** Chronological splits purge training rows whose outcome
  window crosses the boundary (open-ended rows are purged — can't prove they
  closed), plus a configurable embargo. Walk-forward is expanding-window with
  per-fold sample counts and day coverage. `validation_report` computes
  metrics on train/val and walk-forward folds only; the final test slice is
  returned untouched and is **never** evaluated by any tool here — that is a
  one-time, explicit human act. Brier/calibration use only observations with a
  defined binary outcome; day-block bootstrap CIs respect intra-day
  correlation; Bonferroni adjusts for the number of experiments.
* **Shadow ≠ baseline.** Shadow snapshots are written with
  `kind='shadow_candidate'` and are excluded from every baseline consumer
  (replay cohorts, validation observations, component reports, skip
  resolution). Comparison happens only through
  `compare_shadow_to_baseline(ledger, variant)`. There is no promotion code
  path: promoting a variant is a separate, explicit config change by a human.

## Assumptions

* Long single-leg options only (matches the bot): buys lift the ask, sells hit
  the bid; `cost_model.py` is the single cost definition.
* Contract multiplier defaults to 100 unless the snapshot recorded otherwise.
* Quote freshness is only enforceable when `age_s` was recorded with the
  quote; unknown age is flagged, not guessed.
* Alpaca paper fills are treated as evidence of *process*, not of live
  executability — replay scenarios (`wide_spread`, `heavy_slippage`,
  `stale_quotes`, `thin_book`) exist precisely to stress that assumption.
* Timestamps are UTC ISO-8601; the ledger stamps receipt time independently of
  the caller-supplied event time.

## Limitations / known data gaps

* Decisions made while the flag was off are not reconstructible; evidence
  starts when capture starts.
* Expired-without-exit positions carry no settlement value (we do not fetch
  OCC settlement); their P/L stays honest-unknown rather than estimated.
* The skipped-candidate counterfactual uses underlying movement only (no
  option-surface re-pricing) and is clearly labeled `simulated`.
* Intraday option quote history is not backfilled: exact replay uses the
  frozen entry quote; historical simulation requires a caller-supplied
  `quote_fn` and is labeled `mode: historical_sim`.
* `compare shadow` requires the shadow variant to have logged against the same
  decision ids as baseline; disjoint cohorts produce a note, not a verdict.

## Shadow strategies (observe-only strategy experiments)

`shadow_strategies.py` adds a pluggable strategy interface plus two
observe-only strategies that evaluate the same symbol universe as the live
scanner every entry-window cycle and log their would-be trades into the
ledger as `kind='shadow_candidate'` rows (via the existing fail-open
`ShadowComparator`). **They place no orders, feed no gate, and change no
live behavior.** Default OFF.

* `simons_meanrev` — cross-sectional mean reversion: z-score of the last
  close vs a 20-day SMA; z <= -1.5 → CALL (oversold), z >= +1.5 → PUT
  (overbought); only the top-K by |z| per cycle are "accepted", the rest log
  `shadow:not_top_k`.
* `soros_momentum` — 5-day momentum with volume confirmation: |5d return| >
  2% with last volume > 1.2x the 20-day average; exhaustion filter rejects
  moves > 2 sigma*sqrt(3) over 3 days (`shadow:exhausted`).

Enable with `ENABLE_SHADOW_STRATEGIES=1` (requires the decision ledger to be
on). Strategy selection and parameters: `SHADOW_STRATEGY_LIST`,
`SHADOW_SIMONS_WINDOW/Z_ENTRY/TOP_K`, `SHADOW_SOROS_MOM_THRESH/VOL_MULT`,
`SHADOW_CF_HORIZON_MIN` (resolution horizon, default 390 min = one session).
On start the scheduler logs `[SHADOW-STRAT] active: <variants>`.

Scoring: `reconcile --apply` (or the scheduler's periodic reconciler) runs
`resolve_shadow()`, which after the horizon prices the underlying move and
writes an idempotent outcome event with
`payload.class='simulated'`, `policy='underlying_counterfactual_v1'`, and
`counterfactual_pnl_pct` (CALL = underlying up %, PUT = underlying down %).
**All shadow outcomes are underlying-move counterfactuals, not option P/L**
— good for measuring direction edge, not cost/exit tuning. Because the rows
are `kind='shadow_candidate'` and the outcomes `class='simulated'`, nothing
shadow ever enters baseline analytics or validation metrics.

Read results with:

```bash
python research_cli.py compare shadow --variant simons_meanrev
python research_cli.py compare shadow --variant soros_momentum
```

The output includes a `counterfactual_outcomes` block: n entries, n
resolved, hit-rate (counterfactual_pnl_pct > 0), avg/median pnl %, and a
per-direction split. Promotion of a variant to live trading is deliberately
NOT automated — it would be a separate explicit human change.

Rollback: set `ENABLE_SHADOW_STRATEGIES=0` and restart; existing shadow rows
stay in the ledger but nothing new is written or resolved.

## Rollback

1. Set `ENABLE_DECISION_LEDGER=0` (or remove it) and restart the bot — the
   live path is back to pre-integration behavior; hooks become no-ops.
2. Optionally archive/delete `research_ledger.db` and
   `research_experiments.db` (data only; nothing in the bot reads them).
3. Full removal: revert the flag-gated blocks in `smart_trader.py` and delete
   the `research_*.py` modules, `tests/test_research_layer.py`, and this file.
   No other module imports them.
