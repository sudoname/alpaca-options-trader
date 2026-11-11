# Oracle Master-Prompt Implementation — Handoff Report

Base commit at review: `master @ 7a3a074`. All work is local, offline-verifiable,
and read-only toward production: no default execution policy changed, no hard
risk check touched, experimental behavior is opt-in/observe-only. Deployment,
pushes, and arming remain operator actions.

Date: 2026-11-11. Python: local venv 3.10, pytest available locally
(server has unittest only — every new test is a stdlib `unittest` class).

---

## 1. What was implemented, phase by phase (verified behavior)

### Phase 1 — Executable EV accounting (fixed double-charging)
- `oracle/execution/executable_ev.py` + `entry_ev_stamp.py` + `smart_trader.py`:
  the single-leg path previously fed a cost-subtracted stamp EV into
  `compute_executable_ev`, which subtracted modeled costs again. Reproduced the
  review fixture (1 contract, bid 0.99 / ask 1.01, basis 1.01, conf 3,
  delta 0.55, target 25%, stop 15%): old path ≈ **−$3.77**, corrected ≈ **+$2.21**.
- Separate, explicitly-united fields now carried: **gross strategy EV** (total
  dollars, pre-execution-cost), **net EV after modeled execution**, and
  **actual realized net P&L**. Multiplier (100), slippage ($0.02/ct), and OCC
  fee ($0.02/ct) are charged exactly once. Fill probability has one
  interpretation (probability the entry fills; nonfill ⇒ $0 at opportunity
  level). Unusable quotes return unavailable EV with reason codes
  `nonpositive_quote | one_sided_quote | crossed_quote | stale_quote` — never
  silently zero. Legacy ledger rows stay readable.

### Phase 2 — Shared decision/execution policy, lifecycle, calibration
- `oracle/decision/snapshot_builder.py`: scheduler shadow path and replay build
  the same point-in-time snapshot and run the same `decide()` policy; clocks,
  feeds, broker calls stay at the boundaries.
  Integration coverage: `tests/test_shared_policy_integration.py` drives
  captured session events through both entry paths and asserts decision,
  contract choice, quantity, veto reason, and intended-order fields match.
- `oracle/execution/limit_policy.py`: **opt-in** EV-derived buy-limit cap
  (`compute_entry_limit` / `gross_ev_from_contract`). Respects option tick
  size, quote age, spread/size constraints; rounds conservatively; explicit
  expiry; reason codes when no valid cap forms. Wired **observe-only** into
  `run_alpaca_intraday.py` (~line 900): logs what the cap *would* be, never
  alters the intended order. Fail-open.
- `oracle/execution/lifecycle.py`: deterministic order-event state machine —
  ack, reject, cancel, expiry, partial fill, late fill, cancel/fill races,
  idempotent replay. Only actually-filled quantity becomes an open position.
- `oracle/execution/calibration.py`: pre-submission persistence of candidate +
  intended order (including no-trade and unfilled opportunities), join by
  stable decision IDs, fill-rate and slippage calibration fitted on submitted-
  order outcomes **including nonfills**. Missing evidence marked explicitly.

### Phase 3 — Exit replay on the quote path + PoP recalibration
- `exit_replay.py`: replays current exit policy plus stop/trailing/time-exit
  variants on chronological option bid/ask. Trigger and execution are modeled
  separately (a sell trigger is an instruction; executed price comes from later
  executable quotes). Covers gap-through stops, stop-then-recovery winners,
  stale quotes, partial exits, forced time exits, ambiguous ordering.
- `pop_recalibration.py`: PoP calibration fitted on prior sessions only
  (chronological folds, purge/embargo, untouched final period); pooled behavior
  with reported sample size when cohorts are thin; calibration and
  profitability evaluated separately. Quote provenance (historical / indicative
  / synthetic fixture / paper fill / broker fill) stays identifiable.

### Phase 4 — VWAP pullback, EMAs, order-flow confirmation
- `intraday_features.py`: session-anchored VWAP reused; 9/20/50-period EMAs
  with documented warm-up; reproducible VWAP-pullback + confirmation state
  machine with explicit tolerances and expiration. 9-over-20 long condition per
  source; 50-EMA entry rule and mirrored put setup kept as separate hypotheses.
- `order_flow_features.py`: timestamped trade-print (Time & Sales) features and
  freshness-aware L2 confirmation; top-of-book and prints-only are distinct
  variants; missing/stale/crossed inputs explicit — confirmation variant only
  applies when its required inputs qualify.
- `vwap_pullback_study.py`: matched-opportunity offline comparison harness
  (baseline vs confirmation variants) over bars JSONL + optional flow JSONL.

### Phase 5 — Pricing sanity, ranking, skeptic impact
- `smart_trader.py`: per-field Greeks/IV **quality labels**
  (`broker` vs `fallback`) and `quote_ts` stamped onto candidates; labels
  reflect only what `_apply_real_greeks` actually applied (dict shape of
  `_apply_real_greeks` unchanged — verified by existing safety tests). Fixed a
  latent dead-key bug (`implied_volatility` → `iv`) in ledger extra wiring.
  Quality, quote_ts, and `thesis_review` now flow into research-ledger
  `extra_json` for accepted candidates *and* gate rejects.
- `pricing_diagnostics.py` (pure stdlib; QuantLib optional via `_HAS_QUANTLIB`):
  quote classification; Black–Scholes price/delta/vega with
  discounted-forward-intrinsic degenerate handling; bid/mid/ask IV solve with
  per-side failure reasons (`no_price | expired | below_intrinsic |
  above_bound`); put–call parity check that is a *diagnostic with a caveat* on
  American/dividend contracts, never an assertion; `model_itm_prob` explicitly
  labeled `model_implied_risk_neutral` and `not_equal_to:
  empirical_probability_of_profit`; CRR binomial as the exercise-appropriate
  reference (`reference_price` records which engine produced the number).
- `ranking_eval.py`: net-replay evaluation of ranking variants —
  frozen **baseline** (argmax score), **observed_only** (penalize fallback
  IV/delta), **liquidity_tilt** (spread% + OI saturation), **delta_band**
  (|Δ| ∈ [0.45, 0.65], falls back to baseline pick when none qualify).
  Nonfills count as $0 at opportunity level; reports n, fill rate, total net,
  expectancy per opportunity / per fill, win rate, profit factor.
- `skeptic_impact.py`: measures the bull/bear/no-trade reviewer's incremental
  value **beyond** deterministic spread/freshness/net-EV checks, from logged
  ledger decisions: among deterministic-pass rows, compares resolved net
  outcomes of "quiet" vs "doubt" (flags or no_trade_boost>0) groups; verdict
  only with ≥5 resolved per group, otherwise `insufficient_evidence` with
  sample sizes reported; unresolved outcomes never imputed; also reports doubt
  redundancy on deterministic rejects. Reads `research_ledger.db` read-only;
  risk enforcement remains deterministic and reviewer-independent.

---

## 2. Changed / new files

Modified:
- `entry_ev_stamp.py`, `smart_trader.py`, `run_alpaca_intraday.py`
- `oracle/engine/strategy.py`, `oracle/execution/executable_ev.py`,
  `oracle/execution/calibration.py`, `intraday_features.py`

New modules:
- `oracle/decision/snapshot_builder.py`, `oracle/execution/lifecycle.py`,
  `oracle/execution/limit_policy.py`
- `exit_replay.py`, `pop_recalibration.py`
- `order_flow_features.py`, `vwap_pullback_study.py`
- `pricing_diagnostics.py`, `ranking_eval.py`, `skeptic_impact.py`

New tests (all stdlib `unittest`, offline, no credentials/network):
- `tests/test_executable_ev_accounting.py`
- `tests/test_shared_policy_integration.py`, `tests/test_entry_limit_policy.py`,
  `tests/test_order_lifecycle.py`, `tests/test_order_outcome_calibration.py`
- `tests/test_exit_path_replay.py`, `tests/test_pop_recalibration.py`
- `tests/test_vwap_pullback_features.py`
- `tests/test_pricing_diagnostics.py`, `tests/test_ranking_skeptic_eval.py`

Not for commit: `DEPLOYMENT_STATUS.md`, `.tmp_cmp.awk`, `.tmp_hash.sh`
(scratch/operator files in the working tree).

---

## 3. Test commands and results (2026-11-11)

```bash
# Full battery (decision parity, event replay, fill calibration, temporal
# integrity, executable EV, exit replay, VWAP features, pricing, ranking,
# skeptic):
python -m pytest tests/ test_entry_ev_stamp.py -q
# -> 631 passed

# Greeks-wiring safety tests (run ONLY filtered — the unfiltered file
# contains live-network tests that hang offline):
python -m pytest test_safety.py -q -k "ApplyRealGreeks or RealGreeksWiring"
# -> 6 passed, 94 deselected
```

Module self-tests (each exits 0):

```bash
python exit_replay.py            # exit_replay self-test: PASS
python pop_recalibration.py      # pop_recalibration self-test: PASS
python vwap_pullback_study.py    # vwap_pullback_study self-test: PASS
python pricing_diagnostics.py --selftest   # PASS (quantlib=no)
python ranking_eval.py --selftest          # PASS
python skeptic_impact.py --selftest        # PASS (incl. temp-ledger round-trip)
```

---

## 4. Reproducible offline comparison commands (frozen baseline + named variants)

```bash
# Phase 4/5 integrated underlying study: frozen baseline entry vs named
# confirmation variants on identical matched opportunities:
python vwap_pullback_study.py --bars <session_bars.jsonl> [--flow <flow.jsonl>] \
    [--target-pct ...] [--stop-pct ...] [--max-hold-bars ...] [--cost-pct ...]
python vwap_pullback_study.py              # deterministic fixture self-test

# Contract-ranking variants vs frozen baseline, scored by net replay outcomes:
python ranking_eval.py --fixture           # deterministic fixture report
python ranking_eval.py --groups <groups.jsonl>   # real data, same schema

# Skeptic incremental-impact study from the live research ledger:
python skeptic_impact.py --fixture
python skeptic_impact.py --ledger research_ledger.db
```

The `groups.jsonl` schema for `ranking_eval.py` (one JSON object per decision
group): `{"candidates": [{"symbol", "score", "iv", "delta", "greeks_quality",
"bid", "ask", "quote_age_s", "volume", "open_interest",
"net_outcome" (float dollars, or null = unfilled)}]}`.

---

## 5. Remaining data requirements (implementation complete, study pending data)

1. **Historical option bid/ask series.** Alpaca's historical option-quote
   endpoint returned 404 for the needed windows. `exit_replay.py` and
   `ranking_eval.py` therefore ship a JSONL import path + deterministic
   fixtures; the real exit/ranking studies run unchanged once per-contract
   chronological bid/ask JSONL is supplied.
2. **Reconciled outcome events with net P&L.** `skeptic_impact.py` needs
   ledger `outcome` events whose payload carries one of
   `net_pnl | net_pnl_usd | realized_pnl | pnl`, joined by `decision_id`, with
   ≥5 resolved rows in both the quiet and doubt groups before it will issue a
   verdict. The scheduler-side wiring to write these rows is in place; evidence
   accumulates as the shadow/paper path runs.
3. **Session bars + order-flow JSONL** for `vwap_pullback_study.py` on real
   SPY sessions (fields documented in the module header).
4. **Fill/slippage calibration sample.** `oracle/execution/calibration.py`
   fits from submitted-order outcomes including nonfills; needs accumulated
   paper/live submissions to replace priors.

---

## 6. Code correctness vs measured profitability

- **Verified as correct code (offline, deterministic):** the EV accounting
  identity (costs charged once; review fixture sign flip −$3.77 → +$2.21),
  decision-policy parity between scheduler shadow and replay, order-lifecycle
  determinism and idempotency, temporal integrity of PoP folds, pricing
  diagnostics (IV round-trips, parity residuals, CRR↔BSM convergence,
  American ≥ European), nonfill-as-$0 opportunity accounting, and the
  skeptic-impact estimator including its insufficient-evidence floor.
- **NOT yet evidence of profitability:** fixture results (e.g. `ranking_eval`
  variants beating baseline, `skeptic_adds_value` on the fixture, the SPY 0DTE
  puts-only/ATM/fixed-roll replay result of +$4,500 over 105 entries at 61.9%
  POP) are synthetic or single-sample and demonstrate only that the harnesses
  measure what they claim. Profitability claims require the real data in §5 on
  identical held-out sessions, reported with sample sizes, fill rates,
  expectancy per opportunity/fill, profit factor, drawdown, and realized
  slippage — then interpreted through the existing promotion process.
- Default execution policy, hard risk checks (buying power, duplicate
  positions/orders, PDT, loss limits), and mandatory risk exits are unchanged
  and remain independent of every experimental component above.
