# Oracle 3.0 — Phase 1 Validation Report

Scope: Lab (offline research), Temporal Integrity, Unified Decision/Execution,
Adversarial Thesis, Executable EV. Every new behavior is **flag-gated,
default-OFF, fail-open**; the live path is **byte-identical when all flags are
OFF**. All checks below are offline (no creds, no network, no order placement).

How to reproduce everything: `python run_selftests.py`

---

## A. Determinism, safety, and regression

1. **All module self-tests PASS.** `run_selftests.py` imports the 18 new Phase-1
   modules and runs each `_self_test()`; all return 0.

2. **Live entry points still compile.** `smart_trader.py` and
   `oracle_intelligence_reports.py` byte-compile clean under the runner's
   compile-check stage.

3. **Full test suite passes.** `tests/` (unittest-discoverable, pytest-compatible)
   runs green: `test_decision_parity`, `test_temporal_integrity`,
   `test_adversarial_thesis`.

4. **Flags default OFF / shadow.** `.env.example` ships every Phase-1 flag OFF
   (`EXECUTABLE_EV_SHADOW_MODE=true`). With all OFF the decision path is
   unchanged; each layer is additive and fail-open.

5. **Determinism.** Pure functions (`bayesian_probability`, `build_theses`,
   `adversarial_review`, `compute_executable_ev`, lab experiment/sweep) return
   identical output for identical input — asserted in each module's self-test.

6. **Fail-open everywhere.** Every public entry point returns a safe default
   (None / empty / the un-adjusted probability) on malformed input rather than
   raising — asserted with junk inputs (`None, 42, "x", [], {...}`).

7. **Hard risk never bypassed.** No new layer opens/sizes/prices/blocks a real
   trade directly. Adversarial review can only *add* doubt (raise
   `p_no_trade`); executable-EV is *veto-only* and runs before the risk engine.

## B. Upgrade 1 — Unified decision + swappable execution

8. **Single decision kernel.** `oracle/decision/{schema,direction,kernel}.py`
   expose a shared `decide()` producing a typed `Decision`; direction stays an
   OUTPUT (extracted, never an input).

9. **Swappable execution clients.** `oracle/execution/{client,alpaca}.py` present
   one `ExecutionClient` interface so live/paper/sim differ only by adapter.

10. **Parity.** `tests/test_decision_parity.py` asserts the same kernel yields
    the same decision under the live vs backtest client (decision_live ==
    decision_backtest).

## C. Upgrade 2 — Executable EV + fill model

11. **Three-EV split.** `compute_executable_ev` returns theoretical vs executable
    EV plus spread/slippage/fill-probability/execution-risk.

12. **Wide spread lowers executable EV** (more friction) — asserted; spread_cost
    strictly rises with the quoted width.

13. **+theoretical / −executable is a SUCCESSFUL rejection.** Live evidence:
    ```
    REJECT  theo=+3.00  exec=-39.58  spread=50.000  reasons=('fill_probability<1','executable_negative')
    ```
    A thin +3.00 theoretical edge on a 40c-wide market flips negative once
    frictions are applied → correctly rejected.

14. **+theoretical / +executable is accepted.** Live evidence:
    ```
    ACCEPT  theo=+15.00  exec=+10.45  spread=4.400  entry=1.032  exit=0.988
    CAPTURE {'execution_capture_ratio': 0.697, 'realized_capture_ratio': 0.627, 'model_capture_ratio': 0.90}
    ```
    Entry ≥ mid, exit ≤ mid (conservative); capture ratios computed for
    EV-degradation telemetry.

15. **Calibration telemetry.** `oracle/execution/calibration.py` records
    model/executable/realized EV so `execution_capture_ratio` can be tracked
    over time.

## D. Upgrade 3 — Oracle Lab (offline robustness)

16. **Deterministic sweep + walk-forward + stability.**
    `oracle/lab/{parameter_sweep,walk_forward,parameter_stability}.py` compose on
    `run_experiment`/`compute_metrics`. Sweep never selects a single winner on
    the full sample (every `SweepResult.in_sample_only=True`); walk-forward is
    the OOS certifier and raises `oos_collapse` when TEST expectancy collapses
    vs VALIDATE. Stability flags spike-vs-plateau parameters.

## E. Upgrade 4 — Temporal integrity (look-ahead protection)

17. **Injected look-ahead is rejected.** Live evidence:
    ```
    future daily bar valid? False -> (False, 'available_after_decision')
    strict assert raised LookAheadError: [TEMPORAL] look-ahead: 1 datum(s) stamped after as_of=2024-01-05 ...
    ```
    A future daily bar/earnings/analyst-rating/option-quote is each rejected at a
    decision `as_of` that precedes availability; a rolling SMA uses ONLY bars
    with close ≤ as_of; an intraday feature requires a COMPLETED bar; a forward
    return (realized outcome) cannot enter the feature set. Strict mode raises;
    non-strict logs + returns False (fail-open).

## F. Upgrade 5 — Bull / Bear / No-Trade adversarial thesis

18. **All three theses evaluated on the same evidence.** Confidences seeded
    verbatim from the tally. Live evidence (tally=bear, quant=call, stale
    catalyst, counter-regime):
    ```
    BULL conf=0.20  BEAR conf=0.70  NOTR conf=0.10
    REVIEW flags=['direction_tally_mismatch','stale_catalyst','regime_conflict']
    REVIEW adj={'p_call':0.525,'p_put':0.175,'p_no_trade':0.30} boost=0.100
    leading preserved (call>put): True
    ```

19. **Bounded skeptic, never flips.** The review only raised `p_no_trade`
    (0.20 → 0.30), the boost saturated at the `THESIS_MAX_NO_TRADE_BOOST` cap
    (0.10) despite three flags, direction was preserved (call still leads), and
    the result renormalized to 1.0. An optional LLM provider may add only
    natural-language text/flags — it cannot move the numbers.

20. **Semantic trade memory.** `oracle/trade_memory.py` appends a postmortem
    reflection on close (JSONL, fold-by-id idiom) and retrieves context-only
    lessons by `{symbol, sector, regime, catalyst, strategy_mode, failure_mode}`,
    newest-first. It never overrides a rule.

---

### Rollout posture
Stage 1 (research/offline: Lab, Temporal, Sim) and Stage 2 (shadow: thesis +
executable-EV record "would-do") are supported now. Paper (Stage 3) and gated
promotion (Stage 4) remain flag-guarded; **no live-money enablement** is
included.
