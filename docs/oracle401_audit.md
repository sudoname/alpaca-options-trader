# Oracle 4.0.1 Remediation Audit

Status date: 2026-11-20 (UTC)
Branch: `oracle4.0.1` (based on `oracle4-research`)
Current HEAD: `93336d0` — "Oracle 4.0.1 Stage G: strategy books, forward evidence per book, hedge-aware order investigation"
Companion document: `docs/oracle401_experiments.md` (all empirical results, CIs, hashes)

This audit maps every defect class in the Oracle 4.0.1 master prompt to its fix,
commit, and verification evidence, and classifies each item as **implemented**,
**tested**, **validated** (exercised on real historical data end-to-end), or
**blocked** (cannot be completed with data/infrastructure available offline).

---

## 1. Baseline freeze

All comparisons reference the frozen Oracle 4.0 baseline, unchanged by 4.0.1:

| Item | Value |
|---|---|
| Baseline manifest | `docs/oracle4_baseline_manifest.json` (manifest_version 1) |
| Baseline code SHA | `8af6d5bd6d91bfc473ade43a21f1e413c863603e` (oracle4-research) |
| Frozen ledger | `research_ledger_server_2026-11-13.db`, sha256 `d8ff8bc3ccbe8693ce336b70d2476803b0af1040943490e465696ef1316d9d89` |
| Frozen decisions export | `export_decisions.jsonl`, sha256 `c90cf1e47f28ab3fbb9b8dae5c9fef57cb97f1be021dc4421ebcdc001075f34a` |
| Candidate resolution labels | sha256 `ed7c9ec559c870b4df9e2345016de53ccf26456c0e67a72be4226b8cb55073bd` (70 lines; never committed per standing rule) |
| Oracle 4.0 end commit | `7edd1e6` (oracle4-research HEAD when 4.0.1 began) |
| Frozen 4.0 benchmark | 6 dev sessions, 688 mark-derived trades: baseline_frozen −$3,137.85 (PF 0.55); best 4.0 variant combined −$1,123 (PF 0.82); all paired CIs include zero |

The baseline replay variant `baseline_frozen` is re-derived in every 4.0.1
replay from the same frozen inputs; accounting parity holds exactly
(max_abs_diff 0.0) in both the v1-window replay (n=688) and the v2 replay
(n=757).

## 2. Production trading path — verified unchanged

```
git diff --stat 7edd1e6..93336d0 -- run_alpaca_intraday.py strategy.py \
    order_manager.py scheduler.py single_leg_reports.py
(empty output)
```

The full 4.0.1 diff is 49 files, +10,361 / −505 lines, confined to
`oracle/ml/*`, `oracle/lab/*`, `tests/`, and `docs/`. The only production
touchpoint remains the pre-existing Oracle 4.0 shadow hook in
`run_alpaca_intraday.py` (~16 lines, env-gated `ENABLE_ORACLE4_SHADOW`,
default OFF, fail-open) — not modified by 4.0.1. No order-placement API is
imported anywhere under `oracle/ml/` (enforced by test). No gates were
relaxed; no live flags, sizing, exits, or entry logic changed.

## 3. Stage-by-stage defect-to-fix mapping

### Stage A — PIT data integrity (P0) — commit `a7887f4`

| Defect (master prompt) | Fix | Evidence |
|---|---|---|
| `available_at` bypass allowed look-ahead | Hard PIT enforcement in dataset builder; every feature row carries `available_at` and queries are `available_at <= as_of` only | `tests/test_oracle401_dataset.py`; dataset builds reproduce fixed hashes (v1 `f7f662e0…`, v2 `50b5b783…`) |
| Synthetic bar grid / rescaled-daily bars | Removed; only real timestamped underlying observations from the ledger and user-supplied SPY bar files, each with a provenance manifest | dataset `meta` table records sources + hashes |
| Unbounded LOCF staleness | Bounded LOCF (`max_gap_steps`); gaps beyond bound yield missing, not carried values | unit tests on gap bridging |
| `_first_touch_label` used marks as fill evidence | Label types separated: `fills_actual` (32), `mark_derived` (757), `counterfactual`, `policy_skip`, `missing`; first-touch on marks can never be recorded as a fill | outcome-status accounting in `v2_replay_401.json` (`trade_universe`) |
| Censoring ignored | Explicit censored label state (v2 60m/direction: ok 24,170 / missing 7,491 / censored 1,133) | training reports carry the mask counts |

Status: **implemented, tested, validated** (both datasets built from real data;
hashes pinned in experiments doc).

### Stage B — verified model inference (P0) — commit `cfe5581`

| Defect | Fix | Evidence |
|---|---|---|
| Inference path not provably the trained model | Shared `ForecastProvider` interface used by training eval, valuation, and replay; registry entries carry artifact hash + schema; load verifies both | `tests/test_oracle401_provider.py` parity test: provider outputs == training-eval outputs bit-for-bit |
| Silent fallback to heuristics | Explicit abstention with reason codes; no silent default predictions | abstain reasons surfaced in valuation artifact |
| Shadow worker could block trading | Shadow path non-blocking and default disabled | flag default OFF; fail-open hook unchanged |
| Order API reachable from ML code | Import ban test (no alpaca order modules importable from `oracle/ml`) | negative test in suite |

Status: **implemented, tested, validated** (provider drove all v1/v2 valuations
and replays).

### Stage C — forecast-conditioned valuation (P0) — commit `e22f8a5`

| Defect | Fix | Evidence |
|---|---|---|
| Hard-coded drift / fabricated dynamics | Scenarios generated from model forecast with common random numbers (CRN) across candidates | `v2_valuation.json` scenario diagnostics |
| IV handling unsound | Sticky-IV stress; `iv_below_intrinsic` abstention (2 occurrences in v2) | abstain-reason accounting |
| Exit policy mismatch vs production | Valuation applies the actual exit policy, DTE rules, and exercise handling | exit/cost parity tests |
| EV ignored fill probability | EV = P(fill)·E(net|fill); null-vs-zero distinction; fees applied exactly once | `tests/test_oracle4_valuation.py` (56 lines of assertions) |
| Only top candidate valued | All candidates ranked per scan (22,517 candidates over 13,001 scans in v2) | valuation counts |
| Insufficient PIT history silently tolerated | `min_history_returns = 30` on PIT-clean gap-bridged 5-min step returns (`oracle/ml/valuation.py:133`, check at `:462`) | v2: 19,220 `insufficient_history` abstentions vs 991 in v1 — stricter, documented behaviour change (experiments doc §3) |

Status: **implemented, tested, validated.** Note: the stricter history gate
restricted v2 trade scans to 2026-11-16/17, which is the root cause of the
valuation_ranked/combined n=0 replay cohort (see §5, Blocked/limits).

### Stage D — P&L reconciliation + calibration (P0) — commit `5b32490`

| Defect | Fix | Evidence |
|---|---|---|
| Double-counted fees, duplicate orders | Reconciliation dedupes orders, weights by fills, books fees exactly once, reconciles cash vs realized | reconciliation tests |
| EV/PoP never validated out-of-sample | OOS calibration report; **result: EV slope −0.26601, corr −0.07 → `NO_VALIDATED_EDGE` declared, not hidden** | `artifacts/oracle4/execution_calibration.json` (fingerprint `1f14ed01…`), embedded in promotion artifacts |
| Fill probability estimated on non-submitted intents | Fill-prob computed only on actually submitted orders (limit TP fill rate 0.6186 raw, PAPER-only caveat attached) | provenance note in calibration artifact |
| Stop overshoot unexamined | Overshoot decomposition (4.0 finding: 525/527 stops overshoot) retained and stress-tested in replay | `stress_2x_and_overshoot` gate |

Status: **implemented, tested, validated** (the negative calibration result is
itself the validated output).

### Stage E — honest direction modeling (P1) — commit `b4cffb7`

| Defect | Fix | Evidence |
|---|---|---|
| Binary up-vs-not-up conflated neutral/bearish | 3-class `_direction_y` + `gate_direction`; binary retained only with explicit non-comparability note | training reports carry verbatim comparability warning |
| GBM was a stub | Real LightGBM 4.7.0 integration | `v2_train_gbm_*.json` (native + flags variants) |
| Label/mask mismatches across heads | Matched labels/masks per head; class weights computed on train folds only | unit tests |
| 2-class vs 3-class accuracy compared | Forbidden in reports; models compared only within the same class definition | experiments doc §2 |

Status: **implemented, tested, validated.** Validated results are mostly
*negative*: logistic and GBM direction models significantly worse than the
constant predictor on paired Δlog-loss (CIs exclude zero on the wrong side);
MLP direction marginally beats constant (acc .6162, Brier .2395); MLP hazard
head is the one CI-positive model result (ΔNLL +0.1937, CI [+0.0149, +0.4962]).

### Stage F — paired replay + coherent promotion (P0) — commit `6f14548`

| Defect | Fix | Evidence |
|---|---|---|
| Unpaired, cohort-confounded comparisons | Matched-cohort analysis separating no-prediction decisions (318 excluded) from gated decisions; session-paired bootstrap (2,000 iters, seed 7) | `v2_replay_401.json` `matched_cohort`: classical gate +$447.34/session, CI [+17.15, +915.32], n=437, 6 sessions |
| Promotion metrics could mix sources | Immutable forward-evidence artifact is the single source for promotion metrics; replay/paper/mark-derived data cannot satisfy fill gates | promotion gate `evidence_basis` → INSUFFICIENT with observed `mark_derived_replay_outcomes: 757` |
| Stress gates OR-ed / optional | 2x friction AND stop-overshoot both required | `stress_2x_and_overshoot_examined` gate |
| Human review spoofable by a string | Named review protocol (`reviewer_name`, `decision`, `report_fingerprint` match) verified by `oracle.ml.evidence.verify_human_review` | promotion artifact `human_review_protocol` |
| Verdict taxonomy conflated fail/insufficient | PASS / FAIL / INSUFFICIENT per gate; overall INSUFFICIENT_EVIDENCE never upgraded by thin samples | both v2 promotion artifacts: 2 pass / 0 fail / 13 insufficient |
| Holdout contamination (v1 11-12/13 used) | Fresh write-once holdout registration: v2 = 2026-11-18/19, pledged in `artifacts/oracle401/holdout_registration_v2.json`, never evaluated | replay `holdout_refusal` PASS (sessions listed, exclusions counted); negative tests for tamper attempts |

Status: **implemented, tested, validated** (replay + promotion run end-to-end
on real data; verdicts honest: INSUFFICIENT_EVIDENCE for every variant).

### Stage G — strategy books + forward pipeline (P1) — commit `93336d0`

| Defect | Fix | Evidence |
|---|---|---|
| 0DTE/swing/multi-leg P&L aggregated across legs | Strategy books (0DTE / swing / multi-leg) with per-book accounting; no leg aggregation | `tests/test_oracle401_books.py` (33 passed) |
| Nov-18 0DTE session misusable as training data | Pinned as reconciliation fixture only | fixture test |
| Duplicate orders vs hedges indistinguishable | Hedge-aware duplicate investigation | book tests |
| No forward-evidence accrual path | Per-book forward evidence writer (observe-only), observer runbook (`docs/oracle4_runbook.md` update), dashboard summary, new-prospective-window rule | `tests/test_oracle401_e2e.py` (5 passed) |

Status: **implemented, tested.** Forward-evidence accrual is **fixture-tested
only** — no genuine forward sessions have been collected yet (observed 10
replay sessions vs 60 required forward sessions), so this stage is not yet
*validated* on real forward data by design.

## 4. Classification summary

**Implemented + tested + validated on real historical data:**
- PIT dataset builds v1/v2 (hashes pinned)
- ForecastProvider parity inference
- Forecast-conditioned valuation (v1 + v2 artifacts)
- P&L reconciliation and OOS EV/PoP calibration (negative result declared)
- 3-class training battery (logistic/GBM/MLP; mostly negative results)
- Paired replay with matched cohort, friction sweeps, parity, holdout refusal
- Promotion reports with coherent verdict taxonomy (all INSUFFICIENT_EVIDENCE)

**Implemented + tested (fixtures only), not yet validated:**
- Forward-evidence accrual per strategy book (needs real forward sessions)
- Shadow worker integration beyond the existing 4.0 hook (default OFF)
- Named human-review verification flow (no promotion candidate exists to review)

**Blocked (cannot complete offline / with current data):**
- 1–3 year multi-underlying bar backfill: only the user-supplied SPY sample is
  available; Alpaca provides no historical option bid/ask backfill.
- ≥100 actual fills gate: only 32 `fills_actual` outcomes exist.
- ≥60 forward sessions gate: 10 sessions exist, all replay (not forward).
- Final v2 holdout evaluation (2026-11-18/19): deliberately untouched per the
  write-once pledge; must remain so until a pre-registered final evaluation.
- Live-execution validation: all execution evidence is PAPER; paper fill rates
  bound internal consistency only.

**Explicitly NOT done (per prompt rules):**
- No promotion, no registry changes, no flag changes, no deploys/restarts,
  no pushes, no gate relaxation, no fabricated or extrapolated results.

## 5. Known limits and honest negatives

1. **No variant is net-profitable.** Best replay variant (classical_gated,
   v2 window) loses −$1,117 at 1.0x friction and degrades monotonically to
   −$2,954 at 2.0x. The matched-cohort uplift (+$447.34/session, CI LB +17.15)
   is *loss reduction*, not profit.
2. **valuation_ranked/combined n=0 in v2 replay** is a documented data-overlap
   artifact (strict PIT history gate → trade scans only 11-16/17; mark-derived
   outcomes only 11-04…11-13; intersection empty). The +537.82 "paired uplift"
   for these variants is degenerate (cash vs losing baseline) and excluded
   from all conclusions.
3. **EV calibration slope is negative (−0.266):** predicted EV is
   anti-correlated with realized P&L → `NO_VALIDATED_EDGE` stands.
4. Effective sample sizes are tiny (ESS 1.9–6.2 sessions), one market regime,
   paper account only.

## 6. Test evidence

| Battery | Result |
|---|---|
| Full suite at HEAD `93336d0`: `python -m pytest -q --ignore=tests/test_safety.py` | **2560 passed, 3 failed (all pre-existing)**, ~116 s |
| Pre-existing failures (also present at baseline `8af6d5b`) | `test_oracle_intelligence_reports.py::TestTelegramDispatch::test_eight_commands_dispatch`; `test_safety.py::TestEVEntryGate::test_disabled_skips_gate`; `test_safety.py::TestEVEntryGate::test_passes_when_above_floor` (env-dependent) |
| `tests/test_oracle401_books.py` | 33 passed |
| `tests/test_oracle401_e2e.py` | 5 passed |
| Dashboard selftest `python oracle_dashboard.py --selftest` | PASS |

No credentials, raw server data, or banned files are staged or committed;
`candidate_resolution.jsonl` remains modified-unstaged per standing rule.

## 7. Commit ledger (oracle4.0.1)

| Stage | Commit | Subject |
|---|---|---|
| A | `a7887f4` | PIT dataset integrity |
| B | `cfe5581` | Verified model inference via shared ForecastProvider |
| C | `e22f8a5` | Forecast-conditioned valuation + exit/cost parity |
| D | `5b32490` | P&L reconciliation + out-of-sample EV/PoP calibration |
| E | `b4cffb7` | 3-class direction models + real LightGBM |
| F | `6f14548` | Immutable forward evidence, coherent promotion gates, replay cohort accounting |
| G | `93336d0` | Strategy books, forward evidence per book, hedge-aware order investigation |

Final verdict: **remediation complete; evidence honest; nothing promotable.**
Promotion requires ≥60 genuine forward sessions and ≥100 actual fills accrued
through the Stage G observe-only pipeline before any gate can move past
INSUFFICIENT.
