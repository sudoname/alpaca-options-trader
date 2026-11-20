# Oracle 4.0.1 — Experiment Report (actual results)

Branch `oracle4.0.1`, code `93336d05920e2e8c0fba14ce08ba2a918bb51adc` (Stage G checkpoint).
All experiments below were run locally against real research-ledger data pulled from the
production server; nothing in this document is simulated beyond what is explicitly labelled
as a counterfactual replay. No order was placed, no service restarted, no gate relaxed.

Evidence basis for every number in this file:

| basis | meaning |
|---|---|
| `real-ledger` | derived from `server_data/2026-11-20/research_ledger.db` (production shadow ledger) |
| `paper` | fills/quotes from the PAPER Alpaca account (simulated fills; not proof of live execution) |
| `counterfactual` | replayed P&L on the bot's own historical fills; no fills fabricated |
| `fixture` | synthetic unit-test data (used only in the test suite, never in results below) |

All P&L figures are **net** of the measured execution-cost model
(one-way friction fraction `0.01904477825`, stop overshoot `+11.302 pts`, from
`artifacts/oracle4/execution_calibration.json`, fingerprint `1f14ed01…`), which itself
was calibrated on paper fills — see §7 caveats.

---

## 1. Datasets and holdout pledges

| | dataset_v1 (frozen) | dataset_v2 (extended) |
|---|---|---|
| path | `artifacts/oracle4/dataset_v1.sqlite` | `artifacts/oracle4/dataset_v2.sqlite` |
| decisions | 14,264 | 32,794 |
| sessions | 8 (2026-11-04 … 2026-11-13) | 12 (2026-11-04 … 2026-11-19) |
| dataset hash | `f7f662e00b78aeb267953439b341425dd542fc0b2fbf1b4ddf5a827c13e96aa3` | `50b5b783e8676f337039a765d39a8e5038e2582f82ad5e485c60ca31d88af1aa` |
| holdout (pledged, write-once) | 2026-11-12, 2026-11-13 | 2026-11-18, 2026-11-19 |
| registration | `artifacts/oracle4/holdout_registration.json` | `artifacts/oracle401/holdout_registration_v2.json` |

The v1 holdout (11-12/11-13) is **not pristine**: those sessions were examined during
Oracle 4.0 development. For that reason a *new, never-inspected* holdout (11-18/11-19)
was pledged for v2 before any v2 training. `register_holdout` is write-once and refuses
mismatched re-registration (`oracle/lab/purged_validation.py:476`).

v2 label coverage at 60m/direction: total 32,794 → ok 24,170, missing 7,491, censored 1,133.
Fold sessions: the 10 non-holdout sessions; purged walk-forward folds throughout.

Outcome statuses in v2 (non-holdout): counterfactual 17,608 · **mark_derived 757** ·
fills_actual 32 · policy_skip 2,355 · missing 1,765. Only mark-derived/actual outcomes
carry P&L; everything else is accounted, never invented.

---

## 2. Model training on dataset_v2 (seed 42, purged folds, pooled test n = 14,012, base rate 0.38603)

Command pattern (identical for all three; MLP on CUDA):

```
python -m oracle.ml.train --model {logistic|gbm|mlp} \
  --dataset artifacts/oracle4/dataset_v2.sqlite --horizon 60m --label-type direction \
  --out artifacts/oracle401/v2_train_{model}_60m.json --seed 42 \
  --holdout-registration artifacts/oracle401/holdout_registration_v2.json
```

Artifacts: logistic `810fdedf…`, gbm `b47ec5f8…`, mlp `aae3e05e…` (report fingerprints).

> **Comparability warning (recorded in the artifacts verbatim):** binary (up-vs-not-up)
> and 3-class metrics are NOT comparable; the binary collapse merges neutral and bearish
> outcomes. Models are compared only within the same class definition.

### 2.1 Logistic — binary head (ridge logistic + Platt) — **NEGATIVE RESULT**

| model | n | acc | brier | ece | log loss |
|---|---|---|---|---|---|
| challenger (pooled) | 14,012 | 0.5020 | 0.28567 | 0.19812 | 0.77567 |
| constant (base-rate) | 14,012 | **0.6140** | 0.24046 | 0.05962 | **0.67404** |
| momentum baseline | 9,937 | 0.6221 | — | — | 0.67852 |
| production `p_win` | 1,914 | 0.5799 | — | — | 0.69382 |

Per-session bootstrap of Δlog-loss (8 sessions, challenger − baseline sign convention:
negative = challenger worse):

- vs constant: mean **−0.0795**, CI **[−0.1679, −0.0138]** → significantly worse
- vs momentum: mean **−0.0970**, CI **[−0.2080, −0.0055]** → significantly worse
- vs production p_win: mean −0.0429, CI [−0.1500, +0.0420] → inconclusive (12,098 rows have no recorded p_win)

### 2.2 Logistic — 3-class head (multinomial + temperature scaling) — **NEGATIVE RESULT**

| model | acc | log loss | macro-F1 |
|---|---|---|---|
| challenger | 0.37803 | 1.20655 | 0.27470 |
| constant | 0.36562 | **1.04809** | 0.25211 |
| momentum | 0.30140 | 1.07700 | 0.20490 |

Δll vs constant mean −0.1133 CI [−0.2585, −0.0182]; vs momentum −0.1322 CI [−0.3020, −0.0174].
Slightly better accuracy/macro-F1 than constant, but significantly worse calibrated likelihood.

### 2.3 GBM — LightGBM 4.7.0, 3-class — **mixed, mostly negative**

| variant | acc | brier | log loss | macro-F1 |
|---|---|---|---|---|
| native-categorical | 0.34941 | 0.66521 | 1.09598 | **0.33942** |
| missing-flags | 0.35505 | — | 1.09125 | **0.34320** |
| constant | 0.36562 | — | **1.04809** | 0.25211 |

Δll bootstrap (all CIs exclude zero, challenger worse): flags vs constant −0.0462
[−0.0649, −0.0307]; native vs constant −0.0496 [−0.0677, −0.0343]; flags vs momentum
−0.0398 [−0.0664, −0.0149]. Native vs flags +0.0035 (no material difference).
Best macro-F1 of all models (0.343 vs constant 0.252) — the GBM finds minority-class
structure but is poorly calibrated in likelihood.

Quantile head: coverage_50 = 0.5278, coverage_90 = 0.9069; pinball ≈ marginal (no edge).

Net-P&L objective: **refused by design** — artifact records
`"no qualified monetary labels: frozen ledger fees are unknown (None) and exit prices are
sparse; a net-P&L objective would be trained on invented costs"`.

### 2.4 MLP — multitask (CUDA; hidden [128,64,32], dropout 0.1, lr 1e-3, batch 256, ≤40 epochs, patience 5)

Direction head (3-class collapsed metrics on binary panel for comparability with constant):

| model | acc | brier | ece | log loss |
|---|---|---|---|---|
| MLP direction | **0.61619** | **0.23952** | **0.04415** | **0.67280** |
| constant | 0.61397 | 0.24046 | 0.05962 | 0.67404 |

Marginally better than the constant on every metric — the only direction model to do so —
but the margin is small (Δll ≈ 0.0012) and not established as significant.

Hazard head (exit-cause CIF): coherent (`cif_coherent: true`); ΔNLL vs marginal
mean **+0.1937**, CI **[+0.0149, +0.4962]** (positive = model better) → the hazard head
**beats the marginal baseline with a CI excluding zero**. This is the strongest positive
training result in the program.

Quantile head: coverage_50 0.4571 (nominal .50), coverage_90 0.9592 (over-covered vs .90);
Δpinball vs marginal mean −0.0025 CI [−0.0039, −0.0016] → **worse than marginal**. RV head: empty.

### 2.5 Training summary

- Direction prediction at 60m remains ≈ unlearnable on this feature set: 2 of 3 models are
  *significantly worse* than a constant; the MLP is at best marginally better.
- Real signal exists only in (a) GBM minority-class structure (macro-F1) and
  (b) the MLP hazard head (exit-cause timing), not in direction likelihood.

---

## 3. Valuation (scenario engine) on dataset_v2

```
python -m oracle.ml.valuation --dataset artifacts/oracle4/dataset_v2.sqlite \
  --out artifacts/oracle401/v2_valuation.json --seed 42 --n-paths 512
```

Same config as the frozen v1 run (step 5m, 78 steps, CRR 200, block_len 6, EWMA λ=0.94,
rfr 4%, SL 10% / TP 20%, max spread 40%, costs: OCC 0.02 + slip 0.02 per contract).
Fingerprint `2579efac…`. Result: 13,001 scans, 22,517 candidates, 507 trade scans,
12,494 scan-level abstains.

Abstain reasons (candidate level): insufficient_history 19,220 · missing_quote 2,178 ·
missing_underlying_price 17 · iv_below_intrinsic 2.

**Material behaviour change vs v1 code:** the Stage-C valuation builds PIT-clean
5-minute step-return chains (gap-bridged, ≥30 returns required). Under the real ledger's
sparse observation spacing (median ≈ 811 s), that threshold is first met late in the
window — all 507 trade scans fall on **2026-11-16 (44) and 2026-11-17 (463)** only.
329/507 selections were `kshadow-*` (shadow-only candidates), 178 production decision IDs.

---

## 4. Counterfactual replay

Replays execute **only** the bot's own historical fills (mark-derived outcomes);
decisions a variant would have traded that the bot did not trade are counted as
`outcome_unavailable` — fills are never fabricated. Holdout decisions are refused at the
decision level. Accounting parity (engine vs independent recomputation) held exactly in
both runs: v1 `max_abs_diff = 0.0` over 688 trades; v2 `0.0` over 757.

### 4.1 v1 window (`artifacts/oracle401/replay_401.json`, dev sessions 11-04…11-11, holdout 11-12/13 refused, 66 decisions excluded)

| variant | n trades | net P&L @1.0x | PF | paired Δ/session vs frozen [95% CI] |
|---|---|---|---|---|
| baseline_frozen | 688 exec'd / 45 positions | −3,137.85 | 0.552 | — |
| baseline_quality | 676 | −3,420.73 | 0.548 | −40.41 [−95.46, 0.00] |
| classical_gated | 527 | −1,268.09 | 0.770 | +267.11 [−186.86, +806.46] |
| valuation_ranked | 183 (71 unavailable of 254 selected) | −1,510.16 | 0.781 | +232.53 [−706.30, +1456.97] |
| combined | 143 | −3,251.30 | 0.351 | −16.21 [−355.91, +377.56] |
| neural_gated | — | unavailable in this run | — | — |

Matched cohort (identical decisions, n = 368): baseline −3,735.05 vs challenger −1,790.60;
paired mean +388.89/session, CI [−126.21, +949.34] — includes zero.

### 4.2 v2 window (`artifacts/oracle401/v2_replay_401.json`, 10 fold sessions 11-04…11-17, holdout 11-18/19 refused; fingerprint `348aeaa0…`)

```
python -m oracle.ml.replay --dataset artifacts/oracle4/dataset_v2.sqlite \
  --valuation artifacts/oracle401/v2_valuation.json \
  --execution-calibration artifacts/oracle4/execution_calibration.json \
  --out artifacts/oracle401/v2_replay_401.json --holdout 2026-11-18 2026-11-19 --seed 42
```

| variant | n | net @1.0x | PF | net @1.5x | net @2.0x | Sharpe (limited) |
|---|---|---|---|---|---|---|
| baseline_frozen | 52 | −4,302.52 | 0.488 | −5,526.28 | −6,750.04 | −12.35 |
| baseline_quality | 54 | −4,585.40 | 0.489 | −5,888.10 | −7,190.80 | −14.00 |
| classical_gated | 44 | **−1,117.37** | **0.806** | −2,035.56 | −2,953.75 | −2.69 |
| neural_gated | 47 | −4,502.28 | 0.510 | −5,657.43 | −6,812.57 | −9.50 |
| valuation_ranked | **0** | 0.00 | — | — | — | — |
| combined | **0** | 0.00 | — | — | — | — |

Paired Δ/session vs baseline_frozen (8 traded sessions, 2,000-draw bootstrap, seed 7):

| variant | mean | 95% CI | nonzero sessions / ESS |
|---|---|---|---|
| classical_gated | +398.14 | [−92.93, +975.74] | 5 / 3.45 |
| neural_gated | −24.97 | [−422.54, +463.69] | 6 / 2.89 |
| baseline_quality | −35.36 | [−83.53, 0.00] | 2 / 1.86 |
| valuation_ranked / combined | +537.82 | [+125.96, +872.24] | 8 / 6.21 — **degenerate, see below** |

**Matched-cohort result (the cleanest comparison in the program).** On the n = 437
decisions where baseline and classical challenger both had a prediction and an eligible
quote: baseline executed 38 for −4,899.72 (PF 0.251); challenger executed 31 for
−2,215.69 (PF 0.458). Paired per-session uplift mean **+447.34**, 95% CI
**[+17.15, +915.32]** over 6 sessions (ESS 4.06) — the first uplift CI in the program
that excludes zero. Caveats: only 6 sessions, effective sample ≈ 4, CI lower bound barely
above zero, and **both arms lose money** — this is loss *reduction*, not profitability.

### 4.3 Why valuation_ranked and combined executed zero trades on v2 (root cause, not a bug)

1. v2 valuation trade scans exist only on 2026-11-16/17 (§3: strict PIT history gate).
2. Mark-derived outcomes exist only on 2026-11-04…11-13 (150/168/80/105/101/86/66/1 per
   session = 757); candidate resolution was not running on 11-16/17.
3. Intersection = ∅ → accounting records `selected_total 507,
   selected_with_recorded_outcome 0, outcome_unavailable 507`; the replay reports n = 0
   rather than inventing fills. 329/507 selections were shadow-only candidates that were
   never tradable in the baseline universe regardless.

Consequently the +537.82 "paired uplift" for these variants is **degenerate**: it is just
the negation of the baseline's per-session losses (the variant holds cash on every traded
session) and carries **no evidence of selection skill**. It is listed only for accounting
completeness and is excluded from all conclusions. The valid valuation evidence remains
the v1-window run (§4.1: n = 183, PF 0.781, CI includes zero).

### 4.4 Structural limits (recorded in the artifact)

- Selection bias: the replay universe is the baseline bot's own fills; challengers can
  only *drop* trades, so challenger upside requiring non-baseline picks is invisible
  (`outcome_unavailable` counts only).
- Risk metrics are flagged `limited`: positions carried at entry cost between entry and
  exit (no daily option marks) → drawdown/Sharpe understated; < 30 sessions.
- OOF predictions for gating: 8/8 purged folds used, 19,153 OOF predictions for both the
  classical multinomial and the multitask MLP (no fold skipped).

---

## 5. Promotion verdicts (gates never relaxed)

```
python -m oracle.ml.promotion --replay artifacts/oracle401/v2_replay_401.json \
  --execution artifacts/oracle4/execution_calibration.json \
  --variant {combined|classical_gated} --baseline baseline_frozen --out …
```

| run | verdict | pass | fail | insufficient |
|---|---|---|---|---|
| v1 window, combined (`promotion_401.json`) | INSUFFICIENT_EVIDENCE | 2 | 0 | 13 |
| v2 window, combined (`v2_promotion_401.json`, fingerprint `b9865570…`) | INSUFFICIENT_EVIDENCE | 2 | 0 | 13 |
| v2 window, classical_gated (`v2_promotion_classical.json`) | INSUFFICIENT_EVIDENCE | 2 | 0 | 13 |

Passing gates: `accounting_parity` (exact) and `holdout_refusal_demonstrated`. Every
performance gate is INSUFFICIENT because **no validated forward-evidence artifact exists**
(preregistered minimums: 60 forward sessions, 200 resolved opportunities, 100 actual
fills; PF > 1.3; Sharpe > 1.0; net-positive at 1.5× friction; bootstrap LBs > 0).
Replay-only evidence cannot promote by design. Observed forward sessions: 10.

**Promotion verdict: DO NOT PROMOTE. No variant is eligible; no variant is net-profitable.**

---

## 6. Comparison against the frozen Oracle 4.0 baseline

Direct numeric comparison is **not period-matched** and is flagged as such:
frozen 4.0 figures were computed at code `a6eea43` over 6 dev sessions (688 mark-derived
trades); 4.0.1 v2 figures are at `93336d0` over 10 fold sessions (757 trades) with
repaired accounting (Stages A–F changed timestamp handling, label provenance, exit/cost
parity, and cohort accounting — some deltas are *measurement* corrections, not market).

| metric | frozen 4.0 (6 sessions) | 4.0.1 v2 (10 sessions) |
|---|---|---|
| baseline_frozen | 45 trades, −3,138, PF 0.55 | 52 positions, −4,303, PF 0.49 |
| classical_gated | 42, −2,623, PF 0.62 | 44, −1,117, PF 0.81 |
| neural_gated | 41, −2,751, PF 0.60 | 47, −4,502, PF 0.51 |
| valuation_ranked | 41, −1,510, PF 0.78 | 0 (coverage mismatch, §4.3) |
| combined | 39, −1,123, PF 0.82 | 0 (coverage mismatch, §4.3) |
| best uplift CI | all include zero | matched-cohort classical: [+17.2, +915.3] excludes zero |
| 60m classical acc | 0.536 | 3-class 0.378 (binary head 0.502 — worse than constant) |
| 60m MLP | acc 0.415 | acc 0.616 (vs constant 0.614); hazard head beats marginal |
| EV calibration | pred +51.9 vs real −57.8, corr −0.07 | slope −0.266 recorded in execution calibration (unchanged artifact) |
| stop overshoot | 525/527 overshot | same calibration input (+11.302 pts stress applied) |

Honest reading: 4.0.1 did not make any strategy profitable. It made the measurements
trustworthy (exact parity, holdout refusal, provenance-bound artifacts, refusal to
fabricate), surfaced one modest but statistically supported effect (classical gate
reduces losses on a matched cohort), one genuine model signal (MLP hazard head), and
several cleanly documented negative results.

---

## 7. Caveats that bound every conclusion

1. All execution evidence is from a PAPER account or never-submitted shadow intents —
   not proof of attainable live execution (recorded verbatim in the promotion artifact).
2. EV calibration slope is −0.266: the valuation's EV estimates are anti-correlated with
   realized P&L at current calibration; EV magnitudes must not be trusted.
3. ≤ 10 sessions everywhere; ESS of paired tests 1.9–6.2. One regime.
4. The v1 holdout was contaminated by 4.0 development; only the v2 holdout
   (11-18/11-19) is pristine and it has **never been evaluated against**. It remains
   pledged for one final evaluation after all choices are frozen.
5. Friction stress is monotone adverse: classical_gated stays net-negative at 1.0×/1.5×/2.0×.

## 8. Reproduction

All artifacts in `artifacts/oracle401/` (git-ignored; fingerprints above bind them to
dataset hashes and code SHA). Exact commands in §§2–5. Seeds: training/valuation/replay 42,
promotion bootstrap 7, n_boot 2,000. MLP device: CUDA (RTX 3060 Ti); CPU fallback
produces the same artifact schema (determinism is per-device for torch).

## 9. Next experiments (not started)

1. Run candidate resolution continuously so future sessions accrue mark-derived outcomes
   on the same days the valuation can trade (fixes the §4.3 coverage mismatch forward).
2. Accrue ≥ 60 forward sessions of per-book forward evidence (Stage G `oracle/ml/forward.py`)
   before touching the pledged v2 holdout.
3. Recalibrate EV (slope −0.266) before any valuation-ranked variant is considered again.
4. Investigate the MLP hazard head as an *exit-timing* input (the only CI-positive model
   signal), not as a direction signal.
