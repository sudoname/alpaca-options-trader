# Oracle 4.0.2 — Experiments and Results

**Historical snapshot:** The results and fingerprints below were produced
before the strict missing-prediction gate was added. They cannot be used to
evaluate the corrected GBM/ensemble variants. Re-run the backtest and
promotion report on the original inputs before claiming a new result.

Every number below comes from a fingerprinted artifact; reproduction commands
are in `docs/oracle402_runbook.md`.

* Backtest artifact: `artifacts/oracle4/replay402.json`
  fingerprint `bc9647af74742085efdd8c5500f650eafd2b8c8d58a46710322e573e9b8803c5`
* Promotion report: `artifacts/oracle4/promotion402.json`
  fingerprint `6394a806efb1e36d132cf4829572e4e59c5d478a253b9eb37d9fa0472f72ec7d`
* EV model study: `artifacts/oracle402/ev_model_report.json`
* Dataset: `artifacts/oracle4/dataset_bt1121.sqlite`
  (hash `ebba3012…`; 32,794 decisions / 426,322 label rows / 12 sessions /
  88 underlyings)

## 1. Comprehensive backtest (P10)

Window: sessions 2026-11-04 → 2026-11-19 (10 replay sessions, 6 with trades).
Holdout sessions 2026-11-12/13 excluded from the universe (67 trades removed);
757 rows scanned, **690 trades kept**. Parity vs the frozen 4.0.1 replayer:
ok, n=690, max_abs_diff=0.0. OOF prediction coverage: classical/GBM/neural each
24,668 predictions, status ok.

### Variant comparison (constrained portfolio, 1.0× calibrated friction)

| Variant | n fills | Net P&L | PF | Sharpe (ann.) |
|---|---:|---:|---:|---:|
| baseline_frozen | 45 | −$3,137.85 | 0.552 | −8.65 |
| baseline_quality | 47 | −$3,420.73 | 0.548 | — |
| classical_gated | 39 | −$1,268.09 | 0.770 | −2.83 |
| neural_gated | 43 | −$3,649.27 | 0.553 | — |
| **gbm_gated** | **35** | **−$333.44** | **0.927** | **−0.88** |
| ensemble_gated | 40 | −$1,231.14 | 0.784 | — |
| valuation_ranked / combined / gbm_combined | 0 | $0.00 | — | — |

`gbm_gated` funnel: 365 candidate opportunities → 318 no-prediction →
47 gate-passed → 113 gated-out → 212 abstained → 35 executed
(win rate 42.9%, expectancy −$9.53/trade, max drawdown −0.88%).

### Friction sweep + stress (gbm_gated)

| Scenario | Net P&L | PF | Expectancy/trade |
|---|---:|---:|---:|
| 1.0× calibrated | −$333.44 | 0.927 | −$9.53 |
| 1.5× | −$1,082.66 | 0.783 | −$30.93 |
| 2.0× | −$1,831.89 | 0.659 | −$52.34 |
| Stop-overshoot stress (+11.30 pts avg) | −$2,631.68 | — | Δ −$2,298.24 |

The edge-vs-baseline does **not** survive higher friction in absolute terms,
and stop overshoot is the dominant tail risk.

### Paired per-session deltas vs baseline_frozen (bootstrap 95% CI)

| Variant | Mean Δ/session | 95% CI | n sessions |
|---|---:|---|---:|
| **gbm_gated** | **+$400.63** | **[+1.29, +905.29]** | 7 |
| classical_gated | +$267.11 | [−186.86, +806.46] | 7 |
| ensemble_gated | +$272.39 | [−168.74, +777.31] | 7 |
| neural_gated | −$63.93 | [−440.56, +375.59] | 8 |
| baseline_quality | −$40.41 | [−95.46, 0.00] | 7 |
| combined / gbm_combined / valuation_ranked (n=0) | +$448.26 | [+41.49, +825.89] | 7 |

Caution on the last row: an n=0 variant "beats" a losing baseline simply by
trading nothing. The promotion gate correctly refuses these (missing PF/
Sharpe/expectancy fail C1/C4/C5; 0 fills fail E3).

### Matched cohorts (same 370 opportunities per arm, bootstrap n=5 sessions)

| Cohort | Mean Δ | 95% CI |
|---|---:|---|
| **GBM-gated vs ungated** | **+$751.19** | **[+118.69, +1,536.59]** |
| Classical-gated vs ungated | +$388.89 | [−126.21, +949.34] |

The GBM gate's matched-cohort CI excludes zero — the strongest single result in
this release — but on 5 bootstrap sessions it is evidence of *direction*, not a
promotable edge. `risk_metrics_limited=True` is stamped in the artifact (single
window, <30 sessions).

### Outcome provenance (trade universe)

counterfactual 21,554 · mark_derived 690 · fills_actual 21 · policy_skip 3,107 ·
missing 2,660. Only mark-derived/actual outcomes enter the replay P&L.

## 2. Model comparison (classical vs GBM vs neural vs ensemble)

Ranking by every backtest measure: **GBM > ensemble ≈ classical ≫ neural**.
The neural gate is indistinguishable from baseline (paired Δ −$63.93, CI
spanning zero, net −$3,649 — worse than no gate), consistent with the 4.0.1
finding of no neural directional edge. The ensemble inherits the neural arm's
noise; intersection-gating diluted rather than improved the GBM signal on this
window.

## 3. EV calibration study (P0-4)

`artifacts/oracle402/ev_model_report.json` — train 98 opportunities /
3 sessions, eval 10 opportunities / 1 session (2026-11-16); lightgbm + torch
available. **Verdict: `INSUFFICIENT_EVIDENCE` (10 < 50 eval opportunities);
`challenger_policy: abstain`** — no EV model replaces the stamp.

Profit models (OOS Spearman vs realized net P&L; MAE USD):

| Model | Spearman | MAE ($) |
|---|---:|---:|
| **lgbm_quantile_q50** | **+0.564** | **243.87** |
| cohort_profit | +0.009 | 269.34 |
| lgbm_profit | −0.255 | 424.68 |
| neural_profit | −0.309 | 1,694.01 |
| stamped_ev (production) | **−0.661** | 341.53 |

PoP models (OOS Brier, lower better): stamped_pop 0.250 · cohort_pop 0.302 ·
base_rate 0.315 · logistic_recal 0.318 · lgbm_pop 0.455.

Reading: the production EV stamp is strongly *anti*-correlated with realized
P&L on this eval slice (−0.661, consistent with the charter's −0.07/
slope −0.266 measured on more data), while the quantile-regression median is
the only positively-correlated ranking signal. The stamped PoP is, however,
still the best-calibrated probability. Action: keep the stamp's PoP, distrust
its dollar EV, and re-run the study once ≥50 eval opportunities accumulate.

## 4. Kalshi ablation (P1-6)

The matched ablation harness (A base / B base+kalshi / C kalshi_only /
D base+placebo / E base+lagged; identical samples, folds, learner,
hyperparameters) is implemented and tested, but **no live Kalshi snapshots
have been collected yet**, so there is no real-data result to report. The
feature builder is strictly point-in-time and fail-open (all-None) — enabling
the collector cannot affect trading. Result will be `kalshi helps` only if
B > A while D ≈ A (placebo clean) and E quantifies staleness decay.

## 5. Promotion verdict (P11)

`artifacts/oracle4/promotion402.json`:

| Variant | Verdict | Failed criteria |
|---|---|---:|
| baseline_quality | **DO_NOT_PROMOTE** (paired CI UB = 0 — confidently no better) | 8 |
| gbm_gated | INSUFFICIENT_EVIDENCE | 6 |
| classical_gated | INSUFFICIENT_EVIDENCE | 7 |
| ensemble_gated | INSUFFICIENT_EVIDENCE | 7 |
| neural_gated | INSUFFICIENT_EVIDENCE | 8 |
| combined / gbm_combined / valuation_ranked | INSUFFICIENT_EVIDENCE | 6 |

**Overall: `INSUFFICIENT_EVIDENCE`.** Binding constraints: 10 sessions < 60
(E1), 35 best-variant fills < 100 (E3), and no variant with positive absolute
expectancy / PF > 1.3 / Sharpe > 1.0 (C1/C4/C5). Nothing was promoted; nothing
in this release can self-promote.

## 6. Conclusions

1. The GBM-gated policy shows a real, CI-backed *incremental* improvement over
   the frozen baseline (+$400.63/session paired; +$751.19 matched-cohort,
   both CIs > 0) — the gap-closure work produced a measurable signal.
2. Absolute expectancy remains negative everywhere; the system should keep
   **not** trading this policy with real size.
3. The production EV dollar stamp is anti-correlated with outcomes and must
   not be used for ranking; the quantile-median challenger is promising but
   under-evidenced (abstain policy correctly engaged).
4. Required next evidence: ~60 forward replay sessions and ≥100 fills under
   identical frozen criteria (runbook §4), plus Kalshi snapshot accumulation
   for the ablation.
