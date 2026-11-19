# Oracle 4.0 — Experiment & Checkpoint Log

All work on branch `oracle4-research`. Baseline freeze:
`8af6d5bd6d91bfc473ade43a21f1e413c863603e` (see
`docs/oracle4_baseline_manifest.json`). Research-only: nothing in
`oracle/ml/` is imported by any production trading path.

---

## Phase 0 — Gap audit & frozen baseline (committed `4e7a56f`)

- `docs/oracle4_gap_audit.md`, `docs/oracle4_baseline_manifest.json`.
- Test baseline: `python -m pytest -q --ignore=tests/test_safety.py` →
  **2082 passed, 3 failed** (all three pre-existing and recorded in the
  manifest; none touched by this work).

---

## Phase 1 — Dataset & labels (point-in-time, honest missingness)

### Code

- `oracle/ml/schema.py` — typed records, strict UTC parsing (naive
  timestamps rejected), `OutcomeStatus` / `LabelStatus` taxonomies,
  canonical JSON + order-independent `rows_hash`.
- `oracle/ml/ingest.py` — read-only (`mode=ro` URI) ledger readers with
  keep-first dedup and per-reason drop counters. Nullable
  `events.event_at_utc` falls back to `received_at_utc` (counted, never
  invented).
- `oracle/ml/features.py` — `ObsIndex` with availability-aware as-of
  joins (feature inputs require `event_at <= as_of` AND
  `available_at <= as_of`); trailing returns with staleness bounds,
  session RSI/EMA/RV, contract features (moneyness, DTE, spread), ledger
  signals, `baseline_*`-prefixed baseline opinions.
- `oracle/ml/labels.py` — horizons 15m / 60m / 1-session / 5-session:
  log return, direction (neutral band 0.001), future RV, IV change,
  first-touch time-to-event from exit/outcome events. Explicit
  `CENSORED` (target beyond coverage) vs `MISSING` (gap inside
  coverage, with `meta.why`). `classify_outcome` maps ledger evidence to
  FILLS_ACTUAL / MARK_DERIVED / CONFIRMED_NONFILL / POLICY_SKIP /
  COUNTERFACTUAL / MISSING.
- `oracle/ml/dataset.py` — CLI `python -m oracle.ml.dataset --ledger ...
  --output ... --manifest ...`; sqlite store + JSON manifest with ingest
  reports, coverage, label-status matrix, named gaps, deterministic
  `dataset_hash`.

### Acceptance tests

`tests/test_oracle4_dataset.py` — **33 passed** (timestamp edge cases,
dedup counting, read-only enforcement, no-future-joins incl.
late-availability exclusion and boundary inclusion, label censoring vs
missing, outcome taxonomy incl. all MISSING stages, end-to-end
reproducibility, window filtering, CLI smoke).

Full suite after Phase 1: 2115 passed, 3 failed — the same 3
pre-existing failures as the Phase 0 baseline; zero regressions.

### Real-data build (frozen ledger `research_ledger_server_2026-11-13.db`)

```
dataset_hash = f7f662e00b78aeb267953439b341425dd542fc0b2fbf1b4ddf5a827c13e96aa3
decisions=14264  features=14264  labels=185432  outcomes=14264  sessions=8
underlyings=88   underlying_obs=13051
```

Rebuild from the same ledger reproduces the identical hash.

Outcome taxonomy on real data: `counterfactual=11738` (shadow),
`policy_skip=1213`, `mark_derived=754`, `missing=559`,
**`fills_actual=0`** — fills-based net-dollar booking only began at the
baseline freeze, so every realized P&L in this ledger is mark-derived
(gross % known, net dollars null). All P&L conclusions from this
dataset are therefore mark-basis until newer ledger data accrues.

Named gaps (from the manifest, verbatim mechanism):

- 15m labels ~68% unresolved — median same-underlying observation gap is
  ~16 min, so a 15-min horizon with 1.5× tolerance often has no
  qualifying future observation. 60m and session labels are much denser.
- `iv_change` effectively unavailable (IV present on only 98/14264
  rows); `iv`, `delta`, `sig_*` 5–6% coverage (candidates only);
  quote/moneyness/DTE ~18% (quotes recorded for candidates only).
- 5s horizon ~64% censored: only 8 sessions of coverage.
- `first_touch` resolved only for the 754 filled-and-exited decisions.

### Stop-loss overshoot study (P0)

Question: do realized losses exceed the configured stop floor?
Data: 754 realized exits; 527 `dynamic_stop_loss`, 211
`dynamic_trailing_stop`, 16 roll exits. Configured
`exit_policy.stop_loss_pct` from each decision's `prediction_json`
(mostly 0.15, dynamic range 0.10–0.14). Basis: `gross_pnl_pct`
(fraction of entry premium); net dollars unavailable (see above).

Overshoot := (−stop_loss_pct) − gross_return. Positive = realized worse
than the floor.

| metric | hard stops (n=527) |
|---|---|
| exits worse than floor | **525 / 527 (99.6%)** |
| mean overshoot | **+5.4 pts of premium** |
| median | +2.6 pts |
| p75 / p90 / p99 | +6.7 / +13.8 / +34.4 pts |
| worst | +41.3 pts (−13% stop realized −54.7% gross) |
| aggregate excess gross loss | ≈ 28.3× premium across 525 trades |

Trailing-stop exits never breach the hard floor (worst gross −10.4%;
mean +18.5%).

Interpretation: the hard stop is evaluated on the scan cadence
(~16-min median observation gap), so by the time the stop fires the
mark has typically moved several points past the floor; fast moves
produce tail overshoots of 30–40 points. This is a real, quantified
execution cost, not a modeling artifact. Remediation options (tighter
exit-check cadence, broker-side stop orders, overshoot-aware stop
placement) are **production changes and out of scope here**; Phase 6
will instead calibrate the exit-path simulator to this measured
overshoot distribution so challenger EV never assumes the stop floor is
achievable.

### Artifacts

- `artifacts/oracle4/dataset_v1.sqlite` + `dataset_v1_manifest.json`
  (local artifacts; not committed — rebuildable from the frozen ledger
  via the CLI, hash-verified).

---

## Phase 2 — Purged session-based walk-forward validation

### Code

- `oracle/lab/purged_validation.py` — session-level expanding-window
  folds (whole exchange sessions, never row fractions); grouping by
  `decision_id` with a hard `AssertionError` if any group appears in
  both train and test; **purging** of train rows whose `label_end` or
  `label_available_at` reaches into the test window; **embargo** (per
  horizon: 1 session, 5 for the 5-session horizon) dropping train rows
  with *unknown* label end near the test block; fold-train-only
  `Standardizer` (None stays None — no imputation); DTE buckets
  (0dte / short / swing / unknown); deterministic
  `metrics_fingerprint`; whole-session paired bootstrap
  (n=2000, seed 42); **negative controls** (future-feature probe must
  detect leakage at ≥0.95 accuracy, shuffled labels must collapse to
  chance) via a dependency-free ridge-logistic probe; **pre-registered
  write-once holdout** (`register_holdout` refuses any mismatch against
  the original pledge).
- CLI: `python -m oracle.lab.purged_validation --dataset ... --horizon
  60m --label-type direction --holdout-sessions 2 --register-holdout
  artifacts/oracle4/holdout_registration.json --out ...`.

### Acceptance tests

`tests/test_oracle4_validation.py` — **23 passed** (chronology,
group-leak assertion, label-end/availability purging, unknown-end
embargo semantics, standardizer missingness, DTE buckets, fingerprint
determinism, bootstrap determinism, both negative controls, holdout
write-then-verify and mismatch refusal, end-to-end CLI with
registration). Combined oracle4 suite: 56 passed.

### Real-data fold reports (dataset `f7f662e0…`, seed 42)

Holdout pledged **2026-11-12, 2026-11-13** (4,694 samples) in
`holdout_registration.json` — reserved for one final frozen evaluation;
never used for training/tuning. Fold sessions: 2026-11-04 … 2026-11-11
(4 expanding folds, test block = 1 session).

| horizon | label_ok | censored | missing | folds | purged (total) | controls |
|---|---|---|---|---|---|---|
| 60m direction | 9,810 | 669 | 3,785 | 4 | 0 | **PASS** (leak probe 1.00; shuffle 0.499 vs base 0.551) |
| 1s direction | 11,054 | 3,166 | 44 | 4 | 0 | **PASS** (leak probe 1.00; shuffle 0.513 vs base 0.596) |
| 5s direction | 5,130 | 9,123 | 11 | 4 | 17,138 | **NOT RUN** (0 usable control folds) |

Report fingerprints: 60m `21eb7292…`, 1s `895f21f4…`, 5s `e351a2b1…`.
27 features per row.

Named finding: the **5-session horizon is not validatable yet** — with
only 8 sessions of coverage, nearly every train row's 5-session label
window overlaps the test block, so purging removes 3.3k–5.1k rows per
fold (train shrinks to 26–4,788) and the negative controls have no
usable folds. This is the purge working as designed, not a bug; the 5s
horizon is excluded from challenger evaluation until more sessions
accrue. 60m and 1-session folds purge zero rows because those label
windows resolve within the same session.

### Artifacts

- `artifacts/oracle4/folds_{60m,1s,5s}_direction.json`,
  `holdout_registration.json` (local; regenerable via the CLI —
  fingerprints recorded above).

---

## Phase 3 — Classical challengers vs naive comparators

### Code

- `oracle/ml/baselines.py` — numpy-only, deterministic, full-batch GD
  (no sklearn/lightgbm/torch): design matrix with standardized features
  **plus `*_missing` indicator columns** (missing = 0 after standardize
  + flag 1; never silently imputed); `RidgeLogisticModel`; Platt
  recalibration fit on a chronological tail (last train session);
  reliability bins / ECE; naive comparators returning `(probs, mask)`
  so every comparison is on the rows where the comparator is actually
  available — `constant` (train base rate), `momentum` (`ret_60m` sign
  buckets), `baseline_p_win` (the production Bayesian opinion);
  `QuantilePinballModel` (τ ladder 0.05–0.95, warm-started at marginal
  quantiles, per-row sort for non-crossing), pinball + interval
  coverage; RV forecasters — persistence, per-underlying point-in-time
  EWMA (λ=0.94, carry-forward never updates state on missing), HAR-lite
  (ridge-stabilized lstsq, prediction clamped ≥0), QLIKE (skips
  non-positive); discrete-time competing-risks hazards — person-period
  expansion over bins (0,30,60,120,240,480,1440,∞] min, multinomial
  logistic with time-bin dummies, causes stop/target/other_exit with
  right censoring, cumulative incidence, Laplace-smoothed marginal
  per-bin table comparator, empirical Aalen–Johansen CIF.
- `oracle/ml/train.py` — CLI `python -m oracle.ml.train --model
  {logistic,naive,quantile,rv,hazard,gbm,mlp} --dataset ... --horizon
  ... --label-type ... --out ... --seed 42 --device auto
  [--holdout-registration ...]`. Evaluates strictly through the Phase 2
  harness (purged folds, embargo, fold-train-only standardizer, holdout
  verified against the write-once pledge on every run). Per-fold
  challenger/comparator metrics on identical masked rows; per-session
  loss deltas (comparator − challenger, positive = challenger better)
  fed to the whole-session paired bootstrap (n=2000, seed 42); per-DTE
  and per-kind calibration breakdowns; deterministic
  `report_fingerprint`. `--model gbm` reports LightGBM unavailable
  honestly; `--model mlp` reports `deferred` (Phase 4).

### Acceptance tests

`tests/test_oracle4_baselines.py` — **37 passed** (design-matrix
missingness encoding, logistic separability, Platt slope recovery,
reliability/ECE hand cases, all comparator mask semantics, quantile
non-crossing/coverage/conditional-beats-marginal, EWMA point-in-time
order-invariance, HAR exact recovery of a planted model, QLIKE guards,
person-period expansion hand case, planted-signal hazard model beats
marginal NLL, CIF monotone/bounded, Aalen–Johansen hand cases, and 9
end-to-end CLI tests incl. determinism, fold-config override, and
holdout-tamper refusal). Combined oracle4 suite: 93 passed. Full suite:
**2175 passed, 3 failed** — same 3 pre-existing failures; zero
regressions.

### Real-data results (dataset `f7f662e0…`, seed 42, holdout untouched)

All numbers are purged walk-forward test folds (sessions 11-06…11-11);
the pledged holdout (11-12/13) was verified but never scored. Deltas
are per-session log-loss (or QLIKE/NLL) improvements over the
comparator with whole-session paired bootstrap 95% CIs; positive =
challenger better.

**Direction — 60m (fp `3dfb5ece…`, n=4,174):** challenger (ridge
logistic + Platt) acc 0.536 / log loss 0.743 / ECE 0.134 does **not**
beat the constant base rate (ll 0.708; Δ mean −0.032, CI [−0.076,
+0.013]) or momentum (Δ −0.025, CI [−0.121, +0.061]); vs
`baseline_p_win` (available on 534 rows) Δ +0.025, CI [−0.049, +0.099]
— inconclusive. Honest read: **no 60m directional edge from these
features.**

**Direction — 1-session (fp `ead8b5cb…`, n=6,190):** challenger acc
0.635 vs base rate 0.525, log loss 0.668, ECE 0.077. Beats **all three
comparators with CIs clear of zero**: vs constant Δ +0.062 [+0.014,
+0.104]; vs momentum Δ +0.053 [+0.016, +0.101]; vs baseline_p_win
(785 rows) Δ +0.099 [+0.036, +0.178]. Calibration holds across strata
(candidate ECE 0.073 / shadow 0.085). Only 4 test sessions — promising,
not promotion evidence.

**Quantile log-return (fps `3f8c9961…` 60m / `1d14007e…` 1s):** the
conditional model does **not** beat the marginal-quantile comparator
(60m Δ pinball −0.00041 [−0.00114, +0.00001]; 1s −0.00016 [−0.00074,
+0.00018]) and its intervals under-cover (90% interval: 0.84 vs
marginal 0.94 at 60m). Marginal quantiles are the better uncertainty
model on this data so far.

**RV — 60m future_rv (fp `d2bb3d22…`):** EWMA and HAR both crush
persistence on QLIKE (10.05 → 1.08 / 1.15; Δ ≈ +11.2, CIs [+2.4,
+27.1]). HAR has the best RMSE (0.0019 vs EWMA 0.0029); EWMA the best
QLIKE. Persistence of a 16-min-cadence session RV is simply a bad
forecaster — the useful result is that cheap EWMA is already near-HAR.

**Hazards — first_touch (fp `7d41b0d2…`, 1,367 usable subjects; test
events: stop 367 / other_exit 210 / target 3):** model NLL beats the
marginal hazard table in all 4 folds (e.g. 0.99 vs 1.85 in fold 0);
Δ NLL +0.364 [+0.135, +0.683]. Empirical 1-day CIF: stop 0.46 /
other_exit 0.37 / target 0.01. Caveats, stated plainly: **target
touches are nearly nonexistent (n=3 in test, 16 overall)** so the
target cause is unlearnable; censored subjects carry label_end =
coverage end so the purge removes them from every train window
(training uses resolved events only — 25 rows affected, noted in the
report); this models the *bot's own exit process* (stop-dominated), not
market first-passage.

**GBM (fp `352a7d74…`):** `lightgbm not installed (optional
dependency); result reported honestly, never faked`.

Not run: 5s horizon (not validatable, Phase 2 finding), `iv_change`
(0 ok labels), 15m direction (deferred; 68% unresolved labels).

### Artifacts

- `artifacts/oracle4/train_{logistic_60m,logistic_1s,naive_60m,
  quantile_60m,quantile_1s,rv_60m,hazard,gbm}.json` (local; regenerable
  via the CLI — fingerprints recorded above).

## Phase 4 — Multitask MLP challenger (PyTorch, CPU fallback)

### Code

- `oracle/ml/mlp.py` — torch behind a guarded import
  (`TORCH_AVAILABLE`/`TORCH_WHY`; without torch the CLI reports
  `unavailable` honestly, never fakes). `MultitaskMLP`: trunk
  `Linear→128→LayerNorm→GELU→dropout(0.1)→64→GELU→dropout→32` with four
  heads — (a) 3-class direction CE with **train-only** class weights;
  (b) return quantiles parameterized as `q05 + cumsum(softplus(gaps))`
  so the ladder is ordered by construction, pinball loss; (c) log-RV
  robust (smooth-L1) regression, forecast `exp(clamp(·))` > 0 always;
  (d) discrete-time competing-risk hazards — softmax over
  (survive, stop, target, other_exit) per time bin (same bins as Phase
  3), censored likelihood, `cif_from_hazards` producing nondecreasing
  cause-specific CIFs that sum ≤ 1. Combined loss = sum of available
  head losses, each normalized by its valid-label count and a running
  loss scale; **missing targets are masked, never imputed**. AdamW
  (lr 1e-3, wd 1e-4), grad clipping, batch 256, max 40 epochs, early
  stopping on a chronological validation session (patience 5), fixed
  seeds, CPU fallback + single-CUDA. Post-hoc temperature scaling fit
  on validation logits only. `save_model`/`load_model`: weights +
  feature schema + standardizer state + temperature + git SHA + dataset
  hash in a manifest with SHA-256 checksum; any mismatch (checksum,
  schema length, d_in, state_dict) raises `ModelIntegrityError` → the
  caller abstains. `_purge_safe` re-applies the leakage gate to every
  auxiliary label join (aux labels whose window or availability crosses
  the test boundary are masked out).
- `oracle/ml/train.py` — `run_mlp`: same purged fold harness and
  holdout pledge as Phase 3; validation session = last train session of
  each fold (never test); direction head evaluated against the same
  three comparators with whole-session paired bootstrap; quantile vs
  marginal pinball, RV vs point-in-time EWMA QLIKE, hazards vs the
  Laplace-smoothed marginal table (pooled only when ≥20 train
  subjects). After folds, one artifact is trained on fold sessions only
  (holdout untouched), saved with manifest, and **reloaded for an
  inference-parity check** recorded in the report.

### Acceptance tests

`tests/test_oracle4_mlp.py` — **26 passed** (direction-class mapping,
purge-gate truth table, head assembly incl. rv-positivity and censored
hazards, output shapes/coherence, CIF monotone + bounded, temperature
flattening, masked-loss training, all-masked → explicit error, class
weights hand case, deterministic early stopping incl. max-epoch cap,
CPU seed determinism, planted-signal learnability (fixture only — no
profitability claim), temperature recovery on synthetic overconfident
logits, save/load parity, tampered-checksum / missing-file /
schema-mismatch / unreadable-manifest refusals, CLI label-type
rejection, and an end-to-end CLI run on a 4-head fixture asserting
untouched holdout + saved artifact + reload parity). Torch-gated tests
skip honestly when torch is absent; the no-torch CLI path is covered in
`test_oracle4_baselines.py`. Oracle4 battery: **119 passed**. Full suite:
**2103 passed, 1 failed** — the known pre-existing Telegram dispatch
failure; zero regressions.

### Real-data results (dataset `f7f662e0…`, seed 42, `--device cpu`, holdout untouched)

**60m (fp `f7cf0645…`):** direction n=4,174, acc 0.415 / ll 0.715 /
ECE 0.106 — does **not** beat constant (Δ −0.008 [−0.024, +0.003]);
vs baseline_p_win +0.026 [−0.000, +0.053] (borderline, not clear of
zero). Confirms Phase 3: **no 60m directional edge**. Fold-0
temperature hit the 20.0 clamp (2-session train fold, severely
overconfident — reported, not hidden). Quantile head is **worse** than
marginal (Δ pinball −0.0023 [−0.0039, −0.0009]; 90% interval
over-covers at 0.995). RV head **loses to EWMA** on QLIKE (1.279 vs
1.082; Δ −0.237 [−0.386, −0.089]) despite slightly better RMSE. Hazard
head beats the marginal table: Δ NLL +0.352 [+0.046, +0.873], CIF
coherent — on par with the Phase 3 classical hazard model (+0.364).

**1s (fp `3b27f66e…`):** direction n=6,190, acc 0.488 / ll 0.690 —
all comparator deltas have positive means but **CIs cross zero** (vs
constant +0.038 [−0.011, +0.108]; vs momentum +0.030 [−0.005, +0.089]).
The Phase 3 ridge logistic (+0.062, CI clear of zero) **remains the
better 1-session direction challenger.** Quantile head slightly worse
than marginal (Δ −0.00052 [−0.00058, −0.00044]). RV head: no 1s RV
labels → reported `null`, never imputed. Hazard head: Δ NLL +0.371
[+0.082, +0.873], CIF coherent.

Train/val gap reported per fold (60m folds show val < train partly
because dropout/class weights apply to the train loss; 1s fold 2 shows
the opposite, +0.49). Early stopping fired in 7/8 folds. Both final
artifacts saved with manifest + checksum; reload-parity
`max_abs_diff = 0.0` on both.

Honest read: on six training sessions the multitask MLP **adds no
directional edge** over the classical challengers, its quantile/RV
heads underperform the marginal/EWMA comparators, and only the hazard
head matches (not beats) the classical hazard model. Its value at this
stage is the registry-ready artifact path (schema-checked save/load
with abstain-on-mismatch) for Phases 5–7, not predictive skill.

### Artifacts

- `artifacts/oracle4/train_mlp_{60m,1s}.json` +
  `train_mlp_{60m,1s}_model.pt` / `_manifest.json` (local; regenerable
  via the CLI — fingerprints above, checksums `a12a77ca…` / `6f10181f…`).
