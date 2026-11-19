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
