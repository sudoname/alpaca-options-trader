# Oracle 4.0.2 — Runbook

How to reproduce every 4.0.2 result and how to collect the forward evidence the
promotion gate requires. All commands run from the repo root on the research
machine (Windows Git Bash, `python` = the project venv interpreter). Nothing
here touches the production trading path.

## 0. Prerequisites

* Branch `oracle4.0.2` checked out.
* Optional deps for the full suite of challengers: `lightgbm` (GBM arms),
  `torch` (neural arms). Both are optional — absent deps produce honest
  `unavailable` stubs, never failures or fake numbers.
* Windows/pytest gotcha: remove any stray `nul` file first (`rm -f nul`),
  and always exclude the hanging safety suite:
  `python -m pytest --ignore=tests/test_safety.py -q`
  (expect exactly 4 known pre-existing failures unrelated to 4.0.2).

## 1. Rebuild training data (P1-5)

```bash
# Backfill historical bars (read-only market data; forming bars never stored)
python -m oracle.ml.bars_backfill --symbols SPY QQQ \
    --from 2024-01-01 --to 2026-11-19 \
    --db artifacts/oracle4/bars/bars.sqlite

# Build a bars-based dataset in the standard dataset-store schema
python -m oracle.ml.bars_dataset \
    --bars-db artifacts/oracle4/bars/bars.sqlite \
    --symbols SPY QQQ \
    --output artifacts/oracle4/dataset_bars.sqlite \
    --manifest artifacts/oracle4/dataset_bars_manifest.json
```

Each run writes a manifest with a `dataset_hash`; downstream tools crosscheck
hashes and refuse mixed artifacts.

## 2. Reproduce the headline backtest + promotion (P10/P11)

```bash
# Comprehensive 9-variant backtest (friction sweep, stress, paired CIs)
python -m oracle.ml.backtest402 \
    --dataset artifacts/oracle4/dataset_bt1121.sqlite \
    --valuation artifacts/oracle4/valuation_bt1121.json \
    --execution-calibration artifacts/oracle4/execution_calibration.json \
    --out artifacts/oracle4/replay402.json

# Promotion gate (never promotes; emits a fingerprinted verdict report)
python -m oracle.ml.promotion402 \
    --artifact artifacts/oracle4/replay402.json \
    --out artifacts/oracle4/promotion402.json
```

Expected (with the committed inputs): backtest fingerprint `bc9647af…`,
parity ok (n=690, max_abs_diff=0.0), best variant `gbm_gated`
(net −$333.44, PF 0.927), promotion overall `INSUFFICIENT_EVIDENCE`,
report fingerprint `6394a806…`. Any change to inputs changes the fingerprints —
that is the point; never compare reports across fingerprints.

## 3. Reproduce the EV model study (P0-4)

```bash
python -m oracle.ml.ev_models \
    --history server_data/2026-11-19/trading_history.json \
    --out artifacts/oracle402/ev_model_report.json
```

Expected: verdict `INSUFFICIENT_EVIDENCE`, `challenger_policy: abstain`
(10 eval opportunities < 50 floor). The study auto-upgrades its verdict once
the history file contains ≥50 post-split opportunities — just re-run it on a
newer `server_data/<date>/trading_history.json` export.

## 4. Forward evidence collection (the path to a promotable verdict)

The promotion gate requires ≥60 replay sessions, ≥200 resolved opportunities,
≥100 fills. Daily loop:

1. **Server export** (read-only): pull the latest research ledger/history into
   `server_data/<date>/` (same export flow used for 2026-11-19/20).
2. **Dataset refresh**: rebuild the dataset sqlite from the ledger
   (`oracle.ml.dataset`) or bars (`oracle.ml.bars_dataset`).
3. **Re-run §2 verbatim.** The criteria are frozen in `promotion402.py`
   constants; do not tune thresholds between runs — that is the whole value of
   the gate.
4. Track the verdict trajectory. The honest success condition is a
   `gbm_gated`-style variant reaching `ELIGIBLE_PENDING_HUMAN_REVIEW`;
   `DO_NOT_PROMOTE` on a challenger means stop spending sessions on it.
5. **Human decision**: even a fully-passing report changes nothing in
   production. Promotion = a human reads `promotion402.json`, checks the
   fingerprint chain (dataset hash → backtest fingerprint → report
   fingerprint), and makes a deliberate config change.

### Kalshi ablation (pending data)

```bash
# Start snapshot collection (research-only; safe to leave running)
python -m oracle.ml.kalshi.collector --series <kalshi series tickers...> \
    --db artifacts/oracle4/kalshi/snapshots.sqlite

# After >=2 weeks of snapshots, run the matched ablation A-E
python -m oracle.ml.kalshi.validation \
    --dataset <dataset.sqlite> \
    --kalshi-db artifacts/oracle4/kalshi/snapshots.sqlite \
    --underlying SPY
```

Interpretation rule: Kalshi features "help" only if arm B beats arm A while
the placebo arm D does **not** (D ≈ A); otherwise the improvement is leakage
or overfitting by construction.

### Async research worker (optional, default off)

Set `ENABLE_ORACLE4_ASYNC_RESEARCH=1` for the scheduler process to move bars
feeding/shadow scans onto the bounded worker. Health: the worker writes an
atomic status JSON (submitted/completed/failed/dropped/slow counters) surfaced
on the dashboard. Leave unset to keep 4.0.1 behavior exactly.

## 5. Tests

```bash
rm -f nul
# Per-phase suites
python -m pytest tests/test_oracle402_p01.py tests/test_oracle402_p03.py \
    tests/test_oracle402_p04.py tests/test_oracle402_p15.py \
    tests/test_oracle402_p16.py tests/test_oracle402_p17.py \
    tests/test_oracle402_p18.py tests/test_oracle402_p10.py \
    tests/test_oracle402_p11.py -q
# Full suite (expect only the 4 known pre-existing failures)
python -m pytest --ignore=tests/test_safety.py -q
```

## 6. Deployment notes

4.0.2 is research-only; **no server deployment is required or recommended** for
this release. If research modules are wanted on the server (e.g. to run the
collector there): push the branch, then on the server
(`ssh -i ~/.ssh/id_ed25519_1 root@147.182.242.177`, `/var/www/alps`):
`git fetch && git checkout <sha>`, restart only the unit that needs it
(`systemctl restart alps-dashboard` for dashboard changes). The server venv has
no pytest and CPU-only torch — never run training or the test suite there.

## 7. Artifact hygiene

Never commit: `artifacts/`, `server_data/`, `orderbook_shadow.jsonl`,
`candidate_resolution.jsonl`, `DEPLOYMENT_STATUS.md`,
`trading_history_server_2026-11-13.json`. Artifacts are reproducible from the
commands above plus the committed inputs; the fingerprints inside them are the
audit trail.
