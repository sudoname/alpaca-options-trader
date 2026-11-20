# Oracle 4.0 Runbook

Operational guide for the Oracle4 research pipeline (`oracle/ml/`).
Everything here is **research-side**: with the shadow flag OFF (the
default), nothing in this document affects production trading. The only
production-adjacent surface is a 16-line, env-gated, fail-open hook in
`run_alpaca_intraday.py::_rank_and_enter`.

All commands run from the repo root. On the server use
`/var/www/alps/venv/bin/python`; locally plain `python`.

---

## 1. End-to-end pipeline (training → replay → report)

Order matters — each step consumes the previous step's artifact.
Artifacts live in `artifacts/oracle4/` (git-ignored; regenerable).

### 1.1 Build the point-in-time dataset

```bash
python -m oracle.ml.dataset \
  --ledger research_ledger.db \
  --output artifacts/oracle4/dataset_v1.sqlite \
  --manifest artifacts/oracle4/dataset_v1.manifest.json
```

Optional: `--from/--to` (session date bounds), `--kinds` (snapshot
kinds), `--neutral-band` (default 0.001), `--seed` (default 42). The
manifest records row counts, missingness and a content fingerprint —
keep it next to the sqlite.

### 1.2 Train challengers (purged walk-forward, OOF predictions)

```bash
# classical (ridge logistic etc.)
python -m oracle.ml.train --model classical \
  --dataset artifacts/oracle4/dataset_v1.sqlite \
  --out artifacts/oracle4/classical.json

# multitask MLP (uses torch if installed, CPU fallback otherwise)
python -m oracle.ml.train --model mlp \
  --dataset artifacts/oracle4/dataset_v1.sqlite \
  --out artifacts/oracle4/mlp.json --device auto
```

Useful knobs: `--horizon` (default `60m`), `--label-type` (default
`direction`), `--seed`, `--n-boot`, `--fold-config`,
`--holdout-registration`. Negative controls (shuffled labels) must come
out at chance; if they don't, stop — the fold purging is leaking.

### 1.3 Scenario valuation (all-candidate ranking with abstention)

```bash
python -m oracle.ml.valuation \
  --dataset artifacts/oracle4/dataset_v1.sqlite \
  --out artifacts/oracle4/valuation.json \
  --n-paths 512 --step-minutes 5
```

`--sessions` limits to specific dates. Abstention reasons
(missing_quote, no_scenarios:*, iv_unresolved:* …) are counted, never
silently dropped.

### 1.4 Execution calibration (paper evidence, labeled as such)

```bash
python -m oracle.ml.execution \
  --execution-log server_data/<date>/execution_calibration.jsonl \
  --history server_data/<date>/trading_history.json \
  --dataset artifacts/oracle4/dataset_v1.sqlite \
  --out artifacts/oracle4/execution_calibration.json
```

The artifact's `provenance_note` states that all evidence is PAPER /
indicative — it bounds internal consistency only, never attainable live
execution.

### 1.5 Paired portfolio replay

```bash
python -m oracle.ml.replay \
  --dataset artifacts/oracle4/dataset_v1.sqlite \
  --valuation artifacts/oracle4/valuation.json \
  --execution-calibration artifacts/oracle4/execution_calibration.json \
  --out artifacts/oracle4/replay.json
```

Defaults: `--starting-cash 100000 --max-concurrent 5 --n-boot 2000
--seed 42`; `--holdout` refuses decision dates (default holdout is
baked in); `--no-neural` skips the MLP variant. Invariants checked in
the artifact: accounting parity (`parity.ok`) and holdout refusal.

### 1.6 Promotion report (read-only verdict)

```bash
python -m oracle.ml.promotion \
  --replay artifacts/oracle4/replay.json \
  --execution artifacts/oracle4/execution_calibration.json \
  --variant combined --baseline baseline_frozen \
  --out artifacts/oracle4/promotion_report.json
```

Optional `--forward-evidence FILE` supplies the forward-shadow summary
(keys: `forward_sessions`, `resolved_opportunities`, `actual_fills`,
`calibration_improved`, `coverage_disparity`, `live_fill_based`).
Verdicts: `INSUFFICIENT_EVIDENCE` (any thin/uncovered gate — thresholds
are never relaxed), `DO_NOT_PROMOTE` (sufficient evidence, ≥1 failed
gate), `ELIGIBLE_PENDING_HUMAN_REVIEW` (best case; there is no PROMOTE
output). The report never changes the registry, flags or trading
behavior. Current real-artifact verdict: **INSUFFICIENT_EVIDENCE**
(no forward sessions, 39/200 resolved, mark-derived outcomes only).

---

## 2. Model registry (champion/challenger governance)

Registry file: `artifacts/oracle4/registry.json` (JSON; atomic writes;
append-only history).

```bash
# register a trained artifact (always starts unapproved challenger)
python -m oracle.ml.registry register \
  --model-id mlp_v1 --kind neural \
  --artifact artifacts/oracle4/mlp.json --notes "phase4 run"

# human approval (approved-by must be a real person; empty is rejected)
python -m oracle.ml.registry approve --model-id mlp_v1 --approved-by alice

# promote to champion (requires prior approval; demotes old champion)
python -m oracle.ml.registry set-champion --model-id mlp_v1 --approved-by alice

# inspect
python -m oracle.ml.registry show
```

Rules enforced by code, not convention:

- artifact bytes are pinned by SHA-256; re-registering the same id with
  different bytes is rejected (`immutable`) — use a new model id;
- there is no automatic approval path;
- **rollback** = `set-champion` back to the previous model id. It is a
  reference change only; no files move, checksums prove it.

---

## 3. Forward shadow (observe-only)

### 3.1 What it does / does not do

When enabled, every production scan's candidate list (the exact dicts
`_rank_and_enter` just scored) is also handed to
`oracle.ml.shadow.maybe_shadow_scan`, which logs one JSON line per scan
to `oracle4_shadow.jsonl`: baseline score order, challenger scenario-EV
ranking, inference ms, model identity (registry checksums + approval
state), per-candidate missingness, and an explicit "no order was
submitted" note. It **never** submits, modifies or cancels orders — the
module contains no broker imports (test-enforced) and is fail-open: any
exception is swallowed and logged, production entries proceed untouched.

### 3.2 Configuration diff (operator action — do NOT apply unattended)

The shadow is DISABLED by default. To enable on the server, a human
appends to `/var/www/alps/.env` (take a backup first):

```bash
ssh -i ~/.ssh/id_ed25519_1 root@147.182.242.177
cd /var/www/alps
cp .env .env.bak.$(date +%F)-oracle4shadow
cat >> .env << 'EOF'
# --- Oracle4 observe-only shadow (research; submits nothing) ---
ENABLE_ORACLE4_SHADOW=true
ORACLE4_MODEL_ID=mlp_v1
ORACLE4_MAX_CANDIDATES=16
ORACLE4_MAX_SCENARIOS=512
ORACLE4_SHADOW_JSONL=oracle4_shadow.jsonl
ORACLE4_REGISTRY=artifacts/oracle4/registry.json
EOF
systemctl restart alps-bot alps-scheduler
systemctl show alps-bot -p ExecMainStartTimestamp -p MainPID   # verify real restart
```

### 3.3 Verify

```bash
# after one scan cycle:
tail -n 1 /var/www/alps/oracle4_shadow.jsonl | python -m json.tool | head -30
/var/www/alps/venv/bin/python -m oracle.ml.shadow --report --log oracle4_shadow.jsonl
```

Expect `type: oracle4_shadow_scan`, a populated `baseline.order`, a
`challenger.decision` of `trade`/`abstain`, and the no-order note.
Service logs show `[ORACLE4-SHADOW] ...` only on (ignored) failures.

### 3.4 Disable / rollback

```bash
# flip the single gate; the other vars are inert without it
sed -i 's/^ENABLE_ORACLE4_SHADOW=.*/ENABLE_ORACLE4_SHADOW=false/' /var/www/alps/.env
systemctl restart alps-bot alps-scheduler
```

Or restore the `.env.bak.*` backup. With the flag off the hook costs one
`os.getenv` per scan and writes nothing.

---

## 4. Testing

```bash
# full Oracle4 battery (248 tests)
python -m pytest tests/test_oracle4_dataset.py tests/test_oracle4_validation.py \
  tests/test_oracle4_baselines.py tests/test_oracle4_mlp.py \
  tests/test_oracle4_valuation.py tests/test_oracle4_execution.py \
  tests/test_oracle4_replay.py tests/test_oracle4_registry.py \
  tests/test_oracle4_shadow.py tests/test_oracle4_promotion.py -q
```

Repo-wide runs: always `--ignore=tests/test_safety.py` (hangs in this
environment).

---

## 5. Troubleshooting

| symptom | cause / fix |
|---|---|
| no `oracle4_shadow.jsonl` appears | flag not `true` (exact lowercase word after strip/lower), or service not actually restarted — check `ExecMainStartTimestamp`, never trust `is-active` alone |
| `[ORACLE4-SHADOW] scan failed (ignored): …` in logs | fail-open path fired; production unaffected. Read the message; common: registry path wrong (identity still logs with `registry_error`), malformed candidate |
| every record `decision: abstain` with `no_scenarios:*` | `get_price_history` returned < 30 closes per underlying — expected off-hours or for thin symbols; not an error |
| `history_grid_assumed: true` | always true today: daily closes are mapped to a synthetic uniform grid (`history_grid_minutes`). A timestamped intraday history source would remove the assumption |
| registry `ValueError: … immutable` | artifact bytes changed under an existing model id; register under a new id |
| promotion verdict `INSUFFICIENT_EVIDENCE` despite good numbers | by design: any gate without forward evidence (sessions, resolved opps, actual fills, calibration, coverage, evidence basis) forces it; accumulate forward shadow data, never relax thresholds |
| replay `parity.ok: false` | accounting bug or corrupted artifact — regenerate from the dataset; do not hand-edit artifacts |
| torch missing | `oracle.ml.mlp` falls back to the pure-NumPy CPU path automatically; `--device auto` handles it |

---

## 6. Hard safety rules (recap)

1. The challenger never submits orders; promotion is only ever
   `registry.approve_model` + `registry.set_champion` performed by a
   human, after a report whose best verdict is
   `ELIGIBLE_PENDING_HUMAN_REVIEW`.
2. No online gradient updates, no automatic threshold relaxation, no
   automatic production promotion.
3. Thin samples ⇒ `INSUFFICIENT_EVIDENCE`, never adjusted gates.
4. Mark-derived / synthetic outcomes cannot satisfy the evidence-basis
   gate; only genuine forward fills count.
5. Rollback of any model decision is a registry reference change.
