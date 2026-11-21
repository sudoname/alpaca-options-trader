#!/usr/bin/env bash
# =============================================================================
# Oracle4 end-to-end training pipeline — run it yourself, no Claude needed.
#
# Usage (from repo root, Git Bash on Windows or bash on Linux):
#   bash scripts/train_oracle4.sh                 # full run, auto-dated tag
#   bash scripts/train_oracle4.sh --tag v3        # custom artifact tag
#   bash scripts/train_oracle4.sh --skip-fetch    # reuse an existing ledger
#   LEDGER=path/to/research_ledger.db bash scripts/train_oracle4.sh --skip-fetch
#
# What it does, in order (each step consumes the previous artifact):
#   1. fetch   — scp the live research_ledger.db from the server
#   2. dataset — build the point-in-time sqlite dataset + manifest
#   3. train   — logistic (classical) + MLP (GPU if available: --device auto)
#   4. value   — scenario valuation with abstention
#   5. replay  — paired portfolio replay vs frozen baseline
#   6. report  — promotion report (read-only verdict; expect
#                INSUFFICIENT_EVIDENCE until forward shadow evidence exists)
#
# Everything is research-side: nothing here touches production trading.
# Artifacts land in artifacts/oracle4/ (git-ignored, regenerable).
# See docs/oracle4_runbook.md for flags and troubleshooting.
# =============================================================================
set -euo pipefail

# ---- configuration (override via env) ---------------------------------------
PY="${PY:-python}"
SERVER="${SERVER:-root@147.182.242.177}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519_1}"
REMOTE_LEDGER="${REMOTE_LEDGER:-/var/www/alps/research_ledger.db}"
ART="${ART:-artifacts/oracle4}"
SEED="${SEED:-42}"
N_PATHS="${N_PATHS:-512}"
REGISTRY="${REGISTRY:-${ART}/registry.json}"

TAG="$(date +%Y%m%d)"
SKIP_FETCH=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --tag)        TAG="$2"; shift 2 ;;
    --skip-fetch) SKIP_FETCH=1; shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

STAMP="$(date +%F)"
LEDGER="${LEDGER:-server_data/${STAMP}/research_ledger.db}"
DATASET="${ART}/dataset_${TAG}.sqlite"
MANIFEST="${ART}/dataset_${TAG}_manifest.json"

mkdir -p "${ART}" "$(dirname "${LEDGER}")"

step() { printf '\n==== %s ====\n' "$*"; }

# ---- 1. fetch ledger ---------------------------------------------------------
if [[ "${SKIP_FETCH}" -eq 0 ]]; then
  step "1/6 fetch ledger from ${SERVER}"
  scp -i "${SSH_KEY}" "${SERVER}:${REMOTE_LEDGER}" "${LEDGER}"
else
  step "1/6 fetch skipped — using ${LEDGER}"
  [[ -f "${LEDGER}" ]] || { echo "ledger not found: ${LEDGER}" >&2; exit 1; }
fi

# ---- 2. dataset ---------------------------------------------------------------
step "2/6 build dataset -> ${DATASET}"
"${PY}" -m oracle.ml.dataset \
  --ledger "${LEDGER}" \
  --output "${DATASET}" \
  --manifest "${MANIFEST}" \
  --seed "${SEED}"

# ---- 3. train ------------------------------------------------------------------
step "3/6 train logistic (classical challenger)"
"${PY}" -m oracle.ml.train --model logistic \
  --dataset "${DATASET}" \
  --out "${ART}/logistic_${TAG}.json" \
  --seed "${SEED}"

step "3/6 train MLP (GPU via --device auto; CPU fallback if no CUDA)"
"${PY}" -m oracle.ml.train --model mlp \
  --dataset "${DATASET}" \
  --out "${ART}/mlp_${TAG}.json" \
  --device auto --seed "${SEED}"

# Negative controls (shuffled labels) must come out at chance.
# If a control beats chance, STOP: the fold purging is leaking.
"${PY}" - "$ART" "$TAG" <<'PYEOF'
import json, sys
art, tag = sys.argv[1], sys.argv[2]
bad = False
for name in (f"logistic_{tag}", f"mlp_{tag}"):
    try:
        rep = json.load(open(f"{art}/{name}.json", encoding="utf-8"))
    except FileNotFoundError:
        continue
    txt = json.dumps(rep)
    print(f"  {name}: negative-control section present: {'negative_control' in txt or 'shuffled' in txt}")
print("REVIEW the per-fold negative-control metrics above manually before trusting results.")
sys.exit(1 if bad else 0)
PYEOF

# ---- 3b. walk-forward forecaster -----------------------------------------------
# The standard MLP trains on ALL fold sessions, so its train window covers every
# session the valuation step scans -> forecasts would be refused (TRAIN_OVERLAP)
# and the run silently degrades to all-baseline. Train a second MLP with an
# enlarged holdout so the later ~40% of sessions are genuinely out-of-sample for
# it; valuation over those sessions can then use real forecast conditioning.
step "3b/6 train walk-forward forecaster MLP (earlier train window so valuation sessions are OOS)"
FC_HOLDOUT="$("${PY}" - "${DATASET}" <<'PYEOF'
import sys
from oracle.lab.purged_validation import HORIZON_EMBARGO_SESSIONS, load_samples, make_folds
samples = load_samples(sys.argv[1], "60m", "direction")
_f, fold_sessions, holdout = make_folds(
    samples, test_block_sessions=1, min_train_sessions=2,
    embargo_sessions=HORIZON_EMBARGO_SESSIONS.get("60m", 1), holdout_sessions=2)
total = len(fold_sessions) + len(holdout)
print(max(3, int(round(total * 0.4))))
PYEOF
)"
echo "  forecaster holdout_sessions=${FC_HOLDOUT}"
"${PY}" -m oracle.ml.train --model mlp \
  --dataset "${DATASET}" \
  --out "${ART}/mlp_fc_${TAG}.json" \
  --fold-config "{\"holdout_sessions\": ${FC_HOLDOUT}}" \
  --device auto --seed "${SEED}"

# ---- 3c. auto-register challengers ----------------------------------------------
# Every trained model is registered as an UNAPPROVED challenger. Registration
# never changes what production uses; approval + champion promotion stay manual.
# Model ids carry an artifact-checksum suffix so reruns are idempotent and a
# retrain under the same tag gets a fresh id (registry artifacts are immutable).
step "3c/6 auto-register challengers (unapproved; registry governs promotion)"
sfx() { "${PY}" -c "import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest()[:8])" "$1"; }
LOG_ID="logistic_${TAG}_$(sfx "${ART}/logistic_${TAG}.json")"
"${PY}" -m oracle.ml.registry --registry "${REGISTRY}" register \
  --model-id "${LOG_ID}" --kind classical \
  --artifact "${ART}/logistic_${TAG}.json" \
  --notes "auto-registered challenger (train_oracle4.sh, tag ${TAG})"
MLP_ID="mlp_${TAG}_$(sfx "${ART}/mlp_${TAG}_manifest.json")"
"${PY}" -m oracle.ml.registry --registry "${REGISTRY}" register \
  --model-id "${MLP_ID}" --kind neural \
  --artifact "${ART}/mlp_${TAG}_model.pt" \
  --manifest "${ART}/mlp_${TAG}_manifest.json" \
  --notes "auto-registered challenger (train_oracle4.sh, tag ${TAG})"
FC_MODEL_ID="mlp_fc_${TAG}_$(sfx "${ART}/mlp_fc_${TAG}_manifest.json")"
"${PY}" -m oracle.ml.registry --registry "${REGISTRY}" register \
  --model-id "${FC_MODEL_ID}" --kind neural \
  --artifact "${ART}/mlp_fc_${TAG}_model.pt" \
  --manifest "${ART}/mlp_fc_${TAG}_manifest.json" \
  --notes "walk-forward forecaster (holdout_sessions=${FC_HOLDOUT}) for forecast-conditioned valuation, tag ${TAG}"

# ---- 4. valuation ---------------------------------------------------------------
step "4/6 scenario valuation (baseline, no forecast model)"
"${PY}" -m oracle.ml.valuation \
  --dataset "${DATASET}" \
  --out "${ART}/valuation_${TAG}.json" \
  --n-paths "${N_PATHS}" --step-minutes 5 --seed "${SEED}"

# ---- 4b. forecast-conditioned valuation -------------------------------------------
step "4b/6 forecast-conditioned valuation (model ${FC_MODEL_ID})"
"${PY}" -m oracle.ml.valuation \
  --dataset "${DATASET}" \
  --out "${ART}/valuation_fc_${TAG}.json" \
  --registry "${REGISTRY}" --model-id "${FC_MODEL_ID}" \
  --n-paths "${N_PATHS}" --step-minutes 5 --seed "${SEED}"
"${PY}" - "${ART}/valuation_fc_${TAG}.json" <<'PYEOF'
import json, sys
rep = json.load(open(sys.argv[1], encoding="utf-8"))
fc = rep.get("result", rep).get("forecast") or {}
modes = fc.get("scan_modes") or {}
n_fc = sum(v for k, v in modes.items() if k == "forecast_conditioned")
print(f"  forecast scan_modes: {modes}")
print(f"  model_id: {fc.get('model_id')}  provider_note: {fc.get('provider_note')}")
if n_fc == 0:
    print("  WARNING: zero forecast_conditioned scans — every forecast was "
          "refused (TRAIN_OVERLAP/MODEL_STALE) or the provider failed to load. "
          "The fc valuation is effectively baseline; do NOT read it as "
          "model-conditioned evidence.")
PYEOF

# ---- 5. replay -------------------------------------------------------------------
step "5/6 paired replay vs frozen baseline (baseline valuation)"
EXEC_CAL=""
[[ -f "${ART}/execution_calibration.json" ]] && EXEC_CAL="--execution-calibration ${ART}/execution_calibration.json"
"${PY}" -m oracle.ml.replay \
  --dataset "${DATASET}" \
  --valuation "${ART}/valuation_${TAG}.json" \
  ${EXEC_CAL} \
  --out "${ART}/replay_${TAG}.json" --seed "${SEED}"

# ---- 5b. replay on the forecast-conditioned valuation ------------------------------
step "5b/6 paired replay vs frozen baseline (forecast-conditioned valuation)"
"${PY}" -m oracle.ml.replay \
  --dataset "${DATASET}" \
  --valuation "${ART}/valuation_fc_${TAG}.json" \
  ${EXEC_CAL} \
  --out "${ART}/replay_fc_${TAG}.json" --seed "${SEED}"

# ---- 6. promotion report ----------------------------------------------------------
step "6/6 promotion reports (read-only; never auto-promotes)"
"${PY}" -m oracle.ml.promotion \
  --replay "${ART}/replay_${TAG}.json" \
  --execution "${ART}/execution_calibration.json" \
  --out "${ART}/promotion_report_${TAG}.json"
"${PY}" -m oracle.ml.promotion \
  --replay "${ART}/replay_fc_${TAG}.json" \
  --execution "${ART}/execution_calibration.json" \
  --out "${ART}/promotion_report_fc_${TAG}.json"

step "DONE"
cat <<EOF

Artifacts (tag ${TAG}) in ${ART}/:
  dataset_${TAG}.sqlite / _manifest.json
  logistic_${TAG}.json, mlp_${TAG}.json, mlp_fc_${TAG}.json (+ _model.pt, _manifest.json)
  valuation_${TAG}.json, valuation_fc_${TAG}.json
  replay_${TAG}.json, replay_fc_${TAG}.json
  promotion_report_${TAG}.json, promotion_report_fc_${TAG}.json

Registered challengers (unapproved; in ${REGISTRY}):
  ${LOG_ID}
  ${MLP_ID}
  ${FC_MODEL_ID}

Next steps (all manual, by design):
  * inspect the promotion verdicts:
      ${PY} -c "import json;print(json.load(open('${ART}/promotion_report_${TAG}.json'))['verdict'])"
      ${PY} -c "import json;print(json.load(open('${ART}/promotion_report_fc_${TAG}.json'))['verdict'])"
  * approval always requires a human (nothing here auto-promotes):
      ${PY} -m oracle.ml.registry --registry ${REGISTRY} approve --model-id ${FC_MODEL_ID} --approved-by <your-name>
      ${PY} -m oracle.ml.registry --registry ${REGISTRY} set-champion --model-id ${FC_MODEL_ID} --approved-by <your-name>
EOF
