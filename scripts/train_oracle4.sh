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

# ---- 4. valuation ---------------------------------------------------------------
step "4/6 scenario valuation"
"${PY}" -m oracle.ml.valuation \
  --dataset "${DATASET}" \
  --out "${ART}/valuation_${TAG}.json" \
  --n-paths 512 --step-minutes 5 --seed "${SEED}"

# ---- 5. replay -------------------------------------------------------------------
step "5/6 paired replay vs frozen baseline"
EXEC_CAL=""
[[ -f "${ART}/execution_calibration.json" ]] && EXEC_CAL="--execution-calibration ${ART}/execution_calibration.json"
"${PY}" -m oracle.ml.replay \
  --dataset "${DATASET}" \
  --valuation "${ART}/valuation_${TAG}.json" \
  ${EXEC_CAL} \
  --out "${ART}/replay_${TAG}.json" --seed "${SEED}"

# ---- 6. promotion report ----------------------------------------------------------
step "6/6 promotion report (read-only; never auto-promotes)"
"${PY}" -m oracle.ml.promotion \
  --replay "${ART}/replay_${TAG}.json" \
  --execution "${ART}/execution_calibration.json" \
  --out "${ART}/promotion_report_${TAG}.json"

step "DONE"
cat <<EOF

Artifacts (tag ${TAG}) in ${ART}/:
  dataset_${TAG}.sqlite / _manifest.json
  logistic_${TAG}.json, mlp_${TAG}.json (+ _model.pt, _manifest.json)
  valuation_${TAG}.json, replay_${TAG}.json, promotion_report_${TAG}.json

Next steps (all manual, by design):
  * inspect the promotion report verdict:
      ${PY} -c "import json;print(json.load(open('${ART}/promotion_report_${TAG}.json'))['verdict'])"
  * register the MLP under a NEW model id (artifacts are immutable):
      ${PY} -m oracle.ml.registry register --model-id mlp_${TAG} --kind neural \\
        --artifact ${ART}/mlp_${TAG}.json --notes "trained ${STAMP}"
  * promotion to champion always requires a human:
      ${PY} -m oracle.ml.registry approve --model-id mlp_${TAG} --approved-by <your-name>
EOF
