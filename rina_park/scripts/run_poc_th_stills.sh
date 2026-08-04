#!/usr/bin/env bash
# Tim Hortons-adjacent study PoC: 5 SFW coffee/lifestyle stills (no official brand assets).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

export SSL_CERT_FILE="${SSL_CERT_FILE:-$HOME/combined-cert.pem}"
export REQUESTS_CA_BUNDLE="${REQUESTS_CA_BUNDLE:-$HOME/combined-cert.pem}"
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTHONUNBUFFERED=1
export PYTHONPATH=rina_park

POC="$ROOT/rina_park/out/poc_th_ads_study"
STILLS="$POC/stills"
mkdir -p "$STILLS" "$POC/reels"

TS=$(date -u +%Y%m%dT%H%M%SZ)
LOG="$POC/poc_stills_${TS}.log"
BEFORE=$(ls -1dt "$ROOT/rina_park/out/i2v_heroes"/2026* 2>/dev/null | head -1 || true)

echo "===== POC TH STILLS START $TS =====" | tee "$LOG"
echo "POC=$POC" | tee -a "$LOG"

.venv/bin/python -u rina_park/scripts/gen_i2v_heroes.py \
  --poses hand_holding_cup_soft \
  --seeds-per-pose 5 \
  --base-seed 30082026 \
  --lora 0.80 \
  --skip-face-detailer \
  --combo \
  --combo-mode random_seeded \
  --combo-track sfw \
  --scene-glue "cafe window soft daylight, warm red-brown knit" \
  2>&1 | tee -a "$LOG"

EC=$?
AFTER=$(ls -1dt "$ROOT/rina_park/out/i2v_heroes"/2026* 2>/dev/null | head -1 || true)
if [[ -n "$AFTER" && "$AFTER" != "$BEFORE" ]]; then
  echo "run_dir=$AFTER" | tee -a "$LOG"
  echo "$AFTER" > "$POC/SOURCE_RUN_DIR.txt"
  find "$AFTER" -type f -name 's*_seed*.jpg' ! -name '*_gen*' -exec cp -p {} "$STILLS"/ \;
  find "$AFTER" -type f -name 's*_seed*.jpg.json' -exec cp -p {} "$STILLS"/ \; 2>/dev/null || true
  cp -p "$AFTER"/summary.json "$POC/summary_${TS}.json" 2>/dev/null || true
fi
echo "===== POC TH STILLS EXIT=$EC $(date -u +%Y%m%dT%H%M%SZ) =====" | tee -a "$LOG"
exit "$EC"
