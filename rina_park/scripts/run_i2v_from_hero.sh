#!/bin/bash
# Run local Wan A14B I2V from a promoted hero still (detached-friendly).
# Usage:
#   ./rina_park/scripts/run_i2v_from_hero.sh hand_resting_lap_seated
#   screen -dmS wan_i2v ./rina_park/scripts/run_i2v_from_hero.sh fitness_mat_soft_seated
set -euo pipefail
POSE="${1:-hand_resting_lap_seated}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
STILL="$ROOT/rina_park/out/i2v_heroes/current/${POSE}.jpg"
MODEL="$ROOT/rina_park/models/wan/wan2.2-i2v-a14b-diffusers-8bit"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT="$ROOT/rina_park/out/reels/${STAMP}_${POSE}_a14b8_from_hero.mp4"
LOG="$ROOT/rina_park/out/reels/${STAMP}_${POSE}_a14b_from_hero.log"

if [[ ! -f "$STILL" && ! -L "$STILL" ]]; then
  echo "No promoted still at $STILL — pass STILL_GATE / promote_i2v_heroes first." >&2
  exit 1
fi
if [[ ! -d "$MODEL" ]]; then
  echo "Missing A14B model at $MODEL" >&2
  exit 1
fi

export PYTHONUNBUFFERED=1
export SSL_CERT_FILE="${SSL_CERT_FILE:-$HOME/combined-cert.pem}"
export REQUESTS_CA_BUNDLE="${REQUESTS_CA_BUNDLE:-$HOME/combined-cert.pem}"

PROMPT='subtle natural motion only, soft slow blink, gentle breathing, tiny hair strand sway, keep exact face identity and pose, photoreal soft glam skin, stable hands, no head turn, no morphing'
NEG='warp, morphing, melting face, identity drift, deformed hands, rubber body, liquid skin, jitter, flicker, blue noise, painterly, illustration, text, watermark'

cd "$ROOT"
echo "STILL=$STILL"
echo "OUT=$OUT"
.venv/bin/python -u .venv/bin/mlxgen-generate-wan \
  -m "$MODEL" \
  --image-path "$STILL" \
  --prompt "$PROMPT" \
  --negative-prompt "$NEG" \
  --width 720 --height 1280 \
  --frames 49 --fps 16 \
  --steps 24 \
  --guidance 3.5 \
  --flow-shift 5.0 \
  --quantize 8 \
  --mlx-cache-limit-gb 48 \
  --metadata \
  --output "$OUT" \
  --progress \
  2>&1 | tee "$LOG"

echo "DONE $OUT"
