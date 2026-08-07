#!/usr/bin/env bash
# Sync Mac assets to Google Drive (character LoRA, CivitAI weights, identity/private).
# Requires: rclone remote named "gdrive" (already configured on this machine).
#
# Usage:
#   ./rina_park/scripts/sync_local_only_to_drive.sh
#   ./rina_park/scripts/sync_local_only_to_drive.sh --dry-run
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
RINA="$ROOT/rina_park"
REMOTE="${RCLONE_REMOTE:-gdrive:rina_park_colab}"
DRY_RUN=0

if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
  shift
fi

rclone_run() {
  if [[ "$DRY_RUN" -eq 1 ]]; then
    rclone "$@" --dry-run
  else
    rclone "$@"
  fi
}

if ! command -v rclone >/dev/null 2>&1; then
  echo "rclone not found" >&2
  exit 1
fi

echo "LOCAL  $RINA"
echo "REMOTE $REMOTE"
echo

mkdir -p "$RINA/models/loras" "$RINA/models/checkpoints"

echo "==> loras (incl. rina_park_person + CivitAI; follow symlinks)"
rclone_run copy "$RINA/models/loras" "$REMOTE/models/loras" \
  --progress --transfers 4 --checkers 8 \
  --copy-links \
  --exclude ".cache/**"

echo "==> CivitAI checkpoints (juggernaut / cyberrealistic / lightning)"
# Single files need copyto. Symlinks must be resolved (rclone won't follow them).
for f in \
  juggernautXL_ragnarok.safetensors \
  cyberrealisticXL_v100.safetensors \
  realvisxlV50_v50LightningBakedvae.safetensors
do
  src="$RINA/models/checkpoints/$f"
  if [[ -e "$src" ]]; then
    src="$(realpath "$src")"
    rclone_run copyto "$src" "$REMOTE/models/checkpoints/$f" --progress
  else
    echo "  skip missing $f"
  fi
done

# Optional ultrasharp upscaler (CivitAI / local)
if [[ -f "$RINA/models/upscale/4xUltrasharp_4xUltrasharpV10.pt" ]]; then
  echo "==> upscale/4xUltrasharp"
  rclone_run copyto "$RINA/models/upscale/4xUltrasharp_4xUltrasharpV10.pt" \
    "$REMOTE/models/upscale/4xUltrasharp_4xUltrasharpV10.pt" --progress
fi

echo "==> identity"
rclone_run copy "$RINA/identity" "$REMOTE/identity" \
  --progress --exclude "__pycache__/**" --exclude "*.pyc"

echo "==> moodboard"
if [[ -d "$RINA/moodboard" ]]; then
  rclone_run copy "$RINA/moodboard" "$REMOTE/moodboard" --progress
else
  echo "  skip (no moodboard dir)"
fi

echo "==> private"
if [[ -d "$RINA/private" ]]; then
  rclone_run copy "$RINA/private" "$REMOTE/private" \
    --progress --exclude "__pycache__/**"
else
  echo "  skip (no private dir)"
fi

echo
echo "Done. On Colab: mount Drive → run colab_bootstrap.py → HF download for public models."
