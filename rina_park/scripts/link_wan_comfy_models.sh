#!/usr/bin/env bash
# Link rina_park/models/wan/* into ComfyUI/models when weights arrive.
# Safe to re-run (idempotent). Does not download models.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
RINA_WAN="${RINA_WAN:-$ROOT/rina_park/models/wan}"
COMFY="${COMFYUI_ROOT:-$ROOT/ComfyUI}"
COMFY_MODELS="$COMFY/models"

if [[ ! -d "$COMFY_MODELS" ]]; then
  echo "ERROR: ComfyUI models dir missing: $COMFY_MODELS" >&2
  exit 1
fi

mkdir -p \
  "$RINA_WAN/diffusion_models" \
  "$RINA_WAN/text_encoders" \
  "$RINA_WAN/vae" \
  "$RINA_WAN/loras"

link_tree() {
  local src="$1"
  local dest="$2"
  mkdir -p "$dest"
  shopt -s nullglob
  local files=("$src"/*)
  shopt -u nullglob
  if ((${#files[@]} == 0)); then
    echo "skip (empty): $src"
    return 0
  fi
  local f base target
  for f in "${files[@]}"; do
    base="$(basename "$f")"
    [[ "$base" == ".gitkeep" ]] && continue
    target="$dest/$base"
    if [[ -L "$target" ]]; then
      local cur
      cur="$(readlink "$target")"
      if [[ "$cur" == "$f" ]]; then
        echo "ok: $target"
        continue
      fi
      echo "replace symlink: $target -> $f"
      ln -sfn "$f" "$target"
    elif [[ -e "$target" ]]; then
      echo "WARN: exists (not symlink), leave alone: $target" >&2
    else
      ln -s "$f" "$target"
      echo "linked: $target -> $f"
    fi
  done
}

echo "RINA_WAN=$RINA_WAN"
echo "COMFY_MODELS=$COMFY_MODELS"

link_tree "$RINA_WAN/diffusion_models" "$COMFY_MODELS/diffusion_models"
link_tree "$RINA_WAN/text_encoders" "$COMFY_MODELS/text_encoders"
link_tree "$RINA_WAN/vae" "$COMFY_MODELS/vae"
link_tree "$RINA_WAN/loras" "$COMFY_MODELS/loras"

echo "done. Drop Wan weights under rina_park/models/wan/{diffusion_models,text_encoders,vae,loras} then re-run."
echo "ComfyUI root: $COMFY (override with COMFYUI_ROOT=...)"
