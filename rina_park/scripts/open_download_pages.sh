#!/usr/bin/env bash
# Print download destinations and open CivitAI/HF pages in browser (manual download).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
echo "Model root: $ROOT/models"
echo "See: $ROOT/models/CIVITAI_DOWNLOADS.md"
echo ""
echo "Suggested browser tabs:"
URLS=(
  "https://civitai.com/models/139562"
  "https://civitai.com/models/301776/ip-adapter-faceid"
  "https://civitai.com/models/580857"
  "https://huggingface.co/h94/IP-Adapter-FaceID"
  "https://huggingface.co/thibaud/controlnet-openpose-sdxl-1.0"
  "https://apps.apple.com/app/draw-things-ai-generation/id6444050820"
)
for u in "${URLS[@]}"; do
  echo "  $u"
done
if [[ "${1:-}" == "--open" ]]; then
  for u in "${URLS[@]}"; do open "$u"; done
fi
