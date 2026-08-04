#!/usr/bin/env bash
set -euo pipefail
export SSL_CERT_FILE="${SSL_CERT_FILE:-$HOME/combined-cert.pem}"
cd /Users/RBIPK031/ai_influencer/ComfyUI
exec .venv/bin/python main.py --force-fp16 "$@"
