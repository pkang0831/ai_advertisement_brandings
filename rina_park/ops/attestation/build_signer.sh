#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
OUTPUT_DIR="$SCRIPT_DIR/bin"
OUTPUT="$OUTPUT_DIR/rina-promotion-signer"

mkdir -p "$OUTPUT_DIR"
xcrun swiftc \
  -O \
  -framework CryptoKit \
  -framework LocalAuthentication \
  -framework Security \
  "$SCRIPT_DIR/RinaPromotionSigner.swift" \
  -o "$OUTPUT"
chmod 0755 "$OUTPUT"
printf '%s\n' "$OUTPUT"
