#!/usr/bin/env bash
# Package the code, the trained model and the docs (no training data) into crownzero.zip
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-$ROOT/dist/crownzero.zip}"
mkdir -p "$(dirname "$OUT")"
cd "$(dirname "$ROOT")"
rm -f "$OUT"
zip -qr "$OUT" crownzero \
  -x "crownzero/data/*" "crownzero/logs*" "crownzero/dist/*" "crownzero/replays/*.msgpack" \
     "crownzero/models/_*" "crownzero/models/prelim*" "*/__pycache__/*" "*.pyc" "crownzero/.pytest_cache/*"
echo "$OUT"
