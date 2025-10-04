#!/usr/bin/env bash
# Demonstration script for tools/audio_splitter.py
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <input-file>" >&2
  exit 1
fi

INPUT=$1

python tools/audio_splitter.py -i "$INPUT" --parts 3 --dry-run --verbose
python tools/audio_splitter.py -i "$INPUT" --part-duration 30 --pattern "{stem}_seg{idx}.{ext}" --dry-run
