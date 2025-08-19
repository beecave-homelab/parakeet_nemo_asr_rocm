#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <audio_file>" >&2
  exit 1
fi

FILE="$1"
if [[ ! -f "$FILE" ]]; then
  echo "Error: file not found: $FILE" >&2
  exit 1
fi

# Ensure output directories exist
mkdir -p \
  data/output/default \
  data/output/stabilize \
  data/output/stabilize_vad_demucs

# Prefer running through PDM if available to ensure the correct environment
if command -v pdm >/dev/null 2>&1; then
  RUNNER=(pdm run parakeet-rocm)
else
  RUNNER=(parakeet-rocm)
fi

set -x
"${RUNNER[@]}" transcribe --word-timestamps --output-format srt --output-dir data/output/default/ "$FILE"
"${RUNNER[@]}" transcribe --word-timestamps --output-format srt --output-dir data/output/stabilize/ --stabilize "$FILE"
"${RUNNER[@]}" transcribe --word-timestamps --output-format srt --output-dir data/output/stabilize_vad_demucs/ --stabilize --vad --demucs "$FILE"
set +x

echo "All transcriptions completed. Outputs in:"
echo "  - data/output/default/"
echo "  - data/output/stabilize/"
echo "  - data/output/stabilize_vad_demucs/"
