#!/usr/bin/env bash
set -euo pipefail

# Compare three generated SRT outputs (default, stabilize, stabilize_vad_demucs)
# by running scripts/srt_diff_report.py on all pairwise combinations and
# emitting both Markdown and JSON reports into data/test_results/.
#
# Usage:
#   bash scripts/srt_diff_three.sh <audio_file>
#
# Notes:
# - Assumes scripts/transcribe_three.sh was already run for the same <audio_file>.
# - Prefers `pdm run srt-diff-report` when PDM is installed; otherwise falls back
#   to `srt-diff-report` if available, then to `python -m scripts.srt_diff_report`.

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <audio_file>" >&2
  exit 1
fi

INPUT_FILE="$1"
if [[ ! -f "$INPUT_FILE" ]]; then
  echo "Error: file not found: $INPUT_FILE" >&2
  exit 1
fi

STEM="$(basename "${INPUT_FILE}")"
STEM="${STEM%.*}"

OUT_DIR="data/test_results"
mkdir -p "$OUT_DIR"

# Locations containing the three SRT variants
D_DEFAULT="data/output/default"
D_STAB="data/output/stabilize"
D_SVD="data/output/stabilize_vad_demucs"

# Helper to locate the most likely SRT file for a given stem in a directory.
# Tries exact match first, else newest matching glob.
find_srt() {
  local dir="$1"
  local stem="$2"
  local exact="$dir/$stem.srt"
  if [[ -f "$exact" ]]; then
    echo "$exact"
    return 0
  fi
  # Fallback: newest matching by prefix
  local candidate
  candidate=$(ls -1t "$dir/$stem"*.srt 2>/dev/null | head -n1 || true)
  if [[ -n "${candidate:-}" && -f "$candidate" ]]; then
    echo "$candidate"
    return 0
  fi
  return 1
}

SRT_DEFAULT="$(find_srt "$D_DEFAULT" "$STEM" || true)"
SRT_STAB="$(find_srt "$D_STAB" "$STEM" || true)"
SRT_SVD="$(find_srt "$D_SVD" "$STEM" || true)"

if [[ -z "${SRT_DEFAULT:-}" || -z "${SRT_STAB:-}" || -z "${SRT_SVD:-}" ]]; then
  echo "Error: could not locate all three SRT files for stem '$STEM'." >&2
  echo " Looked for:" >&2
  echo "  - default:   $D_DEFAULT/${STEM}.srt (or newest ${STEM}*.srt)" >&2
  echo "  - stabilize: $D_STAB/${STEM}.srt (or newest ${STEM}*.srt)" >&2
  echo "  - vad+demucs: $D_SVD/${STEM}.srt (or newest ${STEM}*.srt)" >&2
  exit 1
fi

# Determine the diff runner
if command -v pdm >/dev/null 2>&1; then
  RUNNER=(pdm run srt-diff-report)
elif command -v srt-diff-report >/dev/null 2>&1; then
  RUNNER=(srt-diff-report)
else
  RUNNER=(python -m scripts.srt_diff_report)
fi

pairs=(
  "default:$SRT_DEFAULT" "stabilize:$SRT_STAB"
  "default:$SRT_DEFAULT" "stabilize_vad_demucs:$SRT_SVD"
  "stabilize:$SRT_STAB" "stabilize_vad_demucs:$SRT_SVD"
)

# Run all pairwise diffs: (A vs B)
set -x
for ((i=0; i<${#pairs[@]}; i+=2)); do
  left_label="${pairs[i]%%:*}"
  left_path="${pairs[i]#*:}"
  right_label="${pairs[i+1]%%:*}"
  right_path="${pairs[i+1]#*:}"

  base_name="srt_diff_${left_label}_vs_${right_label}_${STEM}"
  md_out="$OUT_DIR/${base_name}.md"
  json_out="$OUT_DIR/${base_name}.json"

  # Markdown
  "${RUNNER[@]}" "$left_path" "$right_path" \
    --output-format markdown \
    -o "$md_out"

  # JSON
  "${RUNNER[@]}" "$left_path" "$right_path" \
    --output-format json \
    -o "$json_out"

done
set +x

echo "SRT diffs generated in: $OUT_DIR"
ls -1 "$OUT_DIR" | sed 's/^/  - /'
