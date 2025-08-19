#!/usr/bin/env bash
set -euo pipefail

# Unified helper to:
#  1) Transcribe an input file into three variants (default, stabilize, stabilize+vad+demucs)
#  2) Run pairwise SRT readability diffs on the three generated outputs
#
# By default it runs BOTH steps in sequence. Use --transcribe or --report to
# select a single step.
#
# Usage:
#   bash scripts/transcribe_and_diff.sh [--transcribe | --report] [--show-violations N] [--out-dir DIR] <audio_file>
#
# Examples:
#   # Do everything (transcribe all 3 variants, then diff all pairs)
#   bash scripts/transcribe_and_diff.sh data/samples/sample.wav
#
#   # Only transcribe
#   bash scripts/transcribe_and_diff.sh --transcribe data/samples/sample.wav
#
#   # Only report (requires SRTs already present)
#   bash scripts/transcribe_and_diff.sh --report data/samples/sample.wav
#
#   # Report with top-5 violations per category, results in a custom dir
#   bash scripts/transcribe_and_diff.sh --report --show-violations 5 --out-dir data/test_results/ data/samples/sample.wav

MODE="both"            # both | transcribe | report
SHOW_VIOLATIONS="0"    # integer
OUT_DIR="data/test_results"  # report output dir

# Parse flags
ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --transcribe)
      MODE="transcribe"; shift ;;
    --report)
      MODE="report"; shift ;;
    --show-violations)
      SHOW_VIOLATIONS="${2:-0}"; shift 2 ;;
    --out-dir)
      OUT_DIR="${2:-$OUT_DIR}"; shift 2 ;;
    -h|--help)
      sed -n '1,80p' "$0"; exit 0 ;;
    --) shift; break ;;
    -*) echo "Unknown option: $1" >&2; exit 2 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done

if [[ ${#ARGS[@]} -lt 1 ]]; then
  echo "Usage: $0 [--transcribe | --report] [--show-violations N] [--out-dir DIR] <audio_file>" >&2
  exit 1
fi

INPUT_FILE="${ARGS[0]}"
if [[ ! -f "$INPUT_FILE" ]]; then
  echo "Error: file not found: $INPUT_FILE" >&2
  exit 1
fi

STEM="$(basename -- "$INPUT_FILE")"
STEM="${STEM%.*}"

# Ensure directories
mkdir -p "$OUT_DIR"
mkdir -p data/output/default data/output/stabilize data/output/stabilize_vad_demucs

# Determine runners
if command -v pdm >/dev/null 2>&1; then
  TRANSCRIBE_RUNNER=(pdm run parakeet-rocm)
  DIFF_RUNNER=(pdm run python -m scripts.srt_diff_report)
elif command -v parakeet-rocm >/dev/null 2>&1; then
  TRANSCRIBE_RUNNER=(parakeet-rocm)
  if command -v srt-diff-report >/dev/null 2>&1; then
    DIFF_RUNNER=(srt-diff-report)
  else
    DIFF_RUNNER=(python -m scripts.srt_diff_report)
  fi
else
  TRANSCRIBE_RUNNER=(python -m parakeet_nemo_asr_rocm.cli)
  if command -v srt-diff-report >/dev/null 2>&1; then
    DIFF_RUNNER=(srt-diff-report)
  else
    DIFF_RUNNER=(python -m scripts.srt_diff_report)
  fi
fi

# Paths for three outputs
D_DEFAULT="data/output/default"
D_STAB="data/output/stabilize"
D_SVD="data/output/stabilize_vad_demucs"

transcribe_three() {
  set -x
  "${TRANSCRIBE_RUNNER[@]}" transcribe --word-timestamps --output-format srt \
    --output-dir "$D_DEFAULT" "$INPUT_FILE"
  "${TRANSCRIBE_RUNNER[@]}" transcribe --word-timestamps --output-format srt \
    --output-dir "$D_STAB" --stabilize "$INPUT_FILE"
  "${TRANSCRIBE_RUNNER[@]}" transcribe --word-timestamps --output-format srt \
    --output-dir "$D_SVD" --stabilize --vad --demucs "$INPUT_FILE"
  set +x
}

# Helper to find SRT for a stem within a directory
find_srt() {
  local dir="$1"; local stem="$2"
  local exact="$dir/$stem.srt"
  if [[ -f "$exact" ]]; then
    echo "$exact"; return 0
  fi
  local candidate
  candidate=$(ls -1t "$dir/$stem"*.srt 2>/dev/null | head -n1 || true)
  if [[ -n "${candidate:-}" && -f "$candidate" ]]; then
    echo "$candidate"; return 0
  fi
  return 1
}

report_diffs() {
  local srt_default srt_stab srt_svd
  srt_default="$(find_srt "$D_DEFAULT" "$STEM" || true)"
  srt_stab="$(find_srt "$D_STAB" "$STEM" || true)"
  srt_svd="$(find_srt "$D_SVD" "$STEM" || true)"

  if [[ -z "${srt_default:-}" || -z "${srt_stab:-}" || -z "${srt_svd:-}" ]]; then
    echo "Error: missing SRT(s) for '$STEM'. Ensure transcription step ran." >&2
    echo "  default:   $D_DEFAULT/${STEM}.srt (or newest ${STEM}*.srt)" >&2
    echo "  stabilize: $D_STAB/${STEM}.srt (or newest ${STEM}*.srt)" >&2
    echo "  vad+demucs: $D_SVD/${STEM}.srt (or newest ${STEM}*.srt)" >&2
    return 2
  fi

  local pairs=(
    "default:$srt_default" "stabilize:$srt_stab"
    "default:$srt_default" "stabilize_vad_demucs:$srt_svd"
    "stabilize:$srt_stab" "stabilize_vad_demucs:$srt_svd"
  )

  set -x
  for ((i=0; i<${#pairs[@]}; i+=2)); do
    local left_label="${pairs[i]%%:*}"
    local left_path="${pairs[i]#*:}"
    local right_label="${pairs[i+1]%%:*}"
    local right_path="${pairs[i+1]#*:}"

    local base_name="srt_diff_${left_label}_vs_${right_label}_${STEM}"
    local md_out="$OUT_DIR/${base_name}.md"
    local json_out="$OUT_DIR/${base_name}.json"

    # Markdown
    "${DIFF_RUNNER[@]}" "$left_path" "$right_path" \
      --output-format markdown \
      -o "$md_out" \
      ${SHOW_VIOLATIONS:+--show-violations "$SHOW_VIOLATIONS"}

    # JSON
    "${DIFF_RUNNER[@]}" "$left_path" "$right_path" \
      --output-format json \
      -o "$json_out" \
      ${SHOW_VIOLATIONS:+--show-violations "$SHOW_VIOLATIONS"}
  done
  set +x
}

case "$MODE" in
  transcribe)
    transcribe_three
    ;;
  report)
    report_diffs
    ;;
  both)
    transcribe_three
    report_diffs
    ;;
  *)
    echo "Invalid MODE: $MODE" >&2
    exit 2
    ;;
esac

echo "Done."
