#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/.demo/model-benchmark}"
DEVICE="${DEVICE:-auto}"

cd "$PROJECT_DIR"
mkdir -p "$OUTPUT_ROOT"

PYTHON_BIN="$PYTHON_BIN" \
OUTPUT_DIR="$OUTPUT_ROOT/escoxlmr" \
HF_MODEL="${HF_MODEL:-jjzha/escoxlmr_skill_extraction}" \
DEVICE="$DEVICE" \
bash examples/run_eval_hf_suite.sh

PYTHON_BIN="$PYTHON_BIN" \
OUTPUT_DIR="$OUTPUT_ROOT/gliner-large-v2.1" \
GLINER_MODEL="${GLINER_MODEL_1:-urchade/gliner_large-v2.1}" \
DEVICE="$DEVICE" \
bash examples/run_eval_gliner_suite.sh

PYTHON_BIN="$PYTHON_BIN" \
OUTPUT_DIR="$OUTPUT_ROOT/gliner-bi-large-v1.0" \
GLINER_MODEL="${GLINER_MODEL_2:-knowledgator/gliner-bi-large-v1.0}" \
DEVICE="$DEVICE" \
bash examples/run_eval_gliner_suite.sh

"$PYTHON_BIN" examples/compare_model_benchmarks.py \
  "escoxlmr/easy" "$OUTPUT_ROOT/escoxlmr/metrics-easy.json" \
  "escoxlmr/hard" "$OUTPUT_ROOT/escoxlmr/metrics-hard.json" \
  "gliner-large/easy" "$OUTPUT_ROOT/gliner-large-v2.1/metrics-easy.json" \
  "gliner-large/hard" "$OUTPUT_ROOT/gliner-large-v2.1/metrics-hard.json" \
  "gliner-bi/easy" "$OUTPUT_ROOT/gliner-bi-large-v1.0/metrics-easy.json" \
  "gliner-bi/hard" "$OUTPUT_ROOT/gliner-bi-large-v1.0/metrics-hard.json"

echo "Benchmark outputs:"
echo "  $OUTPUT_ROOT/escoxlmr"
echo "  $OUTPUT_ROOT/gliner-large-v2.1"
echo "  $OUTPUT_ROOT/gliner-bi-large-v1.0"
