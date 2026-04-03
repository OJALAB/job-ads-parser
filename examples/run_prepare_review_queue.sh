#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
INPUT_FILE="${INPUT_FILE:-$PROJECT_DIR/examples/kprm_warszawa_live_10.jsonl}"
TEXT_FIELD="${TEXT_FIELD:-skills_text}"
ID_FIELD="${ID_FIELD:-id}"
ESCO_CSV="${ESCO_CSV:-$PROJECT_DIR/examples/sample_esco_skills.csv}"
INDEX_DIR="${INDEX_DIR:-$PROJECT_DIR/.demo/review-index}"
OUTPUT_FILE="${OUTPUT_FILE:-$PROJECT_DIR/.demo/review-queue.jsonl}"
EXTRACTOR="${EXTRACTOR:-gliner}"
GLINER_MODEL="${GLINER_MODEL:-urchade/gliner_large-v2.1}"
DEVICE="${DEVICE:-auto}"

cd "$PROJECT_DIR"

"$PYTHON_BIN" -m pip install -e '.[gliner]'

PYTHONPATH=src "$PYTHON_BIN" -m esco_skill_batch build-index \
  --esco-csv "$ESCO_CSV" \
  --output-dir "$INDEX_DIR"

PYTHONPATH=src "$PYTHON_BIN" -m esco_skill_batch prepare-review-queue \
  --input "$INPUT_FILE" \
  --output "$OUTPUT_FILE" \
  --index-dir "$INDEX_DIR" \
  --text-field "$TEXT_FIELD" \
  --id-field "$ID_FIELD" \
  --extractor "$EXTRACTOR" \
  --gliner-model "$GLINER_MODEL" \
  --device "$DEVICE" \
  --mapping-backend lexical \
  --top-k 5 \
  --score-threshold 0.0

echo "Review queue: $OUTPUT_FILE"
