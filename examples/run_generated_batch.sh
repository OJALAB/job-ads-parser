#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
INDEX_DIR="${INDEX_DIR:-$PROJECT_DIR/.demo/generated-index}"
OUTPUT_FILE="${OUTPUT_FILE:-$PROJECT_DIR/.demo/generated-results.jsonl}"
INPUT_FILE="${INPUT_FILE:-$PROJECT_DIR/examples/generated_jobs_multilingual.jsonl}"
ESCO_CSV="${ESCO_CSV:-$PROJECT_DIR/examples/sample_esco_skills.csv}"

cd "$PROJECT_DIR"

"$PYTHON_BIN" examples/generate_synthetic_jobs.py

PYTHONPATH="${PYTHONPATH:-src}" "$PYTHON_BIN" -m esco_skill_batch build-index \
  --esco-csv "$ESCO_CSV" \
  --output-dir "$INDEX_DIR"

PYTHONPATH="${PYTHONPATH:-src}" "$PYTHON_BIN" -m esco_skill_batch extract-batch \
  --input "$INPUT_FILE" \
  --output "$OUTPUT_FILE" \
  --index-dir "$INDEX_DIR" \
  --text-field description \
  --id-field id \
  --extractor passthrough \
  --mentions-field skills_raw \
  --mapping-backend lexical \
  --top-k 5

echo
echo "Generated dataset: $INPUT_FILE"
echo "Batch results: $OUTPUT_FILE"
