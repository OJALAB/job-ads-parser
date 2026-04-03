#!/usr/bin/env bash
set -euo pipefail

# Example usage on the target server.
# Assumptions:
# - you are inside the project directory
# - Ollama is available on 127.0.0.1:11434
# - ESCO skills CSV has already been downloaded locally

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
PYTHON_BIN="${PYTHON_BIN:-python}"
ESCO_CSV="${ESCO_CSV:-$PROJECT_DIR/examples/sample_esco_skills.csv}"
INPUT_FILE="${INPUT_FILE:-$PROJECT_DIR/examples/sample_jobs.jsonl}"
INDEX_DIR="${INDEX_DIR:-$PROJECT_DIR/.demo/index}"
OUTPUT_FILE="${OUTPUT_FILE:-$PROJECT_DIR/.demo/results.jsonl}"

cd "$PROJECT_DIR"

"$PYTHON_BIN" -m pip install -e .

"$PYTHON_BIN" -m esco_skill_batch build-index \
  --esco-csv "$ESCO_CSV" \
  --output-dir "$INDEX_DIR"

"$PYTHON_BIN" -m esco_skill_batch extract-batch \
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
echo "Results written to: $OUTPUT_FILE"
