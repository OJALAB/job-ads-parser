#!/usr/bin/env bash
set -euo pipefail

# Real server example using Ollama for extraction and lexical or hybrid matching.
# Recommended:
#   export ESCO_CSV=/path/to/esco/skills.csv
#   export INPUT_FILE=/path/to/job_ads.jsonl
# Optional:
#   export PYTHON_BIN=/opt/miniconda3/envs/llm/bin/python

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
PYTHON_BIN="${PYTHON_BIN:-python}"
ESCO_CSV="${ESCO_CSV:?Set ESCO_CSV to the downloaded ESCO skills CSV}"
INPUT_FILE="${INPUT_FILE:?Set INPUT_FILE to your JSONL/CSV/TXT input file}"
INDEX_DIR="${INDEX_DIR:-$PROJECT_DIR/.server/index}"
OUTPUT_FILE="${OUTPUT_FILE:-$PROJECT_DIR/.server/results.jsonl}"
OLLAMA_MODEL="${OLLAMA_MODEL:-qwen3:14b}"
TEXT_FIELD="${TEXT_FIELD:-description}"
ID_FIELD="${ID_FIELD:-id}"

cd "$PROJECT_DIR"

"$PYTHON_BIN" -m pip install -e .

"$PYTHON_BIN" -m esco_skill_batch build-index \
  --esco-csv "$ESCO_CSV" \
  --output-dir "$INDEX_DIR"

"$PYTHON_BIN" -m esco_skill_batch extract-batch \
  --input "$INPUT_FILE" \
  --output "$OUTPUT_FILE" \
  --index-dir "$INDEX_DIR" \
  --text-field "$TEXT_FIELD" \
  --id-field "$ID_FIELD" \
  --extractor ollama \
  --ollama-model "$OLLAMA_MODEL" \
  --mapping-backend lexical \
  --top-k 5 \
  --score-threshold 0.35

echo
echo "Results written to: $OUTPUT_FILE"
