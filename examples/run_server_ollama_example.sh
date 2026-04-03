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
OLLAMA_URL="${OLLAMA_URL:-http://127.0.0.1:11434}"
TEXT_FIELD="${TEXT_FIELD:-description}"
ID_FIELD="${ID_FIELD:-id}"
MAPPING_BACKEND="${MAPPING_BACKEND:-lexical}"
TOP_K="${TOP_K:-5}"
SCORE_THRESHOLD="${SCORE_THRESHOLD:-0.35}"
OLLAMA_TIMEOUT_SECONDS="${OLLAMA_TIMEOUT_SECONDS:-600}"

cd "$PROJECT_DIR"
PYTHONPATH="${PYTHONPATH:-src}" "$PYTHON_BIN" -m esco_skill_batch build-index \
  --esco-csv "$ESCO_CSV" \
  --output-dir "$INDEX_DIR"

PYTHONPATH="${PYTHONPATH:-src}" "$PYTHON_BIN" -m esco_skill_batch extract-batch \
  --input "$INPUT_FILE" \
  --output "$OUTPUT_FILE" \
  --index-dir "$INDEX_DIR" \
  --text-field "$TEXT_FIELD" \
  --id-field "$ID_FIELD" \
  --extractor ollama \
  --ollama-model "$OLLAMA_MODEL" \
  --ollama-url "$OLLAMA_URL" \
  --ollama-timeout-seconds "$OLLAMA_TIMEOUT_SECONDS" \
  --mapping-backend "$MAPPING_BACKEND" \
  --top-k "$TOP_K" \
  --score-threshold "$SCORE_THRESHOLD"

echo
echo "Results written to: $OUTPUT_FILE"
