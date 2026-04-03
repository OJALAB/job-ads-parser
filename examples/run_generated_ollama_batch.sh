#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
INPUT_FILE="${INPUT_FILE:-$PROJECT_DIR/examples/generated_jobs_multilingual.jsonl}"
ESCO_CSV="${ESCO_CSV:-$PROJECT_DIR/examples/sample_esco_skills.csv}"
INDEX_DIR="${INDEX_DIR:-$PROJECT_DIR/.demo/generated-ollama-index}"
OUTPUT_FILE="${OUTPUT_FILE:-$PROJECT_DIR/.demo/generated-ollama-results.jsonl}"
OLLAMA_MODEL="${OLLAMA_MODEL:-qwen3:14b}"
OLLAMA_URL="${OLLAMA_URL:-http://127.0.0.1:11434}"
TEXT_FIELD="${TEXT_FIELD:-description}"
ID_FIELD="${ID_FIELD:-id}"
MAPPING_BACKEND="${MAPPING_BACKEND:-lexical}"
TOP_K="${TOP_K:-5}"
SCORE_THRESHOLD="${SCORE_THRESHOLD:-0.35}"
MAX_RECORDS="${MAX_RECORDS:-3}"
OLLAMA_TIMEOUT_SECONDS="${OLLAMA_TIMEOUT_SECONDS:-600}"

cd "$PROJECT_DIR"

"$PYTHON_BIN" examples/generate_synthetic_jobs.py

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
  --score-threshold "$SCORE_THRESHOLD" \
  --max-records "$MAX_RECORDS"

echo
echo "Generated dataset: $INPUT_FILE"
echo "Batch results: $OUTPUT_FILE"
echo "Preview:"
sed -n '1,3p' "$OUTPUT_FILE"
