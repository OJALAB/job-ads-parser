#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
GOLD_FILE="${GOLD_FILE:-$PROJECT_DIR/examples/eval_gold_skills_pl.jsonl}"
ESCO_CSV="${ESCO_CSV:-$PROJECT_DIR/examples/sample_esco_skills.csv}"
INDEX_DIR="${INDEX_DIR:-$PROJECT_DIR/.demo/eval-ollama-index}"
PREDICTIONS_FILE="${PREDICTIONS_FILE:-$PROJECT_DIR/.demo/eval-ollama-results.jsonl}"
OLLAMA_MODEL="${OLLAMA_MODEL:-bielik-pl-7b}"
OLLAMA_URL="${OLLAMA_URL:-http://127.0.0.1:11434}"
OLLAMA_PROMPT_PRESET="${OLLAMA_PROMPT_PRESET:-bielik_pl}"
OLLAMA_TIMEOUT_SECONDS="${OLLAMA_TIMEOUT_SECONDS:-600}"
TOP_K="${TOP_K:-5}"
MAX_RECORDS="${MAX_RECORDS:-8}"

cd "$PROJECT_DIR"

PYTHONPATH="${PYTHONPATH:-src}" "$PYTHON_BIN" -m esco_skill_batch build-index \
  --esco-csv "$ESCO_CSV" \
  --output-dir "$INDEX_DIR"

PYTHONPATH="${PYTHONPATH:-src}" "$PYTHON_BIN" -m esco_skill_batch extract-batch \
  --input "$GOLD_FILE" \
  --output "$PREDICTIONS_FILE" \
  --index-dir "$INDEX_DIR" \
  --text-field description \
  --id-field id \
  --extractor ollama \
  --ollama-model "$OLLAMA_MODEL" \
  --ollama-url "$OLLAMA_URL" \
  --ollama-prompt-preset "$OLLAMA_PROMPT_PRESET" \
  --ollama-timeout-seconds "$OLLAMA_TIMEOUT_SECONDS" \
  --mapping-backend lexical \
  --top-k "$TOP_K" \
  --max-records "$MAX_RECORDS"

PYTHONPATH="${PYTHONPATH:-src}" "$PYTHON_BIN" -m esco_skill_batch evaluate \
  --gold "$GOLD_FILE" \
  --predictions "$PREDICTIONS_FILE" \
  --top-k "$TOP_K"
