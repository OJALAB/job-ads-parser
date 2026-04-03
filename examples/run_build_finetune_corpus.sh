#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
INPUT_FILE="${INPUT_FILE:-$PROJECT_DIR/examples/kprm_warszawa_live_10.jsonl}"
REVIEWED_QUEUE="${REVIEWED_QUEUE:-$PROJECT_DIR/.demo/reviewed-queue.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/.demo/review-corpus}"
TEXT_FIELD="${TEXT_FIELD:-skills_text}"
HOLDOUT_RATIO="${HOLDOUT_RATIO:-0.2}"
SEED="${SEED:-42}"

cd "$PROJECT_DIR"

PYTHONPATH=src "$PYTHON_BIN" -m esco_skill_batch build-finetune-corpus \
  --input "$INPUT_FILE" \
  --reviewed-queue "$REVIEWED_QUEUE" \
  --output-dir "$OUTPUT_DIR" \
  --text-field "$TEXT_FIELD" \
  --holdout-ratio "$HOLDOUT_RATIO" \
  --seed "$SEED"

echo "Silver train: $OUTPUT_DIR/silver_train.jsonl"
echo "Manual holdout: $OUTPUT_DIR/manual_gold_holdout.jsonl"
echo "Aliases: $OUTPUT_DIR/review_aliases.jsonl"
