#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
QUEUE_FILE="${QUEUE_FILE:-$PROJECT_DIR/.demo/review-queue.jsonl}"
OUTPUT_CSV="${OUTPUT_CSV:-$PROJECT_DIR/.demo/review-queue.csv}"
DEVICE="${DEVICE:-auto}"

cd "$PROJECT_DIR"

PYTHON_BIN="$PYTHON_BIN" DEVICE="$DEVICE" bash "$PROJECT_DIR/examples/run_prepare_review_queue.sh"

PYTHONPATH=src "$PYTHON_BIN" -m esco_skill_batch export-review-csv \
  --input "$QUEUE_FILE" \
  --output "$OUTPUT_CSV"

echo "Review CSV: $OUTPUT_CSV"
