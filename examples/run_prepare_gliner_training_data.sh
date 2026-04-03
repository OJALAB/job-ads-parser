#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/.demo/gliner-training-data}"
TEXT_FIELD="${TEXT_FIELD:-description}"
SKILLS_FIELD="${SKILLS_FIELD:-gold_skills}"
MAX_TOKENS="${MAX_TOKENS:-300}"
WINDOW_STRIDE="${WINDOW_STRIDE:-300}"
DEV_RATIO="${DEV_RATIO:-0.25}"
SEED="${SEED:-42}"
KEEP_EMPTY="${KEEP_EMPTY:-0}"
TRAIN_FILE="${TRAIN_FILE:-$PROJECT_DIR/examples/train_gold_skills_pl.jsonl}"

INPUT_FILES=("$TRAIN_FILE")

CMD=(
  "$PYTHON_BIN" -m esco_skill_batch prepare-gliner-data
  --input "${INPUT_FILES[@]}"
  --output-dir "$OUTPUT_DIR"
  --text-field "$TEXT_FIELD"
  --skills-field "$SKILLS_FIELD"
  --dev-ratio "$DEV_RATIO"
  --seed "$SEED"
  --max-tokens "$MAX_TOKENS"
  --window-stride "$WINDOW_STRIDE"
)

if [[ "$KEEP_EMPTY" == "1" ]]; then
  CMD+=(--keep-empty)
fi

cd "$PROJECT_DIR"
PYTHONPATH=src "${CMD[@]}"
