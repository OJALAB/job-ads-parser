#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GLINER_MODEL="${GLINER_MODEL:-$PROJECT_DIR/.demo/gliner-finetune-smoke/model/final-model}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/.demo/gliner-finetuned-eval}"

if [[ ! -d "$GLINER_MODEL" ]]; then
  echo "Model directory not found: $GLINER_MODEL" >&2
  exit 1
fi

cd "$PROJECT_DIR"
GLINER_MODEL="$GLINER_MODEL" OUTPUT_DIR="$OUTPUT_DIR" bash "$PROJECT_DIR/examples/run_eval_gliner_suite.sh"
