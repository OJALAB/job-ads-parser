#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_DIR="${RUN_DIR:-$PROJECT_DIR/.demo/gliner-finetune-fair}"
DATA_DIR="$RUN_DIR/data"
MODEL_DIR="$RUN_DIR/model"
BASELINE_DIR="$RUN_DIR/baseline"
FINETUNED_DIR="$RUN_DIR/finetuned"
GLINER_BASE_MODEL="${GLINER_BASE_MODEL:-urchade/gliner_large-v2.1}"
MAX_STEPS="${MAX_STEPS:-250}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"
SAVE_STEPS="${SAVE_STEPS:-50}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
RUN_BASELINE="${RUN_BASELINE:-1}"

cd "$PROJECT_DIR"

"$PYTHON_BIN" -m pip install -e '.[gliner]'

PYTHON_BIN="$PYTHON_BIN" OUTPUT_DIR="$DATA_DIR" bash "$PROJECT_DIR/examples/run_prepare_gliner_training_data.sh"

if [[ "$RUN_BASELINE" == "1" ]]; then
  GLINER_MODEL="$GLINER_BASE_MODEL" OUTPUT_DIR="$BASELINE_DIR" bash "$PROJECT_DIR/examples/run_eval_gliner_suite.sh"
fi

PYTHONPATH=src "$PYTHON_BIN" -m esco_skill_batch train-gliner \
  --train-data "$DATA_DIR/train.json" \
  --dev-data "$DATA_DIR/dev.json" \
  --model-name "$GLINER_BASE_MODEL" \
  --output-dir "$MODEL_DIR" \
  --max-steps "$MAX_STEPS" \
  --train-batch-size "$TRAIN_BATCH_SIZE" \
  --eval-batch-size "$EVAL_BATCH_SIZE" \
  --save-steps "$SAVE_STEPS" \
  --logging-steps "$LOGGING_STEPS" \
  --save-total-limit 2 \
  --use-cpu

GLINER_MODEL="$MODEL_DIR/final-model" OUTPUT_DIR="$FINETUNED_DIR" bash "$PROJECT_DIR/examples/run_eval_gliner_suite.sh"

if [[ "$RUN_BASELINE" == "1" ]]; then
  "$PYTHON_BIN" "$PROJECT_DIR/examples/compare_model_benchmarks.py" \
    "baseline/easy" "$BASELINE_DIR/metrics-easy.json" \
    "baseline/hard" "$BASELINE_DIR/metrics-hard.json" \
    "finetuned/easy" "$FINETUNED_DIR/metrics-easy.json" \
    "finetuned/hard" "$FINETUNED_DIR/metrics-hard.json"
fi

echo
echo "Prepared data: $DATA_DIR"
echo "Base model: $GLINER_BASE_MODEL"
if [[ "$RUN_BASELINE" == "1" ]]; then
  echo "Baseline evaluation: $BASELINE_DIR"
fi
echo "Fine-tuned model: $MODEL_DIR/final-model"
echo "Fine-tuned evaluation: $FINETUNED_DIR"
