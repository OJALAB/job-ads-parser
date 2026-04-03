#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
ESCO_CSV="${ESCO_CSV:-$PROJECT_DIR/examples/sample_esco_skills.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/.demo/eval-suite/hf-token}"
HF_MODEL="${HF_MODEL:-jjzha/escoxlmr_skill_extraction}"
HF_AGGREGATION_STRATEGY="${HF_AGGREGATION_STRATEGY:-simple}"
HF_ENTITY_LABELS="${HF_ENTITY_LABELS:-}"
HF_DEVICE="${HF_DEVICE:--1}"
TOP_K="${TOP_K:-5}"

cd "$PROJECT_DIR"
mkdir -p "$OUTPUT_DIR"

for case_name in easy hard; do
  if [[ "$case_name" == "easy" ]]; then
    GOLD_FILE="$PROJECT_DIR/examples/eval_gold_skills_pl.jsonl"
  else
    GOLD_FILE="$PROJECT_DIR/examples/eval_gold_skills_pl_hard.jsonl"
  fi

  INDEX_DIR="$OUTPUT_DIR/index-$case_name"
  PREDICTIONS_FILE="$OUTPUT_DIR/predictions-$case_name.jsonl"
  METRICS_FILE="$OUTPUT_DIR/metrics-$case_name.json"
  REPORT_FILE="$OUTPUT_DIR/report-$case_name.md"

  PYTHONPATH="${PYTHONPATH:-src}" "$PYTHON_BIN" -m esco_skill_batch build-index \
    --esco-csv "$ESCO_CSV" \
    --output-dir "$INDEX_DIR" >/dev/null

  PYTHONPATH="${PYTHONPATH:-src}" "$PYTHON_BIN" -m esco_skill_batch extract-batch \
    --input "$GOLD_FILE" \
    --output "$PREDICTIONS_FILE" \
    --index-dir "$INDEX_DIR" \
    --text-field description \
    --id-field id \
    --extractor hf_token_classifier \
    --hf-model "$HF_MODEL" \
    --hf-aggregation-strategy "$HF_AGGREGATION_STRATEGY" \
    --hf-entity-labels "$HF_ENTITY_LABELS" \
    --hf-device "$HF_DEVICE" \
    --mapping-backend lexical \
    --top-k "$TOP_K"

  PYTHONPATH="${PYTHONPATH:-src}" "$PYTHON_BIN" -m esco_skill_batch evaluate \
    --gold "$GOLD_FILE" \
    --predictions "$PREDICTIONS_FILE" \
    --top-k "$TOP_K" > "$METRICS_FILE"

  PYTHONPATH="${PYTHONPATH:-src}" "$PYTHON_BIN" -m esco_skill_batch report \
    --gold "$GOLD_FILE" \
    --predictions "$PREDICTIONS_FILE" \
    --output "$REPORT_FILE" \
    --top-k "$TOP_K" >/dev/null
done

"$PYTHON_BIN" examples/summarize_eval_suite.py \
  "$OUTPUT_DIR/metrics-easy.json" \
  "$OUTPUT_DIR/metrics-hard.json"

echo "Reports:"
echo "  $OUTPUT_DIR/report-easy.md"
echo "  $OUTPUT_DIR/report-hard.md"
