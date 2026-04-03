#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
GOLD_FILE="${GOLD_FILE:-$PROJECT_DIR/examples/eval_gold_skills_pl.jsonl}"
ESCO_CSV="${ESCO_CSV:-$PROJECT_DIR/examples/sample_esco_skills.csv}"
INDEX_DIR="${INDEX_DIR:-$PROJECT_DIR/.demo/eval-index}"
PREDICTIONS_FILE="${PREDICTIONS_FILE:-$PROJECT_DIR/.demo/eval-passthrough-results.jsonl}"
METRICS_FILE="${METRICS_FILE:-$PROJECT_DIR/.demo/eval-passthrough-metrics.json}"
REPORT_FILE="${REPORT_FILE:-$PROJECT_DIR/.demo/eval-passthrough-report.md}"

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
  --extractor passthrough \
  --mentions-field gold_skills \
  --mapping-backend lexical \
  --top-k 5 \
  --no-progress

PYTHONPATH="${PYTHONPATH:-src}" "$PYTHON_BIN" -m esco_skill_batch evaluate \
  --gold "$GOLD_FILE" \
  --predictions "$PREDICTIONS_FILE" \
  --top-k 5 | tee "$METRICS_FILE"

PYTHONPATH="${PYTHONPATH:-src}" "$PYTHON_BIN" -m esco_skill_batch report \
  --gold "$GOLD_FILE" \
  --predictions "$PREDICTIONS_FILE" \
  --output "$REPORT_FILE" \
  --top-k 5 >/dev/null

echo "Report written to: $REPORT_FILE"
