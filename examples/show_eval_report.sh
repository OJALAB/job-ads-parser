#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
REPORT_FILE="${REPORT_FILE:-$PROJECT_DIR/.demo/eval-ollama-report.md}"

cd "$PROJECT_DIR"
sed -n '1,220p' "$REPORT_FILE"
