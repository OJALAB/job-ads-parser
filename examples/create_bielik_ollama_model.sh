#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
OLLAMA_MODEL_NAME="${OLLAMA_MODEL_NAME:-bielik-pl:7b}"
BIELIK_GGUF="${BIELIK_GGUF:?Set BIELIK_GGUF to the local path of Bielik GGUF}"

cd "$PROJECT_DIR"

TMP_MODELFILE="$(mktemp)"
trap 'rm -f "$TMP_MODELFILE"' EXIT

cat > "$TMP_MODELFILE" <<EOF
FROM $BIELIK_GGUF

TEMPLATE """<s>{{ if .System }}<|start_header_id|>system<|end_header_id|>
{{ .System }}<|eot_id|>{{ end }}{{ if .Prompt }}<|start_header_id|>user<|end_header_id|>
{{ .Prompt }}<|eot_id|>{{ end }}<|start_header_id|>assistant<|end_header_id|>
{{ .Response }}<|eot_id|>"""

PARAMETER stop "<|start_header_id|>"
PARAMETER stop "<|end_header_id|>"
PARAMETER stop "<|eot_id|>"
PARAMETER temperature 0.1
EOF

ollama create "$OLLAMA_MODEL_NAME" -f "$TMP_MODELFILE"

echo
echo "Created Ollama model: $OLLAMA_MODEL_NAME"
