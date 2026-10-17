#!/usr/bin/env bash
# Start Sandwich.cpp OpenAI-Compatible Server for OpenCode & Local Agents
set -euo pipefail

MODEL_PATH="${1:-/home/sandeep/ai-models/moe/Nemotron-3-Nano-30B-A3B-GGUF/Nemotron-3-Nano-30B-A3B-UD-IQ3_XXS.gguf}"
PORT="${PORT:-8081}"
HOST="${HOST:-127.0.0.1}"
ALIAS="${ALIAS:-sandwich-local}"
CTX="${CTX:-32768}"

echo "Starting Sandwich.cpp server on http://${HOST}:${PORT}/v1 (alias: ${ALIAS})..."
exec /home/sandeep/dev/sandwich/sandwich-cpp/build/bin/llama-server \
    -m "${MODEL_PATH}" \
    --sandwich-auto \
    --alias "${ALIAS}" \
    --host "${HOST}" \
    --port "${PORT}" \
    -c "${CTX}" \
    "$@"
