#!/usr/bin/env bash
# ==============================================================================
# 🥪 Sandwich.cpp — 125B Qwen3.8-Flash-Next MoE Maiden Run on 8GB VRAM + 24GB RAM
# Hardware Target: HP Victus 15 (RTX 5050 8GB GDDR6, i7-13620H, 24GB DDR5)
# ==============================================================================
set -euo pipefail

MODEL_DIR="/home/sandeep/ai-models/moe/Qwen3.8-Flash-Next-GSQ-RCO-GGUF"
MODEL_SHARD1="${MODEL_DIR}/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_XXS-00001-of-00002.gguf"
MODEL_SHARD2="${MODEL_DIR}/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_XXS-00002-of-00002.gguf"
MTP_DRAFTER="${MODEL_DIR}/mtp-Qwen3.8-Flash-Next-Q4_0.gguf"
CLI_BIN="/home/sandeep/dev/sandwich/sandwich-cpp/build/bin/llama-cli"

echo "======================================================================"
echo "🥪 Sandwich.cpp — Qwen 125B MoE Engine"
echo "======================================================================"

if [ ! -f "${MODEL_SHARD1}" ]; then
    echo "❌ Model shard 1 not found at ${MODEL_SHARD1}"
    exit 1
fi
if [ ! -f "${MODEL_SHARD2}" ]; then
    echo "❌ Model shard 2 not found at ${MODEL_SHARD2}"
    exit 1
fi

PROMPT="${1:-Write a high-performance concurrent work-stealing thread pool in modern C++20 with full lock-free ring buffers.}"
N_PREDICT="${2:-64}"
shift 2 2>/dev/null || shift $# 2>/dev/null || true

echo ">> Model Shards : Shard 1 (47.3 GB) & Shard 2 (28.8 GB)"
if [ -f "${MTP_DRAFTER}" ]; then
    echo ">> Spec Drafter : ${MTP_DRAFTER} (Native GPU MTP Head Decoding, -ngld 99)"
    DRAFT_ARG="-md ${MTP_DRAFTER} -ngld 99"
else
    echo ">> Spec Drafter : Disabled"
    DRAFT_ARG=""
fi
echo ">> VRAM Fit     : Target margin 1200 MiB (Strict 7,100 MB Wayland VRAM Guard)"
echo ">> KV Cache     : Turbo-KVarN Unified 4-Bit (turbo4 + closed-form unit-variance)"
echo ">> MoE Streaming: 3-Tier io_uring direct NVMe (depth 128) + 2Q Ghost Cache (14GB DDR5 pool)"
echo ">> Strata Assim.: Batched PCIe DMA, pointer rotation, disjoint guard"
echo "======================================================================"

exec "${CLI_BIN}" \
    -m "${MODEL_SHARD1}" \
    ${DRAFT_ARG} \
    --sandwich-auto \
    --sched-async-cpu \
    --dma-batch \
    --moe-exchange-rot \
    --moe-disjoint-guard \
    --moe-2q \
    --io-uring \
    --io-uring-qd 128 \
    --perf \
    --single-turn \
    -fitt 1200 \
    -ctk turbo4 -ctv turbo4 \
    -fa on \
    -t 6 \
    -p "${PROMPT}" \
    -n "${N_PREDICT}" \
    "$@"
