#!/usr/bin/env bash
# ==============================================================================
# 🥪 Sandwich.cpp — 125B Qwen3.8-Flash-Next MoE Maiden Run on 8GB VRAM + 24GB RAM
# Hardware Target: HP Victus 15 (RTX 5050 8GB GDDR6, i7-13620H, 24GB DDR5)
# ==============================================================================
set -euo pipefail

MODEL_DIR="/home/sandeep/ai-models/moe/Qwen3.8-Flash-Next-GSQ-RCO-GGUF"
MODEL_SHARD1="${MODEL_DIR}/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_XXS-00001-of-00002.gguf"
MTP_DRAFTER="${MODEL_DIR}/mtp-Qwen3.8-Flash-Next-Q4_0.gguf"
CLI_BIN="/home/sandeep/dev/sandwich/sandwich-cpp/build/bin/llama-cli"

echo "======================================================================"
echo "🥪 Sandwich.cpp — Qwen 125B MoE Engine"
echo "======================================================================"

if [ ! -f "${MODEL_SHARD1}" ]; then
    echo "❌ Model shard 1 not found at ${MODEL_SHARD1}"
    echo "Checking current download progress in .cache..."
    du -sh "${MODEL_DIR}/.cache/huggingface/download/" 2>/dev/null || true
    exit 1
fi

PROMPT="${1:-Write a high-performance concurrent work-stealing thread pool in modern C++20 with full lock-free ring buffers.}"
N_PREDICT="${2:-128}"

echo ">> Model        : ${MODEL_SHARD1}"
if [ -f "${MTP_DRAFTER}" ]; then
    echo ">> Spec Drafter : ${MTP_DRAFTER}"
    DRAFT_ARG="--draft-model ${MTP_DRAFTER}"
else
    echo ">> Spec Drafter : Native Embedded MTP Head"
    DRAFT_ARG="--native-mtp"
fi
echo ">> VRAM Fit     : 7,600 MiB (Squeezed 97.4% saturation)"
echo ">> KV Cache     : TurboQuant 4-Bit (turbo4)"
echo ">> MoE Streaming: 3-Tier io_uring direct NVMe + 2Q Ghost Cache (14GB DDR5 pool)"
echo ">> Strata Assim.: Batched PCIe DMA, 4 prefetcher threads, pointer rotation"
echo "======================================================================"

exec "${CLI_BIN}" \
    -m "${MODEL_SHARD1}" \
    ${DRAFT_ARG} \
    --sandwich-auto \
    --sched-async-cpu 1 \
    --moe-prefetch-threads 4 \
    --dma-batch \
    --moe-exchange-rot \
    --moe-disjoint-guard \
    --moe-2q \
    --io-uring \
    -fitt 7600 \
    -ctk turbo4 -ctv turbo4 \
    -fa 1 \
    -t 6 \
    -p "${PROMPT}" \
    -n "${N_PREDICT}"
