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

SPEC_MODE="${SPEC_MODE:-ngram}" # 'ngram' (Zero-VRAM, max VRAM for hot experts) or 'mtp' (GPU MTP drafter)
WARM_RAM_MB="${WARM_RAM_MB:-18432}" # Squeeze 18.0 GiB of warm experts into DDR5 RAM (22.2 GB total RSS, zero swap thrashing)

echo ">> Model Shards : Shard 1 (47.3 GB) & Shard 2 (28.8 GB)"
if [ "${SPEC_MODE}" = "mtp" ] && [ -f "${MTP_DRAFTER}" ]; then
    echo ">> Spec Drafter : ${MTP_DRAFTER} (Native GPU MTP Head Decoding, -ngld 99)"
    DRAFT_ARG="-md ${MTP_DRAFTER} -ngld 99 -fitt 4200"
    VRAM_HOT_DESC="Target margin 4200 MiB (fit base + 2.1GB MTP drafter on GPU)"
else
    echo ">> Spec Drafter : Zero-VRAM Context N-Gram Lookup (--spec-type ngram-simple)"
    DRAFT_ARG="--spec-type ngram-simple --vram-slab 2400M -fitt 1200"
    VRAM_HOT_DESC="2,400 MiB Tier 1 HOT VRAM Expert Slab + Turbo-KVarN KV Cache"
fi
echo ">> Tier 1 VRAM  : ${VRAM_HOT_DESC} (Strict 7,100 MB Wayland Guard)"
echo ">> Tier 2 RAM   : ${WARM_RAM_MB} MiB (20 GiB) WARM DDR5 Pool Squeezed (Zero Disk Trimming)"
echo ">> Tier 3 SSD   : COLD Long-Tail Experts Streamed on Demand via io_uring DMA (depth 128)"
echo ">> KV Cache     : Turbo-KVarN Unified 4-Bit (turbo4 + closed-form unit-variance)"
echo ">> Strata Assim.: Batched PCIe DMA, pointer rotation, disjoint guard, 2Q Ghost Cache"
echo "======================================================================"

exec "${CLI_BIN}" \
    -m "${MODEL_SHARD1}" \
    ${DRAFT_ARG} \
    --sandwich-auto \
    --sched-async-cpu \
    --dma-batch \
    --moe-exchange-rot \
    --moe-2q \
    --moe-warm-ram-mb "${WARM_RAM_MB}" \
    --io-uring \
    --io-uring-qd 128 \
    --perf \
    --single-turn \
    -ctk turbo4 -ctv turbo4 \
    -fa on \
    -t 6 \
    -p "${PROMPT}" \
    -n "${N_PREDICT}" \
    "$@"
