# 📖 Complete Guide to Sandwich.cpp Flags

sandwich.cpp merges upstream llama.cpp parameters with custom MoE expert caching, TurboQuant compression, and speculative drafter flags.
──────
## 1. 🥪 Sandwich Special Flags (Core Engine)

 Flag                     │ Argument │ Description                                                                                                                                       │     Default
──────────────────────────┼──────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼─────────────────
 --sandwich-auto          │ None     │ Master Auto-Dispatcher: Inspects GGUF metadata, detects MoE vs. Dense topologies, sizes VRAM slabs, enables P-core CPU pinning, selects drafter   │      false
                          │          │ pairing, activates TurboQuant, and sets the Anti-Pattern Guard.                                                                                   │
 --sandwich-moe-cache-mb  │ N (MiB)  │ Explicitly overrides the size of the GPU VRAM expert slab for MoE models (e.g. 4633 for 4.6 GB).                                                  │  Auto-budgeted
 --sandwich-pcore-threads │ N        │ Explicitly overrides P-core thread allocation on Intel hybrid CPUs (Alder/Raptor Lake).                                                           │ 6 (all P-cores)
 --no-extra-bufts         │ None     │ The Anti-Pattern Guard: Prohibits -rtr (row-interleaved repacking) from allocating host CPU memory that destroys GPU VRAM staging in hybrid mode. │      true
──────
## 2. ⚡ Mixture-of-Experts (MoE) & Caching Flags

 Flag                           │ Argument                      │ Description                                                                                              │            Default
────────────────────────────────┼───────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────┼───────────────────────────────
 --moe-cache-slots              │ N                             │ Thecodacus Dual-Path MoE Cache: number of routed experts per layer kept resident in GPU VRAM hot slabs.   │   Auto (24-32 in auto)
 --moe-cache-size               │ BYTES                         │ BeeLlama Dynamic Slab Cache: total byte size of O(1) LRU VRAM expert cache slab (e.g., 4294967296 for 4GB)│            0 (off)
 --moe-cache-profile            │ FNAME (CSV)                   │ Routing profile CSV (from llama-moe-trace) used to pick which specific experts to cache in GPU VRAM.    │        Auto-discovered
 --moe-prefetch-threads         │ N                             │ Number of worker threads for the asynchronous lookahead prefetcher (madvise(MADV_WILLNEED) & MADV_COLD). │               2
 -ncmoe, --n-cpu-moe            │ N                             │ Number of expert layers to force onto host CPU RAM while keeping remaining layers in VRAM slabs.         │               0
 -ncffn, --n-cpu-ffn            │ N                             │ Keep dense FFN weights of first N layers in host CPU RAM (for dense/hybrid layer offloading).           │               0
 --override-tensor              │ PAT=BUFT                      │ Force specific tensor patterns to host RAM or VRAM (e.g. ffn_.*_exps=CPU).                               │             Auto
 --moe-hot-only                 │ None                          │ Skip CPU cold experts and run hot VRAM-resident experts only for high-speed speculative drafting.        │            false
 --sched-async-cpu              │ None                          │ Run CPU graph splits on async worker thread to overlap GPU splits in parallel (zero PCIe bubble).        │     Auto (--sandwich-auto)
 --moe-3tier                    │ None                          │ Enable Phase 12 3-tier hierarchical MoE engine (HOT VRAM -> WARM DDR5 THP -> COLD NVMe io_uring DMA).  │             true
 --no-moe-3tier                 │ None                          │ Disable Phase 12 3-tier hierarchical MoE memory engine.                                                  │             false
 --moe-warm-ram-mb              │ N (MiB)                       │ Phase 12 Tier 2 WARM DDR5 RAM pool budget in MiB (e.g. 12288 for 12 GiB, THP-aligned).                   │             12288
 --moe-2q                       │ None                          │ Enable Phase 12 2Q & Ghost-Cache replacement policy (A1in 25% probation + Am 75% frequent + A1out ghost)│             true
 --no-moe-2q                    │ None                          │ Disable Phase 12 2Q replacement policy and revert to naive single-queue LRU.                             │             false
──────
## 3. 🚀 Speculative Decoding Flags

 Flag                                 │ Argument                            │ Description                                                                            │               Default
──────────────────────────────────────┼─────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────┼─────────────────────────────────────
 -md, --model-draft                   │ PATH                                │ Path to the draft model GGUF (e.g., small 1B drafter, DFlash2 head, or MTP assistant). │            Auto-detected
 --draft-max, -dft-n                  │ N                                   │ Maximum number of candidate tokens drafted per iteration (usually 3 to 8).             │                  5
 --draft-min, -dft-p                  │ FLOAT                               │ Minimum cumulative probability threshold for draft token acceptance.                   │                0.75
 -ngld, --gpu-layers-draft            │ N                                   │ Number of GPU layers offloaded for the draft model (-1 or 99 for 100% VRAM).           │                 -1
 --speculative-type                   │ TYPE                                │ Speculative algorithm: draft-simple, draft-dflash, draft-mtp, ngram-simple.            │            Auto-detected
 --ngram-simple-n                     │ N                                   │ Context n-gram search window length (used for zero-VRAM prompt lookup).                │                 12
 --ngram-simple-m                     │ N                                   │ Maximum number of speculative tokens proposed from matching context n-grams.           │                 48
 --spec-tree                          │ None                                │ Enable EAGLE3 / P-EAGLE speculative tree decoding (amortizes target forward pass).      │     Auto (--sandwich-auto)
 --no-spec-tree                       │ None                                │ Disable speculative tree decoding and force classic linear sequence drafting.           │                false
 --spec-tree-depth                    │ N (1-8)                             │ Maximum depth D of the speculative candidate tree.                                     │                  3
 --spec-tree-width                    │ N (1-8)                             │ Maximum branching width W per node for candidate exploration.                          │                  2
 -ctkd, --cache-type-k-draft          │ TYPE                                │ KV cache data type for K for draft model (turbo4, f16, q8_0, etc.).                    │       Same as main model
 -ctvd, --cache-type-v-draft          │ TYPE                                │ KV cache data type for V for draft model (turbo4, f16, q8_0, etc.).                    │       Same as main model
──────
## 4. 💾 Quantization & KV Cache Compression Flags (TurboQuant)

 Flag                             │ Argument                         │ Description                                                                                       │             Default
──────────────────────────────────┼──────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────┼─────────────────────────────────
 -ctk, --cache-type-k             │ TYPE                             │ Key cache quantization type:                                                                      │   turbo4 (--sandwich-auto)
                                  │                                  │ • turbo4 / turbo4_0: 4-bit PolarQuant (75% VRAM compression, Tensor Core Flash Attention).        │
                                  │                                  │ • turbo3 / turbo3_0: 3-bit PolarQuant (81.25% compression, host RAM offload via -nkvo).          │
                                  │                                  │ • turbo2 / turbo2_0: 2-bit PolarQuant (87.5% compression, eliminates DDR5 bus bottleneck).       │
                                  │                                  │ • Standard types: f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1.                           │
 -ctv, --cache-type-v             │ TYPE                             │ Value cache quantization type (supports turbo4, turbo3, turbo2, f16, q8_0, etc.).                 │   turbo4 (--sandwich-auto)
 -fa, --flash-attn                │ on|off|auto                      │ Enables Flash Attention. Accelerates attention computation and cuts peak activation memory.       │               on
 -nkvo, --no-kv-offload           │ None (or 0|1)                    │ Force KV cache into host system DDR5 RAM instead of GPU VRAM (critical for 16K-64K on large MoE). │           0 (in VRAM)
 -kvo, --kv-offload               │ None                             │ Keep KV cache resident in GPU VRAM (default behavior).                                            │             enabled
 -cram, --cache-ram               │ N (MiB)                          │ Persistent prompt prefix cache RAM buffer size in MiB (e.g. 4096 for 4 GiB fast prefix reuse).    │        4096 (in auto)
 --cache-idle-slots               │ None                             │ Save idle slots to prompt cache on new task, and clear them when using unified KV.                │             enabled
 --prompt-cache                   │ FNAME                            │ File to save/load persistent prompt cache state for instant session resumption.                   │              none
 --prompt-cache-all               │ None                             │ Remember all prompt inputs in the cache across turns.                                             │             false
 --prompt-cache-ro                │ None                             │ Open prompt cache file in read-only mode (prevents modification).                                 │             false
 -kvu, --kv-unified               │ None                             │ Use single unified KV buffer shared across all sequences (reduces multi-slot fragmentation).      │              auto
 --sink-tokens, --attn-sinks      │ N                                │ Number of initial attention sink tokens to permanently preserve during context sliding-window.     │               4
 --context-shift                  │ None                             │ Enable context shifting for infinite text generation beyond model context length.                 │             enabled
──────
## 5. ⚙ Compute, Layer Offload & Threading Flags

 Flag                                     │ Argument                                │ Description                                                                │                 Default
──────────────────────────────────────────┼─────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────┼─────────────────────────────────────────
 -m, --model                              │ PATH                                    │ Path to the primary GGUF model file.                                       │                Required
 -ngl, --n-gpu-layers                     │ N                                       │ Number of layers to offload to GPU VRAM (99 = all layers; 0 = CPU only).   │             Auto-calculated
 --split-kv                               │ None                                    │ Technique #32 & #42: Asymmetric Split-KV Engine. Pins KV cache & attention mechanism into VRAM while 100% of dense weights execute on CPU (eliminates PCIe ping-pong when >= 70% in RAM). │ Auto (--sandwich-auto)
 --no-split-kv                            │ None                                    │ Disable Asymmetric Split-KV and revert to standard device-matched KV buffer placement.     │                 false
 -t, --threads                            │ N                                       │ Number of CPU compute threads during token generation (pinned to P-cores). │                    6
 -tb, --threads-batch                     │ N                                       │ Number of compute threads during batch/prompt evaluation (prefill).        │                    6
 -c, --ctx-size                           │ N                                       │ Context window size in tokens (0 = load default from GGUF metadata).       │                    0
 -b, --batch-size                         │ N                                       │ Logical batch size for prompt processing.                                  │                  2048
 -ub, --ubatch-size                       │ N                                       │ Physical micro-batch size for GPU compute tiles (avoids GPU VRAM spikes).  │                   512
 -sm, --split-mode                        │ none|layer|row                          │ GPU tensor splitting mode for multi-device setups.                         │                  layer
 --no-host                                │ None                                    │ Bypass host buffer allowing extra device buffers to be used.               │                  false
──────
## 6. 🌐 HTTP Server & Web UI Flags (llama-server)

 Flag                                │ Argument                            │ Description                                                                             │               Default
─────────────────────────────────────┼─────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────┼─────────────────────────────────────
 --host                              │ IP                                  │ Network interface IP to bind to (0.0.0.0 for all interfaces, 127.0.0.1 for local only). │              127.0.0.1
 --port                              │ PORT                                │ Listening TCP port for HTTP and Web UI.                                                 │                8080
 --path                              │ DIR                                 │ Path to static Web UI directory (serves index.html, style.css, chat.js).                │         tools/server/public
 --api-prefix                        │ PREFIX                              │ URL path prefix for all endpoints (e.g. /api).                                          │                 ""
 --cors-origins                      │ LIST                                │ Comma-separated list of allowed CORS origins (* for all).                               │                 ""
 --timeout                           │ SEC                                 │ Server network read/write timeout in seconds.                                           │                 600
 -np, --parallel                     │ N                                   │ Number of parallel request slots for continuous batching.                               │          1 (or 4 in auto)
──────
## 7. 🎯 Sampling & Generation Flags

 Flag                                       │ Argument                                  │ Description                                                          │                  Default
────────────────────────────────────────────┼───────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────┼───────────────────────────────────────────
 -temp, --temp                              │ FLOAT                                     │ Generation temperature (0.0 = greedy deterministic, 0.7 = creative). │                    0.8
 --top-p                                    │ FLOAT                                     │ Top-P (nucleus) sampling threshold.                                  │                    0.9
 --top-k                                    │ N                                         │ Top-K sampling cutoff (0 = disabled).                                │                    40
 --min-p                                    │ FLOAT                                     │ Minimum token probability relative to top token (0.05 recommended).  │                   0.05
 --repeat-penalty                           │ FLOAT                                     │ Repetition penalty factor (1.0 = disabled, 1.1 = mild penalty).      │                    1.0
 -n, --predict                              │ N                                         │ Number of tokens to predict (-1 = infinity, -2 = fill context).      │                    -1
──────
## 8. 🛡️ Phase 8 Dense Wing Overhaul Flags (Techniques #25 - #28)

 Flag                                │ Argument │ Description                                                                                                   │            Default
─────────────────────────────────────┼──────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────┼───────────────────────────────
 --jit-down-quant                    │ None     │ [P1] In-Memory JIT CPU Down-Quantization: Transcode CPU layers to Q2_K in-memory during mmap (cuts DDR5 load ~50%).│     Auto (--sandwich-auto)
 --no-jit-down-quant                 │ None     │ Disable in-memory JIT down-quantization of CPU-offloaded layers.                                              │             false
 --jit-down-quant-start-layer        │ N        │ First layer index to down-quantize in CPU memory (-1 = first CPU layer).                                      │              -1
 --cpu-cache-path                    │ FNAME    │ Explicit path to save/load persistent transcoded CPU layer cache sidecar (<model>.sandwich-cpu.bin).         │ <model>.sandwich-cpu.bin
 --thp-align                         │ None     │ [P2] 2MB Transparent HugePages (THP) Alignment: Align CPU allocations to 2MB boundaries & madvise(MADV_HUGEPAGE). │     Auto (--sandwich-auto)
 --no-thp-align                      │ None     │ Disable 2MB THP alignment for host CPU memory allocations.                                                    │             false
 --early-exit-thresh                 │ THRESH   │ [P4] Boundary Early-Exit / Dynamic LayerSkip: Softmax confidence threshold to emit token directly from VRAM.  │      0.92 (0.0 = off)
──────
## 9. 🧠 Phase 9 Predictive Hardware-Aware Topology & TurboQuant Auto-Router (Technique #29)

 Flag                                │ Argument │ Description                                                                                                   │            Default
─────────────────────────────────────┼──────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────┼───────────────────────────────
 --sandwich-predict                  │ None     │ [Technique #29] Automatically predicts Asymmetric Split vs. Hybrid Offload and auto-routes TurboQuant KV.     │      true (in auto)
 --no-sandwich-predict               │ None     │ Disable predictive solver and TurboQuant auto-routing (forces static layer fitting).                          │             false
──────
## 10. 🏎️ Phase 10 & 11 Pipeline Overhaul & Hardware Push (Techniques #30 - #35)

 Flag                                │ Argument │ Description                                                                                                   │            Default
─────────────────────────────────────┼──────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────┼───────────────────────────────
 --spec-pipe                         │ None     │ [P1 / #30] Pipelined Double-Buffered Speculative Execution: Overlaps GPU drafting and CPU verification.      │     Auto (--sandwich-auto)
 --no-spec-pipe                      │ None     │ Disable double-buffered speculative pipelining (forces sequential lock-step execution).                       │             false
 --spec-tree-adaptive                │ None     │ [P2 / #31] Adaptive Spec-Tree PID Controller: Dynamically tunes tree depth D(1-7) & split p(0.05-0.40).       │     Auto (--sandwich-auto)
 --no-spec-tree-adaptive             │ None     │ Disable PID dynamic speculative tree parameter adaptation.                                                    │             false
 --split-kv, --kv-vram               │ None     │ [P3 / #32] Asymmetric Split-KV Engine: Pure VRAM KV Cache on CPU-offloaded models (n_gpu_layers=0).          │     Auto (--sandwich-auto)
 --vnni, --avx-vnni                  │ None     │ [P4 / #33] AVX2-VNNI P-Core Vector Execution: Native dpbusd dot-product vectorization for Q2_K & IQ2_XXS.     │     Auto (--sandwich-auto)
 --preset agent                      │ None     │ [P5 / #34] One-Click Agent Server Preset: Configures 64K context, turbo4 KV, persistent prompt cache & slots.│             false
 --fp4, --mxfp4                      │ None     │ [P6 / #35] Blackwell Native FP4 / MXFP4 Tensor Offload: Offloads NVFP4/MXFP4 layers to sm_120 Tensor Cores.  │             false
──────
## 11. 🚀 Phase 12 External High-Impact Native Kernels & Latency Eliminators (Techniques #36 - #39)

 Flag                                │ Argument │ Description                                                                                                   │            Default
─────────────────────────────────────┼──────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────┼───────────────────────────────
 -fanq, --flash-attn-native-quants   │ None     │ [Technique #36] Quantized-Native CUDA FlashAttention: Fuses quant dequantization into MMA Tensor Cores (saves 1.25GB)│     Auto (--sandwich-auto)
 --no-flash-attn-native-quants       │ None     │ Disable quantized-native MMA FlashAttention kernels.                                                          │             false
 --logits-topk                       │ N        │ [Technique #37] GPU Top-K Logits Reduction: Reduces logits to top-K on GPU, cutting PCIe transfer by 99.9%.  │       64 (in spec)
 --turbo-mma                         │ None     │ [Technique #38] Fused Turbo MMA FlashAttention: Fuses PolarQuant turbo4/3/2 tile loading directly into MMA.   │     Auto (--sandwich-auto)
 --sched-async-cpu                   │ None     │ [Technique #39] Asynchronous CPU Split Worker Overlap: Runs CPU graph splits on worker thread behind GPU.   │     Auto (--sandwich-auto)
──────
## 12. ⚡ Phase 12 3-Tier Hierarchical MoE Memory System Engine (Technique #40)

 Flag                                │ Argument │ Description                                                                                                   │            Default
─────────────────────────────────────┼──────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────┼───────────────────────────────
 --moe-3tier                         │ None     │ [Phase 12] 3-Tier Hierarchical MoE Memory System: HOT VRAM (GDDR6) -> WARM DDR5 (THP) -> COLD NVMe (io_uring).│     Auto (--sandwich-auto)
 --no-moe-3tier                      │ None     │ Disable 3-tier hierarchical MoE memory system (falls back to 2-tier).                                         │             false
 --moe-warm-ram-mb                   │ N (MiB)  │ Dedicated Tier 2 WARM DDR5 RAM pool budget in MiB backed by 2MB Transparent Huge Pages (default: 12288 / 12GB)│            12288
──────
## 13. 🔬 Phase 13 & 14 Frontier Architecture & Compression Techniques (Techniques #43 - #52)

 Flag                                │ Argument │ Description                                                                                                   │            Default
─────────────────────────────────────┼──────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────┼───────────────────────────────
 --uvm-zero-copy                     │ None     │ [Technique #43] Unified Zero-Copy Memory & UVM Prefetching: registers host DDR5 mmap via cudaMemAdviseSetAccessedBy.│     Auto (--sandwich-auto)
 --no-uvm-zero-copy                  │ None     │ Disable unified zero-copy memory and UVM driver prefetching.                                                  │             false
 --radix-cache                       │ None     │ [Technique #45] Radix-Tree Persistent Prefix Caching (RadixAttention): sub-ms KV reuse on multi-turn prompts. │     Auto (--sandwich-auto)
 --no-radix-cache                    │ None     │ Disable Radix-Tree persistent prefix caching.                                                                 │             false
 --radix-cache-tokens                │ N        │ [Technique #45] Maximum number of prompt prefix tokens cached in the dynamic Radix Tree (default: 65536).     │            65536
 --moe-mass-thresh                   │ FLOAT    │ [Technique #47] Dynamic MoE Softmax Entropy Pruning: drops routed experts once cumulative mass >= tau (0.96). │             0.96
 --moe-gds                           │ None     │ [Technique #48] GPUDirect Storage (GDS / libcufile): direct NVMe-to-VRAM PCIe DMA bypass for cold experts.   │             false
 --no-moe-gds                        │ None     │ Disable GPUDirect Storage and fall back to io_uring / host buffer staging.                                    │             true
 --native-mtp                        │ None     │ [Technique #49] Zero-Memory Native MTP Speculative Rollout: binds directly to base model MTP heads (0MB draft).│            false
 --no-native-mtp                     │ None     │ Disable native Multi-Token Prediction speculative rollout.                                                    │             true
 --asym-kv-squeeze                   │ None     │ [Technique #50] Asymmetric KV Outlier Channelling: Keys to turbo4, Values to turbo2 (2.75 bits/tok squeeze). │             false
 --chunked-prefill                   │ None     │ [Technique #51] Chunked Context Prefill & Ping-Pong Activation Staging: caps prefill transient VRAM < 350MB.  │     Auto (ctx >= 32k)
 --no-chunked-prefill                │ None     │ Disable chunked context prefill and evaluate entire prompt in a single pass.                                  │             false
 --latent-mla, --flash-mla           │ None     │ [Technique #52] Latent FlashMLA Direct Tensor Core MMA: fuses uncompressed MLA manifold projection into MMA.  │     Auto (MLA models)
 --no-latent-mla, --no-flash-mla     │ None     │ Disable direct Latent FlashMLA and uncompress full multi-head tensors in memory.                              │             false
 --spec-select                       │ None     │ [Piggidragon/Strata Port] Dynamic Multi-Drafter Arbitration: per-step expected token utility max over multiple drafters. │ Auto (--sandwich-auto)
 --turboquant-varn                   │ None     │ [Technique #54] Closed-Form Dual-Axis Variance Normalization: eliminates long-horizon reasoning drift.          │ Auto (ctx >= 8192)
 --no-turboquant-varn                │ None     │ Disable TurboQuant dual-axis variance normalization.                                                          │             false
──────





