# 🏛️ The 6 Core Architectural Pillars of Sandwich.cpp

**Engine Foundation:** Sandwich.cpp v3.0.0 (Upstream Master Synchronized `b12200+` / `3837bf0bc`)  
**Design Philosophy:** Maximum Physical Silicon Saturation via Strict Invariants, Zero Bloat, and Automated Hardware Calibration.

---

## Executive Summary & Consolidation

Across 17 development eras and 82 recorded experimental techniques in the engineering diary, Sandwich.cpp has matured from an exploratory research codebase into a hardened inference engine. To eliminate redundant flag aliases, placebo environment variables, and cognitive bloat, the 82 historical techniques are consolidated into **6 Core Architectural Pillars**.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        SANDWICH.CPP CORE ARCHITECTURE                                 │
├──────────────────────────┬──────────────────────────┬──────────────────────────────────┤
│ 1. 3-TIER MEMORY ENGINE  │ 2. TURBO-KVARN KV CACHE  │ 3. ZERO-MEMORY SPECULATION       │
│ • Linux io_uring DMA     │ • WHT Gaussian Rotation  │ • Native MTP Heads Rollout       │
│ • 2Q + Ghost-Cache       │ • Polar Lloyd-Max Quant  │ • DFlash2 / EAGLE Tree Decoding  │
│ • Warm DDR5 THP Slabs    │ • Unit-Variance Scaling  │ • Pipelined Double-Buffering     │
├──────────────────────────┼──────────────────────────┼──────────────────────────────────┤
│ 4. CONTEXT & ATTENTION   │ 5. UNIVERSAL AUTO-TUNER  │ 6. DISTRIBUTED SWARM INTERCONNECT│
│ • Flash-Decoding (SMs)   │ • Zero-Config Silicon Cal│ • Thunderbolt 4 / USB4 DMA       │
│ • Radix Tree Prefix Cache│ • Dynamic VRAM Guard     │ • 2-Bit Polar Network Quant      │
│ • H2O Dynamic Eviction   │ • P-Core Topology Pinning│ • Hot Zero-Crash Failover        │
└──────────────────────────┴──────────────────────────┴──────────────────────────────────┘
```

---

## Pillar 1: 3-Tier Hierarchical MoE Memory System
* **Primary Objective:** Break the VRAM capacity wall by streaming Mixture-of-Experts weights from NVMe through RAM into VRAM with zero CPU memory copies.
* **Consolidated Techniques:** #2 (Thecodacus Dual-Path), #3 (BeeLlama Dynamic Slab), #5 (Async CPU execution), #6 (Pinned host DMA), #7 (Early-lookahead prefetch), #8 (CUDA graph capture), #37 (3-Tier Engine: VRAM $\to$ DDR5 $\to$ NVMe), #38 (2Q + Ghost-Cache replacement), #51 (Disjoint peer cache guard), #52 (Batched DMA), #53 (Exchange buffer rotation), #74 (Asymmetric parallel prefetch), #75 (Dedicated RAM cache), #80 (GPUDirect Storage cuFile DMA), and #81 (Upstream CPU-GPU miss sharing `--moe-cache-cpu-share`).
* **Active Runtime Flags:**
  * `--io-uring`, `--io-uring-qd <64|128>`
  * `--moe-2q`, `--moe-probation <float>`
  * `--moe-warm-ram-mb <MB>`, `--ram-cache <MB>`
  * `--moe-cache-cpu-share <float>`
  * `--asym-prefetch`, `--dma-batch`
* **Physical Hardware Impact:** Enables 35B–125B MoE models (e.g. Qwen 35B, GPT-OSS 120B) to run on 8GB consumer GPUs at up to 34+ tokens/sec.

---

## Pillar 2: Turbo-KVarN Unified Polar KV Cache Compression
* **Primary Objective:** Compress Key/Value cache tensors to 4-bit, 3-bit, and 2-bit with mathematically proven unit-variance reconstruction, completely preventing long-horizon attention collapse.
* **Consolidated Techniques:** #9 (TurboQuant WHT + Polar Lloyd-Max), #33 (Quantized-Native FlashAttention), #35 (Fused TurboQuant MMA Tensor Core decode), #44 (Asymmetric KV outlier channeling), and #48 (Turbo-KVarN Closed-Form Dual-Axis Variance Normalization).
* **Active Runtime Flags:**
  * `-ctk turbo4 -ctv turbo4`, `--turbo-kvran`
  * `--turbo-mma`, `-fanq` (`--flash-attn-native-quants`)
  * `--asym-kv-squeeze`
* **Physical Hardware Impact:** Reduces KV cache VRAM footprint by **72.5%** ($1.25\text{ GB}$ saved at 64k context) with zero intermediate F16 buffer allocations and zero scale drift across 128k context windows.

---

## Pillar 3: Zero-Memory Native MTP & Speculative Decoding Engine
* **Primary Objective:** Multiply generation throughput by $1.3\times\text{--}2.1\times$ using zero-memory latent Multi-Token Prediction heads or companion ultra-fast drafters.
* **Consolidated Techniques:** #12 (Multi-Model Speculative Subsystem), #20 (Hot-Only Speculative Drafting), #24 (EAGLE3 / P-EAGLE Spec-Tree), #26 (VRAM-Pinned Drafter), #28 (Pipelined Double-Buffered Speculation), #43 (Zero-Memory Native MTP), and #47 (Dynamic Drafter Arbitration).
* **Active Runtime Flags:**
  * `--native-mtp` (binds model's internal Multi-Token Prediction heads)
  * `-md <drafter.gguf> --spec-type <draft-dflash|draft-mtp>`
  * `--spec-tree`
* **Physical Hardware Impact:** Achieves **28.7 – 31.9 tok/s** decode on Qwen 35B MoE (+39% to +46% over baseline) on an RTX 5050 Mobile.

---

## Pillar 4: Context Scaling & High-Efficiency Attention
* **Primary Objective:** Eliminate quadratic attention bottlenecks, memory thrashing, and prompt prefill stalls across long contexts.
* **Consolidated Techniques:** #10 / #41 (Radix-Tree Prefix Caching), #11 / #46 (Latent FlashMLA MMA), #21 / #59 (Streaming Attention Sinks), #45 / #76 (Chunked Prefill & Ping-Pong Activation Staging), #49 (Prefill Head Aliasing), #50 (KVA Late-Layer Recurrence), #54 (Message-Boundary Delta Caching), #58 (Dynamic YaRN Scaling), #60 (Unrotated KV Staging), #77 (Heavy Hitter Oracle H2O Dynamic Eviction), #78 (PagedAttention dynamic block cache), and #79 (Sequence-Parallel Flash-Decoding across SMs).
* **Active Runtime Flags:**
  * `--flash-decoding`, `--flash-decoding-chunk-size <N>`
  * `--radix-cache`, `--radix-cache-tokens <N>`
  * `--h2o`, `--h2o-budget <N>`, `--h2o-heavy-hitters <N>`
  * `--chunked-prefill`, `--prefill-chunk-size <128|256>`
  * `--prefill-head-alias`, `--unrotated-kv`, `--dynamic-yarn`
* **Physical Hardware Impact:** Yields a **2.18x prompt processing speedup** (70+ tok/s prompt prefill on 35B MoE) while capping transient activation memory spikes under 350 MB.

---

## Pillar 5: Universal Hardware Auto-Calibration & Invariant Guardrails
* **Primary Objective:** Provide zero-config execution on any machine by auto-detecting silicon topology and enforcing strict memory invariants.
* **Consolidated Techniques:** #1 (Hardware-Aware Auto-Dispatcher), #13 (Intel P-Core / AMD CCX Pinning), #19 (Online Warm-Up Auto-Profiling), #25 (2MB THP Buffer Alignment), #27 (Predictive Topology Solver), #30 (AVX2-VNNI P-Core Vector Execution), #39 (70% Host RAM Rule), #56 (Dynamic Elastic VRAM Resizing), #73 (Dynamic Memory Watermarking), and the Universal Dynamic VRAM Guard.
* **Active Runtime Flags:**
  * `./sandwich run -m <model.gguf>` (Automatically activates all optimal flags with zero manual configuration)
  * `--vram-guard <auto|LIMIT>` (Dynamic DRM-aware compositor protection)
  * `-t <optimal_p_cores>`
  * `--dynamic-watermark` (HWM 85% / LWM 70% proactive throttling)
* **Physical Hardware Impact:** 100% elimination of CUDA OOM crashes and OS OOM-killer invocations across laptops, desktops, and headless multi-GPU clusters.

---

## Pillar 6: Heterogeneous Distributed Swarm & Interconnect
* **Primary Objective:** Federate multiple disparate consumer devices (laptops, desktops, mini-PCs) into a unified compute cluster.
* **Consolidated Techniques:** #67 (P2P Distributed Heterogeneous VRAM Pooling), #68 (Heterogeneous Topology Solver), #69 (Zero-Copy Thunderbolt/USB4 DMA), #70 (2-Bit Polar Network Compression), #71 (Cross-Node Speculative Verification Pipeline), and #72 (Dynamic Zero-Crash Failover).
* **Active Runtime Flags:**
  * `--peer-nodes <endpoints>`
  * `--peer-dma`, `--peer-split-kv`
  * `--peer-turbo-net` (8x payload compression)
  * `--peer-spec-pipe`, `--peer-fallback`
* **Physical Hardware Impact:** Enables multi-machine offloading over standard USB4 / Thunderbolt 4 or 10GbE networking with automatic zero-crash fallback if a peer disconnects.

---

## Ablation & Pruning Ledger

The following historical technique designations have been formally audited, pruned, or aliased:
* **Pruned No-Ops & Flag Duplicates:**
  * `--turbo-varn`, `--turbo-vran`, `--turboquant-varn` $\to$ Standardized to `--turbo-kvran`.
  * `#22` and `#32` (Blackwell FP4 duplicate numbering) $\to$ Consolidated under Pillar 2.
  * `#61–#66` vs `#67–#72` (Duplicate peer networking numbering) $\to$ Consolidated under Pillar 6.
  * `setenv("STRATA_EXPDEC", "1")` / `setenv("STRATA_QSA_ROWS", "1")` $\to$ Deprecated as standalone techniques; integrated into native shared-memory kernels.
* **Pruned Deprecated Techniques:**
  * `--moe-gds` (consumer laptop PCIe P2P switch constraint) $\to$ Unified under GDS cuFile DMA with automatic direct io_uring fallback.
  * `--entropy-pruning` $\to$ Deprecated to safeguard reasoning fidelity.
  * `--uvm-zero-copy` $\to$ Pruned in favor of deterministic pinned host DMA.
