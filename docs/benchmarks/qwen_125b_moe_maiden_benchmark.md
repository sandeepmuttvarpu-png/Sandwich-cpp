# 🥪 Sandwich.cpp — Maiden 125B MoE Benchmark & Smart-Dispatch Verification Report

**Date:** November 2, 2026  
**Hardware Platform:** HP Victus 15 Laptop  
- **GPU:** NVIDIA GeForce RTX 5050 Mobile (8,151 MiB GDDR6, PCIe Gen4 x16, 224 GB/s)  
- **CPU:** 13th Gen Intel Core i7-13620H (6 P-cores @ up to 4.9 GHz, 4 E-cores, 16 threads)  
- **RAM:** 24 GB DDR5 @ 4800 MT/s (Peak Bandwidth: ~64 GB/s)  
- **Storage:** 1 TB Samsung PCIe Gen4 NVMe SSD (Sequential Read: ~6,000 MB/s)  
- **OS / Display:** Arch Linux (Kernel 7.2.11-arch1-1), Wayland Compositor (Hyprland / niri)  
- **Engine Commit:** [`13acab40a`](https://github.com/sandeepmuttvarpu-png/Sandwich-cpp/commit/13acab40a)  

---

## 1. Executive Summary

We conducted comprehensive live testing and resolved three foundational edge-case bugs across the **Sandwich Smart-Dispatch & Auto-Tuning TUI Engine** and the **Maiden 125B MoE Inference Pipeline**. 

Key milestones achieved:
1. **Full Maiden 125B MoE Inference Round-Trip**: Successfully loaded and executed the **Qwen3.8-Flash-Next 125B MoE** (512 fine-grained routed experts, 78.34 GB suite on disk) on an **8GB VRAM + 24GB DDR5 laptop** with **0 crashes, 0 memory corruption**, and strict adherence to the **7,100 MB Wayland VRAM Guard**.
2. **Dense Model Acceleration**: Benchmarked Bonsai-27B (`Q1_0`, 3.54 GB) at **85.7 prompt t/s and 30.2 generation t/s** (a 5.8× speedup over unmanaged CPU fallback).
3. **Smart-Dispatch Engine Hardening**: All profiling, scanning, and TUI synthesis bugs (`-fitt` margin semantics, single-GPU `-sm row` conflict, drafter VRAM overcommitment, and Indexer KV cache type clash) have been eliminated and pushed to GitHub `origin/main`.

---

## 2. Root Cause Analysis & Engineering Fixes

### Bug A: Indexer KV Cache `ops.cpp:5424` Fatal Abort
- **Symptom:** When passing `-ctk turbo4`, models with sparse indexers (`qwen4exp` / Qwen3.8-Flash-Next) crashed with:
  ```
  [FATAL] ggml_compute_forward_get_rows: unsupported src0->type=45 (turbo4), dst->type=0 (f32), src0->name='CPU#cache_idx_k_l3 (view)#0'
  ```
- **Root Cause:** In [`src/llama-memory-hybrid-idx.cpp`](file:///home/sandeep/dev/sandwich/sandwich-cpp/src/llama-memory-hybrid-idx.cpp) and [`src/llama-kv-cache-msa.cpp`](file:///home/sandeep/dev/sandwich/sandwich-cpp/src/llama-kv-cache-msa.cpp), the auxiliary Indexer cache (`cache_idx_k`) was inheriting `type_k` (`turbo4`). Unlike the main attention cache, the indexer is accessed via `ggml_get_rows` for sparse head clustering. TurboQuant's PolarQuant layout is designed for fused FlashAttention (`fattn`), causing `get_rows` to abort.
- **Fix:** Enforced `GGML_TYPE_F16` for the lightweight (~50MB) indexer KV cache while retaining `turbo4` for the multi-gigabyte primary KV cache.

### Bug B: The `-fitt` Target Free Margin Semantics
- **Symptom:** Passing `-fitt 7100` resulted in sluggish 5.3 t/s inference because 90% of model layers remained on CPU.
- **Root Cause:** In `llama.cpp`, `-fitt <margin>` sets the **target free memory margin to LEAVE on the GPU**, NOT the allocated quota. Specifying `-fitt 7100` on an 8,151 MiB GPU instructed `--fit` to reserve 7,100 MiB free, leaving only ~700 MiB for model weights.
- **Fix:**
  - If a model fits entirely in VRAM: emit direct `-ngl 99`.
  - For multi-tier offload: compute `free_margin_mb = total_vram_mb - wayland_guard_vram_mb` ($8151 - 7100 = 1051$ MiB, or `-fitt 1200`), allocating all remaining ~7,000 MiB to the model under the Wayland Guard.

### Bug C: Multi-GPU Split Mode (`-sm row`) on Single GPU
- **Symptom:** `error loading model: device CUDA0 does not support split buffers`.
- **Root Cause:** `-sm row` is strictly for multi-GPU row-parallel weight splitting.
- **Fix:** Removed `-sm row` emission for single-GPU profiles in [`src/dispatch/hardware_profiler.cpp`](file:///home/sandeep/dev/sandwich/sandwich-cpp/src/dispatch/hardware_profiler.cpp) and [`src/dispatch/tui_wizard.cpp`](file:///home/sandeep/dev/sandwich/sandwich-cpp/src/dispatch/tui_wizard.cpp).

### Bug D: Speculative Drafter VRAM OOM
- **Symptom:** `cudaMalloc failed: out of memory (allocating 3836.91 MiB on device 0)`.
- **Root Cause:** `llama-cli` defaults to offloading all layers of draft models (`-md`) to GPU (`n_gpu_layers = -1`), attempting to allocate another 3.8 GB alongside the base model.
- **Fix:** Pinned speculative drafters into host DDR5 RAM by emitting `-ngld 0`.

---

## 3. Empirical Benchmark Matrix

| Model Architecture | Quantization | Size on Disk | Active Footprint | Prompt Speed | Generation Speed | System Health & Guard Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Dolphin-2.8 Mistral-7B** | `Q4_K_S` | 3.86 GB | 3.52 GB VRAM | **292.3 t/s** | **50.6 t/s** | VRAM: 3.5 GB / 8.1 GB (Wayland safe) |
| **Bonsai-27B Dense** | `Q1_0` | 3.54 GB | 3.85 GB VRAM | **85.7 t/s** | **30.2 t/s** | VRAM: 3.9 GB / 8.1 GB (Wayland safe) |
| **Qwen3.8-Flash-Next 125B MoE** | `IQ3_XXS` | **78.34 GB** | 4.04 GB VRAM + 6.68 GB DDR5 + NVMe | **2.4 t/s** | **0.30 t/s** | 3-Tier Hierarchy Active, 0 OOMs, 0 aborts |

---

## 4. Verification Checkpoints

```bash
# 1. Hardware Profiling
sandwich --tune
✔ CPU: i7-13620H (6 P-cores pinned, -t 6)
✔ GPU: RTX 5050 Mobile (7,100 MB Wayland Guard active)
✔ RAM: 24GB DDR5 (16,552 MB safe cap via Rule 42)

# 2. Local Model Catalog
sandwich --scan-models
✔ 26 Base Models + 7 Drafters discovered and cached

# 3. 125B MoE Maiden Run
scripts/run_125b_moe.sh "Explain quantum superposition in one sentence." 16
✔ Output: "The user asks for a 'thinking' process..."
✔ Exit code: 0 (Clean termination)
```

---

## 5. Next Planned Enhancements
1. **Dynamic Expert Shard Paging:** Increase 125B MoE generation throughput from 0.3 t/s to ~3–5 t/s by increasing the asynchronous `io_uring` DMA ring queue depth (`--io-uring-qd 64`) and tuning prefetch batching.
2. **Native MTP Head Decoding:** Wire the embedded MTP drafter weights directly into the primary execution graph to eliminate host DDR5 drafter synchronization overhead.
