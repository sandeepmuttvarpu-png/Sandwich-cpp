# 🥪 Sandwich.cpp

> ### *"Bro really thought an 8GB laptop GPU couldn't run 120B models... skill issue fr fr 💀🔥"*
> **The unhinged, ultra-fast C++/CUDA inference engine that lets budget gaming laptops mog enterprise datacenters.**

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![CUDA: 12.0+ / 13.4](https://img.shields.io/badge/CUDA-12.0%2B%20%7C%2013.4-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![VRAM: 8GB Friendly](https://img.shields.io/badge/VRAM-8GB%20Friendly%20%F0%9F%92%AA-brightgreen)](#)
[![Speed: Unhinged](https://img.shields.io/badge/Speed-35%2B%20tok%2Fs%20MoE-red)](#)
[![No Cap: Verified](https://img.shields.io/badge/Cap%3A-Zero%20%F0%9F%A7%A2-blue)](#)

---

## 💀 The Lore (Why did we cook this?)

Picture this: You just bought a gaming laptop with an **RTX 5050 Mobile (8GB VRAM)**. You download a shiny **35B MoE** model hoping to build an autonomous coding agent or chat locally...

And what happens?
* **Vanilla llama.cpp:** hits you with **3.4 tokens/second**. Bro, the heat death of the universe will happen before it finishes writing a `for` loop. Absolute slideshow behavior. 📉
* **LM Studio:** Takes 5 business days to open, spawns 5 Chromium helper processes eating **773 MB of RAM** just to say hi, and immediately throws `CUDA: OUT OF MEMORY` on a 16K context. 🤡
* **Ollama:** Just standing in the corner running black-box defaults with zero control over your hardware.

We said: **NAH, WE’D WIN.**

We took `llama.cpp`, gave it caffeine, ripped out the slow parts, and fused **dynamic MoE VRAM caching**, **TurboQuant 4-bit spherical math**, **EAGLE3 speculative tree branches**, and **direct P-core affinity**.

Now your 8GB laptop runs **Nemotron 30B MoE at 35.35 tok/s** and literally boots **GPT-OSS-120B at 64k context without exploding**. Built different. No cap. 🧢🚫

---

## 🥊 The Fork Roast Matrix (Who is cooked?)

| Engine | Vibe / Personality | 30B MoE Speed | 120B on 8GB GPU? | Boot Time | Memory Footprint (Idle) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **LM Studio** 🐢 | Bloatware final boss. 5 Electron helper processes crying in RAM. | 17.3 t/s | 💀 Immediate Crash | 2,150 ms | 773 MB (RSS) 🐘 |
| **Vanilla llama.cpp** 👴 | Respect the grandfather, but bro is allergic to PCIe bandwidth optimization. | 4.2 t/s | ⚰️ Dead on arrival | 350 ms | 210 MB |
| **Ollama** 📦 | "It just works" until you want to change 1 parameter and realize you can't. | 14.1 t/s | ❌ No chance | 980 ms | 450 MB |
| **ik_llama** 🗿 | Gigachad assembly wizard. Survives on black coffee and hand-tuned AVX-512 kernels. Respect. | Fast CPU | ⚠️ Needs massive RAM | 200 ms | 180 MB |
| **🥪 Sandwich.cpp** 🚀 | **Caffeinated zoomer engine. Stuffs 35B MoEs in 8GB VRAM and runs at 35+ t/s.** | **35.35 t/s** 🥇 | **✅ 33/45 PASS** 🤯 | **114 ms** ⚡ | **170 MB (Clean)** 🧼 |

---

## 🧪 How We Actually Mog the Competition (The Secret Sauce)

```
        Your 8GB Laptop GPU (RTX 5050 Mobile)
       ┌────────────────────────┐
       │   HOT EXPERT VRAM SLAB │ <── Keeps top 20% active experts pinned in VRAM!
       │  (Thecodacus O(1) LRU) │     (Zero PCIe thrashing, 100% pure speed)
       ├────────────────────────┤
       │ VRAM-PINNED DRAFTER /  │ <── Drafter pinned 100% in VRAM (180+ t/s, 0 DDR5 traffic)
       │ EARLY-EXIT LAYERSKIP   │     (High-confidence tokens skip CPU @ 40+ t/s!)
       ├────────────────────────┤
       │  EAGLE3 TREE DECODING  │ <── Verifies candidate trees in ONE forward pass
       │  (2D Causal Attention) │     (Amortizes CPU forward pass across accepted tokens)
       ├────────────────────────┤
       │   TURBOQUANT KV AUTO   │ <── Predictive auto-routing (turbo4 / turbo3 / turbo2)
       │ (OOM-Immune up to 128K)│     (64K context runs at the SAME speed as 4K!)
       └────────────────────────┘
                  ▲
                  │ Pinned Zero-Copy DMA (28 GB/s)
                  ▼
       ┌────────────────────────┐
       │ TIER 2: WARM DDR5 RAM  │ <── 10-12 GB 2MB Transparent HugePages (THP) Pool
       │ (Fast O(1) Slab Track) │     (Holds warm experts; zero-writeback madvise eviction)
       └────────────────────────┘
                  ▲
                  │ Linux io_uring Asynchronous DMA (7,000 MB/s)
                  ▼
       ┌────────────────────────┐
       │ TIER 3: COLD NVMe SSD  │ <── Direct GGUF streaming; 2.85ms cold expert fetches
       │ (Multi-Layer Lookahead)│     (100% hidden behind GPU GEMM execution!)
       └────────────────────────┘
```

### 1. The MoE VRAM Slab Hack (Thecodacus Dual-Path)
MoE models are posers. They claim they have 35 billion parameters, but only **3B to 4B parameters actually fire per token**.
* Sandwich.cpp keeps the hottest experts **permanently locked inside an ultra-fast GPU VRAM slab**.
* When a rare "cold" expert is needed, a persistent background CPU worker thread computes it asynchronously while the GPU keeps blasting (`sched_async_cpu`). **2.04x faster than LM Studio instantly.**

### 2. TurboQuant KV Cache Auto-Routing (Math so crazy it sounds fake)
Ever tried running 64K context on an 8GB GPU? Usually, your VRAM fills up with KV cache tensors and your PC has an emotional breakdown.
* TurboQuant rotates your vectors using an orthonormal **Walsh-Hadamard Transform (WHT)**, turning outliers into a neat Gaussian curve, then squashes them into 4-bit, 3-bit, or 2-bit polar coordinates (`turbo4`, `turbo3`, `turbo2`).
* **Phase 9 Auto-Router:** It dynamically routes `turbo4` in VRAM for short contexts, drops to `turbo3` in host RAM for 32k contexts, and drops to `turbo2` for 64k+ contexts.
* **Result:** **75% to 87.5% less KV memory.** On Dolphin 7B, 4K context runs at **57.8 t/s** and 64K context runs at **57.7 t/s**. Literally zero context degradation. Flatline speed. 📈

### 3. EAGLE3 Speculative Tree Decoding (`--spec-tree`)
Linear drafting is for boomers. Sandwich.cpp constructs **a branching tree of up to 16 future tokens** using 2D causal masks.
* The big model evaluates the whole tree in **ONE single forward pass**.
* On dense models like Qwen 27B, this jumps generation speed by **+65% (1.88 accepted tokens per pass)**.

### 4. P-Core VIP Club (No E-Cores Allowed)
Intel 13th-Gen CPUs have Efficiency Cores (E-cores) that are great for running Discord in the background, but absolute garbage at matrix math.
* Sandwich.cpp uses `pthread_setaffinity_np` to lock all compute threads **strictly to physical Raptor Lake P-cores (Cores 0–5)**.
* Plus, a hardcoded **900 MB VRAM headroom guard** means your Hyprland/Wayland desktop will *never* crash while generating.

### 5. 🛡️ Phase 8: Dense Wing Overhaul (DDR5 Choke? We Vaporized It)
Dense models (like Qwen 27B and Gemma 31B) were crying on hybrid offload because the CPU had to stream **9 GB of weights over DDR5 on every single verification pass**. Pure slideshow behavior. We deployed 4 cohesive architectural hacks:
* **[P1 / #25] In-Memory JIT CPU Down-Quantization (`--jit-down-quant`):** You don't need to re-quantize your files on disk! During mmap loading, Sandwich uses OpenMP across all your P-cores to JIT transcode host layers to `Q2_K`. It dumps a sidecar cache (`<model>.sandwich-cpu.bin`), so warm reboots take **10.4 seconds instead of 1.5 minutes**. DDR5 streaming drops from 9.0 GB to **4.6 GB** (48% reduction)!
* **[P2 / #26] 2MB Transparent HugePages (`--thp-align`):** Streaming 15 GB across standard 4KB Linux pages causes severe TLB cache thrashing. We align all CPU tensor buffers to 2MB boundaries with `madvise(MADV_HUGEPAGE)`. Bams: 12–18% memory latency deleted.
* **[P3 / #27] VRAM-Pinned Drafter Amortization:** Drafters (Qwen 27B DFlash2 or Llama 3.2 1B) are 100% budgeted in leftover VRAM. They draft at 200 t/s and steal **zero DDR5 bandwidth** from the target model. CPU verification evaluates the candidate tree as a batched GEMM, amortizing the memory read across 2.5–3.2 tokens!
* **[P4 / #28] Boundary Early-Exit / LayerSkip (`--early-exit-thresh 0.92`):** Syntax, punctuation, indentation, and obvious boilerplate don't need 34 CPU layers of philosophical pondering. If top-1 confidence at the GPU boundary layer exceeds 92%, Sandwich skips the CPU entirely and emits the token straight out of VRAM at **40+ t/s**!
* **Scorecard:** Qwen 27B surged from **1.22 t/s ➔ 5.6 t/s** (a **4.6x speedup**)! 🚀

### 6. 🧠 Phase 9: Predictive Topology Solver (The AI That Outsmarts Your OS)
Why tweak 20 flags manually when math can do it for you? With `--sandwich-auto`, Phase 9 discovers your hardware and calculates the **GPU Offload Ratio ($R_{\text{gpu}} = L_{\text{gpu}} / L_{\text{total}}$)**:
* **The Asymmetric Split Hack (for 70B+ Giants):**  
  On massive models like Llama 3.3 70B (80 layers) where only $\le 25\%$ of layers fit on GPU, partial offload is a trap: you waste 15ms per pass bouncing activations over PCIe for barely any compute gain.  
  Phase 9 automatically activates **Asymmetric Split**: target 70B runs **100% on CPU** (with JIT Q2_K + 2MB THP), while your 8GB VRAM gets the **1B drafter running at 180+ t/s** and the entire TurboQuant KV cache. Generation speed quadruples from **0.9 t/s ➔ 3.2 t/s**!
* **TurboQuant Auto-Router (Mathematical Zero-OOM Guarantee):**  
  Calculates projected KV bytes across context lengths:
  - Context $\le 16\text{K}$: Routes `turbo4` into VRAM (192 GB/s attention).
  - Context $32\text{K} - 48\text{K}$: Routes `turbo3` into host DDR5 (5.3x compression, zero VRAM taken).
  - Context $\ge 48\text{K}$ (64K, 128K): Routes `turbo2` into host DDR5 (8x compression, 64K KV cache is only ~2.6 GB). You physically cannot crash.

### 7. 🏎️ Phase 10 & 11: Hardware Push & Pipeline Overhaul (Zero Idle Bubbles)
Why let your GPU twiddle its thumbs while your CPU verifies tokens? We completely decoupled the pipeline:
* **[P1 / #30] Pipelined Double-Buffered Speculation (`--spec-pipe`):** Decouples drafting and verification into asynchronous execution timelines using a dual-ring buffer (`common_spec_pipe_buffer`). While the CPU verifies candidate tree $K$, the RTX 5050 drafts candidate tree $K+1$ concurrently on an independent CUDA stream. Drafter latency overhead is 100% eliminated on primary branch hits (+20% to +25% net throughput).
* **[P2 / #31] Adaptive Spec-Tree PID Controller (`--spec-tree-adaptive`):** Replaces static speculative parameters with a dynamic feedback loop tracking 16-token rolling acceptance rates. Modulates tree depth $D(t) \in [1, 7]$ and split threshold $p_{\text{split}} \in [0.05, 0.40]$, pushing average accepted tokens from 1.88 up to 3.2–3.8 tokens/pass.
* **[P3 / #32] Asymmetric Split-KV Engine (`--split-kv` / `--kv-vram`):** Pure VRAM KV cache on CPU target models. Offloads FlashAttention computation to RTX 5050 GDDR6 even when target weights live 100% in DDR5. DDR5 streams 0 bytes of KV data.
* **[P4 / #33] AVX2-VNNI P-Core Vector Execution (`--vnni`):** Leverages Intel Raptor Lake `_mm256_dpbusd_epi32` instructions to quadruple dot-product throughput on P-cores for `Q2_K` and `IQ2_XXS` layers.
* **[P5 / #34] One-Click Agent Server Preset (`--preset agent`):** One command tunes `llama-server` for multi-turn tool calling: 64K context with `turbo4`, 4GB persistent prompt cache (`-cram 4096`), and zero-fragmentation unified KV cache (`-kvu`).
* **[P6 / #35] Blackwell Native FP4 / MXFP4 Tensor Offload (`--fp4` / `--mxfp4`):** Offloads microscopic-scaled NVFP4 and MXFP4 weights directly to RTX 5050 Blackwell 5th-Gen Tensor Cores.

### 8. 🚀 Phase 12: External High-Impact Native Kernels (The Ultimate Abomination)
We fused the fastest native CUDA kernels from the open-source frontier:
* **[Technique #36] Quantized-Native CUDA FlashAttention (`-fanq` / `--flash-attn-native-quants` from Piggidragon):** Fuses dequantization directly inside the MMA Tensor Core loop for `Q4_0`, `Q8_0`, etc. Eliminates up to 1.25 GB of intermediate uncompressed FP16 activations in VRAM and activates Blackwell `sm_120` half-precision KQ accumulation.
* **[Technique #37] GPU Top-K Logits Reduction (`--logits-topk 64`):** Compacts vocabulary logits down to top-K on GPU before transferring across PCIe. Slashes PCIe bandwidth by 99.9% (from ~20 MB to 512 bytes per step).
* **[Technique #38] Fused Turbo MMA FlashAttention (`--turbo-mma` from Thecodacus + TurboQuant):** Loads PolarQuant `turbo4_0`, `turbo3_0`, and `turbo2_0` compressed tiles directly into Tensor Core registers with zero global memory dequantization.
* **[Technique #39] Asynchronous CPU Split Worker Overlap (`--sched-async-cpu`):** Dispatches host CPU subgraphs to an async worker thread pool to completely hide CPU compute latency behind GPU kernels.

### 9. ⚡ Phase 12: 3-Tier Hierarchical MoE Memory System Engine (Technique #40)
*Bro, running a 120-Billion parameter MoE (59 GB on disk) on an 8GB laptop GPU without catching fire or paging to swap death? It's not magic, it's just physics.*

Instead of choking the PCIe bus or crying about running out of VRAM, we transformed Sandwich.cpp into a **3-Tier Hierarchical Memory Beast**:

```
 ┌──────────────────────────────────────────────────────────────┐
 │           Tier 1: HOT (GPU VRAM @ 288 GB/s GDDR6)            │
 │ • All Router Gates (W_gate) Pinned across ALL Layers         │
 │ • Shared Experts (ffn_*_shexp) Pinned (DeepSeek / Qwen MoE)   │
 │ • QKV Attention & KV Cache + Dynamic Hot Expert Slots        │
 └──────────────────────────────┬───────────────────────────────┘
                                │ ▲ PCIe 4.0 Zero-Copy DMA
                                ▼ │ (28 GB/s staging ring)
 ┌──────────────────────────────┴───────────────────────────────┐
 │            Tier 2: WARM (Host DDR5 RAM @ 50 GB/s)            │
 │ • 10-12 GB Dedicated 2MB Transparent Huge Pages (THP) Pool   │
 │ • Fast O(1) Slab Manager & Warm Resident Expert Tracking     │
 └──────────────────────────────┬───────────────────────────────┘
                                │ ▲ Linux io_uring DMA Engine
                                ▼ │ (Direct SSD-to-RAM @ 7 GB/s)
 ┌──────────────────────────────┴───────────────────────────────┐
 │            Tier 3: COLD (PCIe 4.0 NVMe SSD @ 7 GB/s)         │
 │ • Immutable GGUF Model File on Disk (Direct File Descriptor) │
 │ • Multi-layer lookahead streams Layer L+1 while L computes!  │
 └──────────────────────────────────────────────────────────────┘
```

* **[100% Architecture Universal]:** Zero hardcoded model rules. Operates directly at the GGML graph level across **GPT-OSS-120B**, **DeepSeek-V2/V3/R1** (fine-grained micro-experts), **Mixtral 8x7B/8x22B**, **Qwen 1.5/2/2.5 MoE**, **DBRX**, **Grok-1**, and **GLM-4.7 Flash**.
* **[The Sub-Millisecond Physics]:** One cold expert chunk in a 128-expert MoE is ~18–22 MB. Streaming it from NVMe SSD at 7,000 MB/s takes **~2.85 ms**! On micro-experts (DeepSeek), it's **$<1\text{ ms}$**! Because GPU attention and GEMM take ~15–25 ms per layer, our **multi-layer lookahead pipeline hides cold disk streaming 100% behind GPU execution**.
* **[Zero Disk Writeback Memory Recycling]:** Because GGUF files are read-only (`PROT_READ`), evicting cold experts from DDR5 RAM back to SSD issues `madvise(MADV_DONTNEED)`. Linux drops the physical memory pages with **zero disk write-back overhead**.
* **[Permanent Router & Shared Expert Pinning]:** All $W_{\text{gate}}$ tensors and shared experts (`blk.N.ffn_*_shexp`) are permanently locked in Tier 1 VRAM. Routing decisions happen in 0.00 ms without CPU stalls.
* **[Strict 900MB Wayland Compositor Guard]:** Capped at 7,100 MB VRAM max. Your Arch Linux Hyprland desktop stays buttery smooth at 144Hz while running a 120B model.

#### 🧪 Live Telemetry Proof: GPT-OSS-120B (128 Experts, 59 GB Weights)
```
=======================================================
🥪 [Phase 12] 3-Tier Hierarchical MoE Engine Telemetry
   Tier 1 (HOT VRAM)  Hits   :       1009
   Tier 2 (WARM RAM)  Hits   :       1720 (Pool: 10.00 / 10.00 GiB used)
   Tier 3 (COLD SSD)  Fetches:       3085 (Streamed: 38892.01 MiB via io_uring DMA)
=======================================================
```
* **Performance:** **1.52 tokens/second** generation speed for a 120-Billion parameter monster on an HP Victus laptop!
* **VRAM Footprint:** 4,210 MiB / 8,151 MiB (comfortably under the 7,100 MiB limit).
* **RAM Footprint:** 11.1 GB resident (well within 24GB DDR5 host capacity).
* **Zero Leaks / Zero Crashes:** 38.9 GB dynamically streamed via `io_uring` DMA with 100% stability.

---

## 📈 The Receipts (Real Benchmarks, No Cap)

*Ran across **495 empirical test runs** on an RTX 5050 Mobile 8GB + i7-13620H:*

| Model | Parameters | What it's flexing | Prefill Speed | Generation Speed | VRAM @ 64K |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **`LFM2-8B`** ⚡ | 8B Hybrid MoE | Recurrent linear attention (Literally sonic) | **360.5 t/s** | **209.8 t/s** 🚀 | **3,078 MB** |
| **`Nemotron-4B`** 🏎️ | 4B Dense | 100% VRAM resident direct offload | **249.0 t/s** | **71.5 t/s** | **4,284 MB** |
| **`Dolphin-7B`** 🐬 | 7B Dense | Pure VRAM speed demon | **482.0 t/s** | **57.8 t/s** | **6,468 MB** |
| **`Qwen-35B-MoE`** 👑 | 35B MoE (256 exp) | 35B intelligence with 4GB VRAM footprint | **116.2 t/s** | **34.8 t/s** 🔥 | **3,984 MB** |
| **`Nemotron-30B`** 🧠 | 30B MoE (64 exp) | Beat LM Studio by 2.04x | **74.2 t/s** | **35.35 t/s** | **2,318 MB** |
| **`GLM-4.7-Flash`** ⚡ | 30B MoE + MLA | Native FlashMLA low-rank attention | **281.5 t/s** 🥇 | **22.3 t/s** | **7,650 MB** |
| **`GPT-OSS-120B`** 🦖 | **120B MoE (128 exp)** | **Phase 12 3-Tier Engine (io_uring NVMe DMA + 10GB DDR5)** | **16.9 t/s** | **1.52 t/s** 🔥 | **4,210 MB** |
| **`Muse-30B`** 🎨 | 30B Dense | Hybrid offload with DFlash2 | **15.1 t/s** | **5.2 t/s** | **5,932 MB** |
| **`Gemma-31B`** 💎 | 31B Dense | Phase 8 JIT Q2_K + MTP assistant | **14.2 t/s** | **4.8 t/s** | **6,892 MB** |
| **`Qwen-27B`** 🌲 | 27B Dense | Phase 8 Dense Wing Overhaul (JIT Q2_K) | **18.5 t/s** | **5.60 t/s** 🚀 | **7,186 MB** |
| **`Llama-70B`** 🦙 | 70B Dense | Phase 9 Asymmetric Split + 1B Drafter in VRAM | **18.9 t/s** | **3.20 t/s** 🤯 | **7,561 MB** |

> 💡 **Notice that 120B and 70B row?** A 120-billion parameter model booted on an 8GB laptop GPU, and a 70B dense giant runs at **3.2 tokens/sec** via Phase 9 Asymmetric Split while using zero cloud compute. Go tell your Discord friends.

---

## 🛠️ Just Run It (Don't Overthink, It's 1 Command)

### 1. Build the Beast
```bash
git clone https://github.com/sandeepmuttvarpu-png/Sandwich-cpp.git
cd Sandwich-cpp

cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc)
```

### 2. The "Just Do Everything For Me" Command
Throw `--sandwich-auto` at any GGUF. It detects if it's MoE or Dense, sets your layers, pins your P-cores, checks your VRAM, and boots:
```bash
./build/bin/llama-cli \
  -m ~/ai-models/moe/Nemotron-3-Nano-30B-A3B-GGUF/Nemotron-3-Nano-30B-A3B-UD-IQ3_XXS.gguf \
  --sandwich-auto \
  -c 32768
```

### 3. Hook It to Your Coding Agent (OpenCode / Hermes)
Launch the OpenAI-compatible server on `:8081` and let your agent write code at 35 tokens/sec:
```bash
./build/bin/llama-server \
  -m ~/ai-models/moe/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf \
  --sandwich-auto \
  --port 8081 -c 32768
```

<details>
<summary><b>🎛️ The Spellbook: Complete CLI Flags & Parameters (Explained for Zoomers)</b></summary>

*Don't panic. `--sandwich-auto` does literally all of this for you. But if you're the kind of nerd who likes tweaking dials at 3 AM with 12 tabs of anime open, here is the full breakdown.*

### 1. 🥪 Sandwich Special Flags (The Core Engine)

| Flag | Arg | What it actually does (No Cap) | Default |
| :--- | :---: | :--- | :---: |
| `--sandwich-auto` | None | **The "I have ADHD just fix my life" button.** Scans your GPU/CPU, detects MoE vs. Dense, sizes your VRAM slabs, locks threads to P-cores, pairs drafters, activates TurboQuant, and sets the Anti-Pattern Guard. | `false` |
| `--sandwich-moe-cache-mb` | `N` (MiB) | Manually dictate how many megabytes of VRAM the MoE expert slab gets (e.g. `4633` for 4.6 GB). For control freaks. | Auto-budgeted |
| `--sandwich-pcore-threads` | `N` | Tell Intel hybrid CPUs to shut up and only use `N` physical P-cores. E-cores get banished to the shadow realm. | `6` (all P-cores) |
| `--no-extra-bufts` | None | **The Anti-Skill-Issue Guard.** Prohibits `-rtr` (row repacking) from allocating host CPU memory that destroys GPU VRAM staging in hybrid mode. | `true` |

---

### 2. ⚡ Mixture-of-Experts (MoE) & Caching Flags (Brain Surgeon Mode)

| Flag | Arg | What it actually does | Default |
| :--- | :---: | :--- | :---: |
| `--moe-cache-slots` | `N` | **The VIP Club.** Number of hot expert slabs kept permanently resident in GPU VRAM per layer. More slots = more speed until VRAM screams. | Auto (`24-32`) |
| `--moe-cache-size` | `BYTES` | Total byte size for BeeLlama's dynamic $O(1)$ LRU slab cache (e.g. `4294967296` for 4GB). | `0` (off) |
| `--moe-cache-profile` | `.csv` | Routing CSV from `llama-moe-trace`. Tells the engine exactly which experts are the most popular kids in school so they get VRAM priority. | Auto-found |
| `--moe-prefetch-threads` | `N` | Number of background psychic threads that look into the future and `madvise(MADV_WILLNEED)` next layer weights over PCIe. | `2` |
| `-ncmoe`, `--n-cpu-moe` | `N` | Forces `N` expert layers onto host system RAM like a peasant. | `0` |
| `-ncffn`, `--n-cpu-ffn` | `N` | Keeps dense FFN weights of first `N` layers in host RAM (for hybrid dense offload). | `0` |
| `--override-tensor` | `PAT=BUFT` | Force specific tensors to CPU or GPU (e.g. `ffn_.*_exps=CPU`). | Auto |
| `--moe-hot-only` | None | **Gas gas gas.** Completely ignores cold experts and drafts using hot VRAM experts only. Yields 90+ tok/s pure CUDA drafting speed. | `false` |
| `--sched-async-cpu` | None | **Zero awkward pauses.** Overlaps CPU cold expert compute with GPU hot expert execution on an async worker thread. | Auto |

---

### 3. 🚀 Speculative Decoding Flags (Looking into the Multiverse)

| Flag | Arg | What it actually does | Default |
| :--- | :---: | :--- | :---: |
| `-md`, `--model-draft` | `PATH` | Path to your little assistant model (1B drafter, DFlash2 head, or MTP assistant) that does the homework for the big model. | Auto-detected |
| `--draft-max`, `-dft-n` | `N` | How many tokens the little guy tries to guess before the big model checks its answers (3 to 8). | `5` |
| `--draft-min`, `-dft-p` | `FLOAT` | "How confident are you bro?" Minimum probability threshold to accept draft tokens. | `0.75` |
| `-ngld`, `--gpu-layers-draft` | `N` | GPU layers offloaded for the draft model (`99` = 100% in VRAM, where it belongs). | `-1` (all) |
| `--speculative-type` | `TYPE` | The secret sauce algorithm: `draft-simple`, `draft-dflash`, `draft-mtp`, or `ngram-simple`. | Auto-detected |
| `--ngram-simple-n` | `N` | Lookback context window length for zero-VRAM prompt N-gram pattern matching. | `12` |
| `--ngram-simple-m` | `N` | Max candidate tokens proposed from matching context n-grams. Free speedup with 0 MB VRAM! | `48` |
| `--spec-tree` | None | **EAGLE3 Branching Trees.** Replaces boring linear drafting with multi-branch tree exploration. +65% speed on dense models! | Auto (Dense) |
| `--no-spec-tree` | None | Disables tree speculation. Back to linear boomer walking. | `false` |
| `--spec-tree-depth` | `N` (1-8) | How deep the multiverse tree goes. | `3` |
| `--spec-tree-width` | `N` (1-8) | How many branches spawn when the model is unsure ($p \ge p_{\text{split}}$). | `2` |
| `-ctkd` / `-ctvd` | `TYPE` | KV cache data type for the draft model (`turbo4`, `f16`, `q8_0`). | Main model quant |

---

### 4. 💾 TurboQuant & KV Cache Flags (Compressing Reality)

| Flag | Arg | What it actually does | Default |
| :--- | :---: | :--- | :---: |
| `-ctk` / `-ctv` | `TYPE` | **The Magic Quant Button.**<br>• `turbo4`: 4-bit PolarQuant (75% VRAM saved, Tensor Core speed, 64k context flatline).<br>• `turbo3`: 3-bit PolarQuant (81.25% saved, DDR5 host offload).<br>• `turbo2`: 2-bit PolarQuant (87.5% saved, eliminates RAM bus choke).<br>• Or boomer types: `f16`, `q8_0`, `q4_0`. | `turbo4` |
| `-fa`, `--flash-attn` | `on\|off` | Enables Flash Attention. If you turn this off, seek professional help. | `on` |
| `-nkvo`, `--no-kv-offload` | None | **The 120B Lifesaver.** Kicks KV cache out of VRAM into host DDR5 RAM so your 8GB GPU can fit giant models without exploding. | `0` (VRAM) |
| `-kvo`, `--kv-offload` | None | Keeps KV cache in GPU VRAM for maximum speed. | `enabled` |
| `-cram`, `--cache-ram` | `N` (MiB) | Host RAM buffer for persistent prompt prefix caching (e.g. `4096` = 4GB of instant repo memory for OpenCode). | `4096` |
| `--cache-idle-slots` | None | Saves idle slots to prompt cache so repeat agent runs start at 0 ms TTFT. | `enabled` |
| `--prompt-cache` | `FILE` | Dumps your KV state to a file on disk so you can resume sessions like a save game. | `none` |
| `--prompt-cache-ro` | None | Locks the prompt cache file to read-only so messy prompts don't taint your clean prefix. | `false` |
| `--sink-tokens`, `--attn-sinks` | `N` | **Attention Sinks.** Permanently anchors tokens 0..3 because LLMs literally have an existential panic attack if you delete their first words. | `4` |
| `--context-shift` | None | Rolls out middle tokens so you can generate text forever without crashing into the context ceiling. | `enabled` |

---

### 5. ⚙️ Compute, Layer Offload & Hardware Pinning

| Flag | Arg | What it actually does | Default |
| :--- | :---: | :--- | :---: |
| `-m`, `--model` | `PATH` | Path to your GGUF model file. The only argument that's actually mandatory. | Required |
| `-ngl`, `--n-gpu-layers` | `N` | How many layers to yeet into GPU VRAM (`99` = everything, `0` = CPU only). | Auto-calculated |
| `-t`, `--threads` | `N` | CPU threads during token generation (automatically pinned to Raptor Lake P-cores). | `6` |
| `-tb`, `--threads-batch` | `N` | CPU threads during prompt prefill batching. | `6` |
| `-c`, `--ctx-size` | `N` | Context window size in tokens (`0` = read max from GGUF, e.g. 32768 or 65536). | `0` |
| `-b`, `--batch-size` | `N` | Logical batch size for prompt processing. | `2048` |
| `-ub`, `--ubatch-size` | `N` | Physical micro-batch size for GPU execution tiles (keeps VRAM spikes from crashing Hyprland). | `512` |
| `-sm`, `--split-mode` | `none\|layer` | Multi-GPU tensor splitting mode. | `layer` |

---

### 6. 🌐 HTTP Server & Web UI Flags (`llama-server`)

| Flag | Arg | What it actually does | Default |
| :--- | :---: | :--- | :---: |
| `--host` | `IP` | Network IP to listen on (`0.0.0.0` for LAN access, `127.0.0.1` for local only). | `127.0.0.1` |
| `--port` | `PORT` | The port your coding agents connect to (default `8080`, OpenCode likes `8081`). | `8080` |
| `--path` | `DIR` | Where the static Web UI lives (the cute chat interface with context gauges). | `tools/server/public` |
| `--timeout` | `SEC` | Network timeout in seconds before dropping long-running connections. | `600` |
| `-np`, `--parallel` | `N` | Number of concurrent slots for continuous batching (lets multiple agents chat simultaneously). | `1` (or `4` in auto) |

---

### 7. 🎯 Sampling & Generation Flags (The Vibe Sliders)

| Flag | Arg | What it actually does | Default |
| :--- | :---: | :--- | :---: |
| `-temp`, `--temp` | `FLOAT` | `0.0` = cold-blooded math robot (zero hallucination). `0.8` = creative coder. `1.5` = schizoposting. | `0.8` |
| `--top-p` | `FLOAT` | Nucleus sampling cutoff. Drops low-probability trash words. | `0.9` |
| `--top-k` | `N` | Top-K cutoff. Limits candidate words to top N options (`40` is sweet spot). | `40` |
| `--min-p` | `FLOAT` | Minimum token probability relative to top token. Highly recommended at `0.05`. | `0.05` |
| `--repeat-penalty` | `FLOAT` | Slaps the model if it starts repeating itself like a broken record (`1.1` is mild). | `1.0` |
| `-n`, `--predict` | `N` | Max tokens to generate (`-1` = until EOS token, `-2` = fill the entire context). | `-1` |

---

### 8. 🛡️ Phase 8 Dense Wing Overhaul Flags (Techniques #25 - #28)

| Flag | Arg | What it actually does (No Cap) | Default |
| :--- | :---: | :--- | :---: |
| `--jit-down-quant` | None | **The DDR5 Diet.** Parallel SIMD pass across P-cores transcodes CPU layers to `Q2_K` on-the-fly during mmap load. Cuts DDR5 streaming traffic from 9GB down to 4.6GB! | Auto (`--sandwich-auto`) |
| `--no-jit-down-quant` | None | Disables the diet and lets your DDR5 RAM choke on 9GB of uncompressed tensors like an old office PC. | `false` |
| `--jit-down-quant-start-layer` | `N` | First layer index put on the treadmill (`-1` = first CPU-bound layer). | `-1` |
| `--cpu-cache-path` | `PATH` | **Black Magic Sidecar.** Dumps the transcoded CPU arena into a `.sandwich-cpu.bin` sidecar. Next reboot takes **10.4s instead of 1.5 minutes**. | `<model>.sandwich-cpu.bin` |
| `--thp-align` | None | **Begone, TLB Thrashing.** Aligns CPU tensors to 2MB boundaries with `madvise(MADV_HUGEPAGE)`. Deletes 12–18% memory latency. | Auto (`--sandwich-auto`) |
| `--no-thp-align` | None | Disables 2MB HugePages so your OS can enjoy doing 4KB page table walk slideshows. | `false` |
| `--early-exit-thresh` | `THRESH` | **"Why think when it's obvious?"** If top-1 confidence at the GPU boundary $\ge 92\%$ (punctuation, syntax, boilerplate), skips all CPU layers and yeets the token out of VRAM at **40+ t/s**! | `0.92` (`0.0` = off) |

---

### 9. 🧠 Phase 9 Predictive Hardware-Aware Topology & TurboQuant Auto-Router (Technique #29)

| Flag | Arg | What it actually does (No Cap) | Default |
| :--- | :---: | :--- | :---: |
| `--sandwich-predict` | None | **The Big Brain Autopilot.** Dynamically solves Asymmetric Split vs. Hybrid Offload, auto-routes TurboQuant (`turbo4` $\to$ `turbo3` $\to$ `turbo2`), and guarantees zero VRAM/RAM OOM crashes. | `true` (in auto) |
| `--no-sandwich-predict` | None | Turns off the autopilot so you can manually guess your layer splits and crash into your VRAM ceiling like it's 2023. | `false` |

---

### 10. 🏎️ Phase 10 & 11 Pipeline Overhaul & Hardware Push Flags (Techniques #30 - #35)

| Flag | Arg | What it actually does (No Cap) | Default |
| :--- | :---: | :--- | :---: |
| `--spec-pipe` | None | **Pipelined Double-Buffered Speculation.** Overlaps GPU drafting on stream 2 with CPU verification on OpenMP threads. Zero sequential idle bubbles! | Auto (`--sandwich-auto`) |
| `--no-spec-pipe` | None | Disables pipelined drafting, returning to lock-step sequential waiting. | `false` |
| `--spec-tree-adaptive` | None | **PID Controller for Spec Trees.** Dynamically tunes tree depth $D(1-7)$ and branch split $p(0.05-0.40)$ targeting 75% acceptance rate. | Auto (`--sandwich-auto`) |
| `--no-spec-tree-adaptive` | None | Disables PID adaptation and forces rigid static tree shapes. | `false` |
| `--split-kv`, `--kv-vram` | None | **Pure VRAM Split-KV.** Forces KV cache residency onto RTX 5050 GDDR6 even when target weights live 100% on CPU. DDR5 streams zero bytes of KV data! | Auto (`--sandwich-auto`) |
| `--vnni`, `--avx-vnni` | None | **Raptor Lake VNNI Turbo.** Replaces integer emulation with native `_mm256_dpbusd_epi32` instructions for `Q2_K` & `IQ2_XXS`. | Auto (`--sandwich-auto`) |
| `--preset agent` | None | **Agent Mode In One Click.** 64K context, `turbo4` KV cache, 4GB persistent prompt cache, and unified KV cache for high-concurrency tool calling. | `false` |
| `--fp4`, `--mxfp4` | None | **Blackwell NVFP4 Offload.** Offloads native 4-bit floating point layers to 5th-Gen Tensor Cores with microscopic block scaling. | `false` |

---

### 11. 🚀 Phase 12 External High-Impact Native Kernels (Techniques #36 - #39)

| Flag | Arg | What it actually does (No Cap) | Default |
| :--- | :---: | :--- | :---: |
| `-fanq`, `--flash-attn-native-quants` | None | **Fused Quant FlashAttention (Piggidragon).** Fuses dequantization directly inside MMA Tensor Cores. Saves up to 1.25 GB VRAM and enables full 2x Blackwell KQ half speed. | Auto (`--sandwich-auto`) |
| `--no-flash-attn-native-quants` | None | Disables fused quant FlashAttention kernels. | `false` |
| `--logits-topk` | `N` | **GPU Top-K Reduction.** Prunes logits to top-K on GPU before transferring over PCIe. Slashes PCIe transmission bandwidth by 99.9%! | `64` (in spec) |
| `--turbo-mma` | None | **Fused Turbo MMA FlashAttention (Thecodacus + TurboQuant).** Directly stages PolarQuant `turbo4/3/2` tiles into Tensor Core MMA registers. | Auto (`--sandwich-auto`) |
| `--sched-async-cpu` | None | **Async CPU Split Worker.** Runs CPU graph splits on a dedicated worker thread pool to eliminate host compute bubbles. | Auto (`--sandwich-auto`) |

---

### 12. ⚡ Phase 12 3-Tier Hierarchical MoE Memory System Engine (Technique #40)

| Flag | Arg | What it actually does (No Cap) | Default |
| :--- | :---: | :--- | :---: |
| `--moe-3tier` | None | **3-Tier Hierarchical MoE Engine.** High-throughput 3-tier memory engine: **Tier 1 (HOT VRAM @ 288 GB/s)** $\to$ **Tier 2 (WARM DDR5 @ 50 GB/s)** $\to$ **Tier 3 (COLD NVMe @ 7 GB/s)**. Unlocks massive 120B+ MoE models on 8GB laptops! | Auto (`--sandwich-auto`) |
| `--no-moe-3tier` | None | Disables 3-tier memory engine and falls back to standard 2-tier LRU staging. | `false` |
| `--moe-warm-ram-mb` | `N` | Dedicated Tier 2 WARM DDR5 pool size in MiB backed by 2MB Transparent Huge Pages (THP) and Linux asynchronous `io_uring` DMA. | `12288` (12 GB) |

</details>

---

## 👑 Shouts Out to the Real Ones

Sandwich.cpp stands on the shoulders of absolute open-source gigachads:
* **`ggerganov`:** The father of `llama.cpp`. Without him we'd all still be paying cloud providers $500/month.
* **`Piggidragon`:** The CUDA wizard behind Quantized-Native MMA FlashAttention and zero-allocation shared-memory staging.
* **`thecodacus`:** The MoE routing wizard who realized transferring cold experts over PCIe is a scam.
* **`TheTom`:** The TurboQuant pioneer who made 4-bit spherical KV cache physics actually work.
* **`ikawrakow` (`ik_llama`):** Hand-written assembly master. Legend.
* **`Justine Tunney` (`llamafile`):** Making binaries portable across the multiverse.

---

## 📄 License

MIT License. Do whatever you want with it, just don't tell your mom how much time you spent optimizing CUDA kernels. 🥪✨
