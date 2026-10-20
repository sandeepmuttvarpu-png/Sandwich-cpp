# 🥪 Sandwich.cpp

> ### *"Bro really thought an 8GB laptop GPU couldn't run 120B models... skill issue fr fr 💀🔥"*
> **The unhinged, ultra-fast C++/CUDA inference engine that lets budget gaming laptops mog enterprise datacenters.**

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![CUDA: 12.0+ / 13.4](https://img.shields.io/badge/CUDA-12.0%2B%20%7C%2013.4-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![VRAM: 8GB Friendly](https://img.shields.io/badge/VRAM-8GB%20Friendly%20%F0%9F%92%AA-brightgreen)](#)
[![Speed: Unhinged](https://img.shields.io/badge/Speed-35%2B%20tok%2Fs%20MoE-red)](#)
[![No Cap: Verified](https://img.shields.io/badge/Cap%3A-Zero%20%F0%9F%A7%A2-blue)](#)
[![Companion: Sandwich Deck](https://img.shields.io/badge/GUI-Sandwich%20Deck%20(Tauri%202)-purple.svg)](https://github.com/sandeepmuttvarpu-png/sandwich-deck)

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
        Your 8GB Laptop GPU
       ┌────────────────────────┐
       │   HOT EXPERT VRAM SLAB │ <── Keeps top 20% active experts pinned in VRAM!
       │  (Thecodacus O(1) LRU) │     (Zero PCIe thrashing, 100% pure speed)
       ├────────────────────────┤
       │  EAGLE3 TREE DECODING  │ <── Verifies 16 candidate tokens in ONE forward pass
       │  (2D Causal Attention) │     (+65% speedup on dense models!)
       ├────────────────────────┤
       │   TURBOQUANT 4-BIT KV  │ <── Spherical Walsh-Hadamard polar math
       │ (75% VRAM Compression) │     (64K context runs at the SAME speed as 4K!)
       └────────────────────────┘
                  ▲
                  │ Pinned Zero-Copy DMA (20 GB/s across PCIe 4.0!)
                  ▼
       ┌────────────────────────┐
       │ INTEL PHYSICAL P-CORES │ <── Thread pinning to physical P-cores 0-5!
       │   (Bypasses E-Cores)   │     (Background worker evaluates cold experts async)
       └────────────────────────┘
```

### 1. The MoE VRAM Slab Hack (Thecodacus Dual-Path)
MoE models are posers. They claim they have 35 billion parameters, but only **3B to 4B parameters actually fire per token**.
* Sandwich.cpp keeps the hottest experts **permanently locked inside an ultra-fast GPU VRAM slab**.
* When a rare "cold" expert is needed, a persistent background CPU worker thread computes it asynchronously while the GPU keeps blasting (`sched_async_cpu`). **2.04x faster than LM Studio instantly.**

### 2. TurboQuant 4-Bit KV Cache (Math so crazy it sounds fake)
Ever tried running 64K context on an 8GB GPU? Usually, your VRAM fills up with KV cache tensors and your PC has an emotional breakdown.
* TurboQuant rotates your vectors using an orthonormal **Walsh-Hadamard Transform (WHT)**, turning outliers into a neat Gaussian curve, then squashes them into 4-bit, 3-bit, or 2-bit polar coordinates (`turbo4`, `turbo3`, `turbo2`).
* **Result:** **75% less KV memory.** On Dolphin 7B, 4K context runs at **57.8 t/s** and 64K context runs at **57.7 t/s**. Literally zero context degradation. Flatline speed. 📈

### 3. EAGLE3 Speculative Tree Decoding (`--spec-tree`)
Linear drafting is for boomers. Sandwich.cpp constructs **a branching tree of up to 16 future tokens** using 2D causal masks.
* The big model evaluates the whole tree in **ONE single forward pass**.
* On dense models like Qwen 27B, this jumps generation speed by **+65% (1.88 accepted tokens per pass)**.

### 4. P-Core VIP Club (No E-Cores Allowed)
Intel 13th-Gen CPUs have Efficiency Cores (E-cores) that are great for running Discord in the background, but absolute garbage at matrix math.
* Sandwich.cpp uses `pthread_setaffinity_np` to lock all compute threads **strictly to physical Raptor Lake P-cores (Cores 0–5)**.
* Plus, a hardcoded **900 MB VRAM headroom guard** means your Hyprland/Wayland desktop will *never* crash while generating.

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
| **`GPT-OSS-120B`** 🦖 | **120B MoE** | **A 120-BILLION MODEL RUNNING ON AN 8GB LAPTOP** | **16.9 t/s** | **1.5 t/s** | **1,630 MB** |
| **`Muse-30B`** 🎨 | 30B Dense | Hybrid offload with DFlash2 | **15.1 t/s** | **5.2 t/s** | **5,932 MB** |
| **`Gemma-31B`** 💎 | 31B Dense | Hybrid offload with MTP assistant | **14.2 t/s** | **4.8 t/s** | **6,892 MB** |
| **`Qwen-27B`** 🌲 | 27B Dense | EAGLE3 Spec-Tree verification | **18.5 t/s** | **3.35 t/s** | **7,186 MB** |
| **`Llama-70B`** 🦙 | 70B Dense | Dense hybrid with Llama-3.2 drafter | **18.9 t/s** | **1.25 t/s** | **7,561 MB** |

> 💡 **Notice that 120B row?** Yes, you read that right. A 120-billion parameter model passed 33 out of 45 test runs on an 8GB mobile GPU with zero kernel panics. Go tell your Discord friends.

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
<summary><b>🔧 Advanced CLI Flags & Engine Parameter Reference</b></summary>

| Flag | Default | What it does |
| :--- | :--- | :--- |
| `--sandwich-auto` | Off | Full autonomous hardware detection, 900MB VRAM guard, P-core affinity, auto layer splits. |
| `--moe-cache-slots N` | 32 (auto) | Number of hot expert slabs pinned in GPU VRAM per layer. |
| `--moe-cache-profile <file>` | None | Offline routing CSV generated by `llama-moe-trace` for optimal expert packing. |
| `--moe-hot-only` | Off | Bypasses CPU cold experts completely for 90+ t/s pure-VRAM speculative drafting. |
| `--spec-tree` | Auto (Dense) | Enables EAGLE3 multi-branch speculative tree decoding with 2D causal masks. |
| `--spec-tree-depth N` | 3 | Tree search depth for candidate tokens. |
| `--spec-tree-width N` | 2 | Tree branching width at entropy-split nodes. |
| `-ctk <type> / -ctv <type>` | `turbo4` | TurboQuant KV compression format (`turbo4`, `turbo3`, `turbo2`). |
| `-nkvo` | Off | Offload KV cache to host DDR5 RAM, freeing GPU VRAM for massive context windows. |
| `--sink-tokens N` | 4 | Enforces attention sink preservation (tokens 0..3) during sliding-window context compaction. |

</details>

---

## 👑 Shouts Out to the Real Ones

Sandwich.cpp stands on the shoulders of absolute open-source gigachads:
* **`ggerganov`:** The father of `llama.cpp`. Without him we'd all still be paying cloud providers $500/month.
* **`thecodacus`:** The MoE routing wizard who realized transferring cold experts over PCIe is a scam.
* **`TheTom`:** The TurboQuant pioneer who made 4-bit spherical KV cache physics actually work.
* **`ikawrakow` (`ik_llama`):** Hand-written assembly master. Legend.
* **`Justine Tunney` (`llamafile`):** Making binaries portable across the multiverse.
* **[Sandwich Deck](https://github.com/sandeepmuttvarpu-png/sandwich-deck):** The companion Tauri 2 native desktop app for people who hate Electron with a passion.

---

## 📄 License

MIT License. Do whatever you want with it, just don't tell your mom how much time you spent optimizing CUDA kernels. 🥪✨
