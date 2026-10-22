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

| Flag | Arg | What it actually does | Default |
| :--- | :---: | :--- | :---: |
| `--jit-down-quant` | None | Transcodes CPU layers to `Q2_K` in-memory during mmap loading (cuts DDR5 load ~50%). | Auto (`--sandwich-auto`) |
| `--no-jit-down-quant` | None | Disables in-memory JIT down-quantization of CPU layers. | `false` |
| `--jit-down-quant-start-layer` | `N` | First layer index to down-quantize in host CPU memory (`-1` = first CPU layer). | `-1` |
| `--cpu-cache-path` | `PATH` | Explicit sidecar cache path for transcoded CPU layers (`<model>.sandwich-cpu.bin`). | `<model>.sandwich-cpu.bin` |
| `--thp-align` | None | Aligns CPU tensor allocations to 2MB boundaries & issues `madvise(MADV_HUGEPAGE)`. | Auto (`--sandwich-auto`) |
| `--no-thp-align` | None | Disables 2MB Transparent HugePages alignment for host memory. | `false` |
| `--early-exit-thresh` | `THRESH` | Softmax probability margin threshold to emit high-confidence tokens directly from VRAM. | `0.92` (`0.0` = off) |

---

### 9. 🧠 Phase 9 Predictive Hardware-Aware Topology & TurboQuant Auto-Router (Technique #29)

| Flag | Arg | What it actually does | Default |
| :--- | :---: | :--- | :---: |
| `--sandwich-predict` | None | Automatically predicts Asymmetric Split vs. Hybrid Offload and auto-routes TurboQuant KV. | `true` (in auto) |
| `--no-sandwich-predict` | None | Disables predictive solver and TurboQuant auto-routing (forces static layer fitting). | `false` |

</details>

---

## 👑 Shouts Out to the Real Ones

Sandwich.cpp stands on the shoulders of absolute open-source gigachads:
* **`ggerganov`:** The father of `llama.cpp`. Without him we'd all still be paying cloud providers $500/month.
* **`thecodacus`:** The MoE routing wizard who realized transferring cold experts over PCIe is a scam.
* **`TheTom`:** The TurboQuant pioneer who made 4-bit spherical KV cache physics actually work.
* **`ikawrakow` (`ik_llama`):** Hand-written assembly master. Legend.
* **`Justine Tunney` (`llamafile`):** Making binaries portable across the multiverse.

---

## 📄 License

MIT License. Do whatever you want with it, just don't tell your mom how much time you spent optimizing CUDA kernels. 🥪✨
