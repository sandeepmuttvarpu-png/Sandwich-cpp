# 🥪 Sandwich.cpp

> **Unified High-Performance Local LLM Engine for Consumer GPUs**  
> *Dynamic MoE Expert Caching + TurboQuant 4-Bit KV Compression + Architecture-Aware Auto-Dispatch*

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![CUDA: 12.0+](https://img.shields.io/badge/CUDA-12.0%2B-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Arch: x86__64](https://img.shields.io/badge/Arch-x86__64-orange.svg)](#)
[![Sandwich Deck: Native](https://img.shields.io/badge/Companion-Sandwich%20Deck-blueviolet.svg)](https://github.com/sandeepmuttvarpu-png/sandwich-deck)

**Sandwich.cpp** is a high-performance fork of [`llama.cpp`](https://github.com/ggerganov/llama.cpp) engineered to solve the memory and bandwidth bottleneck of running **30B–35B Mixture of Experts (MoE)** and large dense models on consumer hardware (such as 8GB VRAM laptop and desktop GPUs).

By fusing **BeeLlama's dynamic MoE slab offloading** with **TheTom's TurboQuant 4-bit compressed KV cache**, patching critical C++ buffer assertion crashes on GLM architectures, and introducing an **Architecture-Aware Auto-Dispatcher (`--sandwich-auto`)**, Sandwich.cpp achieves up to **+78.3% faster MoE inference** with zero manual configuration.

---

## 🏆 Key Performance Highlights (8GB RTX 5050 Laptop GPU)

| Model Architecture | Context | BeeLlama | TurboQuant | Sandwich.cpp (Ours) | Delta vs Prior Forks |
|---|---|---|---|---|:---:|
| **Nemotron-3-Nano-30B MoE** | 32K | 23.6 tk/s (8K) | 19.5 tk/s | **34.8 tk/s** 🥇 | **+47.5% vs BeeLlama**<br>**+78.3% vs TurboQuant** |
| **Qwen3.6-35B MoE** | 8K / 32K | 25.1 tk/s | 15.7 tk/s | **33.4 tk/s** 🥇 | **+33.1% vs BeeLlama**<br>**+116.0% vs TurboQuant** |
| **GLM-4.7 Flash 30B MoE** | 8K / 32K | ❌ **CRASH** | 16.2 tk/s | **27.6 tk/s** (61.8 PP) 🥇 | **+70.4% vs TurboQuant**<br>*(BeeLlama crashed outright)* |
| **Muse-Glimmer 30B Dense** | 8K | 2.5 tk/s | 2.6 tk/s | **2.56 tk/s** (20.3 PP) 🥇 | Auto CPU WHT fallback |
| **Dolphin 2.8 Mistral 7B** | 8K | — | — | **55.82 tk/s** (1,978 PP) 🥇 | 100% GPU offload |
| **NVIDIA Nemotron 4B** | 8K | — | — | **66.67 tk/s** (3,050 PP) 🥇 | 100% GPU offload |

*See full benchmark details and methodology in [`docs/benchmarks.md`](docs/benchmarks.md).*

---

## 🌟 The Killer Pros of Sandwich.cpp

1. **Consumer VRAM Multiplier (Run 30B–35B MoE on 8GB GPUs):**
   - Enables models that previously required 24GB–48GB VRAM cards to run on 8GB consumer laptops and desktops at fluid, interactive generation speeds (**27–35 tokens/second**).
2. **TurboQuant 4-Bit KV Cache Compression (`turbo4`):**
   - Reduces KV cache memory footprint by **75%** with negligible loss in reasoning accuracy, unlocking 32K–64K context windows without exhausting VRAM.
3. **True Dynamic MoE Slab Caching:**
   - Active expert pathways are dynamically staged into GPU VRAM slabs instead of static layer-by-layer offloading, eliminating PCIe host paging stalls.
4. **Zero-Configuration Simplicity (`--sandwich-auto`):**
   - Automatically detects MoE vs. Dense topologies from GGUF metadata and sets optimal GPU layer offloads, cache slots, continuous batching, and quantization schemes.
5. **Fixed GLM & Modern MoE Architectures:**
   - Out-of-the-box stability for GLM-4.7 Flash and Multi-Head Latent Attention (MLA) models that crash on other forks.
6. **Native Speculative Decoding (`-md`):**
   - Accelerates generation throughput by up to 1.5x via speculative draft model verification with zero loss in generation fidelity.
7. **Drop-in OpenAI Server HUD:**
   - Full compatibility with the OpenAI REST specification (`/v1/chat/completions`, `/v1/models`, `/health`), compatible with any UI, client library, or autonomous agent framework.

---

## 🐛 The C++ Bug: Why GLM Crashed in BeeLlama (And How We Fixed It)

When running **GLM-4.7 Flash** (or DeepSeek-style MoE models with Multi-Head Latent Attention) under BeeLlama (`GenerelSchwerz/llama.cpp`), the engine crashed immediately with an abort assertion during model initialization.

### Bug 1: The Null Backend Buffer Assertion Crash

#### Root Cause:
In BeeLlama, `ggml_backend_buffer_get_usage()` in `ggml/src/ggml-backend.cpp` contained an unconditional assertion:

```cpp
// ❌ FAILING CODE IN BEELLAMA & UPSTREAM:
enum ggml_backend_buffer_usage ggml_backend_buffer_get_usage(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);  // <--- CRASHES when buffer == nullptr!
    return buffer->usage;
}
```

During dynamic MoE allocation and expert slab staging on GLM models, split virtual buffers and unallocated temporary tensors are queried to determine usage policy. Passing a `nullptr` buffer triggered an instant `GGML_ASSERT(buffer)` abort.

#### Sandwich.cpp Fix:
We replaced the brittle assertion with a null-safe fallback:

```cpp
// ✅ FIXED IN SANDWICH.CPP:
enum ggml_backend_buffer_usage ggml_backend_buffer_get_usage(ggml_backend_buffer_t buffer) {
    if (buffer == nullptr) {
        return GGML_BACKEND_BUFFER_USAGE_ANY;
    }
    GGML_ASSERT(buffer);
    return buffer->usage;
}
```

---

### Bug 2: Missing Multi-Head Latent Attention (MLA) Kernels (`dkq=640, dv=512`)

#### Root Cause:
GLM-4.7 Flash uses an atypical attention head dimension (`dkq = 640`, `dv = 512`) for its Multi-Head Latent Attention layer. Neither BeeLlama nor upstream llama.cpp had CUDA Flash Attention MMA (Matrix Multiply-Accumulate) configurations or tile kernels compiled for `dkq = 640`. As a result, Flash Attention either failed silently, fell back to unoptimized paths, or crashed on tensor shape checks.

#### Sandwich.cpp Fix:
1. **Added MMA Configuration Cases** (`ggml/src/ggml-cuda/fattn-mma-f16.cuh`):
   ```cpp
   GGML_CUDA_FATTN_MMA_CONFIG_CASE(640, 512,  8,  64, 4,  32, 288, 256, 128, 1, false);
   GGML_CUDA_FATTN_MMA_CONFIG_CASE(640, 512, 16,  64, 4,  32, 288, 256, 128, 1, false);
   GGML_CUDA_FATTN_MMA_CONFIG_CASE(640, 512, 32, 128, 2,  32, 160, 128, 128, 1, false);
   GGML_CUDA_FATTN_MMA_CONFIG_CASE(640, 512, 64, 256, 1,  32, 160, 128, 128, 1, false);
   ```
2. **Added Flash Attention Tile Template Instance** (`ggml/src/ggml-cuda/fattn-tile-instance-dkq640-dv512.cu`):
   Compiled dedicated CUDA kernels for `dkq=640` and `dv=512`.

---

### Bug 3: CPU Walsh-Hadamard Transform (WHT) Fallback

#### Root Cause:
The original TurboQuant implementation lacked CPU fallback routines for its 4-bit Walsh-Hadamard transform quantization (`turbo4`). If a large dense model (e.g. 30B+) required hybrid offloading (`-ngl 18` on GPU, rest on CPU), the CPU worker threads crashed on unhandled TurboQuant tensor operations.

#### Sandwich.cpp Fix:
Ported C++ reference Walsh-Hadamard transform operations into `ggml/src/ggml-cpu/ops.cpp`, enabling dense models with TurboQuant KV caches to execute seamlessly across partial GPU + CPU host memory with zero crashes.

---

## ⚠️ What Still Needs to Be Fixed (Known Limitations & Roadmap)

While Sandwich.cpp is production-stable on single-GPU systems, the following areas are under active development:

1. **Multi-GPU Dynamic MoE Slicing:**
   - The dynamic MoE expert cache currently optimizes for a single primary GPU. Multi-GPU setups (`-ts` / `--tensor-split`) with dynamic expert slabs require deeper synchronization to avoid bus stalls across PCIe/NVLink.
2. **Cross-Tokenizer Speculative Drafting:**
   - Speculative decoding (`-md`) strictly requires identical vocabulary representations (e.g., Llama-3 128k target + Llama-3 128k drafter). Pairing cross-family models (e.g. Mistral 32k target + Llama 128k drafter) triggers token ID mismatches. A dynamic vocabulary projection/remapping layer is planned.
3. **AMD ROCm (HIP) & Apple Silicon (Metal) TurboQuant Kernels:**
   - TurboQuant Walsh-Hadamard kernels are currently accelerated natively in CUDA and reference C++ on CPU. Native Metal Shading Language (MSL) and ROCm/HIP kernel implementations are in progress.
4. **Runtime Context KV Downscaling:**
   - Under extremely long context sessions (64K+), automatic dynamic downscaling from `turbo4` to `turbo2` without restarting the session or resetting the KV cache slots.
5. **Vision-Language MoE Projector Offloading:**
   - Multimodal projector tensors (`mmproj`) for vision-MoE models currently bypass the dynamic expert cache and require static allocation.

---

## 🥪 Architecture-Aware Auto-Dispatcher (`--sandwich-auto`)

The `--sandwich-auto` flag inspects model topology at load time and configures hardware-optimal parameters automatically:

```bash
./build/bin/llama-server -m /path/to/model.gguf --sandwich-auto
```

### Auto Policy Behavior:
- **MoE Topology (`expert_count > 0`):**
  - Sets `-ngl 99` (100% layer offload into dynamic slabs).
  - Sets `--moe-expert-cache-size 32` (allocates 32 active expert cache slots in VRAM).
  - Sets `--cache-ram 0` (eliminates slow PCIe host RAM paging).
  - Activates `turbo4` KV compression (`-ctk turbo4 -ctv turbo4`) and Flash Attention.
- **Dense Topology (`expert_count == 0`):**
  - Automatically calculates GPU headroom (defaults to `-ngl 18` on 8GB cards).
  - Enables continuous batching and parallel decoding (`-np 4`).
  - Activates `turbo4` with CPU Walsh-Hadamard fallback for remaining layers.

---

## 🛠️ How to Use It

### 1. Interactive Terminal CLI
```bash
# Launch interactive chat with automatic hardware optimization
./build/bin/llama-cli \
  -m ~/ai-models/moe/Nemotron-3-Nano-30B-A3B-GGUF/Nemotron-3-Nano-30B-A3B-UD-IQ3_XXS.gguf \
  --sandwich-auto \
  -c 32768
```

### 2. OpenAI-Compatible API Server (`llama-server`)
```bash
# Run server on port 8081
./build/bin/llama-server \
  -m ~/ai-models/moe/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf \
  --sandwich-auto \
  --host 0.0.0.0 --port 8081 \
  -c 32768
```

### 3. Speculative Decoding with Drafter Model
Accelerate generation up to 1.5x by pairing a target model with an aligned drafter:
```bash
./build/bin/llama-server \
  -m ~/ai-models/dense/Llama-3.3-70B-Instruct-GGUF/Llama-3.3-70B-Instruct-UD-IQ2_XXS.gguf \
  -md ~/ai-models/drafters/Llama-3.2-1B-Instruct-GGUF/Llama-3.2-1B-Instruct-UD-IQ2_XXS.gguf \
  -ngl 16 -c 8192 --port 8081
```

### 4. Querying the Server (curl / Python)

```bash
curl http://127.0.0.1:8081/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sandwich",
    "messages": [{"role": "user", "content": "Explain quantum computing in 2 sentences."}],
    "temperature": 0.7
  }'
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8081/v1", api_key="sandwich")
response = client.chat.completions.create(
    model="sandwich",
    messages=[{"role": "user", "content": "Write a fast matrix multiplication function in C++."}],
    stream=True,
)

for chunk in response:
    content = chunk.choices[0].delta.content or ""
    print(content, end="", flush=True)
```

---

## 🚀 How to Deploy It

### Method 1: Systemd Service (Linux Server / Workstation)

Create a persistent system service to run Sandwich.cpp automatically on boot:

1. Create `/etc/systemd/system/sandwich.service`:
   ```ini
   [Unit]
   Description=Sandwich.cpp High-Performance Local LLM Server
   After=network.target

   [Service]
   Type=simple
   User=sandeep
   WorkingDirectory=/home/sandeep/dev/sandwich/sandwich-cpp
   ExecStart=/home/sandeep/dev/sandwich/sandwich-cpp/build/bin/llama-server \
       -m /home/sandeep/ai-models/moe/Nemotron-3-Nano-30B-A3B-GGUF/Nemotron-3-Nano-30B-A3B-UD-IQ3_XXS.gguf \
       --sandwich-auto \
       --host 0.0.0.0 --port 8081 \
       -c 32768
   Restart=always
   RestartSec=3
   LimitNOFILE=65535

   [Install]
   WantedBy=multi-user.target
   ```

2. Enable and start the service:
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable --now sandwich
   sudo systemctl status sandwich
   ```

---

### Method 2: Docker Deployment with NVIDIA GPU Acceleration

1. Build the Docker container:
   ```dockerfile
   # Dockerfile.sandwich
   FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder

   RUN apt-get update && apt-get install -y git cmake build-essential
   WORKDIR /app
   COPY . .
   RUN mkdir build && cd build && \
       cmake .. -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release && \
       cmake --build . --config Release -j$(nproc)

   FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04
   RUN apt-get update && apt-get install -y libgomp1 curl && rm -rf /var/lib/apt/lists/*
   WORKDIR /app
   COPY --from=builder /app/build/bin/llama-server /usr/local/bin/llama-server

   EXPOSE 8081
   ENTRYPOINT ["llama-server"]
   CMD ["--help"]
   ```

2. Run with GPU support:
   ```bash
   docker build -t sandwich-cpp:latest -f Dockerfile.sandwich .

   docker run -d --gpus all \
     -p 8081:8081 \
     -v /home/sandeep/ai-models:/models \
     --name sandwich-server \
     sandwich-cpp:latest \
     -m /models/moe/Nemotron-3-Nano-30B-A3B-GGUF/Nemotron-3-Nano-30B-A3B-UD-IQ3_XXS.gguf \
     --sandwich-auto --host 0.0.0.0 --port 8081 -c 32768
   ```

---

### Method 3: Reverse Proxy with Nginx (Streaming SSE Optimized)

Ensure `proxy_buffering off;` is set so tokens stream smoothly with zero latency:

```nginx
server {
    listen 80;
    server_name llm.internal;

    location / {
        proxy_pass http://127.0.0.1:8081;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_set_header Host $host;

        # Disable buffering for realtime Server-Sent Events (SSE) token typewriter
        proxy_buffering off;
        proxy_cache off;
        chunked_transfer_encoding on;

        proxy_read_timeout 600s;
        proxy_send_timeout 600s;
    }
}
```

---

## 🔨 Building From Source

```bash
git clone https://github.com/sandeepmuttvarpu-png/Sandwich-cpp.git
cd Sandwich-cpp

mkdir build
cd build
cmake .. -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release -j$(nproc)
```

Binaries will be placed in `build/bin/`:
- `llama-server`: High-performance OpenAI-compatible HTTP server.
- `llama-cli`: Terminal interactive engine.
- `llama-bench`: Automated benchmark runner.

---

## 🤝 Credits & Upstream Lineage

Sandwich.cpp is built upon pioneering work across the local AI open-source community:

- **Upstream Engine:** [`ggerganov/llama.cpp`](https://github.com/ggerganov/llama.cpp)
- **Dynamic MoE Offload:** [`GenerelSchwerz/llama.cpp`](https://github.com/GenerelSchwerz/llama.cpp) (BeeLlama)
- **TurboQuant KV Cache:** [`TheTom/llama-cpp-turboquant`](https://github.com/TheTom/llama-cpp-turboquant)
- **GUI Companion:** Pair Sandwich.cpp with **[Sandwich Deck](https://github.com/sandeepmuttvarpu-png/sandwich-deck)** for an ultra-lightweight LM Studio desktop experience.

---

## 📄 License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for details.
