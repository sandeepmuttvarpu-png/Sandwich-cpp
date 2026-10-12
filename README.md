# 🥪 Sandwich.cpp

> **Unified High-Performance Local LLM Engine for Consumer GPUs**  
> *Dynamic MoE Expert Caching + TurboQuant 4-Bit KV Compression + Architecture-Aware Auto-Dispatch*

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![CUDA: 12.0+](https://img.shields.io/badge/CUDA-12.0%2B-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Arch: x86__64](https://img.shields.io/badge/Arch-x86__64-orange.svg)](#)

**Sandwich.cpp** is a specialized fork of [`llama.cpp`](https://github.com/ggerganov/llama.cpp) engineered to solve the memory and bandwidth bottleneck of running **30B–35B Mixture of Experts (MoE)** and large dense models on consumer hardware (such as 8GB VRAM cards).

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

## 🥪 Architecture-Aware Auto-Dispatcher (`--sandwich-auto`)

No more guessing layer budgets or tweaking 6+ command-line flags. The `--sandwich-auto` flag inspects model topology at load time and configures hardware-optimal parameters automatically:

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

## 🚀 Building From Source

### Prerequisites
- **CMake** 3.14+
- **GCC / Clang** with C++17 support
- **CUDA Toolkit** 12.0+ (for NVIDIA GPU acceleration)

### Compile with CUDA Support

```bash
git clone https://github.com/sandeepmuttvarpu-png/sandwich-cpp.git
cd sandwich-cpp

mkdir build
cd build
cmake .. -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release -j$(nproc)
```

The resulting binaries will be in `build/bin/`:
- `llama-server`: High-performance OpenAI-compatible HTTP server.
- `llama-cli`: Terminal interactive completion engine.
- `llama-bench`: Automated benchmarking suite.

---

## 📖 Usage Examples

### 1. Launching MoE Models (Nemotron 30B / Qwen 35B / GLM 30B)
```bash
./build/bin/llama-server \
  -m ~/ai-models/moe/Nemotron-3-Nano-30B-A3B-GGUF/Nemotron-3-Nano-30B-A3B-UD-IQ3_XXS.gguf \
  --sandwich-auto \
  -c 32768 --port 8081
```

### 2. Speculative Decoding with Drafter Model (`-md`)
Accelerate generation up to 1.5x by pairing a large target model with a compatible draft model:
```bash
./build/bin/llama-server \
  -m ~/ai-models/dense/Llama-3.3-70B-Instruct-GGUF/Llama-3.3-70B-Instruct-UD-IQ2_XXS.gguf \
  -md ~/ai-models/drafters/Llama-3.2-1B-Instruct-GGUF/Llama-3.2-1B-Instruct-UD-IQ2_XXS.gguf \
  -ngl 16 -c 8192 --port 8081
```

### 3. Granular TurboQuant KV Cache Selection
```bash
./build/bin/llama-cli \
  -m ~/ai-models/dense/dolphin-2.8-mistral-7b-v02-GGUF/dolphin-2.8-mistral-7b-v02-Q4_K_S.gguf \
  -ngl 99 -c 32768 \
  -ctk turbo4 -ctv turbo4 -fa
```

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
