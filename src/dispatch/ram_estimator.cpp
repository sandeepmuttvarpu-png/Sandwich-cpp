#include "ram_estimator.hpp"

#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <cstdlib>

#if defined(__has_include)
    #if __has_include(<cuda_runtime.h>)
        #define HAS_CUDA_RUNTIME 1
        #include <cuda_runtime.h>
    #endif
#endif

namespace sandwich {
namespace dispatch {

RealtimeRAMStats RAMEstimator::sample_realtime_ram() {
    RealtimeRAMStats stats;

    std::ifstream meminfo("/proc/meminfo");
    if (!meminfo.is_open()) {
        // Fallback if /proc/meminfo cannot be opened
        stats.total_mb = 16384;
        stats.available_mb = 8192;
        stats.free_mb = 4096;
        stats.used_percentage = 50.0;
        return stats;
    }

    std::string line;
    uint64_t mem_total_kb = 0;
    uint64_t mem_free_kb = 0;
    uint64_t mem_available_kb = 0;
    uint64_t buffers_kb = 0;
    uint64_t cached_kb = 0;
    uint64_t swap_total_kb = 0;
    uint64_t swap_free_kb = 0;

    while (std::getline(meminfo, line)) {
        std::istringstream iss(line);
        std::string key;
        uint64_t value = 0;
        std::string unit;

        if (iss >> key >> value) {
            if (key == "MemTotal:")       mem_total_kb = value;
            else if (key == "MemFree:")      mem_free_kb = value;
            else if (key == "MemAvailable:") mem_available_kb = value;
            else if (key == "Buffers:")      buffers_kb = value;
            else if (key == "Cached:")       cached_kb = value;
            else if (key == "SwapTotal:")    swap_total_kb = value;
            else if (key == "SwapFree:")     swap_free_kb = value;
        }
    }

    stats.total_mb      = mem_total_kb / 1024;
    stats.free_mb       = mem_free_kb / 1024;
    stats.available_mb  = (mem_available_kb > 0) ? (mem_available_kb / 1024) : ((mem_free_kb + cached_kb) / 1024);
    stats.cached_mb     = cached_kb / 1024;
    stats.buffers_mb    = buffers_kb / 1024;
    stats.swap_total_mb = swap_total_kb / 1024;
    stats.swap_free_mb  = swap_free_kb / 1024;

    if (stats.total_mb > 0) {
        uint64_t used_mb = (stats.total_mb > stats.available_mb) ? (stats.total_mb - stats.available_mb) : 0;
        stats.used_percentage = (static_cast<double>(used_mb) / stats.total_mb) * 100.0;
    }

    return stats;
}

RealtimeVRAMStats RAMEstimator::sample_realtime_vram() {
    RealtimeVRAMStats stats;

#if defined(HAS_CUDA_RUNTIME)
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    cudaError_t err = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (err == cudaSuccess && total_bytes > 0) {
        stats.cuda_available = true;
        stats.total_mb = total_bytes / (1024 * 1024);
        stats.free_mb = free_bytes / (1024 * 1024);
        stats.used_mb = (stats.total_mb > stats.free_mb) ? (stats.total_mb - stats.free_mb) : 0;
        stats.used_percentage = (static_cast<double>(stats.used_mb) / stats.total_mb) * 100.0;

        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) {
            stats.device_name = prop.name;
        } else {
            stats.device_name = "NVIDIA CUDA Device";
        }
        return stats;
    }
#endif

    // Fallback: nvidia-smi pipe query
    FILE * pipe = popen("nvidia-smi --query-gpu=name,memory.total,memory.free,memory.used --format=csv,noheader,nounits 2>/dev/null", "r");
    if (pipe) {
        char buf[256];
        if (fgets(buf, sizeof(buf), pipe)) {
            std::string line(buf);
            std::vector<std::string> tokens;
            std::stringstream ss(line);
            std::string item;
            while (std::getline(ss, item, ',')) {
                while (!item.empty() && (item.front() == ' ' || item.front() == '\t')) item.erase(0, 1);
                while (!item.empty() && (item.back() == ' ' || item.back() == '\t' || item.back() == '\n' || item.back() == '\r')) item.pop_back();
                tokens.push_back(item);
            }

            if (tokens.size() >= 3) {
                stats.cuda_available = true;
                stats.device_name = tokens[0];
                stats.total_mb = std::strtoull(tokens[1].c_str(), nullptr, 10);
                stats.free_mb  = std::strtoull(tokens[2].c_str(), nullptr, 10);
                stats.used_mb  = (tokens.size() >= 4) ? std::strtoull(tokens[3].c_str(), nullptr, 10) : (stats.total_mb - stats.free_mb);
                if (stats.total_mb > 0) {
                    stats.used_percentage = (static_cast<double>(stats.used_mb) / stats.total_mb) * 100.0;
                }
            }
        }
        pclose(pipe);
    }

    if (!stats.cuda_available) {
        stats.device_name = "CPU Direct Compute (No CUDA)";
        stats.total_mb = 0;
        stats.free_mb = 0;
        stats.used_mb = 0;
        stats.used_percentage = 0.0;
    }

    return stats;
}

ModelMemoryFootprint RAMEstimator::calculate_footprint(
    uint64_t model_bytes,
    int total_layers,
    int context_length,
    const std::string & kv_quant,
    bool is_moe,
    int expert_count,
    int active_experts,
    int n_embd,
    int n_head_kv,
    int head_dim
) {
    ModelMemoryFootprint fp;
    fp.total_weights_mb = model_bytes / (1024 * 1024);
    int safe_layers = (total_layers > 0) ? total_layers : 32;

    if (is_moe && expert_count > 0 && active_experts > 0) {
        // MoE models: ~20% dense shared projections, ~80% sparse expert MLPs
        fp.dense_weights_mb = static_cast<uint64_t>(fp.total_weights_mb * 0.20);
        fp.moe_weights_mb   = fp.total_weights_mb - fp.dense_weights_mb;

        fp.mb_per_expert = static_cast<double>(fp.moe_weights_mb) / (safe_layers * expert_count);
        fp.mb_per_dense_layer = static_cast<double>(fp.dense_weights_mb) / safe_layers;
        fp.mb_per_moe_layer = fp.mb_per_dense_layer + (fp.mb_per_expert * active_experts);
    } else {
        fp.dense_weights_mb = fp.total_weights_mb;
        fp.moe_weights_mb   = 0;
        fp.mb_per_dense_layer = static_cast<double>(fp.total_weights_mb) / safe_layers;
        fp.mb_per_expert = 0.0;
        fp.mb_per_moe_layer = fp.mb_per_dense_layer;
    }

    // Exact KV Cache footprint calculation
    // Formula: 2 * n_layers * n_kv_heads * head_dim * context_length * bytes_per_element
    double bytes_per_elem = 2.0; // f16 default
    if (kv_quant == "turbo4" || kv_quant == "q4_0") {
        bytes_per_elem = 0.55; // 4-bit Polar Lloyd-Max + scales
    } else if (kv_quant == "turbo3" || kv_quant == "q3_k") {
        bytes_per_elem = 0.42; // 3-bit Polar Lloyd-Max
    } else if (kv_quant == "turbo2" || kv_quant == "q2_k") {
        bytes_per_elem = 0.30; // 2-bit Polar Lloyd-Max
    } else if (kv_quant == "q8_0") {
        bytes_per_elem = 1.0625;
    } else if (kv_quant == "bf16") {
        bytes_per_elem = 2.0;
    }

    int safe_n_head_kv = (n_head_kv > 0) ? n_head_kv : 8;
    int safe_head_dim  = (head_dim > 0) ? head_dim : 128;
    int safe_ctx       = (context_length > 0) ? context_length : 4096;

    double kv_raw_bytes = 2.0 * safe_layers * safe_n_head_kv * safe_head_dim * safe_ctx * bytes_per_elem;
    // Include 10% allocation padding for page alignment and metadata tables
    fp.kv_cache_mb = static_cast<uint64_t>((kv_raw_bytes * 1.10) / (1024.0 * 1024.0));
    if (fp.kv_cache_mb < 64) fp.kv_cache_mb = 64; // minimum buffer

    // Scratchpad activation workspace during prefill
    // Scales with chunk size (e.g. 128 or 512) and model embedding dimension
    int safe_embd = (n_embd > 0) ? n_embd : 4096;
    fp.scratchpad_vram_mb = static_cast<uint64_t>((512.0 * safe_embd * 4.0 * 8.0) / (1024.0 * 1024.0));
    if (fp.scratchpad_vram_mb < 120) fp.scratchpad_vram_mb = 120;
    if (fp.scratchpad_vram_mb > 450) fp.scratchpad_vram_mb = 450;

    fp.estimated_total_mb = fp.total_weights_mb + fp.kv_cache_mb + fp.scratchpad_vram_mb;

    return fp;
}

DynamicDispatchRecommendation RAMEstimator::compute_realtime_dispatch(
    const ModelMemoryFootprint & footprint,
    const RealtimeRAMStats & ram,
    const RealtimeVRAMStats & vram,
    bool is_moe,
    int context_length,
    uint64_t max_wayland_guard_mb
) {
    DynamicDispatchRecommendation rec;

    // 1. Determine safe GPU VRAM allocation considering live free memory and Wayland guard
    uint64_t vram_safe_ceiling = max_wayland_guard_mb;
    if (vram.cuda_available && vram.total_mb > 0) {
        // Leave at least 800 MB cushion for OS / Wayland compositor / browser spikes
        uint64_t dynamic_cushion = 800;
        uint64_t live_ceiling = (vram.free_mb > dynamic_cushion) ? (vram.free_mb - dynamic_cushion) : (vram.free_mb / 2);
        vram_safe_ceiling = std::min(max_wayland_guard_mb, live_ceiling);
    }

    // 2. Reserve VRAM for KV cache and scratchpad activations
    uint64_t vram_reserved_for_attn = footprint.kv_cache_mb + footprint.scratchpad_vram_mb;
    uint64_t vram_for_weights = 0;
    if (vram_safe_ceiling > vram_reserved_for_attn + 200) {
        vram_for_weights = vram_safe_ceiling - vram_reserved_for_attn;
    } else {
        vram_for_weights = vram_safe_ceiling / 2;
    }

    rec.target_vram_budget_mb = vram_safe_ceiling;

    // 3. Compute GPU Layer Offload
    if (vram.cuda_available && footprint.mb_per_dense_layer > 0.0) {
        if (footprint.total_weights_mb <= vram_for_weights) {
            // Full offload
            rec.recommended_gpu_layers = 999;
            rec.warm_ddr5_pool_mb = 0;
            rec.status_reason = "Model fits 100% in GPU VRAM with full KV cache allocation.";
        } else {
            // Partial offload: Sandwich 3-Tier Split
            int layers_offload = static_cast<int>(vram_for_weights / footprint.mb_per_dense_layer);
            rec.recommended_gpu_layers = std::max(0, layers_offload);

            uint64_t unoffloaded_weights_mb = (footprint.total_weights_mb > vram_for_weights)
                                                ? (footprint.total_weights_mb - vram_for_weights)
                                                : 0;

            // Apply live 70% available host RAM rule (Technique #42)
            uint64_t max_safe_host_ram = static_cast<uint64_t>(ram.available_mb * 0.70);
            rec.warm_ddr5_pool_mb = std::min(unoffloaded_weights_mb, max_safe_host_ram);

            rec.status_reason = "3-Tier Hybrid: Offloaded " + std::to_string(rec.recommended_gpu_layers) +
                                " layers to VRAM, " + std::to_string(rec.warm_ddr5_pool_mb) + " MB to WARM DDR5 pool.";
        }
    } else {
        // Pure CPU execution
        rec.recommended_gpu_layers = 0;
        rec.warm_ddr5_pool_mb = std::min(footprint.total_weights_mb, static_cast<uint64_t>(ram.available_mb * 0.70));
        rec.status_reason = "Pure CPU execution using host DDR5 memory.";
    }

    // 4. Chunked Prefill & Dynamic Scheduling Configuration
    if (is_moe) {
        // In MoE models (e.g. Qwen Flash Next 125B), large ubatches cause all experts to activate,
        // saturating memory channels and stalling PCIe. Micro-chunking (64 to 128) concentrates
        // activations on a small subset of experts, yielding >4x expert cache locality!
        rec.enable_chunked_prefill = true;
        rec.prefill_chunk_size = (context_length > 16384) ? 64 : 128;
        rec.enable_asym_prefetch = true;
    } else {
        // Dense models: chunked prefill activated for contexts >= 16k to cap transient activation spikes
        if (context_length >= 16384) {
            rec.enable_chunked_prefill = true;
            rec.prefill_chunk_size = 256;
        } else {
            rec.enable_chunked_prefill = false;
            rec.prefill_chunk_size = 512;
        }
        rec.enable_asym_prefetch = (rec.warm_ddr5_pool_mb > 0);
    }

    // 5. Host RAM Caching Layer
    if (rec.warm_ddr5_pool_mb > 0) {
        rec.enable_ram_cache = true;
        // Allocate 15% of the warm pool or at least 1024 MB for evicted KV blocks and hot expert cache
        rec.ram_cache_budget_mb = std::max<uint64_t>(1024, static_cast<uint64_t>(rec.warm_ddr5_pool_mb * 0.15));
    } else {
        rec.enable_ram_cache = false;
        rec.ram_cache_budget_mb = 0;
    }

    return rec;
}

} // namespace dispatch
} // namespace sandwich
