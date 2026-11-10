#pragma once

#include <string>
#include <cstdint>
#include <vector>
#include <memory>

namespace sandwich {
namespace dispatch {

// Real-time snapshot of system host memory
struct RealtimeRAMStats {
    uint64_t total_mb         = 0;
    uint64_t free_mb          = 0;
    uint64_t available_mb     = 0; // True available memory considering reclaimable pagecache
    uint64_t cached_mb        = 0;
    uint64_t buffers_mb       = 0;
    uint64_t swap_total_mb    = 0;
    uint64_t swap_free_mb     = 0;
    double   used_percentage  = 0.0;
};

// Real-time snapshot of GPU VRAM
struct RealtimeVRAMStats {
    bool     cuda_available   = false;
    std::string device_name   = "";
    uint64_t total_mb         = 0;
    uint64_t free_mb          = 0;
    uint64_t used_mb          = 0;
    double   used_percentage  = 0.0;
};

// Detailed breakdown of estimated model memory requirements
struct ModelMemoryFootprint {
    uint64_t total_weights_mb     = 0;
    uint64_t dense_weights_mb     = 0;
    uint64_t moe_weights_mb       = 0;
    uint64_t kv_cache_mb          = 0;
    uint64_t scratchpad_vram_mb   = 0;
    uint64_t estimated_total_mb   = 0;
    int      total_layers         = 32;

    // Per-layer breakdowns
    double   mb_per_dense_layer   = 0.0;
    double   mb_per_expert        = 0.0;
    double   mb_per_moe_layer     = 0.0;
};

// Auto-dispatcher dynamic recommendation
struct DynamicDispatchRecommendation {
    int      recommended_gpu_layers = 0;
    uint64_t target_vram_budget_mb  = 0;
    uint64_t warm_ddr5_pool_mb      = 0;
    uint32_t prefill_chunk_size     = 512;
    bool     enable_chunked_prefill = false;
    bool     enable_asym_prefetch   = false;
    bool     enable_ram_cache       = false;
    uint64_t ram_cache_budget_mb    = 0;
    std::string status_reason;
};

class RAMEstimator {
public:
    // Samples real-time host RAM directly from /proc/meminfo
    static RealtimeRAMStats sample_realtime_ram();

    // Samples real-time GPU VRAM directly from CUDA driver / sysfs
    static RealtimeVRAMStats sample_realtime_vram();

    // Calculates precise model memory footprint based on exact architecture parameters
    static ModelMemoryFootprint calculate_footprint(
        uint64_t model_bytes,
        int total_layers,
        int context_length,
        const std::string & kv_quant,
        bool is_moe,
        int expert_count = 0,
        int active_experts = 0,
        int n_embd = 4096,
        int n_head_kv = 8,
        int head_dim = 128
    );

    // Dynamic auto-dispatcher engine: computes optimal tiering and prefill chunk size
    // using live real-time RAM/VRAM state rather than static constants
    static DynamicDispatchRecommendation compute_realtime_dispatch(
        const ModelMemoryFootprint & footprint,
        const RealtimeRAMStats & ram,
        const RealtimeVRAMStats & vram,
        bool is_moe,
        int context_length,
        uint64_t max_wayland_guard_mb = 7800
    );
};

} // namespace dispatch
} // namespace sandwich
