#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace sandwich {
namespace dispatch {

enum class PrefetchPriority {
    CRITICAL_SHARED_DENSE, // Attention QKV, LayerNorms, Router Gates (always needed, zero wait)
    SPECULATIVE_MOE_LOOKAHEAD, // Layer L+1 / L+2 predicted experts (high-priority lookahead)
    BACKGROUND_COLD_PAGING    // Deep tier cold reads from NVMe SSD via io_uring
};

struct ChunkedPrefillSchedule {
    bool     enabled             = false;
    uint32_t chunk_size          = 128;
    uint32_t total_prompt_tokens = 0;
    uint32_t estimated_chunks    = 0;
    double   est_chunk_ms        = 0.0;
    double   est_total_prefill_s = 0.0;
    std::string strategy_desc;
};

struct AsymmetricPrefetchConfig {
    bool     enabled                 = false;
    int      shared_dense_threads    = 2;
    int      moe_lookahead_threads   = 4;
    int      io_uring_queue_depth    = 128;
    bool     use_p_core_affinity     = true;
    size_t   lookahead_depth_layers  = 2;
};

class PrefetchScheduler {
public:
    // Calculates optimal chunked prefill schedule based on model topology and prompt length
    static ChunkedPrefillSchedule calculate_chunked_schedule(
        bool is_moe,
        int total_layers,
        int expert_count,
        int active_experts,
        int prompt_tokens,
        uint64_t available_vram_mb,
        double ddr5_bandwidth_gbps = 64.0
    );

    // Determines optimal asymmetric prefetch worker configuration
    static AsymmetricPrefetchConfig calculate_prefetch_config(
        int p_cores,
        int e_cores,
        bool is_moe,
        bool has_nvme_gen4
    );

    // Synthesizes recommended CLI arguments for execution
    static std::string synthesize_scheduler_flags(
        const ChunkedPrefillSchedule & chunk_sched,
        const AsymmetricPrefetchConfig & prefetch_cfg
    );
};

} // namespace dispatch
} // namespace sandwich
