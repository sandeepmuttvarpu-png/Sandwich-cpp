#include "prefetch_scheduler.hpp"
#include <algorithm>
#include <cmath>
#include <sstream>

namespace sandwich {
namespace dispatch {

ChunkedPrefillSchedule PrefetchScheduler::calculate_chunked_schedule(
    bool is_moe,
    int total_layers,
    int expert_count,
    int active_experts,
    int prompt_tokens,
    uint64_t available_vram_mb,
    double ddr5_bandwidth_gbps
) {
    ChunkedPrefillSchedule sched;
    sched.total_prompt_tokens = (prompt_tokens > 0) ? prompt_tokens : 2048;

    if (is_moe && expert_count > 0 && active_experts > 0) {
        // MoE models: large chunks trigger massive expert thrashing.
        // Smaller chunks (64 - 128) maximize expert cache locality within the chunk.
        sched.enabled = true;
        if (sched.total_prompt_tokens > 8192 || available_vram_mb < 3000) {
            sched.chunk_size = 64;
            sched.strategy_desc = "MoE Micro-Chunking (64 tokens): Minimizes active expert set to fit resident WARM DDR5 pool.";
        } else {
            sched.chunk_size = 128;
            sched.strategy_desc = "MoE Dynamic Chunking (128 tokens): Optimal balance between tensor core throughput and expert cache hits.";
        }
    } else {
        // Dense models: chunking is beneficial for long contexts (>= 16k) to cap activation memory
        if (sched.total_prompt_tokens >= 16384 || available_vram_mb < 2000) {
            sched.enabled = true;
            sched.chunk_size = 256;
            sched.strategy_desc = "Dense Context Chunking (256 tokens): Caches transient activation memory below 350 MB.";
        } else {
            sched.enabled = false;
            sched.chunk_size = 512;
            sched.strategy_desc = "Standard Prefill (512 tokens): Maximizes GEMM compute efficiency on resident weights.";
        }
    }

    sched.estimated_chunks = (sched.total_prompt_tokens + sched.chunk_size - 1) / sched.chunk_size;

    // Approximate latency per chunk:
    // Memory traffic per chunk = chunk_size * active_params_per_layer * total_layers
    double safe_bw = (ddr5_bandwidth_gbps > 10.0) ? ddr5_bandwidth_gbps : 64.0;
    double mb_per_chunk = (is_moe) ? (sched.chunk_size * 0.45 * total_layers) : (sched.chunk_size * 0.85 * total_layers);
    sched.est_chunk_ms = std::max(5.0, (mb_per_chunk / (safe_bw * 1024.0)) * 1000.0);
    sched.est_total_prefill_s = (sched.estimated_chunks * sched.est_chunk_ms) / 1000.0;

    return sched;
}

AsymmetricPrefetchConfig PrefetchScheduler::calculate_prefetch_config(
    int p_cores,
    int e_cores,
    bool is_moe,
    bool has_nvme_gen4
) {
    AsymmetricPrefetchConfig cfg;
    cfg.enabled = true;
    cfg.use_p_core_affinity = (p_cores >= 4);

    if (is_moe) {
        // Dedicated workers: 2 for shared dense weights (pinned DMA), remaining for MoE lookahead
        cfg.shared_dense_threads = 2;
        int remaining_p = std::max(2, p_cores - 2);
        cfg.moe_lookahead_threads = std::min(4, remaining_p);
        cfg.lookahead_depth_layers = 2;
        cfg.io_uring_queue_depth = has_nvme_gen4 ? 128 : 64;
    } else {
        cfg.shared_dense_threads = 2;
        cfg.moe_lookahead_threads = 0;
        cfg.lookahead_depth_layers = 1;
        cfg.io_uring_queue_depth = 64;
    }

    return cfg;
}

std::string PrefetchScheduler::synthesize_scheduler_flags(
    const ChunkedPrefillSchedule & chunk_sched,
    const AsymmetricPrefetchConfig & prefetch_cfg
) {
    std::ostringstream ss;
    if (chunk_sched.enabled) {
        ss << "--chunked-prefill -ub " << chunk_sched.chunk_size << " ";
    } else {
        ss << "-ub " << chunk_sched.chunk_size << " ";
    }

    if (prefetch_cfg.enabled) {
        ss << "--io-uring-qd " << prefetch_cfg.io_uring_queue_depth << " ";
    }

    return ss.str();
}

} // namespace dispatch
} // namespace sandwich
