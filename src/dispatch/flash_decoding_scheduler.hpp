#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>
#include <string>

namespace sandwich {
namespace dispatch {

// Technique #79 (Phase 16): Sequence-Parallel Flash-Decoding (Split-KV Attention)
// Breaks long-context KV caches (4k - 128k) into sequence-parallel chunks across
// GPU Streaming Multiprocessors (SMs) during single-token decode (Q=1).
// Computes local maximums and partial sums in parallel, then merges via a logarithmic tree.

struct PartialAttentionReduction {
    float              max_logit = -1e30f;
    float              sum_exp   = 0.0f;
    std::vector<float> accum_out; // Size = head_dim
};

struct FlashDecodingPlan {
    bool        enabled               = false;
    size_t      kv_length             = 0;
    size_t      chunk_size            = 256;
    size_t      num_splits            = 1;
    int         target_sms            = 32;
    int         active_sms_utilized   = 1;
    double      theoretical_speedup   = 1.0;
    std::string diagnostics;
};

class FlashDecodingScheduler {
public:
    // Calculates optimal split-KV partitioning based on sequence length and GPU SM count
    static FlashDecodingPlan calculate_plan(
        size_t kv_length,
        int gpu_sm_count = 32,
        size_t min_split_threshold = 1024,
        size_t chunk_size = 256
    );

    // Numerically stable logarithmic merge of partial chunk reductions
    static std::vector<float> merge_partial_reductions(
        const std::vector<PartialAttentionReduction> & partials,
        size_t head_dim
    );
};

} // namespace dispatch
} // namespace sandwich
