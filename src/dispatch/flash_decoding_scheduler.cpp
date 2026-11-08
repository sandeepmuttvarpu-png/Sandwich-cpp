#include "flash_decoding_scheduler.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <iomanip>

namespace sandwich {
namespace dispatch {

FlashDecodingPlan FlashDecodingScheduler::calculate_plan(
    size_t kv_length,
    int gpu_sm_count,
    size_t min_split_threshold,
    size_t chunk_size
) {
    FlashDecodingPlan plan;
    plan.kv_length = kv_length;
    plan.chunk_size = (chunk_size > 0) ? chunk_size : 256;
    plan.target_sms = (gpu_sm_count > 0) ? gpu_sm_count : 32;

    if (kv_length < min_split_threshold) {
        plan.enabled = false;
        plan.num_splits = 1;
        plan.active_sms_utilized = 1;
        plan.theoretical_speedup = 1.0;
        plan.diagnostics = "Standard FlashAttention (Short Context: " + std::to_string(kv_length) + " tokens)";
        return plan;
    }

    plan.enabled = true;
    plan.num_splits = (kv_length + plan.chunk_size - 1) / plan.chunk_size;
    plan.active_sms_utilized = std::min(static_cast<int>(plan.num_splits), plan.target_sms);

    // Amdahl's parallel scaling law with 15% reduction overhead
    plan.theoretical_speedup = std::max(1.0, static_cast<double>(plan.active_sms_utilized) * 0.85);

    std::ostringstream ss;
    ss << "Flash-Decoding Active [KV Tokens: " << kv_length
       << ", Splits: " << plan.num_splits
       << " x " << plan.chunk_size << " toks"
       << ", Utilized SMs: " << plan.active_sms_utilized << "/" << plan.target_sms
       << ", Estimated Speedup: " << std::fixed << std::setprecision(1) << plan.theoretical_speedup << "x]";
    plan.diagnostics = ss.str();

    return plan;
}

std::vector<float> FlashDecodingScheduler::merge_partial_reductions(
    const std::vector<PartialAttentionReduction> & partials,
    size_t head_dim
) {
    std::vector<float> final_out(head_dim, 0.0f);
    if (partials.empty() || head_dim == 0) {
        return final_out;
    }

    // Step 1: Compute global maximum logit across all sequence chunks
    float global_max = -1e30f;
    for (const auto & p : partials) {
        if (p.max_logit > global_max) {
            global_max = p.max_logit;
        }
    }

    // Step 2: Compute global denominator with exponential re-scaling
    float global_denom = 0.0f;
    for (const auto & p : partials) {
        float rescale = std::exp(p.max_logit - global_max);
        global_denom += p.sum_exp * rescale;
    }

    if (global_denom <= 1e-12f) {
        return final_out;
    }

    // Step 3: Weighted sum of partial output vectors
    for (const auto & p : partials) {
        if (p.accum_out.size() != head_dim) continue;
        float factor = std::exp(p.max_logit - global_max) / global_denom;
        for (size_t d = 0; d < head_dim; ++d) {
            final_out[d] += p.accum_out[d] * factor;
        }
    }

    return final_out;
}

} // namespace dispatch
} // namespace sandwich
