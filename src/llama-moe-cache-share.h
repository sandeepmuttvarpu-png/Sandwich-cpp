#pragma once

#include <vector>
#include <cmath>
#include <cstddef>
#include <algorithm>
#include <stdexcept>

namespace llama_moe_cache_detail {

// Computes normalized per-device fractional shares from cumulative split points.
// Handles single device, asymmetric splits, zero fallback (equal distribution among eligible devices),
// and filters out ineligible devices (e.g. GPUs with no host-resident layers assigned).
inline std::vector<double> device_shares(
        const std::vector<float> & cumulative,
        const std::vector<bool>  & eligible) {
    if (cumulative.size() != eligible.size()) {
        throw std::invalid_argument("cumulative and eligible vectors must have the same size");
    }

    const size_t n_dev = cumulative.size();
    std::vector<double> shares(n_dev, 0.0);
    double prev_point = 0.0;

    for (size_t i = 0; i < n_dev; ++i) {
        const double cur_point = (std::isfinite(cumulative[i]) && cumulative[i] >= 0.0f)
            ? std::max(prev_point, (double) cumulative[i])
            : prev_point;
        shares[i] = std::max(0.0, cur_point - prev_point);
        prev_point = cur_point;
    }

    double total_share = 0.0;
    size_t n_eligible = 0;
    for (size_t i = 0; i < n_dev; ++i) {
        if (eligible[i]) {
            total_share += shares[i];
            n_eligible++;
        } else {
            shares[i] = 0.0;
        }
    }

    if (n_eligible == 0) {
        return shares;
    }

    // Fallback: if all cumulative values are zero or sum is non-positive, distribute equally
    if (!std::isfinite(total_share) || total_share <= 0.0) {
        const double uniform_share = 1.0 / (double) n_eligible;
        for (size_t i = 0; i < n_dev; ++i) {
            shares[i] = eligible[i] ? uniform_share : 0.0;
        }
        return shares;
    }

    // Normalize so shares of eligible devices sum to 1.0
    for (size_t i = 0; i < n_dev; ++i) {
        shares[i] = eligible[i] ? (shares[i] / total_share) : 0.0;
    }

    return shares;
}

// Phase 6: Fine-Grained MoE Topology Support (Micro-Experts)
// Detects architectures with thousands of fine-grained micro-experts (e.g. Qwen3.8-Flash-Next, DeepSeek V2/V3)
inline bool is_fine_grained_moe(int64_t n_expert, int64_t n_expert_used) {
    return (n_expert >= 256) || (n_expert > 0 && n_expert_used >= 16);
}

// Solves for the maximum number of expert slots that fit into a VRAM budget using binary search in O(log N)
// Avoids slow linear while loops when sizing caches for models with up to 24,576 micro-experts
inline int32_t compute_optimal_slots(
        size_t vram_budget,
        size_t expert_slice_bytes,
        size_t alignment,
        int32_t max_possible_slots) {
    if (vram_budget == 0 || expert_slice_bytes == 0 || max_possible_slots <= 0) {
        return 0;
    }
    int32_t low = 0;
    int32_t high = max_possible_slots;
    int32_t best = 0;
    while (low <= high) {
        int32_t mid = low + (high - low) / 2;
        const size_t raw_size = expert_slice_bytes * (size_t)(mid + 1);
        const size_t padded_size = alignment > 0 ? ((raw_size + alignment - 1) / alignment) * alignment : raw_size;
        if (padded_size <= vram_budget) {
            best = mid;
            low = mid + 1;
        } else {
            high = mid - 1;
        }
    }
    return best;
}

// Micro-expert routing map: maps routed micro-expert IDs to cache slots
inline int32_t map_micro_expert_to_slot(int32_t expert_id, int32_t n_slots) {
    if (n_slots <= 0) return -1;
    return expert_id % n_slots;
}

} // namespace llama_moe_cache_detail
