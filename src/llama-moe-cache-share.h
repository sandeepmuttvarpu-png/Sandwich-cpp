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

} // namespace llama_moe_cache_detail
