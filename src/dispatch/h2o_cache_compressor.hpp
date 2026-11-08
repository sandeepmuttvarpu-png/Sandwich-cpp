#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>
#include <unordered_set>
#include <mutex>
#include <string>

namespace sandwich {
namespace dispatch {

// Technique #67 (Phase 16): Heavy Hitter Oracle (H2O) KV Cache Compression
// Implements dynamic cumulative attention tracking, preserving Attention Sinks (S),
// Heavy Hitters (H), and Rolling Local Window (W), evicting low-activation middle tokens.
struct H2OConfig {
    bool     enabled         = false;
    size_t   sink_tokens     = 4;      // Initial prompt sinks (positions 0..3)
    size_t   heavy_hitters   = 1024;   // Dynamic heavy hitter tokens with high accumulated attention
    size_t   rolling_window  = 128;    // Recent local sliding window
    size_t   max_budget      = 2048;   // Total allowed active KV tokens (sink + heavy + window)
    float    decay_factor    = 0.99f;  // Exponential moving average decay for attention history
};

struct H2OTelemetry {
    size_t total_tokens_seen     = 0;
    size_t current_active_tokens = 0;
    size_t total_evictions       = 0;
    double compression_ratio     = 1.0;
    float  mean_heavy_score      = 0.0f;
    std::string status_desc;
};

class H2OCacheCompressor {
public:
    explicit H2OCacheCompressor(const H2OConfig & config = H2OConfig{});

    // Reset internal state
    void reset();

    // Accumulates attention scores for active tokens during decode step
    void accumulate_attention(const std::vector<float> & step_attention);

    // Determines if KV eviction is required based on budget
    bool needs_eviction(size_t current_context_length) const;

    // Computes indices of tokens to be evicted from the KV cache
    std::vector<int32_t> compute_eviction_indices(size_t current_context_length);

    // Queries current retained active set
    std::vector<int32_t> get_retained_indices(size_t current_context_length) const;

    // Telemetry and statistics
    H2OTelemetry get_telemetry() const;
    const H2OConfig & get_config() const { return m_config; }

private:
    H2OConfig                  m_config;
    std::vector<float>         m_accumulated_scores;
    size_t                     m_total_evictions = 0;
    mutable std::mutex         m_mutex;
};

} // namespace dispatch
} // namespace sandwich
