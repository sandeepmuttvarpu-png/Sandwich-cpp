#include "h2o_cache_compressor.hpp"

#include <algorithm>
#include <numeric>
#include <sstream>
#include <iomanip>

namespace sandwich {
namespace dispatch {

H2OCacheCompressor::H2OCacheCompressor(const H2OConfig & config)
    : m_config(config) {
}

void H2OCacheCompressor::reset() {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_accumulated_scores.clear();
    m_total_evictions = 0;
}

void H2OCacheCompressor::accumulate_attention(const std::vector<float> & step_attention) {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (!m_config.enabled || step_attention.empty()) {
        return;
    }

    if (m_accumulated_scores.size() < step_attention.size()) {
        m_accumulated_scores.resize(step_attention.size(), 0.0f);
    }

    for (size_t i = 0; i < step_attention.size(); ++i) {
        m_accumulated_scores[i] = (m_accumulated_scores[i] * m_config.decay_factor) + step_attention[i];
    }
}

bool H2OCacheCompressor::needs_eviction(size_t current_context_length) const {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (!m_config.enabled) {
        return false;
    }
    return current_context_length > m_config.max_budget;
}

std::vector<int32_t> H2OCacheCompressor::compute_eviction_indices(size_t current_context_length) {
    std::lock_guard<std::mutex> lock(m_mutex);
    std::vector<int32_t> evicted;
    if (!m_config.enabled || current_context_length <= m_config.max_budget) {
        return evicted;
    }

    size_t sink_end = std::min(m_config.sink_tokens, current_context_length);
    size_t window_size = std::min(m_config.rolling_window, current_context_length);
    size_t window_start = (current_context_length > window_size) ? (current_context_length - window_size) : current_context_length;

    if (sink_end >= window_start) {
        // Not enough middle tokens to evict
        return evicted;
    }

    // Candidate middle tokens
    struct ScoredToken {
        int32_t index;
        float   score;
    };

    std::vector<ScoredToken> middle_tokens;
    middle_tokens.reserve(window_start - sink_end);

    for (size_t i = sink_end; i < window_start; ++i) {
        float score = (i < m_accumulated_scores.size()) ? m_accumulated_scores[i] : 0.0f;
        middle_tokens.push_back({static_cast<int32_t>(i), score});
    }

    // Sort by score descending
    std::sort(middle_tokens.begin(), middle_tokens.end(), [](const ScoredToken & a, const ScoredToken & b) {
        return a.score > b.score;
    });

    // How many tokens can we keep in the middle?
    size_t reserved_sinks_and_window = sink_end + window_size;
    size_t allowed_heavy_hitters = (m_config.max_budget > reserved_sinks_and_window)
                                     ? std::min(m_config.heavy_hitters, m_config.max_budget - reserved_sinks_and_window)
                                     : 0;

    // Tokens after the top `allowed_heavy_hitters` are evicted
    for (size_t i = allowed_heavy_hitters; i < middle_tokens.size(); ++i) {
        evicted.push_back(middle_tokens[i].index);
    }

    m_total_evictions += evicted.size();
    return evicted;
}

std::vector<int32_t> H2OCacheCompressor::get_retained_indices(size_t current_context_length) const {
    std::lock_guard<std::mutex> lock(m_mutex);
    std::vector<int32_t> retained;
    if (current_context_length == 0) return retained;

    if (!m_config.enabled || current_context_length <= m_config.max_budget) {
        retained.resize(current_context_length);
        std::iota(retained.begin(), retained.end(), 0);
        return retained;
    }

    size_t sink_end = std::min(m_config.sink_tokens, current_context_length);
    size_t window_size = std::min(m_config.rolling_window, current_context_length);
    size_t window_start = (current_context_length > window_size) ? (current_context_length - window_size) : current_context_length;

    // 1. Sinks
    for (size_t i = 0; i < sink_end; ++i) {
        retained.push_back(static_cast<int32_t>(i));
    }

    // 2. Heavy Hitters
    struct ScoredToken {
        int32_t index;
        float   score;
    };
    std::vector<ScoredToken> middle_tokens;
    for (size_t i = sink_end; i < window_start; ++i) {
        float score = (i < m_accumulated_scores.size()) ? m_accumulated_scores[i] : 0.0f;
        middle_tokens.push_back({static_cast<int32_t>(i), score});
    }

    std::sort(middle_tokens.begin(), middle_tokens.end(), [](const ScoredToken & a, const ScoredToken & b) {
        return a.score > b.score;
    });

    size_t reserved_sinks_and_window = sink_end + window_size;
    size_t allowed_heavy_hitters = (m_config.max_budget > reserved_sinks_and_window)
                                     ? std::min(m_config.heavy_hitters, m_config.max_budget - reserved_sinks_and_window)
                                     : 0;

    for (size_t i = 0; i < allowed_heavy_hitters && i < middle_tokens.size(); ++i) {
        retained.push_back(middle_tokens[i].index);
    }

    // 3. Rolling Window
    for (size_t i = window_start; i < current_context_length; ++i) {
        retained.push_back(static_cast<int32_t>(i));
    }

    std::sort(retained.begin(), retained.end());
    return retained;
}

H2OTelemetry H2OCacheCompressor::get_telemetry() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    H2OTelemetry telem;
    telem.total_tokens_seen     = m_accumulated_scores.size();
    telem.total_evictions       = m_total_evictions;
    telem.current_active_tokens = (telem.total_tokens_seen > m_total_evictions)
                                    ? (telem.total_tokens_seen - m_total_evictions)
                                    : telem.total_tokens_seen;

    if (telem.current_active_tokens > 0 && telem.total_tokens_seen > 0) {
        telem.compression_ratio = static_cast<double>(telem.total_tokens_seen) / static_cast<double>(telem.current_active_tokens);
    }

    std::ostringstream ss;
    if (m_config.enabled) {
        ss << "H2O Active [Budget: " << m_config.max_budget
           << ", Sinks: " << m_config.sink_tokens
           << ", HeavyHitters: " << m_config.heavy_hitters
           << ", Evicted: " << m_total_evictions << "]";
    } else {
        ss << "H2O Disabled (Full Context Retained)";
    }
    telem.status_desc = ss.str();
    return telem;
}

} // namespace dispatch
} // namespace sandwich
