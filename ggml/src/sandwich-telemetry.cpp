#include "sandwich-telemetry.h"

#include <cstdio>
#include <cinttypes>
#include <atomic>
#include <mutex>
#include <unordered_map>

static std::atomic<uint64_t> g_hot_hits{0};
static std::atomic<uint64_t> g_cold_misses{0};
static std::atomic<uint64_t> g_warmup_tokens{0};

static std::mutex g_freq_mutex;
static std::unordered_map<uint64_t, int64_t> g_expert_freqs;

void sandwich_telemetry_record_hits(uint64_t hits) {
    g_hot_hits.fetch_add(hits, std::memory_order_relaxed);
}

void sandwich_telemetry_record_misses(uint64_t misses) {
    g_cold_misses.fetch_add(misses, std::memory_order_relaxed);
}

void sandwich_telemetry_record_expert(int32_t il, int32_t expert_id) {
    if (expert_id < 0 || il < 0) return;
    const uint64_t key = (static_cast<uint64_t>(il) << 32) | static_cast<uint32_t>(expert_id);
    std::lock_guard<std::mutex> lock(g_freq_mutex);
    g_expert_freqs[key]++;
}

int64_t sandwich_telemetry_get_expert_freq(int32_t il, int32_t expert_id) {
    const uint64_t key = (static_cast<uint64_t>(il) << 32) | static_cast<uint32_t>(expert_id);
    std::lock_guard<std::mutex> lock(g_freq_mutex);
    auto it = g_expert_freqs.find(key);
    return (it != g_expert_freqs.end()) ? it->second : 0;
}

uint64_t sandwich_telemetry_get_total_warmup_tokens(void) {
    return g_warmup_tokens.load(std::memory_order_relaxed);
}

void sandwich_telemetry_increment_warmup_tokens(void) {
    g_warmup_tokens.fetch_add(1, std::memory_order_relaxed);
}

sandwich_telemetry_stats_t sandwich_telemetry_get_stats(void) {
    sandwich_telemetry_stats_t stats;
    stats.hot_hits    = g_hot_hits.load(std::memory_order_relaxed);
    stats.cold_misses = g_cold_misses.load(std::memory_order_relaxed);
    stats.total_evals = stats.hot_hits + stats.cold_misses;
    stats.hit_rate_pct = stats.total_evals > 0
        ? (100.0 * (double) stats.hot_hits / (double) stats.total_evals)
        : 0.0;
    return stats;
}

static uint64_t g_last_printed_evals = 0;

void sandwich_telemetry_reset(void) {
    g_hot_hits.store(0, std::memory_order_relaxed);
    g_cold_misses.store(0, std::memory_order_relaxed);
    g_warmup_tokens.store(0, std::memory_order_relaxed);
    g_last_printed_evals = 0;
    std::lock_guard<std::mutex> lock(g_freq_mutex);
    g_expert_freqs.clear();
}

void sandwich_telemetry_print(void) {
    const auto stats = sandwich_telemetry_get_stats();
    if (stats.total_evals == 0 || stats.total_evals == g_last_printed_evals) {
        return;
    }
    g_last_printed_evals = stats.total_evals;

    fprintf(stderr, "\n");
    fprintf(stderr, "=======================================================\n");
    fprintf(stderr, "🥪 [Project Sandwich] MoE Cache Runtime Telemetry\n");
    fprintf(stderr, "   Total Routed Expert Invocations : %10" PRIu64 "\n", stats.total_evals);
    fprintf(stderr, "   GPU Hot-Pack Cache Hits        : %10" PRIu64 " (%.2f%%)\n", stats.hot_hits, stats.hit_rate_pct);
    fprintf(stderr, "   CPU Cold-Pack Misses           : %10" PRIu64 " (%.2f%%)\n", stats.cold_misses, 100.0 - stats.hit_rate_pct);
    fprintf(stderr, "   PCIe Transfer Bandwidth Saved  : %10.2f%%\n", stats.hit_rate_pct);
    fprintf(stderr, "=======================================================\n");
    fprintf(stderr, "\n");
}
