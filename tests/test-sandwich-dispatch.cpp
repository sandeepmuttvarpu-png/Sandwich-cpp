#include "ram_estimator.hpp"
#include "memory_watermark.hpp"
#include "prefetch_scheduler.hpp"
#include "ram_cache_layer.hpp"
#include "h2o_cache_compressor.hpp"

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>

#define SANDWICH_TEST_CHECK(cond) \
    do { \
        if (!(cond)) { \
            fprintf(stderr, "TEST ASSERTION FAILED: %s at %s:%d\n", #cond, __FILE__, __LINE__); \
            std::abort(); \
        } \
    } while (0)

using namespace sandwich::dispatch;

static void test_ram_estimator() {
    printf("[test_ram_estimator] Starting...\n");

    // 1. Host RAM sampling
    RealtimeRAMStats ram = RAMEstimator::sample_realtime_ram();
    SANDWICH_TEST_CHECK(ram.total_mb > 0);
    SANDWICH_TEST_CHECK(ram.available_mb > 0);
    SANDWICH_TEST_CHECK(ram.available_mb <= ram.total_mb);
    printf("  RAM: Total=%zu MB, Available=%zu MB, Free=%zu MB\n",
           (size_t)ram.total_mb, (size_t)ram.available_mb, (size_t)ram.free_mb);

    // 2. VRAM sampling
    RealtimeVRAMStats vram = RAMEstimator::sample_realtime_vram();
    if (vram.cuda_available) {
        SANDWICH_TEST_CHECK(vram.total_mb > 0);
        SANDWICH_TEST_CHECK(vram.free_mb <= vram.total_mb);
        printf("  VRAM: Detected %s (Total=%zu MB, Free=%zu MB)\n",
               vram.device_name.c_str(), (size_t)vram.total_mb, (size_t)vram.free_mb);
    } else {
        printf("  VRAM: No CUDA device active in current environment (handled gracefully)\n");
    }

    // 3. Footprint calculation for Qwen Flash Next 125B
    uint64_t model_bytes = 70ULL * 1024ULL * 1024ULL * 1024ULL; // ~70 GB file
    int layers = 48;
    int ctx_64k = 65536;
    int ctx_4k  = 4096;

    ModelMemoryFootprint fp_turbo4 = RAMEstimator::calculate_footprint(
        model_bytes, layers, ctx_64k, "turbo4", true, 64, 4, 4096, 8, 128
    );
    SANDWICH_TEST_CHECK(fp_turbo4.total_weights_mb > 0);
    SANDWICH_TEST_CHECK(fp_turbo4.kv_cache_mb > 0);
    SANDWICH_TEST_CHECK(fp_turbo4.estimated_total_mb >= fp_turbo4.total_weights_mb + fp_turbo4.kv_cache_mb);

    // KV cache scaling test: 64k context vs 4k context
    ModelMemoryFootprint fp_4k = RAMEstimator::calculate_footprint(
        model_bytes, layers, ctx_4k, "turbo4", true, 64, 4, 4096, 8, 128
    );
    SANDWICH_TEST_CHECK(fp_turbo4.kv_cache_mb > fp_4k.kv_cache_mb);

    // KV quantization scaling: turbo2 vs turbo4
    ModelMemoryFootprint fp_turbo2 = RAMEstimator::calculate_footprint(
        model_bytes, layers, ctx_64k, "turbo2", true, 64, 4, 4096, 8, 128
    );
    SANDWICH_TEST_CHECK(fp_turbo2.kv_cache_mb < fp_turbo4.kv_cache_mb);

    // 4. Real-time dynamic dispatch recommendation
    DynamicDispatchRecommendation rec = RAMEstimator::compute_realtime_dispatch(
        fp_turbo4, ram, vram, true, ctx_64k, 7100
    );
    SANDWICH_TEST_CHECK(rec.recommended_gpu_layers >= 0);
    SANDWICH_TEST_CHECK(rec.prefill_chunk_size > 0);
    printf("  Recommendation: %d GPU layers, chunk size %u, reason: %s\n",
           rec.recommended_gpu_layers, rec.prefill_chunk_size, rec.status_reason.c_str());

    printf("  [PASS] test_ram_estimator\n");
}

static void test_memory_watermarker() {
    printf("[test_memory_watermarker] Starting...\n");

    WatermarkThresholds thresholds;
    thresholds.vram_hwm_ratio = 0.90;
    thresholds.ram_hwm_ratio = 0.85;
    thresholds.ram_min_headroom_mb = 2048;

    MemoryWatermarker wm(thresholds);

    bool callback_triggered = false;
    wm.set_pressure_callback([&](WatermarkLevel lvl) {
        if (lvl == WatermarkLevel::CRITICAL_THROTTLE || lvl == WatermarkLevel::PRESSURE_WARNING) {
            callback_triggered = true;
        }
    });

    // Case 1: Healthy normal conditions
    RealtimeRAMStats normal_ram;
    normal_ram.total_mb = 32768;
    normal_ram.available_mb = 20000;
    normal_ram.free_mb = 18000;

    RealtimeVRAMStats normal_vram;
    normal_vram.cuda_available = true;
    normal_vram.total_mb = 8192;
    normal_vram.free_mb = 4096;

    WatermarkLevel lvl1 = wm.update_realtime(normal_ram, normal_vram);
    SANDWICH_TEST_CHECK(lvl1 == WatermarkLevel::NORMAL);
    SANDWICH_TEST_CHECK(!wm.is_throttled());
    SANDWICH_TEST_CHECK(!wm.is_critical());
    SANDWICH_TEST_CHECK(!callback_triggered);

    // Case 2: Severe RAM exhaustion (available memory drops below headroom)
    RealtimeRAMStats exhausted_ram = normal_ram;
    exhausted_ram.available_mb = 500; // < 2048 MB headroom

    WatermarkLevel lvl2 = wm.update_realtime(exhausted_ram, normal_vram);
    SANDWICH_TEST_CHECK(lvl2 == WatermarkLevel::CRITICAL_THROTTLE);
    SANDWICH_TEST_CHECK(wm.is_throttled());
    SANDWICH_TEST_CHECK(wm.is_critical());
    SANDWICH_TEST_CHECK(callback_triggered);

    // Adapt prefill chunk size under pressure
    uint32_t base_chunk = 512;
    uint32_t adapted = wm.adapt_prefill_chunk_size(base_chunk);
    SANDWICH_TEST_CHECK(adapted < base_chunk);

    // Case 3: Recovery back to normal
    RealtimeRAMStats recovered_ram = normal_ram;
    recovered_ram.available_mb = 22000;
    wm.update_realtime(recovered_ram, normal_vram);
    WatermarkTelemetry telem = wm.get_telemetry();
    SANDWICH_TEST_CHECK(telem.recovery_events > 0);

    printf("  [PASS] test_memory_watermarker\n");
}

static void test_prefetch_scheduler() {
    printf("[test_prefetch_scheduler] Starting...\n");

    // 1. Chunked Prefill Schedule calculation
    int prompt_tokens = 4096;
    ChunkedPrefillSchedule sched = PrefetchScheduler::calculate_chunked_schedule(
        true,   // is_moe
        48,     // total_layers
        64,     // expert_count
        4,      // active_experts
        prompt_tokens,
        6000,   // available_vram_mb
        64.0    // ddr5_bandwidth_gbps
    );

    SANDWICH_TEST_CHECK(sched.enabled == true);
    SANDWICH_TEST_CHECK(sched.chunk_size <= 512);
    SANDWICH_TEST_CHECK(sched.chunk_size > 0);
    SANDWICH_TEST_CHECK(sched.estimated_chunks == (uint32_t)std::ceil((double)prompt_tokens / sched.chunk_size));
    SANDWICH_TEST_CHECK(sched.est_chunk_ms > 0.0);
    printf("  Schedule: chunk_size=%u, chunks=%u, est_total=%.2fs\n",
           sched.chunk_size, sched.estimated_chunks, sched.est_total_prefill_s);

    // 2. Asymmetric Prefetch Worker Config
    AsymmetricPrefetchConfig cfg = PrefetchScheduler::calculate_prefetch_config(
        8,  // p_cores
        8,  // e_cores
        true, // is_moe
        true  // has_nvme_gen4
    );

    SANDWICH_TEST_CHECK(cfg.enabled == true);
    SANDWICH_TEST_CHECK(cfg.shared_dense_threads >= 2);
    SANDWICH_TEST_CHECK(cfg.moe_lookahead_threads >= 2);
    SANDWICH_TEST_CHECK(cfg.io_uring_queue_depth >= 64);
    SANDWICH_TEST_CHECK(cfg.lookahead_depth_layers >= 2);

    // 3. Flags synthesis
    std::string flags = PrefetchScheduler::synthesize_scheduler_flags(sched, cfg);
    SANDWICH_TEST_CHECK(!flags.empty());
    SANDWICH_TEST_CHECK(flags.find("-ub") != std::string::npos);
    SANDWICH_TEST_CHECK(flags.find("--chunked-prefill") != std::string::npos);
    printf("  Synthesized flags: %s\n", flags.c_str());

    printf("  [PASS] test_prefetch_scheduler\n");
}

static void test_ram_cache_layer() {
    printf("[test_ram_cache_layer] Starting...\n");

    // Allocate 4 MB test cache
    RAMCacheLayer cache(4);
    bool init_ok = cache.initialize();
    SANDWICH_TEST_CHECK(init_ok);

    RAMCacheStats s1 = cache.get_stats();
    SANDWICH_TEST_CHECK(s1.capacity_bytes == 4ULL * 1024ULL * 1024ULL);
    SANDWICH_TEST_CHECK(s1.cache_hits == 0);
    SANDWICH_TEST_CHECK(s1.cache_misses == 0);

    // Record cache hit and miss events
    cache.record_hit(65536);
    cache.record_hit(65536);
    cache.record_miss(65536);
    cache.record_eviction(65536);

    RAMCacheStats s2 = cache.get_stats();
    SANDWICH_TEST_CHECK(s2.cache_hits == 2);
    SANDWICH_TEST_CHECK(s2.cache_misses == 1);
    SANDWICH_TEST_CHECK(s2.evictions == 1);
    SANDWICH_TEST_CHECK(s2.hit_ratio > 0.60 && s2.hit_ratio < 0.70); // 2/3 = 66.7%

    cache.release();
    printf("  [PASS] test_ram_cache_layer\n");
}

static void test_h2o_cache_compressor() {
    printf("[test_h2o_cache_compressor] Starting...\n");

    H2OConfig cfg;
    cfg.enabled = true;
    cfg.sink_tokens = 4;
    cfg.heavy_hitters = 16;
    cfg.rolling_window = 16;
    cfg.max_budget = 36; // 4 sinks + 16 heavy + 16 window

    H2OCacheCompressor h2o(cfg);

    // Context length = 20 (below budget of 36) -> no eviction
    SANDWICH_TEST_CHECK(!h2o.needs_eviction(20));
    std::vector<int32_t> ev1 = h2o.compute_eviction_indices(20);
    SANDWICH_TEST_CHECK(ev1.empty());

    // Generate simulated attention distribution over 100 tokens
    size_t ctx_len = 100;
    std::vector<float> attn(ctx_len, 0.01f);
    // Spike attention on specific key tokens (e.g. tokens 10, 25, 42)
    attn[10] = 5.0f;
    attn[25] = 8.0f;
    attn[42] = 12.0f;

    h2o.accumulate_attention(attn);

    SANDWICH_TEST_CHECK(h2o.needs_eviction(ctx_len));
    std::vector<int32_t> ev_indices = h2o.compute_eviction_indices(ctx_len);
    SANDWICH_TEST_CHECK(!ev_indices.empty());

    // Total tokens minus evictions must be <= max_budget
    size_t retained_count = ctx_len - ev_indices.size();
    SANDWICH_TEST_CHECK(retained_count <= cfg.max_budget);

    // Verify Sinks (0..3) are NEVER evicted
    for (int32_t idx : ev_indices) {
        SANDWICH_TEST_CHECK(idx >= 4);
        SANDWICH_TEST_CHECK(idx < static_cast<int32_t>(ctx_len - cfg.rolling_window));
    }

    // Verify top Heavy Hitters (10, 25, 42) are retained
    std::vector<int32_t> retained = h2o.get_retained_indices(ctx_len);
    SANDWICH_TEST_CHECK(std::find(retained.begin(), retained.end(), 10) != retained.end());
    SANDWICH_TEST_CHECK(std::find(retained.begin(), retained.end(), 25) != retained.end());
    SANDWICH_TEST_CHECK(std::find(retained.begin(), retained.end(), 42) != retained.end());

    H2OTelemetry telem = h2o.get_telemetry();
    SANDWICH_TEST_CHECK(telem.total_evictions > 0);
    SANDWICH_TEST_CHECK(telem.compression_ratio > 1.0);
    printf("  H2O Telemetry: %s, compression ratio: %.2fx\n",
           telem.status_desc.c_str(), telem.compression_ratio);

    printf("  [PASS] test_h2o_cache_compressor\n");
}

int main() {
    printf("=== Running sandwich.cpp Dispatch Subsystem Unit Tests ===\n");
    test_ram_estimator();
    test_memory_watermarker();
    test_prefetch_scheduler();
    test_ram_cache_layer();
    test_h2o_cache_compressor();
    printf("=== All Sandwich Dispatch Tests Passed Successfully! ===\n");
    return 0;
}
