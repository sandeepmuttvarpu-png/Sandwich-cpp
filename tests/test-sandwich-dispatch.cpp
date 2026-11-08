#include "ram_estimator.hpp"
#include "memory_watermark.hpp"
#include "prefetch_scheduler.hpp"
#include "ram_cache_layer.hpp"
#include "h2o_cache_compressor.hpp"
#include "paged_block_allocator.hpp"
#include "flash_decoding_scheduler.hpp"
#include "gpudirect_storage.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <unistd.h>

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

static void test_paged_block_allocator() {
    printf("[test_paged_block_allocator] Starting...\n");

    PagedBlockConfig cfg;
    cfg.tokens_per_block = 16;
    cfg.total_blocks = 128; // 128 * 16 = 2048 tokens capacity

    PagedBlockAllocator alloc(cfg);

    // 1. Create Sequence and append tokens
    int32_t seq1 = alloc.create_sequence();
    SANDWICH_TEST_CHECK(seq1 > 0);

    // Append 40 tokens -> requires ceil(40/16) = 3 physical blocks
    bool ok1 = alloc.append_tokens(seq1, 40);
    SANDWICH_TEST_CHECK(ok1);
    SANDWICH_TEST_CHECK(alloc.get_sequence_tokens(seq1) == 40);

    std::vector<int32_t> tbl1 = alloc.get_block_table(seq1);
    SANDWICH_TEST_CHECK(tbl1.size() == 3);

    PagedBlockStats st1 = alloc.get_stats();
    SANDWICH_TEST_CHECK(st1.allocated_blocks == 3);
    SANDWICH_TEST_CHECK(st1.shared_blocks == 0);

    // 2. Fork Sequence (Copy-on-Write for Speculative Trees / Chat Prefix)
    int32_t seq2 = alloc.fork_sequence(seq1);
    SANDWICH_TEST_CHECK(seq2 > seq1);
    SANDWICH_TEST_CHECK(alloc.get_sequence_tokens(seq2) == 40);

    std::vector<int32_t> tbl2 = alloc.get_block_table(seq2);
    SANDWICH_TEST_CHECK(tbl2 == tbl1); // Shares identical physical blocks!

    PagedBlockStats st2 = alloc.get_stats();
    SANDWICH_TEST_CHECK(st2.allocated_blocks == 3); // Zero new blocks allocated!
    SANDWICH_TEST_CHECK(st2.shared_blocks == 3);    // All 3 blocks shared via refcount!

    // 3. Append to forked sequence (CoW test)
    bool ok2 = alloc.append_tokens(seq2, 10);
    SANDWICH_TEST_CHECK(ok2);
    SANDWICH_TEST_CHECK(alloc.get_sequence_tokens(seq2) == 50);

    // 4. Free sequence 1
    alloc.free_sequence(seq1);
    PagedBlockStats st3 = alloc.get_stats();
    SANDWICH_TEST_CHECK(st3.active_sequences == 1);

    // 5. Free sequence 2 -> all blocks returned to free list
    alloc.free_sequence(seq2);
    PagedBlockStats st4 = alloc.get_stats();
    SANDWICH_TEST_CHECK(st4.allocated_blocks == 0);
    SANDWICH_TEST_CHECK(st4.free_blocks == cfg.total_blocks);
    printf("  PagedAttention Stats: %s\n", st2.diagnostics.c_str());

    printf("  [PASS] test_paged_block_allocator\n");
}

static void test_flash_decoding_scheduler() {
    printf("[test_flash_decoding_scheduler] Starting...\n");

    // Case 1: Short context (512 tokens) -> Standard FlashAttention (disabled)
    FlashDecodingPlan p_short = FlashDecodingScheduler::calculate_plan(512, 32, 1024, 256);
    SANDWICH_TEST_CHECK(!p_short.enabled);
    SANDWICH_TEST_CHECK(p_short.num_splits == 1);
    SANDWICH_TEST_CHECK(p_short.active_sms_utilized == 1);

    // Case 2: Long context (65,536 tokens on 32-SM RTX 5050 Mobile)
    FlashDecodingPlan p_long = FlashDecodingScheduler::calculate_plan(65536, 32, 1024, 256);
    SANDWICH_TEST_CHECK(p_long.enabled);
    SANDWICH_TEST_CHECK(p_long.num_splits == 256); // 65536 / 256 = 256 chunks
    SANDWICH_TEST_CHECK(p_long.active_sms_utilized == 32); // All 32 SMs saturated!
    SANDWICH_TEST_CHECK(p_long.theoretical_speedup > 20.0);
    printf("  Flash-Decoding 64k: %s\n", p_long.diagnostics.c_str());

    // Case 3: Logarithmic reduction merge of partial reductions
    size_t head_dim = 128;
    std::vector<PartialAttentionReduction> partials;

    // Chunk 0: max logit 2.0, sum 10.0, unnormalized output = 1.0f * 10.0 = 10.0f
    PartialAttentionReduction r0;
    r0.max_logit = 2.0f;
    r0.sum_exp = 10.0f;
    r0.accum_out.assign(head_dim, 10.0f);
    partials.push_back(r0);

    // Chunk 1: max logit 5.0 (dominant chunk!), sum 20.0, unnormalized output = 3.0f * 20.0 = 60.0f
    PartialAttentionReduction r1;
    r1.max_logit = 5.0f;
    r1.sum_exp = 20.0f;
    r1.accum_out.assign(head_dim, 60.0f);
    partials.push_back(r1);

    std::vector<float> merged = FlashDecodingScheduler::merge_partial_reductions(partials, head_dim);
    SANDWICH_TEST_CHECK(merged.size() == head_dim);
    // Since chunk 1 has max_logit 5.0 vs 2.0 (e^(2-5) = e^-3 ~= 0.05), chunk 1 heavily dominates output
    SANDWICH_TEST_CHECK(merged[0] > 2.8f && merged[0] <= 3.0f);
    printf("  Logarithmic reduction verified: merged head[0] = %.4f (expected ~2.99f)\n", merged[0]);

    printf("  [PASS] test_flash_decoding_scheduler\n");
}

static void test_gpudirect_storage() {
    printf("[test_gpudirect_storage] Starting...\n");

    // 1. Test 4KB alignment arithmetic
    uint64_t aligned_off = 0;
    size_t aligned_len = 0;
    size_t pad = 0;

    GDSManager::align_bounds(0, 4096, 4096, aligned_off, aligned_len, pad);
    SANDWICH_TEST_CHECK(aligned_off == 0);
    SANDWICH_TEST_CHECK(aligned_len == 4096);
    SANDWICH_TEST_CHECK(pad == 0);

    GDSManager::align_bounds(100, 500, 4096, aligned_off, aligned_len, pad);
    SANDWICH_TEST_CHECK(aligned_off == 0);
    SANDWICH_TEST_CHECK(aligned_len == 4096);
    SANDWICH_TEST_CHECK(pad == 100);

    GDSManager::align_bounds(4500, 8192, 4096, aligned_off, aligned_len, pad);
    SANDWICH_TEST_CHECK(aligned_off == 4096);
    SANDWICH_TEST_CHECK(aligned_len == 12288);
    SANDWICH_TEST_CHECK(pad == (4500 - 4096));

    // 2. Test initialization and diagnostics
    GDSConfig cfg;
    cfg.enabled = true;
    GDSManager gds(cfg);
    SANDWICH_TEST_CHECK(gds.is_active());

    GDSStatus st = gds.get_status();
    SANDWICH_TEST_CHECK(st.enabled);
    SANDWICH_TEST_CHECK(!st.backend_name.empty());
    printf("  Initial GDS Backend: %s\n", st.backend_name.c_str());

    // 3. Test direct DMA transfer on a temporary file
    char tmp_path[] = "/tmp/sandwich_gds_test_XXXXXX";
    int tmp_fd = mkstemp(tmp_path);
    SANDWICH_TEST_CHECK(tmp_fd >= 0);

    // Populate file with 16KB of pseudo-random known data
    const size_t file_size = 16384;
    std::vector<uint8_t> test_data(file_size);
    for (size_t i = 0; i < file_size; ++i) {
        test_data[i] = static_cast<uint8_t>((i * 31 + 17) & 0xFF);
    }
    ssize_t written = write(tmp_fd, test_data.data(), file_size);
    SANDWICH_TEST_CHECK(written == static_cast<ssize_t>(file_size));
    fsync(tmp_fd);

    // Test Case A: Aligned 4096 byte read
    std::vector<uint8_t> read_buf_a(4096, 0);
    GDSTransferRequest req_a;
    req_a.fd = tmp_fd;
    req_a.file_offset = 4096;
    req_a.length = 4096;
    req_a.dest_ptr = read_buf_a.data();
    req_a.is_gpu_vram = false;

    GDSTransferResult res_a = gds.transfer(req_a);
    SANDWICH_TEST_CHECK(res_a.success);
    SANDWICH_TEST_CHECK(res_a.bytes_transferred == 4096);
    SANDWICH_TEST_CHECK(std::memcmp(read_buf_a.data(), test_data.data() + 4096, 4096) == 0);

    // Test Case B: Unaligned read (offset 512, length 1024)
    std::vector<uint8_t> read_buf_b(1024, 0);
    GDSTransferRequest req_b;
    req_b.fd = tmp_fd;
    req_b.file_offset = 512;
    req_b.length = 1024;
    req_b.dest_ptr = read_buf_b.data();
    req_b.is_gpu_vram = false;

    GDSTransferResult res_b = gds.transfer(req_b);
    SANDWICH_TEST_CHECK(res_b.success);
    SANDWICH_TEST_CHECK(res_b.bytes_transferred == 1024);
    SANDWICH_TEST_CHECK(std::memcmp(read_buf_b.data(), test_data.data() + 512, 1024) == 0);

    // 4. Test Batch DMA transfers
    std::vector<uint8_t> batch_buf(4096 * 2, 0);
    std::vector<GDSTransferRequest> batch_reqs(2);
    batch_reqs[0].fd = tmp_fd;
    batch_reqs[0].file_offset = 0;
    batch_reqs[0].length = 4096;
    batch_reqs[0].dest_ptr = batch_buf.data();

    batch_reqs[1].fd = tmp_fd;
    batch_reqs[1].file_offset = 8192;
    batch_reqs[1].length = 4096;
    batch_reqs[1].dest_ptr = batch_buf.data() + 4096;

    auto batch_results = gds.transfer_batch(batch_reqs);
    SANDWICH_TEST_CHECK(batch_results.size() == 2);
    SANDWICH_TEST_CHECK(batch_results[0].success && batch_results[0].bytes_transferred == 4096);
    SANDWICH_TEST_CHECK(batch_results[1].success && batch_results[1].bytes_transferred == 4096);

    // Cleanup temp file
    close(tmp_fd);
    unlink(tmp_path);

    // 5. Test metrics & diagnostics
    GDSStatus st_after = gds.get_status();
    SANDWICH_TEST_CHECK(st_after.total_transfers == 4); // req_a + req_b + 2 batch
    SANDWICH_TEST_CHECK(st_after.total_bytes == (4096 + 1024 + 4096 + 4096));
    printf("  Diagnostics after transfers: %s\n", st_after.diagnostics.c_str());

    gds.reset_metrics();
    SANDWICH_TEST_CHECK(gds.get_status().total_transfers == 0);
    SANDWICH_TEST_CHECK(gds.get_status().total_bytes == 0);

    // 6. Test Error Handling
    GDSTransferRequest req_err;
    req_err.fd = -1;
    req_err.dest_ptr = read_buf_a.data();
    req_err.length = 4096;
    GDSTransferResult res_err = gds.transfer(req_err);
    SANDWICH_TEST_CHECK(!res_err.success);

    printf("  [PASS] test_gpudirect_storage\n");
}

int main() {
    printf("=== Running sandwich.cpp Dispatch Subsystem Unit Tests ===\n");
    test_ram_estimator();
    test_memory_watermarker();
    test_prefetch_scheduler();
    test_ram_cache_layer();
    test_h2o_cache_compressor();
    test_paged_block_allocator();
    test_flash_decoding_scheduler();
    test_gpudirect_storage();
    printf("=== All Sandwich Dispatch Tests Passed Successfully! ===\n");
    return 0;
}
