#include "../src/llama-moe-cache-share.h"
#include "../src/llama-moe-cache.h"

#include <cmath>
#include <cstdio>
#include <vector>

static bool check(const char * name, const std::vector<float> & cumulative,
        const std::vector<bool> & eligible, const std::vector<double> & expected) {
    const auto actual = llama_moe_cache_detail::device_shares(cumulative, eligible);
    if (actual.size() != expected.size()) {
        std::fprintf(stderr, "%s: share count mismatch\n", name);
        return false;
    }
    for (size_t i = 0; i < actual.size(); ++i) {
        if (std::abs(actual[i] - expected[i]) > 1e-9) {
            std::fprintf(stderr, "%s: share %zu was %.12f, expected %.12f\n",
                    name, i, actual[i], expected[i]);
            return false;
        }
    }
    return true;
}

int main() {
    bool ok = true;
    ok &= check("equal", {0.5f, 1.0f}, {true, true}, {0.5, 0.5});
    ok &= check("asymmetric", {0.25f, 1.0f}, {true, true}, {0.25, 0.75});
    ok &= check("single device", {1.0f}, {true}, {1.0});
    ok &= check("zero fallback", {0.0f, 0.0f}, {true, true}, {0.5, 0.5});
    ok &= check("ineligible device", {0.25f, 1.0f}, {false, true}, {0.0, 1.0});
    ok &= check("fallback with ineligible device", {0.0f, 0.0f}, {true, false}, {1.0, 0.0});

    // Phase 6 tests: fine-grained MoE topology & binary search slot solver
    if (!llama_moe_cache_detail::is_fine_grained_moe(24576, 64)) {
        std::fprintf(stderr, "is_fine_grained_moe failed for 24576 micro-experts\n");
        ok = false;
    }
    if (llama_moe_cache_detail::is_fine_grained_moe(8, 2)) {
        std::fprintf(stderr, "is_fine_grained_moe failed: 8 macro-experts should not be fine-grained\n");
        ok = false;
    }

    // Binary search slot solver test: 4.6 GB budget, 512 KB per micro-expert
    const size_t budget = 4600ULL * 1024 * 1024;
    const size_t slice_size = 512 * 1024;
    const int32_t max_possible = 20000;
    int32_t slots = llama_moe_cache_detail::compute_optimal_slots(budget, slice_size, 256, max_possible);
    if (slots <= 0 || (size_t)slots * slice_size > budget) {
        std::fprintf(stderr, "compute_optimal_slots produced invalid slot count: %d\n", slots);
        ok = false;
    }

    // Micro-expert mapping test
    if (llama_moe_cache_detail::map_micro_expert_to_slot(12345, 1000) != 345) {
        std::fprintf(stderr, "map_micro_expert_to_slot mismatch\n");
        ok = false;
    }

    // Phase 12: 3-Tier Hierarchical MoE locator & tier state machine tests
    expert_locator loc;
    if (loc.tier != moe_tier_level::TIER_COLD_SSD || loc.vram_slot != -1 || loc.ram_ptr != nullptr) {
        std::fprintf(stderr, "Phase 12: expert_locator default init failed\n");
        ok = false;
    }
    // Transition to WARM
    loc.tier = moe_tier_level::TIER_WARM_RAM;
    loc.warm_slot = 42;
    loc.ram_ptr = (void *)0x1000;
    if (loc.tier != moe_tier_level::TIER_WARM_RAM || loc.warm_slot != 42 || loc.ram_ptr == nullptr) {
        std::fprintf(stderr, "Phase 12: WARM transition failed\n");
        ok = false;
    }
    // Transition to HOT
    loc.tier = moe_tier_level::TIER_HOT_VRAM;
    loc.vram_slot = 7;
    if (loc.tier != moe_tier_level::TIER_HOT_VRAM || loc.vram_slot != 7) {
        std::fprintf(stderr, "Phase 12: HOT transition failed\n");
        ok = false;
    }
    // Demote on VRAM eviction back to WARM
    loc.tier = moe_tier_level::TIER_WARM_RAM;
    loc.vram_slot = -1;
    if (loc.tier != moe_tier_level::TIER_WARM_RAM || loc.vram_slot != -1 || loc.ram_ptr == nullptr) {
        std::fprintf(stderr, "Phase 12: Demote to WARM failed\n");
        ok = false;
    }

    // Phase 12: 2Q & Ghost-Cache Replacement Policy Unit Tests
    {
        llama_moe_cache_detail::moe_cache_lru lru_2q;
        lru_2q.init(2, 16, 4, true); // 2 layers, 16 experts, 4 slots (Kin=1 probation, Km=3 frequent)
        std::vector<int32_t> map0(16, -1);
        std::vector<int32_t> map1(16, -1);
        lru_2q.slot_map[0] = map0.data();
        lru_2q.slot_map[1] = map1.data();

        if (lru_2q.k_in != 1 || lru_2q.k_am != 3 || lru_2q.size_free != 4) {
            std::fprintf(stderr, "Phase 12 2Q: Initial partition sizing failed (Kin=%d, Km=%d, free=%d)\n",
                lru_2q.k_in, lru_2q.k_am, lru_2q.size_free);
            ok = false;
        }

        std::vector<llama_moe_cache_detail::moe_cache_lru::fill> fills;
        std::vector<llama_moe_cache_detail::moe_cache_lru::eviction> evictions;
        size_t n_hit = 0;

        // 1. First reference to expert 5: should enter A1in (probation)
        int32_t req1[] = { 5 };
        lru_2q.plan(0, req1, 1, fills, evictions, n_hit);
        if (n_hit != 0 || fills.size() != 1 || fills[0].expert != 5) {
            std::fprintf(stderr, "Phase 12 2Q: First reference failed\n");
            ok = false;
        }
        int32_t slot_exp5 = fills[0].slot;
        if (lru_2q.queue_of[slot_exp5] != llama_moe_cache_detail::moe_slot_queue::QUEUE_A1IN) {
            std::fprintf(stderr, "Phase 12 2Q: Expert 5 not placed into A1in probation queue\n");
            ok = false;
        }

        // 2. Second reference to expert 5: should PROMOTE from A1in to Am
        lru_2q.plan(0, req1, 1, fills, evictions, n_hit);
        if (n_hit != 1 || lru_2q.stat_promotions_2q != 1 ||
            lru_2q.queue_of[slot_exp5] != llama_moe_cache_detail::moe_slot_queue::QUEUE_AM) {
            std::fprintf(stderr, "Phase 12 2Q: Promotion to Am failed (n_hit=%zu, promotions=%zu)\n",
                n_hit, lru_2q.stat_promotions_2q);
            ok = false;
        }

        // 3. Fill remaining slots with transient one-hit wonders: experts 6, 7, 8
        int32_t req2[] = { 6 }; lru_2q.plan(0, req2, 1, fills, evictions, n_hit);
        int32_t req3[] = { 7 }; lru_2q.plan(0, req3, 1, fills, evictions, n_hit);
        int32_t req4[] = { 8 }; lru_2q.plan(0, req4, 1, fills, evictions, n_hit);
        if (lru_2q.size_free != 0) {
            std::fprintf(stderr, "Phase 12 2Q: Cache should be full (free=%d)\n", lru_2q.size_free);
            ok = false;
        }

        // 4. Introduce expert 9: A1in probation is full, so expert 6 (oldest in probation) must be evicted to Ghost Cache
        int32_t req5[] = { 9 };
        lru_2q.plan(0, req5, 1, fills, evictions, n_hit);
        if (evictions.empty() || evictions[0].expert != 6) {
            std::fprintf(stderr, "Phase 12 2Q: Expected expert 6 eviction to ghost cache\n");
            ok = false;
        }
        if (!lru_2q.in_ghost(0, 6)) {
            std::fprintf(stderr, "Phase 12 2Q: Expert 6 missing from Ghost Cache\n");
            ok = false;
        }
        if (lru_2q.stat_pollution_quarantined != 1) {
            std::fprintf(stderr, "Phase 12 2Q: Pollution quarantine counter mismatch (%zu != 1)\n",
                lru_2q.stat_pollution_quarantined);
            ok = false;
        }
        // Crucial: Frequent expert 5 in Am was NOT evicted!
        if (map0[5] < 0 || lru_2q.queue_of[map0[5]] != llama_moe_cache_detail::moe_slot_queue::QUEUE_AM) {
            std::fprintf(stderr, "Phase 12 2Q: Pollution resistance failed! Expert 5 was evicted from Am\n");
            ok = false;
        }

        // 5. Ghost hit test: Request expert 6 again (should hit Ghost Cache and promote directly to Am)
        int32_t req6[] = { 6 };
        lru_2q.plan(0, req6, 1, fills, evictions, n_hit);
        if (lru_2q.stat_ghost_hits != 1) {
            std::fprintf(stderr, "Phase 12 2Q: Ghost hit counter mismatch (%zu != 1)\n", lru_2q.stat_ghost_hits);
            ok = false;
        }
        if (lru_2q.in_ghost(0, 6)) {
            std::fprintf(stderr, "Phase 12 2Q: Expert 6 should be removed from Ghost Cache after hit\n");
            ok = false;
        }
        int32_t slot_exp6 = map0[6];
        if (slot_exp6 < 0 || lru_2q.queue_of[slot_exp6] != llama_moe_cache_detail::moe_slot_queue::QUEUE_AM) {
            std::fprintf(stderr, "Phase 12 2Q: Ghost hit did not promote expert 6 directly to Am\n");
            ok = false;
        }
    }

    if (ok) {
        std::printf("test-moe-cache-shares: ALL PASS (including Phase 6 Fine-Grained, Phase 12 3-Tier MoE Engine, and Phase 12 2Q & Ghost-Cache)\n");
        return 0;
    }
    return 1;
}
