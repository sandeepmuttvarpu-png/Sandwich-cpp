#include "../src/llama-moe-cache-share.h"

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

    if (ok) {
        std::printf("test-moe-cache-shares: ALL PASS (including Phase 6 Fine-Grained MoE)\n");
        return 0;
    }
    return 1;
}
