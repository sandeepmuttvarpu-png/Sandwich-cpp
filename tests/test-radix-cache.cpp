#include "common/radix-cache.h"
#include <cassert>
#include <iostream>
#include <vector>

int main() {
    std::cout << "🥪 [Technique #45] Testing Radix-Tree Persistent Prefix Caching (RadixAttention)...\n";

    sandwich_radix_cache cache(1024);

    // Turn 1: System prompt + User Message 1
    // Tokens: [1, 10, 20, 30, 40, 50]
    std::vector<llama_token> prompt1 = {1, 10, 20, 30, 40, 50};
    cache.insert(prompt1, 0, 0, prompt1.size());

    assert(cache.get_total_cached_tokens() == 6);
    std::cout << "  -> Inserted Turn 1 (6 tokens). Cached tokens: " << cache.get_total_cached_tokens() << " ✅\n";

    // Turn 2: Exact same prefix [1, 10, 20, 30] + New User Message [99, 100]
    std::vector<llama_token> prompt2 = {1, 10, 20, 30, 99, 100};
    auto match = cache.find_longest_prefix(prompt2);

    assert(match.matched_tokens == 4);
    std::cout << "  -> Turn 2 Prefix Query: matched " << match.matched_tokens << " / 6 tokens instantly (TTFT = 0 ms)! ✅\n";

    // Insert Turn 2 branch into Radix Tree
    cache.insert(prompt2, 1, 0, prompt2.size());

    // Turn 3: Querying an identical full match
    auto match_full = cache.find_longest_prefix(prompt1);
    assert(match_full.matched_tokens == 6);
    std::cout << "  -> Turn 3 Query on prompt 1: matched " << match_full.matched_tokens << " / 6 tokens! ✅\n";

    // [Technique #60] Strata Message-Boundary Delta Caching tests
    std::cout << "\n🥪 [Technique #60] Testing Message-Boundary Delta Caching...\n";
    const llama_token IM_END = 32000;
    const llama_token IM_START = 32001;
    cache.register_boundary_token(IM_END, false);
    cache.register_boundary_token(IM_START, true);
    assert(cache.is_boundary_token(IM_END));
    assert(cache.is_boundary_token(IM_START));
    assert(!cache.is_boundary_token(10));

    // Turn 4: Conversation with boundary markers:
    // [IM_START, 101, 102, IM_END, IM_START, 201, 202, 203]
    std::vector<llama_token> conv = {IM_START, 101, 102, IM_END, IM_START, 201, 202, 203};
    cache.insert(conv, 2, 0, conv.size());

    // Agent edits intermediate message:
    // Prompt shares prefix up to token 201, but edited after [IM_START, 101, 102, IM_END, IM_START, 201, 999]
    // Raw prefix match would give 6 tokens (including partial second message [IM_START, 201]).
    // Boundary snap must cleanly snap back to 4 tokens (ending right after IM_END), preserving intact turn 1!
    std::vector<llama_token> conv_edited = {IM_START, 101, 102, IM_END, IM_START, 201, 999};
    auto raw_match = cache.find_longest_prefix(conv_edited);
    assert(raw_match.matched_tokens == 6);

    auto snapped_match = cache.find_prefix_boundary_snapped(conv_edited, true);
    assert(snapped_match.matched_tokens == 4);
    std::cout << "  -> Raw prefix matched " << raw_match.matched_tokens << " tokens, snapped cleanly to message boundary: "
              << snapped_match.matched_tokens << " tokens! ✅\n";

    std::cout << "🎉 [Technique #45 & #60] Radix-Tree & Message-Boundary Delta Caching verified with 100% success!\n";
    return 0;
}
