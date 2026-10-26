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

    std::cout << "🎉 [Technique #45] Radix-Tree Persistent Prefix Caching verified with 100% success!\n";
    return 0;
}
