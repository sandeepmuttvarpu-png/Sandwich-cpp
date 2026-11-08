#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <string>

namespace sandwich {
namespace dispatch {

// Technique #78 (Phase 16): PagedAttention / Dynamic Block Cache Allocation
// Divides KV cache memory into fixed-size physical blocks (B=16 or B=32),
// using virtual block translation tables to eliminate internal/external fragmentation
// and enable zero-copy branching (Copy-on-Write) for speculative drafter trees and multi-turn chat.

struct PagedBlockConfig {
    size_t tokens_per_block = 16;     // Standard PagedAttention block size (B=16 or 32)
    size_t total_blocks     = 4096;   // Max physical blocks in pool (4096 * 16 = 65,536 tokens)
};

struct PagedBlockStats {
    size_t total_blocks       = 0;
    size_t allocated_blocks   = 0;
    size_t free_blocks        = 0;
    size_t active_sequences   = 0;
    size_t shared_blocks      = 0;    // Number of blocks referenced by multiple sequences (CoW savings)
    double memory_utilization = 0.0;
    std::string diagnostics;
};

class PagedBlockAllocator {
public:
    explicit PagedBlockAllocator(const PagedBlockConfig & config = PagedBlockConfig{});

    // Reset allocator and free all physical blocks
    void reset();

    // Allocate an empty sequence and return its unique ID
    int32_t create_sequence();

    // Free a sequence and decrement reference counts of its physical blocks
    void free_sequence(int32_t seq_id);

    // Append N tokens to a sequence, allocating physical blocks on demand
    bool append_tokens(int32_t seq_id, size_t num_tokens);

    // Fork a sequence for speculative decoding / prefix sharing (Copy-on-Write)
    int32_t fork_sequence(int32_t src_seq_id);

    // Get the physical block mapping table for a sequence (passed to FlashAttention kernels)
    std::vector<int32_t> get_block_table(int32_t seq_id) const;

    // Get sequence token count
    size_t get_sequence_tokens(int32_t seq_id) const;

    // Query stats and telemetry
    PagedBlockStats get_stats() const;

private:
    PagedBlockConfig                                   m_config;
    std::vector<int32_t>                               m_free_list;
    std::vector<int32_t>                               m_ref_counts; // index = block_id
    std::unordered_map<int32_t, std::vector<int32_t>>  m_block_tables; // seq_id -> block IDs
    std::unordered_map<int32_t, size_t>                m_seq_token_counts;
    int32_t                                            m_next_seq_id = 1;
    mutable std::mutex                                 m_mutex;
};

} // namespace dispatch
} // namespace sandwich
