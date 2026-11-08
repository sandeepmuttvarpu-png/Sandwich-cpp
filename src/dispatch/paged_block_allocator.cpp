#include "paged_block_allocator.hpp"

#include <algorithm>
#include <sstream>
#include <iomanip>

namespace sandwich {
namespace dispatch {

PagedBlockAllocator::PagedBlockAllocator(const PagedBlockConfig & config)
    : m_config(config) {
    reset();
}

void PagedBlockAllocator::reset() {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_free_list.clear();
    m_free_list.reserve(m_config.total_blocks);
    for (int32_t i = static_cast<int32_t>(m_config.total_blocks) - 1; i >= 0; --i) {
        m_free_list.push_back(i);
    }
    m_ref_counts.assign(m_config.total_blocks, 0);
    m_block_tables.clear();
    m_seq_token_counts.clear();
    m_next_seq_id = 1;
}

int32_t PagedBlockAllocator::create_sequence() {
    std::lock_guard<std::mutex> lock(m_mutex);
    int32_t seq_id = m_next_seq_id++;
    m_block_tables[seq_id] = {};
    m_seq_token_counts[seq_id] = 0;
    return seq_id;
}

void PagedBlockAllocator::free_sequence(int32_t seq_id) {
    std::lock_guard<std::mutex> lock(m_mutex);
    auto it = m_block_tables.find(seq_id);
    if (it == m_block_tables.end()) {
        return;
    }

    for (int32_t block_id : it->second) {
        if (block_id >= 0 && block_id < static_cast<int32_t>(m_ref_counts.size())) {
            m_ref_counts[block_id]--;
            if (m_ref_counts[block_id] <= 0) {
                m_ref_counts[block_id] = 0;
                m_free_list.push_back(block_id);
            }
        }
    }

    m_block_tables.erase(it);
    m_seq_token_counts.erase(seq_id);
}

bool PagedBlockAllocator::append_tokens(int32_t seq_id, size_t num_tokens) {
    std::lock_guard<std::mutex> lock(m_mutex);
    auto it = m_block_tables.find(seq_id);
    if (it == m_block_tables.end()) {
        return false;
    }

    size_t cur_tokens = m_seq_token_counts[seq_id];
    size_t new_tokens = cur_tokens + num_tokens;
    size_t req_blocks = (new_tokens + m_config.tokens_per_block - 1) / m_config.tokens_per_block;
    size_t cur_blocks = it->second.size();

    if (req_blocks > cur_blocks) {
        size_t blocks_to_add = req_blocks - cur_blocks;
        if (m_free_list.size() < blocks_to_add) {
            return false; // Out of physical blocks
        }

        // Check if last block is shared and needs Copy-on-Write
        if (cur_blocks > 0) {
            int32_t last_blk = it->second.back();
            if (m_ref_counts[last_blk] > 1 && (cur_tokens % m_config.tokens_per_block) != 0) {
                // Must unshare last block
                int32_t new_blk = m_free_list.back();
                m_free_list.pop_back();
                m_ref_counts[last_blk]--;
                m_ref_counts[new_blk] = 1;
                it->second.back() = new_blk;
            }
        }

        for (size_t b = 0; b < blocks_to_add; ++b) {
            int32_t blk = m_free_list.back();
            m_free_list.pop_back();
            m_ref_counts[blk] = 1;
            it->second.push_back(blk);
        }
    }

    m_seq_token_counts[seq_id] = new_tokens;
    return true;
}

int32_t PagedBlockAllocator::fork_sequence(int32_t src_seq_id) {
    std::lock_guard<std::mutex> lock(m_mutex);
    auto it = m_block_tables.find(src_seq_id);
    if (it == m_block_tables.end()) {
        return -1;
    }

    int32_t dst_seq_id = m_next_seq_id++;
    m_block_tables[dst_seq_id] = it->second;
    m_seq_token_counts[dst_seq_id] = m_seq_token_counts[src_seq_id];

    // Increment reference count for all shared physical blocks (Copy-on-Write)
    for (int32_t block_id : it->second) {
        if (block_id >= 0 && block_id < static_cast<int32_t>(m_ref_counts.size())) {
            m_ref_counts[block_id]++;
        }
    }

    return dst_seq_id;
}

std::vector<int32_t> PagedBlockAllocator::get_block_table(int32_t seq_id) const {
    std::lock_guard<std::mutex> lock(m_mutex);
    auto it = m_block_tables.find(seq_id);
    if (it != m_block_tables.end()) {
        return it->second;
    }
    return {};
}

size_t PagedBlockAllocator::get_sequence_tokens(int32_t seq_id) const {
    std::lock_guard<std::mutex> lock(m_mutex);
    auto it = m_seq_token_counts.find(seq_id);
    if (it != m_seq_token_counts.end()) {
        return it->second;
    }
    return 0;
}

PagedBlockStats PagedBlockAllocator::get_stats() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    PagedBlockStats st;
    st.total_blocks     = m_config.total_blocks;
    st.free_blocks      = m_free_list.size();
    st.allocated_blocks = (st.total_blocks > st.free_blocks) ? (st.total_blocks - st.free_blocks) : 0;
    st.active_sequences = m_block_tables.size();

    size_t shared = 0;
    for (int32_t count : m_ref_counts) {
        if (count > 1) {
            shared++;
        }
    }
    st.shared_blocks = shared;

    if (st.total_blocks > 0) {
        st.memory_utilization = static_cast<double>(st.allocated_blocks) / static_cast<double>(st.total_blocks);
    }

    std::ostringstream ss;
    ss << "PagedAttention [Blocks: " << st.allocated_blocks << "/" << st.total_blocks
       << " (" << std::fixed << std::setprecision(1) << (st.memory_utilization * 100.0) << "% used)"
       << ", Sequences: " << st.active_sequences
       << ", CoW Shared Blocks: " << st.shared_blocks << "]";
    st.diagnostics = ss.str();

    return st;
}

} // namespace dispatch
} // namespace sandwich
