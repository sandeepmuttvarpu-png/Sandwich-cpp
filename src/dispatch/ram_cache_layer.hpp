#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>
#include <unordered_map>
#include <mutex>

namespace sandwich {
namespace dispatch {

struct RAMCacheStats {
    uint64_t capacity_bytes      = 0;
    uint64_t used_bytes          = 0;
    uint64_t kv_blocks_cached    = 0;
    uint64_t expert_slices_cached= 0;
    size_t   cache_hits          = 0;
    size_t   cache_misses        = 0;
    size_t   evictions           = 0;
    double   hit_ratio           = 0.0;
    bool     thp_2mb_aligned     = true;
    bool     pinned_cuda_dma     = false;
};

class RAMCacheLayer {
public:
    explicit RAMCacheLayer(uint64_t capacity_mb = 2048);
    ~RAMCacheLayer();

    // Prevent copy
    RAMCacheLayer(const RAMCacheLayer &) = delete;
    RAMCacheLayer & operator=(const RAMCacheLayer &) = delete;

    // Initializes the host RAM cache buffer with 2MB THP alignment
    bool initialize();

    // Releases allocated host memory
    void release();

    // Query status and telemetry
    RAMCacheStats get_stats() const;

    // Registers a memory segment as pinned for zero-copy DMA streaming
    bool pin_buffer_dma(void * ptr, size_t size);

    // Unregisters pinned memory segment
    void unpin_buffer_dma(void * ptr);

    // Records cache access events for telemetry
    void record_hit(size_t bytes_transferred);
    void record_miss(size_t bytes_requested);
    void record_eviction(size_t bytes_freed);

private:
    uint64_t            m_capacity_bytes = 0;
    void *              m_raw_buffer     = nullptr;
    size_t              m_aligned_size   = 0;
    RAMCacheStats       m_stats;
    std::vector<void *> m_pinned_buffers;
    mutable std::mutex  m_mutex;
};

} // namespace dispatch
} // namespace sandwich
