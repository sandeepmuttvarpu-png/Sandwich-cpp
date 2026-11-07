#include "ram_cache_layer.hpp"

#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <iostream>

#if defined(__has_include)
    #if __has_include(<sys/mman.h>)
        #include <sys/mman.h>
    #endif
    #if __has_include(<unistd.h>)
        #include <unistd.h>
    #endif
    #if __has_include(<cuda_runtime.h>)
        #define HAS_CUDA_DMA 1
        #include <cuda_runtime.h>
    #endif
#endif

#ifndef MADV_HUGEPAGE
#define MADV_HUGEPAGE 14
#endif

namespace sandwich {
namespace dispatch {

constexpr size_t THP_ALIGNMENT = 2 * 1024 * 1024; // 2MB Transparent HugePage boundary

RAMCacheLayer::RAMCacheLayer(uint64_t capacity_mb) {
    m_capacity_bytes = capacity_mb * 1024 * 1024;
    m_stats.capacity_bytes = m_capacity_bytes;
}

RAMCacheLayer::~RAMCacheLayer() {
    release();
}

bool RAMCacheLayer::initialize() {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_raw_buffer) return true; // already initialized
    if (m_capacity_bytes == 0) return false;

    // Round up size to 2MB boundary
    m_aligned_size = (m_capacity_bytes + THP_ALIGNMENT - 1) & ~(THP_ALIGNMENT - 1);

#if defined(MAP_ANONYMOUS) && defined(MAP_PRIVATE)
    m_raw_buffer = mmap(nullptr, m_aligned_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (m_raw_buffer == MAP_FAILED) {
        m_raw_buffer = nullptr;
    }
#endif

    if (!m_raw_buffer) {
        // Fallback to posix_memalign
        int ret = posix_memalign(&m_raw_buffer, THP_ALIGNMENT, m_aligned_size);
        if (ret != 0) {
            m_raw_buffer = nullptr;
            return false;
        }
    }

#if defined(MADV_HUGEPAGE)
    // Advise kernel to back this L2 RAM cache with 2MB hugepages to reduce TLB misses
    if (m_raw_buffer) {
        madvise(m_raw_buffer, m_aligned_size, MADV_HUGEPAGE);
        m_stats.thp_2mb_aligned = true;
    }
#endif

#if defined(HAS_CUDA_DMA)
    // Register host memory with CUDA driver for zero-copy pinned DMA transfers
    if (m_raw_buffer) {
        cudaError_t err = cudaHostRegister(m_raw_buffer, m_aligned_size, cudaHostRegisterDefault);
        if (err == cudaSuccess) {
            m_stats.pinned_cuda_dma = true;
            m_pinned_buffers.push_back(m_raw_buffer);
        }
    }
#endif

    return (m_raw_buffer != nullptr);
}

void RAMCacheLayer::release() {
    std::lock_guard<std::mutex> lock(m_mutex);

#if defined(HAS_CUDA_DMA)
    for (void * ptr : m_pinned_buffers) {
        cudaHostUnregister(ptr);
    }
    m_pinned_buffers.clear();
#endif

    if (m_raw_buffer) {
#if defined(MAP_ANONYMOUS)
        munmap(m_raw_buffer, m_aligned_size);
#else
        free(m_raw_buffer);
#endif
        m_raw_buffer = nullptr;
    }

    m_stats.capacity_bytes = 0;
    m_stats.used_bytes = 0;
    m_stats.pinned_cuda_dma = false;
}

RAMCacheStats RAMCacheLayer::get_stats() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    RAMCacheStats s = m_stats;
    size_t total_ops = s.cache_hits + s.cache_misses;
    s.hit_ratio = (total_ops > 0) ? (static_cast<double>(s.cache_hits) / total_ops) : 0.0;
    return s;
}

bool RAMCacheLayer::pin_buffer_dma(void * ptr, size_t size) {
    if (!ptr || size == 0) return false;
    std::lock_guard<std::mutex> lock(m_mutex);

#if defined(HAS_CUDA_DMA)
    cudaError_t err = cudaHostRegister(ptr, size, cudaHostRegisterDefault);
    if (err == cudaSuccess) {
        m_pinned_buffers.push_back(ptr);
        return true;
    }
#endif
    return false;
}

void RAMCacheLayer::unpin_buffer_dma(void * ptr) {
    if (!ptr) return;
    std::lock_guard<std::mutex> lock(m_mutex);

#if defined(HAS_CUDA_DMA)
    auto it = std::find(m_pinned_buffers.begin(), m_pinned_buffers.end(), ptr);
    if (it != m_pinned_buffers.end()) {
        cudaHostUnregister(ptr);
        m_pinned_buffers.erase(it);
    }
#endif
}

void RAMCacheLayer::record_hit(size_t bytes_transferred) {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_stats.cache_hits++;
}

void RAMCacheLayer::record_miss(size_t bytes_requested) {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_stats.cache_misses++;
}

void RAMCacheLayer::record_eviction(size_t bytes_freed) {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_stats.evictions++;
}

} // namespace dispatch
} // namespace sandwich
