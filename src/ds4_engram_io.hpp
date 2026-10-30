#pragma once

#include "llama.h"
#include "ggml.h"
#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <mutex>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>

#ifdef GGML_USE_IO_URING
#include <liburing.h>
#endif

// DeepSeek V4.1 Flash (GSQ_RCO / ISTA-DASLab) Engram Direct NVMe I/O Engine
// Manages sparse retrieval from the 111 GB external Engram memory table
// (DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf) directly on NVMe SSD with an
// in-memory hot row cache and asynchronous batch prefetching.

struct engram_row_entry {
    int64_t row_idx = -1;
    std::vector<uint8_t> data;
    uint64_t access_count = 0;
};

class engram_row_cache {
public:
    engram_row_cache(size_t max_rows = 32768, size_t row_bytes = 2048)
        : max_cached_rows(max_rows), row_size_bytes(row_bytes) {}

    bool lookup(int64_t row_idx, void * out_buf) {
        std::lock_guard<std::mutex> lock(cache_mutex);
        auto it = entries.find(row_idx);
        if (it != entries.end()) {
            it->second.access_count++;
            std::memcpy(out_buf, it->second.data.data(), row_size_bytes);
            hits++;
            return true;
        }
        misses++;
        return false;
    }

    void insert(int64_t row_idx, const void * src_buf) {
        std::lock_guard<std::mutex> lock(cache_mutex);
        if (entries.size() >= max_cached_rows) {
            // Evict lowest access count
            int64_t evict_key = -1;
            uint64_t min_count = UINT64_MAX;
            for (const auto & kv : entries) {
                if (kv.second.access_count < min_count) {
                    min_count = kv.second.access_count;
                    evict_key = kv.first;
                }
            }
            if (evict_key >= 0) {
                entries.erase(evict_key);
            }
        }
        auto & entry = entries[row_idx];
        entry.row_idx = row_idx;
        entry.data.resize(row_size_bytes);
        entry.access_count = 1;
        std::memcpy(entry.data.data(), src_buf, row_size_bytes);
    }

    size_t get_hits() const { return hits; }
    size_t get_misses() const { return misses; }
    size_t get_cached_count() const { return entries.size(); }

private:
    std::mutex cache_mutex;
    size_t max_cached_rows;
    size_t row_size_bytes;
    std::unordered_map<int64_t, engram_row_entry> entries;
    size_t hits = 0;
    size_t misses = 0;
};

class ds4_engram_io {
public:
    ds4_engram_io() = default;
    ~ds4_engram_io() {
        close();
    }

    bool open(const std::string & file_path, size_t row_bytes = 2048, size_t cache_rows = 32768) {
        std::lock_guard<std::mutex> lock(io_mutex);
        engram_path = file_path;
        row_size = row_bytes;
        row_cache = std::make_unique<engram_row_cache>(cache_rows, row_bytes);

        fd = ::open(file_path.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd < 0) {
            return false;
        }

        struct stat st;
        if (fstat(fd, &st) == 0) {
            total_file_bytes = st.st_size;
            total_rows = total_file_bytes / row_size;
        }

#ifdef GGML_USE_IO_URING
        if (io_uring_queue_init(32, &ring, 0) == 0) {
            uring_ready = true;
        }
#endif

        return true;
    }

    void close() {
        std::lock_guard<std::mutex> lock(io_mutex);
#ifdef GGML_USE_IO_URING
        if (uring_ready) {
            io_uring_queue_exit(&ring);
            uring_ready = false;
        }
#endif
        if (fd >= 0) {
            ::close(fd);
            fd = -1;
        }
        row_cache.reset();
    }

    bool is_open() const { return fd >= 0; }
    int64_t get_total_rows() const { return total_rows; }
    size_t get_row_bytes() const { return row_size; }

    // Read an n-gram row embedding (synchronous with row cache acceleration)
    bool read_row(int64_t row_idx, void * out_buf) {
        if (row_idx < 0 || row_idx >= total_rows || !out_buf) {
            return false;
        }

        if (row_cache && row_cache->lookup(row_idx, out_buf)) {
            return true; // Fast RAM hit
        }

        // Direct NVMe pread
        off_t offset = (off_t)row_idx * (off_t)row_size;
        ssize_t bytes_read = ::pread(fd, out_buf, row_size, offset);
        if (bytes_read == (ssize_t)row_size) {
            if (row_cache) {
                row_cache->insert(row_idx, out_buf);
            }
            return true;
        }

        return false;
    }

    // Prefetch a batch of n-gram rows into row cache
    void prefetch_rows(const std::vector<int64_t> & row_indices) {
        std::vector<uint8_t> temp_buf(row_size);
        for (int64_t r : row_indices) {
            read_row(r, temp_buf.data());
        }
    }

    // [Technique #61]: Fused unquantization / EXPDEC helper for IQ4_NL / GSQ_RCO rows
    static void fused_expdec_dot(
        const uint8_t * __restrict__ row_data,
        const float   * __restrict__ query_vec,
        float         * __restrict__ out_score,
        size_t                       dim) {
        float sum = 0.0f;
        for (size_t i = 0; i < dim; ++i) {
            uint8_t byte = row_data[i / 2];
            int8_t nibble = (i % 2 == 0) ? (byte & 0x0F) : (byte >> 4);
            float val = ((float)nibble - 8.0f) * 0.125f;
            sum += val * query_vec[i];
        }
        *out_score = sum;
    }

private:
    std::mutex io_mutex;
    std::string engram_path;
    int fd = -1;
    size_t row_size = 2048;
    int64_t total_rows = 0;
    size_t total_file_bytes = 0;
    std::unique_ptr<engram_row_cache> row_cache;

#ifdef GGML_USE_IO_URING
    struct io_uring ring;
    bool uring_ready = false;
#endif
};
