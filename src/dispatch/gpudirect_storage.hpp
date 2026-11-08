#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>
#include <mutex>

namespace sandwich {
namespace dispatch {

// Technique #80 (Phase 16) / Technique #48: GPUDirect Storage (cuFile) Direct NVMe -> VRAM DMA Engine
// Bypasses CPU bounce buffers and page-cache lock contention by streaming GGUF weight tensors
// and offloaded KV/MoE slices directly from NVMe SSD into GPU VRAM (or pinned host memory fallback)
// with hardware-aligned 4KB Direct I/O DMA transfers.

enum class GDSBackendMode {
    DISABLED,               // GDS explicitly turned off
    DIRECT_CUFILE,          // Native NVIDIA cuFile kernel-bypass DMA directly to VRAM
    COMPAT_IO_URING_DIRECT, // Linux io_uring asynchronous O_DIRECT DMA to pinned staging
    COMPAT_POSIX_DIRECT     // POSIX aligned O_DIRECT pread DMA fallback
};

struct GDSConfig {
    bool   enabled             = true;
    size_t block_alignment     = 4096; // NVMe hardware sector alignment (4KB)
    size_t max_batch_size      = 32;   // Max simultaneous DMA transfers
    bool   force_compatibility = false;// Force compat fallback mode (for testing / systems without nvidia-fs)
};

struct GDSStatus {
    bool           enabled            = false;
    GDSBackendMode active_backend     = GDSBackendMode::DISABLED;
    bool           has_cufile_driver  = false; // /dev/cufile or nvidia-fs driver present
    bool           has_io_uring       = false; // Linux io_uring available
    uint64_t       total_transfers    = 0;
    uint64_t       total_bytes        = 0;
    double         peak_bandwidth_gbs = 0.0;
    std::string    backend_name;
    std::string    diagnostics;
};

struct GDSTransferRequest {
    int         fd          = -1;     // File descriptor of the weights file (open with O_RDONLY / O_DIRECT)
    uint64_t    file_offset = 0;      // Byte offset within file
    size_t      length      = 0;      // Number of bytes to transfer
    void *      dest_ptr    = nullptr;// GPU device pointer (for cuFile) or pinned host buffer
    bool        is_gpu_vram = false;  // True if dest_ptr is VRAM
};

struct GDSTransferResult {
    bool        success           = false;
    size_t      bytes_transferred = 0;
    double      elapsed_ms        = 0.0;
    double      bandwidth_gbs     = 0.0;
    std::string error_message;
};

class GDSManager {
public:
    explicit GDSManager(const GDSConfig & config = GDSConfig{});
    ~GDSManager();

    // Check hardware & kernel driver capabilities
    void probe_capabilities();

    // Initialize the engine with config
    bool init();

    // Shutdown and release handles
    void shutdown();

    // Check if GDS is active and functioning
    bool is_active() const;

    // Get current status and transfer counters
    GDSStatus get_status() const;

    // Execute a synchronous or pipelined direct DMA transfer
    GDSTransferResult transfer(const GDSTransferRequest & req);

    // Batch transfer execution for multi-expert or multi-head DMA
    std::vector<GDSTransferResult> transfer_batch(const std::vector<GDSTransferRequest> & reqs);

    // Reset performance counters
    void reset_metrics();

    // Format human-readable diagnostics
    std::string get_diagnostics() const;

    // Utility: calculate 4KB aligned offset and length padding
    static void align_bounds(
        uint64_t offset,
        size_t length,
        size_t alignment,
        uint64_t & aligned_offset,
        size_t & aligned_length,
        size_t & offset_padding
    );

private:
    GDSConfig          config_;
    mutable std::mutex mutex_;
    GDSStatus          status_;
    bool               initialized_ = false;

    // Internal execution helpers
    GDSTransferResult execute_cufile(const GDSTransferRequest & req);
    GDSTransferResult execute_io_uring(const GDSTransferRequest & req);
    GDSTransferResult execute_posix_direct(const GDSTransferRequest & req);
};

} // namespace dispatch
} // namespace sandwich
