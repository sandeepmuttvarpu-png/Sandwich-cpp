#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <memory>

namespace sandwich {
namespace dispatch {

// Invariant Constants
constexpr uint64_t WAYLAND_VRAM_GUARD_MB = 7100; // Hard invariant: 7,100 MB max VRAM allocation
constexpr double   RAM_RULE_42_RATIO     = 0.70; // Hard invariant: Technique #42 (70% Max Host RAM)
constexpr int      FALLBACK_THREADS      = 4;
constexpr uint64_t FALLBACK_VRAM_MB      = 4096;
constexpr uint64_t FALLBACK_RAM_MB       = 16384;

struct CPUProfile {
    std::string model_name           = "Generic x86_64";
    int         total_logical_cores  = 8;
    int         p_cores              = 4;  // Performance cores
    int         e_cores              = 0;  // Efficiency cores
    int         optimal_threads      = 4;  // Bound to physical P-cores
    double      max_freq_ghz         = 2.5;
};

struct RAMProfile {
    uint64_t total_mb               = FALLBACK_RAM_MB;
    uint64_t available_mb           = FALLBACK_RAM_MB / 2;
    uint64_t max_safe_allocation_mb = static_cast<uint64_t>(FALLBACK_RAM_MB * RAM_RULE_42_RATIO);
    double   est_bandwidth_gbps     = 64.0; // DDR5 typical dual-channel bandwidth
};

struct GPUProfile {
    std::string device_name            = "NVIDIA GPU (Autodetected)";
    uint64_t    total_vram_mb          = FALLBACK_VRAM_MB;
    uint64_t    free_vram_mb           = FALLBACK_VRAM_MB;
    uint64_t    wayland_guard_vram_mb  = std::min(FALLBACK_VRAM_MB, WAYLAND_VRAM_GUARD_MB);
    double      est_bandwidth_gbps     = 224.0; // GDDR6 mobile bus bandwidth
    bool        cuda_available         = false;
};

struct StorageProfile {
    std::string nvme_model             = "Standard NVMe / SSD";
    std::string pcie_link_speed        = "16.0 GT/s PCIe";
    int         pcie_link_width        = 4;
    double      est_read_bandwidth_mbps = 5000.0; // Gen4 x4 sequential
};

struct OSProfile {
    std::string pretty_name            = "Linux";
    std::string id                     = "linux";
    std::string kernel_release         = "";
    bool        is_wayland             = false;
};

struct HardwareProfile {
    std::string    timestamp;
    OSProfile      os;
    CPUProfile     cpu;
    RAMProfile     ram;
    GPUProfile     gpu;
    StorageProfile storage;

    std::string to_json() const;
    static HardwareProfile from_json(const std::string & json_str);
    static HardwareProfile get_conservative_fallback();
};

struct TieringPlan {
    uint64_t model_total_mb       = 0;
    uint64_t tier1_hot_vram_mb    = 0; // GPU VRAM (capped by WAYLAND_VRAM_GUARD_MB)
    uint64_t tier2_warm_ddr5_mb   = 0; // Host RAM (capped by 70% RAM Rule)
    uint64_t tier3_cold_nvme_mb   = 0; // Direct NVMe mmap paging
    int      gpu_layers           = 0;
    int      total_layers         = 0;
    double   est_tokens_sec_best  = 0.0;
    double   est_tokens_sec_hybrid= 0.0;
    double   est_tokens_sec_worst = 0.0;
    std::string recommended_flags;
};

class HardwareProfiler {
public:
    static bool profile_exists();
    static std::string get_profile_path();
    static std::string get_flag_path();
    static HardwareProfile get_conservative_fallback();

    // Runs full hardware profiling and writes to ~/.config/sandwich/hardware_profile.json
    static HardwareProfile run_profiling(bool force = false);

    // Loads cached profile, or runs profiling if missing or corrupted
    static HardwareProfile load_or_profile();

    // Calculates 3-tier memory allocation & throughput simulation for a given model
    static TieringPlan calculate_tiering(
        const HardwareProfile & profile,
        uint64_t model_bytes,
        int total_layers,
        bool is_moe,
        int expert_count = 0,
        int active_experts = 0,
        int context_length = 4096,
        const std::string & kv_quant = "turbo3"
    );

private:
    static OSProfile      detect_os();
    static CPUProfile     detect_cpu();
    static RAMProfile     detect_ram();
    static GPUProfile     detect_gpu();
    static StorageProfile detect_storage();
};

} // namespace dispatch
} // namespace sandwich
