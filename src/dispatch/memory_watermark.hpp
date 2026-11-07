#pragma once

#include "ram_estimator.hpp"
#include <cstdint>
#include <string>
#include <functional>
#include <mutex>

namespace sandwich {
namespace dispatch {

enum class WatermarkLevel {
    NORMAL,            // Memory usage within safe operational envelope
    PRESSURE_WARNING,  // Approaching High Watermark (HWM): throttle prefetch depth
    CRITICAL_THROTTLE  // Exceeded High Watermark (HWM): evict probation slabs, clamp chunks
};

struct WatermarkThresholds {
    double vram_hwm_ratio   = 0.90; // 90% max VRAM allocation
    double vram_lwm_ratio   = 0.75; // 75% safe recovery point
    uint64_t vram_min_headroom_mb = 750; // Minimum 750 MB cushion for Wayland compositor

    double ram_hwm_ratio    = 0.85; // 85% max Host RAM allocation
    double ram_lwm_ratio    = 0.70; // 70% safe recovery point (Technique #42)
    uint64_t ram_min_headroom_mb  = 2048; // Minimum 2,048 MB cushion for OS & page cache
};

struct WatermarkTelemetry {
    WatermarkLevel current_level = WatermarkLevel::NORMAL;
    uint64_t vram_free_mb        = 0;
    uint64_t vram_safe_budget_mb = 0;
    uint64_t ram_available_mb    = 0;
    uint64_t ram_safe_budget_mb  = 0;
    size_t   throttle_events     = 0;
    size_t   recovery_events     = 0;
    std::string diagnostics_msg;
};

class MemoryWatermarker {
public:
    explicit MemoryWatermarker(const WatermarkThresholds & thresholds = WatermarkThresholds{});

    // Evaluates real-time system memory and updates watermark level with hysteresis
    WatermarkLevel update_realtime(const RealtimeRAMStats & ram, const RealtimeVRAMStats & vram);

    // Queries current telemetry and level
    WatermarkTelemetry get_telemetry() const;

    // Direct checks
    bool is_throttled() const;
    bool is_critical() const;

    // Computes dynamic safe budget taking into account live system watermarks
    uint64_t get_dynamic_vram_limit_mb(uint64_t total_vram_mb) const;
    uint64_t get_dynamic_ram_limit_mb(uint64_t total_ram_mb) const;

    // Recommended prefill chunk size adaptation based on current watermark pressure
    uint32_t adapt_prefill_chunk_size(uint32_t base_chunk_size) const;

    // Register callback for when memory exceeds critical threshold
    using EvictionCallback = std::function<void(WatermarkLevel)>;
    void set_pressure_callback(EvictionCallback cb);

private:
    WatermarkThresholds m_thresholds;
    WatermarkLevel      m_current_level = WatermarkLevel::NORMAL;
    size_t              m_throttle_count = 0;
    size_t              m_recovery_count = 0;
    EvictionCallback    m_callback;
    mutable std::mutex  m_mutex;
};

} // namespace dispatch
} // namespace sandwich
