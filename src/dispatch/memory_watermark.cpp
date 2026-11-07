#include "memory_watermark.hpp"
#include <algorithm>
#include <sstream>

namespace sandwich {
namespace dispatch {

MemoryWatermarker::MemoryWatermarker(const WatermarkThresholds & thresholds)
    : m_thresholds(thresholds) {}

WatermarkLevel MemoryWatermarker::update_realtime(const RealtimeRAMStats & ram, const RealtimeVRAMStats & vram) {
    std::lock_guard<std::mutex> lock(m_mutex);

    bool vram_critical = false;
    bool vram_warning  = false;
    bool ram_critical  = false;
    bool ram_warning   = false;

    // 1. Evaluate VRAM pressure if CUDA available
    if (vram.cuda_available && vram.total_mb > 0) {
        double vram_used_ratio = static_cast<double>(vram.used_mb) / vram.total_mb;
        if (vram_used_ratio >= m_thresholds.vram_hwm_ratio || vram.free_mb < m_thresholds.vram_min_headroom_mb) {
            vram_critical = true;
        } else if (vram_used_ratio >= (m_thresholds.vram_lwm_ratio + 0.05)) {
            vram_warning = true;
        }
    }

    // 2. Evaluate Host RAM pressure
    if (ram.total_mb > 0) {
        double ram_used_ratio = ram.used_percentage / 100.0;
        if (ram_used_ratio >= m_thresholds.ram_hwm_ratio || ram.available_mb < m_thresholds.ram_min_headroom_mb) {
            ram_critical = true;
        } else if (ram_used_ratio >= (m_thresholds.ram_lwm_ratio + 0.05)) {
            ram_warning = true;
        }
    }

    WatermarkLevel new_level = m_current_level;

    if (vram_critical || ram_critical) {
        if (m_current_level != WatermarkLevel::CRITICAL_THROTTLE) {
            m_throttle_count++;
        }
        new_level = WatermarkLevel::CRITICAL_THROTTLE;
    } else if (vram_warning || ram_warning) {
        new_level = WatermarkLevel::PRESSURE_WARNING;
    } else {
        // Hysteresis recovery check: must be strictly below LWM to return to NORMAL
        bool vram_recovered = true;
        if (vram.cuda_available && vram.total_mb > 0) {
            double vram_used_ratio = static_cast<double>(vram.used_mb) / vram.total_mb;
            vram_recovered = (vram_used_ratio <= m_thresholds.vram_lwm_ratio && vram.free_mb >= m_thresholds.vram_min_headroom_mb + 500);
        }

        bool ram_recovered = true;
        if (ram.total_mb > 0) {
            double ram_used_ratio = ram.used_percentage / 100.0;
            ram_recovered = (ram_used_ratio <= m_thresholds.ram_lwm_ratio && ram.available_mb >= m_thresholds.ram_min_headroom_mb + 1000);
        }

        if (vram_recovered && ram_recovered) {
            if (m_current_level != WatermarkLevel::NORMAL) {
                m_recovery_count++;
            }
            new_level = WatermarkLevel::NORMAL;
        }
    }

    m_current_level = new_level;

    if (m_current_level == WatermarkLevel::CRITICAL_THROTTLE && m_callback) {
        m_callback(m_current_level);
    }

    return m_current_level;
}

WatermarkTelemetry MemoryWatermarker::get_telemetry() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    WatermarkTelemetry tel;
    tel.current_level = m_current_level;
    tel.throttle_events = m_throttle_count;
    tel.recovery_events = m_recovery_count;

    std::ostringstream ss;
    switch (m_current_level) {
        case WatermarkLevel::NORMAL:
            ss << "NORMAL [Headroom Safe]";
            break;
        case WatermarkLevel::PRESSURE_WARNING:
            ss << "WARNING [Approaching HWM: Throttling Lookahead]";
            break;
        case WatermarkLevel::CRITICAL_THROTTLE:
            ss << "CRITICAL [Above HWM: Proactive Eviction Active]";
            break;
    }
    tel.diagnostics_msg = ss.str();
    return tel;
}

bool MemoryWatermarker::is_throttled() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_current_level != WatermarkLevel::NORMAL;
}

bool MemoryWatermarker::is_critical() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_current_level == WatermarkLevel::CRITICAL_THROTTLE;
}

uint64_t MemoryWatermarker::get_dynamic_vram_limit_mb(uint64_t total_vram_mb) const {
    if (total_vram_mb == 0) return 0;
    // Always preserve at least min headroom for Wayland compositor
    uint64_t max_usable = (total_vram_mb > m_thresholds.vram_min_headroom_mb)
                            ? (total_vram_mb - m_thresholds.vram_min_headroom_mb)
                            : (total_vram_mb / 2);
    uint64_t ratio_limit = static_cast<uint64_t>(total_vram_mb * m_thresholds.vram_hwm_ratio);
    return std::min(max_usable, ratio_limit);
}

uint64_t MemoryWatermarker::get_dynamic_ram_limit_mb(uint64_t total_ram_mb) const {
    if (total_ram_mb == 0) return 0;
    // Enforce 70% RAM Rule #42 as ceiling, leaving OS cushion
    uint64_t max_usable = (total_ram_mb > m_thresholds.ram_min_headroom_mb)
                            ? (total_ram_mb - m_thresholds.ram_min_headroom_mb)
                            : (total_ram_mb / 2);
    uint64_t ratio_limit = static_cast<uint64_t>(total_ram_mb * m_thresholds.ram_lwm_ratio);
    return std::min(max_usable, ratio_limit);
}

uint32_t MemoryWatermarker::adapt_prefill_chunk_size(uint32_t base_chunk_size) const {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_current_level == WatermarkLevel::CRITICAL_THROTTLE) {
        // Clamp chunk size aggressively to prevent transient VRAM spike
        return std::max<uint32_t>(32, base_chunk_size / 4);
    } else if (m_current_level == WatermarkLevel::PRESSURE_WARNING) {
        return std::max<uint32_t>(64, base_chunk_size / 2);
    }
    return base_chunk_size;
}

void MemoryWatermarker::set_pressure_callback(EvictionCallback cb) {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_callback = std::move(cb);
}

} // namespace dispatch
} // namespace sandwich
