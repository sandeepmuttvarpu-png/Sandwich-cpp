#include "hardware_profiler.hpp"
#include "ram_estimator.hpp"
#include "memory_watermark.hpp"
#include "prefetch_scheduler.hpp"

#include <nlohmann/json.hpp>

#include <fstream>
#include <sstream>
#include <iostream>
#include <filesystem>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <set>
#include <map>
#include <algorithm>
#include <cstring>
#include <unistd.h>
#include <sys/utsname.h>
#include <sys/sysinfo.h>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace sandwich {
namespace dispatch {

std::string HardwareProfiler::get_config_dir() {
    const char * home = std::getenv("HOME");
    if (!home) {
        home = "/tmp";
    }
    fs::path config_dir = fs::path(home) / ".config" / "sandwich";
    std::error_code ec;
    fs::create_directories(config_dir, ec);
    return config_dir.string();
}

std::string HardwareProfiler::get_profile_path() {
    return (fs::path(get_config_dir()) / "hardware_profile.json").string();
}

std::string HardwareProfiler::get_flag_path() {
    return (fs::path(get_config_dir()) / "profile_exists.flag").string();
}

bool HardwareProfiler::profile_exists() {
    std::error_code ec;
    return fs::exists(get_flag_path(), ec) && fs::exists(get_profile_path(), ec);
}

HardwareProfile HardwareProfiler::get_conservative_fallback() {
    HardwareProfile fb;
    fb.timestamp = "fallback-default";
    fb.os.pretty_name = "Linux (Conservative Fallback)";
    fb.os.id = "linux";
    fb.os.kernel_release = "unknown";
    fb.os.is_wayland = true;

    fb.cpu.model_name = "x86_64 CPU (Safe Mode)";
    fb.cpu.total_logical_cores = 8;
    fb.cpu.p_cores = FALLBACK_THREADS;
    fb.cpu.e_cores = 0;
    fb.cpu.optimal_threads = FALLBACK_THREADS;
    fb.cpu.max_freq_ghz = 2.4;

    fb.ram.total_mb = FALLBACK_RAM_MB;
    fb.ram.available_mb = FALLBACK_RAM_MB / 2;
    fb.ram.max_safe_allocation_mb = static_cast<uint64_t>(FALLBACK_RAM_MB * RAM_RULE_42_RATIO);
    fb.ram.est_bandwidth_gbps = 48.0;

    fb.gpu.device_name = "GPU (Safe Mode)";
    fb.gpu.total_vram_mb = FALLBACK_VRAM_MB;
    fb.gpu.free_vram_mb = FALLBACK_VRAM_MB;
    fb.gpu.wayland_guard_vram_mb = std::min(FALLBACK_VRAM_MB, WAYLAND_VRAM_GUARD_MB);
    fb.gpu.est_bandwidth_gbps = 150.0;
    fb.gpu.cuda_available = false;

    fb.storage.nvme_model = "Storage Fallback";
    fb.storage.pcie_link_speed = "Gen3";
    fb.storage.pcie_link_width = 4;
    fb.storage.est_read_bandwidth_mbps = 2500.0;

    return fb;
}

std::string HardwareProfile::to_json() const {
    json j;
    j["timestamp"] = timestamp;

    j["os"] = {
        {"pretty_name", os.pretty_name},
        {"id",          os.id},
        {"kernel",      os.kernel_release},
        {"is_wayland",  os.is_wayland}
    };

    j["cpu"] = {
        {"model_name",          cpu.model_name},
        {"total_logical_cores", cpu.total_logical_cores},
        {"p_cores",             cpu.p_cores},
        {"e_cores",             cpu.e_cores},
        {"optimal_threads",     cpu.optimal_threads},
        {"max_freq_ghz",        cpu.max_freq_ghz}
    };

    j["ram"] = {
        {"total_mb",               ram.total_mb},
        {"available_mb",           ram.available_mb},
        {"max_safe_allocation_mb", ram.max_safe_allocation_mb},
        {"safety_ratio",           RAM_RULE_42_RATIO},
        {"est_bandwidth_gbps",     ram.est_bandwidth_gbps}
    };

    j["gpu"] = {
        {"device_name",           gpu.device_name},
        {"total_vram_mb",         gpu.total_vram_mb},
        {"free_vram_mb",          gpu.free_vram_mb},
        {"wayland_guard_vram_mb", gpu.wayland_guard_vram_mb},
        {"est_bandwidth_gbps",    gpu.est_bandwidth_gbps},
        {"cuda_available",        gpu.cuda_available}
    };

    j["storage"] = {
        {"nvme_model",              storage.nvme_model},
        {"pcie_link_speed",         storage.pcie_link_speed},
        {"pcie_link_width",         storage.pcie_link_width},
        {"est_read_bandwidth_mbps", storage.est_read_bandwidth_mbps}
    };

    j["invariants"] = {
        {"wayland_vram_guard_mb", WAYLAND_VRAM_GUARD_MB},
        {"ram_rule_42_ratio",     RAM_RULE_42_RATIO}
    };

    return j.dump(4);
}

HardwareProfile HardwareProfile::from_json(const std::string & json_str) {
    HardwareProfile p;
    try {
        json j = json::parse(json_str);
        p.timestamp = j.value("timestamp", "unknown");

        if (j.contains("os")) {
            p.os.pretty_name    = j["os"].value("pretty_name", "Linux");
            p.os.id             = j["os"].value("id", "linux");
            p.os.kernel_release = j["os"].value("kernel", "");
            p.os.is_wayland     = j["os"].value("is_wayland", false);
        }

        if (j.contains("cpu")) {
            p.cpu.model_name          = j["cpu"].value("model_name", "Unknown CPU");
            p.cpu.total_logical_cores = j["cpu"].value("total_logical_cores", 8);
            p.cpu.p_cores             = j["cpu"].value("p_cores", 4);
            p.cpu.e_cores             = j["cpu"].value("e_cores", 0);
            p.cpu.optimal_threads     = j["cpu"].value("optimal_threads", 4);
            p.cpu.max_freq_ghz        = j["cpu"].value("max_freq_ghz", 2.5);
        }

        if (j.contains("ram")) {
            p.ram.total_mb               = j["ram"].value("total_mb", FALLBACK_RAM_MB);
            p.ram.available_mb           = j["ram"].value("available_mb", FALLBACK_RAM_MB / 2);
            p.ram.max_safe_allocation_mb = j["ram"].value("max_safe_allocation_mb", static_cast<uint64_t>(FALLBACK_RAM_MB * RAM_RULE_42_RATIO));
            p.ram.est_bandwidth_gbps     = j["ram"].value("est_bandwidth_gbps", 64.0);
        }

        if (j.contains("gpu")) {
            p.gpu.device_name           = j["gpu"].value("device_name", "GPU");
            p.gpu.total_vram_mb         = j["gpu"].value("total_vram_mb", FALLBACK_VRAM_MB);
            p.gpu.free_vram_mb          = j["gpu"].value("free_vram_mb", FALLBACK_VRAM_MB);
            p.gpu.wayland_guard_vram_mb = std::min(static_cast<uint64_t>(j["gpu"].value("wayland_guard_vram_mb", WAYLAND_VRAM_GUARD_MB)), WAYLAND_VRAM_GUARD_MB);
            p.gpu.est_bandwidth_gbps    = j["gpu"].value("est_bandwidth_gbps", 224.0);
            p.gpu.cuda_available        = j["gpu"].value("cuda_available", true);
        }

        if (j.contains("storage")) {
            p.storage.nvme_model              = j["storage"].value("nvme_model", "NVMe SSD");
            p.storage.pcie_link_speed         = j["storage"].value("pcie_link_speed", "Gen4");
            p.storage.pcie_link_width         = j["storage"].value("pcie_link_width", 4);
            p.storage.est_read_bandwidth_mbps = j["storage"].value("est_read_bandwidth_mbps", 5000.0);
        }

        return p;
    } catch (...) {
        return HardwareProfiler::get_conservative_fallback();
    }
}

OSProfile HardwareProfiler::detect_os() {
    OSProfile os;
    struct utsname uts;
    if (uname(&uts) == 0) {
        os.kernel_release = uts.release;
    }

    std::ifstream f("/etc/os-release");
    if (f.is_open()) {
        std::string line;
        while (std::getline(f, line)) {
            if (line.rfind("PRETTY_NAME=", 0) == 0) {
                std::string val = line.substr(12);
                if (val.size() >= 2 && val.front() == '"' && val.back() == '"') {
                    val = val.substr(1, val.size() - 2);
                }
                os.pretty_name = val;
            } else if (line.rfind("ID=", 0) == 0) {
                std::string val = line.substr(3);
                if (val.size() >= 2 && val.front() == '"' && val.back() == '"') {
                    val = val.substr(1, val.size() - 2);
                }
                os.id = val;
            }
        }
    }

    const char * wayland = std::getenv("WAYLAND_DISPLAY");
    const char * xdg_session = std::getenv("XDG_SESSION_TYPE");
    if (wayland || (xdg_session && std::string(xdg_session) == "wayland")) {
        os.is_wayland = true;
    }

    return os;
}

CPUProfile HardwareProfiler::detect_cpu() {
    CPUProfile cpu;

    // 1. Read /proc/cpuinfo for model name & total processors
    std::ifstream f("/proc/cpuinfo");
    if (f.is_open()) {
        std::string line;
        int proc_count = 0;
        while (std::getline(f, line)) {
            if (line.rfind("model name", 0) == 0) {
                auto pos = line.find(':');
                if (pos != std::string::npos && cpu.model_name == "Generic x86_64") {
                    std::string m = line.substr(pos + 1);
                    while (!m.empty() && (m.front() == ' ' || m.front() == '\t')) m.erase(0, 1);
                    cpu.model_name = m;
                }
            } else if (line.rfind("processor", 0) == 0) {
                proc_count++;
            }
        }
        if (proc_count > 0) {
            cpu.total_logical_cores = proc_count;
        }
    }

    // 2. Topology Detection: Identify P-cores (hyperthreaded) vs E-cores (single-threaded)
    // On Intel Hybrid (12th/13th/14th Gen), P-cores have 2 thread siblings (e.g., 0-1),
    // while E-cores have 1 thread sibling (e.g., 12).
    std::set<std::string> p_core_groups;
    int e_core_count = 0;

    for (int i = 0; i < cpu.total_logical_cores; ++i) {
        std::string path = "/sys/devices/system/cpu/cpu" + std::to_string(i) + "/topology/thread_siblings_list";
        std::ifstream sf(path);
        if (sf.is_open()) {
            std::string sib;
            std::getline(sf, sib);
            if (sib.find('-') != std::string::npos || sib.find(',') != std::string::npos) {
                p_core_groups.insert(sib);
            } else {
                e_core_count++;
            }
        }
    }

    if (!p_core_groups.empty()) {
        cpu.p_cores = static_cast<int>(p_core_groups.size());
        cpu.e_cores = e_core_count;
        // Optimal threads: lock precisely to physical P-cores to eliminate thread migration overhead
        cpu.optimal_threads = cpu.p_cores;
    } else {
        // Fallback for non-hybrid architectures (e.g. standard AMD or pure cores)
        cpu.p_cores = std::max(1, cpu.total_logical_cores / 2);
        cpu.e_cores = 0;
        cpu.optimal_threads = cpu.p_cores;
    }

    // 3. Max frequency
    std::ifstream freq_f("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq");
    if (freq_f.is_open()) {
        long long khz = 0;
        if (freq_f >> khz && khz > 0) {
            cpu.max_freq_ghz = khz / 1000000.0;
        }
    }

    return cpu;
}

RAMProfile HardwareProfiler::detect_ram() {
    RAMProfile ram;
    RealtimeRAMStats s = RAMEstimator::sample_realtime_ram();
    ram.total_mb = s.total_mb;
    ram.available_mb = s.available_mb;
    ram.max_safe_allocation_mb = static_cast<uint64_t>(ram.total_mb * RAM_RULE_42_RATIO);
    ram.est_bandwidth_gbps = 64.0; // Typical DDR5 4800/5600 MT/s dual-channel
    return ram;
}

GPUProfile HardwareProfiler::detect_gpu() {
    GPUProfile gpu;
    RealtimeVRAMStats s = RAMEstimator::sample_realtime_vram();
    gpu.cuda_available = s.cuda_available;
    gpu.device_name    = s.device_name;
    gpu.total_vram_mb  = s.total_mb;
    gpu.free_vram_mb   = s.free_mb;

    // Dynamic Universal VRAM Guard & Display Topology Detection
    // 1. Probes if discrete GPU drives active display via DRM sysfs.
    //    If driving display: leaves 800MB - 1GB cushion for desktop compositor (Wayland/X11).
    //    If headless or secondary compute device (e.g. laptop hybrid iGPU mode): leaves 350-400MB cushion for cuBLAS/CUDA workspace.
    bool dgpu_has_display = false;
    std::error_code drm_ec;
    if (fs::exists("/sys/class/drm", drm_ec)) {
        for (const auto & entry : fs::directory_iterator("/sys/class/drm", drm_ec)) {
            std::string name = entry.path().filename().string();
            if (name.find("card") != std::string::npos && name.find("-") != std::string::npos) {
                std::ifstream sf(entry.path() / "status");
                std::string status;
                if (sf >> status && status == "connected") {
                    std::string dev_link = fs::canonical(entry.path() / "device", drm_ec).string();
                    if (dev_link.find("01:00.0") != std::string::npos || dev_link.find("nvidia") != std::string::npos) {
                        dgpu_has_display = true;
                        break;
                    }
                }
            }
        }
    }

    uint64_t vram_cushion = dgpu_has_display
        ? std::max<uint64_t>(800, static_cast<uint64_t>(gpu.total_vram_mb * 0.08))
        : std::max<uint64_t>(350, static_cast<uint64_t>(gpu.total_vram_mb * 0.045));

    gpu.wayland_guard_vram_mb = (gpu.total_vram_mb > vram_cushion) ? (gpu.total_vram_mb - vram_cushion) : gpu.total_vram_mb;

    // Estimate bandwidth
    if (gpu.device_name.find("5050") != std::string::npos) {
        gpu.est_bandwidth_gbps = 224.0; // GDDR6 128-bit mobile
    } else if (gpu.device_name.find("4060") != std::string::npos || gpu.device_name.find("4070") != std::string::npos) {
        gpu.est_bandwidth_gbps = 272.0;
    } else {
        gpu.est_bandwidth_gbps = 200.0;
    }

    return gpu;
}

StorageProfile HardwareProfiler::detect_storage() {
    StorageProfile st;

    // Check sysfs for NVMe model
    std::ifstream mf("/sys/block/nvme0n1/device/model");
    if (mf.is_open()) {
        std::string m;
        std::getline(mf, m);
        while (!m.empty() && (m.back() == ' ' || m.back() == '\n' || m.back() == '\r')) m.pop_back();
        if (!m.empty()) {
            st.nvme_model = m;
        }
    }

    // Check GPU PCIe link speed in sysfs (0000:01:00.0 is typical discrete GPU bus)
    std::ifstream lsf("/sys/bus/pci/devices/0000:01:00.0/max_link_speed");
    if (lsf.is_open()) {
        std::string sp;
        std::getline(lsf, sp);
        while (!sp.empty() && (sp.back() == ' ' || sp.back() == '\n' || sp.back() == '\r')) sp.pop_back();
        if (!sp.empty()) {
            st.pcie_link_speed = sp;
        }
    }

    std::ifstream lwf("/sys/bus/pci/devices/0000:01:00.0/max_link_width");
    if (lwf.is_open()) {
        int width = 0;
        if (lwf >> width && width > 0) {
            st.pcie_link_width = width;
        }
    }

    // PCIe Gen4 x4 typically achieves 5,000 - 7,000 MB/s sequential
    if (st.pcie_link_speed.find("32.0") != std::string::npos || st.pcie_link_speed.find("16.0") != std::string::npos) {
        st.est_read_bandwidth_mbps = 6000.0;
    } else {
        st.est_read_bandwidth_mbps = 3200.0;
    }

    return st;
}

HardwareProfile HardwareProfiler::run_profiling(bool force) {
    if (!force && profile_exists()) {
        return load_or_profile();
    }

    HardwareProfile p;

    // Timestamp
    auto now = std::chrono::system_clock::now();
    std::time_t tt = std::chrono::system_clock::to_time_t(now);
    std::tm tm_buf;
    localtime_r(&tt, &tm_buf);
    std::ostringstream ss;
    ss << std::put_time(&tm_buf, "%Y-%m-%dT%H:%M:%S%z");
    p.timestamp = ss.str();

    try {
        p.os      = detect_os();
        p.cpu     = detect_cpu();
        p.ram     = detect_ram();
        p.gpu     = detect_gpu();
        p.storage = detect_storage();
    } catch (...) {
        p = get_conservative_fallback();
    }

    // Write hardware_profile.json
    std::string json_str = p.to_json();
    std::string p_path = get_profile_path();
    std::ofstream out(p_path);
    if (out.is_open()) {
        out << json_str << std::endl;
        out.close();
    }

    // Touch profile_exists.flag
    std::string f_path = get_flag_path();
    std::ofstream flag_out(f_path);
    if (flag_out.is_open()) {
        flag_out << "INITIALIZED=" << p.timestamp << std::endl;
        flag_out.close();
    }

    return p;
}

HardwareProfile HardwareProfiler::load_or_profile() {
    std::string p_path = get_profile_path();
    std::ifstream in(p_path);
    if (in.is_open()) {
        std::stringstream buffer;
        buffer << in.rdbuf();
        try {
            return HardwareProfile::from_json(buffer.str());
        } catch (...) {
            // Corrupted profile: force re-profiling
            return run_profiling(true);
        }
    }
    return run_profiling(true);
}

TieringPlan HardwareProfiler::calculate_tiering(
    const HardwareProfile & profile,
    uint64_t model_bytes,
    int total_layers,
    bool is_moe,
    int expert_count,
    int active_experts,
    int context_length,
    const std::string & kv_quant
) {
    TieringPlan plan;
    plan.model_total_mb = model_bytes / (1024 * 1024);
    plan.total_layers = (total_layers > 0) ? total_layers : 32;

    // 1. Real-time Live RAM & VRAM Sampling
    RealtimeRAMStats live_ram = RAMEstimator::sample_realtime_ram();
    RealtimeVRAMStats live_vram = RAMEstimator::sample_realtime_vram();

    // 2. Exact GGUF Footprint Estimation
    ModelMemoryFootprint fp = RAMEstimator::calculate_footprint(
        model_bytes,
        plan.total_layers,
        context_length,
        kv_quant,
        is_moe,
        expert_count,
        active_experts
    );

    // 3. Dynamic Dispatch Recommendation with Real-Time Watermarking
    DynamicDispatchRecommendation rec = RAMEstimator::compute_realtime_dispatch(
        fp,
        live_ram,
        live_vram,
        is_moe,
        context_length,
        profile.gpu.wayland_guard_vram_mb
    );

    // 4. Memory Tiering Assignment
    if (rec.recommended_gpu_layers == 999) {
        // Fits 100% in VRAM
        plan.tier1_hot_vram_mb  = fp.total_weights_mb;
        plan.tier2_warm_ddr5_mb = 0;
        plan.tier3_cold_nvme_mb = 0;
        plan.gpu_layers         = plan.total_layers;
    } else {
        // Multi-tier Sandwich Partitioning
        plan.gpu_layers = std::min(plan.total_layers, rec.recommended_gpu_layers);
        plan.tier1_hot_vram_mb = static_cast<uint64_t>(plan.gpu_layers * fp.mb_per_dense_layer);
        plan.tier2_warm_ddr5_mb = rec.warm_ddr5_pool_mb;

        uint64_t allocated_mb = plan.tier1_hot_vram_mb + plan.tier2_warm_ddr5_mb;
        plan.tier3_cold_nvme_mb = (fp.total_weights_mb > allocated_mb) ? (fp.total_weights_mb - allocated_mb) : 0;
    }

    // 5. Scenario Simulation (Tokens / Second)
    double active_fraction = 1.0;
    if (is_moe && expert_count > 0 && active_experts > 0) {
        double dense_ratio = 0.20;
        double moe_ratio = 0.80;
        active_fraction = dense_ratio + (moe_ratio * static_cast<double>(active_experts) / expert_count);
    }
    double active_model_mb = plan.model_total_mb * active_fraction;

    double gpu_bw_mbps = profile.gpu.est_bandwidth_gbps * 1024.0;
    plan.est_tokens_sec_best = std::min(45.0, (gpu_bw_mbps / (active_model_mb + 100.0)) * 0.28);

    double ddr5_bw_mbps = profile.ram.est_bandwidth_gbps * 1024.0;
    plan.est_tokens_sec_hybrid = std::min(22.0, (ddr5_bw_mbps / (active_model_mb * 0.75 + 100.0)) * 0.32);

    double nvme_bw_mbps = profile.storage.est_read_bandwidth_mbps;
    double cold_expert_mb_per_tok = (plan.tier3_cold_nvme_mb > 0) ? (active_model_mb * 0.30) : 50.0;
    plan.est_tokens_sec_worst = std::max(2.5, (nvme_bw_mbps / cold_expert_mb_per_tok) * 0.20);

    // 6. Synthesize Recommended CLI Flags
    std::ostringstream flags;
    if (plan.tier2_warm_ddr5_mb == 0 && plan.tier3_cold_nvme_mb == 0) {
        flags << "-ngl 99 ";
    } else {
        uint64_t free_margin_mb = (profile.gpu.total_vram_mb > profile.gpu.wayland_guard_vram_mb)
                                    ? (profile.gpu.total_vram_mb - profile.gpu.wayland_guard_vram_mb)
                                    : 1024;
        flags << "-fitt " << free_margin_mb << " ";
    }
    flags << "-t " << profile.cpu.optimal_threads << " ";
    flags << "-c " << context_length << " ";
    flags << "-fa ";
    if (kv_quant == "turbo4") {
        flags << "-ctk turbo4 -ctv turbo4 ";
    } else if (kv_quant == "turbo-kvran") {
        flags << "-ctk turbo4 -ctv turbo4 --turbo-kvran ";
    } else if (kv_quant == "turbo2") {
        flags << "-ctk turbo2 -ctv turbo2 ";
    } else if (kv_quant == "q8_0") {
        flags << "-ctk q8_0 -ctv q8_0 ";
    } else if (kv_quant == "f16") {
        flags << "-ctk f16 -ctv f16 ";
    }

    if (rec.enable_chunked_prefill) {
        flags << "--chunked-prefill -ub " << rec.prefill_chunk_size << " ";
    }
    if (rec.enable_asym_prefetch) {
        flags << "--asym-prefetch ";
    }
    if (rec.enable_ram_cache) {
        flags << "--ram-cache-mb " << rec.ram_cache_budget_mb << " ";
    }
    flags << "--dynamic-watermark ";

    plan.recommended_flags = flags.str();
    return plan;
}

} // namespace dispatch
} // namespace sandwich
