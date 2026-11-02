#include "model_scanner.hpp"

#include <nlohmann/json.hpp>

#include <fstream>
#include <sstream>
#include <iostream>
#include <filesystem>
#include <chrono>
#include <algorithm>
#include <regex>
#include <thread>
#include <set>
#include <sys/stat.h>
#include <unistd.h>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace sandwich {
namespace dispatch {

static std::string get_config_dir() {
    const char * home = std::getenv("HOME");
    if (!home) {
        home = "/tmp";
    }
    fs::path config_dir = fs::path(home) / ".config" / "sandwich";
    std::error_code ec;
    fs::create_directories(config_dir, ec);
    return config_dir.string();
}

std::string ModelScanner::get_cache_path() {
    return (fs::path(get_config_dir()) / "model_cache.json").string();
}

bool ModelScanner::cache_exists() {
    std::error_code ec;
    return fs::exists(get_cache_path(), ec);
}

bool ModelScanner::is_cache_fresh(int max_age_seconds) {
    std::string path = get_cache_path();
    struct stat st;
    if (stat(path.c_str(), &st) != 0) {
        return false;
    }
    auto now = std::time(nullptr);
    return (now - st.st_mtime) < max_age_seconds;
}

std::string ModelCatalog::to_json() const {
    json j;
    j["scan_timestamp"] = scan_timestamp;
    j["scanned_paths"]   = scanned_paths;

    auto serialize_model = [](const ModelMetadata & m) {
        json item;
        item["file_path"]           = m.file_path;
        item["filename"]            = m.filename;
        item["display_name"]        = m.display_name;
        item["architecture"]        = m.architecture;
        item["quantization"]        = m.quantization;
        item["size_bytes"]          = m.size_bytes;
        item["size_gb"]             = m.size_gb;
        item["total_layers"]        = m.total_layers;
        item["expert_count"]        = m.expert_count;
        item["active_experts"]      = m.active_experts;
        item["context_length"]      = m.context_length;
        item["is_moe"]              = m.is_moe;
        item["is_drafter"]          = m.is_drafter;
        item["drafter_target"]      = m.drafter_target;
        item["last_modified_epoch"] = m.last_modified_epoch;
        return item;
    };

    json m_arr = json::array();
    for (const auto & m : models) {
        m_arr.push_back(serialize_model(m));
    }
    j["models"] = m_arr;

    json d_arr = json::array();
    for (const auto & d : drafters) {
        d_arr.push_back(serialize_model(d));
    }
    j["drafters"] = d_arr;

    return j.dump(4);
}

ModelCatalog ModelCatalog::from_json(const std::string & json_str) {
    ModelCatalog cat;
    try {
        json j = json::parse(json_str);
        cat.scan_timestamp = j.value("scan_timestamp", 0LL);
        if (j.contains("scanned_paths") && j["scanned_paths"].is_array()) {
            for (const auto & p : j["scanned_paths"]) {
                cat.scanned_paths.push_back(p.get<std::string>());
            }
        }

        auto deserialize_model = [](const json & item) {
            ModelMetadata m;
            m.file_path           = item.value("file_path", "");
            m.filename            = item.value("filename", "");
            m.display_name        = item.value("display_name", "");
            m.architecture        = item.value("architecture", "unknown");
            m.quantization        = item.value("quantization", "Q4_K_M");
            m.size_bytes          = item.value("size_bytes", 0ULL);
            m.size_gb             = item.value("size_gb", 0.0);
            m.total_layers        = item.value("total_layers", 32);
            m.expert_count        = item.value("expert_count", 0);
            m.active_experts      = item.value("active_experts", 0);
            m.context_length      = item.value("context_length", 4096);
            m.is_moe              = item.value("is_moe", false);
            m.is_drafter          = item.value("is_drafter", false);
            m.drafter_target      = item.value("drafter_target", "");
            m.last_modified_epoch = item.value("last_modified_epoch", 0LL);
            return m;
        };

        if (j.contains("models") && j["models"].is_array()) {
            for (const auto & item : j["models"]) {
                cat.models.push_back(deserialize_model(item));
            }
        }
        if (j.contains("drafters") && j["drafters"].is_array()) {
            for (const auto & item : j["drafters"]) {
                cat.drafters.push_back(deserialize_model(item));
            }
        }
    } catch (...) {
        // Return empty on parse failure
    }
    return cat;
}

std::vector<std::string> ModelScanner::get_default_scan_paths() {
    std::vector<std::string> paths;
    const char * home = std::getenv("HOME");
    if (home) {
        fs::path h(home);
        paths.push_back((h / "ai-models").string());
        paths.push_back((h / "models").string());
        paths.push_back((h / ".cache" / "huggingface" / "hub").string());
    }
    // Also include current working directory
    std::error_code ec;
    paths.push_back(fs::current_path(ec).string());
    return paths;
}

void ModelScanner::parse_heuristics_from_filename(const std::string & filename, ModelMetadata & meta) {
    std::string lower = filename;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

    // Quantization tags
    if (lower.find("iq3_xxs") != std::string::npos) meta.quantization = "IQ3_XXS";
    else if (lower.find("iq3_s") != std::string::npos) meta.quantization = "IQ3_S";
    else if (lower.find("iq2_xxs") != std::string::npos) meta.quantization = "IQ2_XXS";
    else if (lower.find("q4_k_m") != std::string::npos) meta.quantization = "Q4_K_M";
    else if (lower.find("q4_0") != std::string::npos) meta.quantization = "Q4_0";
    else if (lower.find("q8_0") != std::string::npos) meta.quantization = "Q8_0";
    else if (lower.find("q5_k_m") != std::string::npos) meta.quantization = "Q5_K_M";
    else if (lower.find("f16") != std::string::npos) meta.quantization = "F16";

    // MoE Detection
    if (lower.find("moe") != std::string::npos || lower.find("a3b") != std::string::npos || lower.find("flash-next") != std::string::npos) {
        meta.is_moe = true;
        if (lower.find("a3b") != std::string::npos) {
            meta.expert_count = 128;
            meta.active_experts = 8;
        } else if (lower.find("flash-next") != std::string::npos || lower.find("125b") != std::string::npos) {
            meta.expert_count = 128;
            meta.active_experts = 8;
        }
    }

    // Drafter / Speculative head detection
    if (lower.find("dflash") != std::string::npos || lower.find("mtp") != std::string::npos ||
        lower.find("drafter") != std::string::npos || lower.find("eagle") != std::string::npos) {
        meta.is_drafter = true;
        if (lower.find("qwen3.6-35b") != std::string::npos || lower.find("35b") != std::string::npos) {
            meta.drafter_target = "Qwen3.6-35B-A3B";
        } else if (lower.find("qwen3.8") != std::string::npos || lower.find("flash-next") != std::string::npos) {
            meta.drafter_target = "Qwen3.8-Flash-Next";
        }
    }

    // Architecture family
    if (lower.find("qwen") != std::string::npos) meta.architecture = "qwen2moe";
    else if (lower.find("llama") != std::string::npos) meta.architecture = "llama";
    else if (lower.find("deepseek") != std::string::npos) meta.architecture = "deepseek2";
    else if (lower.find("gemma") != std::string::npos) meta.architecture = "gemma";
    else if (lower.find("mistral") != std::string::npos) meta.architecture = "mistral";

    // Display Name Clean-up
    meta.display_name = filename;
    // Strip shard suffixes like -00001-of-00002.gguf or .gguf
    if (meta.display_name.size() > 5 && meta.display_name.substr(meta.display_name.size() - 5) == ".gguf") {
        meta.display_name = meta.display_name.substr(0, meta.display_name.size() - 5);
    }
}

bool ModelScanner::inspect_gguf_file(const std::string & file_path, ModelMetadata & out_meta) {
    std::error_code ec;
    uintmax_t sz = fs::file_size(file_path, ec);
    if (ec || sz < 1024) {
        return false;
    }

    out_meta.file_path = file_path;
    out_meta.filename  = fs::path(file_path).filename().string();
    out_meta.size_bytes = sz;
    out_meta.size_gb    = static_cast<double>(sz) / (1024.0 * 1024.0 * 1024.0);

    auto ftime = fs::last_write_time(file_path, ec);
    if (!ec) {
        // Convert file_time_type to epoch
        out_meta.last_modified_epoch = std::chrono::duration_cast<std::chrono::seconds>(ftime.time_since_epoch()).count();
    }

    // Binary read GGUF header
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        parse_heuristics_from_filename(out_meta.filename, out_meta);
        // If file is tiny (< 50 MB) and not explicitly marked as a drafter, filter out test vocabularies
    if (out_meta.size_bytes < 50 * 1024 * 1024 && !out_meta.is_drafter) {
        return false;
    }

    return true;
    }

    char magic[5] = {0};
    file.read(magic, 4);
    if (std::string(magic, 4) != "GGUF") {
        // Not a standard GGUF header
        return false;
    }

    uint32_t version = 0;
    file.read(reinterpret_cast<char *>(&version), sizeof(version));

    uint64_t tensor_count = 0;
    uint64_t kv_count = 0;
    file.read(reinterpret_cast<char *>(&tensor_count), sizeof(tensor_count));
    file.read(reinterpret_cast<char *>(&kv_count), sizeof(kv_count));

    // Apply baseline heuristics first
    parse_heuristics_from_filename(out_meta.filename, out_meta);

    // If small (< 3.0 GB) and filename hints, flag as drafter
    if (out_meta.size_gb < 3.5 && (out_meta.filename.find("DFlash") != std::string::npos ||
                                  out_meta.filename.find("mtp") != std::string::npos ||
                                  out_meta.filename.find("draft") != std::string::npos)) {
        out_meta.is_drafter = true;
    }

    // If file is tiny (< 50 MB) and not explicitly marked as a drafter, filter out test vocabularies
    if (out_meta.size_bytes < 50 * 1024 * 1024 && !out_meta.is_drafter) {
        return false;
    }

    return true;
}

ModelCatalog ModelScanner::run_full_scan(const std::vector<std::string> & custom_paths) {
    ModelCatalog cat;
    cat.scan_timestamp = std::time(nullptr);

    std::vector<std::string> targets = custom_paths.empty() ? get_default_scan_paths() : custom_paths;
    cat.scanned_paths = targets;

    std::set<std::string> visited_files;

    for (const auto & dir_path : targets) {
        std::error_code ec;
        if (!fs::exists(dir_path, ec) || !fs::is_directory(dir_path, ec)) {
            continue;
        }

        // Recursive directory search with error suppression for permissions
        for (auto it = fs::recursive_directory_iterator(dir_path, fs::directory_options::skip_permission_denied, ec);
             it != fs::recursive_directory_iterator(); it.increment(ec)) {
            if (ec) {
                ec.clear();
                continue;
            }

            const auto & entry = *it;
            if (entry.is_regular_file(ec) && entry.path().extension() == ".gguf") {
                std::string full_path = entry.path().string();
                if (visited_files.count(full_path)) {
                    continue;
                }
                visited_files.insert(full_path);

                ModelMetadata meta;
                if (inspect_gguf_file(full_path, meta)) {
                    if (meta.is_drafter) {
                        cat.drafters.push_back(meta);
                    } else {
                        cat.models.push_back(meta);
                    }
                }
            }
        }
    }

    // Sort models alphabetically by display name
    auto comp = [](const ModelMetadata & a, const ModelMetadata & b) {
        return a.display_name < b.display_name;
    };
    std::sort(cat.models.begin(), cat.models.end(), comp);
    std::sort(cat.drafters.begin(), cat.drafters.end(), comp);

    // Save to ~/.config/sandwich/model_cache.json
    std::string cache_p = get_cache_path();
    std::ofstream out(cache_p);
    if (out.is_open()) {
        out << cat.to_json() << std::endl;
        out.close();
    }

    return cat;
}

ModelCatalog ModelScanner::load_or_scan() {
    std::string cache_p = get_cache_path();
    if (cache_exists()) {
        std::ifstream in(cache_p);
        if (in.is_open()) {
            std::stringstream buffer;
            buffer << in.rdbuf();
            ModelCatalog cat = ModelCatalog::from_json(buffer.str());
            if (!cat.models.empty() || !cat.drafters.empty()) {
                // If cache is older than 24 hours, fire background thread to refresh cache asynchronously
                if (!is_cache_fresh(86400)) {
                    std::thread([]() {
                        ModelScanner::run_full_scan();
                    }).detach();
                }
                return cat;
            }
        }
    }

    // If cache missing or corrupt, run synchronous scan
    return run_full_scan();
}

} // namespace dispatch
} // namespace sandwich
