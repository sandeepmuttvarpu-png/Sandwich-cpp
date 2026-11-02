#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <memory>
#include <future>

namespace sandwich {
namespace dispatch {

struct ModelMetadata {
    std::string file_path;
    std::string filename;
    std::string display_name;
    std::string architecture        = "unknown";
    std::string quantization        = "Q4_K_M";
    uint64_t    size_bytes          = 0;
    double      size_gb             = 0.0;
    int         total_layers        = 32;
    int         expert_count        = 0;
    int         active_experts      = 0;
    int         context_length      = 4096;
    bool        is_moe              = false;
    bool        is_drafter          = false;
    std::string drafter_target      = "";
    int64_t     last_modified_epoch = 0;
};

struct ModelCatalog {
    int64_t                    scan_timestamp = 0;
    std::vector<std::string>   scanned_paths;
    std::vector<ModelMetadata> models;
    std::vector<ModelMetadata> drafters;

    std::string to_json() const;
    static ModelCatalog from_json(const std::string & json_str);
};

class ModelScanner {
public:
    static std::string get_cache_path();
    static bool cache_exists();
    static bool is_cache_fresh(int max_age_seconds = 86400); // 24 hours

    // High-level method: loads cache instantly if present; if stale, kicks off async rescan
    static ModelCatalog load_or_scan();

    // Performs full directory scan and updates ~/.config/sandwich/model_cache.json
    static ModelCatalog run_full_scan(const std::vector<std::string> & custom_paths = {});

    // Inspects a single GGUF file header and extracts metadata
    static bool inspect_gguf_file(const std::string & file_path, ModelMetadata & out_meta);

private:
    static std::vector<std::string> get_default_scan_paths();
    static void parse_heuristics_from_filename(const std::string & filename, ModelMetadata & meta);
};

} // namespace dispatch
} // namespace sandwich
