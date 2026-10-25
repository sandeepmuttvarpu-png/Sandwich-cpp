#pragma once

#include "ggml-backend.h"

#include <map>
#include <memory>
#include <vector>

struct llama_model;

// Phase 12: 3-Tier Hierarchical MoE Memory System
enum class moe_tier_level : uint8_t {
    TIER_HOT_VRAM  = 0, // In GPU VRAM bank (zero-copy compute)
    TIER_WARM_RAM  = 1, // In pinned DDR5 host RAM (ready for PCIe DMA)
    TIER_COLD_SSD  = 2  // On NVMe disk (requires io_uring fetch)
};

struct expert_locator {
    moe_tier_level tier        = moe_tier_level::TIER_COLD_SSD;
    int32_t        vram_slot   = -1;      // index in Tier 1 VRAM bank (-1 if not in VRAM)
    int32_t        warm_slot   = -1;      // index in Tier 2 WARM pool (-1 if not in RAM)
    void *         ram_ptr     = nullptr; // pointer in Tier 2 WARM RAM pool (nullptr if on disk)
    size_t         file_offset = 0;       // file offset on NVMe SSD
    size_t         bytes       = 0;       // size in bytes
};

// keeps the most recently used experts of host-resident MoE layers in a device buffer
// each layer has a slot map in host memory: when the scheduler copies it to the device, the copy callback uploads the missing experts
class llama_moe_cache {
public:
    // backends are all the backends of the context, each GPU gets its own cache of the given size for the layers assigned to it
    llama_moe_cache(const llama_model & model,
                    const std::vector<ggml_backend_t> & backends,
                    const std::vector<ggml_backend_buffer_type_t> & bufts,
                    size_t size,
                    size_t warm_ram_bytes = 0,
                    bool   moe_3tier = false,
                    bool   moe_2q = true);
    ~llama_moe_cache();

    // the device that caches layer il
    ggml_backend_t backend(int32_t il) const;

    // the slot map of layer il, if its experts can be read from the cache for n_tokens tokens, nullptr otherwise
    ggml_tensor * get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used) const;

    // the experts of w in the cache, nullptr if w is not cached
    ggml_tensor * get_experts(const ggml_tensor * w) const;

    // ggml_backend_sched copy callback, returns false if src is not a slot map
    bool copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph);

    // for large batches: copy the experts of w that are in the cache, starting at expert e and up to expert last, to the copy dst of w
    // returns the number of experts copied, 0 if expert e is not in the cache
    int64_t copy_experts(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t e, int64_t last);

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const;

    // Phase 5: Query pinned host memory DMA status and capacity
    bool is_pinned_dma_enabled() const;
    size_t pinned_buffer_size() const;

    // Phase 6: Query fine-grained micro-expert status and total slot capacity
    bool is_fine_grained() const;
    int32_t total_cached_slots() const;

    // Phase 12: 3-Tier Hierarchical MoE Cache Engine
    const expert_locator * get_locator(int32_t il, int32_t expert) const;
    moe_tier_level         get_tier(int32_t il, int32_t expert) const;
    size_t                 warm_pool_size() const;
    size_t                 warm_pool_used() const;
    size_t                 cold_fetches() const;
    bool                   is_3tier_enabled() const;

    // Phase 12: 2Q & Ghost-Cache Replacement Policy Telemetry
    bool   is_2q_enabled() const;
    size_t ghost_hits() const;
    size_t promotions_2q() const;
    size_t pollution_quarantined() const;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

using llama_moe_cache_ptr = std::unique_ptr<llama_moe_cache>;
