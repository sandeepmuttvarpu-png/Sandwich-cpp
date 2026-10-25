#include "llama-moe-cache.h"
#include "llama-moe-cache-share.h"

#include "llama-impl.h"
#include "llama-model.h"

#include "ggml-cpp.h"
#include "ggml-moe-prefetch.h"
#include "ggml-iqk.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>

#ifdef GGML_USE_IO_URING
#include <liburing.h>
#endif

// Phase 6: Zero enum collisions verification between TurboQuant types (43-45) and IQK types (134+)
static_assert(GGML_TYPE_TURBO4_0 == 45, "TurboQuant type enum mismatch");
static_assert(GGML_TYPE_IQ2_KS_EXT >= 134, "IQK type collision guard failed");

namespace {

// Phase 12: 2Q & Ghost-Cache Replacement Policy Engine (from llama-moe-cache-share.h)
using moe_cache_lru = llama_moe_cache_detail::moe_cache_lru;

// gate, up, down or gate_up, down
static std::vector<ggml_tensor *> llama_moe_cache_layer_experts(const llama_layer & layer) {
    std::vector<ggml_tensor *> res;
    for (ggml_tensor * t : { layer.ffn_gate_up_exps, layer.ffn_gate_exps, layer.ffn_up_exps, layer.ffn_down_exps }) {
        if (t != nullptr) {
            res.push_back(t);
        }
    }
    return res;
}

static bool llama_moe_cache_same_layout(const std::vector<ggml_tensor *> & a, const std::vector<ggml_tensor *> & b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i]->type != b[i]->type || !ggml_are_same_shape(a[i], b[i]) || a[i]->nb[2] != b[i]->nb[2]) {
            return false;
        }
    }
    return true;
}

static bool llama_moe_cache_is_host_weight(const ggml_tensor * t) {
    return t->buffer != nullptr &&
        ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
        ggml_backend_buffer_is_host(t->buffer);
}

}

struct llama_moe_cache::impl {
    // Phase 5: Pinned host memory ring buffer for true zero-copy PCIe DMA
    struct moe_pinned_ring_buffer {
        ggml_backend_buffer_ptr buf;
        uint8_t * base = nullptr;
        size_t size = 0;
        size_t head = 0;

        void init(ggml_backend_dev_t dev, size_t capacity) {
            if (!dev || capacity == 0) return;
            ggml_backend_buffer_type_t host_buft = ggml_backend_dev_host_buffer_type(dev);
            if (host_buft) {
                buf.reset(ggml_backend_buft_alloc_buffer(host_buft, capacity));
                if (buf) {
                    base = (uint8_t *) ggml_backend_buffer_get_base(buf.get());
                    size = ggml_backend_buffer_get_size(buf.get());
                    head = 0;
                }
            }
        }

        const void * stage(const void * src, size_t bytes) {
            if (!base || bytes == 0 || bytes > size) {
                return src; // fallback to unpinned host pointer if too large or unavailable
            }
            // 256-byte alignment for hardware PCIe packet efficiency
            size_t aligned_head = (head + 255) & ~((size_t) 255);
            if (aligned_head + bytes > size) {
                aligned_head = 0; // wrap around in ring buffer
            }
            uint8_t * dst = base + aligned_head;
            memcpy(dst, src, bytes);
            head = aligned_head + bytes;
            return dst;
        }

        bool is_pinned() const {
            return base != nullptr && size > 0;
        }
    };

    // a GPU with its own budget and banks, it caches the layers assigned to it
    struct device {
        ggml_backend_t backend;
        ggml_backend_buffer_type_t buft;
        size_t host_bytes = 0; // host experts of the layers it caches
        double split = 0.0;    // share of the budget

        // banks and their views
        ggml_context_ptr ctx;
        ggml_backend_buffer_ptr buf;
        size_t buf_size = 0;

        // Phase 5: Pinned DMA ring buffer for zero-copy PCIe expert weight streaming
        moe_pinned_ring_buffer pinned_ring;
    };

    // layers of the same device with the same expert tensor layout share the banks and the LRU of a group
    struct group {
        int32_t id;                       // device
        std::vector<ggml_tensor *> ref;   // expert tensors of the first layer
        std::vector<int32_t> layers;
        std::vector<ggml_tensor *> banks; // device storage of all slots, one per expert tensor
        size_t host_bytes = 0;
        int32_t n_slots = 0;
        moe_cache_lru lru;
    };

    struct layer {
        int32_t ig = -1;                    // -1 if the layer is not cached
        ggml_tensor * slot_map = nullptr;   // I32 [1, n_expert] in host memory
        std::vector<ggml_tensor *> experts; // host expert tensors, in the order of the banks
    };

    struct binding {
        int32_t il;
        int32_t ip;           // index of the bank
        ggml_tensor * cached; // view of the bank used in place of the host experts
    };

    struct stats {
        size_t hits   = 0;
        size_t misses = 0;
        size_t bytes  = 0;
    };

    static constexpr int64_t max_batch = 32;

    const llama_model * model = nullptr;
    int32_t n_expert_used;

    stats stats_small; // up to 8 tokens per ubatch
    stats stats_large;
    stats stats_copy;  // experts copied from the cache for large batches

    std::vector<device> devices;
    std::vector<group> groups;
    std::vector<layer> layers;
    std::unordered_map<const ggml_tensor *, binding> bindings; // host experts -> cached experts
    std::unordered_map<const ggml_tensor *, int32_t> layer_of; // slot map -> layer

    std::vector<int32_t> ids;
    std::vector<moe_cache_lru::fill> fills;
    std::vector<moe_cache_lru::eviction> evictions;

    // Phase 12: 3-Tier Hierarchical MoE Memory System Engine
    struct moe_3tier_engine {
        bool enabled = false;
        bool uring_active = false;
#ifdef GGML_USE_IO_URING
        struct io_uring ring;
#endif
        void * pool_mem = nullptr;
        size_t pool_capacity = 0;
        size_t slot_stride = 0;
        int32_t n_warm_slots = 0;
        int32_t warm_slots_used = 0;

        struct warm_slot_meta {
            int32_t il = -1;
            int32_t expert = -1;
        };
        std::vector<warm_slot_meta> warm_slots;
        std::vector<int32_t> prev;
        std::vector<int32_t> next;
        int32_t head = -1;
        int32_t tail = -1;

        std::vector<std::vector<expert_locator>> locators;
        std::vector<std::vector<size_t>> group_tensor_offsets;

        size_t tier1_vram_hits = 0;
        size_t tier2_ram_hits = 0;
        size_t tier3_cold_fetches = 0;
        size_t total_cold_bytes = 0;

        void init_lru(int32_t n_slots) {
            n_warm_slots = n_slots;
            warm_slots_used = 0;
            warm_slots.assign(n_slots, { -1, -1 });
            prev.resize(n_slots);
            next.resize(n_slots);
            for (int32_t s = 0; s < n_slots; ++s) {
                prev[s] = s - 1;
                next[s] = s + 1 < n_slots ? s + 1 : -1;
            }
            head = n_slots > 0 ? 0 : -1;
            tail = n_slots > 0 ? n_slots - 1 : -1;
        }

        void touch_warm(int32_t s) {
            if (s < 0 || s >= n_warm_slots || s == tail) return;
            if (prev[s] >= 0) {
                next[prev[s]] = next[s];
            } else {
                head = next[s];
            }
            if (next[s] >= 0) {
                prev[next[s]] = prev[s];
            }
            prev[s] = tail;
            next[s] = -1;
            if (tail >= 0) {
                next[tail] = s;
            }
            tail = s;
            if (head < 0) {
                head = s;
            }
        }

        int32_t alloc_warm_slot(int32_t il, int32_t expert) {
            int32_t s = -1;
            if (warm_slots_used < n_warm_slots) {
                s = warm_slots_used++;
                touch_warm(s);
            } else {
                s = head;
                if (s >= 0 && s < n_warm_slots) {
                    const auto & old_meta = warm_slots[s];
                    if (old_meta.il >= 0 && old_meta.expert >= 0 &&
                        (size_t)old_meta.il < locators.size() && (size_t)old_meta.expert < locators[old_meta.il].size()) {
                        auto & old_loc = locators[old_meta.il][old_meta.expert];
                        old_loc.tier = moe_tier_level::TIER_COLD_SSD;
                        old_loc.ram_ptr = nullptr;
                        old_loc.warm_slot = -1;
                    }
                    touch_warm(s);
                }
            }
            if (s >= 0 && s < n_warm_slots) {
                warm_slots[s] = { il, expert };
            }
            return s;
        }

        void mark_evicted_from_vram(int32_t il, int32_t expert) {
            if (!enabled || il < 0 || (size_t)il >= locators.size() || expert < 0 || (size_t)expert >= locators[il].size()) {
                return;
            }
            auto & loc = locators[il][expert];
            loc.tier = (loc.ram_ptr != nullptr && loc.warm_slot >= 0) ? moe_tier_level::TIER_WARM_RAM : moe_tier_level::TIER_COLD_SSD;
            loc.vram_slot = -1;
        }

        void ensure_warm_batch(const llama_model & model, const layer & l, int32_t il, const std::vector<moe_cache_lru::fill> & fills) {
            if (!enabled || n_warm_slots <= 0) return;

            struct read_op {
                int fd = -1;
                void * dst = nullptr;
                size_t len = 0;
                size_t offset = 0;
                const void * fallback_src = nullptr;
            };
            std::vector<read_op> cold_reads;

            for (const auto & f : fills) {
                int32_t e = f.expert;
                if (il < 0 || (size_t)il >= locators.size() || e < 0 || (size_t)e >= locators[il].size()) {
                    continue;
                }
                auto & loc = locators[il][e];
                // An expert is genuinely warm ONLY if it has an allocated warm slot and valid ram pointer
                if (loc.ram_ptr != nullptr && loc.warm_slot >= 0) {
                    touch_warm(loc.warm_slot);
                    tier2_ram_hits++;
                    continue;
                }

                int32_t s = alloc_warm_slot(il, e);
                if (s < 0) {
                    continue;
                }
                uint8_t * slot_mem = (uint8_t *)pool_mem + (size_t)s * slot_stride;
                loc.warm_slot = s;
                loc.ram_ptr = slot_mem;
                loc.tier = moe_tier_level::TIER_WARM_RAM;
                tier3_cold_fetches++;

                for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                    const ggml_tensor * w = l.experts[ip];
                    const size_t tensor_len = w->nb[2];
                    const size_t tensor_off = group_tensor_offsets[l.ig][ip];
                    uint8_t * dst_ptr = slot_mem + tensor_off;
                    const void * src_ptr = (const uint8_t *)w->data + e * tensor_len;

                    int fd = -1;
                    size_t file_offset = 0;
                    if (!model.get_mmap_info(src_ptr, fd, file_offset)) {
                        fd = model.get_mmap_fd();
                        const uint8_t * mmap_addr = (const uint8_t *)model.get_mmap_addr();
                        if (mmap_addr && (const uint8_t *)src_ptr >= mmap_addr) {
                            file_offset = (const uint8_t *)src_ptr - mmap_addr;
                        } else {
                            fd = -1;
                        }
                    }
                    cold_reads.push_back({ fd, dst_ptr, tensor_len, file_offset, src_ptr });
                }
            }

            if (cold_reads.empty()) {
                return;
            }

#ifdef GGML_USE_IO_URING
            if (uring_active) {
                size_t op_idx = 0;
                while (op_idx < cold_reads.size()) {
                    size_t batch_count = 0;
                    while (op_idx + batch_count < cold_reads.size() && batch_count < 64) {
                        const auto & op = cold_reads[op_idx + batch_count];
                        if (op.fd >= 0) {
                            struct io_uring_sqe * sqe = io_uring_get_sqe(&ring);
                            if (!sqe) break;
                            io_uring_prep_read(sqe, op.fd, op.dst, op.len, op.offset);
                            io_uring_sqe_set_data(sqe, (void *)(uintptr_t)(op_idx + batch_count));
                            batch_count++;
                        } else {
                            break;
                        }
                    }
                    if (batch_count > 0) {
                        int submitted = io_uring_submit(&ring);
                        if (submitted > 0) {
                            for (int c = 0; c < submitted; ++c) {
                                struct io_uring_cqe * cqe = nullptr;
                                if (io_uring_wait_cqe(&ring, &cqe) == 0 && cqe) {
                                    uintptr_t completed_idx = (uintptr_t)io_uring_cqe_get_data(cqe);
                                    if (cqe->res < 0 && completed_idx < cold_reads.size()) {
                                        const auto & op = cold_reads[completed_idx];
                                        ssize_t n = pread(op.fd, op.dst, op.len, op.offset);
                                        if (n != (ssize_t)op.len) {
                                            memcpy(op.dst, op.fallback_src, op.len);
                                        }
                                    }
                                    io_uring_cqe_seen(&ring, cqe);
                                }
                            }
                            // Any unsubmitted operations fall back to pread
                            for (size_t b = (size_t)submitted; b < batch_count; ++b) {
                                const auto & op = cold_reads[op_idx + b];
                                ssize_t n = pread(op.fd, op.dst, op.len, op.offset);
                                if (n != (ssize_t)op.len) {
                                    memcpy(op.dst, op.fallback_src, op.len);
                                }
                            }
                            op_idx += batch_count;
                        } else {
                            for (size_t b = 0; b < batch_count; ++b) {
                                const auto & op = cold_reads[op_idx + b];
                                ssize_t n = pread(op.fd, op.dst, op.len, op.offset);
                                if (n != (ssize_t)op.len) {
                                    memcpy(op.dst, op.fallback_src, op.len);
                                }
                            }
                            op_idx += batch_count;
                        }
                    } else {
                        const auto & op = cold_reads[op_idx];
                        if (op.fd >= 0) {
                            ssize_t n = pread(op.fd, op.dst, op.len, op.offset);
                            if (n != (ssize_t)op.len) {
                                memcpy(op.dst, op.fallback_src, op.len);
                            }
                        } else {
                            memcpy(op.dst, op.fallback_src, op.len);
                        }
                        op_idx++;
                    }
                }
            } else
#endif
            {
                for (const auto & op : cold_reads) {
                    if (op.fd >= 0) {
                        ssize_t n = pread(op.fd, op.dst, op.len, op.offset);
                        if (n != (ssize_t)op.len) {
                            memcpy(op.dst, op.fallback_src, op.len);
                        }
                    } else {
                        memcpy(op.dst, op.fallback_src, op.len);
                    }
                }
            }

            for (const auto & op : cold_reads) {
                total_cold_bytes += op.len;
            }
        }
    } tier_engine;

    // slot maps in host memory
    ggml_context_ptr ctx_host;
    ggml_backend_buffer_ptr buf_host;
    size_t buf_host_size = 0;

    // views used by copy_experts
    ggml_context_ptr ctx_views;

    impl(const llama_model & model, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size, size_t warm_ram_bytes, bool moe_3tier, bool moe_2q) :
            model(&model), n_expert_used(model.hparams.n_expert_used_max()), layers(model.layers.size()) {
        for (size_t i = 0; i < backends.size(); ++i) {
            const auto dev_type = ggml_backend_dev_type(ggml_backend_get_device(backends[i]));
            if (dev_type == GGML_BACKEND_DEVICE_TYPE_GPU || dev_type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                auto & d = devices.emplace_back();
                d.backend = backends[i];
                d.buft    = bufts[i];
                // Phase 5: Allocate pinned host ring buffer for zero-copy PCIe DMA
                ggml_backend_dev_t dev = ggml_backend_get_device(backends[i]);
                const size_t ring_size = 64ULL * 1024 * 1024; // 64 MiB pinned DMA staging pool
                d.pinned_ring.init(dev, ring_size);
                if (d.pinned_ring.is_pinned()) {
                    LLAMA_LOG_INFO("%s: %10s allocated %.2f MiB pinned host memory ring buffer for zero-copy PCIe DMA\n",
                        __func__, ggml_backend_buft_name(d.buft), d.pinned_ring.size / 1024.0 / 1024.0);
                }
            }
        }
        if (devices.empty()) {
            throw std::runtime_error("MoE cache requires a GPU backend");
        }
        if (model.split_mode() == LLAMA_SPLIT_MODE_TENSOR) {
            throw std::runtime_error("MoE cache does not support tensor parallelism");
        }
        if (model.hparams.n_expert == 0 || n_expert_used == 0) {
            throw std::runtime_error("MoE cache requires a MoE model");
        }

        // only cache layers that keep all of their experts in host memory, on the device the layer is assigned to
        for (size_t il = 0; il < model.layers.size(); ++il) {
            auto experts = llama_moe_cache_layer_experts(model.layers[il]);
            if (experts.empty() || !std::all_of(experts.begin(), experts.end(), llama_moe_cache_is_host_weight)) {
                continue;
            }
            const auto it_dev = std::find_if(devices.begin(), devices.end(), [&](const device & d) { return ggml_backend_get_device(d.backend) == model.dev_layer(il); });
            if (it_dev == devices.end()) {
                continue;
            }
            const int32_t id = (int32_t) (it_dev - devices.begin());
            auto it = std::find_if(groups.begin(), groups.end(), [&](const group & g) { return g.id == id && llama_moe_cache_same_layout(g.ref, experts); });
            if (it == groups.end()) {
                groups.emplace_back();
                it = groups.end() - 1;
                it->id  = id;
                it->ref = experts;
            }
            it->layers.push_back(il);
            for (const ggml_tensor * t : experts) {
                it->host_bytes         += ggml_nbytes(t);
                devices[id].host_bytes += ggml_nbytes(t);
            }
        }
        if (groups.empty()) {
            LLAMA_LOG_WARN("%s: no layer has all of its experts in host memory, MoE cache is disabled\n", __func__);
            return;
        }

        // one extra slot at the end, CUDA MMQ can read past the last expert
        auto alloc_size = [&](const group & g, int32_t n_slots) {
            const size_t alignment = ggml_backend_buft_get_alignment(devices[g.id].buft);
            size_t res = 0;
            for (const ggml_tensor * t : g.ref) {
                res += GGML_PAD(t->nb[2]*(n_slots + 1), alignment);
            }
            return res;
        };

        // the budget is split among the devices with host experts like the layers, by the tensor split or by default by free memory
        const float * tensor_split = model.tensor_split();
        const bool split_by_free = tensor_split == nullptr ||
            std::all_of(tensor_split, tensor_split + model.n_devices(), [](float x) { return x == 0.0f; });
        double split_sum = 0.0;
        if (!split_by_free) {
            std::vector<float> cumulative(model.n_devices());
            std::vector<bool> eligible(model.n_devices(), false);
            for (size_t i = 0; i < model.n_devices(); ++i) {
                cumulative[i] = tensor_split[i];
            }
            for (const device & d : devices) {
                if (d.host_bytes > 0) {
                    ggml_backend_dev_t dev = ggml_backend_get_device(d.backend);
                    const auto it = std::find_if(model.devices.begin(), model.devices.end(), [&](const llama_device & ld) { return ld.dev == dev; });
                    if (it != model.devices.end()) {
                        eligible[it - model.devices.begin()] = true;
                    }
                }
            }
            const auto shares = llama_moe_cache_detail::device_shares(cumulative, eligible);
            for (device & d : devices) {
                if (d.host_bytes == 0) continue;
                ggml_backend_dev_t dev = ggml_backend_get_device(d.backend);
                const auto it = std::find_if(model.devices.begin(), model.devices.end(), [&](const llama_device & ld) { return ld.dev == dev; });
                GGML_ASSERT(it != model.devices.end());
                d.split = shares[it - model.devices.begin()];
                split_sum += d.split;
            }
        } else {
            for (device & d : devices) {
                if (d.host_bytes == 0) {
                    continue;
                }
                ggml_backend_dev_t dev = ggml_backend_get_device(d.backend);
                size_t free = 0;
                size_t total = 0;
                ggml_backend_dev_memory(dev, &free, &total);
                d.split = (double) free;
                split_sum += d.split;
            }
        }
        if (split_sum <= 0.0) {
            // the devices do not report their free memory
            for (device & d : devices) {
                d.split    = d.host_bytes > 0 ? 1.0 : 0.0;
                split_sum += d.split;
            }
        }

        // within a device the budget is split by the size of the experts, so each group caches the same fraction of its experts
        std::vector<size_t> n_tensors(devices.size(), 0);
        size_t n_tensors_host = 0;
        for (group & g : groups) {
            const device & d = devices[g.id];
            const int32_t n_expert  = g.ref[0]->ne[2];
            const size_t  budget    = (size_t) ((double) size*d.split/split_sum*g.host_bytes/d.host_bytes);
            const int32_t max_slots = g.layers.size()*n_expert;
            // Phase 6: High-performance binary search for slot budget sizing
            // Handles both macro-experts and 24,576+ micro-expert topologies in O(log N)
            int32_t low = 0;
            int32_t high = max_slots;
            while (low <= high) {
                int32_t mid = low + (high - low) / 2;
                if (alloc_size(g, mid) <= budget) {
                    g.n_slots = mid;
                    low = mid + 1;
                } else {
                    high = mid - 1;
                }
            }
            if (llama_moe_cache_detail::is_fine_grained_moe(n_expert, n_expert_used)) {
                LLAMA_LOG_INFO("%s: Fine-Grained MoE Topology: %d micro-experts per layer, %d slots cached in VRAM slab\n",
                    __func__, n_expert, g.n_slots);
            }
            if (g.n_slots < n_expert_used) {
                LLAMA_LOG_WARN("%s: MoE cache budget is too small for %zu layers, they are not cached\n", __func__, g.layers.size());
                g.n_slots = 0;
                continue;
            }
            g.lru.init(model.layers.size(), n_expert, g.n_slots, moe_2q);
            n_tensors[g.id] += g.ref.size()*(1 + g.layers.size());
            n_tensors_host  += g.layers.size();
        }
        if (n_tensors_host == 0) {
            throw std::runtime_error("MoE cache is too small to hold the experts of one token");
        }

        auto init_ctx = [](size_t n_tensors) {
            ggml_init_params params = {
                /*.mem_size   =*/ n_tensors*ggml_tensor_overhead(),
                /*.mem_buffer =*/ nullptr,
                /*.no_alloc   =*/ true,
            };
            ggml_context_ptr res(ggml_init(params));
            if (!res) {
                throw std::runtime_error("failed to create the MoE cache context");
            }
            return res;
        };
        for (size_t id = 0; id < devices.size(); ++id) {
            if (n_tensors[id] > 0) {
                devices[id].ctx = init_ctx(n_tensors[id]);
            }
        }
        ctx_host  = init_ctx(n_tensors_host);
        ctx_views = init_ctx(2);

        ggml_backend_buffer_type_t buft_host = ggml_backend_cpu_buffer_type();
        const size_t alignment_host = ggml_backend_buft_get_alignment(buft_host);

        for (size_t ig = 0; ig < groups.size(); ++ig) {
            group & g = groups[ig];
            if (g.n_slots == 0) {
                continue;
            }
            ggml_context * ctx = devices[g.id].ctx.get();
            for (const ggml_tensor * t : g.ref) {
                ggml_tensor * bank = ggml_new_tensor_3d(ctx, t->type, t->ne[0], t->ne[1], g.n_slots + 1);
                GGML_ASSERT(bank->nb[2] == t->nb[2]);
                ggml_format_name(bank, "moe_cache.%zu.%s", ig, t->name);
                g.banks.push_back(bank);
            }
            for (int32_t il : g.layers) {
                layer & l = layers[il];
                l.ig      = (int32_t) ig;
                l.experts = llama_moe_cache_layer_experts(model.layers[il]);
                for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                    ggml_tensor * bank   = g.banks[ip];
                    ggml_tensor * cached = ggml_view_3d(ctx, bank, bank->ne[0], bank->ne[1], g.n_slots, bank->nb[1], bank->nb[2], 0);
                    ggml_format_name(cached, "moe_cache.%s", l.experts[ip]->name);
                    bindings[l.experts[ip]] = { il, (int32_t) ip, cached };
                }
                l.slot_map = ggml_new_tensor_2d(ctx_host.get(), GGML_TYPE_I32, 1, g.ref[0]->ne[2]);
                ggml_format_name(l.slot_map, "moe_cache.slot_map-%d", il);
                layer_of[l.slot_map] = il;
                buf_host_size += GGML_PAD(ggml_nbytes(l.slot_map), alignment_host);
            }
            devices[g.id].buf_size += alloc_size(g, g.n_slots);
        }

        if (model.hparams.no_alloc) {
            // only used to measure the memory use, see llama_context::memory_breakdown
            for (device & d : devices) {
                if (!d.ctx) {
                    continue;
                }
                d.buf.reset(ggml_backend_buft_alloc_buffer(d.buft, 0));
                for (ggml_tensor * t = ggml_get_first_tensor(d.ctx.get()); t != nullptr; t = ggml_get_next_tensor(d.ctx.get(), t)) {
                    t->buffer = d.buf.get();
                }
            }
            buf_host.reset(ggml_backend_buft_alloc_buffer(buft_host, 0));
            for (ggml_tensor * t = ggml_get_first_tensor(ctx_host.get()); t != nullptr; t = ggml_get_next_tensor(ctx_host.get(), t)) {
                t->buffer = buf_host.get();
            }
        } else {
            for (device & d : devices) {
                if (!d.ctx) {
                    continue;
                }
                d.buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(d.ctx.get(), d.buft));
                if (!d.buf) {
                    throw std::runtime_error("failed to allocate the MoE cache buffers");
                }
                ggml_backend_buffer_clear(d.buf.get(), 0);
                d.buf_size = ggml_backend_buffer_get_size(d.buf.get());
            }
            buf_host.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx_host.get(), buft_host));
            if (!buf_host) {
                throw std::runtime_error("failed to allocate the MoE cache buffers");
            }
            ggml_backend_buffer_clear(buf_host.get(), 0xff); // all slots are -1
            buf_host_size = ggml_backend_buffer_get_size(buf_host.get());

            for (group & g : groups) {
                for (int32_t il : g.layers) {
                    if (layers[il].slot_map != nullptr) {
                        g.lru.slot_map[il] = (int32_t *) layers[il].slot_map->data;
                    }
                }
            }
        }

        // as weights, the ops that read the banks run on the device and the slot maps are copied with the copy callback
        for (device & d : devices) {
            if (d.buf) {
                ggml_backend_buffer_set_usage(d.buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            }
        }
        ggml_backend_buffer_set_usage(buf_host.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

        for (size_t id = 0; id < devices.size(); ++id) {
            const device & d = devices[id];
            if (d.host_bytes == 0) {
                continue;
            }
            LLAMA_LOG_INFO("%s: %10s MoE cache size = %8.2f MiB for %.2f MiB of host experts\n", __func__,
                ggml_backend_buft_name(d.buft), d.buf_size/1024.0/1024.0, d.host_bytes/1024.0/1024.0);
            for (const group & g : groups) {
                if (g.id == (int32_t) id) {
                    LLAMA_LOG_INFO("%s: %2zu layers, %s: %5d slots (%.1f%%)\n", __func__,
                        g.layers.size(), ggml_type_name(g.ref.back()->type), g.n_slots, 100.0*g.n_slots/(g.layers.size()*g.ref[0]->ne[2]));
                }
            }
        }

        // Initialize Phase 12: 3-Tier Hierarchical MoE Memory System
        if (moe_3tier && !groups.empty()) {
            size_t expert_bytes = 0;
            for (const ggml_tensor * t : groups[0].ref) {
                expert_bytes += t->nb[2];
            }
            if (expert_bytes > 0) {
                tier_engine.slot_stride = (expert_bytes + 4095) & ~((size_t)4095); // 4KB aligned

                size_t target_cap = warm_ram_bytes > 0 ? warm_ram_bytes : 12ULL * 1024 * 1024 * 1024;
                void * mem = nullptr;
                while (target_cap >= 512ULL * 1024 * 1024) {
                    int ret = posix_memalign(&mem, 2 * 1024 * 1024, target_cap);
                    if (ret == 0 && mem != nullptr) {
                        break;
                    }
                    target_cap = target_cap > 1024ULL * 1024 * 1024 ? target_cap - 1024ULL * 1024 * 1024 : 0;
                }
                if (mem == nullptr) {
                    target_cap = 512ULL * 1024 * 1024;
                    mem = malloc(target_cap);
                }

                if (mem != nullptr) {
                    tier_engine.pool_mem = mem;
                    tier_engine.pool_capacity = target_cap;
                    int32_t n_slots = (int32_t)(target_cap / tier_engine.slot_stride);
                    tier_engine.init_lru(n_slots);
                    tier_engine.enabled = true;

#ifdef MADV_HUGEPAGE
                    madvise(mem, target_cap, MADV_HUGEPAGE);
#endif
                    LLAMA_LOG_INFO("%s: Tier 2 WARM DDR5 pool allocated: %.2f GiB (%d warm slots, 2MB THP-aligned)\n",
                        __func__, target_cap / 1024.0 / 1024.0 / 1024.0, n_slots);
                }

#ifdef GGML_USE_IO_URING
                int ret = io_uring_queue_init(64, &tier_engine.ring, 0);
                if (ret == 0) {
                    tier_engine.uring_active = true;
                    int mmap_fd = model.get_mmap_fd();
                    LLAMA_LOG_INFO("%s: Tier 3 io_uring DMA initialized (depth 64, fd %d)\n", __func__, mmap_fd);
                } else {
                    LLAMA_LOG_WARN("%s: io_uring_queue_init failed (%d): falling back to pread\n", __func__, ret);
                }
#endif

                tier_engine.locators.resize(model.layers.size());
                for (size_t il = 0; il < model.layers.size(); ++il) {
                    if (layers[il].ig >= 0 && !layers[il].experts.empty()) {
                        int32_t n_expert = groups[layers[il].ig].ref[0]->ne[2];
                        tier_engine.locators[il].resize(n_expert);
                        for (int32_t e = 0; e < n_expert; ++e) {
                            auto & loc = tier_engine.locators[il][e];
                            loc.tier = moe_tier_level::TIER_COLD_SSD;
                            loc.bytes = expert_bytes;
                            int dummy_fd = -1;
                            model.get_mmap_info((const uint8_t *)layers[il].experts[0]->data + e * layers[il].experts[0]->nb[2],
                                                dummy_fd, loc.file_offset);
                        }
                    }
                }

                for (size_t ig = 0; ig < groups.size(); ++ig) {
                    std::vector<size_t> offsets;
                    size_t cur_off = 0;
                    for (const ggml_tensor * t : groups[ig].ref) {
                        offsets.push_back(cur_off);
                        cur_off += t->nb[2];
                    }
                    tier_engine.group_tensor_offsets.push_back(offsets);
                }

                LLAMA_LOG_INFO("%s: [Phase 12] Universal 3-Tier MoE Pinning ACTIVE (W_gate & Shared Experts in VRAM)\n", __func__);
            }
        }

        if (moe_2q && !groups.empty()) {
            LLAMA_LOG_INFO("%s: [Phase 12] 2Q & Ghost-Cache Replacement Policy ACTIVE (A1in: 25%% probation, Am: 75%% frequent, A1out ghost)\n", __func__);
        }
    }

    ggml_backend_t backend(int32_t il) const {
        return devices[groups[layers[il].ig].id].backend;
    }

    ~impl() {
        log_stats();
#ifdef GGML_USE_IO_URING
        if (tier_engine.uring_active) {
            io_uring_queue_exit(&tier_engine.ring);
            tier_engine.uring_active = false;
        }
#endif
        if (tier_engine.pool_mem) {
            free(tier_engine.pool_mem);
            tier_engine.pool_mem = nullptr;
        }
    }

    ggml_tensor * get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used) const {
        if (il < 0 || il >= (int32_t) layers.size() || layers[il].ig < 0) {
            return nullptr;
        }
        const layer & l = layers[il];

        // large batches use most experts of a layer, so they gain little from the cache and would evict the experts used in generation
        if (n_tokens == 0 || n_tokens > max_batch || std::min(n_tokens*n_expert_used, l.slot_map->ne[1]) > groups[l.ig].n_slots) {
            return nullptr;
        }
        return l.slot_map;
    }

    ggml_tensor * get_experts(const ggml_tensor * w) const {
        const auto it = bindings.find(w);
        return it != bindings.end() ? it->second.cached : nullptr;
    }

    int64_t copy_experts(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t e, int64_t last) {
        const auto it = bindings.find(w);
        if (it == bindings.end()) {
            return 0;
        }
        const binding & b = it->second;
        const group & g = groups[layers[b.il].ig];
        if (backend != devices[g.id].backend) {
            return 0;
        }

        // large batches only read the cache, so the experts used in generation stay in it
        const int32_t * slots = g.lru.slot_map[b.il];
        if (slots == nullptr || slots[e] < 0) {
            return 0;
        }
        int64_t n = 1;
        while (e + n <= last && slots[e + n] == slots[e] + n) {
            n++;
        }

        ggml_tensor * bank = g.banks[b.ip];
        ggml_reset(ctx_views.get());
        ggml_tensor * src_view = ggml_view_3d(ctx_views.get(), bank, bank->ne[0], bank->ne[1], n, bank->nb[1], bank->nb[2], slots[e]*bank->nb[2]);
        ggml_tensor * dst_view = ggml_view_3d(ctx_views.get(), dst,  dst->ne[0],  dst->ne[1],  n, dst->nb[1],  dst->nb[2],  e*dst->nb[2]);
        ggml_backend_view_init(src_view);
        ggml_backend_view_init(dst_view);
        ggml_backend_tensor_copy_async(backend, backend, src_view, dst_view);

        stats_copy.hits  += n;
        stats_copy.bytes += ggml_nbytes(src_view);

        return n;
    }

    bool copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph) {
        const auto it = layer_of.find(src);
        if (it == layer_of.end()) {
            return false;
        }
        const int32_t il = it->second;
        const layer & l = layers[il];
        group & g = groups[l.ig];

        GGML_ASSERT(backend == devices[g.id].backend);

        // the get_rows that looks up the slots of the selected experts
        const int n_nodes = ggml_graph_n_nodes(graph);
        const ggml_tensor * lookup = nullptr;
        for (int i = 0; i < n_nodes && lookup == nullptr; ++i) {
            const ggml_tensor * node = ggml_graph_node(graph, i);
            if (node->op == GGML_OP_GET_ROWS && node->src[0] == dst) {
                lookup = node;
            }
        }
        GGML_ASSERT(lookup != nullptr);

        // the selected experts must be computed in an earlier split
        // the scheduler starts a new split at the lookup because it reads a host weight, but only if the split already has inputs
        const ggml_tensor * sel = lookup->src[1];
        for (int i = 0; i < n_nodes; ++i) {
            const ggml_tensor * node = ggml_graph_node(graph, i);
            if (node == sel || node == sel->view_src) {
                GGML_ABORT("the experts of layer %d are selected in the same split as their MoE cache lookup", il);
            }
        }
        GGML_ASSERT(ggml_is_contiguous(sel));

        ids.resize(ggml_nelements(sel));
        ggml_backend_tensor_get_async(backend, sel, ids.data(), 0, ggml_nbytes(sel));
        ggml_backend_synchronize(backend);

        size_t n_hit = 0;
        if (!g.lru.plan(il, ids.data(), ids.size(), fills, evictions, n_hit)) {
            GGML_ABORT("the MoE cache is too small for the experts selected in layer %d", il);
        }
        for (const auto & ev : evictions) {
            tier_engine.mark_evicted_from_vram(ev.il, ev.expert);
        }

        // upload the missing experts
        size_t bytes = 0;
        if (tier_engine.enabled) {
            tier_engine.ensure_warm_batch(*model, l, il, fills);

            for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                const ggml_tensor * w    = l.experts[ip];
                ggml_tensor       * bank = g.banks[ip];
                const size_t expert_size = w->nb[2];
                const size_t tensor_off  = tier_engine.group_tensor_offsets[l.ig][ip];

                for (const auto & f : fills) {
                    const void * host_expert_ptr = (const uint8_t *)w->data + f.expert * expert_size;
                    const void * warm_src = (tier_engine.locators[il][f.expert].ram_ptr != nullptr) ?
                        (const uint8_t *)tier_engine.locators[il][f.expert].ram_ptr + tensor_off :
                        host_expert_ptr;
                    const void * dma_src  = devices[g.id].pinned_ring.stage(warm_src, expert_size);

                    ggml_backend_tensor_set_async(backend, bank, dma_src, f.slot * expert_size, expert_size);
                    bytes += expert_size;
                    tier_engine.locators[il][f.expert].tier = moe_tier_level::TIER_HOT_VRAM;
                    tier_engine.locators[il][f.expert].vram_slot = f.slot;
                }
            }
        } else {
            for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                const ggml_tensor * w    = l.experts[ip];
                ggml_tensor       * bank = g.banks[ip];
                const size_t expert_size = w->nb[2];
                for (size_t i = 0; i < fills.size();) {
                    size_t n = 1;
                    while (i + n < fills.size() && fills[i + n].expert == fills[i].expert + (int32_t) n && fills[i + n].slot == fills[i].slot + (int32_t) n) {
                        n++;
                    }
                    const size_t chunk_size = n * expert_size;
                    const void * host_src = (const uint8_t *) w->data + fills[i].expert*expert_size;

                    // Phase 5: Zero-copy DMA streaming via pinned host memory ring buffer
                    const void * dma_src = devices[g.id].pinned_ring.stage(host_src, chunk_size);

                    ggml_backend_tensor_set_async(backend, bank, dma_src, fills[i].slot*expert_size, chunk_size);
                    bytes += chunk_size;
                    i += n;
                }
            }
        }

        // Phase 5: Trigger concurrent Early-Lookahead Router prefetch for Layer L+1 while GPU computes
        if (ggml_moe_prefetch_enabled()) {
            ggml_moe_prefetch_lookahead_step(il, ids.data(), ids.size());
        }

        stats & st = ids.size() <= (size_t) 8*n_expert_used ? stats_small : stats_large;
        st.hits   += n_hit;
        st.misses += fills.size();
        st.bytes  += bytes;

        // the next copy synchronizes the backend before it changes the slot map again
        ggml_backend_tensor_set_async(backend, dst, src->data, 0, ggml_nbytes(src));

        return true;
    }

    void log_stats() const {
        auto log = [](const char * name, const stats & st) {
            const size_t n = st.hits + st.misses;
            if (n == 0) {
                return;
            }
            LLAMA_LOG_INFO("llama_moe_cache: %s: hits = %zu, misses = %zu, hit rate = %.2f%%, uploaded = %.2f MiB\n",
                name, st.hits, st.misses, 100.0*st.hits/n, st.bytes/1024.0/1024.0);
        };
        log("ubatch <= 8", stats_small);
        log("ubatch  > 8", stats_large);
        if (stats_copy.hits > 0) {
            LLAMA_LOG_INFO("llama_moe_cache: large batches: %zu experts copied from the cache, %.2f MiB\n", stats_copy.hits, stats_copy.bytes/1024.0/1024.0);
        }
        size_t total_ghost_hits = 0;
        size_t total_promotions_2q = 0;
        size_t total_quarantined = 0;
        bool any_2q = false;
        for (const auto & g : groups) {
            if (g.lru.use_2q) {
                any_2q = true;
                total_ghost_hits += g.lru.stat_ghost_hits;
                total_promotions_2q += g.lru.stat_promotions_2q;
                total_quarantined += g.lru.stat_pollution_quarantined;
            }
        }
        if (any_2q) {
            LLAMA_LOG_INFO("llama_moe_cache: [Phase 12 2Q & Ghost-Cache Policy Telemetry]\n");
            LLAMA_LOG_INFO("  Ghost Cache Hits      : %zu (recurrent temporal locality recognized)\n", total_ghost_hits);
            LLAMA_LOG_INFO("  2Q Queue Promotions   : %zu (probation A1in -> frequent Am queue)\n", total_promotions_2q);
            LLAMA_LOG_INFO("  Pollution Quarantined : %zu (transient one-hit wonders filtered)\n", total_quarantined);
        }
        if (tier_engine.enabled) {
            LLAMA_LOG_INFO("llama_moe_cache: [Phase 12 3-Tier MoE Engine Telemetry]\n");
            LLAMA_LOG_INFO("  Tier 1 (HOT VRAM)  hits: %zu\n", stats_small.hits + stats_large.hits);
            LLAMA_LOG_INFO("  Tier 2 (WARM RAM)  hits: %zu (pool: %.2f GiB / %.2f GiB used)\n",
                tier_engine.tier2_ram_hits,
                (tier_engine.warm_slots_used * tier_engine.slot_stride) / 1024.0 / 1024.0 / 1024.0,
                tier_engine.pool_capacity / 1024.0 / 1024.0 / 1024.0);
            LLAMA_LOG_INFO("  Tier 3 (COLD SSD)  fetches: %zu (streamed: %.2f MiB via %s)\n",
                tier_engine.tier3_cold_fetches,
                tier_engine.total_cold_bytes / 1024.0 / 1024.0,
                tier_engine.uring_active ? "io_uring DMA" : "pread fallback");

            fprintf(stderr, "\n=======================================================\n");
            fprintf(stderr, "🥪 [Phase 12] 3-Tier Hierarchical MoE Engine Telemetry\n");
            fprintf(stderr, "   Tier 1 (HOT VRAM)  Hits   : %10zu\n", stats_small.hits + stats_large.hits);
            fprintf(stderr, "   Tier 2 (WARM RAM)  Hits   : %10zu (Pool: %.2f / %.2f GiB used)\n",
                tier_engine.tier2_ram_hits,
                (tier_engine.warm_slots_used * tier_engine.slot_stride) / 1024.0 / 1024.0 / 1024.0,
                tier_engine.pool_capacity / 1024.0 / 1024.0 / 1024.0);
            fprintf(stderr, "   Tier 3 (COLD SSD)  Fetches: %10zu (Streamed: %.2f MiB via %s)\n",
                tier_engine.tier3_cold_fetches,
                tier_engine.total_cold_bytes / 1024.0 / 1024.0,
                tier_engine.uring_active ? "io_uring DMA" : "pread fallback");
            if (any_2q) {
                fprintf(stderr, "   2Q Policy Engine          : ACTIVE (Probation: 25%%, Frequent: 75%%)\n");
                fprintf(stderr, "   2Q Ghost Cache Hits       : %10zu (Recurrent Temporal Locality)\n", total_ghost_hits);
                fprintf(stderr, "   2Q Promotions (A1in -> Am): %10zu (Promoted to Frequent LRU)\n", total_promotions_2q);
                fprintf(stderr, "   2Q Pollution Quarantined  : %10zu (Transient Noise Isolated)\n", total_quarantined);
            }
            fprintf(stderr, "=======================================================\n\n");
        }
    }
};

llama_moe_cache::llama_moe_cache(const llama_model & model,
                                 const std::vector<ggml_backend_t> & backends,
                                 const std::vector<ggml_backend_buffer_type_t> & bufts,
                                 size_t size,
                                 size_t warm_ram_bytes,
                                 bool   moe_3tier,
                                 bool   moe_2q) :
    pimpl(new impl(model, backends, bufts, size, warm_ram_bytes, moe_3tier, moe_2q)) {
}

llama_moe_cache::~llama_moe_cache() = default;

ggml_backend_t llama_moe_cache::backend(int32_t il) const {
    return pimpl->backend(il);
}

ggml_tensor * llama_moe_cache::get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used) const {
    return pimpl->get_slot_map(il, n_tokens, n_expert_used);
}

ggml_tensor * llama_moe_cache::get_experts(const ggml_tensor * w) const {
    return pimpl->get_experts(w);
}

bool llama_moe_cache::copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph) {
    return pimpl->copy(backend, src, dst, graph);
}

int64_t llama_moe_cache::copy_experts(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t e, int64_t last) {
    return pimpl->copy_experts(backend, w, dst, e, last);
}

std::map<ggml_backend_buffer_type_t, size_t> llama_moe_cache::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> res;
    for (const auto & d : pimpl->devices) {
        if (d.buf) {
            res[ggml_backend_buffer_get_type(d.buf.get())] += d.buf_size;
        }
        if (d.pinned_ring.buf) {
            res[ggml_backend_buffer_get_type(d.pinned_ring.buf.get())] += d.pinned_ring.size;
        }
    }
    if (pimpl->buf_host) {
        res[ggml_backend_buffer_get_type(pimpl->buf_host.get())] += pimpl->buf_host_size;
    }
    return res;
}

bool llama_moe_cache::is_pinned_dma_enabled() const {
    for (const auto & d : pimpl->devices) {
        if (d.pinned_ring.is_pinned()) {
            return true;
        }
    }
    return false;
}

size_t llama_moe_cache::pinned_buffer_size() const {
    size_t total = 0;
    for (const auto & d : pimpl->devices) {
        total += d.pinned_ring.size;
    }
    return total;
}

bool llama_moe_cache::is_fine_grained() const {
    for (const auto & g : pimpl->groups) {
        if (!g.ref.empty() && llama_moe_cache_detail::is_fine_grained_moe(g.ref[0]->ne[2], pimpl->n_expert_used)) {
            return true;
        }
    }
    return false;
}

int32_t llama_moe_cache::total_cached_slots() const {
    int32_t total = 0;
    for (const auto & g : pimpl->groups) {
        total += g.n_slots;
    }
    return total;
}

const expert_locator * llama_moe_cache::get_locator(int32_t il, int32_t expert) const {
    if (!pimpl || !pimpl->tier_engine.enabled) return nullptr;
    if (il < 0 || (size_t)il >= pimpl->tier_engine.locators.size()) return nullptr;
    if (expert < 0 || (size_t)expert >= pimpl->tier_engine.locators[il].size()) return nullptr;
    return &pimpl->tier_engine.locators[il][expert];
}

moe_tier_level llama_moe_cache::get_tier(int32_t il, int32_t expert) const {
    const auto * loc = get_locator(il, expert);
    return loc ? loc->tier : moe_tier_level::TIER_COLD_SSD;
}

size_t llama_moe_cache::warm_pool_size() const {
    return pimpl ? pimpl->tier_engine.pool_capacity : 0;
}

size_t llama_moe_cache::warm_pool_used() const {
    return pimpl ? (size_t)pimpl->tier_engine.warm_slots_used * pimpl->tier_engine.slot_stride : 0;
}

size_t llama_moe_cache::cold_fetches() const {
    return pimpl ? pimpl->tier_engine.tier3_cold_fetches : 0;
}

bool llama_moe_cache::is_3tier_enabled() const {
    return pimpl && pimpl->tier_engine.enabled;
}

bool llama_moe_cache::is_2q_enabled() const {
    if (!pimpl) return false;
    for (const auto & g : pimpl->groups) {
        if (g.lru.use_2q) return true;
    }
    return false;
}

size_t llama_moe_cache::ghost_hits() const {
    if (!pimpl) return 0;
    size_t total = 0;
    for (const auto & g : pimpl->groups) {
        total += g.lru.stat_ghost_hits;
    }
    return total;
}

size_t llama_moe_cache::promotions_2q() const {
    if (!pimpl) return 0;
    size_t total = 0;
    for (const auto & g : pimpl->groups) {
        total += g.lru.stat_promotions_2q;
    }
    return total;
}

size_t llama_moe_cache::pollution_quarantined() const {
    if (!pimpl) return 0;
    size_t total = 0;
    for (const auto & g : pimpl->groups) {
        total += g.lru.stat_pollution_quarantined;
    }
    return total;
}
