#include "ggml-moe-prefetch.h"
#include "ggml-backend.h"

#include <vector>
#include <deque>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <cinttypes>

#if defined(__has_include)
    #if __has_include(<unistd.h>)
        #include <unistd.h>
    #endif
    #if __has_include(<sys/mman.h>)
        #include <sys/mman.h>
    #endif
#endif

#ifndef MADV_WILLNEED
#define MADV_WILLNEED 3
#endif

#ifndef MADV_COLD
#define MADV_COLD 20
#endif

namespace {

struct mapped_region {
    uintptr_t start;
    uintptr_t end;
};

struct prefetch_job {
    void * addr;
    size_t len;
    bool is_cold;
};

class prefetch_service {
public:
    static prefetch_service & instance() {
        static prefetch_service svc;
        return svc;
    }

    void register_mapping(const void * base, size_t size) {
        if (!base || size == 0) return;
        std::lock_guard<std::mutex> lock(m_registry_mutex);
        uintptr_t start = reinterpret_cast<uintptr_t>(base);
        m_mappings.push_back({ start, start + size });
    }

    void unregister_mapping(const void * base) {
        if (!base) return;
        std::lock_guard<std::mutex> lock(m_registry_mutex);
        uintptr_t start = reinterpret_cast<uintptr_t>(base);
        m_mappings.erase(
            std::remove_if(m_mappings.begin(), m_mappings.end(),
                [start](const mapped_region & r) { return r.start == start; }),
            m_mappings.end());
    }

    bool is_registered(const void * ptr, size_t bytes) const {
        if (!ptr || bytes == 0) return false;
        uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
        std::lock_guard<std::mutex> lock(m_registry_mutex);
        for (const auto & r : m_mappings) {
            if (addr >= r.start && (addr + bytes) <= r.end) {
                return true;
            }
        }
        return false;
    }

    void set_n_threads(int n_threads) {
        std::unique_lock<std::mutex> lock(m_queue_mutex);
        if (n_threads <= 0) {
            if (m_running) {
                m_stop_requested = true;
                m_queue_cv.notify_all();
                lock.unlock();
                for (auto & t : m_workers) {
                    if (t.joinable()) t.join();
                }
                m_workers.clear();
                lock.lock();
                m_running = false;
                m_stop_requested = false;
                m_jobs.clear();
            }
            return;
        }

        if (m_running && (int) m_workers.size() == n_threads) {
            return;
        }

        // Restart worker pool with desired thread count
        m_stop_requested = true;
        m_queue_cv.notify_all();
        lock.unlock();
        for (auto & t : m_workers) {
            if (t.joinable()) t.join();
        }
        m_workers.clear();

        lock.lock();
        m_running = true;
        m_stop_requested = false;
        m_jobs.clear();
        for (int i = 0; i < n_threads; ++i) {
            m_workers.emplace_back(&prefetch_service::worker_loop, this);
        }
    }

    bool is_enabled() const {
        return m_running;
    }

    void bump_epoch() {
        m_epoch.fetch_add(1, std::memory_order_relaxed);
    }

    void enqueue(void * addr, size_t len, bool is_cold) {
        if (!m_running || !addr || len == 0) return;
        std::lock_guard<std::mutex> lock(m_queue_mutex);
        m_jobs.push_back({ addr, len, is_cold });
        m_queue_cv.notify_one();
    }

    void wait_flush() {
        std::unique_lock<std::mutex> lock(m_queue_mutex);
        m_idle_cv.wait(lock, [this]() {
            return m_jobs.empty() && (m_active_workers == 0);
        });
    }

    struct moe_layer_meta {
        int32_t il;
        const struct ggml_tensor * gate_inp;
        const struct ggml_tensor * up_exps;
        const struct ggml_tensor * gate_exps;
        const struct ggml_tensor * down_exps;
        int64_t n_expert;
        int64_t n_expert_used;
    };

    void register_moe_layer(int32_t il,
                            const struct ggml_tensor * gate_inp,
                            const struct ggml_tensor * up_exps,
                            const struct ggml_tensor * gate_exps,
                            const struct ggml_tensor * down_exps,
                            int64_t n_expert,
                            int64_t n_expert_used) {
        std::lock_guard<std::mutex> lock(m_layer_mutex);
        for (auto & l : m_moe_layers) {
            if (l.il == il) {
                l = { il, gate_inp, up_exps, gate_exps, down_exps, n_expert, n_expert_used };
                return;
            }
        }
        m_moe_layers.push_back({ il, gate_inp, up_exps, gate_exps, down_exps, n_expert, n_expert_used });
        std::sort(m_moe_layers.begin(), m_moe_layers.end(), [](const moe_layer_meta & a, const moe_layer_meta & b) {
            return a.il < b.il;
        });
    }

    void clear_moe_layers() {
        std::lock_guard<std::mutex> lock(m_layer_mutex);
        m_moe_layers.clear();
    }

    void lookahead_step(int32_t current_il, const int32_t * current_ids, size_t n_ids) {
        if (!is_enabled()) return;
        std::lock_guard<std::mutex> lock(m_layer_mutex);
        if (m_moe_layers.empty() || !current_ids || n_ids == 0) return;

        // Find next MoE layer after current_il
        const moe_layer_meta * next_layer = nullptr;
        for (const auto & l : m_moe_layers) {
            if (l.il > current_il) {
                next_layer = &l;
                break;
            }
        }
        if (!next_layer) {
            // Autoregressive wrap-around to first MoE layer for token loop
            next_layer = &m_moe_layers.front();
        }

        if (!next_layer || next_layer->n_expert <= 0) return;

        const int64_t next_n_expert = next_layer->n_expert;
        std::vector<int32_t> predicted_experts;
        predicted_experts.reserve(n_ids);

        for (size_t i = 0; i < n_ids; ++i) {
            int32_t exp = current_ids[i];
            if (exp >= 0) {
                int32_t mapped_exp = (int32_t) ((int64_t) exp % next_n_expert);
                if (std::find(predicted_experts.begin(), predicted_experts.end(), mapped_exp) == predicted_experts.end()) {
                    predicted_experts.push_back(mapped_exp);
                }
            }
        }

        // Prefetch predicted expert slices across up_exps, gate_exps, and down_exps asynchronously
        for (int32_t exp : predicted_experts) {
            auto prefetch_expert_tensor = [this, exp](const struct ggml_tensor * t) {
                if (!t || !t->data) return;
                if (t->buffer && !ggml_backend_buffer_is_host(t->buffer)) return;
                const size_t expert_size = t->nb[2];
                void * addr = (uint8_t *) t->data + (size_t) exp * expert_size;
                if (is_registered(addr, expert_size)) {
                    enqueue(addr, expert_size, /* is_cold = */ false);
                }
            };

            prefetch_expert_tensor(next_layer->up_exps);
            prefetch_expert_tensor(next_layer->gate_exps);
            prefetch_expert_tensor(next_layer->down_exps);
        }
    }

private:
    prefetch_service() : m_running(false), m_stop_requested(false), m_active_workers(0), m_epoch(0) {}
    ~prefetch_service() {
        set_n_threads(0);
    }

    void worker_loop() {
        const size_t page_size = get_sys_page_size();
        const uintptr_t page_mask = page_size - 1;

        while (true) {
            prefetch_job job;
            {
                std::unique_lock<std::mutex> lock(m_queue_mutex);
                m_queue_cv.wait(lock, [this]() {
                    return m_stop_requested || !m_jobs.empty();
                });
                if (m_stop_requested && m_jobs.empty()) {
                    return;
                }
                job = m_jobs.front();
                m_jobs.pop_front();
                m_active_workers++;
            }

            // Align to host system page boundary
            uintptr_t uaddr = reinterpret_cast<uintptr_t>(job.addr);
            uintptr_t start = uaddr & ~page_mask;
            uintptr_t end   = (uaddr + job.len + page_mask) & ~page_mask;
            size_t aligned_len = end - start;

            if (aligned_len > 0) {
#if defined(__linux__)
                if (job.is_cold) {
                    madvise(reinterpret_cast<void *>(start), aligned_len, MADV_COLD);
                } else {
                    madvise(reinterpret_cast<void *>(start), aligned_len, MADV_WILLNEED);
                }
#elif defined(_POSIX_MAPPED_FILES)
                if (job.is_cold) {
#ifdef POSIX_MADV_DONTNEED
                    posix_madvise(reinterpret_cast<void *>(start), aligned_len, POSIX_MADV_DONTNEED);
#endif
                } else {
#ifdef POSIX_MADV_WILLNEED
                    posix_madvise(reinterpret_cast<void *>(start), aligned_len, POSIX_MADV_WILLNEED);
#endif
                }
#endif
            }

            {
                std::lock_guard<std::mutex> lock(m_queue_mutex);
                m_active_workers--;
                if (m_jobs.empty() && m_active_workers == 0) {
                    m_idle_cv.notify_all();
                }
            }
        }
    }

    static size_t get_sys_page_size() {
#if defined(_SC_PAGESIZE)
        long sz = sysconf(_SC_PAGESIZE);
        return (sz > 0) ? (size_t) sz : 4096;
#else
        return 4096;
#endif
    }

    mutable std::mutex m_registry_mutex;
    std::vector<mapped_region> m_mappings;

    mutable std::mutex m_layer_mutex;
    std::vector<moe_layer_meta> m_moe_layers;

    std::mutex m_queue_mutex;
    std::condition_variable m_queue_cv;
    std::condition_variable m_idle_cv;
    std::deque<prefetch_job> m_jobs;
    std::vector<std::thread> m_workers;
    std::atomic<bool> m_running;
    std::atomic<bool> m_stop_requested;
    int m_active_workers;
    std::atomic<uint64_t> m_epoch;
};

} // namespace

void ggml_moe_prefetch_register_mapping(const void * base, size_t size) {
    prefetch_service::instance().register_mapping(base, size);
}

void ggml_moe_prefetch_unregister_mapping(const void * base) {
    prefetch_service::instance().unregister_mapping(base);
}

void ggml_moe_prefetch_set_n_threads(int n_threads) {
    prefetch_service::instance().set_n_threads(n_threads);
}

bool ggml_moe_prefetch_enabled(void) {
    return prefetch_service::instance().is_enabled();
}

void ggml_moe_prefetch_new_epoch(void) {
    prefetch_service::instance().bump_epoch();
}

void ggml_moe_prefetch_slice(const void * addr, size_t size) {
    if (!addr || size == 0) return;
    auto & svc = prefetch_service::instance();
    if (!svc.is_enabled()) return;
    if (svc.is_registered(addr, size)) {
        svc.enqueue(const_cast<void *>(addr), size, /* is_cold = */ false);
    }
}

void ggml_moe_prefetch_tensor(const struct ggml_tensor * w) {
    if (!w || !w->data) return;
    if (w->buffer && !ggml_backend_buffer_is_host(w->buffer)) return;
    auto & svc = prefetch_service::instance();
    size_t nbytes = ggml_nbytes(w);
    if (svc.is_registered(w->data, nbytes)) {
        svc.enqueue(w->data, nbytes, /* is_cold = */ false);
    }
}

void ggml_moe_prefetch_cold(const struct ggml_tensor * w) {
    if (!w || !w->data) return;
    if (w->buffer && !ggml_backend_buffer_is_host(w->buffer)) return;
    auto & svc = prefetch_service::instance();
    size_t nbytes = ggml_nbytes(w);
    if (svc.is_registered(w->data, nbytes)) {
        svc.enqueue(w->data, nbytes, /* is_cold = */ true);
    }
}

void ggml_moe_prefetch_wait(const struct ggml_tensor * /*w*/) {
    prefetch_service::instance().wait_flush();
}

void ggml_moe_prefetch_node(const struct ggml_tensor * node) {
    if (!node || node->op != GGML_OP_MUL_MAT_ID) return;
    auto & svc = prefetch_service::instance();
    if (!svc.is_enabled()) return;

    const struct ggml_tensor * weights = node->src[0];
    const struct ggml_tensor * ids_tensor = node->src[2];

    if (!weights || !weights->data || !ids_tensor || !ids_tensor->data) return;

    // Safety: ensure both tensors are located in host (CPU) memory before accessing ->data
    if (weights->buffer && !ggml_backend_buffer_is_host(weights->buffer)) return;
    if (ids_tensor->buffer && !ggml_backend_buffer_is_host(ids_tensor->buffer)) return;

    // ids_tensor shape: [n_expert_used, n_tokens]
    const int64_t n_expert_used = ids_tensor->ne[0];
    const int64_t n_tokens      = ids_tensor->ne[1];
    const int64_t total_ids     = n_expert_used * n_tokens;

    if (total_ids <= 0) return;

    // Expert slice byte stride in weights tensor
    const size_t expert_stride = weights->nb[2];
    const int64_t n_experts = weights->ne[2];

    const int32_t * ids = reinterpret_cast<const int32_t *>(ids_tensor->data);

    // Enqueue prefetch for uniquely selected experts in this dispatch step
    uint64_t seen_mask = 0; // fast-path bitmask for up to 64 experts
    std::vector<bool> seen_vec;
    if (n_experts > 64) {
        seen_vec.assign((size_t) n_experts, false);
    }

    for (int64_t i = 0; i < total_ids; ++i) {
        int32_t exp = ids[i];
        if (exp < 0 || exp >= n_experts) continue;

        bool already_seen = false;
        if (n_experts <= 64) {
            uint64_t bit = (1ULL << exp);
            if (seen_mask & bit) {
                already_seen = true;
            } else {
                seen_mask |= bit;
            }
        } else {
            if (seen_vec[(size_t) exp]) {
                already_seen = true;
            } else {
                seen_vec[(size_t) exp] = true;
            }
        }

        if (already_seen) continue;

        void * exp_addr = reinterpret_cast<char *>(weights->data) + (size_t) exp * expert_stride;
        if (svc.is_registered(exp_addr, expert_stride)) {
            svc.enqueue(exp_addr, expert_stride, /* is_cold = */ false);
        }
    }
}

void ggml_moe_prefetch_kernel_hook(const struct ggml_tensor * node, int ith) {
    if (ith == 0) {
        ggml_moe_prefetch_node(node);
    }
}

void ggml_moe_prefetch_register_moe_layer(
    int32_t il,
    const struct ggml_tensor * gate_inp,
    const struct ggml_tensor * up_exps,
    const struct ggml_tensor * gate_exps,
    const struct ggml_tensor * down_exps,
    int64_t n_expert,
    int64_t n_expert_used) {
    prefetch_service::instance().register_moe_layer(il, gate_inp, up_exps, gate_exps, down_exps, n_expert, n_expert_used);
}

void ggml_moe_prefetch_clear_moe_layers(void) {
    prefetch_service::instance().clear_moe_layers();
}

void ggml_moe_prefetch_lookahead_step(int32_t current_il, const int32_t * current_ids, size_t n_ids) {
    if (!current_ids || n_ids == 0) return;
    prefetch_service::instance().lookahead_step(current_il, current_ids, n_ids);
}
