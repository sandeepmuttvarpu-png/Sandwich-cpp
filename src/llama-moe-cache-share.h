#pragma once

#include <vector>
#include <deque>
#include <unordered_set>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <algorithm>
#include <functional>
#include <stdexcept>

#ifndef GGML_ASSERT
#include <cassert>
#define GGML_ASSERT(x) assert(x)
#endif

namespace llama_moe_cache_detail {

// Computes normalized per-device fractional shares from cumulative split points.
// Handles single device, asymmetric splits, zero fallback (equal distribution among eligible devices),
// and filters out ineligible devices (e.g. GPUs with no host-resident layers assigned).
inline std::vector<double> device_shares(
        const std::vector<float> & cumulative,
        const std::vector<bool>  & eligible) {
    if (cumulative.size() != eligible.size()) {
        throw std::invalid_argument("cumulative and eligible vectors must have the same size");
    }

    const size_t n_dev = cumulative.size();
    std::vector<double> shares(n_dev, 0.0);
    double prev_point = 0.0;

    for (size_t i = 0; i < n_dev; ++i) {
        const double cur_point = (std::isfinite(cumulative[i]) && cumulative[i] >= 0.0f)
            ? std::max(prev_point, (double) cumulative[i])
            : prev_point;
        shares[i] = std::max(0.0, cur_point - prev_point);
        prev_point = cur_point;
    }

    double total_share = 0.0;
    size_t n_eligible = 0;
    for (size_t i = 0; i < n_dev; ++i) {
        if (eligible[i]) {
            total_share += shares[i];
            n_eligible++;
        } else {
            shares[i] = 0.0;
        }
    }

    if (n_eligible == 0) {
        return shares;
    }

    // Fallback: if all cumulative values are zero or sum is non-positive, distribute equally
    if (!std::isfinite(total_share) || total_share <= 0.0) {
        const double uniform_share = 1.0 / (double) n_eligible;
        for (size_t i = 0; i < n_dev; ++i) {
            shares[i] = eligible[i] ? uniform_share : 0.0;
        }
        return shares;
    }

    // Normalize so shares of eligible devices sum to 1.0
    for (size_t i = 0; i < n_dev; ++i) {
        shares[i] = eligible[i] ? (shares[i] / total_share) : 0.0;
    }

    return shares;
}

// Phase 6: Fine-Grained MoE Topology Support (Micro-Experts)
// Detects architectures with thousands of fine-grained micro-experts (e.g. Qwen3.8-Flash-Next, DeepSeek V2/V3)
inline bool is_fine_grained_moe(int64_t n_expert, int64_t n_expert_used) {
    return (n_expert >= 256) || (n_expert > 0 && n_expert_used >= 16);
}

// Solves for the maximum number of expert slots that fit into a VRAM budget using binary search in O(log N)
// Avoids slow linear while loops when sizing caches for models with up to 24,576 micro-experts
inline int32_t compute_optimal_slots(
        size_t vram_budget,
        size_t expert_slice_bytes,
        size_t alignment,
        int32_t max_possible_slots) {
    if (vram_budget == 0 || expert_slice_bytes == 0 || max_possible_slots <= 0) {
        return 0;
    }
    int32_t low = 0;
    int32_t high = max_possible_slots;
    int32_t best = 0;
    while (low <= high) {
        int32_t mid = low + (high - low) / 2;
        const size_t raw_size = expert_slice_bytes * (size_t)(mid + 1);
        const size_t padded_size = alignment > 0 ? ((raw_size + alignment - 1) / alignment) * alignment : raw_size;
        if (padded_size <= vram_budget) {
            best = mid;
            low = mid + 1;
        } else {
            high = mid - 1;
        }
    }
    return best;
}

// Micro-expert routing map: maps routed micro-expert IDs to cache slots
inline int32_t map_micro_expert_to_slot(int32_t expert_id, int32_t n_slots) {
    if (n_slots <= 0) return -1;
    return expert_id % n_slots;
}

// Phase 12: 2Q & Ghost-Cache Replacement Policy Engine
enum class moe_slot_queue : uint8_t {
    QUEUE_FREE = 0,
    QUEUE_A1IN = 1, // Probation queue (FIFO)
    QUEUE_AM   = 2  // Main frequent queue (LRU)
};

struct moe_cache_lru {
    int32_t n_expert = 0;
    int32_t n_slots  = 0;
    bool    use_2q   = true;

    // Phase 6: Fast struct key to eliminate 64-bit hardware integer division on hot eviction path
    struct slot_key {
        int32_t il = -1;
        int32_t expert = -1;
    };

    struct fill {
        int32_t expert;
        int32_t slot;
    };

    struct eviction {
        int32_t il;
        int32_t expert;
        int32_t slot;
    };

    std::vector<int32_t *> slot_map; // [n_layer] data of the slot maps, -1 if the expert is not cached
    std::vector<slot_key>  key_of;   // [n_slots] layer and expert key, {-1, -1} if empty

    // doubly linked list of the slots
    std::vector<int32_t> prev;
    std::vector<int32_t> next;

    // standard single-queue LRU state (when use_2q == false)
    int32_t head = -1;
    int32_t tail = -1;

    // 2Q state (when use_2q == true)
    std::vector<moe_slot_queue> queue_of;
    int32_t head_free = -1;
    int32_t tail_free = -1;
    int32_t size_free = 0;

    int32_t head_in = -1;
    int32_t tail_in = -1;
    int32_t size_in = 0;

    int32_t head_am = -1;
    int32_t tail_am = -1;
    int32_t size_am = 0;

    int32_t k_in = 0;
    int32_t k_am = 0;

    // Ghost Cache (A1out) - metadata tracking evicted keys with zero weight storage
    size_t max_ghost_keys = 0;
    std::deque<uint64_t> ghost_queue;
    std::unordered_set<uint64_t> ghost_set;

    // Current batch lock tracking to prevent in-flight eviction of slots needed in the current plan() call
    std::vector<uint32_t> slot_locked;
    uint32_t plan_gen = 0;

    // Technique #57: Disjoint Peer Cache Guard (STRATA_DISJOINT_ADAPT)
    bool disjoint_guard_enabled = true;
    std::function<bool(int32_t il, int32_t expert)> peer_holds_fn;
    size_t stat_disjoint_conflicts_prevented = 0;

    bool peer_holds(int32_t il, int32_t expert) const {
        if (!disjoint_guard_enabled || !peer_holds_fn) return false;
        return peer_holds_fn(il, expert);
    }

    // Telemetry statistics
    size_t stat_ghost_hits = 0;
    size_t stat_promotions_2q = 0;
    size_t stat_pollution_quarantined = 0;

    std::vector<uint32_t> seen; // [n_expert]
    uint32_t seen_gen = 0;
    std::vector<int32_t> uniq;

    void list_remove(int32_t & l_head, int32_t & l_tail, int32_t & l_size, int32_t s) {
        if (s < 0) return;
        if (prev[s] >= 0) {
            next[prev[s]] = next[s];
        } else if (l_head == s) {
            l_head = next[s];
        }
        if (next[s] >= 0) {
            prev[next[s]] = prev[s];
        } else if (l_tail == s) {
            l_tail = prev[s];
        }
        prev[s] = -1;
        next[s] = -1;
        l_size--;
    }

    void list_push_tail(int32_t & l_head, int32_t & l_tail, int32_t & l_size, int32_t s) {
        prev[s] = l_tail;
        next[s] = -1;
        if (l_tail >= 0) {
            next[l_tail] = s;
        }
        l_tail = s;
        if (l_head < 0) {
            l_head = s;
        }
        l_size++;
    }

    int32_t list_pop_head(int32_t & l_head, int32_t & l_tail, int32_t & l_size) {
        if (l_head < 0) return -1;
        int32_t s = l_head;
        list_remove(l_head, l_tail, l_size, s);
        return s;
    }

    int32_t pop_unlocked(int32_t & l_head, int32_t & l_tail, int32_t & l_size) {
        int32_t curr = l_head;
        while (curr >= 0) {
            if (slot_locked[curr] != plan_gen) {
                list_remove(l_head, l_tail, l_size, curr);
                return curr;
            }
            curr = next[curr];
        }
        return -1;
    }

    static inline uint64_t make_ghost_key(int32_t il, int32_t expert) {
        return ((uint64_t)(uint32_t)il << 32) | (uint64_t)(uint32_t)expert;
    }

    bool in_ghost(int32_t il, int32_t expert) const {
        return ghost_set.count(make_ghost_key(il, expert)) > 0;
    }

    void add_ghost(int32_t il, int32_t expert) {
        if (il < 0 || expert < 0) return;
        uint64_t k = make_ghost_key(il, expert);
        if (ghost_set.insert(k).second) {
            ghost_queue.push_back(k);
            while (ghost_set.size() > max_ghost_keys && !ghost_queue.empty()) {
                uint64_t oldest = ghost_queue.front();
                ghost_queue.pop_front();
                ghost_set.erase(oldest);
            }
        }
    }

    void remove_ghost(int32_t il, int32_t expert) {
        uint64_t k = make_ghost_key(il, expert);
        ghost_set.erase(k);
        while (!ghost_queue.empty() && ghost_set.count(ghost_queue.front()) == 0) {
            ghost_queue.pop_front();
        }
    }

    void init(int32_t n_layer, int32_t n_expert, int32_t n_slots, bool use_2q = true) {
        this->n_expert = n_expert;
        this->n_slots  = n_slots;
        this->use_2q   = use_2q;

        slot_map.assign(n_layer, nullptr);
        key_of.assign(n_slots, { -1, -1 });
        prev.assign(n_slots, -1);
        next.assign(n_slots, -1);
        seen.assign(n_expert, 0);
        seen_gen = 0;

        stat_ghost_hits = 0;
        stat_promotions_2q = 0;
        stat_pollution_quarantined = 0;

        if (use_2q) {
            queue_of.assign(n_slots, moe_slot_queue::QUEUE_FREE);
            slot_locked.assign(n_slots, 0);
            plan_gen = 0;

            head_free = -1;
            tail_free = -1;
            size_free = 0;

            head_in = -1;
            tail_in = -1;
            size_in = 0;

            head_am = -1;
            tail_am = -1;
            size_am = 0;

            for (int32_t s = 0; s < n_slots; ++s) {
                list_push_tail(head_free, tail_free, size_free, s);
            }

            k_in = std::max(1, n_slots / 4);
            k_am = n_slots - k_in;
            max_ghost_keys = std::max((size_t)128, std::min((size_t)n_slots * 2, (size_t)2048));
            ghost_queue.clear();
            ghost_set.clear();
        } else {
            for (int32_t s = 0; s < n_slots; ++s) {
                prev[s] = s - 1;
                next[s] = s + 1 < n_slots ? s + 1 : -1;
            }
            head = n_slots > 0 ? 0 : -1;
            tail = n_slots > 0 ? n_slots - 1 : -1;
        }
    }

    // Standard LRU touch (move slot s to tail)
    void touch_lru(int32_t s) {
        if (s == tail) return;
        if (prev[s] >= 0) {
            next[prev[s]] = next[s];
        } else {
            head = next[s];
        }
        prev[next[s]] = prev[s];

        prev[s] = tail;
        next[s] = -1;
        next[tail] = s;
        tail = s;
    }

    bool plan(int32_t il, const int32_t * ids, size_t n_ids, std::vector<fill> & fills, std::vector<eviction> & evictions, size_t & n_hit) {
        fills.clear();
        evictions.clear();
        n_hit = 0;

        if (++seen_gen == 0) {
            std::fill(seen.begin(), seen.end(), 0);
            seen_gen = 1;
        }
        uniq.clear();
        for (size_t i = 0; i < n_ids; ++i) {
            GGML_ASSERT(ids[i] >= 0 && ids[i] < n_expert);
            if (seen[ids[i]] != seen_gen) {
                seen[ids[i]] = seen_gen;
                uniq.push_back(ids[i]);
            }
        }
        if (uniq.size() > (size_t) n_slots) {
            return false;
        }

        int32_t * slots = slot_map[il];

        if (!use_2q) {
            // Naive single-queue LRU fallback
            for (int32_t e : uniq) {
                if (slots[e] >= 0) {
                    touch_lru(slots[e]);
                    n_hit++;
                }
            }
            std::sort(uniq.begin(), uniq.end());
            for (int32_t e : uniq) {
                if (slots[e] >= 0) {
                    continue;
                }
                const int32_t s = head;
                if (key_of[s].il >= 0 && key_of[s].expert >= 0) {
                    evictions.push_back({ key_of[s].il, key_of[s].expert, s });
                    slot_map[key_of[s].il][key_of[s].expert] = -1;
                }
                key_of[s] = { il, e };
                slots[e] = s;
                touch_lru(s);
                fills.push_back({ e, s });
            }
            return true;
        }

        // Phase 12: 2Q & Ghost-Cache Replacement Policy
        if (++plan_gen == 0) {
            std::fill(slot_locked.begin(), slot_locked.end(), 0);
            plan_gen = 1;
        }

        // Step 1: Process HITS
        for (int32_t e : uniq) {
            if (slots[e] >= 0) {
                const int32_t s = slots[e];
                n_hit++;
                slot_locked[s] = plan_gen;

                if (queue_of[s] == moe_slot_queue::QUEUE_A1IN) {
                    // Hit while in probation: recurrent access proven -> promote to Am!
                    list_remove(head_in, tail_in, size_in, s);
                    list_push_tail(head_am, tail_am, size_am, s);
                    queue_of[s] = moe_slot_queue::QUEUE_AM;
                    stat_promotions_2q++;
                } else if (queue_of[s] == moe_slot_queue::QUEUE_AM) {
                    // Hit in frequent queue: touch to MRU of Am
                    list_remove(head_am, tail_am, size_am, s);
                    list_push_tail(head_am, tail_am, size_am, s);
                }
            }
        }

        // Step 2: Process MISSES
        std::sort(uniq.begin(), uniq.end());
        for (int32_t e : uniq) {
            if (slots[e] >= 0) {
                continue;
            }

            // Technique #57: Disjoint Peer Cache Guard (STRATA_DISJOINT_ADAPT)
            // If peer tier holds this expert, skip local VRAM promotion to eliminate duplicate slots
            if (disjoint_guard_enabled && peer_holds(il, e)) {
                stat_disjoint_conflicts_prevented++;
                continue;
            }

            const bool is_ghost = in_ghost(il, e);
            if (is_ghost) {
                stat_ghost_hits++;
                remove_ghost(il, e);
            }

            int32_t s = -1;
            // 1. Take free slot if available
            if (size_free > 0) {
                s = list_pop_head(head_free, tail_free, size_free);
            } else if (is_ghost) {
                // Ghost hit target is Am: evict from Am if over capacity, else A1in, else Am
                if (size_am > k_am) {
                    s = pop_unlocked(head_am, tail_am, size_am);
                }
                if (s < 0 && size_in > 0) {
                    s = pop_unlocked(head_in, tail_in, size_in);
                }
                if (s < 0 && size_am > 0) {
                    s = pop_unlocked(head_am, tail_am, size_am);
                }
            } else {
                // Cold miss target is A1in (probation): evict from A1in if at capacity, else Am, else A1in
                if (size_in >= k_in) {
                    s = pop_unlocked(head_in, tail_in, size_in);
                }
                if (s < 0 && size_am > 0) {
                    s = pop_unlocked(head_am, tail_am, size_am);
                }
                if (s < 0 && size_in > 0) {
                    s = pop_unlocked(head_in, tail_in, size_in);
                }
            }

            GGML_ASSERT(s >= 0 && "MoE cache 2Q allocator could not find an unlocked slot");

            const bool was_in_a1in = (queue_of[s] == moe_slot_queue::QUEUE_A1IN);
            if (key_of[s].il >= 0 && key_of[s].expert >= 0) {
                evictions.push_back({ key_of[s].il, key_of[s].expert, s });
                slot_map[key_of[s].il][key_of[s].expert] = -1;
                if (was_in_a1in) {
                    // Evicted from probation without second reference: add key to Ghost Cache
                    add_ghost(key_of[s].il, key_of[s].expert);
                    stat_pollution_quarantined++;
                }
            }

            key_of[s] = { il, e };
            slots[e] = s;
            slot_locked[s] = plan_gen;

            if (is_ghost) {
                // Direct promotion into Am
                queue_of[s] = moe_slot_queue::QUEUE_AM;
                list_push_tail(head_am, tail_am, size_am, s);
            } else {
                // New transient candidate into A1in (probation)
                queue_of[s] = moe_slot_queue::QUEUE_A1IN;
                list_push_tail(head_in, tail_in, size_in, s);
            }
            fills.push_back({ e, s });
        }

        return true;
    }
};

} // namespace llama_moe_cache_detail
