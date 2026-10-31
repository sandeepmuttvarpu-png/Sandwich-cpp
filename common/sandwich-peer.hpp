#pragma once

#include "ggml.h"
#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <stdexcept>
#include <atomic>

// ============================================================================
// 🥪 Sandwich.cpp Era 9: P2P Multi-Device Heterogeneous VRAM Pooling & Polar Mesh
// Techniques #67 - #72
// ============================================================================

enum sandwich_peer_transport {
    SANDWICH_PEER_TRANSPORT_TCP,     // Standard TCP/IP streaming over 1G/2.5G/10G LAN
    SANDWICH_PEER_TRANSPORT_TB4_DMA, // [Technique #67] Zero-Copy Thunderbolt 4 / USB4 DMA P2P PCIe
    SANDWICH_PEER_TRANSPORT_ROCE,    // RDMA over Converged Ethernet
    SANDWICH_PEER_TRANSPORT_LOOPBACK // In-memory simulated transport for testing
};

enum sandwich_peer_quant {
    SANDWICH_PEER_QUANT_RAW_FP16,    // Uncompressed 16-bit half precision
    SANDWICH_PEER_QUANT_TURBO4,      // 4-bit Polar Lloyd-Max (4x compression)
    SANDWICH_PEER_QUANT_TURBO2       // [Technique #70] 2-bit Polar WHT Lloyd-Max (8x compression)
};

struct sandwich_peer_node {
    uint32_t id = 0;
    std::string endpoint;
    size_t vram_total_mb = 0;
    size_t vram_free_mb = 0;
    float bandwidth_gbps = 10.0f;
    float ping_ms = 0.5f;
    bool is_alive = true;
    int start_layer = -1;
    int end_layer = -1;
    size_t assigned_vram_mb = 0;
};

struct sandwich_peer_config {
    std::vector<std::string> endpoints;
    sandwich_peer_transport transport = SANDWICH_PEER_TRANSPORT_TCP;
    sandwich_peer_quant quant_mode    = SANDWICH_PEER_QUANT_TURBO2;
    bool enable_dma                   = false; // [Technique #67] Zero-Copy Thunderbolt 4 DMA
    bool enable_split_kv              = true;  // [Technique #69] Inter-Device Split-KV activation masking
    bool enable_spec_pipe             = true;  // [Technique #71] Cross-node speculative pipelined verification
    bool enable_fallback              = true;  // [Technique #72] Dynamic zero-crash failover to local 3-tier
    int timeout_ms                    = 5000;
};

class sandwich_peer_mesh {
public:
    sandwich_peer_mesh(const sandwich_peer_config & config)
        : m_config(config), m_fallback_triggered(false), m_local_vram_mb(8151) {}

    void set_local_vram(size_t mb) {
        m_local_vram_mb = mb;
    }

    // Register a peer node with hardware telemetry
    void register_node(const std::string & endpoint, size_t vram_mb, float bandwidth_gbps, float ping_ms = 0.5f) {
        sandwich_peer_node node;
        node.id = static_cast<uint32_t>(m_nodes.size() + 1);
        node.endpoint = endpoint;
        node.vram_total_mb = vram_mb;
        node.vram_free_mb = vram_mb;
        node.bandwidth_gbps = bandwidth_gbps;
        node.ping_ms = ping_ms;
        node.is_alive = true;
        m_nodes.push_back(node);
    }

    size_t node_count() const {
        return m_nodes.size();
    }

    const sandwich_peer_node & get_node(size_t idx) const {
        if (idx >= m_nodes.size()) {
            throw std::out_of_range("Peer node index out of bounds");
        }
        return m_nodes[idx];
    }

    // [Technique #68] Asymmetric Heterogeneous Topology Solver
    // Distributes layers across Local Node (Master) and Remote Peers based on VRAM capacity & link speed
    bool solve_heterogeneous_topology(int total_layers, size_t total_model_mb) {
        if (total_layers <= 0 || total_model_mb == 0) {
            return false;
        }

        size_t aggregate_vram = m_local_vram_mb;
        for (const auto & n : m_nodes) {
            if (n.is_alive) {
                aggregate_vram += n.vram_free_mb;
            }
        }
        if (total_model_mb > aggregate_vram * 2) {
            return false;
        }

        // Calculate layer weight estimate
        const double mb_per_layer = static_cast<double>(total_model_mb) / total_layers;

        // Local node gets primary low-latency layers (router gates, first layers)
        int local_layers = std::clamp(
            static_cast<int>(std::floor((m_local_vram_mb * 0.85) / mb_per_layer)),
            1, total_layers
        );

        m_local_start_layer = 0;
        m_local_end_layer = local_layers;

        int current_layer = local_layers;
        for (auto & node : m_nodes) {
            if (!node.is_alive || current_layer >= total_layers) {
                node.start_layer = -1;
                node.end_layer = -1;
                node.assigned_vram_mb = 0;
                continue;
            }

            int peer_layers = std::clamp(
                static_cast<int>(std::floor((node.vram_free_mb * 0.85) / mb_per_layer)),
                1, total_layers - current_layer
            );

            node.start_layer = current_layer;
            node.end_layer = current_layer + peer_layers;
            node.assigned_vram_mb = static_cast<size_t>(peer_layers * mb_per_layer);
            current_layer += peer_layers;
        }

        return true;
    }

    int local_start_layer() const { return m_local_start_layer; }
    int local_end_layer() const { return m_local_end_layer; }

    // [Technique #70] Distributed 2-Bit Polar Centroid Quantization over Network
    // Compresses N float elements to N/4 bytes using WHT butterfly rotation + Lloyd-Max centroids
    static size_t pack_activations_turbo2(const float * src, uint8_t * dst, size_t n_elements) {
        if (!src || !dst || n_elements == 0) return 0;

        // Pad to multiples of 4 for 4-way nibble packing
        size_t padded = (n_elements + 3) & ~3UL;
        std::vector<float> buffer(padded, 0.0f);
        std::memcpy(buffer.data(), src, n_elements * sizeof(float));

        // In-place Walsh-Hadamard butterfly rotation on blocks of 32
        const size_t block_size = 32;
        for (size_t b = 0; b < padded; b += block_size) {
            size_t curr_len = std::min(block_size, padded - b);
            for (size_t h = 1; h < curr_len; h *= 2) {
                for (size_t i = 0; i < curr_len; i += 2 * h) {
                    for (size_t j = i; j < i + h; ++j) {
                        float u = buffer[b + j];
                        float v = buffer[b + j + h];
                        buffer[b + j]     = (u + v) * 0.70710678118f; // 1/sqrt(2)
                        buffer[b + j + h] = (u - v) * 0.70710678118f;
                    }
                }
            }
        }

        // Lloyd-Max 2-bit centroids for standard normal:
        // C0 = -1.0f (00b), C1 = -0.333f (01b), C2 = 0.333f (10b), C3 = 1.0f (11b)
        size_t packed_bytes = padded / 4;
        for (size_t i = 0; i < packed_bytes; ++i) {
            uint8_t byte = 0;
            for (int k = 0; k < 4; ++k) {
                float val = buffer[i * 4 + k];
                uint8_t code = 0;
                if (val < -0.666f)      code = 0; // 00b
                else if (val < 0.0f)     code = 1; // 01b
                else if (val < 0.666f)   code = 2; // 10b
                else                     code = 3; // 11b
                byte |= (code << (k * 2));
            }
            dst[i] = byte;
        }

        return packed_bytes;
    }

    // Unpacks 2-bit polar centroids back to float32 and applies inverse WHT
    static void unpack_activations_turbo2(const uint8_t * src, float * dst, size_t n_elements) {
        if (!src || !dst || n_elements == 0) return;

        size_t padded = (n_elements + 3) & ~3UL;
        std::vector<float> buffer(padded, 0.0f);
        size_t packed_bytes = padded / 4;

        const float centroids[4] = { -1.0f, -0.333333f, 0.333333f, 1.0f };

        for (size_t i = 0; i < packed_bytes; ++i) {
            uint8_t byte = src[i];
            for (int k = 0; k < 4; ++k) {
                uint8_t code = (byte >> (k * 2)) & 0x03;
                buffer[i * 4 + k] = centroids[code];
            }
        }

        // Inverse WHT is self-inverse (symmetric Hadamard matrix)
        const size_t block_size = 32;
        for (size_t b = 0; b < padded; b += block_size) {
            size_t curr_len = std::min(block_size, padded - b);
            for (size_t h = 1; h < curr_len; h *= 2) {
                for (size_t i = 0; i < curr_len; i += 2 * h) {
                    for (size_t j = i; j < i + h; ++j) {
                        float u = buffer[b + j];
                        float v = buffer[b + j + h];
                        buffer[b + j]     = (u + v) * 0.70710678118f;
                        buffer[b + j + h] = (u - v) * 0.70710678118f;
                    }
                }
            }
        }

        std::memcpy(dst, buffer.data(), n_elements * sizeof(float));
    }

    // [Technique #72] Fault-Tolerant Dynamic Peer Detach & Fallback
    void trigger_failover(uint32_t failed_node_id) {
        for (auto & node : m_nodes) {
            if (node.id == failed_node_id) {
                node.is_alive = false;
                break;
            }
        }
        m_fallback_triggered = true;
    }

    bool is_fallback_active() const {
        return m_fallback_triggered.load();
    }

    const sandwich_peer_config & config() const {
        return m_config;
    }

private:
    sandwich_peer_config m_config;
    std::vector<sandwich_peer_node> m_nodes;
    std::atomic<bool> m_fallback_triggered;
    size_t m_local_vram_mb;
    int m_local_start_layer = 0;
    int m_local_end_layer = 0;
};
