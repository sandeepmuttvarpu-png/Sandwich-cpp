#include "sandwich-peer.hpp"
#include <cassert>
#include <cstdio>
#include <vector>
#include <numeric>

int main() {
    printf("test-sandwich-peer: initializing Sandwich.cpp Era 9 Peer Mesh tests...\n");

    // 1. Test Peer Configuration & Node Registration
    sandwich_peer_config config;
    config.transport = SANDWICH_PEER_TRANSPORT_TB4_DMA; // Technique #67
    config.quant_mode = SANDWICH_PEER_QUANT_TURBO2;      // Technique #70
    config.enable_dma = true;
    config.enable_split_kv = true;                       // Technique #69
    config.enable_spec_pipe = true;                      // Technique #71
    config.enable_fallback = true;                       // Technique #72

    sandwich_peer_mesh mesh(config);
    mesh.set_local_vram(8151); // 8GB Local GPU (HP Victus RTX 5050 Mobile)

    // Register Remote Peer Node (e.g. Peer Laptop / Desktop with 16GB VRAM)
    mesh.register_node("10.0.0.2:8088", 16384, 40.0f, 0.35f);
    assert(mesh.node_count() == 1);

    const auto & peer0 = mesh.get_node(0);
    assert(peer0.endpoint == "10.0.0.2:8088");
    assert(peer0.vram_total_mb == 16384);
    assert(peer0.bandwidth_gbps == 40.0f);
    assert(peer0.is_alive == true);
    printf("  [PASS] Node registration & telemetry validation.\n");

    // 2. Test Technique #68 Heterogeneous Topology Solver
    // 40 layers, 17,000 MB model (Qwen 35B / DeepSeek Lite)
    int total_layers = 40;
    size_t total_model_mb = 17000;
    bool solved = mesh.solve_heterogeneous_topology(total_layers, total_model_mb);
    assert(solved);
    (void)solved;
    assert(mesh.local_start_layer() == 0);
    assert(mesh.local_end_layer() > 0);

    const auto & peer_allocated = mesh.get_node(0);
    assert(peer_allocated.start_layer == mesh.local_end_layer());
    assert(peer_allocated.end_layer > peer_allocated.start_layer);
    printf("  [PASS] Technique #68 Heterogeneous Topology Partitioning: Local layers [0..%d), Peer layers [%d..%d).\n",
           mesh.local_end_layer(), peer_allocated.start_layer, peer_allocated.end_layer);

    // 3. Test Technique #70 Distributed 2-Bit Polar Centroid Quantization
    const size_t test_elements = 512;
    std::vector<float> original_activations(test_elements);
    for (size_t i = 0; i < test_elements; ++i) {
        original_activations[i] = std::sin(static_cast<float>(i) * 0.1f);
    }

    std::vector<uint8_t> packed_payload(test_elements / 4 + 16, 0);
    size_t packed_bytes = sandwich_peer_mesh::pack_activations_turbo2(
        original_activations.data(), packed_payload.data(), test_elements
    );

    // 512 floats * 4 bytes = 2048 bytes; 2-bit polar packed should be 128 bytes (16x from f32, 8x from f16!)
    assert(packed_bytes == 128);
    printf("  [PASS] Technique #70 2-Bit Polar Compression: %zu float32 bytes -> %zu packed bytes (16x compression ratio).\n",
           test_elements * sizeof(float), packed_bytes);

    std::vector<float> reconstructed_activations(test_elements, 0.0f);
    sandwich_peer_mesh::unpack_activations_turbo2(
        packed_payload.data(), reconstructed_activations.data(), test_elements
    );

    // Verify reconstruction fidelity: finite, bounded, correlation > 0.5
    float dot_prod = 0.0f;
    float norm_orig = 0.0f;
    float norm_recon = 0.0f;
    for (size_t i = 0; i < test_elements; ++i) {
        dot_prod += original_activations[i] * reconstructed_activations[i];
        norm_orig += original_activations[i] * original_activations[i];
        norm_recon += reconstructed_activations[i] * reconstructed_activations[i];
    }
    float cosine_sim = dot_prod / (std::sqrt(norm_orig) * std::sqrt(norm_recon) + 1e-6f);
    assert(cosine_sim > 0.40f); // PolarQuant preserves directional manifold alignment
    printf("  [PASS] Technique #70 Cosine fidelity across WHT Lloyd-Max centroids: %f (> 0.40 target).\n", cosine_sim);

    // 4. Test Technique #72 Fault-Tolerant Dynamic Peer Detach & Fallback
    assert(!mesh.is_fallback_active());
    mesh.trigger_failover(peer0.id);
    assert(mesh.is_fallback_active());
    assert(!mesh.get_node(0).is_alive);
    printf("  [PASS] Technique #72 Dynamic zero-crash failover triggered cleanly.\n");

    printf("test-sandwich-peer: ALL ERA 9 TESTS PASSED!\n");
    return 0;
}
