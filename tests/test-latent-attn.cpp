#include "ggml.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <random>

static void fill_random_f32(float * data, size_t n, float min_val = -1.0f, float max_val = 1.0f) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(min_val, max_val);
    for (size_t i = 0; i < n; ++i) {
        data[i] = dist(rng);
    }
}

static bool test_latent_attn_dense(ggml_backend_t backend_cpu, ggml_backend_t backend_gpu, ggml_type cache_type) {
    const int64_t Dk = 64;
    const int64_t T  = 4;
    const int64_t H  = 2;
    const int64_t N  = 16;
    const int64_t P  = 4;
    const int64_t dv = 32;
    const int64_t dv_off = 0;
    const float scale = 1.0f / sqrtf((float)Dk);

    printf("Testing Mode 0 (Dense Latent Attn), cache_type=%s, Dk=%ld, T=%ld, H=%ld, N=%ld, P=%ld, dv=%ld...\n",
           ggml_type_name(cache_type), Dk, T, H, N, P, dv);

    auto run_on_backend = [&](ggml_backend_t backend, std::vector<float> & out_data) -> bool {
        struct ggml_init_params params = {
            /* .mem_size   = */ 16 * 1024 * 1024,
            /* .mem_buffer = */ NULL,
            /* .no_alloc   = */ true,
        };
        struct ggml_context * ctx = ggml_init(params);

        struct ggml_tensor * q        = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, Dk, T, H);
        struct ggml_tensor * cache    = ggml_new_tensor_2d(ctx, cache_type, Dk, N);
        struct ggml_tensor * prefix_k = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, Dk, P);
        struct ggml_tensor * prefix_v = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, P, dv);
        struct ggml_tensor * mask     = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N, T);

        struct ggml_tensor * dst = ggml_latent_attn_prefix_ext(ctx, q, cache, prefix_k, prefix_v, mask, dv, dv_off, scale, 0.0f);

        struct ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, dst);

        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (!buf) {
            fprintf(stderr, "Failed to allocate backend buffer\n");
            ggml_free(ctx);
            return false;
        }

        std::vector<float> q_host(ggml_nelements(q));
        fill_random_f32(q_host.data(), q_host.size());
        ggml_backend_tensor_set(q, q_host.data(), 0, ggml_nbytes(q));

        if (cache_type == GGML_TYPE_F32) {
            std::vector<float> cache_host(ggml_nelements(cache));
            fill_random_f32(cache_host.data(), cache_host.size());
            ggml_backend_tensor_set(cache, cache_host.data(), 0, ggml_nbytes(cache));
        } else if (cache_type == GGML_TYPE_F16) {
            std::vector<float> cache_f32(ggml_nelements(cache));
            fill_random_f32(cache_f32.data(), cache_f32.size());
            std::vector<ggml_fp16_t> cache_f16(cache_f32.size());
            for (size_t i = 0; i < cache_f32.size(); ++i) {
                cache_f16[i] = ggml_fp32_to_fp16(cache_f32[i]);
            }
            ggml_backend_tensor_set(cache, cache_f16.data(), 0, ggml_nbytes(cache));
        } else if (cache_type == GGML_TYPE_Q8_0) {
            std::vector<float> cache_f32(ggml_nelements(cache));
            fill_random_f32(cache_f32.data(), cache_f32.size());
            std::vector<uint8_t> cache_q8(ggml_nbytes(cache));
            ggml_quantize_chunk(GGML_TYPE_Q8_0, cache_f32.data(), cache_q8.data(), 0, N, Dk, nullptr);
            ggml_backend_tensor_set(cache, cache_q8.data(), 0, ggml_nbytes(cache));
        }

        std::vector<float> pk_host(ggml_nelements(prefix_k));
        fill_random_f32(pk_host.data(), pk_host.size());
        ggml_backend_tensor_set(prefix_k, pk_host.data(), 0, ggml_nbytes(prefix_k));

        std::vector<float> pv_host(ggml_nelements(prefix_v));
        fill_random_f32(pv_host.data(), pv_host.size());
        ggml_backend_tensor_set(prefix_v, pv_host.data(), 0, ggml_nbytes(prefix_v));

        std::vector<float> mask_host(ggml_nelements(mask), 0.0f);
        // Mask out future positions for causal attention
        for (int t = 0; t < T; ++t) {
            for (int n = 0; n < N; ++n) {
                if (n > t + 8) {
                    mask_host[t * N + n] = -INFINITY;
                }
            }
        }
        ggml_backend_tensor_set(mask, mask_host.data(), 0, ggml_nbytes(mask));

        ggml_backend_graph_compute(backend, gf);

        out_data.resize(ggml_nelements(dst));
        ggml_backend_tensor_get(dst, out_data.data(), 0, ggml_nbytes(dst));

        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        return true;
    };

    std::vector<float> out_cpu;
    if (!run_on_backend(backend_cpu, out_cpu)) {
        fprintf(stderr, "CPU run failed!\n");
        return false;
    }

    if (backend_gpu) {
        std::vector<float> out_gpu;
        if (!run_on_backend(backend_gpu, out_gpu)) {
            fprintf(stderr, "GPU run failed!\n");
            return false;
        }

        float max_diff = 0.0f;
        for (size_t i = 0; i < out_cpu.size(); ++i) {
            float diff = fabsf(out_cpu[i] - out_gpu[i]);
            if (diff > max_diff) {
                max_diff = diff;
            }
        }
        printf("  Max diff (CPU vs GPU): %e\n", max_diff);
        const float tol = (cache_type == GGML_TYPE_F32) ? 1e-3f : ((cache_type == GGML_TYPE_F16) ? 2e-3f : 5e-3f);
        if (max_diff > tol) {
            fprintf(stderr, "FAIL: max_diff %e > tol %e\n", max_diff, tol);
            return false;
        }
    }

    printf("  PASS\n");
    return true;
}

static bool test_latent_attn_indexed(ggml_backend_t backend_cpu, ggml_backend_t backend_gpu, ggml_type cache_type) {
    const int64_t Dk = 64;
    const int64_t T  = 4;
    const int64_t H  = 2;
    const int64_t N  = 32;
    const int64_t P  = 4;
    const int64_t topk = 8;
    const int64_t dv = 32;
    const int64_t dv_off = 0;
    const float scale = 1.0f / sqrtf((float)Dk);

    printf("Testing Mode 1 (Indexed Latent Attn), cache_type=%s, Dk=%ld, T=%ld, H=%ld, N=%ld, P=%ld, topk=%ld, dv=%ld...\n",
           ggml_type_name(cache_type), Dk, T, H, N, P, topk, dv);

    auto run_on_backend = [&](ggml_backend_t backend, std::vector<float> & out_data) -> bool {
        struct ggml_init_params params = {
            /* .mem_size   = */ 16 * 1024 * 1024,
            /* .mem_buffer = */ NULL,
            /* .no_alloc   = */ true,
        };
        struct ggml_context * ctx = ggml_init(params);

        struct ggml_tensor * q        = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, Dk, T, H);
        struct ggml_tensor * cache    = ggml_new_tensor_2d(ctx, cache_type, Dk, N);
        struct ggml_tensor * prefix_k = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, Dk, P);
        struct ggml_tensor * prefix_v = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, P, dv);
        struct ggml_tensor * mask     = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N, T);
        struct ggml_tensor * indices  = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, topk, T);

        struct ggml_tensor * dst = ggml_latent_attn_indexed_ext(ctx, q, cache, prefix_k, prefix_v, mask, indices, dv, dv_off, scale, 0.0f);

        struct ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, dst);

        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (!buf) {
            fprintf(stderr, "Failed to allocate backend buffer\n");
            ggml_free(ctx);
            return false;
        }

        std::vector<float> q_host(ggml_nelements(q));
        fill_random_f32(q_host.data(), q_host.size());
        ggml_backend_tensor_set(q, q_host.data(), 0, ggml_nbytes(q));

        if (cache_type == GGML_TYPE_F32) {
            std::vector<float> cache_host(ggml_nelements(cache));
            fill_random_f32(cache_host.data(), cache_host.size());
            ggml_backend_tensor_set(cache, cache_host.data(), 0, ggml_nbytes(cache));
        } else if (cache_type == GGML_TYPE_F16) {
            std::vector<float> cache_f32(ggml_nelements(cache));
            fill_random_f32(cache_f32.data(), cache_f32.size());
            std::vector<ggml_fp16_t> cache_f16(cache_f32.size());
            for (size_t i = 0; i < cache_f32.size(); ++i) {
                cache_f16[i] = ggml_fp32_to_fp16(cache_f32[i]);
            }
            ggml_backend_tensor_set(cache, cache_f16.data(), 0, ggml_nbytes(cache));
        } else if (cache_type == GGML_TYPE_Q8_0) {
            std::vector<float> cache_f32(ggml_nelements(cache));
            fill_random_f32(cache_f32.data(), cache_f32.size());
            std::vector<uint8_t> cache_q8(ggml_nbytes(cache));
            ggml_quantize_chunk(GGML_TYPE_Q8_0, cache_f32.data(), cache_q8.data(), 0, N, Dk, nullptr);
            ggml_backend_tensor_set(cache, cache_q8.data(), 0, ggml_nbytes(cache));
        }

        std::vector<float> pk_host(ggml_nelements(prefix_k));
        fill_random_f32(pk_host.data(), pk_host.size());
        ggml_backend_tensor_set(prefix_k, pk_host.data(), 0, ggml_nbytes(prefix_k));

        std::vector<float> pv_host(ggml_nelements(prefix_v));
        fill_random_f32(pv_host.data(), pv_host.size());
        ggml_backend_tensor_set(prefix_v, pv_host.data(), 0, ggml_nbytes(prefix_v));

        std::vector<float> mask_host(ggml_nelements(mask), 0.0f);
        ggml_backend_tensor_set(mask, mask_host.data(), 0, ggml_nbytes(mask));

        std::vector<int32_t> indices_host(ggml_nelements(indices));
        for (int t = 0; t < T; ++t) {
            for (int k = 0; k < topk; ++k) {
                indices_host[t * topk + k] = (k * 3 + t) % N;
            }
        }
        ggml_backend_tensor_set(indices, indices_host.data(), 0, ggml_nbytes(indices));

        ggml_backend_graph_compute(backend, gf);

        out_data.resize(ggml_nelements(dst));
        ggml_backend_tensor_get(dst, out_data.data(), 0, ggml_nbytes(dst));

        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        return true;
    };

    std::vector<float> out_cpu;
    if (!run_on_backend(backend_cpu, out_cpu)) {
        fprintf(stderr, "CPU run failed!\n");
        return false;
    }

    if (backend_gpu) {
        std::vector<float> out_gpu;
        if (!run_on_backend(backend_gpu, out_gpu)) {
            fprintf(stderr, "GPU run failed!\n");
            return false;
        }

        float max_diff = 0.0f;
        for (size_t i = 0; i < out_cpu.size(); ++i) {
            float diff = fabsf(out_cpu[i] - out_gpu[i]);
            if (diff > max_diff) {
                max_diff = diff;
            }
        }
        printf("  Max diff (CPU vs GPU): %e\n", max_diff);
        const float tol = (cache_type == GGML_TYPE_F32) ? 1e-3f : ((cache_type == GGML_TYPE_F16) ? 2e-3f : 5e-3f);
        if (max_diff > tol) {
            fprintf(stderr, "FAIL: max_diff %e > tol %e\n", max_diff, tol);
            return false;
        }
    }

    printf("  PASS\n");
    return true;
}

int main() {
    ggml_backend_load_all();

    ggml_backend_t backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, NULL);
    if (!backend_cpu) {
        fprintf(stderr, "Failed to initialize CPU backend\n");
        return 1;
    }

    ggml_backend_t backend_gpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, NULL);
    if (backend_gpu) {
        printf("GPU backend initialized: %s\n", ggml_backend_name(backend_gpu));
    } else {
        printf("No GPU backend found, running on CPU only.\n");
    }

    bool ok = true;
    ok &= test_latent_attn_dense(backend_cpu, backend_gpu, GGML_TYPE_F32);
    ok &= test_latent_attn_dense(backend_cpu, backend_gpu, GGML_TYPE_F16);
    ok &= test_latent_attn_dense(backend_cpu, backend_gpu, GGML_TYPE_Q8_0);
    ok &= test_latent_attn_indexed(backend_cpu, backend_gpu, GGML_TYPE_F32);
    ok &= test_latent_attn_indexed(backend_cpu, backend_gpu, GGML_TYPE_F16);
    ok &= test_latent_attn_indexed(backend_cpu, backend_gpu, GGML_TYPE_Q8_0);

    ggml_backend_free(backend_cpu);
    if (backend_gpu) {
        ggml_backend_free(backend_gpu);
    }

    if (ok) {
        printf("\ntest-latent-attn: ALL PASS\n");
        return 0;
    }

    fprintf(stderr, "\ntest-latent-attn: FAILED\n");
    return 1;
}
