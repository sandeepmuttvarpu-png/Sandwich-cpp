#include "ggml-iqk.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <chrono>

struct test_q8_K {
    float   d;
    int8_t  qs[256];
    int16_t bsums[16];
};

struct test_q8_0 {
    uint16_t d; // ggml_half
    int8_t   qs[32];
};

static uint16_t float_to_half_local(float x) {
    uint32_t val;
    __builtin_memcpy(&val, &x, sizeof(float));
    uint16_t sign = (val >> 16) & 0x8000;
    int32_t  exp  = ((val >> 23) & 0xff) - 127 + 15;
    uint32_t mant = val & 0x7fffff;

    if (exp <= 0) {
        return sign;
    } else if (exp >= 31) {
        return sign | 0x7c00;
    }
    return sign | ((uint16_t)exp << 10) | (uint16_t)(mant >> 13);
}

static void fill_random_bytes(uint8_t * p, size_t n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(0, 255);
    for (size_t i = 0; i < n; ++i) {
        p[i] = (uint8_t)dist(rng);
    }
}

static bool test_dequantization() {
    printf("--- Testing Dequantization Math ---\n");

    const int64_t K = 256;
    float y_ref[K];
    float y_opt[K];

    // 1. IQ2_KS
    block_iq2_ks b2;
    fill_random_bytes((uint8_t *)&b2, sizeof(b2), 101);
    iqk_set_simd_override(IQK_SIMD_GENERIC);
    iqk_dequantize_row_iq2_ks(&b2, y_ref, K, 0.125f);
    iqk_set_simd_override(iqk_get_simd_type());
    iqk_dequantize_row_iq2_ks(&b2, y_opt, K, 0.125f);

    float max_diff_2 = 0.0f;
    for (int i = 0; i < K; ++i) {
        max_diff_2 = std::max(max_diff_2, std::abs(y_ref[i] - y_opt[i]));
    }
    printf("  IQ2_KS dequant max diff: %e\n", max_diff_2);
    if (max_diff_2 > 1e-5f) return false;

    // 2. IQ3_KS
    block_iq3_ks b3;
    fill_random_bytes((uint8_t *)&b3, sizeof(b3), 102);
    iqk_set_simd_override(IQK_SIMD_GENERIC);
    iqk_dequantize_row_iq3_ks(&b3, y_ref, K, 0.0625f);
    iqk_set_simd_override(iqk_get_simd_type());
    iqk_dequantize_row_iq3_ks(&b3, y_opt, K, 0.0625f);

    float max_diff_3 = 0.0f;
    for (int i = 0; i < K; ++i) {
        max_diff_3 = std::max(max_diff_3, std::abs(y_ref[i] - y_opt[i]));
    }
    printf("  IQ3_KS dequant max diff: %e\n", max_diff_3);
    if (max_diff_3 > 1e-5f) return false;

    // 3. IQ4_KS
    block_iq4_ks b4;
    fill_random_bytes((uint8_t *)&b4, sizeof(b4), 103);
    iqk_set_simd_override(IQK_SIMD_GENERIC);
    iqk_dequantize_row_iq4_ks(&b4, y_ref, K, 0.03125f);
    iqk_set_simd_override(iqk_get_simd_type());
    iqk_dequantize_row_iq4_ks(&b4, y_opt, K, 0.03125f);

    float max_diff_4 = 0.0f;
    for (int i = 0; i < K; ++i) {
        max_diff_4 = std::max(max_diff_4, std::abs(y_ref[i] - y_opt[i]));
    }
    printf("  IQ4_KS dequant max diff: %e\n", max_diff_4);
    if (max_diff_4 > 1e-5f) return false;

    // 4. IQ4_KS_R16
    block_iq4_ks_r16 b4_r16[8]; // 8 blocks x 32 cols = 256 cols
    fill_random_bytes((uint8_t *)b4_r16, sizeof(b4_r16), 104);
    float d16[16];
    for (int r = 0; r < 16; ++r) d16[r] = 0.1f * (r + 1);

    std::vector<float> yr16_ref(16 * K);
    std::vector<float> yr16_opt(16 * K);

    iqk_set_simd_override(IQK_SIMD_GENERIC);
    iqk_dequantize_row_iq4_ks_r16(b4_r16, yr16_ref.data(), K, d16);
    iqk_set_simd_override(iqk_get_simd_type());
    iqk_dequantize_row_iq4_ks_r16(b4_r16, yr16_opt.data(), K, d16);

    float max_diff_r16 = 0.0f;
    for (size_t i = 0; i < yr16_ref.size(); ++i) {
        max_diff_r16 = std::max(max_diff_r16, std::abs(yr16_ref[i] - yr16_opt[i]));
    }
    printf("  IQ4_KS_R16 dequant max diff: %e\n", max_diff_r16);
    if (max_diff_r16 > 1e-5f) return false;

    printf("  PASS\n");
    return true;
}

static bool test_dot_products() {
    printf("--- Testing Vector Dot Products ---\n");

    const int n = 512; // 2 super-blocks of 256
    const int nb = n / 256;

    test_q8_K y_k[nb];
    for (int b = 0; b < nb; ++b) {
        y_k[b].d = 0.05f;
        fill_random_bytes((uint8_t *)y_k[b].qs, sizeof(y_k[b].qs), 200 + b);
    }

    // 1. IQ2_KS
    block_iq2_ks x2[nb];
    fill_random_bytes((uint8_t *)x2, sizeof(x2), 301);
    float s2_ref = 0.0f, s2_opt = 0.0f;
    iqk_set_simd_override(IQK_SIMD_GENERIC);
    iqk_vec_dot_iq2_ks_q8_k(n, &s2_ref, x2, y_k, 0.125f);
    iqk_set_simd_override(iqk_get_simd_type());
    iqk_vec_dot_iq2_ks_q8_k(n, &s2_opt, x2, y_k, 0.125f);
    printf("  IQ2_KS dot product diff: %e (ref=%f, opt=%f)\n", std::abs(s2_ref - s2_opt), s2_ref, s2_opt);
    if (std::abs(s2_ref - s2_opt) > 1e-3f) return false;

    // 2. IQ3_KS
    block_iq3_ks x3[nb];
    fill_random_bytes((uint8_t *)x3, sizeof(x3), 302);
    float s3_ref = 0.0f, s3_opt = 0.0f;
    iqk_set_simd_override(IQK_SIMD_GENERIC);
    iqk_vec_dot_iq3_ks_q8_k(n, &s3_ref, x3, y_k, 0.0625f);
    iqk_set_simd_override(iqk_get_simd_type());
    iqk_vec_dot_iq3_ks_q8_k(n, &s3_opt, x3, y_k, 0.0625f);
    printf("  IQ3_KS dot product diff: %e (ref=%f, opt=%f)\n", std::abs(s3_ref - s3_opt), s3_ref, s3_opt);
    if (std::abs(s3_ref - s3_opt) > 1e-3f) return false;

    // 3. IQ4_KS
    block_iq4_ks x4[nb];
    fill_random_bytes((uint8_t *)x4, sizeof(x4), 303);
    float s4_ref = 0.0f, s4_opt = 0.0f;
    iqk_set_simd_override(IQK_SIMD_GENERIC);
    iqk_vec_dot_iq4_ks_q8_k(n, &s4_ref, x4, y_k, 0.03125f);
    iqk_set_simd_override(iqk_get_simd_type());
    iqk_vec_dot_iq4_ks_q8_k(n, &s4_opt, x4, y_k, 0.03125f);
    printf("  IQ4_KS dot product diff: %e (ref=%f, opt=%f)\n", std::abs(s4_ref - s4_opt), s4_ref, s4_opt);
    if (std::abs(s4_ref - s4_opt) > 1e-3f) return false;

    // 4. IQ4_KS_R16
    const int nb_r16 = n / 32;
    std::vector<block_iq4_ks_r16> x4_r16(nb_r16);
    fill_random_bytes((uint8_t *)x4_r16.data(), x4_r16.size() * sizeof(block_iq4_ks_r16), 304);
    std::vector<test_q8_0> y_0(nb_r16);
    for (int b = 0; b < nb_r16; ++b) {
        y_0[b].d = float_to_half_local(0.04f);
        fill_random_bytes((uint8_t *)y_0[b].qs, sizeof(y_0[b].qs), 400 + b);
    }
    float d16[16];
    for (int r = 0; r < 16; ++r) d16[r] = 0.1f * (r + 1);

    float s16_ref[16] = {0};
    float s16_opt[16] = {0};
    iqk_set_simd_override(IQK_SIMD_GENERIC);
    iqk_vec_dot_iq4_ks_r16_q8_0(n, s16_ref, x4_r16.data(), y_0.data(), d16);
    iqk_set_simd_override(iqk_get_simd_type());
    iqk_vec_dot_iq4_ks_r16_q8_0(n, s16_opt, x4_r16.data(), y_0.data(), d16);

    float max_diff_s16 = 0.0f;
    for (int r = 0; r < 16; ++r) {
        max_diff_s16 = std::max(max_diff_s16, std::abs(s16_ref[r] - s16_opt[r]));
    }
    printf("  IQ4_KS_R16 dot product max diff: %e\n", max_diff_s16);
    if (max_diff_s16 > 1e-3f) return false;

    printf("  PASS\n");
    return true;
}

static bool test_hybrid_gemm() {
    printf("--- Testing Hybrid Offloaded CPU GEMM ---\n");

    const int nr = 32;
    const int nc = 4;
    const int k  = 256;

    std::vector<block_iq4_ks> A(nr);
    fill_random_bytes((uint8_t *)A.data(), A.size() * sizeof(block_iq4_ks), 501);

    std::vector<test_q8_K> B(nc);
    for (int c = 0; c < nc; ++c) {
        B[c].d = 0.02f;
        fill_random_bytes((uint8_t *)B[c].qs, sizeof(B[c].qs), 601 + c);
    }

    std::vector<float> C_single(nr * nc, 0.0f);
    std::vector<float> C_multi(nr * nc, 0.0f);

    float scales[nr];
    for (int r = 0; r < nr; ++r) scales[r] = 0.1f;

    // Run single-threaded
    iqk_gemm_hybrid_cpu(
        IQK_QUANT_TYPE_IQ4_KS,
        nr, nc, k,
        A.data(), sizeof(block_iq4_ks),
        B.data(), sizeof(test_q8_K),
        C_single.data(), nc,
        scales,
        0, 1
    );

    // Run 4-threaded
    for (int ith = 0; ith < 4; ++ith) {
        iqk_gemm_hybrid_cpu(
            IQK_QUANT_TYPE_IQ4_KS,
            nr, nc, k,
            A.data(), sizeof(block_iq4_ks),
            B.data(), sizeof(test_q8_K),
            C_multi.data(), nc,
            scales,
            ith, 4
        );
    }

    float max_diff = 0.0f;
    for (size_t i = 0; i < C_single.size(); ++i) {
        max_diff = std::max(max_diff, std::abs(C_single[i] - C_multi[i]));
    }
    printf("  Multi-threaded GEMM consistency max diff: %e\n", max_diff);
    if (max_diff > 1e-5f) return false;

    printf("  PASS\n");
    return true;
}

int main() {
    printf("=======================================================\n");
    printf("🥪 [Sandwich.cpp] SOTA IQK Engine & Hybrid CPU Offload\n");
    printf("=======================================================\n");
    printf("Active SIMD Architecture: %s (id=%d)\n\n", iqk_get_simd_name(), (int)iqk_get_simd_type());

    bool ok = true;
    ok &= test_dequantization();
    ok &= test_dot_products();
    ok &= test_hybrid_gemm();

    if (ok) {
        printf("\ntest-iqk: ALL PASS (100%% verified)\n");
        return 0;
    } else {
        fprintf(stderr, "\ntest-iqk: FAILED\n");
        return 1;
    }
}
