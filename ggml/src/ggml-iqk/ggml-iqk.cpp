#include "ggml-iqk.h"
#include "iqk-internal.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <atomic>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#endif

// =============================================================================
// Function Pointers for Dispatch
// =============================================================================

typedef void (*dequant_iq2_ks_t)(const block_iq2_ks *, float *, int64_t, float);
typedef void (*dequant_iq3_ks_t)(const block_iq3_ks *, float *, int64_t, float);
typedef void (*dequant_iq4_ks_t)(const block_iq4_ks *, float *, int64_t, float);
typedef void (*dequant_iq4_ks_r16_t)(const block_iq4_ks_r16 *, float *, int64_t, const float *);

typedef void (*vec_dot_iq2_ks_t)(int, float *, const void *, const void *, float);
typedef void (*vec_dot_iq3_ks_t)(int, float *, const void *, const void *, float);
typedef void (*vec_dot_iq4_ks_t)(int, float *, const void *, const void *, float);
typedef void (*vec_dot_iq4_ks_r16_t)(int, float *, const void *, const void *, const float *);

static dequant_iq2_ks_t     s_dequant_iq2_ks     = iqk_dequantize_row_iq2_ks_generic;
static dequant_iq3_ks_t     s_dequant_iq3_ks     = iqk_dequantize_row_iq3_ks_generic;
static dequant_iq4_ks_t     s_dequant_iq4_ks     = iqk_dequantize_row_iq4_ks_generic;
static dequant_iq4_ks_r16_t s_dequant_iq4_ks_r16 = iqk_dequantize_row_iq4_ks_r16_generic;

static vec_dot_iq2_ks_t     s_vec_dot_iq2_ks     = iqk_vec_dot_iq2_ks_q8_k_generic;
static vec_dot_iq3_ks_t     s_vec_dot_iq3_ks     = iqk_vec_dot_iq3_ks_q8_k_generic;
static vec_dot_iq4_ks_t     s_vec_dot_iq4_ks     = iqk_vec_dot_iq4_ks_q8_k_generic;
static vec_dot_iq4_ks_r16_t s_vec_dot_iq4_ks_r16 = iqk_vec_dot_iq4_ks_r16_q8_0_generic;

static iqk_simd_type s_active_simd = IQK_SIMD_GENERIC;
static std::atomic<bool> s_initialized{false};

// =============================================================================
// CPUID Feature Detection
// =============================================================================

static void iqk_init_dispatch(void) {
    if (s_initialized.exchange(true)) {
        return;
    }

    iqk_simd_type detected = IQK_SIMD_GENERIC;

#if defined(__x86_64__) || defined(_M_X64)
    unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
#if defined(_MSC_VER)
    int cpuInfo[4];
    __cpuid(cpuInfo, 1);
    ecx = cpuInfo[2];
    edx = cpuInfo[3];
#else
    __cpuid(1, eax, ebx, ecx, edx);
#endif

    const bool has_avx = (ecx & (1 << 28)) != 0;

    if (has_avx) {
        unsigned int max_subleaf = 0;
#if defined(_MSC_VER)
        __cpuidex(cpuInfo, 7, 0);
        max_subleaf = cpuInfo[0];
        ebx = cpuInfo[1];
        ecx = cpuInfo[2];
        edx = cpuInfo[3];
#else
        __cpuid_count(7, 0, eax, ebx, ecx, edx);
        max_subleaf = eax;
#endif
        const bool has_avx2        = (ebx & (1 << 5)) != 0;
        const bool has_avx512f     = (ebx & (1 << 16)) != 0;
        const bool has_avx512bw    = (ebx & (1 << 30)) != 0;
        const bool has_avx512vl    = (ebx & (1 << 31)) != 0;
        const bool has_avx512_vnni = (ecx & (1 << 11)) != 0;

        bool has_avx_vnni = false;
        if (max_subleaf >= 1) {
#if defined(_MSC_VER)
            __cpuidex(cpuInfo, 7, 1);
            has_avx_vnni = (cpuInfo[0] & (1 << 4)) != 0;
#else
            unsigned int eax1 = 0, ebx1 = 0, ecx1 = 0, edx1 = 0;
            __cpuid_count(7, 1, eax1, ebx1, ecx1, edx1);
            has_avx_vnni = (eax1 & (1 << 4)) != 0;
#endif
        }

        if (has_avx512f && has_avx512bw && has_avx512vl && has_avx512_vnni) {
            detected = IQK_SIMD_AVX512_VNNI;
        } else if (has_avx2 && has_avx_vnni) {
            detected = IQK_SIMD_AVX_VNNI;
        } else if (has_avx2) {
            detected = IQK_SIMD_AVX2;
        }
    }
#endif

    iqk_set_simd_override(detected);
}

void iqk_set_simd_override(iqk_simd_type type) {
    s_active_simd = type;

    switch (type) {
        case IQK_SIMD_AVX512_VNNI:
            s_dequant_iq2_ks     = iqk_dequantize_row_iq2_ks_avx512;
            s_dequant_iq3_ks     = iqk_dequantize_row_iq3_ks_avx512;
            s_dequant_iq4_ks     = iqk_dequantize_row_iq4_ks_avx512;
            s_dequant_iq4_ks_r16 = iqk_dequantize_row_iq4_ks_r16_avx512;

            s_vec_dot_iq2_ks     = iqk_vec_dot_iq2_ks_q8_k_avx512;
            s_vec_dot_iq3_ks     = iqk_vec_dot_iq3_ks_q8_k_avx512;
            s_vec_dot_iq4_ks     = iqk_vec_dot_iq4_ks_q8_k_avx512;
            s_vec_dot_iq4_ks_r16 = iqk_vec_dot_iq4_ks_r16_q8_0_avx512;
            break;

        case IQK_SIMD_AVX_VNNI:
        case IQK_SIMD_AVX2:
            s_dequant_iq2_ks     = iqk_dequantize_row_iq2_ks_avx2;
            s_dequant_iq3_ks     = iqk_dequantize_row_iq3_ks_avx2;
            s_dequant_iq4_ks     = iqk_dequantize_row_iq4_ks_avx2;
            s_dequant_iq4_ks_r16 = iqk_dequantize_row_iq4_ks_r16_avx2;

            s_vec_dot_iq2_ks     = iqk_vec_dot_iq2_ks_q8_k_avx2;
            s_vec_dot_iq3_ks     = iqk_vec_dot_iq3_ks_q8_k_avx2;
            s_vec_dot_iq4_ks     = iqk_vec_dot_iq4_ks_q8_k_avx2;
            s_vec_dot_iq4_ks_r16 = iqk_vec_dot_iq4_ks_r16_q8_0_avx2;
            break;

        case IQK_SIMD_GENERIC:
        default:
            s_dequant_iq2_ks     = iqk_dequantize_row_iq2_ks_generic;
            s_dequant_iq3_ks     = iqk_dequantize_row_iq3_ks_generic;
            s_dequant_iq4_ks     = iqk_dequantize_row_iq4_ks_generic;
            s_dequant_iq4_ks_r16 = iqk_dequantize_row_iq4_ks_r16_generic;

            s_vec_dot_iq2_ks     = iqk_vec_dot_iq2_ks_q8_k_generic;
            s_vec_dot_iq3_ks     = iqk_vec_dot_iq3_ks_q8_k_generic;
            s_vec_dot_iq4_ks     = iqk_vec_dot_iq4_ks_q8_k_generic;
            s_vec_dot_iq4_ks_r16 = iqk_vec_dot_iq4_ks_r16_q8_0_generic;
            break;
    }
}

iqk_simd_type iqk_get_simd_type(void) {
    if (!s_initialized.load()) {
        iqk_init_dispatch();
    }
    return s_active_simd;
}

const char * iqk_get_simd_name(void) {
    switch (iqk_get_simd_type()) {
        case IQK_SIMD_AVX512_VNNI: return "AVX-512 VNNI (Zen4/Sapphire Rapids+)";
        case IQK_SIMD_AVX_VNNI:    return "AVX2 + AVX_VNNI (Alder Lake/Raptor Lake)";
        case IQK_SIMD_AVX2:        return "AVX2 (Haswell/Zen1-3)";
        case IQK_SIMD_GENERIC:     return "Generic Portable";
        default:                   return "Unknown";
    }
}

// =============================================================================
// Public Dequantization Math Wrappers
// =============================================================================

void iqk_dequantize_row_iq2_ks(const block_iq2_ks * x, float * y, int64_t k, float d) {
    if (!s_initialized.load()) iqk_init_dispatch();
    s_dequant_iq2_ks(x, y, k, d);
}

void iqk_dequantize_row_iq3_ks(const block_iq3_ks * x, float * y, int64_t k, float d) {
    if (!s_initialized.load()) iqk_init_dispatch();
    s_dequant_iq3_ks(x, y, k, d);
}

void iqk_dequantize_row_iq4_ks(const block_iq4_ks * x, float * y, int64_t k, float d) {
    if (!s_initialized.load()) iqk_init_dispatch();
    s_dequant_iq4_ks(x, y, k, d);
}

void iqk_dequantize_row_iq4_ks_r16(const block_iq4_ks_r16 * x, float * y, int64_t k, const float * d16) {
    if (!s_initialized.load()) iqk_init_dispatch();
    s_dequant_iq4_ks_r16(x, y, k, d16);
}

// =============================================================================
// Public Vector Dot-Product Wrappers
// =============================================================================

void iqk_vec_dot_iq2_ks_q8_k(int n, float * s, const void * vx, const void * vy, float d) {
    if (!s_initialized.load()) iqk_init_dispatch();
    s_vec_dot_iq2_ks(n, s, vx, vy, d);
}

void iqk_vec_dot_iq3_ks_q8_k(int n, float * s, const void * vx, const void * vy, float d) {
    if (!s_initialized.load()) iqk_init_dispatch();
    s_vec_dot_iq3_ks(n, s, vx, vy, d);
}

void iqk_vec_dot_iq4_ks_q8_k(int n, float * s, const void * vx, const void * vy, float d) {
    if (!s_initialized.load()) iqk_init_dispatch();
    s_vec_dot_iq4_ks(n, s, vx, vy, d);
}

void iqk_vec_dot_iq4_ks_r16_q8_0(int n, float * s, const void * vx, const void * vy, const float * d16) {
    if (!s_initialized.load()) iqk_init_dispatch();
    s_vec_dot_iq4_ks_r16(n, s, vx, vy, d16);
}

// =============================================================================
// High-Throughput Hybrid Offloaded CPU GEMM
// =============================================================================

bool iqk_gemm_hybrid_cpu(
        iqk_quant_type type,
        int nr, int nc, int k,
        const void * A, size_t stride_A,
        const void * B, size_t stride_B,
        float * C, size_t stride_C,
        const float * scales_row,
        int ith, int nth) {
    if (!s_initialized.load()) {
        iqk_init_dispatch();
    }

    if (nr <= 0 || nc <= 0 || k <= 0 || !A || !B || !C) {
        return false;
    }

    // Work partition across worker threads (parallelized across matrix rows)
    const int r0 = (nr *  ith   ) / nth;
    const int r1 = (nr * (ith+1)) / nth;

    for (int r = r0; r < r1; ++r) {
        const float d_r = scales_row ? scales_row[r] : 1.0f;
        const char * a_row = (const char *)A + (size_t)r * stride_A;

        for (int c = 0; c < nc; ++c) {
            const char * b_col = (const char *)B + (size_t)c * stride_B;
            float * c_out = C + (size_t)r * stride_C + c;

            switch (type) {
                case IQK_QUANT_TYPE_IQ2_KS:
                    s_vec_dot_iq2_ks(k, c_out, a_row, b_col, d_r);
                    break;
                case IQK_QUANT_TYPE_IQ3_KS:
                    s_vec_dot_iq3_ks(k, c_out, a_row, b_col, d_r);
                    break;
                case IQK_QUANT_TYPE_IQ4_KS:
                    s_vec_dot_iq4_ks(k, c_out, a_row, b_col, d_r);
                    break;
                case IQK_QUANT_TYPE_IQ4_KS_R16:
                    // For R16 interleaved blocks, multiple rows are evaluated together
                    if (r % 16 == 0) {
                        float s16[16];
                        s_vec_dot_iq4_ks_r16(k, s16, a_row, b_col, scales_row ? scales_row + r : nullptr);
                        for (int i = 0; i < 16 && r + i < nr; ++i) {
                            C[(size_t)(r + i) * stride_C + c] = s16[i];
                        }
                    }
                    break;
                default:
                    return false;
            }
        }
    }

    return true;
}
