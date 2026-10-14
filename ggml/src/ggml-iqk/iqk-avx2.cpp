#include "iqk-internal.h"

#if defined(__AVX2__)
#include <immintrin.h>

struct q8_K_ref {
    float   d;
    int8_t  qs[256];
    int16_t bsums[16];
};

struct q8_0_ref {
    uint16_t d; // ggml_half
    int8_t   qs[32];
};

static inline float hsum_m256(const __m256 v) {
    const __m128 lo = _mm256_castps256_ps128(v);
    const __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
    sum = _mm_add_ss(sum, _mm_movehdup_ps(sum));
    return _mm_cvtss_f32(sum);
}

static inline int32_t hsum_m256i(const __m256i v) {
    const __m128i lo = _mm256_castsi256_si128(v);
    const __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i sum = _mm_add_epi32(lo, hi);
    sum = _mm_add_epi32(sum, _mm_unpackhi_epi64(sum, sum));
    sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, _MM_SHUFFLE(2, 3, 0, 1)));
    return _mm_cvtsi128_si32(sum);
}

// Unpack 16 bytes of 4-bit nibbles into 32 unsigned 8-bit integers
static inline __m256i unpack_nibbles_16to32(const uint8_t * p) {
    const __m128i raw16 = _mm_loadu_si128((const __m128i *)p);
    const __m128i mask  = _mm_set1_epi8(0x0F);

    const __m128i lo = _mm_and_si128(raw16, mask);
    const __m128i hi = _mm_and_si128(_mm_srli_epi16(raw16, 4), mask);

    const __m128i part0 = _mm_unpacklo_epi8(lo, hi);
    const __m128i part1 = _mm_unpackhi_epi8(lo, hi);

    return _mm256_inserti128_si256(_mm256_castsi128_si256(part0), part1, 1);
}

// =============================================================================
// AVX2 Vector Dot Products
// =============================================================================

void iqk_vec_dot_iq4_ks_q8_k_avx2(int n, float * s, const void * vx, const void * vy, float d) {
    const int nb = n / IQK_QK_K;
    const block_iq4_ks * x = (const block_iq4_ks *)vx;
    const q8_K_ref     * y = (const q8_K_ref *)vy;

    const __m256i ones16 = _mm256_set1_epi16(1);
    const __m256i ones8  = _mm256_set1_epi8(1);

    float sum = 0.0f;
    for (int b = 0; b < nb; ++b) {
        const block_iq4_ks * blk_x = x + b;
        const q8_K_ref     * blk_y = y + b;
        const float dy = blk_y->d;

        for (int g = 0; g < 8; ++g) {
            const float scale = d * (float)(blk_x->scales[g] + 1) * dy;
            const __m256i q_u8 = unpack_nibbles_16to32(blk_x->qs + g * 16);
            const __m256i y_s8 = _mm256_loadu_si256((const __m256i *)(blk_y->qs + g * 32));

            // dot = sum(q_u8 * y_s8)
            const __m256i prod16 = _mm256_maddubs_epi16(q_u8, y_s8);
            const __m256i dot32  = _mm256_madd_epi16(prod16, ones16);

            // sum_y = sum(1 * y_s8)
            const __m256i sum_y16 = _mm256_maddubs_epi16(ones8, y_s8);
            const __m256i sum_y32 = _mm256_madd_epi16(sum_y16, ones16);

            const int32_t dot   = hsum_m256i(dot32);
            const int32_t sum_y = hsum_m256i(sum_y32);

            sum += scale * ((float)dot - 8.0f * (float)sum_y);
        }
    }
    *s = sum;
}

void iqk_vec_dot_iq2_ks_q8_k_avx2(int n, float * s, const void * vx, const void * vy, float d) {
    // Delegate to optimized generic or unpack with AVX2
    iqk_vec_dot_iq2_ks_q8_k_generic(n, s, vx, vy, d);
}

void iqk_vec_dot_iq3_ks_q8_k_avx2(int n, float * s, const void * vx, const void * vy, float d) {
    iqk_vec_dot_iq3_ks_q8_k_generic(n, s, vx, vy, d);
}

void iqk_vec_dot_iq4_ks_r16_q8_0_avx2(int n, float * s, const void * vx, const void * vy, const float * d16) {
    const int nblocks = n / 32;
    const block_iq4_ks_r16 * x = (const block_iq4_ks_r16 *)vx;
    const q8_0_ref         * y = (const q8_0_ref *)vy;

    const __m256i ones16 = _mm256_set1_epi16(1);
    const __m256i ones8  = _mm256_set1_epi8(1);

    for (int r = 0; r < 16; ++r) {
        float sum = 0.0f;
        const float dr = d16 ? d16[r] : 1.0f;

        for (int b = 0; b < nblocks; ++b) {
            const block_iq4_ks_r16 * blk_x = x + b;
            const q8_0_ref         * blk_y = y + b;

            // Load fp16 scale of q8_0
            uint32_t sign     = ((uint32_t)(blk_y->d & 0x8000)) << 16;
            uint32_t exp_bits = (blk_y->d & 0x7c00) >> 10;
            uint32_t mantissa = (blk_y->d & 0x03ff);
            float dy;
            if (exp_bits == 0 && mantissa == 0) {
                dy = 0.0f;
            } else {
                uint32_t res = sign | ((exp_bits + (127 - 15)) << 23) | (mantissa << 13);
                __builtin_memcpy(&dy, &res, sizeof(float));
            }

            const float scale = dr * (float)(blk_x->scales[r] + 1) * dy;
            const __m256i q_u8 = unpack_nibbles_16to32(blk_x->qs + r * 16);
            const __m256i y_s8 = _mm256_loadu_si256((const __m256i *)blk_y->qs);

            const __m256i prod16  = _mm256_maddubs_epi16(q_u8, y_s8);
            const __m256i dot32   = _mm256_madd_epi16(prod16, ones16);
            const __m256i sum_y16 = _mm256_maddubs_epi16(ones8, y_s8);
            const __m256i sum_y32 = _mm256_madd_epi16(sum_y16, ones16);

            const int32_t dot   = hsum_m256i(dot32);
            const int32_t sum_y = hsum_m256i(sum_y32);

            sum += scale * ((float)dot - 8.0f * (float)sum_y);
        }
        s[r] = sum;
    }
}

// =============================================================================
// AVX2 Dequantization
// =============================================================================

void iqk_dequantize_row_iq4_ks_avx2(const block_iq4_ks * x, float * y, int64_t k, float d) {
    const int64_t nb = k / IQK_QK_K;
    const __m256 eight = _mm256_set1_ps(8.0f);

    for (int64_t b = 0; b < nb; ++b) {
        const block_iq4_ks * blk = x + b;
        float * yb = y + b * IQK_QK_K;

        for (int g = 0; g < 8; ++g) {
            const float scale = d * (float)(blk->scales[g] + 1);
            const __m256 vscale = _mm256_set1_ps(scale);
            const __m256i q_u8 = unpack_nibbles_16to32(blk->qs + g * 16);

            // Convert 32 bytes to 32 floats in 4 AVX registers
            const __m128i q0_16 = _mm256_extracti128_si256(q_u8, 0);
            const __m128i q1_16 = _mm256_extracti128_si256(q_u8, 1);

            const __m256i q0_32 = _mm256_cvtepu8_epi32(q0_16);
            const __m256i q1_32 = _mm256_cvtepu8_epi32(_mm_srli_si128(q0_16, 8));
            const __m256i q2_32 = _mm256_cvtepu8_epi32(q1_16);
            const __m256i q3_32 = _mm256_cvtepu8_epi32(_mm_srli_si128(q1_16, 8));

            const __m256 f0 = _mm256_mul_ps(vscale, _mm256_sub_ps(_mm256_cvtepi32_ps(q0_32), eight));
            const __m256 f1 = _mm256_mul_ps(vscale, _mm256_sub_ps(_mm256_cvtepi32_ps(q1_32), eight));
            const __m256 f2 = _mm256_mul_ps(vscale, _mm256_sub_ps(_mm256_cvtepi32_ps(q2_32), eight));
            const __m256 f3 = _mm256_mul_ps(vscale, _mm256_sub_ps(_mm256_cvtepi32_ps(q3_32), eight));

            _mm256_storeu_ps(yb + g * 32 +  0, f0);
            _mm256_storeu_ps(yb + g * 32 +  8, f1);
            _mm256_storeu_ps(yb + g * 32 + 16, f2);
            _mm256_storeu_ps(yb + g * 32 + 24, f3);
        }
    }
}

void iqk_dequantize_row_iq2_ks_avx2(const block_iq2_ks * x, float * y, int64_t k, float d) {
    iqk_dequantize_row_iq2_ks_generic(x, y, k, d);
}

void iqk_dequantize_row_iq3_ks_avx2(const block_iq3_ks * x, float * y, int64_t k, float d) {
    iqk_dequantize_row_iq3_ks_generic(x, y, k, d);
}

void iqk_dequantize_row_iq4_ks_r16_avx2(const block_iq4_ks_r16 * x, float * y, int64_t k, const float * d16) {
    iqk_dequantize_row_iq4_ks_r16_generic(x, y, k, d16);
}

#else

// Stubs for non-AVX2 builds
void iqk_dequantize_row_iq2_ks_avx2(const block_iq2_ks * x, float * y, int64_t k, float d) { iqk_dequantize_row_iq2_ks_generic(x, y, k, d); }
void iqk_dequantize_row_iq3_ks_avx2(const block_iq3_ks * x, float * y, int64_t k, float d) { iqk_dequantize_row_iq3_ks_generic(x, y, k, d); }
void iqk_dequantize_row_iq4_ks_avx2(const block_iq4_ks * x, float * y, int64_t k, float d) { iqk_dequantize_row_iq4_ks_generic(x, y, k, d); }
void iqk_dequantize_row_iq4_ks_r16_avx2(const block_iq4_ks_r16 * x, float * y, int64_t k, const float * d16) { iqk_dequantize_row_iq4_ks_r16_generic(x, y, k, d16); }

void iqk_vec_dot_iq2_ks_q8_k_avx2(int n, float * s, const void * vx, const void * vy, float d) { iqk_vec_dot_iq2_ks_q8_k_generic(n, s, vx, vy, d); }
void iqk_vec_dot_iq3_ks_q8_k_avx2(int n, float * s, const void * vx, const void * vy, float d) { iqk_vec_dot_iq3_ks_q8_k_generic(n, s, vx, vy, d); }
void iqk_vec_dot_iq4_ks_q8_k_avx2(int n, float * s, const void * vx, const void * vy, float d) { iqk_vec_dot_iq4_ks_q8_k_generic(n, s, vx, vy, d); }
void iqk_vec_dot_iq4_ks_r16_q8_0_avx2(int n, float * s, const void * vx, const void * vy, const float * d16) { iqk_vec_dot_iq4_ks_r16_q8_0_generic(n, s, vx, vy, d16); }

#endif
