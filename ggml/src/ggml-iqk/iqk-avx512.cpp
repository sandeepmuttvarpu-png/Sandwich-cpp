#include "iqk-internal.h"

#if defined(__AVX512F__) && defined(__AVX512BW__)
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

static inline int32_t hsum_m256i(const __m256i v) {
    const __m128i lo = _mm256_castsi256_si128(v);
    const __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i sum = _mm_add_epi32(lo, hi);
    sum = _mm_add_epi32(sum, _mm_unpackhi_epi64(sum, sum));
    sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, _MM_SHUFFLE(2, 3, 0, 1)));
    return _mm_cvtsi128_si32(sum);
}

static inline __m256i unpack_nibbles_16to32_512(const uint8_t * p) {
    const __m128i raw16 = _mm_loadu_si128((const __m128i *)p);
    const __m128i mask  = _mm_set1_epi8(0x0F);

    const __m128i lo = _mm_and_si128(raw16, mask);
    const __m128i hi = _mm_and_si128(_mm_srli_epi16(raw16, 4), mask);

    const __m128i part0 = _mm_unpacklo_epi8(lo, hi);
    const __m128i part1 = _mm_unpackhi_epi8(lo, hi);

    return _mm256_inserti128_si256(_mm256_castsi128_si256(part0), part1, 1);
}

// =============================================================================
// AVX-512 VNNI Vector Dot Products
// =============================================================================

void iqk_vec_dot_iq4_ks_q8_k_avx512(int n, float * s, const void * vx, const void * vy, float d) {
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
    const int nb = n / IQK_QK_K;
    const block_iq4_ks * x = (const block_iq4_ks *)vx;
    const q8_K_ref     * y = (const q8_K_ref *)vy;

    const __m256i ones8 = _mm256_set1_epi8(1);

    float sum = 0.0f;
    for (int b = 0; b < nb; ++b) {
        const block_iq4_ks * blk_x = x + b;
        const q8_K_ref     * blk_y = y + b;
        const float dy = blk_y->d;

        for (int g = 0; g < 8; ++g) {
            const float scale = d * (float)(blk_x->scales[g] + 1) * dy;
            const __m256i q_u8 = unpack_nibbles_16to32_512(blk_x->qs + g * 16);
            const __m256i y_s8 = _mm256_loadu_si256((const __m256i *)(blk_y->qs + g * 32));

            // Single instruction 32-way VNNI dot product
            __m256i dot_acc   = _mm256_dpbusd_epi32(_mm256_setzero_si256(), q_u8, y_s8);
            __m256i sum_y_acc = _mm256_dpbusd_epi32(_mm256_setzero_si256(), ones8, y_s8);

            const int32_t dot   = hsum_m256i(dot_acc);
            const int32_t sum_y = hsum_m256i(sum_y_acc);

            sum += scale * ((float)dot - 8.0f * (float)sum_y);
        }
    }
    *s = sum;
#else
    iqk_vec_dot_iq4_ks_q8_k_avx2(n, s, vx, vy, d);
#endif
}

void iqk_vec_dot_iq2_ks_q8_k_avx512(int n, float * s, const void * vx, const void * vy, float d) {
    iqk_vec_dot_iq2_ks_q8_k_avx2(n, s, vx, vy, d);
}

void iqk_vec_dot_iq3_ks_q8_k_avx512(int n, float * s, const void * vx, const void * vy, float d) {
    iqk_vec_dot_iq3_ks_q8_k_avx2(n, s, vx, vy, d);
}

void iqk_vec_dot_iq4_ks_r16_q8_0_avx512(int n, float * s, const void * vx, const void * vy, const float * d16) {
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
    const int nblocks = n / 32;
    const block_iq4_ks_r16 * x = (const block_iq4_ks_r16 *)vx;
    const q8_0_ref         * y = (const q8_0_ref *)vy;

    const __m256i ones8 = _mm256_set1_epi8(1);

    for (int r = 0; r < 16; ++r) {
        float sum = 0.0f;
        const float dr = d16 ? d16[r] : 1.0f;

        for (int b = 0; b < nblocks; ++b) {
            const block_iq4_ks_r16 * blk_x = x + b;
            const q8_0_ref         * blk_y = y + b;

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
            const __m256i q_u8 = unpack_nibbles_16to32_512(blk_x->qs + r * 16);
            const __m256i y_s8 = _mm256_loadu_si256((const __m256i *)blk_y->qs);

            __m256i dot_acc   = _mm256_dpbusd_epi32(_mm256_setzero_si256(), q_u8, y_s8);
            __m256i sum_y_acc = _mm256_dpbusd_epi32(_mm256_setzero_si256(), ones8, y_s8);

            const int32_t dot   = hsum_m256i(dot_acc);
            const int32_t sum_y = hsum_m256i(sum_y_acc);

            sum += scale * ((float)dot - 8.0f * (float)sum_y);
        }
        s[r] = sum;
    }
#else
    iqk_vec_dot_iq4_ks_r16_q8_0_avx2(n, s, vx, vy, d16);
#endif
}

// =============================================================================
// AVX-512 Dequantization
// =============================================================================

void iqk_dequantize_row_iq4_ks_avx512(const block_iq4_ks * x, float * y, int64_t k, float d) {
    const int64_t nb = k / IQK_QK_K;
    const __m512 eight = _mm512_set1_ps(8.0f);

    for (int64_t b = 0; b < nb; ++b) {
        const block_iq4_ks * blk = x + b;
        float * yb = y + b * IQK_QK_K;

        for (int g = 0; g < 8; ++g) {
            const float scale = d * (float)(blk->scales[g] + 1);
            const __m512 vscale = _mm512_set1_ps(scale);

            // 16 bytes of nibbles -> 32 floats
            const __m128i raw16 = _mm_loadu_si128((const __m128i *)(blk->qs + g * 16));
            const __m128i mask  = _mm_set1_epi8(0x0F);
            const __m128i lo = _mm_and_si128(raw16, mask);
            const __m128i hi = _mm_and_si128(_mm_srli_epi16(raw16, 4), mask);

            const __m128i part0 = _mm_unpacklo_epi8(lo, hi);
            const __m128i part1 = _mm_unpackhi_epi8(lo, hi);

            const __m512i q0_16 = _mm512_cvtepu8_epi32(part0);
            const __m512i q1_16 = _mm512_cvtepu8_epi32(part1);

            const __m512 f0 = _mm512_mul_ps(vscale, _mm512_sub_ps(_mm512_cvtepi32_ps(q0_16), eight));
            const __m512 f1 = _mm512_mul_ps(vscale, _mm512_sub_ps(_mm512_cvtepi32_ps(q1_16), eight));

            _mm512_storeu_ps(yb + g * 32 +  0, f0);
            _mm512_storeu_ps(yb + g * 32 + 16, f1);
        }
    }
}

void iqk_dequantize_row_iq2_ks_avx512(const block_iq2_ks * x, float * y, int64_t k, float d) {
    iqk_dequantize_row_iq2_ks_avx2(x, y, k, d);
}

void iqk_dequantize_row_iq3_ks_avx512(const block_iq3_ks * x, float * y, int64_t k, float d) {
    iqk_dequantize_row_iq3_ks_avx2(x, y, k, d);
}

void iqk_dequantize_row_iq4_ks_r16_avx512(const block_iq4_ks_r16 * x, float * y, int64_t k, const float * d16) {
    iqk_dequantize_row_iq4_ks_r16_avx2(x, y, k, d16);
}

#else

// Stubs for non-AVX-512 builds
void iqk_dequantize_row_iq2_ks_avx512(const block_iq2_ks * x, float * y, int64_t k, float d) { iqk_dequantize_row_iq2_ks_avx2(x, y, k, d); }
void iqk_dequantize_row_iq3_ks_avx512(const block_iq3_ks * x, float * y, int64_t k, float d) { iqk_dequantize_row_iq3_ks_avx2(x, y, k, d); }
void iqk_dequantize_row_iq4_ks_avx512(const block_iq4_ks * x, float * y, int64_t k, float d) { iqk_dequantize_row_iq4_ks_avx2(x, y, k, d); }
void iqk_dequantize_row_iq4_ks_r16_avx512(const block_iq4_ks_r16 * x, float * y, int64_t k, const float * d16) { iqk_dequantize_row_iq4_ks_r16_avx2(x, y, k, d16); }

void iqk_vec_dot_iq2_ks_q8_k_avx512(int n, float * s, const void * vx, const void * vy, float d) { iqk_vec_dot_iq2_ks_q8_k_avx2(n, s, vx, vy, d); }
void iqk_vec_dot_iq3_ks_q8_k_avx512(int n, float * s, const void * vx, const void * vy, float d) { iqk_vec_dot_iq3_ks_q8_k_avx2(n, s, vx, vy, d); }
void iqk_vec_dot_iq4_ks_q8_k_avx512(int n, float * s, const void * vx, const void * vy, float d) { iqk_vec_dot_iq4_ks_q8_k_avx2(n, s, vx, vy, d); }
void iqk_vec_dot_iq4_ks_r16_q8_0_avx512(int n, float * s, const void * vx, const void * vy, const float * d16) { iqk_vec_dot_iq4_ks_r16_q8_0_avx2(n, s, vx, vy, d16); }

#endif
