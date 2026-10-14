#pragma once

#include "ggml-iqk.h"

#ifdef __cplusplus
extern "C" {
#endif

// Generic portable implementations
void iqk_dequantize_row_iq2_ks_generic(const block_iq2_ks * x, float * y, int64_t k, float d);
void iqk_dequantize_row_iq3_ks_generic(const block_iq3_ks * x, float * y, int64_t k, float d);
void iqk_dequantize_row_iq4_ks_generic(const block_iq4_ks * x, float * y, int64_t k, float d);
void iqk_dequantize_row_iq4_ks_r16_generic(const block_iq4_ks_r16 * x, float * y, int64_t k, const float * d16);

void iqk_vec_dot_iq2_ks_q8_k_generic(int n, float * s, const void * vx, const void * vy, float d);
void iqk_vec_dot_iq3_ks_q8_k_generic(int n, float * s, const void * vx, const void * vy, float d);
void iqk_vec_dot_iq4_ks_q8_k_generic(int n, float * s, const void * vx, const void * vy, float d);
void iqk_vec_dot_iq4_ks_r16_q8_0_generic(int n, float * s, const void * vx, const void * vy, const float * d16);

// AVX2 implementations
void iqk_dequantize_row_iq2_ks_avx2(const block_iq2_ks * x, float * y, int64_t k, float d);
void iqk_dequantize_row_iq3_ks_avx2(const block_iq3_ks * x, float * y, int64_t k, float d);
void iqk_dequantize_row_iq4_ks_avx2(const block_iq4_ks * x, float * y, int64_t k, float d);
void iqk_dequantize_row_iq4_ks_r16_avx2(const block_iq4_ks_r16 * x, float * y, int64_t k, const float * d16);

void iqk_vec_dot_iq2_ks_q8_k_avx2(int n, float * s, const void * vx, const void * vy, float d);
void iqk_vec_dot_iq3_ks_q8_k_avx2(int n, float * s, const void * vx, const void * vy, float d);
void iqk_vec_dot_iq4_ks_q8_k_avx2(int n, float * s, const void * vx, const void * vy, float d);
void iqk_vec_dot_iq4_ks_r16_q8_0_avx2(int n, float * s, const void * vx, const void * vy, const float * d16);

// AVX-512 VNNI implementations
void iqk_dequantize_row_iq2_ks_avx512(const block_iq2_ks * x, float * y, int64_t k, float d);
void iqk_dequantize_row_iq3_ks_avx512(const block_iq3_ks * x, float * y, int64_t k, float d);
void iqk_dequantize_row_iq4_ks_avx512(const block_iq4_ks * x, float * y, int64_t k, float d);
void iqk_dequantize_row_iq4_ks_r16_avx512(const block_iq4_ks_r16 * x, float * y, int64_t k, const float * d16);

void iqk_vec_dot_iq2_ks_q8_k_avx512(int n, float * s, const void * vx, const void * vy, float d);
void iqk_vec_dot_iq3_ks_q8_k_avx512(int n, float * s, const void * vx, const void * vy, float d);
void iqk_vec_dot_iq4_ks_q8_k_avx512(int n, float * s, const void * vx, const void * vy, float d);
void iqk_vec_dot_iq4_ks_r16_q8_0_avx512(int n, float * s, const void * vx, const void * vy, const float * d16);

#ifdef __cplusplus
}
#endif
