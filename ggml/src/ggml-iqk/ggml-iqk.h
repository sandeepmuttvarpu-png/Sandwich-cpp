#pragma once

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// IQK Quantized Block Structures (Clean-Room Specifications)
// =============================================================================

#define IQK_QK_K 256
#define IQK_QK8_0 32

// IQ2_KS: 2-bit quantized super-block of 256 elements
// Format: 16-bit extra, 4 sub-block scales (64 elements each), 64 bytes (4 values/byte)
typedef struct {
    uint16_t extra;
    uint8_t  scales[4];
    uint8_t  qs[64];
} block_iq2_ks;
#ifdef __cplusplus
static_assert(sizeof(block_iq2_ks) == 2 + 4 + 64, "block_iq2_ks must be 70 bytes");
#endif

// IQ3_KS: 3-bit quantized super-block of 256 elements
// Format: 16-bit extra, 4 sub-block scales, 64 bytes low 2 bits, 32 bytes high 1 bit
typedef struct {
    uint16_t extra;
    uint8_t  scales[4];
    uint8_t  qs[64];
    uint8_t  qh[32];
} block_iq3_ks;
#ifdef __cplusplus
static_assert(sizeof(block_iq3_ks) == 2 + 4 + 64 + 32, "block_iq3_ks must be 102 bytes");
#endif

// IQ4_KS: 4-bit quantized super-block of 256 elements
// Format: 8 sub-block scales (32 elements each), 128 bytes (2 nibbles/byte)
typedef struct {
    uint8_t  scales[8];
    uint8_t  qs[128];
} block_iq4_ks;
#ifdef __cplusplus
static_assert(sizeof(block_iq4_ks) == 8 + 128, "block_iq4_ks must be 136 bytes");
#endif

// IQ4_KS_R16: 16-row interleaved 4-bit quant block (16 rows x 32 elements = 512 elements)
// Format: 16 row scales, 256 bytes (16 bytes per row)
typedef struct {
    uint8_t  scales[16];
    uint8_t  qs[256];
} block_iq4_ks_r16;
#ifdef __cplusplus
static_assert(sizeof(block_iq4_ks_r16) == 16 + 256, "block_iq4_ks_r16 must be 272 bytes");
#endif

// =============================================================================
// SIMD Architecture Flags
// =============================================================================

typedef enum {
    IQK_SIMD_GENERIC     = 0,
    IQK_SIMD_AVX2        = 1,
    IQK_SIMD_AVX_VNNI    = 2,
    IQK_SIMD_AVX512_VNNI = 3,
} iqk_simd_type;

// =============================================================================
// Dequantization Math API (Separated from Compute Kernels)
// =============================================================================

void iqk_dequantize_row_iq2_ks(const block_iq2_ks * x, float * y, int64_t k, float d);
void iqk_dequantize_row_iq3_ks(const block_iq3_ks * x, float * y, int64_t k, float d);
void iqk_dequantize_row_iq4_ks(const block_iq4_ks * x, float * y, int64_t k, float d);
void iqk_dequantize_row_iq4_ks_r16(const block_iq4_ks_r16 * x, float * y, int64_t k, const float * d16);

// =============================================================================
// Vector Dot-Product API
// =============================================================================

void iqk_vec_dot_iq2_ks_q8_k(int n, float * s, const void * vx, const void * vy, float d);
void iqk_vec_dot_iq3_ks_q8_k(int n, float * s, const void * vx, const void * vy, float d);
void iqk_vec_dot_iq4_ks_q8_k(int n, float * s, const void * vx, const void * vy, float d);
void iqk_vec_dot_iq4_ks_r16_q8_0(int n, float * s, const void * vx, const void * vy, const float * d16);

// =============================================================================
// High-Throughput Hybrid Offloaded CPU GEMM
// =============================================================================

typedef enum {
    IQK_QUANT_TYPE_IQ2_KS     = 0,
    IQK_QUANT_TYPE_IQ3_KS     = 1,
    IQK_QUANT_TYPE_IQ4_KS     = 2,
    IQK_QUANT_TYPE_IQ4_KS_R16 = 3,
} iqk_quant_type;

// =============================================================================
// Extended GGML Type IDs for IQK Sparse Formats (Zero Collision with TurboQuant 43-45)
// =============================================================================
#define GGML_TYPE_IQK_BASE       134
#define GGML_TYPE_IQ2_KS_EXT     (GGML_TYPE_IQK_BASE + 0) // 134
#define GGML_TYPE_IQ3_KS_EXT     (GGML_TYPE_IQK_BASE + 1) // 135
#define GGML_TYPE_IQ4_KS_EXT     (GGML_TYPE_IQK_BASE + 2) // 136
#define GGML_TYPE_IQ4_KS_R16_EXT (GGML_TYPE_IQK_BASE + 3) // 137

#ifdef __cplusplus
static_assert(GGML_TYPE_IQK_BASE >= 134, "IQK types must be >= 134 to prevent enum collision with TurboQuant (43-45)");
#endif

bool iqk_gemm_hybrid_cpu(
        iqk_quant_type type,
        int nr, int nc, int k,
        const void * A, size_t stride_A,
        const void * B, size_t stride_B,
        float * C, size_t stride_C,
        const float * scales_row,
        int ith, int nth);

// =============================================================================
// Runtime SIMD Capability Detection
// =============================================================================

iqk_simd_type iqk_get_simd_type(void);
const char *  iqk_get_simd_name(void);
void          iqk_set_simd_override(iqk_simd_type type);

#ifdef __cplusplus
}
#endif
