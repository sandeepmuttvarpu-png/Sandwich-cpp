#include "iqk-internal.h"
#include <cmath>
#include <cstring>

// Reference block_q8_K and block_q8_0 layout compatible with ggml
struct q8_K_ref {
    float   d;
    int8_t  qs[256];
    int16_t bsums[16];
};

struct q8_0_ref {
    uint16_t d; // ggml_half
    int8_t   qs[32];
};

static inline float fp16_to_fp32_local(uint16_t h) {
    uint32_t sign     = ((uint32_t)(h & 0x8000)) << 16;
    uint32_t exp_bits = (h & 0x7c00) >> 10;
    uint32_t mantissa = (h & 0x03ff);

    if (exp_bits == 0) {
        if (mantissa == 0) {
            uint32_t res = sign;
            float out;
            memcpy(&out, &res, sizeof(float));
            return out;
        }
        while (!(mantissa & 0x0400)) {
            mantissa <<= 1;
            exp_bits--;
        }
        exp_bits++;
        mantissa &= ~0x0400;
    } else if (exp_bits == 31) {
        uint32_t res = sign | 0x7f800000 | (mantissa << 13);
        float out;
        memcpy(&out, &res, sizeof(float));
        return out;
    }

    uint32_t res = sign | ((exp_bits + (127 - 15)) << 23) | (mantissa << 13);
    float out;
    memcpy(&out, &res, sizeof(float));
    return out;
}

// =============================================================================
// Dequantization Math: Generic Reference Implementations
// =============================================================================

void iqk_dequantize_row_iq2_ks_generic(const block_iq2_ks * x, float * y, int64_t k, float d) {
    const int64_t nb = k / IQK_QK_K;
    for (int64_t b = 0; b < nb; ++b) {
        const block_iq2_ks * blk = x + b;
        float * yb = y + b * IQK_QK_K;

        for (int g = 0; g < 4; ++g) {
            const float scale = d * (float)(blk->scales[g] + 1);
            const uint8_t * qsrc = blk->qs + g * 16;
            float * yg = yb + g * 64;

            for (int i = 0; i < 16; ++i) {
                const uint8_t byte = qsrc[i];
                const int q0 = (byte     ) & 3;
                const int q1 = (byte >> 2) & 3;
                const int q2 = (byte >> 4) & 3;
                const int q3 = (byte >> 6) & 3;

                yg[i * 4 + 0] = scale * ((float)q0 - 1.5f);
                yg[i * 4 + 1] = scale * ((float)q1 - 1.5f);
                yg[i * 4 + 2] = scale * ((float)q2 - 1.5f);
                yg[i * 4 + 3] = scale * ((float)q3 - 1.5f);
            }
        }
    }
}

void iqk_dequantize_row_iq3_ks_generic(const block_iq3_ks * x, float * y, int64_t k, float d) {
    const int64_t nb = k / IQK_QK_K;
    for (int64_t b = 0; b < nb; ++b) {
        const block_iq3_ks * blk = x + b;
        float * yb = y + b * IQK_QK_K;

        for (int g = 0; g < 4; ++g) {
            const float scale = d * (float)(blk->scales[g] + 1);
            const uint8_t * qsrc = blk->qs + g * 16;
            const uint8_t * hsrc = blk->qh + g * 8;
            float * yg = yb + g * 64;

            for (int i = 0; i < 64; ++i) {
                const int q_lo = (qsrc[i / 4] >> ((i % 4) * 2)) & 3;
                const int q_hi = (hsrc[i / 8] >> (i % 8)) & 1;
                const int q = q_lo | (q_hi << 2);
                yg[i] = scale * ((float)q - 3.5f);
            }
        }
    }
}

void iqk_dequantize_row_iq4_ks_generic(const block_iq4_ks * x, float * y, int64_t k, float d) {
    const int64_t nb = k / IQK_QK_K;
    for (int64_t b = 0; b < nb; ++b) {
        const block_iq4_ks * blk = x + b;
        float * yb = y + b * IQK_QK_K;

        for (int g = 0; g < 8; ++g) {
            const float scale = d * (float)(blk->scales[g] + 1);
            const uint8_t * qsrc = blk->qs + g * 16;
            float * yg = yb + g * 32;

            for (int i = 0; i < 16; ++i) {
                const uint8_t byte = qsrc[i];
                const int q0 = byte & 0x0F;
                const int q1 = byte >> 4;
                yg[i * 2 + 0] = scale * ((float)q0 - 8.0f);
                yg[i * 2 + 1] = scale * ((float)q1 - 8.0f);
            }
        }
    }
}

void iqk_dequantize_row_iq4_ks_r16_generic(const block_iq4_ks_r16 * x, float * y, int64_t k, const float * d16) {
    // 16 rows, k elements per row (k is a multiple of 32)
    const int64_t nblocks = k / 32;
    for (int r = 0; r < 16; ++r) {
        float * yr = y + r * k;
        const float dr = d16 ? d16[r] : 1.0f;
        for (int64_t b = 0; b < nblocks; ++b) {
            const block_iq4_ks_r16 * blk = x + b;
            const float scale = dr * (float)(blk->scales[r] + 1);
            const uint8_t * qsrc = blk->qs + r * 16;
            float * yb = yr + b * 32;

            for (int i = 0; i < 16; ++i) {
                const uint8_t byte = qsrc[i];
                const int q0 = byte & 0x0F;
                const int q1 = byte >> 4;
                yb[i * 2 + 0] = scale * ((float)q0 - 8.0f);
                yb[i * 2 + 1] = scale * ((float)q1 - 8.0f);
            }
        }
    }
}

// =============================================================================
// Vector Dot Products: Generic Reference Implementations
// =============================================================================

void iqk_vec_dot_iq2_ks_q8_k_generic(int n, float * s, const void * vx, const void * vy, float d) {
    const int nb = n / IQK_QK_K;
    const block_iq2_ks * x = (const block_iq2_ks *)vx;
    const q8_K_ref     * y = (const q8_K_ref *)vy;

    float sum = 0.0f;
    for (int b = 0; b < nb; ++b) {
        const block_iq2_ks * blk_x = x + b;
        const q8_K_ref     * blk_y = y + b;
        const float dy = blk_y->d;

        for (int g = 0; g < 4; ++g) {
            const float scale = d * (float)(blk_x->scales[g] + 1) * dy;
            const uint8_t * qsrc = blk_x->qs + g * 16;
            const int8_t  * ysrc = blk_y->qs + g * 64;

            int32_t dot = 0;
            int32_t sum_y = 0;
            for (int i = 0; i < 16; ++i) {
                const uint8_t byte = qsrc[i];
                const int q0 = (byte     ) & 3;
                const int q1 = (byte >> 2) & 3;
                const int q2 = (byte >> 4) & 3;
                const int q3 = (byte >> 6) & 3;

                const int8_t y0 = ysrc[i * 4 + 0];
                const int8_t y1 = ysrc[i * 4 + 1];
                const int8_t y2 = ysrc[i * 4 + 2];
                const int8_t y3 = ysrc[i * 4 + 3];

                dot += q0 * y0 + q1 * y1 + q2 * y2 + q3 * y3;
                sum_y += y0 + y1 + y2 + y3;
            }
            sum += scale * ((float)dot - 1.5f * (float)sum_y);
        }
    }
    *s = sum;
}

void iqk_vec_dot_iq3_ks_q8_k_generic(int n, float * s, const void * vx, const void * vy, float d) {
    const int nb = n / IQK_QK_K;
    const block_iq3_ks * x = (const block_iq3_ks *)vx;
    const q8_K_ref     * y = (const q8_K_ref *)vy;

    float sum = 0.0f;
    for (int b = 0; b < nb; ++b) {
        const block_iq3_ks * blk_x = x + b;
        const q8_K_ref     * blk_y = y + b;
        const float dy = blk_y->d;

        for (int g = 0; g < 4; ++g) {
            const float scale = d * (float)(blk_x->scales[g] + 1) * dy;
            const uint8_t * qsrc = blk_x->qs + g * 16;
            const uint8_t * hsrc = blk_x->qh + g * 8;
            const int8_t  * ysrc = blk_y->qs + g * 64;

            int32_t dot = 0;
            int32_t sum_y = 0;
            for (int i = 0; i < 64; ++i) {
                const int q_lo = (qsrc[i / 4] >> ((i % 4) * 2)) & 3;
                const int q_hi = (hsrc[i / 8] >> (i % 8)) & 1;
                const int q = q_lo | (q_hi << 2);
                const int8_t yi = ysrc[i];

                dot += q * yi;
                sum_y += yi;
            }
            sum += scale * ((float)dot - 3.5f * (float)sum_y);
        }
    }
    *s = sum;
}

void iqk_vec_dot_iq4_ks_q8_k_generic(int n, float * s, const void * vx, const void * vy, float d) {
    const int nb = n / IQK_QK_K;
    const block_iq4_ks * x = (const block_iq4_ks *)vx;
    const q8_K_ref     * y = (const q8_K_ref *)vy;

    float sum = 0.0f;
    for (int b = 0; b < nb; ++b) {
        const block_iq4_ks * blk_x = x + b;
        const q8_K_ref     * blk_y = y + b;
        const float dy = blk_y->d;

        for (int g = 0; g < 8; ++g) {
            const float scale = d * (float)(blk_x->scales[g] + 1) * dy;
            const uint8_t * qsrc = blk_x->qs + g * 16;
            const int8_t  * ysrc = blk_y->qs + g * 32;

            int32_t dot = 0;
            int32_t sum_y = 0;
            for (int i = 0; i < 16; ++i) {
                const uint8_t byte = qsrc[i];
                const int q0 = byte & 0x0F;
                const int q1 = byte >> 4;

                const int8_t y0 = ysrc[i * 2 + 0];
                const int8_t y1 = ysrc[i * 2 + 1];

                dot += q0 * y0 + q1 * y1;
                sum_y += y0 + y1;
            }
            sum += scale * ((float)dot - 8.0f * (float)sum_y);
        }
    }
    *s = sum;
}

void iqk_vec_dot_iq4_ks_r16_q8_0_generic(int n, float * s, const void * vx, const void * vy, const float * d16) {
    const int nblocks = n / 32;
    const block_iq4_ks_r16 * x = (const block_iq4_ks_r16 *)vx;
    const q8_0_ref         * y = (const q8_0_ref *)vy;

    for (int r = 0; r < 16; ++r) {
        float sum = 0.0f;
        const float dr = d16 ? d16[r] : 1.0f;

        for (int b = 0; b < nblocks; ++b) {
            const block_iq4_ks_r16 * blk_x = x + b;
            const q8_0_ref         * blk_y = y + b;
            const float dy = fp16_to_fp32_local(blk_y->d);
            const float scale = dr * (float)(blk_x->scales[r] + 1) * dy;
            const uint8_t * qsrc = blk_x->qs + r * 16;
            const int8_t  * ysrc = blk_y->qs;

            int32_t dot = 0;
            int32_t sum_y = 0;
            for (int i = 0; i < 16; ++i) {
                const uint8_t byte = qsrc[i];
                const int q0 = byte & 0x0F;
                const int q1 = byte >> 4;

                const int8_t y0 = ysrc[i * 2 + 0];
                const int8_t y1 = ysrc[i * 2 + 1];

                dot += q0 * y0 + q1 * y1;
                sum_y += y0 + y1;
            }
            sum += scale * ((float)dot - 8.0f * (float)sum_y);
        }
        s[r] = sum;
    }
}
