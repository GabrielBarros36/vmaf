/**
 * Degenerate Input Test Suite -- Synthetic Data Generators
 *
 * Shared fill functions for producing degenerate/edge-case pixel patterns
 * in VmafPicture buffers. All generators operate on pre-allocated pictures
 * (via vmaf_picture_alloc) and fill luma + chroma planes for YUV420P or
 * luma-only for YUV400P.
 *
 * See plans/specs/degenerate-input-tests.md for the full specification.
 */

#ifndef TEST_DEGENERATE_COMMON_H
#define TEST_DEGENERATE_COMMON_H

#include <stdint.h>
#include <string.h>
#include <math.h>
#include "libvmaf/picture.h"

/* ========== Helpers ========== */

/** Get the maximum pixel value for a given bit depth */
static inline unsigned degen_max_val(unsigned bpc) {
    return (1u << bpc) - 1;
}

/** Get the mid-gray value for a given bit depth */
static inline unsigned degen_mid_val(unsigned bpc) {
    return 1u << (bpc - 1);
}

/** xorshift32 PRNG -- fast, deterministic, reproducible */
static inline uint32_t xorshift32(uint32_t *state) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

/** Number of planes to fill based on pixel format */
static inline unsigned degen_n_planes(const VmafPicture *pic) {
    return (pic->pix_fmt == VMAF_PIX_FMT_YUV400P) ? 1 : 3;
}

/** Set a single pixel in the picture (handles 8-bit and >8-bit) */
static inline void degen_set_pixel(VmafPicture *pic, unsigned plane,
                                   unsigned row, unsigned col, unsigned val)
{
    if (pic->bpc <= 8) {
        uint8_t *r = (uint8_t *)pic->data[plane] + row * pic->stride[plane];
        r[col] = (uint8_t)val;
    } else {
        uint16_t *r = (uint16_t *)((uint8_t *)pic->data[plane] +
                                    row * pic->stride[plane]);
        r[col] = (uint16_t)val;
    }
}

/* ========== Fill Functions (P1-P10) ========== */

/** P1: All pixels = 0 (black) */
static inline void degen_fill_black(VmafPicture *pic) {
    unsigned np = degen_n_planes(pic);
    for (unsigned p = 0; p < np; p++) {
        for (unsigned i = 0; i < pic->h[p]; i++) {
            uint8_t *row = (uint8_t *)pic->data[p] + i * pic->stride[p];
            memset(row, 0, pic->stride[p]);
        }
    }
}

/** P2: All pixels = (1 << bpc) - 1 (white / max) */
static inline void degen_fill_white(VmafPicture *pic) {
    unsigned np = degen_n_planes(pic);
    unsigned max = degen_max_val(pic->bpc);
    for (unsigned p = 0; p < np; p++) {
        for (unsigned i = 0; i < pic->h[p]; i++) {
            for (unsigned j = 0; j < pic->w[p]; j++)
                degen_set_pixel(pic, p, i, j, max);
        }
    }
}

/** P3/P8: All pixels = (1 << bpc) / 2 (mid-gray) */
static inline void degen_fill_midgray(VmafPicture *pic) {
    unsigned np = degen_n_planes(pic);
    unsigned mid = degen_mid_val(pic->bpc);
    for (unsigned p = 0; p < np; p++) {
        for (unsigned i = 0; i < pic->h[p]; i++) {
            for (unsigned j = 0; j < pic->w[p]; j++)
                degen_set_pixel(pic, p, i, j, mid);
        }
    }
}

/** P4: All pixels = mid-gray, then pixel at (0,0) in Y plane = mid+1 */
static inline void degen_fill_single_pixel_diff(VmafPicture *pic) {
    degen_fill_midgray(pic);
    unsigned mid = degen_mid_val(pic->bpc);
    degen_set_pixel(pic, 0, 0, 0, mid + 1);
}

/** P5: Horizontal gradient ramp from 0 to max across each row */
static inline void degen_fill_gradient(VmafPicture *pic) {
    unsigned np = degen_n_planes(pic);
    unsigned max = degen_max_val(pic->bpc);
    for (unsigned p = 0; p < np; p++) {
        unsigned w = pic->w[p];
        for (unsigned i = 0; i < pic->h[p]; i++) {
            for (unsigned j = 0; j < w; j++) {
                unsigned val = (unsigned)((uint64_t)j * max /
                               (w > 1 ? w - 1 : 1));
                degen_set_pixel(pic, p, i, j, val);
            }
        }
    }
}

/** P6: Deterministic PRNG fill (xorshift32, given seed) */
static inline void degen_fill_random(VmafPicture *pic, uint32_t seed) {
    uint32_t state = seed;
    unsigned max = degen_max_val(pic->bpc);
    unsigned np = degen_n_planes(pic);
    for (unsigned p = 0; p < np; p++) {
        for (unsigned i = 0; i < pic->h[p]; i++) {
            for (unsigned j = 0; j < pic->w[p]; j++) {
                unsigned val = xorshift32(&state) % (max + 1);
                degen_set_pixel(pic, p, i, j, val);
            }
        }
    }
}

/** P7: Checkerboard: even positions = 0, odd positions = max */
static inline void degen_fill_checkerboard(VmafPicture *pic) {
    unsigned np = degen_n_planes(pic);
    unsigned max = degen_max_val(pic->bpc);
    for (unsigned p = 0; p < np; p++) {
        for (unsigned i = 0; i < pic->h[p]; i++) {
            for (unsigned j = 0; j < pic->w[p]; j++) {
                unsigned val = ((i + j) & 1) ? max : 0;
                degen_set_pixel(pic, p, i, j, val);
            }
        }
    }
}

/** P8: DC constant -- every pixel set to the given value */
static inline void degen_fill_constant(VmafPicture *pic, unsigned value) {
    unsigned np = degen_n_planes(pic);
    for (unsigned p = 0; p < np; p++) {
        for (unsigned i = 0; i < pic->h[p]; i++) {
            for (unsigned j = 0; j < pic->w[p]; j++)
                degen_set_pixel(pic, p, i, j, value);
        }
    }
}

/** P9: Single pixel at center = max, rest = 0 (impulse) */
static inline void degen_fill_impulse(VmafPicture *pic) {
    unsigned np = degen_n_planes(pic);
    unsigned max = degen_max_val(pic->bpc);
    for (unsigned p = 0; p < np; p++) {
        for (unsigned i = 0; i < pic->h[p]; i++) {
            uint8_t *row = (uint8_t *)pic->data[p] + i * pic->stride[p];
            memset(row, 0, pic->stride[p]);
        }
        /* Set center pixel to max */
        degen_set_pixel(pic, p, pic->h[p] / 2, pic->w[p] / 2, max);
    }
}

/** P10: First/last row and first/last column = max, rest = 0 */
static inline void degen_fill_boundary_stripe(VmafPicture *pic) {
    unsigned np = degen_n_planes(pic);
    unsigned max = degen_max_val(pic->bpc);
    /* First clear everything */
    for (unsigned p = 0; p < np; p++) {
        for (unsigned i = 0; i < pic->h[p]; i++) {
            uint8_t *row = (uint8_t *)pic->data[p] + i * pic->stride[p];
            memset(row, 0, pic->stride[p]);
        }
    }
    /* Then set border pixels to max */
    for (unsigned p = 0; p < np; p++) {
        unsigned w = pic->w[p];
        unsigned h = pic->h[p];
        for (unsigned j = 0; j < w; j++) {
            degen_set_pixel(pic, p, 0, j, max);         /* top row */
            degen_set_pixel(pic, p, h - 1, j, max);     /* bottom row */
        }
        for (unsigned i = 0; i < h; i++) {
            degen_set_pixel(pic, p, i, 0, max);          /* left col */
            degen_set_pixel(pic, p, i, w - 1, max);      /* right col */
        }
    }
}

/* ========== Score Validity Macros ========== */

static inline int score_is_valid(double score) {
    return isfinite(score);
}

#define mu_assert_score_finite(msg, score) \
    mu_assert(msg " produced non-finite score", score_is_valid(score))

#endif /* TEST_DEGENERATE_COMMON_H */
