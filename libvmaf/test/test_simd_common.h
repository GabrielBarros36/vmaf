/**
 * SIMD Oracle Test Common Utilities
 *
 * Shared test data generators, buffer management, and comparison
 * utilities for SIMD correctness oracle tests.
 */

#ifndef TEST_SIMD_COMMON_H_
#define TEST_SIMD_COMMON_H_

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#include "mem.h"

/* ========== Constants ========== */

#define GUARD_SIZE 64
#define GUARD_SENTINEL 0xAA
#define FILL_SENTINEL 0xDE

/* ========== Input Categories ========== */

typedef enum {
    INPUT_ZERO = 0,
    INPUT_MAX,
    INPUT_CONSTANT,
    INPUT_GRADIENT_H,
    INPUT_GRADIENT_V,
    INPUT_CHECKERBOARD,
    INPUT_RANDOM_42,
    INPUT_RANDOM_123,
    INPUT_SINGLE_HOT,
    INPUT_BOUNDARY_STRIPE,
    INPUT_CATEGORY_COUNT
} InputCategory;

static const char *input_category_names[] = {
    "zero", "max", "constant", "gradient_h", "gradient_v",
    "checkerboard", "random_42", "random_123", "single_hot", "boundary_stripe"
};

/* ========== Test Dimensions ========== */

typedef struct {
    unsigned w;
    unsigned h;
} TestDim;

static const TestDim standard_dims[] = {
    {8, 8}, {16, 16}, {24, 24}, {32, 32},
    {64, 64}, {120, 68}, {576, 324}, {1920, 1080}
};
#define NUM_STANDARD_DIMS (sizeof(standard_dims) / sizeof(standard_dims[0]))

/* Additional ADM dimensions for non-multiple-of-8 widths */
static const TestDim adm_fallback_dims[] = {
    {7, 8}, {9, 8}, {15, 8}
};
#define NUM_ADM_FALLBACK_DIMS (sizeof(adm_fallback_dims) / sizeof(adm_fallback_dims[0]))

/* ========== Simple Deterministic PRNG (xorshift32) ========== */

static inline uint32_t prng_next(uint32_t *state) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

/* ========== Guarded Buffer Allocation ========== */

typedef struct {
    void *base;          /* Raw allocation pointer (for free) */
    void *data;          /* Aligned data pointer (past guard region) */
    size_t data_size;    /* Size of the data region in bytes */
    uint8_t *guard_before; /* Start of before-guard region */
    uint8_t *guard_after;  /* Start of after-guard region */
} GuardedBuffer;

static inline int guarded_buf_alloc(GuardedBuffer *gb, size_t data_size) {
    size_t total = GUARD_SIZE + data_size + GUARD_SIZE + MAX_ALIGN;
    gb->base = malloc(total);
    if (!gb->base) return -1;

    /* Align the data region to MAX_ALIGN (32 bytes) past the guard */
    uintptr_t raw = (uintptr_t)gb->base + GUARD_SIZE;
    uintptr_t aligned = (raw + MAX_ALIGN - 1) & ~(uintptr_t)(MAX_ALIGN - 1);

    gb->data = (void *)aligned;
    gb->data_size = data_size;
    gb->guard_before = (uint8_t *)aligned - GUARD_SIZE;
    gb->guard_after = (uint8_t *)aligned + data_size;

    /* Fill guard regions with sentinel */
    memset(gb->guard_before, GUARD_SENTINEL, GUARD_SIZE);
    memset(gb->guard_after, GUARD_SENTINEL, GUARD_SIZE);

    /* Fill data region with known pattern */
    memset(gb->data, FILL_SENTINEL, data_size);

    return 0;
}

static inline void guarded_buf_free(GuardedBuffer *gb) {
    if (gb->base) {
        free(gb->base);
        gb->base = NULL;
        gb->data = NULL;
    }
}

static inline int guard_before_intact(const GuardedBuffer *gb) {
    for (size_t i = 0; i < GUARD_SIZE; i++) {
        if (gb->guard_before[i] != GUARD_SENTINEL) return 0;
    }
    return 1;
}

static inline int guard_after_intact(const GuardedBuffer *gb) {
    for (size_t i = 0; i < GUARD_SIZE; i++) {
        if (gb->guard_after[i] != GUARD_SENTINEL) return 0;
    }
    return 1;
}

/* ========== Input Data Generators ========== */

static inline void generate_input_8(uint8_t *buf, unsigned w, unsigned h,
                                     ptrdiff_t stride, InputCategory cat)
{
    memset(buf, 0, stride * h);
    switch (cat) {
    case INPUT_ZERO:
        break;
    case INPUT_MAX:
        for (unsigned i = 0; i < h; i++)
            memset(buf + i * stride, 255, w);
        break;
    case INPUT_CONSTANT:
        for (unsigned i = 0; i < h; i++)
            memset(buf + i * stride, 128, w);
        break;
    case INPUT_GRADIENT_H:
        for (unsigned i = 0; i < h; i++)
            for (unsigned j = 0; j < w; j++)
                buf[i * stride + j] = (uint8_t)((j * 255) / (w > 1 ? w - 1 : 1));
        break;
    case INPUT_GRADIENT_V:
        for (unsigned i = 0; i < h; i++)
            for (unsigned j = 0; j < w; j++)
                buf[i * stride + j] = (uint8_t)((i * 255) / (h > 1 ? h - 1 : 1));
        break;
    case INPUT_CHECKERBOARD:
        for (unsigned i = 0; i < h; i++)
            for (unsigned j = 0; j < w; j++)
                buf[i * stride + j] = ((i + j) & 1) ? 255 : 0;
        break;
    case INPUT_RANDOM_42: {
        uint32_t state = 42;
        for (unsigned i = 0; i < h; i++)
            for (unsigned j = 0; j < w; j++)
                buf[i * stride + j] = (uint8_t)(prng_next(&state) & 0xFF);
        break;
    }
    case INPUT_RANDOM_123: {
        uint32_t state = 123;
        for (unsigned i = 0; i < h; i++)
            for (unsigned j = 0; j < w; j++)
                buf[i * stride + j] = (uint8_t)(prng_next(&state) & 0xFF);
        break;
    }
    case INPUT_SINGLE_HOT:
        buf[(h / 2) * stride + (w / 2)] = 255;
        break;
    case INPUT_BOUNDARY_STRIPE:
        for (unsigned j = 0; j < w; j++) {
            buf[0 * stride + j] = 255;
            buf[(h - 1) * stride + j] = 255;
        }
        for (unsigned i = 0; i < h; i++) {
            buf[i * stride + 0] = 255;
            buf[i * stride + (w - 1)] = 255;
        }
        break;
    default:
        break;
    }
}

static inline void generate_input_16(uint16_t *buf, unsigned w, unsigned h,
                                      ptrdiff_t stride_elems, InputCategory cat,
                                      uint16_t max_val)
{
    for (unsigned i = 0; i < h; i++)
        memset(buf + i * stride_elems, 0, stride_elems * sizeof(uint16_t));
    switch (cat) {
    case INPUT_ZERO:
        break;
    case INPUT_MAX:
        for (unsigned i = 0; i < h; i++)
            for (unsigned j = 0; j < w; j++)
                buf[i * stride_elems + j] = max_val;
        break;
    case INPUT_CONSTANT:
        for (unsigned i = 0; i < h; i++)
            for (unsigned j = 0; j < w; j++)
                buf[i * stride_elems + j] = max_val / 2;
        break;
    case INPUT_GRADIENT_H:
        for (unsigned i = 0; i < h; i++)
            for (unsigned j = 0; j < w; j++)
                buf[i * stride_elems + j] = (uint16_t)((uint32_t)j * max_val / (w > 1 ? w - 1 : 1));
        break;
    case INPUT_GRADIENT_V:
        for (unsigned i = 0; i < h; i++)
            for (unsigned j = 0; j < w; j++)
                buf[i * stride_elems + j] = (uint16_t)((uint32_t)i * max_val / (h > 1 ? h - 1 : 1));
        break;
    case INPUT_CHECKERBOARD:
        for (unsigned i = 0; i < h; i++)
            for (unsigned j = 0; j < w; j++)
                buf[i * stride_elems + j] = ((i + j) & 1) ? max_val : 0;
        break;
    case INPUT_RANDOM_42: {
        uint32_t state = 42;
        for (unsigned i = 0; i < h; i++)
            for (unsigned j = 0; j < w; j++)
                buf[i * stride_elems + j] = (uint16_t)(prng_next(&state) % (max_val + 1));
        break;
    }
    case INPUT_RANDOM_123: {
        uint32_t state = 123;
        for (unsigned i = 0; i < h; i++)
            for (unsigned j = 0; j < w; j++)
                buf[i * stride_elems + j] = (uint16_t)(prng_next(&state) % (max_val + 1));
        break;
    }
    case INPUT_SINGLE_HOT:
        buf[(h / 2) * stride_elems + (w / 2)] = max_val;
        break;
    case INPUT_BOUNDARY_STRIPE:
        for (unsigned j = 0; j < w; j++) {
            buf[0 * stride_elems + j] = max_val;
            buf[(h - 1) * stride_elems + j] = max_val;
        }
        for (unsigned i = 0; i < h; i++) {
            buf[i * stride_elems + 0] = max_val;
            buf[i * stride_elems + (w - 1)] = max_val;
        }
        break;
    default:
        break;
    }
}

/* ========== Comparison Utilities ========== */

/* Diagnostic message buffer - static to avoid stack overflow */
static char diag_msg[512];

static inline int integers_match(const void *ref, const void *simd,
                                  size_t size, const char *func_name,
                                  const char *category, unsigned w, unsigned h,
                                  int elem_size)
{
    if (memcmp(ref, simd, size) == 0) return 1;

    /* Find first difference */
    const uint8_t *r = (const uint8_t *)ref;
    const uint8_t *s = (const uint8_t *)simd;
    for (size_t i = 0; i < size; i++) {
        if (r[i] != s[i]) {
            size_t elem_offset = i / elem_size;
            if (elem_size == 2) {
                int16_t rv = ((const int16_t *)ref)[elem_offset];
                int16_t sv = ((const int16_t *)simd)[elem_offset];
                snprintf(diag_msg, sizeof(diag_msg),
                    "%s MISMATCH [%s %ux%u]: elem %zu, ref=%d, simd=%d",
                    func_name, category, w, h, elem_offset, rv, sv);
            } else {
                snprintf(diag_msg, sizeof(diag_msg),
                    "%s MISMATCH [%s %ux%u]: byte %zu, ref=0x%02x, simd=0x%02x",
                    func_name, category, w, h, i, r[i], s[i]);
            }
            return 0;
        }
    }
    return 1;
}

static inline int float_eq(float a, float b) {
    if (fabsf(a) < 1e-9f && fabsf(b) < 1e-9f) return 1;
    return fabsf(a - b) / fmaxf(fabsf(a), fabsf(b)) < 1e-6f;
}

static inline int floats_match(float ref_num, float ref_den,
                                float simd_num, float simd_den,
                                const char *func_name, const char *category,
                                unsigned w, unsigned h)
{
    if (float_eq(ref_num, simd_num) && float_eq(ref_den, simd_den))
        return 1;

    float rel_err_num = 0, rel_err_den = 0;
    if (fabsf(ref_num) > 1e-9f)
        rel_err_num = fabsf(ref_num - simd_num) / fabsf(ref_num);
    if (fabsf(ref_den) > 1e-9f)
        rel_err_den = fabsf(ref_den - simd_den) / fabsf(ref_den);

    snprintf(diag_msg, sizeof(diag_msg),
        "%s MISMATCH [%s %ux%u]: num ref=%e simd=%e (rel_err=%e), "
        "den ref=%e simd=%e (rel_err=%e)",
        func_name, category, w, h,
        ref_num, simd_num, rel_err_num,
        ref_den, simd_den, rel_err_den);
    return 0;
}

#endif /* TEST_SIMD_COMMON_H_ */
