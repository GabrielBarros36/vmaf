/**
 * SIMD Oracle Tests — Motion Module
 *
 * Tests x_convolution_16 C reference vs AVX2/AVX-512 implementations.
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdio.h>

#include "test.h"
#include "test_simd_common.h"

#include "config.h"
#include "cpu.h"
#include "feature/integer_motion.h"
#include "feature/common/alignment.h"

#if ARCH_X86
#include "feature/x86/motion_avx2.h"
#if HAVE_AVX512
#include "feature/x86/motion_avx512.h"
#endif
#endif

/* ========== C Reference Implementation ========== */

static void ref_x_convolution_16(const uint16_t *src, uint16_t *dst,
    unsigned width, unsigned height, ptrdiff_t src_stride, ptrdiff_t dst_stride)
{
    const unsigned radius = filter_width / 2;
    const unsigned left_edge = vmaf_ceiln(radius, 1);
    const unsigned right_edge = vmaf_floorn(width - (filter_width - radius), 1);
    const unsigned shift_add_round = 32768;

    uint16_t *src_p = (uint16_t *)src + (left_edge - radius);
    for (unsigned i = 0; i < height; ++i) {
        for (unsigned j = 0; j < left_edge; j++) {
            dst[i * dst_stride + j] =
                (edge_16(true, src, width, height, src_stride, i, j) +
                 shift_add_round) >> 16;
        }

        uint16_t *src_p1 = src_p;
        for (unsigned j = left_edge; j < right_edge; j++) {
            uint32_t accum = 0;
            uint16_t *src_p2 = src_p1;
            for (int k = 0; k < filter_width; ++k) {
                accum += filter[k] * (*src_p2);
                src_p2++;
            }
            src_p1++;
            dst[i * dst_stride + j] = (accum + shift_add_round) >> 16;
        }

        for (unsigned j = right_edge; j < width; j++) {
            dst[i * dst_stride + j] =
                (edge_16(true, src, width, height, src_stride, i, j) +
                 shift_add_round) >> 16;
        }

        src_p += src_stride;
    }
}

/* ========== AVX2 Oracle Test ========== */

#if ARCH_X86
static char *test_x_convolution_16_avx2(void) {
    for (unsigned d = 0; d < NUM_STANDARD_DIMS; d++) {
        unsigned w = standard_dims[d].w;
        unsigned h = standard_dims[d].h;
        ptrdiff_t stride = (ptrdiff_t)(ALIGN_CEIL(w * sizeof(uint16_t)) / sizeof(uint16_t));

        for (int cat = 0; cat < INPUT_CATEGORY_COUNT; cat++) {
            /* Allocate source */
            size_t src_size = stride * h * sizeof(uint16_t);
            void *src_mem = aligned_malloc(src_size, MAX_ALIGN);
            mu_assert("src alloc failed", src_mem != NULL);
            generate_input_16((uint16_t *)src_mem, w, h, stride,
                (InputCategory)cat, 65535);

            /* Allocate output buffers with guards */
            size_t dst_size = stride * h * sizeof(uint16_t);
            GuardedBuffer gb_ref, gb_simd;
            mu_assert("alloc ref failed", guarded_buf_alloc(&gb_ref, dst_size) == 0);
            mu_assert("alloc simd failed", guarded_buf_alloc(&gb_simd, dst_size) == 0);
            memset(gb_ref.data, 0, dst_size);
            memset(gb_simd.data, 0, dst_size);

            /* Run C reference */
            ref_x_convolution_16((const uint16_t *)src_mem,
                (uint16_t *)gb_ref.data, w, h, stride, stride);

            /* Run AVX2 */
            x_convolution_16_avx2((const uint16_t *)src_mem,
                (uint16_t *)gb_simd.data, w, h, stride, stride);

            /* Compare element by element for better diagnostics */
            uint16_t *ref_out = (uint16_t *)gb_ref.data;
            uint16_t *simd_out = (uint16_t *)gb_simd.data;
            for (unsigned i = 0; i < h; i++) {
                for (unsigned j = 0; j < w; j++) {
                    uint16_t rv = ref_out[i * stride + j];
                    uint16_t sv = simd_out[i * stride + j];
                    if (rv != sv) {
                        snprintf(diag_msg, sizeof(diag_msg),
                            "x_convolution_16_avx2 MISMATCH [%s %ux%u]: "
                            "pos (%u,%u), ref=%u, simd=%u",
                            input_category_names[cat], w, h, i, j, rv, sv);
                        mu_assert(diag_msg, 0);
                    }
                }
            }

            /* Check guards */
            mu_assert("x_conv avx2: SIMD guard before corrupted",
                guard_before_intact(&gb_simd));
            mu_assert("x_conv avx2: SIMD guard after corrupted",
                guard_after_intact(&gb_simd));

            aligned_free(src_mem);
            guarded_buf_free(&gb_ref);
            guarded_buf_free(&gb_simd);
        }
    }
    return NULL;
}
#endif

/* ========== AVX-512 Oracle Test ========== */

#if ARCH_X86 && HAVE_AVX512
static char *test_x_convolution_16_avx512(void) {
    for (unsigned d = 0; d < NUM_STANDARD_DIMS; d++) {
        unsigned w = standard_dims[d].w;
        unsigned h = standard_dims[d].h;
        ptrdiff_t stride = (ptrdiff_t)(ALIGN_CEIL(w * sizeof(uint16_t)) / sizeof(uint16_t));

        for (int cat = 0; cat < INPUT_CATEGORY_COUNT; cat++) {
            size_t src_size = stride * h * sizeof(uint16_t);
            void *src_mem = aligned_malloc(src_size, MAX_ALIGN);
            mu_assert("src alloc failed", src_mem != NULL);
            generate_input_16((uint16_t *)src_mem, w, h, stride,
                (InputCategory)cat, 65535);

            size_t dst_size = stride * h * sizeof(uint16_t);
            GuardedBuffer gb_ref, gb_simd;
            mu_assert("alloc ref failed", guarded_buf_alloc(&gb_ref, dst_size) == 0);
            mu_assert("alloc simd failed", guarded_buf_alloc(&gb_simd, dst_size) == 0);
            memset(gb_ref.data, 0, dst_size);
            memset(gb_simd.data, 0, dst_size);

            ref_x_convolution_16((const uint16_t *)src_mem,
                (uint16_t *)gb_ref.data, w, h, stride, stride);

            x_convolution_16_avx512((const uint16_t *)src_mem,
                (uint16_t *)gb_simd.data, w, h, stride, stride);

            uint16_t *ref_out = (uint16_t *)gb_ref.data;
            uint16_t *simd_out = (uint16_t *)gb_simd.data;
            for (unsigned i = 0; i < h; i++) {
                for (unsigned j = 0; j < w; j++) {
                    uint16_t rv = ref_out[i * stride + j];
                    uint16_t sv = simd_out[i * stride + j];
                    if (rv != sv) {
                        snprintf(diag_msg, sizeof(diag_msg),
                            "x_convolution_16_avx512 MISMATCH [%s %ux%u]: "
                            "pos (%u,%u), ref=%u, simd=%u",
                            input_category_names[cat], w, h, i, j, rv, sv);
                        mu_assert(diag_msg, 0);
                    }
                }
            }

            mu_assert("x_conv avx512: SIMD guard before corrupted",
                guard_before_intact(&gb_simd));
            mu_assert("x_conv avx512: SIMD guard after corrupted",
                guard_after_intact(&gb_simd));

            aligned_free(src_mem);
            guarded_buf_free(&gb_ref);
            guarded_buf_free(&gb_simd);
        }
    }
    return NULL;
}
#endif

/* ========== Test Runner ========== */

char *run_tests(void) {
    vmaf_init_cpu();
#if ARCH_X86
    mu_run_test(test_x_convolution_16_avx2);
#if HAVE_AVX512
    if (vmaf_get_cpu_flags() & VMAF_X86_CPU_FLAG_AVX512) {
        mu_run_test(test_x_convolution_16_avx512);
    } else {
        fprintf(stderr, "test_x_convolution_16_avx512: skipped (no AVX-512)\n");
    }
#endif
#endif
    return NULL;
}
