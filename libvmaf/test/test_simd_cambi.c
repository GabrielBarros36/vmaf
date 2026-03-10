/**
 * SIMD Oracle Tests — CAMBI Module
 *
 * Tests increment_range, decrement_range, and get_derivative_data_for_row
 * C reference vs AVX2 implementations.
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "test.h"
#include "test_simd_common.h"

#include "config.h"

#if ARCH_X86
#include "feature/x86/cambi_avx2.h"
#endif

/* ========== C Reference Implementations ========== */

static void ref_increment_range(uint16_t *arr, int left, int right) {
    for (int i = left; i < right; i++) {
        arr[i]++;
    }
}

static void ref_decrement_range(uint16_t *arr, int left, int right) {
    for (int i = left; i < right; i++) {
        arr[i]--;
    }
}

static void ref_get_derivative_data_for_row(const uint16_t *image_data,
    uint16_t *derivative_buffer, int width, int height, int row, int stride)
{
    for (int col = 0; col < width; col++) {
        int horizontal_derivative = (col == width - 1 ||
            image_data[row * stride + col] == image_data[row * stride + col + 1]);
        int vertical_derivative = (row == height - 1 ||
            image_data[row * stride + col] == image_data[(row + 1) * stride + col]);
        derivative_buffer[col] = horizontal_derivative && vertical_derivative;
    }
}

/* ========== Increment Range Tests ========== */

#if ARCH_X86
static char *test_increment_range_avx2(void) {
    for (unsigned d = 0; d < NUM_STANDARD_DIMS; d++) {
        unsigned w = standard_dims[d].w;
        unsigned h = standard_dims[d].h;
        unsigned arr_len = w * h;

        for (int cat = 0; cat < INPUT_CATEGORY_COUNT; cat++) {
            /* Allocate two guarded buffers */
            GuardedBuffer gb_ref, gb_simd;
            size_t buf_size = arr_len * sizeof(uint16_t);
            mu_assert("alloc ref failed", guarded_buf_alloc(&gb_ref, buf_size) == 0);
            mu_assert("alloc simd failed", guarded_buf_alloc(&gb_simd, buf_size) == 0);

            uint16_t *arr_ref = (uint16_t *)gb_ref.data;
            uint16_t *arr_simd = (uint16_t *)gb_simd.data;

            /* Generate input data */
            generate_input_16(arr_ref, w, h, w, (InputCategory)cat, 1023);
            memcpy(arr_simd, arr_ref, buf_size);

            /* Run both on the full range */
            int left = 0;
            int right = (int)arr_len;

            ref_increment_range(arr_ref, left, right);
            cambi_increment_range_avx2(arr_simd, left, right);

            /* Compare */
            mu_assert(diag_msg,
                integers_match(arr_ref, arr_simd, buf_size,
                    "increment_range_avx2", input_category_names[cat], w, h, 2));

            /* Check guards */
            mu_assert("increment_range: guard before corrupted",
                guard_before_intact(&gb_ref));
            mu_assert("increment_range: guard after corrupted",
                guard_after_intact(&gb_ref));
            mu_assert("increment_range: SIMD guard before corrupted",
                guard_before_intact(&gb_simd));
            mu_assert("increment_range: SIMD guard after corrupted",
                guard_after_intact(&gb_simd));

            guarded_buf_free(&gb_ref);
            guarded_buf_free(&gb_simd);

            /* Also test partial ranges */
            if (arr_len >= 4) {
                mu_assert("alloc ref2 failed", guarded_buf_alloc(&gb_ref, buf_size) == 0);
                mu_assert("alloc simd2 failed", guarded_buf_alloc(&gb_simd, buf_size) == 0);
                arr_ref = (uint16_t *)gb_ref.data;
                arr_simd = (uint16_t *)gb_simd.data;

                generate_input_16(arr_ref, w, h, w, (InputCategory)cat, 1023);
                memcpy(arr_simd, arr_ref, buf_size);

                left = (int)(arr_len / 4);
                right = (int)(arr_len * 3 / 4);

                ref_increment_range(arr_ref, left, right);
                cambi_increment_range_avx2(arr_simd, left, right);

                mu_assert(diag_msg,
                    integers_match(arr_ref, arr_simd, buf_size,
                        "increment_range_avx2 (partial)", input_category_names[cat], w, h, 2));

                mu_assert("increment_range partial: SIMD guard before corrupted",
                    guard_before_intact(&gb_simd));
                mu_assert("increment_range partial: SIMD guard after corrupted",
                    guard_after_intact(&gb_simd));

                guarded_buf_free(&gb_ref);
                guarded_buf_free(&gb_simd);
            }
        }
    }
    return NULL;
}
#endif

/* ========== Decrement Range Tests ========== */

#if ARCH_X86
static char *test_decrement_range_avx2(void) {
    for (unsigned d = 0; d < NUM_STANDARD_DIMS; d++) {
        unsigned w = standard_dims[d].w;
        unsigned h = standard_dims[d].h;
        unsigned arr_len = w * h;

        for (int cat = 0; cat < INPUT_CATEGORY_COUNT; cat++) {
            GuardedBuffer gb_ref, gb_simd;
            size_t buf_size = arr_len * sizeof(uint16_t);
            mu_assert("alloc ref failed", guarded_buf_alloc(&gb_ref, buf_size) == 0);
            mu_assert("alloc simd failed", guarded_buf_alloc(&gb_simd, buf_size) == 0);

            uint16_t *arr_ref = (uint16_t *)gb_ref.data;
            uint16_t *arr_simd = (uint16_t *)gb_simd.data;

            /* Use values >= 1 to avoid underflow to avoid wrapping issues with pattern */
            generate_input_16(arr_ref, w, h, w, (InputCategory)cat, 1023);
            /* Ensure minimum value of 1 so decrement doesn't underflow to 65535 */
            for (unsigned i = 0; i < arr_len; i++) {
                if (arr_ref[i] == 0) arr_ref[i] = 1;
            }
            memcpy(arr_simd, arr_ref, buf_size);

            int left = 0;
            int right = (int)arr_len;

            ref_decrement_range(arr_ref, left, right);
            cambi_decrement_range_avx2(arr_simd, left, right);

            mu_assert(diag_msg,
                integers_match(arr_ref, arr_simd, buf_size,
                    "decrement_range_avx2", input_category_names[cat], w, h, 2));

            mu_assert("decrement_range: SIMD guard before corrupted",
                guard_before_intact(&gb_simd));
            mu_assert("decrement_range: SIMD guard after corrupted",
                guard_after_intact(&gb_simd));

            guarded_buf_free(&gb_ref);
            guarded_buf_free(&gb_simd);
        }
    }
    return NULL;
}
#endif

/* ========== Derivative Data Tests ========== */

#if ARCH_X86
static char *test_derivative_avx2(void) {
    for (unsigned d = 0; d < NUM_STANDARD_DIMS; d++) {
        unsigned w = standard_dims[d].w;
        unsigned h = standard_dims[d].h;

        for (int cat = 0; cat < INPUT_CATEGORY_COUNT; cat++) {
            /* Allocate image data */
            ptrdiff_t stride = (ptrdiff_t)ALIGN_CEIL(w * sizeof(uint16_t)) / sizeof(uint16_t);
            size_t img_size = stride * h * sizeof(uint16_t);
            void *img_data = aligned_malloc(img_size, MAX_ALIGN);
            mu_assert("img alloc failed", img_data != NULL);

            uint16_t *image = (uint16_t *)img_data;
            generate_input_16(image, w, h, stride, (InputCategory)cat, 1023);

            /* Allocate derivative buffers with guards */
            size_t deriv_size = w * sizeof(uint16_t);
            GuardedBuffer gb_ref, gb_simd;

            /* Test each row */
            for (unsigned row = 0; row < h; row++) {
                mu_assert("alloc deriv ref failed",
                    guarded_buf_alloc(&gb_ref, deriv_size) == 0);
                mu_assert("alloc deriv simd failed",
                    guarded_buf_alloc(&gb_simd, deriv_size) == 0);

                memset(gb_ref.data, 0, deriv_size);
                memset(gb_simd.data, 0, deriv_size);

                ref_get_derivative_data_for_row(image,
                    (uint16_t *)gb_ref.data, (int)w, (int)h, (int)row, (int)stride);
                get_derivative_data_for_row_avx2(image,
                    (uint16_t *)gb_simd.data, (int)w, (int)h, (int)row, (int)stride);

                mu_assert(diag_msg,
                    integers_match(gb_ref.data, gb_simd.data, deriv_size,
                        "get_derivative_data_for_row_avx2",
                        input_category_names[cat], w, h, 2));

                mu_assert("derivative: SIMD guard before corrupted",
                    guard_before_intact(&gb_simd));
                mu_assert("derivative: SIMD guard after corrupted",
                    guard_after_intact(&gb_simd));

                guarded_buf_free(&gb_ref);
                guarded_buf_free(&gb_simd);
            }

            aligned_free(img_data);
        }
    }
    return NULL;
}
#endif

/* ========== Test Runner ========== */

char *run_tests(void) {
#if ARCH_X86
    mu_run_test(test_increment_range_avx2);
    mu_run_test(test_decrement_range_avx2);
    mu_run_test(test_derivative_avx2);
#endif
    return NULL;
}
