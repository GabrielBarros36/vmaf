/**
 *
 *  Copyright 2016-2020 Netflix, Inc.
 *
 *     Licensed under the BSD+Patent License (the "License");
 *     you may not use this file except in compliance with the License.
 *     You may obtain a copy of the License at
 *
 *         https://opensource.org/licenses/BSDplusPatent
 *
 *     Unless required by applicable law or agreed to in writing, software
 *     distributed under the License is distributed on an "AS IS" BASIS,
 *     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *     See the License for the specific language governing permissions and
 *     limitations under the License.
 *
 */

#include <immintrin.h>
#include <stdint.h>
#include <stdbool.h>

void cambi_increment_range_avx2(uint16_t *arr, int left, int right) {
    __m256i val_vector = _mm256_set1_epi16(1);
    int col = left;
    for (; col + 15 < right; col += 16) {
        __m256i data = _mm256_loadu_si256((__m256i*) &arr[col]);
        data = _mm256_add_epi16(data, val_vector);
        _mm256_storeu_si256((__m256i*) &arr[col], data);
    }
    for (; col < right; col++) {
        arr[col]++;
    } 
}

void cambi_decrement_range_avx2(uint16_t *arr, int left, int right) {
    __m256i val_vector = _mm256_set1_epi16(1);
    int col = left;
    for (; col + 15 < right; col += 16) {
        __m256i data = _mm256_loadu_si256((__m256i*) &arr[col]);
        data = _mm256_sub_epi16(data, val_vector);
        _mm256_storeu_si256((__m256i*) &arr[col], data);
    }
    for (; col < right; col++) {
        arr[col]--;
    } 
}

/*
 * AVX2 mode3: for each position, if any two of {a,b,c} match, return that
 * value; otherwise return min(a,b,c).  Processes 16 uint16 values at once.
 */
static inline __m256i mode3_avx2(__m256i a, __m256i b, __m256i c) {
    __m256i eq_ab = _mm256_cmpeq_epi16(a, b);
    __m256i eq_ac = _mm256_cmpeq_epi16(a, c);
    __m256i eq_bc = _mm256_cmpeq_epi16(b, c);

    /* Default: min of all three */
    __m256i result = _mm256_min_epu16(_mm256_min_epu16(a, b), c);

    /* Where b==c, result is b */
    result = _mm256_blendv_epi8(result, b, eq_bc);
    /* Where a==b or a==c, result is a (higher priority) */
    __m256i a_match = _mm256_or_si256(eq_ab, eq_ac);
    result = _mm256_blendv_epi8(result, a, a_match);

    return result;
}

void cambi_filter_mode_avx2(uint16_t *data, ptrdiff_t stride, int width, int height, uint16_t *buffer) {
    int curr_line = 0;
    for (int i = 0; i < height; i++) {
        uint16_t *buf_row = buffer + curr_line * width;
        uint16_t *src_row = data + i * stride;

        /* First element: copy */
        buf_row[0] = src_row[0];

        /* Horizontal mode3: mode3(data[j-1], data[j], data[j+1]) */
        int j = 1;
        for (; j + 15 < width - 1; j += 16) {
            __m256i a = _mm256_loadu_si256((const __m256i *)&src_row[j - 1]);
            __m256i b = _mm256_loadu_si256((const __m256i *)&src_row[j]);
            __m256i c = _mm256_loadu_si256((const __m256i *)&src_row[j + 1]);
            __m256i res = mode3_avx2(a, b, c);
            _mm256_storeu_si256((__m256i *)&buf_row[j], res);
        }
        for (; j < width - 1; j++) {
            uint16_t va = src_row[j - 1], vb = src_row[j], vc = src_row[j + 1];
            if (va == vb || va == vc) buf_row[j] = va;
            else if (vb == vc) buf_row[j] = vb;
            else buf_row[j] = va <= vb ? (va <= vc ? va : vc) : (vb <= vc ? vb : vc);
        }

        /* Last element: copy */
        buf_row[width - 1] = src_row[width - 1];

        /* Vertical mode3 on 3-line circular buffer */
        if (i > 1) {
            uint16_t *buf0 = buffer + 0 * width;
            uint16_t *buf1 = buffer + 1 * width;
            uint16_t *buf2 = buffer + 2 * width;
            uint16_t *dst = data + (i - 1) * stride;

            int k = 0;
            for (; k + 15 < width; k += 16) {
                __m256i a = _mm256_loadu_si256((const __m256i *)&buf0[k]);
                __m256i b = _mm256_loadu_si256((const __m256i *)&buf1[k]);
                __m256i c = _mm256_loadu_si256((const __m256i *)&buf2[k]);
                __m256i res = mode3_avx2(a, b, c);
                _mm256_storeu_si256((__m256i *)&dst[k], res);
            }
            for (; k < width; k++) {
                uint16_t va = buf0[k], vb = buf1[k], vc = buf2[k];
                if (va == vb || va == vc) dst[k] = va;
                else if (vb == vc) dst[k] = vb;
                else dst[k] = va <= vb ? (va <= vc ? va : vc) : (vb <= vc ? vb : vc);
            }
        }
        curr_line = (curr_line + 1 == 3 ? 0 : curr_line + 1);
    }
}

void cambi_decimate_shift_avx2(const uint16_t *src, uint16_t *dst, ptrdiff_t src_stride,
                                ptrdiff_t dst_stride, unsigned width, unsigned height,
                                int shift_factor, int rounding_offset) {
    for (unsigned i = 0; i < height; i++) {
        const uint16_t *src_row = src + i * src_stride;
        uint16_t *dst_row = dst + i * dst_stride;
        __m256i v_round = _mm256_set1_epi16((int16_t)rounding_offset);
        unsigned j = 0;
        for (; j + 15 < width; j += 16) {
            __m256i vals = _mm256_loadu_si256((const __m256i *)&src_row[j]);
            vals = _mm256_add_epi16(vals, v_round);
            vals = _mm256_srli_epi16(vals, shift_factor);
            _mm256_storeu_si256((__m256i *)&dst_row[j], vals);
        }
        for (; j < width; j++) {
            dst_row[j] = (src_row[j] + rounding_offset) >> shift_factor;
        }
    }
}

int cambi_mask_block_zero_avx2(const uint16_t *mask_block) {
    __m256i m = _mm256_loadu_si256((const __m256i *)mask_block);
    return _mm256_testz_si256(m, m);
}

void get_derivative_data_for_row_avx2(const uint16_t *image_data, uint16_t *derivative_buffer, int width, int height, int row, int stride) {
    // For the last row, we only compute horizontal derivatives
    if (row == height - 1) {
        __m256i ones = _mm256_set1_epi16(1);
        int col = 0;
        for (; col + 15 < width - 1; col += 16) {
            __m256i vals1 = _mm256_loadu_si256((__m256i*) &image_data[row * stride + col]);
            __m256i vals2 = _mm256_loadu_si256((__m256i*) &image_data[row * stride + col + 1]);
            __m256i result = _mm256_cmpeq_epi16(vals1, vals2);
            _mm256_storeu_si256((__m256i*) &derivative_buffer[col], _mm256_and_si256(ones, result));
        }
        for (; col < width - 1; col++) {
            derivative_buffer[col] = (image_data[row * stride + col] == image_data[row * stride + col + 1]);
        }
        derivative_buffer[width - 1] = 1;
    }
    else {
        __m256i ones = _mm256_set1_epi16(1);
        int col = 0;
        for (; col + 15 < width - 1; col += 16) {
            __m256i horiz_vals1 = _mm256_loadu_si256((__m256i*) &image_data[row * stride + col]);
            __m256i horiz_vals2 = _mm256_loadu_si256((__m256i*) &image_data[row * stride + col + 1]);
            __m256i horiz_result = _mm256_and_si256(ones, _mm256_cmpeq_epi16(horiz_vals1, horiz_vals2));
            __m256i vert_vals1 = _mm256_loadu_si256((__m256i*) &image_data[row * stride + col]);
            __m256i vert_vals2 = _mm256_loadu_si256((__m256i*) &image_data[(row + 1) * stride + col]);
            __m256i vert_result = _mm256_and_si256(ones, _mm256_cmpeq_epi16(vert_vals1, vert_vals2));
            _mm256_storeu_si256((__m256i*) &derivative_buffer[col], _mm256_and_si256(horiz_result, vert_result));
        }
        for (; col < width; col++) {
            bool horizontal_derivative = (col == width - 1 || image_data[row * stride + col] == image_data[row * stride + col + 1]);
            bool vertical_derivative = image_data[row * stride + col] == image_data[(row + 1) * stride + col];
            derivative_buffer[col] =  horizontal_derivative && vertical_derivative;
        }
    }
}