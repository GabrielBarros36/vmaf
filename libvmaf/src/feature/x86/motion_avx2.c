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
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>

#include "feature/integer_motion.h"
#include "feature/common/alignment.h"

void x_convolution_16_avx2(const uint16_t *src, uint16_t *dst, unsigned width,
                           unsigned height, ptrdiff_t src_stride,
                           ptrdiff_t dst_stride)
{
    const unsigned radius = filter_width / 2;
    const unsigned left_edge = vmaf_ceiln(radius, 1);
    const unsigned right_edge = vmaf_floorn(width - (filter_width - radius), 1);
    const unsigned shift_add_round = 32768;
    const unsigned vector_loop = width < 16 ? 0 : (width >> 4) - 1;

    uint16_t *src_p = (uint16_t*) src + (left_edge - radius);
    unsigned nr = left_edge + 16 * vector_loop;
    uint16_t *src_pt = (uint16_t*) src + nr -radius;
    for (unsigned i = 0; i < height; ++i) {
        for (unsigned j = 0; j < left_edge; j++) {
            dst[i * dst_stride + j] =
                (edge_16(true, src, width, height, src_stride, i, j) +
                 shift_add_round) >> 16;
        }
    }

    /* Hoist invariant constants outside the loop */
    __m256i kernel1 = _mm256_set1_epi16(3571);
    __m256i kernel2 = _mm256_set1_epi16(16004);
    __m256i kernel3 = _mm256_set1_epi16(26386);
    __m256i addnum = _mm256_set1_epi32(32768);

    for (unsigned i = 0; i < height; ++i) {
        uint16_t *src_p1 = src_p;
        for (unsigned j = 0; j < vector_loop; j = j + 1) {
            __m256i src1 = _mm256_loadu_si256((__m256i*) src_p1);
            __m256i result = _mm256_mulhi_epu16(src1, kernel1);
            __m256i resultlo = _mm256_mullo_epi16(src1, kernel1);

            //src1 = src1 >> 16; //shift by a  pixel
            __m256i src2 = _mm256_loadu_si256((__m256i*) (src_p1 + 1));
            __m256i result2 = _mm256_mulhi_epu16(src2, kernel2);
            __m256i result2lo = _mm256_mullo_epi16(src2, kernel2);
            __m256i accum1_lo = _mm256_unpacklo_epi16(resultlo, result);
            __m256i accum1_hi = _mm256_unpackhi_epi16(resultlo, result);
            __m256i accum2_lo = _mm256_unpacklo_epi16(result2lo, result2);
            __m256i accum2_hi = _mm256_unpackhi_epi16(result2lo, result2);

            //Filter[3] value
            // src1= src1>>32;
            __m256i src3 = _mm256_loadu_si256((__m256i*) (src_p1 + 2));
            __m256i result3 = _mm256_mulhi_epu16(src3, kernel3);
            __m256i result3lo = _mm256_mullo_epi16(src3, kernel3);
            __m256i accum3_lo = _mm256_unpacklo_epi16(result3lo, result3);
            __m256i accum3_hi = _mm256_unpackhi_epi16(result3lo, result3);

            //filter 4
            src1 = _mm256_loadu_si256((__m256i*) (src_p1 + 3));
            result = _mm256_mulhi_epu16(src1, kernel2);
            resultlo = _mm256_mullo_epi16(src1, kernel2);

            //Filter 5
            src2 = _mm256_loadu_si256((__m256i*) (src_p1 + 4));
            result2 = _mm256_mulhi_epu16(src2, kernel1);
            result2lo = _mm256_mullo_epi16(src2, kernel1);

            __m256i accum4_lo =_mm256_unpacklo_epi16(resultlo, result);
            __m256i accum4_hi =_mm256_unpackhi_epi16(resultlo, result);
            __m256i accum5_lo =_mm256_unpacklo_epi16(result2lo, result2);
            __m256i accum5_hi =_mm256_unpackhi_epi16(result2lo, result2);

            __m256i accum_lo = _mm256_add_epi32(accum1_lo, accum2_lo);
            __m256i accumi_lo = _mm256_add_epi32(accum3_lo, accum4_lo);
            accum5_lo = _mm256_add_epi32(accum5_lo, addnum);
            accum_lo = _mm256_add_epi32(accum5_lo, accum_lo);
            accum_lo = _mm256_add_epi32(accumi_lo, accum_lo);
            __m256i accum_hi = _mm256_add_epi32(accum1_hi, accum2_hi);
            __m256i accumi_hi = _mm256_add_epi32(accum3_hi, accum4_hi);
            accum_hi = _mm256_add_epi32(accum5_hi, accum_hi);
            accumi_hi = _mm256_add_epi32(accumi_hi, addnum);
            accum_hi = _mm256_add_epi32(accumi_hi, accum_hi);
            accum_lo = _mm256_srli_epi32(accum_lo, 0x10);
            accum_hi = _mm256_srli_epi32(accum_hi, 0x10);

            result = _mm256_packus_epi32(accum_lo, accum_hi);
            _mm256_storeu_si256(
                (__m256i*) (dst + i * dst_stride + j * 16 + left_edge), result);

            src_p1 += 16;
        }
        src_p += src_stride;
    }

    for (unsigned i = 0; i < height; ++i) {
        uint16_t *src_p1 = src_pt;
        for (unsigned j = nr; j < (right_edge); j++) {
            uint32_t accum = 0;
            uint16_t *src_p2 = src_p1;
            for (int k = 0; k < filter_width; ++k) {
                accum += filter[k] * (*src_p2);
                src_p2++;
            }
            src_p1++;
            dst[i * dst_stride + j] = (accum + shift_add_round) >> 16;
        }
        src_pt += src_stride;
    }

    for (unsigned i = 0; i < height; ++i) {
        for (unsigned j = right_edge; j < width; j++) {
            dst[i * dst_stride + j] =
                (edge_16(true, src, width, height, src_stride, i, j) +
                 shift_add_round) >> 16;
        }
    }
}

void y_convolution_16_avx2(void *src, uint16_t *dst, unsigned width,
                           unsigned height, ptrdiff_t src_stride,
                           ptrdiff_t dst_stride, unsigned inp_size_bits)
{
    const unsigned radius = filter_width / 2;
    const unsigned top_edge = vmaf_ceiln(radius, 1);
    const unsigned bottom_edge = vmaf_floorn(height - (filter_width - radius), 1);
    const unsigned add_before_shift = (int) pow(2, (inp_size_bits - 1));
    const unsigned shift_var = inp_size_bits;

    /* Handle top edge rows with scalar code */
    for (unsigned i = 0; i < top_edge; i++) {
        for (unsigned j = 0; j < width; ++j) {
            dst[i * dst_stride + j] =
                (edge_16(false, src, width, height, src_stride, i, j) +
                 add_before_shift) >> shift_var;
        }
    }

    /* AVX2 kernel constants */
    __m256i k0 = _mm256_set1_epi16(3571);
    __m256i k1 = _mm256_set1_epi16(16004);
    __m256i k2 = _mm256_set1_epi16(26386);
    /* k3 == k1, k4 == k0 (symmetric filter) */
    __m256i addnum = _mm256_set1_epi32(add_before_shift);

    const unsigned vec_width = 16; /* 16 uint16_t per __m256i */
    const unsigned vector_count = width / vec_width;
    const unsigned scalar_start = vector_count * vec_width;

    uint16_t *src_p = (uint16_t*) src + (top_edge - radius) * src_stride;
    for (unsigned i = top_edge; i < bottom_edge; i++) {
        /* Pointers to 5 vertically-adjacent rows */
        uint16_t *row0 = src_p;
        uint16_t *row1 = src_p + src_stride;
        uint16_t *row2 = src_p + 2 * src_stride;
        uint16_t *row3 = src_p + 3 * src_stride;
        uint16_t *row4 = src_p + 4 * src_stride;

        for (unsigned j = 0; j < vector_count; j++) {
            unsigned off = j * vec_width;

            /* Load 16 uint16_t from each of the 5 rows */
            __m256i r0 = _mm256_loadu_si256((__m256i*)(row0 + off));
            __m256i r1 = _mm256_loadu_si256((__m256i*)(row1 + off));
            __m256i r2 = _mm256_loadu_si256((__m256i*)(row2 + off));
            __m256i r3 = _mm256_loadu_si256((__m256i*)(row3 + off));
            __m256i r4 = _mm256_loadu_si256((__m256i*)(row4 + off));

            /* Multiply row0 by filter[0]=3571, get 32-bit results */
            __m256i hi0 = _mm256_mulhi_epu16(r0, k0);
            __m256i lo0 = _mm256_mullo_epi16(r0, k0);
            __m256i prod0_lo = _mm256_unpacklo_epi16(lo0, hi0);
            __m256i prod0_hi = _mm256_unpackhi_epi16(lo0, hi0);

            /* Multiply row1 by filter[1]=16004 */
            __m256i hi1 = _mm256_mulhi_epu16(r1, k1);
            __m256i lo1 = _mm256_mullo_epi16(r1, k1);
            __m256i prod1_lo = _mm256_unpacklo_epi16(lo1, hi1);
            __m256i prod1_hi = _mm256_unpackhi_epi16(lo1, hi1);

            /* Multiply row2 by filter[2]=26386 */
            __m256i hi2 = _mm256_mulhi_epu16(r2, k2);
            __m256i lo2 = _mm256_mullo_epi16(r2, k2);
            __m256i prod2_lo = _mm256_unpacklo_epi16(lo2, hi2);
            __m256i prod2_hi = _mm256_unpackhi_epi16(lo2, hi2);

            /* Multiply row3 by filter[3]=16004 (same as k1) */
            __m256i hi3 = _mm256_mulhi_epu16(r3, k1);
            __m256i lo3 = _mm256_mullo_epi16(r3, k1);
            __m256i prod3_lo = _mm256_unpacklo_epi16(lo3, hi3);
            __m256i prod3_hi = _mm256_unpackhi_epi16(lo3, hi3);

            /* Multiply row4 by filter[4]=3571 (same as k0) */
            __m256i hi4 = _mm256_mulhi_epu16(r4, k0);
            __m256i lo4 = _mm256_mullo_epi16(r4, k0);
            __m256i prod4_lo = _mm256_unpacklo_epi16(lo4, hi4);
            __m256i prod4_hi = _mm256_unpackhi_epi16(lo4, hi4);

            /* Accumulate all 5 products + rounding constant */
            __m256i acc_lo = _mm256_add_epi32(prod0_lo, prod1_lo);
            __m256i tmp_lo = _mm256_add_epi32(prod2_lo, prod3_lo);
            __m256i fin_lo = _mm256_add_epi32(prod4_lo, addnum);
            acc_lo = _mm256_add_epi32(acc_lo, tmp_lo);
            acc_lo = _mm256_add_epi32(acc_lo, fin_lo);

            __m256i acc_hi = _mm256_add_epi32(prod0_hi, prod1_hi);
            __m256i tmp_hi = _mm256_add_epi32(prod2_hi, prod3_hi);
            __m256i fin_hi = _mm256_add_epi32(prod4_hi, addnum);
            acc_hi = _mm256_add_epi32(acc_hi, tmp_hi);
            acc_hi = _mm256_add_epi32(acc_hi, fin_hi);

            /* Shift right by shift_var */
            acc_lo = _mm256_srli_epi32(acc_lo, shift_var);
            acc_hi = _mm256_srli_epi32(acc_hi, shift_var);

            /* Pack 32-bit back to 16-bit with unsigned saturation */
            __m256i result = _mm256_packus_epi32(acc_lo, acc_hi);
            _mm256_storeu_si256(
                (__m256i*)(dst + i * dst_stride + off), result);
        }

        /* Scalar tail for remaining elements */
        uint16_t *src_p1 = src_p + scalar_start;
        for (unsigned j = scalar_start; j < width; ++j) {
            uint16_t *src_p2 = src_p1;
            uint32_t accum = 0;
            for (int kk = 0; kk < filter_width; ++kk) {
                accum += filter[kk] * (*src_p2);
                src_p2 += src_stride;
            }
            dst[i * dst_stride + j] = (accum + add_before_shift) >> shift_var;
            src_p1++;
        }

        src_p += src_stride;
    }

    /* Handle bottom edge rows with scalar code */
    for (unsigned i = bottom_edge; i < height; i++) {
        for (unsigned j = 0; j < width; ++j) {
            dst[i * dst_stride + j] =
                (edge_16(false, src, width, height, src_stride, i, j) +
                 add_before_shift) >> shift_var;
        }
    }
}

static inline uint32_t
edge_8_local(const uint8_t *src, int height, int stride, int i, int j)
{
    int radius = filter_width / 2;
    uint32_t accum = 0;

    for (int k = 0; k < filter_width; ++k) {
        int i_tap = i - radius + k;
        int j_tap = j;

        if (i_tap < 0)
            i_tap = -i_tap;
        else if (i_tap >= height)
            i_tap = height - (i_tap - height + 1);

        accum += filter[k] * src[i_tap * stride + j_tap];
    }
    return accum;
}

void y_convolution_8_avx2(void *src, uint16_t *dst, unsigned width,
                          unsigned height, ptrdiff_t src_stride,
                          ptrdiff_t dst_stride, unsigned inp_size_bits)
{
    (void) inp_size_bits;
    const unsigned radius = filter_width / 2;
    const unsigned top_edge = vmaf_ceiln(radius, 1);
    const unsigned bottom_edge = vmaf_floorn(height - (filter_width - radius), 1);
    const unsigned shift_var = 8;
    const unsigned add_before_shift = (int) pow(2, (shift_var - 1));

    /* Handle top edge rows with scalar code */
    for (unsigned i = 0; i < top_edge; i++) {
        for (unsigned j = 0; j < width; ++j) {
            dst[i * dst_stride + j] =
                (edge_8_local(src, height, src_stride, i, j) +
                 add_before_shift) >> shift_var;
        }
    }

    /* AVX2 kernel constants (16-bit for multiply with widened 8-bit data) */
    __m256i k0 = _mm256_set1_epi16(3571);
    __m256i k1 = _mm256_set1_epi16(16004);
    __m256i k2 = _mm256_set1_epi16(26386);
    __m256i addnum = _mm256_set1_epi32(add_before_shift);

    const unsigned vec_width = 16; /* process 16 uint8_t -> 16 uint16_t per iteration */
    const unsigned vector_count = width / vec_width;
    const unsigned scalar_start = vector_count * vec_width;

    uint8_t *src_p = (uint8_t*) src + (top_edge - radius) * src_stride;
    for (unsigned i = top_edge; i < bottom_edge; i++) {
        /* Pointers to 5 vertically-adjacent rows */
        uint8_t *row0 = src_p;
        uint8_t *row1 = src_p + src_stride;
        uint8_t *row2 = src_p + 2 * src_stride;
        uint8_t *row3 = src_p + 3 * src_stride;
        uint8_t *row4 = src_p + 4 * src_stride;

        for (unsigned j = 0; j < vector_count; j++) {
            unsigned off = j * vec_width;

            /* Load 16 uint8_t from each row and zero-extend to 16-bit */
            __m128i r0_8 = _mm_loadu_si128((__m128i*)(row0 + off));
            __m256i r0 = _mm256_cvtepu8_epi16(r0_8);

            __m128i r1_8 = _mm_loadu_si128((__m128i*)(row1 + off));
            __m256i r1 = _mm256_cvtepu8_epi16(r1_8);

            __m128i r2_8 = _mm_loadu_si128((__m128i*)(row2 + off));
            __m256i r2 = _mm256_cvtepu8_epi16(r2_8);

            __m128i r3_8 = _mm_loadu_si128((__m128i*)(row3 + off));
            __m256i r3 = _mm256_cvtepu8_epi16(r3_8);

            __m128i r4_8 = _mm_loadu_si128((__m128i*)(row4 + off));
            __m256i r4 = _mm256_cvtepu8_epi16(r4_8);

            /* Multiply row0 by filter[0]=3571 */
            __m256i hi0 = _mm256_mulhi_epu16(r0, k0);
            __m256i lo0 = _mm256_mullo_epi16(r0, k0);
            __m256i prod0_lo = _mm256_unpacklo_epi16(lo0, hi0);
            __m256i prod0_hi = _mm256_unpackhi_epi16(lo0, hi0);

            /* Multiply row1 by filter[1]=16004 */
            __m256i hi1 = _mm256_mulhi_epu16(r1, k1);
            __m256i lo1 = _mm256_mullo_epi16(r1, k1);
            __m256i prod1_lo = _mm256_unpacklo_epi16(lo1, hi1);
            __m256i prod1_hi = _mm256_unpackhi_epi16(lo1, hi1);

            /* Multiply row2 by filter[2]=26386 */
            __m256i hi2 = _mm256_mulhi_epu16(r2, k2);
            __m256i lo2 = _mm256_mullo_epi16(r2, k2);
            __m256i prod2_lo = _mm256_unpacklo_epi16(lo2, hi2);
            __m256i prod2_hi = _mm256_unpackhi_epi16(lo2, hi2);

            /* Multiply row3 by filter[3]=16004 */
            __m256i hi3 = _mm256_mulhi_epu16(r3, k1);
            __m256i lo3 = _mm256_mullo_epi16(r3, k1);
            __m256i prod3_lo = _mm256_unpacklo_epi16(lo3, hi3);
            __m256i prod3_hi = _mm256_unpackhi_epi16(lo3, hi3);

            /* Multiply row4 by filter[4]=3571 */
            __m256i hi4 = _mm256_mulhi_epu16(r4, k0);
            __m256i lo4 = _mm256_mullo_epi16(r4, k0);
            __m256i prod4_lo = _mm256_unpacklo_epi16(lo4, hi4);
            __m256i prod4_hi = _mm256_unpackhi_epi16(lo4, hi4);

            /* Accumulate all 5 products + rounding constant */
            __m256i acc_lo = _mm256_add_epi32(prod0_lo, prod1_lo);
            __m256i tmp_lo = _mm256_add_epi32(prod2_lo, prod3_lo);
            __m256i fin_lo = _mm256_add_epi32(prod4_lo, addnum);
            acc_lo = _mm256_add_epi32(acc_lo, tmp_lo);
            acc_lo = _mm256_add_epi32(acc_lo, fin_lo);

            __m256i acc_hi = _mm256_add_epi32(prod0_hi, prod1_hi);
            __m256i tmp_hi = _mm256_add_epi32(prod2_hi, prod3_hi);
            __m256i fin_hi = _mm256_add_epi32(prod4_hi, addnum);
            acc_hi = _mm256_add_epi32(acc_hi, tmp_hi);
            acc_hi = _mm256_add_epi32(acc_hi, fin_hi);

            /* Shift right by shift_var (8) */
            acc_lo = _mm256_srli_epi32(acc_lo, shift_var);
            acc_hi = _mm256_srli_epi32(acc_hi, shift_var);

            /* Pack 32-bit back to 16-bit with unsigned saturation */
            __m256i result = _mm256_packus_epi32(acc_lo, acc_hi);
            _mm256_storeu_si256(
                (__m256i*)(dst + i * dst_stride + off), result);
        }

        /* Scalar tail for remaining elements */
        uint8_t *src_p1 = src_p + scalar_start;
        for (unsigned j = scalar_start; j < width; ++j) {
            uint8_t *src_p2 = src_p1;
            uint32_t accum = 0;
            for (int kk = 0; kk < filter_width; ++kk) {
                accum += filter[kk] * (*src_p2);
                src_p2 += src_stride;
            }
            dst[i * dst_stride + j] = (accum + add_before_shift) >> shift_var;
            src_p1++;
        }

        src_p += src_stride;
    }

    /* Handle bottom edge rows with scalar code */
    for (unsigned i = bottom_edge; i < height; i++) {
        for (unsigned j = 0; j < width; ++j) {
            dst[i * dst_stride + j] =
                (edge_8_local(src, height, src_stride, i, j) +
                 add_before_shift) >> shift_var;
        }
    }
}

void sad_avx2(const uint16_t *a, const uint16_t *b,
              unsigned w, unsigned h,
              ptrdiff_t stride_a, ptrdiff_t stride_b,
              uint64_t *sad)
{
    *sad = 0;

    const __m256i zero = _mm256_setzero_si256();
    const unsigned vec_width = 16; /* 16 uint16_t per __m256i */
    const unsigned vector_count = w / vec_width;
    const unsigned scalar_start = vector_count * vec_width;

    for (unsigned i = 0; i < h; i++) {
        __m256i row_acc = _mm256_setzero_si256();

        for (unsigned j = 0; j < vector_count; j++) {
            unsigned off = j * vec_width;
            __m256i va = _mm256_loadu_si256((__m256i*)(a + off));
            __m256i vb = _mm256_loadu_si256((__m256i*)(b + off));

            /* Unsigned absolute difference via saturating subtraction */
            __m256i diff_ab = _mm256_subs_epu16(va, vb);
            __m256i diff_ba = _mm256_subs_epu16(vb, va);
            __m256i absdiff = _mm256_or_si256(diff_ab, diff_ba);

            /* Widen unsigned 16-bit abs diffs to 32-bit by interleaving
             * with zero, then accumulate. Cannot use _mm256_madd_epi16
             * because abs diffs can exceed 32767. */
            __m256i lo32 = _mm256_unpacklo_epi16(absdiff, zero);
            __m256i hi32 = _mm256_unpackhi_epi16(absdiff, zero);
            row_acc = _mm256_add_epi32(row_acc, lo32);
            row_acc = _mm256_add_epi32(row_acc, hi32);
        }

        /* Horizontal reduce the 8x32-bit accumulator to scalar */
        __m128i lo128 = _mm256_castsi256_si128(row_acc);
        __m128i hi128 = _mm256_extracti128_si256(row_acc, 1);
        __m128i sum128 = _mm_add_epi32(lo128, hi128);
        __m128i sum64 = _mm_add_epi32(sum128, _mm_srli_si128(sum128, 8));
        __m128i sum32 = _mm_add_epi32(sum64, _mm_srli_si128(sum64, 4));
        uint32_t inner_sad = (uint32_t)_mm_cvtsi128_si32(sum32);

        /* Scalar tail */
        for (unsigned j = scalar_start; j < w; j++) {
            inner_sad += abs(a[j] - b[j]);
        }

        *sad += inner_sad;
        a += stride_a;
        b += stride_b;
    }
}
