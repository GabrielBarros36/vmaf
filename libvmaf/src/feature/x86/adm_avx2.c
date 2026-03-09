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

#include "feature/integer_adm.h"
#include "feature/common/macros.h"

#include <immintrin.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327
#endif

#ifndef ADM_BORDER_FACTOR
#define ADM_BORDER_FACTOR (0.1)
#endif

#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#define MAX(x, y) (((x) > (y)) ? (x) : (y))

/*
 * Signed arithmetic right shift by 15 for 128-bit vector of two int64 elements.
 * AVX2 does not have _mm_srai_epi64, so we implement it using:
 *   1. Logical right shift by 15
 *   2. Compute sign extension mask (all 1s if negative, all 0s if positive)
 *   3. Left shift sign mask by (64-15)=49 to fill top 15 bits
 *   4. OR with logical shift result
 */
static inline __m128i srai_epi64_15(__m128i x) {
    __m128i logical = _mm_srli_epi64(x, 15);
    /* Get sign bit into all bits of each 32-bit word */
    __m128i sign32 = _mm_srai_epi32(x, 31);
    /* Broadcast the high 32-bit word of each 64-bit element to both 32-bit positions */
    __m128i sign64 = _mm_shuffle_epi32(sign32, 0xF5); /* 11 11 01 01: replicate words 1,3 */
    /* Shift left by 49 to create mask for top 15 bits */
    __m128i sign_fill = _mm_slli_epi64(sign64, 49);
    return _mm_or_si128(logical, sign_fill);
}

/*
 * Signed arithmetic right shift for 256-bit vector of four int64 elements.
 * AVX2 does not have _mm256_srai_epi64, so we implement it manually.
 */
static inline __m256i srai_epi64_256(__m256i x, int shift) {
    __m256i logical = _mm256_srli_epi64(x, shift);
    __m256i sign32 = _mm256_srai_epi32(x, 31);
    /* Replicate high 32-bit words: positions 1,3,5,7 -> 0,1,2,3,4,5,6,7 */
    __m256i sign64 = _mm256_shuffle_epi32(sign32, 0xF5);
    __m256i sign_fill = _mm256_slli_epi64(sign64, 64 - shift);
    return _mm256_or_si256(logical, sign_fill);
}

/*
 * Multiply 8 int32 elements by a single int16 filter coefficient, accumulating
 * into int64 pairs. Processes even and odd lanes separately using _mm256_mul_epi32.
 * Accumulates into two __m256i vectors holding 4 int64 each (even and odd lanes).
 */
static inline void mul_acc_i32_coeff(
    __m256i src, __m256i coeff32,
    __m256i *acc_even, __m256i *acc_odd)
{
    /* Even lanes: elements 0,2,4,6 */
    *acc_even = _mm256_add_epi64(*acc_even, _mm256_mul_epi32(src, coeff32));
    /* Odd lanes: elements 1,3,5,7 - shift right by 32 bits to move odd to even position */
    __m256i src_odd = _mm256_srli_epi64(src, 32);
    __m256i coeff_odd = _mm256_srli_epi64(coeff32, 32);
    *acc_odd = _mm256_add_epi64(*acc_odd, _mm256_mul_epi32(src_odd, coeff_odd));
}

/*
 * Round-shift int64 accumulators and merge even/odd lanes back into 8 int32 results.
 */
static inline __m256i merge_shift_i64_to_i32(
    __m256i acc_even, __m256i acc_odd,
    __m256i rounding, int shift)
{
    acc_even = _mm256_add_epi64(acc_even, rounding);
    acc_odd  = _mm256_add_epi64(acc_odd, rounding);
    acc_even = srai_epi64_256(acc_even, shift);
    acc_odd  = srai_epi64_256(acc_odd, shift);
    /* acc_even has results in bits [31:0] of each 64-bit lane (positions 0,2,4,6)
     * acc_odd  has results in bits [31:0] of each 64-bit lane (positions 1,3,5,7)
     * Shift odd results left by 32 bits and OR together */
    acc_odd = _mm256_slli_epi64(acc_odd, 32);
    return _mm256_or_si256(acc_even, acc_odd);
}

/*
 * Truncating merge without shift (shift=0) for scale 1 vertical pass.
 * Just takes the low 32 bits of each int64 accumulator and interleaves.
 */
static inline __m256i merge_noshift_i64_to_i32(
    __m256i acc_even, __m256i acc_odd)
{
    /* Mask to keep only low 32 bits of each 64-bit lane */
    __m256i mask32 = _mm256_set1_epi64x(0xFFFFFFFF);
    acc_even = _mm256_and_si256(acc_even, mask32);
    acc_odd  = _mm256_slli_epi64(acc_odd, 32);
    return _mm256_or_si256(acc_even, acc_odd);
}

void adm_dwt2_8_avx2(const uint8_t *src, const adm_dwt_band_t *dst,
                     AdmBuffer *buf, int w, int h, int src_stride,
                     int dst_stride)
{
    const int16_t *filter_lo = dwt2_db2_coeffs_lo;
    const int16_t *filter_hi = dwt2_db2_coeffs_hi;

    const int16_t shift_HP = 16;
    const int32_t add_shift_HP = 32768;
    int **ind_y = buf->ind_y;
    int **ind_x = buf->ind_x;

    int16_t *tmplo = (int16_t *)buf->tmp_ref;
    int16_t *tmphi = tmplo + w;
    int32_t accum;

    __m256i dwt2_db2_coeffs_lo_sum_const = _mm256_set1_epi32(5931776);
    __m256i fl0 =
        _mm256_broadcastd_epi32(_mm_loadu_si128((__m128i *)filter_lo));
    __m256i fl1 =
        _mm256_broadcastd_epi32(_mm_loadu_si128((__m128i *)(filter_lo + 2)));
    __m256i fh0 =
        _mm256_broadcastd_epi32(_mm_loadu_si128((__m128i *)filter_hi));
    __m256i fh1 =
        _mm256_broadcastd_epi32(_mm_loadu_si128((__m128i *)(filter_hi + 2)));
    __m256i add_shift_VP_vex = _mm256_set1_epi32(128);
    __m256i pad_register = _mm256_setzero_si256();
    __m256i add_shift_HP_vex = _mm256_set1_epi32(32768);

    for (int i = 0; i < (h + 1) / 2; ++i) {
        /* Vertical pass - process 32 elements per iteration. */

        const uint8_t *src_row0 = src + (ind_y[0][i] * src_stride);
        const uint8_t *src_row1 = src + (ind_y[1][i] * src_stride);
        const uint8_t *src_row2 = src + (ind_y[2][i] * src_stride);
        const uint8_t *src_row3 = src + (ind_y[3][i] * src_stride);

        int j = 0;
        for (; j + 32 <= w; j += 32) {

            /* First group of 16 elements */
            __m256i accum_lo2_a, accum_hi2_a, accum_lo1_a, accum_hi1_a;
            accum_lo2_a = accum_hi2_a = accum_lo1_a = accum_hi1_a =
                _mm256_setzero_si256();

            __m256i s0a = _mm256_cvtepu8_epi16(
                _mm_loadu_si128((__m128i *)(src_row0 + j)));
            __m256i s1a = _mm256_cvtepu8_epi16(
                _mm_loadu_si128((__m128i *)(src_row1 + j)));
            __m256i s2a = _mm256_cvtepu8_epi16(
                _mm_loadu_si128((__m128i *)(src_row2 + j)));
            __m256i s3a = _mm256_cvtepu8_epi16(
                _mm_loadu_si128((__m128i *)(src_row3 + j)));

            /* Second group of 16 elements */
            __m256i accum_lo2_b, accum_hi2_b, accum_lo1_b, accum_hi1_b;
            accum_lo2_b = accum_hi2_b = accum_lo1_b = accum_hi1_b =
                _mm256_setzero_si256();

            __m256i s0b = _mm256_cvtepu8_epi16(
                _mm_loadu_si128((__m128i *)(src_row0 + j + 16)));
            __m256i s1b = _mm256_cvtepu8_epi16(
                _mm_loadu_si128((__m128i *)(src_row1 + j + 16)));
            __m256i s2b = _mm256_cvtepu8_epi16(
                _mm_loadu_si128((__m128i *)(src_row2 + j + 16)));
            __m256i s3b = _mm256_cvtepu8_epi16(
                _mm_loadu_si128((__m128i *)(src_row3 + j + 16)));

            /* Unpack and multiply - group A */
            __m256i s0lo_a = _mm256_unpacklo_epi16(s0a, s1a);
            __m256i s0hi_a = _mm256_unpackhi_epi16(s0a, s1a);
            accum_lo2_a = _mm256_madd_epi16(s0lo_a, fl0);
            accum_hi2_a = _mm256_madd_epi16(s0hi_a, fl0);

            /* Unpack and multiply - group B */
            __m256i s0lo_b = _mm256_unpacklo_epi16(s0b, s1b);
            __m256i s0hi_b = _mm256_unpackhi_epi16(s0b, s1b);
            accum_lo2_b = _mm256_madd_epi16(s0lo_b, fl0);
            accum_hi2_b = _mm256_madd_epi16(s0hi_b, fl0);

            __m256i s1lo_a = _mm256_unpacklo_epi16(s2a, s3a);
            __m256i s1hi_a = _mm256_unpackhi_epi16(s2a, s3a);
            accum_lo2_a =
                _mm256_add_epi32(accum_lo2_a, _mm256_madd_epi16(s1lo_a, fl1));
            accum_hi2_a =
                _mm256_add_epi32(accum_hi2_a, _mm256_madd_epi16(s1hi_a, fl1));

            __m256i s1lo_b = _mm256_unpacklo_epi16(s2b, s3b);
            __m256i s1hi_b = _mm256_unpackhi_epi16(s2b, s3b);
            accum_lo2_b =
                _mm256_add_epi32(accum_lo2_b, _mm256_madd_epi16(s1lo_b, fl1));
            accum_hi2_b =
                _mm256_add_epi32(accum_hi2_b, _mm256_madd_epi16(s1hi_b, fl1));

            /* Normalize lo filter - group A */
            accum_lo2_a =
                _mm256_sub_epi32(accum_lo2_a, dwt2_db2_coeffs_lo_sum_const);
            accum_hi2_a =
                _mm256_sub_epi32(accum_hi2_a, dwt2_db2_coeffs_lo_sum_const);

            accum_lo2_a = _mm256_add_epi32(accum_lo2_a, add_shift_VP_vex);
            accum_lo2_a = _mm256_srli_epi32(accum_lo2_a, 0x08);
            accum_hi2_a = _mm256_add_epi32(accum_hi2_a, add_shift_VP_vex);
            accum_hi2_a = _mm256_srli_epi32(accum_hi2_a, 0x08);
            accum_lo2_a = _mm256_blend_epi16(accum_lo2_a, pad_register, 0xAA);
            accum_hi2_a = _mm256_blend_epi16(accum_hi2_a, pad_register, 0xAA);

            accum_hi2_a = _mm256_packus_epi32(accum_lo2_a, accum_hi2_a);
            _mm256_storeu_si256((__m256i *)(tmplo + j), accum_hi2_a);

            /* Normalize lo filter - group B */
            accum_lo2_b =
                _mm256_sub_epi32(accum_lo2_b, dwt2_db2_coeffs_lo_sum_const);
            accum_hi2_b =
                _mm256_sub_epi32(accum_hi2_b, dwt2_db2_coeffs_lo_sum_const);

            accum_lo2_b = _mm256_add_epi32(accum_lo2_b, add_shift_VP_vex);
            accum_lo2_b = _mm256_srli_epi32(accum_lo2_b, 0x08);
            accum_hi2_b = _mm256_add_epi32(accum_hi2_b, add_shift_VP_vex);
            accum_hi2_b = _mm256_srli_epi32(accum_hi2_b, 0x08);
            accum_lo2_b = _mm256_blend_epi16(accum_lo2_b, pad_register, 0xAA);
            accum_hi2_b = _mm256_blend_epi16(accum_hi2_b, pad_register, 0xAA);

            accum_hi2_b = _mm256_packus_epi32(accum_lo2_b, accum_hi2_b);
            _mm256_storeu_si256((__m256i *)(tmplo + j + 16), accum_hi2_b);

            /* Hi filter - group A */
            accum_lo1_a = _mm256_madd_epi16(s0lo_a, fh0);
            accum_hi1_a = _mm256_madd_epi16(s0hi_a, fh0);
            accum_lo1_a =
                _mm256_add_epi32(accum_lo1_a, _mm256_madd_epi16(s1lo_a, fh1));
            accum_hi1_a =
                _mm256_add_epi32(accum_hi1_a, _mm256_madd_epi16(s1hi_a, fh1));

            accum_lo1_a = _mm256_add_epi32(accum_lo1_a, add_shift_VP_vex);
            accum_lo1_a = _mm256_srli_epi32(accum_lo1_a, 0x08);
            accum_hi1_a = _mm256_add_epi32(accum_hi1_a, add_shift_VP_vex);
            accum_hi1_a = _mm256_srli_epi32(accum_hi1_a, 0x08);
            accum_lo1_a = _mm256_blend_epi16(accum_lo1_a, pad_register, 0xAA);
            accum_hi1_a = _mm256_blend_epi16(accum_hi1_a, pad_register, 0xAA);
            accum_hi1_a = _mm256_packus_epi32(accum_lo1_a, accum_hi1_a);
            _mm256_storeu_si256((__m256i *)(tmphi + j), accum_hi1_a);

            /* Hi filter - group B */
            accum_lo1_b = _mm256_madd_epi16(s0lo_b, fh0);
            accum_hi1_b = _mm256_madd_epi16(s0hi_b, fh0);
            accum_lo1_b =
                _mm256_add_epi32(accum_lo1_b, _mm256_madd_epi16(s1lo_b, fh1));
            accum_hi1_b =
                _mm256_add_epi32(accum_hi1_b, _mm256_madd_epi16(s1hi_b, fh1));

            accum_lo1_b = _mm256_add_epi32(accum_lo1_b, add_shift_VP_vex);
            accum_lo1_b = _mm256_srli_epi32(accum_lo1_b, 0x08);
            accum_hi1_b = _mm256_add_epi32(accum_hi1_b, add_shift_VP_vex);
            accum_hi1_b = _mm256_srli_epi32(accum_hi1_b, 0x08);
            accum_lo1_b = _mm256_blend_epi16(accum_lo1_b, pad_register, 0xAA);
            accum_hi1_b = _mm256_blend_epi16(accum_hi1_b, pad_register, 0xAA);
            accum_hi1_b = _mm256_packus_epi32(accum_lo1_b, accum_hi1_b);
            _mm256_storeu_si256((__m256i *)(tmphi + j + 16), accum_hi1_b);
        }

        /* Tail: process remaining 16-element chunk if width not multiple of 32 */
        for (; j < w; j += 16) {

            __m256i accum_mu2_lo, accum_mu2_hi, accum_mu1_lo, accum_mu1_hi;
            accum_mu2_lo = accum_mu2_hi = accum_mu1_lo = accum_mu1_hi =
                _mm256_setzero_si256();
            __m256i s0, s1, s2, s3;

            s0 = _mm256_cvtepu8_epi16(_mm_loadu_si128(
                (__m128i *)(src_row0 + j)));
            s1 = _mm256_cvtepu8_epi16(_mm_loadu_si128(
                (__m128i *)(src_row1 + j)));
            s2 = _mm256_cvtepu8_epi16(_mm_loadu_si128(
                (__m128i *)(src_row2 + j)));
            s3 = _mm256_cvtepu8_epi16(_mm_loadu_si128(
                (__m128i *)(src_row3 + j)));

            __m256i s0lo = _mm256_unpacklo_epi16(s0, s1);
            __m256i s0hi = _mm256_unpackhi_epi16(s0, s1);
            accum_mu2_lo =
                _mm256_add_epi32(accum_mu2_lo, _mm256_madd_epi16(s0lo, fl0));
            accum_mu2_hi =
                _mm256_add_epi32(accum_mu2_hi, _mm256_madd_epi16(s0hi, fl0));

            __m256i s1lo = _mm256_unpacklo_epi16(s2, s3);
            __m256i s1hi = _mm256_unpackhi_epi16(s2, s3);
            accum_mu2_lo =
                _mm256_add_epi32(accum_mu2_lo, _mm256_madd_epi16(s1lo, fl1));
            accum_mu2_hi =
                _mm256_add_epi32(accum_mu2_hi, _mm256_madd_epi16(s1hi, fl1));

            accum_mu2_lo =
                _mm256_sub_epi32(accum_mu2_lo, dwt2_db2_coeffs_lo_sum_const);
            accum_mu2_hi =
                _mm256_sub_epi32(accum_mu2_hi, dwt2_db2_coeffs_lo_sum_const);

            accum_mu2_lo = _mm256_add_epi32(accum_mu2_lo, add_shift_VP_vex);
            accum_mu2_lo = _mm256_srli_epi32(accum_mu2_lo, 0x08);
            accum_mu2_hi = _mm256_add_epi32(accum_mu2_hi, add_shift_VP_vex);
            accum_mu2_hi = _mm256_srli_epi32(accum_mu2_hi, 0x08);
            accum_mu2_lo = _mm256_blend_epi16(accum_mu2_lo, pad_register, 0xAA);
            accum_mu2_hi = _mm256_blend_epi16(accum_mu2_hi, pad_register, 0xAA);

            accum_mu2_hi = _mm256_packus_epi32(accum_mu2_lo, accum_mu2_hi);
            _mm256_storeu_si256((__m256i *)(tmplo + j), accum_mu2_hi);

            accum_mu1_lo =
                _mm256_add_epi32(accum_mu1_lo, _mm256_madd_epi16(s0lo, fh0));
            accum_mu1_hi =
                _mm256_add_epi32(accum_mu1_hi, _mm256_madd_epi16(s0hi, fh0));
            accum_mu1_lo =
                _mm256_add_epi32(accum_mu1_lo, _mm256_madd_epi16(s1lo, fh1));
            accum_mu1_hi =
                _mm256_add_epi32(accum_mu1_hi, _mm256_madd_epi16(s1hi, fh1));

            accum_mu1_lo = _mm256_add_epi32(accum_mu1_lo, add_shift_VP_vex);
            accum_mu1_lo = _mm256_srli_epi32(accum_mu1_lo, 0x08);
            accum_mu1_hi = _mm256_add_epi32(accum_mu1_hi, add_shift_VP_vex);
            accum_mu1_hi = _mm256_srli_epi32(accum_mu1_hi, 0x08);
            accum_mu1_lo = _mm256_blend_epi16(accum_mu1_lo, pad_register, 0xAA);
            accum_mu1_hi = _mm256_blend_epi16(accum_mu1_hi, pad_register, 0xAA);
            accum_mu1_hi = _mm256_packus_epi32(accum_mu1_lo, accum_mu1_hi);
            _mm256_storeu_si256((__m256i *)(tmphi + j), accum_mu1_hi);
        }

        int j0 = ind_x[0][0];
        int j1 = ind_x[1][0];
        int j2 = ind_x[2][0];
        int j3 = ind_x[3][0];

        int16_t s0 = tmplo[j0];
        int16_t s1 = tmplo[j1];
        int16_t s2 = tmplo[j2];
        int16_t s3 = tmplo[j3];

        accum = 0;
        accum += (int32_t)filter_lo[0] * s0;
        accum += (int32_t)filter_lo[1] * s1;
        accum += (int32_t)filter_lo[2] * s2;
        accum += (int32_t)filter_lo[3] * s3;
        dst->band_a[i * dst_stride] = (accum + add_shift_HP) >> shift_HP;

        accum = 0;
        accum += (int32_t)filter_hi[0] * s0;
        accum += (int32_t)filter_hi[1] * s1;
        accum += (int32_t)filter_hi[2] * s2;
        accum += (int32_t)filter_hi[3] * s3;
        dst->band_v[i * dst_stride] = (accum + add_shift_HP) >> shift_HP;

        s0 = tmphi[j0];
        s1 = tmphi[j1];
        s2 = tmphi[j2];
        s3 = tmphi[j3];

        accum = 0;
        accum += (int32_t)filter_lo[0] * s0;
        accum += (int32_t)filter_lo[1] * s1;
        accum += (int32_t)filter_lo[2] * s2;
        accum += (int32_t)filter_lo[3] * s3;
        dst->band_h[i * dst_stride] = (accum + add_shift_HP) >> shift_HP;

        accum = 0;
        accum += (int32_t)filter_hi[0] * s0;
        accum += (int32_t)filter_hi[1] * s1;
        accum += (int32_t)filter_hi[2] * s2;
        accum += (int32_t)filter_hi[3] * s3;
        dst->band_d[i * dst_stride] = (accum + add_shift_HP) >> shift_HP;

        /* Horizontal pass - interleaved tmplo and tmphi processing. */
        for (int j = 1; j < (w + 1) / 2; j = j + 16) {
            __m256i lo_accum2_lo, lo_accum2_hi, lo_accum1_lo, lo_accum1_hi;
            __m256i hi_accum2_lo, hi_accum2_hi, hi_accum1_lo, hi_accum1_hi;
            lo_accum2_lo = lo_accum2_hi = lo_accum1_lo = lo_accum1_hi =
                _mm256_setzero_si256();
            hi_accum2_lo = hi_accum2_hi = hi_accum1_lo = hi_accum1_hi =
                _mm256_setzero_si256();

            /* Load from both tmplo and tmphi simultaneously */
            __m256i lo_s00 = _mm256_loadu_si256(
                (__m256i *)(tmplo + ind_x[0][j]));
            __m256i hi_s00 = _mm256_loadu_si256(
                (__m256i *)(tmphi + ind_x[0][j]));

            __m256i lo_s22 = _mm256_loadu_si256(
                (__m256i *)(tmplo + ind_x[2][j]));
            __m256i hi_s22 = _mm256_loadu_si256(
                (__m256i *)(tmphi + ind_x[2][j]));

            __m256i lo_s33 = _mm256_loadu_si256(
                (__m256i *)(tmplo + 16 + ind_x[0][j]));
            __m256i hi_s33 = _mm256_loadu_si256(
                (__m256i *)(tmphi + 16 + ind_x[0][j]));

            __m256i lo_s44 = _mm256_loadu_si256(
                (__m256i *)(tmplo + 16 + ind_x[2][j]));
            __m256i hi_s44 = _mm256_loadu_si256(
                (__m256i *)(tmphi + 16 + ind_x[2][j]));

            /* tmplo: multiply by fl0 */
            lo_accum2_lo = _mm256_madd_epi16(lo_s00, fl0);
            lo_accum2_hi = _mm256_madd_epi16(lo_s33, fl0);

            /* tmphi: multiply by fl0 */
            hi_accum2_lo = _mm256_madd_epi16(hi_s00, fl0);
            hi_accum2_hi = _mm256_madd_epi16(hi_s33, fl0);

            /* tmplo: multiply by fl1 and accumulate */
            lo_accum2_lo =
                _mm256_add_epi32(lo_accum2_lo, _mm256_madd_epi16(lo_s22, fl1));
            lo_accum2_hi =
                _mm256_add_epi32(lo_accum2_hi, _mm256_madd_epi16(lo_s44, fl1));

            /* tmphi: multiply by fl1 and accumulate */
            hi_accum2_lo =
                _mm256_add_epi32(hi_accum2_lo, _mm256_madd_epi16(hi_s22, fl1));
            hi_accum2_hi =
                _mm256_add_epi32(hi_accum2_hi, _mm256_madd_epi16(hi_s44, fl1));

            /* tmplo: shift and pack for band_a */
            lo_accum2_lo = _mm256_add_epi32(lo_accum2_lo, add_shift_HP_vex);
            lo_accum2_lo = _mm256_srli_epi32(lo_accum2_lo, 0x10);
            lo_accum2_hi = _mm256_add_epi32(lo_accum2_hi, add_shift_HP_vex);
            lo_accum2_hi = _mm256_srli_epi32(lo_accum2_hi, 0x10);

            /* tmphi: shift and pack for band_h */
            hi_accum2_lo = _mm256_add_epi32(hi_accum2_lo, add_shift_HP_vex);
            hi_accum2_lo = _mm256_srli_epi32(hi_accum2_lo, 0x10);
            hi_accum2_hi = _mm256_add_epi32(hi_accum2_hi, add_shift_HP_vex);
            hi_accum2_hi = _mm256_srli_epi32(hi_accum2_hi, 0x10);

            /* tmplo: pack and store band_a */
            lo_accum2_hi = _mm256_packus_epi32(lo_accum2_lo, lo_accum2_hi);
            lo_accum2_hi = _mm256_permute4x64_epi64(lo_accum2_hi, 0xD8);
            _mm256_storeu_si256(
                (__m256i *)(dst->band_a + i * dst_stride + j),
                lo_accum2_hi);

            /* tmphi: pack and store band_h */
            hi_accum2_hi = _mm256_packus_epi32(hi_accum2_lo, hi_accum2_hi);
            hi_accum2_hi = _mm256_permute4x64_epi64(hi_accum2_hi, 0xD8);
            _mm256_storeu_si256(
                (__m256i *)(dst->band_h + i * dst_stride + j),
                hi_accum2_hi);

            /* tmplo: multiply by fh0 */
            lo_accum1_lo = _mm256_madd_epi16(lo_s00, fh0);
            lo_accum1_hi = _mm256_madd_epi16(lo_s33, fh0);

            /* tmphi: multiply by fh0 */
            hi_accum1_lo = _mm256_madd_epi16(hi_s00, fh0);
            hi_accum1_hi = _mm256_madd_epi16(hi_s33, fh0);

            /* tmplo: multiply by fh1 and accumulate */
            lo_accum1_lo =
                _mm256_add_epi32(lo_accum1_lo, _mm256_madd_epi16(lo_s22, fh1));
            lo_accum1_hi =
                _mm256_add_epi32(lo_accum1_hi, _mm256_madd_epi16(lo_s44, fh1));

            /* tmphi: multiply by fh1 and accumulate */
            hi_accum1_lo =
                _mm256_add_epi32(hi_accum1_lo, _mm256_madd_epi16(hi_s22, fh1));
            hi_accum1_hi =
                _mm256_add_epi32(hi_accum1_hi, _mm256_madd_epi16(hi_s44, fh1));

            /* tmplo: shift and pack for band_v */
            lo_accum1_lo = _mm256_add_epi32(lo_accum1_lo, add_shift_HP_vex);
            lo_accum1_lo = _mm256_srli_epi32(lo_accum1_lo, 0x10);
            lo_accum1_hi = _mm256_add_epi32(lo_accum1_hi, add_shift_HP_vex);
            lo_accum1_hi = _mm256_srli_epi32(lo_accum1_hi, 0x10);

            /* tmphi: shift and pack for band_d */
            hi_accum1_lo = _mm256_add_epi32(hi_accum1_lo, add_shift_HP_vex);
            hi_accum1_lo = _mm256_srli_epi32(hi_accum1_lo, 0x10);
            hi_accum1_hi = _mm256_add_epi32(hi_accum1_hi, add_shift_HP_vex);
            hi_accum1_hi = _mm256_srli_epi32(hi_accum1_hi, 0x10);

            /* tmplo: pack and store band_v */
            lo_accum1_hi = _mm256_packus_epi32(lo_accum1_lo, lo_accum1_hi);
            lo_accum1_hi = _mm256_permute4x64_epi64(lo_accum1_hi, 0xD8);
            _mm256_storeu_si256(
                (__m256i *)(dst->band_v + i * dst_stride + j),
                lo_accum1_hi);

            /* tmphi: pack and store band_d */
            hi_accum1_hi = _mm256_packus_epi32(hi_accum1_lo, hi_accum1_hi);
            hi_accum1_hi = _mm256_permute4x64_epi64(hi_accum1_hi, 0xD8);
            _mm256_storeu_si256(
                (__m256i *)(dst->band_d + i * dst_stride + j),
                hi_accum1_hi);
        }
    }
}

void adm_csf_avx2(AdmBuffer *buf, int w, int h, int stride,
                   uint16_t i_rfactor[3], uint8_t i_shifts[3],
                   uint16_t i_shiftsadd[3])
{
    const adm_dwt_band_t *src = &buf->decouple_a;
    const adm_dwt_band_t *dst = &buf->csf_a;
    const adm_dwt_band_t *flt = &buf->csf_f;

    const int16_t *src_angles[3] = { src->band_h, src->band_v, src->band_d };
    int16_t *dst_angles[3] = { dst->band_h, dst->band_v, dst->band_d };
    int16_t *flt_angles[3] = { flt->band_h, flt->band_v, flt->band_d };

    const uint16_t FIX_ONE_BY_30 = 4369; /* (1/30)*2^17 */

    int left = w * ADM_BORDER_FACTOR - 0.5 - 1;
    int top = h * ADM_BORDER_FACTOR - 0.5 - 1;
    int right = w - left + 2;
    int bottom = h - top + 2;

    if (left < 0)   left = 0;
    if (right > w)   right = w;
    if (top < 0)     top = 0;
    if (bottom > h)  bottom = h;

    for (int theta = 0; theta < 3; ++theta) {
        const int16_t *src_ptr = src_angles[theta];
        int16_t *dst_ptr = dst_angles[theta];
        int16_t *flt_ptr = flt_angles[theta];

        const uint16_t rfactor = i_rfactor[theta];
        const int shift = i_shifts[theta];
        const int32_t add_val = (int32_t)i_shiftsadd[theta];

        /* Broadcast constants for this theta */
        const __m256i v_add = _mm256_set1_epi32(add_val);
        const __m256i v_fix30 = _mm256_set1_epi32(FIX_ONE_BY_30);
        const __m256i v_flt_add = _mm256_set1_epi32(2048);

        for (int i = top; i < bottom; ++i) {
            const int offset = i * stride;
            int j = left;

            /* AVX2 loop: process 16 int16_t values per iteration */
            for (; j + 15 < right; j += 16) {
                /* Load 16 int16_t source values */
                __m256i src_v = _mm256_loadu_si256(
                    (const __m256i *)(src_ptr + offset + j));

                /*
                 * Multiply src by rfactor:
                 * We need (int32_t)rfactor * (int32_t)src for each element.
                 * rfactor is uint16_t, src is int16_t.
                 * Use mullo/mulhi to get full 32-bit results.
                 *
                 * _mm256_mullo_epi16 gives low 16 bits of each 16x16 product
                 * _mm256_mulhi_epi16 gives high 16 bits (signed)
                 * But rfactor is unsigned, so we need mulhi_epu16 for unsigned*signed.
                 *
                 * Actually, the C code does: i_rfactor[theta] * (int32_t)src_ptr[...]
                 * where i_rfactor is uint16_t. The product is int32_t.
                 *
                 * For correct sign handling with unsigned * signed:
                 * Use _mm256_mulhi_epi16 (treats both as signed) then correct.
                 * Alternatively, widen to 32-bit and multiply.
                 */

                /* Widen src to 32-bit (lo and hi halves) */
                __m256i src_lo = _mm256_cvtepi16_epi32(
                    _mm256_castsi256_si128(src_v));
                __m256i src_hi = _mm256_cvtepi16_epi32(
                    _mm256_extracti128_si256(src_v, 1));

                /* rfactor as 32-bit (unsigned, fits in int32) */
                __m256i v_rfactor32 = _mm256_set1_epi32((int32_t)rfactor);

                /* Multiply: int32 * int32 -> int32 (fits because
                 * rfactor is 16-bit and src is 16-bit, product fits in 32 bits) */
                __m256i dst_lo = _mm256_mullo_epi32(v_rfactor32, src_lo);
                __m256i dst_hi = _mm256_mullo_epi32(v_rfactor32, src_hi);

                /* Add rounding and shift right */
                dst_lo = _mm256_add_epi32(dst_lo, v_add);
                dst_hi = _mm256_add_epi32(dst_hi, v_add);
                dst_lo = _mm256_srai_epi32(dst_lo, shift);
                dst_hi = _mm256_srai_epi32(dst_hi, shift);

                /*
                 * Truncate to int16: C does (int16_t) cast which wraps,
                 * NOT saturation. Use sign-extension of lower 16 bits.
                 */
                __m256i dst_lo_i16 = _mm256_srai_epi32(
                    _mm256_slli_epi32(dst_lo, 16), 16);
                __m256i dst_hi_i16 = _mm256_srai_epi32(
                    _mm256_slli_epi32(dst_hi, 16), 16);

                /* Pack to int16 using packs (values now in int16 range after truncation) */
                __m256i dst_i16 = _mm256_packs_epi32(dst_lo_i16, dst_hi_i16);
                dst_i16 = _mm256_permute4x64_epi64(dst_i16, 0xD8);

                /* Store dst values */
                _mm256_storeu_si256((__m256i *)(dst_ptr + offset + j), dst_i16);

                /*
                 * Compute flt = (FIX_ONE_BY_30 * abs(i16_dst_val) + 2048) >> 12
                 * i16_dst_val is the truncated int16 value (now in dst_lo_i16/dst_hi_i16).
                 * Use abs of the truncated int16 values (in 32-bit form).
                 */
                __m256i abs_lo = _mm256_abs_epi32(dst_lo_i16);
                __m256i abs_hi = _mm256_abs_epi32(dst_hi_i16);

                /* Multiply by FIX_ONE_BY_30 (4369). Product fits in 32 bits:
                 * max abs value is 32768, 32768 * 4369 = ~143,163,392
                 * which fits in int32 */
                __m256i flt_lo = _mm256_mullo_epi32(v_fix30, abs_lo);
                __m256i flt_hi = _mm256_mullo_epi32(v_fix30, abs_hi);

                /* Add 2048 and shift right by 12 */
                flt_lo = _mm256_add_epi32(flt_lo, v_flt_add);
                flt_hi = _mm256_add_epi32(flt_hi, v_flt_add);
                flt_lo = _mm256_srai_epi32(flt_lo, 12);
                flt_hi = _mm256_srai_epi32(flt_hi, 12);

                /* Pack to int16 (flt values are always positive and fit in int16) */
                __m256i flt_i16 = _mm256_packs_epi32(flt_lo, flt_hi);
                flt_i16 = _mm256_permute4x64_epi64(flt_i16, 0xD8);

                /* Store flt values */
                _mm256_storeu_si256((__m256i *)(flt_ptr + offset + j), flt_i16);
            }

            /* Scalar tail */
            for (; j < right; ++j) {
                int32_t dst_val = rfactor * (int32_t)src_ptr[offset + j];
                int16_t i16_dst_val = (int16_t)((dst_val + add_val) >> shift);
                dst_ptr[offset + j] = i16_dst_val;
                flt_ptr[offset + j] = (int16_t)(((FIX_ONE_BY_30 *
                    abs((int32_t)i16_dst_val)) + 2048) >> 12);
            }
        }
    }
}

void adm_decouple_avx2(AdmBuffer *buf, int w, int h, int stride,
                       double adm_enhn_gain_limit,
                       const int32_t *div_lookup_ptr)
{
    const float cos_1deg_sq = (float)(cos(1.0 * M_PI / 180.0) *
                                      cos(1.0 * M_PI / 180.0));

    const adm_dwt_band_t *ref = &buf->ref_dwt2;
    const adm_dwt_band_t *dis = &buf->dis_dwt2;
    const adm_dwt_band_t *r = &buf->decouple_r;
    const adm_dwt_band_t *a = &buf->decouple_a;

    int left = w * ADM_BORDER_FACTOR - 0.5 - 1;
    int top = h * ADM_BORDER_FACTOR - 0.5 - 1;
    int right = w - left + 2;
    int bottom = h - top + 2;

    if (left < 0)   left = 0;
    if (right > w)   right = w;
    if (top < 0)     top = 0;
    if (bottom > h)  bottom = h;

    /* Hoist loop-invariant constant vectors */
    __m256d v_inv_4096_d = _mm256_set1_pd(1.0 / 4096.0);
    __m256d v_zero_d = _mm256_setzero_pd();
    __m256d v_cos_d = _mm256_set1_pd((double)cos_1deg_sq);
    __m256i v_32768 = _mm256_set1_epi32(32768);
    __m256i v_16384 = _mm256_set1_epi32(16384);
    __m256i v_zero = _mm256_setzero_si256();
    __m256 v_gain = _mm256_set1_ps((float)adm_enhn_gain_limit);
    __m256 v_inv_32768_f = _mm256_set1_ps(1.0f / 32768.0f);
    __m256 v_inv_64_f = _mm256_set1_ps(1.0f / 64.0f);
    __m256 v_zero_f = _mm256_setzero_ps();

    for (int i = top; i < bottom; ++i) {
        int j = left;

        /* Process 8 pixels at a time with AVX2 */
        for (; j + 7 < right; j += 8) {
            const int idx = i * stride + j;

            /* Load 8 int16_t values for each band */
            __m128i oh_128 = _mm_loadu_si128((const __m128i *)(ref->band_h + idx));
            __m128i ov_128 = _mm_loadu_si128((const __m128i *)(ref->band_v + idx));
            __m128i od_128 = _mm_loadu_si128((const __m128i *)(ref->band_d + idx));
            __m128i th_128 = _mm_loadu_si128((const __m128i *)(dis->band_h + idx));
            __m128i tv_128 = _mm_loadu_si128((const __m128i *)(dis->band_v + idx));
            __m128i td_128 = _mm_loadu_si128((const __m128i *)(dis->band_d + idx));

            /* Sign-extend to 32-bit */
            __m256i oh = _mm256_cvtepi16_epi32(oh_128);
            __m256i ov = _mm256_cvtepi16_epi32(ov_128);
            __m256i od = _mm256_cvtepi16_epi32(od_128);
            __m256i th = _mm256_cvtepi16_epi32(th_128);
            __m256i tv = _mm256_cvtepi16_epi32(tv_128);
            __m256i td = _mm256_cvtepi16_epi32(td_128);

            /*
             * Compute angle_flag matching the C reference exactly.
             * C reference does: (float)ot_dp / 4096.0 which promotes to double.
             * All comparisons happen in double precision.
             * Process in two batches of 4 elements using __m256d.
             */
            __m256i ot_dp_i = _mm256_add_epi32(
                _mm256_mullo_epi32(oh, th),
                _mm256_mullo_epi32(ov, tv));
            __m256i o_mag_sq_i = _mm256_add_epi32(
                _mm256_mullo_epi32(oh, oh),
                _mm256_mullo_epi32(ov, ov));
            __m256i t_mag_sq_i = _mm256_add_epi32(
                _mm256_mullo_epi32(th, th),
                _mm256_mullo_epi32(tv, tv));

            /*
             * C does: (float)ot_dp -> float, then / 4096.0 -> double
             * We replicate: int32 -> float -> double, then /4096.0 in double
             */
            __m256 ot_dp_ps = _mm256_cvtepi32_ps(ot_dp_i);
            __m256 o_mag_ps = _mm256_cvtepi32_ps(o_mag_sq_i);
            __m256 t_mag_ps = _mm256_cvtepi32_ps(t_mag_sq_i);

            /* Split into lo/hi 4 floats, convert to double */
            __m256d ot_dp_d_lo = _mm256_cvtps_pd(_mm256_castps256_ps128(ot_dp_ps));
            __m256d ot_dp_d_hi = _mm256_cvtps_pd(_mm256_extractf128_ps(ot_dp_ps, 1));
            __m256d o_mag_d_lo = _mm256_cvtps_pd(_mm256_castps256_ps128(o_mag_ps));
            __m256d o_mag_d_hi = _mm256_cvtps_pd(_mm256_extractf128_ps(o_mag_ps, 1));
            __m256d t_mag_d_lo = _mm256_cvtps_pd(_mm256_castps256_ps128(t_mag_ps));
            __m256d t_mag_d_hi = _mm256_cvtps_pd(_mm256_extractf128_ps(t_mag_ps, 1));

            /* ot_dp / 4096.0 (multiply by exact reciprocal) */
            __m256d dp_lo = _mm256_mul_pd(ot_dp_d_lo, v_inv_4096_d);
            __m256d dp_hi = _mm256_mul_pd(ot_dp_d_hi, v_inv_4096_d);
            __m256d omag_lo = _mm256_mul_pd(o_mag_d_lo, v_inv_4096_d);
            __m256d omag_hi = _mm256_mul_pd(o_mag_d_hi, v_inv_4096_d);
            __m256d tmag_lo = _mm256_mul_pd(t_mag_d_lo, v_inv_4096_d);
            __m256d tmag_hi = _mm256_mul_pd(t_mag_d_hi, v_inv_4096_d);

            /* cond1: dp >= 0 */
            __m256d cond1_lo = _mm256_cmp_pd(dp_lo, v_zero_d, _CMP_GE_OQ);
            __m256d cond1_hi = _mm256_cmp_pd(dp_hi, v_zero_d, _CMP_GE_OQ);

            /* cond2: dp^2 >= cos_1deg_sq * omag * tmag */
            __m256d dp_sq_lo = _mm256_mul_pd(dp_lo, dp_lo);
            __m256d dp_sq_hi = _mm256_mul_pd(dp_hi, dp_hi);
            __m256d mag_prod_lo = _mm256_mul_pd(_mm256_mul_pd(v_cos_d, omag_lo), tmag_lo);
            __m256d mag_prod_hi = _mm256_mul_pd(_mm256_mul_pd(v_cos_d, omag_hi), tmag_hi);
            __m256d cond2_lo = _mm256_cmp_pd(dp_sq_lo, mag_prod_lo, _CMP_GE_OQ);
            __m256d cond2_hi = _mm256_cmp_pd(dp_sq_hi, mag_prod_hi, _CMP_GE_OQ);

            /* Combine conditions */
            __m256d angle_d_lo = _mm256_and_pd(cond1_lo, cond2_lo);
            __m256d angle_d_hi = _mm256_and_pd(cond1_hi, cond2_hi);

            /*
             * _mm256_cvtpd_ps converts double mask bits (all 1s per 64-bit) to
             * float values. NaN/-0 issues possible. Instead, extract sign bits
             * and build mask manually.
             */
            /* Use the double mask to create a 32-bit integer mask */
            int mask_lo = _mm256_movemask_pd(angle_d_lo); /* 4-bit mask */
            int mask_hi = _mm256_movemask_pd(angle_d_hi); /* 4-bit mask */
            int full_mask = mask_lo | (mask_hi << 4);      /* 8-bit mask */

            /* Create int32 mask from the 8-bit mask */
            __m256i angle_mask = _mm256_set_epi32(
                (full_mask & 0x80) ? -1 : 0,
                (full_mask & 0x40) ? -1 : 0,
                (full_mask & 0x20) ? -1 : 0,
                (full_mask & 0x10) ? -1 : 0,
                (full_mask & 0x08) ? -1 : 0,
                (full_mask & 0x04) ? -1 : 0,
                (full_mask & 0x02) ? -1 : 0,
                (full_mask & 0x01) ? -1 : 0
            );
            __m256 angle_mask_f = _mm256_castsi256_ps(angle_mask);

            /*
             * Division via lookup table:
             * tmp_k = (o == 0) ? 32768 : ((div_lookup[o + 32768] * t) + 16384) >> 15
             * k = clamp(tmp_k, 0, 32768)
             *
             * For the gather: index = o + 32768 (o is int16, so index is [0, 65536])
             * div_lookup is int32_t[65537]
             */

            /* --- band_h: kh --- */
            __m256i oh_idx = _mm256_add_epi32(oh, v_32768);
            __m256i div_h = _mm256_i32gather_epi32(div_lookup_ptr, oh_idx, 4);
            /* tmp_kh = (div_lookup[oh+32768] * th + 16384) >> 15
             * div_lookup values are int32, th values are int32 (sign-extended from int16)
             * Product can be up to ~1e9 * 32767 which overflows int32.
             * We need 64-bit multiply. Process in two halves.
             */
            {
                /* Extract low 4 and high 4 elements */
                __m128i div_h_lo = _mm256_castsi256_si128(div_h);
                __m128i div_h_hi = _mm256_extracti128_si256(div_h, 1);
                __m128i th_lo = _mm256_castsi256_si128(th);
                __m128i th_hi = _mm256_extracti128_si256(th, 1);

                /* Compute 64-bit products for 8 elements, 2 at a time with _mm_mul_epi32 */
                /* _mm_mul_epi32 multiplies elements 0,2 (low 32 bits of 64-bit pairs) */

                /* For elements 0,1,2,3 (lo 128-bit lane) */
                __m128i prod_02_lo = _mm_mul_epi32(div_h_lo, th_lo);  /* elements 0,2 */
                __m128i prod_13_lo = _mm_mul_epi32(
                    _mm_srli_si128(div_h_lo, 4),
                    _mm_srli_si128(th_lo, 4));  /* elements 1,3 */

                /* Add 16384 and shift right by 15 */
                __m128i v_16384_128 = _mm_set1_epi64x(16384);
                __m128i r02_lo = srai_epi64_15(
                    _mm_add_epi64(prod_02_lo, v_16384_128));
                __m128i r13_lo = srai_epi64_15(
                    _mm_add_epi64(prod_13_lo, v_16384_128));

                /* For elements 4,5,6,7 (hi 128-bit lane) */
                __m128i prod_02_hi = _mm_mul_epi32(div_h_hi, th_hi);
                __m128i prod_13_hi = _mm_mul_epi32(
                    _mm_srli_si128(div_h_hi, 4),
                    _mm_srli_si128(th_hi, 4));

                __m128i r02_hi = srai_epi64_15(
                    _mm_add_epi64(prod_02_hi, v_16384_128));
                __m128i r13_hi = srai_epi64_15(
                    _mm_add_epi64(prod_13_hi, v_16384_128));

                /* Recombine: need to interleave elements 0,1,2,3 from the pairs */
                /* r02 has [result0, 0, result2, 0], r13 has [result1, 0, result3, 0] */
                /* Shuffle to get [r0, r1, r2, r3] as int32 */
                __m128i r_lo_shuf = _mm_castps_si128(_mm_shuffle_ps(
                    _mm_castsi128_ps(r02_lo), _mm_castsi128_ps(r13_lo),
                    _MM_SHUFFLE(2, 0, 2, 0)));
                /* Result is [r0, r2, r1, r3], need to reorder to [r0, r1, r2, r3] */
                r_lo_shuf = _mm_shuffle_epi32(r_lo_shuf, _MM_SHUFFLE(3, 1, 2, 0));

                __m128i r_hi_shuf = _mm_castps_si128(_mm_shuffle_ps(
                    _mm_castsi128_ps(r02_hi), _mm_castsi128_ps(r13_hi),
                    _MM_SHUFFLE(2, 0, 2, 0)));
                r_hi_shuf = _mm_shuffle_epi32(r_hi_shuf, _MM_SHUFFLE(3, 1, 2, 0));

                __m256i tmp_kh = _mm256_setr_m128i(r_lo_shuf, r_hi_shuf);

                /* Handle oh == 0 case: where oh == 0, use 32768 */
                __m256i oh_zero_mask = _mm256_cmpeq_epi32(oh, v_zero);
                tmp_kh = _mm256_blendv_epi8(tmp_kh, v_32768, oh_zero_mask);

                /* Clamp to [0, 32768] */
                __m256i kh = _mm256_max_epi32(tmp_kh, v_zero);
                kh = _mm256_min_epi32(kh, v_32768);

                /* rst_h = (kh * oh + 16384) >> 15 */
                __m256i rst_h = _mm256_srai_epi32(
                    _mm256_add_epi32(_mm256_mullo_epi32(kh, oh), v_16384), 15);

                /* --- band_v: kv --- */
                __m256i ov_idx = _mm256_add_epi32(ov, v_32768);
                __m256i div_v = _mm256_i32gather_epi32(div_lookup_ptr, ov_idx, 4);

                __m128i div_v_lo = _mm256_castsi256_si128(div_v);
                __m128i div_v_hi = _mm256_extracti128_si256(div_v, 1);
                __m128i tv_lo = _mm256_castsi256_si128(tv);
                __m128i tv_hi = _mm256_extracti128_si256(tv, 1);

                prod_02_lo = _mm_mul_epi32(div_v_lo, tv_lo);
                prod_13_lo = _mm_mul_epi32(
                    _mm_srli_si128(div_v_lo, 4),
                    _mm_srli_si128(tv_lo, 4));
                r02_lo = srai_epi64_15(
                    _mm_add_epi64(prod_02_lo, v_16384_128));
                r13_lo = srai_epi64_15(
                    _mm_add_epi64(prod_13_lo, v_16384_128));

                prod_02_hi = _mm_mul_epi32(div_v_hi, tv_hi);
                prod_13_hi = _mm_mul_epi32(
                    _mm_srli_si128(div_v_hi, 4),
                    _mm_srli_si128(tv_hi, 4));
                r02_hi = srai_epi64_15(
                    _mm_add_epi64(prod_02_hi, v_16384_128));
                r13_hi = srai_epi64_15(
                    _mm_add_epi64(prod_13_hi, v_16384_128));

                r_lo_shuf = _mm_castps_si128(_mm_shuffle_ps(
                    _mm_castsi128_ps(r02_lo), _mm_castsi128_ps(r13_lo),
                    _MM_SHUFFLE(2, 0, 2, 0)));
                r_lo_shuf = _mm_shuffle_epi32(r_lo_shuf, _MM_SHUFFLE(3, 1, 2, 0));

                r_hi_shuf = _mm_castps_si128(_mm_shuffle_ps(
                    _mm_castsi128_ps(r02_hi), _mm_castsi128_ps(r13_hi),
                    _MM_SHUFFLE(2, 0, 2, 0)));
                r_hi_shuf = _mm_shuffle_epi32(r_hi_shuf, _MM_SHUFFLE(3, 1, 2, 0));

                __m256i tmp_kv = _mm256_setr_m128i(r_lo_shuf, r_hi_shuf);
                __m256i ov_zero_mask = _mm256_cmpeq_epi32(ov, v_zero);
                tmp_kv = _mm256_blendv_epi8(tmp_kv, v_32768, ov_zero_mask);
                __m256i kv = _mm256_max_epi32(tmp_kv, v_zero);
                kv = _mm256_min_epi32(kv, v_32768);

                __m256i rst_v = _mm256_srai_epi32(
                    _mm256_add_epi32(_mm256_mullo_epi32(kv, ov), v_16384), 15);

                /* --- band_d: kd --- */
                __m256i od_idx = _mm256_add_epi32(od, v_32768);
                __m256i div_d = _mm256_i32gather_epi32(div_lookup_ptr, od_idx, 4);

                __m128i div_d_lo = _mm256_castsi256_si128(div_d);
                __m128i div_d_hi = _mm256_extracti128_si256(div_d, 1);
                __m128i td_lo = _mm256_castsi256_si128(td);
                __m128i td_hi = _mm256_extracti128_si256(td, 1);

                prod_02_lo = _mm_mul_epi32(div_d_lo, td_lo);
                prod_13_lo = _mm_mul_epi32(
                    _mm_srli_si128(div_d_lo, 4),
                    _mm_srli_si128(td_lo, 4));
                r02_lo = srai_epi64_15(
                    _mm_add_epi64(prod_02_lo, v_16384_128));
                r13_lo = srai_epi64_15(
                    _mm_add_epi64(prod_13_lo, v_16384_128));

                prod_02_hi = _mm_mul_epi32(div_d_hi, td_hi);
                prod_13_hi = _mm_mul_epi32(
                    _mm_srli_si128(div_d_hi, 4),
                    _mm_srli_si128(td_hi, 4));
                r02_hi = srai_epi64_15(
                    _mm_add_epi64(prod_02_hi, v_16384_128));
                r13_hi = srai_epi64_15(
                    _mm_add_epi64(prod_13_hi, v_16384_128));

                r_lo_shuf = _mm_castps_si128(_mm_shuffle_ps(
                    _mm_castsi128_ps(r02_lo), _mm_castsi128_ps(r13_lo),
                    _MM_SHUFFLE(2, 0, 2, 0)));
                r_lo_shuf = _mm_shuffle_epi32(r_lo_shuf, _MM_SHUFFLE(3, 1, 2, 0));

                r_hi_shuf = _mm_castps_si128(_mm_shuffle_ps(
                    _mm_castsi128_ps(r02_hi), _mm_castsi128_ps(r13_hi),
                    _MM_SHUFFLE(2, 0, 2, 0)));
                r_hi_shuf = _mm_shuffle_epi32(r_hi_shuf, _MM_SHUFFLE(3, 1, 2, 0));

                __m256i tmp_kd = _mm256_setr_m128i(r_lo_shuf, r_hi_shuf);
                __m256i od_zero_mask = _mm256_cmpeq_epi32(od, v_zero);
                tmp_kd = _mm256_blendv_epi8(tmp_kd, v_32768, od_zero_mask);
                __m256i kd = _mm256_max_epi32(tmp_kd, v_zero);
                kd = _mm256_min_epi32(kd, v_32768);

                __m256i rst_d = _mm256_srai_epi32(
                    _mm256_add_epi32(_mm256_mullo_epi32(kd, od), v_16384), 15);

                /*
                 * Enhancement gain limit:
                 * rst_f = (k/32768.0) * (o/64.0)
                 * if angle_flag && rst_f > 0: rst = MIN(rst * gain_limit, t)
                 * if angle_flag && rst_f < 0: rst = MAX(rst * gain_limit, t)
                 *
                 * Process each component using float arithmetic matching C reference.
                 */

                /* Helper macro-like: apply enhancement gain limit */
                /* band_h */
                {
                    __m256 kh_f = _mm256_mul_ps(_mm256_cvtepi32_ps(kh), v_inv_32768_f);
                    __m256 oh_f = _mm256_mul_ps(_mm256_cvtepi32_ps(oh), v_inv_64_f);
                    __m256 rst_h_f = _mm256_mul_ps(kh_f, oh_f);

                    __m256 rst_h_ps = _mm256_cvtepi32_ps(rst_h);
                    __m256 rst_scaled = _mm256_mul_ps(rst_h_ps, v_gain);
                    __m256 th_f = _mm256_cvtepi32_ps(th);

                    /* if rst_h_f > 0: result = MIN(rst_scaled, th) */
                    __m256 pos_mask = _mm256_cmp_ps(rst_h_f, v_zero_f, _CMP_GT_OQ);
                    __m256 neg_mask = _mm256_cmp_ps(rst_h_f, v_zero_f, _CMP_LT_OQ);

                    __m256 min_val = _mm256_min_ps(rst_scaled, th_f);
                    __m256 max_val = _mm256_max_ps(rst_scaled, th_f);

                    /* Apply: if angle_flag && pos, use min; if angle_flag && neg, use max */
                    __m256 current = rst_h_ps;
                    current = _mm256_blendv_ps(current, min_val,
                        _mm256_and_ps(angle_mask_f, pos_mask));
                    current = _mm256_blendv_ps(current, max_val,
                        _mm256_and_ps(angle_mask_f, neg_mask));

                    /* Convert back to int32 (truncation, matching C cast) */
                    rst_h = _mm256_cvttps_epi32(current);
                }

                /* band_v */
                {
                    __m256 kv_f = _mm256_mul_ps(_mm256_cvtepi32_ps(kv), v_inv_32768_f);
                    __m256 ov_f = _mm256_mul_ps(_mm256_cvtepi32_ps(ov), v_inv_64_f);
                    __m256 rst_v_f = _mm256_mul_ps(kv_f, ov_f);

                    __m256 rst_v_ps = _mm256_cvtepi32_ps(rst_v);
                    __m256 rst_scaled = _mm256_mul_ps(rst_v_ps, v_gain);
                    __m256 tv_f = _mm256_cvtepi32_ps(tv);

                    __m256 pos_mask = _mm256_cmp_ps(rst_v_f, v_zero_f, _CMP_GT_OQ);
                    __m256 neg_mask = _mm256_cmp_ps(rst_v_f, v_zero_f, _CMP_LT_OQ);

                    __m256 min_val = _mm256_min_ps(rst_scaled, tv_f);
                    __m256 max_val = _mm256_max_ps(rst_scaled, tv_f);

                    __m256 current = rst_v_ps;
                    current = _mm256_blendv_ps(current, min_val,
                        _mm256_and_ps(angle_mask_f, pos_mask));
                    current = _mm256_blendv_ps(current, max_val,
                        _mm256_and_ps(angle_mask_f, neg_mask));

                    rst_v = _mm256_cvttps_epi32(current);
                }

                /* band_d */
                {
                    __m256 kd_f = _mm256_mul_ps(_mm256_cvtepi32_ps(kd), v_inv_32768_f);
                    __m256 od_f = _mm256_mul_ps(_mm256_cvtepi32_ps(od), v_inv_64_f);
                    __m256 rst_d_f = _mm256_mul_ps(kd_f, od_f);

                    __m256 rst_d_ps = _mm256_cvtepi32_ps(rst_d);
                    __m256 rst_scaled = _mm256_mul_ps(rst_d_ps, v_gain);
                    __m256 td_f = _mm256_cvtepi32_ps(td);

                    __m256 pos_mask = _mm256_cmp_ps(rst_d_f, v_zero_f, _CMP_GT_OQ);
                    __m256 neg_mask = _mm256_cmp_ps(rst_d_f, v_zero_f, _CMP_LT_OQ);

                    __m256 min_val = _mm256_min_ps(rst_scaled, td_f);
                    __m256 max_val = _mm256_max_ps(rst_scaled, td_f);

                    __m256 current = rst_d_ps;
                    current = _mm256_blendv_ps(current, min_val,
                        _mm256_and_ps(angle_mask_f, pos_mask));
                    current = _mm256_blendv_ps(current, max_val,
                        _mm256_and_ps(angle_mask_f, neg_mask));

                    rst_d = _mm256_cvttps_epi32(current);
                }

                /* Store results: pack int32 -> int16 and store */
                /* r->band_h = rst_h, a->band_h = th - rst_h */
                __m256i r_h_16 = _mm256_packs_epi32(rst_h, rst_h);
                r_h_16 = _mm256_permute4x64_epi64(r_h_16, 0xD8);
                __m256i r_v_16 = _mm256_packs_epi32(rst_v, rst_v);
                r_v_16 = _mm256_permute4x64_epi64(r_v_16, 0xD8);
                __m256i r_d_16 = _mm256_packs_epi32(rst_d, rst_d);
                r_d_16 = _mm256_permute4x64_epi64(r_d_16, 0xD8);

                __m256i a_h_32 = _mm256_sub_epi32(th, rst_h);
                __m256i a_v_32 = _mm256_sub_epi32(tv, rst_v);
                __m256i a_d_32 = _mm256_sub_epi32(td, rst_d);

                __m256i a_h_16 = _mm256_packs_epi32(a_h_32, a_h_32);
                a_h_16 = _mm256_permute4x64_epi64(a_h_16, 0xD8);
                __m256i a_v_16 = _mm256_packs_epi32(a_v_32, a_v_32);
                a_v_16 = _mm256_permute4x64_epi64(a_v_16, 0xD8);
                __m256i a_d_16 = _mm256_packs_epi32(a_d_32, a_d_32);
                a_d_16 = _mm256_permute4x64_epi64(a_d_16, 0xD8);

                /* Store only lower 128 bits (8 int16 values) */
                _mm_storeu_si128((__m128i *)(r->band_h + idx),
                    _mm256_castsi256_si128(r_h_16));
                _mm_storeu_si128((__m128i *)(r->band_v + idx),
                    _mm256_castsi256_si128(r_v_16));
                _mm_storeu_si128((__m128i *)(r->band_d + idx),
                    _mm256_castsi256_si128(r_d_16));

                _mm_storeu_si128((__m128i *)(a->band_h + idx),
                    _mm256_castsi256_si128(a_h_16));
                _mm_storeu_si128((__m128i *)(a->band_v + idx),
                    _mm256_castsi256_si128(a_v_16));
                _mm_storeu_si128((__m128i *)(a->band_d + idx),
                    _mm256_castsi256_si128(a_d_16));
            }
        }

        /* Scalar tail */
        for (; j < right; ++j) {
            const int idx = i * stride + j;

            int16_t oh = ref->band_h[idx];
            int16_t ov = ref->band_v[idx];
            int16_t od = ref->band_d[idx];
            int16_t th = dis->band_h[idx];
            int16_t tv = dis->band_v[idx];
            int16_t td = dis->band_d[idx];
            int16_t rst_h, rst_v, rst_d;

            int64_t ot_dp = (int64_t)oh * th + (int64_t)ov * tv;
            int64_t o_mag_sq = (int64_t)oh * oh + (int64_t)ov * ov;
            int64_t t_mag_sq = (int64_t)th * th + (int64_t)tv * tv;

            int angle_flag = (((float)ot_dp / 4096.0) >= 0.0f) &&
                (((float)ot_dp / 4096.0) * ((float)ot_dp / 4096.0) >=
                    cos_1deg_sq * ((float)o_mag_sq / 4096.0) *
                    ((float)t_mag_sq / 4096.0));

            int32_t tmp_kh = (oh == 0) ?
                32768 : (((int64_t)div_lookup_ptr[oh + 32768] * th) + 16384) >> 15;
            int32_t tmp_kv = (ov == 0) ?
                32768 : (((int64_t)div_lookup_ptr[ov + 32768] * tv) + 16384) >> 15;
            int32_t tmp_kd = (od == 0) ?
                32768 : (((int64_t)div_lookup_ptr[od + 32768] * td) + 16384) >> 15;

            int32_t kh = tmp_kh < 0 ? 0 : (tmp_kh > 32768 ? 32768 : tmp_kh);
            int32_t kv = tmp_kv < 0 ? 0 : (tmp_kv > 32768 ? 32768 : tmp_kv);
            int32_t kd = tmp_kd < 0 ? 0 : (tmp_kd > 32768 ? 32768 : tmp_kd);

            rst_h = ((kh * oh) + 16384) >> 15;
            rst_v = ((kv * ov) + 16384) >> 15;
            rst_d = ((kd * od) + 16384) >> 15;

            const float rst_h_f = ((float)kh / 32768) * ((float)oh / 64);
            const float rst_v_f = ((float)kv / 32768) * ((float)ov / 64);
            const float rst_d_f = ((float)kd / 32768) * ((float)od / 64);

            if (angle_flag && (rst_h_f > 0.))
                rst_h = MIN((rst_h * adm_enhn_gain_limit), th);
            if (angle_flag && (rst_h_f < 0.))
                rst_h = MAX((rst_h * adm_enhn_gain_limit), th);

            if (angle_flag && (rst_v_f > 0.))
                rst_v = MIN(rst_v * adm_enhn_gain_limit, tv);
            if (angle_flag && (rst_v_f < 0.))
                rst_v = MAX(rst_v * adm_enhn_gain_limit, tv);

            if (angle_flag && (rst_d_f > 0.))
                rst_d = MIN(rst_d * adm_enhn_gain_limit, td);
            if (angle_flag && (rst_d_f < 0.))
                rst_d = MAX(rst_d * adm_enhn_gain_limit, td);

            r->band_h[idx] = rst_h;
            r->band_v[idx] = rst_v;
            r->band_d[idx] = rst_d;

            a->band_h[idx] = th - rst_h;
            a->band_v[idx] = tv - rst_v;
            a->band_d[idx] = td - rst_d;
        }
    }
}

/* ================================================================
 * AVX2 implementation of adm_decouple for scales 1-3 (int32 data).
 *
 * This mirrors the C reference adm_decouple_s123() in integer_adm.c.
 * Key difference from scale 0: data is int32_t (in i4_adm_dwt_band_t)
 * and division uses get_best15_from32() to normalise int32 denominators
 * down to 15-bit range before table lookup.
 * ================================================================ */

/*
 * Vectorised get_best15_from32 + table lookup for one band (8 elements).
 *
 * For each element:
 *   abs_o = abs(o)
 *   if abs_o < 32768:  msb = abs_o, shift = 0
 *   else:  k = 17 - clz(abs_o); msb = (abs_o + (1<<(k-1))) >> k; shift = k
 *   looked_up = div_lookup[msb + 32768]
 *
 * Returns looked_up values in *out_div, shift values in *out_shift.
 * Also returns the sign mask (all-ones where o < 0) in *out_neg.
 */
static inline void get_best15_gather_avx2(
    __m256i o,             /* 8 x int32 original values */
    const int32_t *div_lookup_ptr,
    __m256i *out_div,      /* 8 x int32 div_lookup results */
    __m256i *out_shift,    /* 8 x int32 shift amounts (k) */
    __m256i *out_neg)      /* 8 x int32 sign mask (all 1s where o < 0) */
{
    __m256i v_zero = _mm256_setzero_si256();
    __m256i v_32768 = _mm256_set1_epi32(32768);
    __m256i v_one = _mm256_set1_epi32(1);

    /* Sign of original value */
    __m256i neg_mask = _mm256_cmpgt_epi32(v_zero, o);  /* all-1s where o < 0 */
    *out_neg = neg_mask;

    /* abs_o = abs(o) */
    __m256i abs_o = _mm256_abs_epi32(o);

    /*
     * Compute per-element CLZ using float-exponent trick.
     *
     * Converting an integer to float gives us the exponent in the IEEE 754
     * representation, which encodes floor(log2(abs_val)).
     * exponent_field = floor(log2(val)) + 127 (for val > 0)
     * clz = 31 - floor(log2(val)) = 31 - (exponent_field - 127) = 158 - exponent_field
     * k = max(0, 17 - clz) = max(0, exponent_field - 141)
     *
     * Caveat: cvtepi32_ps may round up past a power of 2 for certain values
     * (e.g., 33554431 -> 33554432.0f), giving exponent off by 1 high.
     * We fix this after computing temp by checking bit 14 of abs_o >> k.
     */
    __m256 f = _mm256_cvtepi32_ps(abs_o);
    __m256i float_bits = _mm256_castps_si256(f);
    __m256i exponent = _mm256_srli_epi32(float_bits, 23);
    /* clz = 158 - exponent */
    __m256i clz = _mm256_sub_epi32(_mm256_set1_epi32(158), exponent);
    /* k = max(0, 17 - clz) = max(0, exponent - 141) */
    __m256i k = _mm256_max_epi32(
        _mm256_sub_epi32(_mm256_set1_epi32(17), clz),
        v_zero);

    /* For abs_o < 32768, k should be 0 */
    __m256i small_mask = _mm256_cmpgt_epi32(v_32768, abs_o);  /* all-1s where abs_o < 32768 */
    k = _mm256_andnot_si256(small_mask, k);  /* zero out k where abs_o < 32768 */

    /*
     * Variable right-shift: temp = (abs_o + round) >> k
     * round = (k > 0) ? (1 << (k-1)) : 0
     */
    __m256i k_minus_1 = _mm256_max_epi32(_mm256_sub_epi32(k, v_one), v_zero);
    __m256i round_val = _mm256_sllv_epi32(v_one, k_minus_1);
    __m256i k_nonzero = _mm256_cmpgt_epi32(k, v_zero);
    round_val = _mm256_and_si256(round_val, k_nonzero);
    __m256i temp = _mm256_srlv_epi32(_mm256_add_epi32(abs_o, round_val), k);

    /*
     * Fixup for float rounding: cvtepi32_ps uses round-to-nearest-even,
     * which can round UP past a power of 2. This makes the float exponent
     * (and thus k) 1 too high, causing temp to be shifted too much.
     *
     * With round-to-nearest, the exponent can only stay correct or increase
     * by 1 (never decrease). So k is either correct or 1 too high.
     *
     * To detect: for the correct k, abs_o has its MSB at bit position
     * (14 + k), so (abs_o >> k) has bit 14 set. If k is 1 too high,
     * (abs_o >> k) has bit 14 clear (MSB at bit 13 instead).
     */
    __m256i unrounded = _mm256_srlv_epi32(abs_o, k);
    __m256i bit14 = _mm256_and_si256(unrounded, _mm256_set1_epi32(1 << 14));
    /* k is too high where bit 14 is NOT set AND k > 0 */
    __m256i bit14_set = _mm256_cmpeq_epi32(bit14, _mm256_set1_epi32(1 << 14));
    __m256i k_too_high = _mm256_andnot_si256(bit14_set, k_nonzero);
    /* Where k is too high: decrement k by 1 and recompute temp */
    __m256i k_dec = _mm256_sub_epi32(k, v_one);
    __m256i k_dec_m1 = _mm256_max_epi32(_mm256_sub_epi32(k_dec, v_one), v_zero);
    __m256i round_dec = _mm256_sllv_epi32(v_one, k_dec_m1);
    __m256i k_dec_nz = _mm256_cmpgt_epi32(k_dec, v_zero);
    round_dec = _mm256_and_si256(round_dec, k_dec_nz);
    __m256i temp_dec = _mm256_srlv_epi32(_mm256_add_epi32(abs_o, round_dec), k_dec);
    k = _mm256_blendv_epi8(k, k_dec, k_too_high);
    temp = _mm256_blendv_epi8(temp, temp_dec, k_too_high);

    /* For abs_o < 32768, temp = abs_o (no shifting) */
    temp = _mm256_blendv_epi8(temp, abs_o, small_mask);

    /* Table lookup: div_lookup[temp + 32768] */
    __m256i idx = _mm256_add_epi32(temp, v_32768);
    __m256i looked_up = _mm256_i32gather_epi32(div_lookup_ptr, idx, 4);

    *out_div = looked_up;
    *out_shift = k;
}

/*
 * Compute tmp_k for one band with variable shift:
 *   tmp_k = (div * t * sign + (1 << (14 + shift))) >> (15 + shift)
 *
 * where div = div_lookup[msb+32768], t = distorted band value,
 * sign = +1 or -1 based on reference sign, shift = get_best15_from32 shift.
 *
 * This is a 64-bit operation. We process in two halves of 4 elements each,
 * using _mm_mul_epi32 for the signed 32x32->64 multiply.
 *
 * For o == 0, the caller will blend in 32768 afterwards.
 */
static inline __m256i compute_tmp_k_s123(
    __m256i div_val,   /* 8 x int32: div_lookup values */
    __m256i t,         /* 8 x int32: distorted band values */
    __m256i neg_mask,  /* 8 x int32: all-1s where ref < 0 */
    __m256i shift)     /* 8 x int32: per-element shift amounts */
{
    /*
     * In the C reference:
     *   result = ((int64_t)div_lookup[msb+32768] * t * sign + (1 << (14+shift))) >> (15+shift)
     *
     * sign is +1 when ref >= 0, -1 when ref < 0.
     * Equivalently: negate div_val where ref < 0, then multiply by t.
     *   signed_div = (ref < 0) ? -div_val : div_val
     *   result = (signed_div * t + round) >> total_shift
     */
    /* Apply sign: negate div_val where neg_mask is set */
    __m256i neg_div = _mm256_sub_epi32(_mm256_setzero_si256(), div_val);
    __m256i signed_div = _mm256_blendv_epi8(div_val, neg_div, neg_mask);

    /* total_shift = 15 + shift */
    __m256i total_shift = _mm256_add_epi32(_mm256_set1_epi32(15), shift);

    /*
     * Compute round = 1 << (14 + shift) = 1 << (total_shift - 1)
     * For 64-bit: we need this as int64. Since total_shift <= 15+17 = 32,
     * (total_shift - 1) <= 31, so the round value fits in int32, and we
     * can sign-extend to int64.
     */
    __m256i ts_minus1 = _mm256_sub_epi32(total_shift, _mm256_set1_epi32(1));
    __m256i round32 = _mm256_sllv_epi32(_mm256_set1_epi32(1), ts_minus1);

    /* Process in two 128-bit halves */
    __m128i sd_lo = _mm256_castsi256_si128(signed_div);
    __m128i sd_hi = _mm256_extracti128_si256(signed_div, 1);
    __m128i t_lo  = _mm256_castsi256_si128(t);
    __m128i t_hi  = _mm256_extracti128_si256(t, 1);
    __m128i r_lo  = _mm256_castsi256_si128(round32);
    __m128i r_hi  = _mm256_extracti128_si256(round32, 1);
    __m128i ts_lo = _mm256_castsi256_si128(total_shift);
    __m128i ts_hi = _mm256_extracti128_si256(total_shift, 1);

    /*
     * For each 128-bit half, we need to multiply elements 0,1,2,3 and
     * do a variable arithmetic right shift. _mm_mul_epi32 does elements 0,2
     * (treating them as pairs of [low32, ignored] in each 64-bit lane).
     * We shift inputs right by 4 bytes to access elements 1,3.
     */

    /* --- Low half (elements 0-3) --- */
    /* Products for elements 0,2 */
    __m128i prod_02_lo = _mm_mul_epi32(sd_lo, t_lo);
    /* Products for elements 1,3 */
    __m128i prod_13_lo = _mm_mul_epi32(
        _mm_srli_si128(sd_lo, 4), _mm_srli_si128(t_lo, 4));

    /* Add rounding: need round as int64 */
    /* round32 elements are in positions 0,1,2,3 of the 128-bit register.
     * For elements 0,2: we need round[0] as int64 in lane 0, round[2] as int64 in lane 1.
     * _mm_cvtepi32_epi64 sign-extends elements 0,1 of the input.
     * For elements 0,2: shuffle to put element 2 next to element 0 first. */
    __m128i r02_lo_64 = _mm_cvtepi32_epi64(
        _mm_shuffle_epi32(r_lo, _MM_SHUFFLE(2, 0, 2, 0)));
    __m128i r13_lo_64 = _mm_cvtepi32_epi64(
        _mm_shuffle_epi32(r_lo, _MM_SHUFFLE(3, 1, 3, 1)));

    __m128i sum_02_lo = _mm_add_epi64(prod_02_lo, r02_lo_64);
    __m128i sum_13_lo = _mm_add_epi64(prod_13_lo, r13_lo_64);

    /* Variable arithmetic right shift by total_shift.
     * AVX2 has _mm_srlv_epi64 (logical) but no _mm_srav_epi64 (arithmetic).
     * We implement arithmetic right shift as:
     *   result = (val >> shift) | (sign_extension)
     * where sign_extension fills in the top bits with the sign bit.
     *
     * Extract per-element shift values as 64-bit. */
    __m128i ts02_lo_64 = _mm_cvtepi32_epi64(
        _mm_shuffle_epi32(ts_lo, _MM_SHUFFLE(2, 0, 2, 0)));
    __m128i ts13_lo_64 = _mm_cvtepi32_epi64(
        _mm_shuffle_epi32(ts_lo, _MM_SHUFFLE(3, 1, 3, 1)));

    /* Arithmetic right shift for 64-bit:
     * logical_shift = val >>> shift
     * sign_bits = val >> 63  (all 0s or all 1s)
     * sign_fill = sign_bits << (64 - shift)   (but only if shift > 0)
     * result = logical_shift | sign_fill */
    __m128i lsh_02_lo = _mm_srlv_epi64(sum_02_lo, ts02_lo_64);
    __m128i sign_02_lo = _mm_srai_epi32(
        _mm_shuffle_epi32(sum_02_lo, 0xF5), 31); /* broadcast high word sign */
    sign_02_lo = _mm_shuffle_epi32(sign_02_lo, 0xF5);
    __m128i fill_02_lo = _mm_sllv_epi64(sign_02_lo,
        _mm_sub_epi64(_mm_set1_epi64x(64), ts02_lo_64));
    __m128i res_02_lo = _mm_or_si128(lsh_02_lo, fill_02_lo);

    __m128i lsh_13_lo = _mm_srlv_epi64(sum_13_lo, ts13_lo_64);
    __m128i sign_13_lo = _mm_srai_epi32(
        _mm_shuffle_epi32(sum_13_lo, 0xF5), 31);
    sign_13_lo = _mm_shuffle_epi32(sign_13_lo, 0xF5);
    __m128i fill_13_lo = _mm_sllv_epi64(sign_13_lo,
        _mm_sub_epi64(_mm_set1_epi64x(64), ts13_lo_64));
    __m128i res_13_lo = _mm_or_si128(lsh_13_lo, fill_13_lo);

    /* Pack results back to int32: take low 32 bits of each 64-bit result.
     * res_02 has [result0, X, result2, X], res_13 has [result1, X, result3, X]
     * Shuffle to get [r0, r2, r1, r3] then reorder to [r0, r1, r2, r3]. */
    __m128i pack_lo = _mm_castps_si128(_mm_shuffle_ps(
        _mm_castsi128_ps(res_02_lo), _mm_castsi128_ps(res_13_lo),
        _MM_SHUFFLE(2, 0, 2, 0)));
    pack_lo = _mm_shuffle_epi32(pack_lo, _MM_SHUFFLE(3, 1, 2, 0));

    /* --- High half (elements 4-7) --- */
    __m128i prod_02_hi = _mm_mul_epi32(sd_hi, t_hi);
    __m128i prod_13_hi = _mm_mul_epi32(
        _mm_srli_si128(sd_hi, 4), _mm_srli_si128(t_hi, 4));

    __m128i r02_hi_64 = _mm_cvtepi32_epi64(
        _mm_shuffle_epi32(r_hi, _MM_SHUFFLE(2, 0, 2, 0)));
    __m128i r13_hi_64 = _mm_cvtepi32_epi64(
        _mm_shuffle_epi32(r_hi, _MM_SHUFFLE(3, 1, 3, 1)));

    __m128i sum_02_hi = _mm_add_epi64(prod_02_hi, r02_hi_64);
    __m128i sum_13_hi = _mm_add_epi64(prod_13_hi, r13_hi_64);

    __m128i ts02_hi_64 = _mm_cvtepi32_epi64(
        _mm_shuffle_epi32(ts_hi, _MM_SHUFFLE(2, 0, 2, 0)));
    __m128i ts13_hi_64 = _mm_cvtepi32_epi64(
        _mm_shuffle_epi32(ts_hi, _MM_SHUFFLE(3, 1, 3, 1)));

    __m128i lsh_02_hi = _mm_srlv_epi64(sum_02_hi, ts02_hi_64);
    __m128i sign_02_hi = _mm_srai_epi32(
        _mm_shuffle_epi32(sum_02_hi, 0xF5), 31);
    sign_02_hi = _mm_shuffle_epi32(sign_02_hi, 0xF5);
    __m128i fill_02_hi = _mm_sllv_epi64(sign_02_hi,
        _mm_sub_epi64(_mm_set1_epi64x(64), ts02_hi_64));
    __m128i res_02_hi = _mm_or_si128(lsh_02_hi, fill_02_hi);

    __m128i lsh_13_hi = _mm_srlv_epi64(sum_13_hi, ts13_hi_64);
    __m128i sign_13_hi = _mm_srai_epi32(
        _mm_shuffle_epi32(sum_13_hi, 0xF5), 31);
    sign_13_hi = _mm_shuffle_epi32(sign_13_hi, 0xF5);
    __m128i fill_13_hi = _mm_sllv_epi64(sign_13_hi,
        _mm_sub_epi64(_mm_set1_epi64x(64), ts13_hi_64));
    __m128i res_13_hi = _mm_or_si128(lsh_13_hi, fill_13_hi);

    __m128i pack_hi = _mm_castps_si128(_mm_shuffle_ps(
        _mm_castsi128_ps(res_02_hi), _mm_castsi128_ps(res_13_hi),
        _MM_SHUFFLE(2, 0, 2, 0)));
    pack_hi = _mm_shuffle_epi32(pack_hi, _MM_SHUFFLE(3, 1, 2, 0));

    return _mm256_setr_m128i(pack_lo, pack_hi);
}

/*
 * Compute rst = (k * o + 16384) >> 15 where k and o are int32,
 * and the product can exceed int32 range, requiring 64-bit arithmetic.
 * k is clamped to [0, 32768], o is int32.  k*o can be up to 32768 * 2^31
 * which overflows int32 but fits in int64.
 *
 * We process 8 elements, returning 8 int32 results.
 */
static inline __m256i compute_rst_s123(__m256i k, __m256i o) {
    __m128i k_lo = _mm256_castsi256_si128(k);
    __m128i k_hi = _mm256_extracti128_si256(k, 1);
    __m128i o_lo = _mm256_castsi256_si128(o);
    __m128i o_hi = _mm256_extracti128_si256(o, 1);
    __m128i v_16384_128 = _mm_set1_epi64x(16384);

    /* Low half elements 0,2 and 1,3 */
    __m128i prod_02_lo = _mm_mul_epi32(k_lo, o_lo);
    __m128i prod_13_lo = _mm_mul_epi32(
        _mm_srli_si128(k_lo, 4), _mm_srli_si128(o_lo, 4));

    __m128i r02_lo = srai_epi64_15(_mm_add_epi64(prod_02_lo, v_16384_128));
    __m128i r13_lo = srai_epi64_15(_mm_add_epi64(prod_13_lo, v_16384_128));

    __m128i pack_lo = _mm_castps_si128(_mm_shuffle_ps(
        _mm_castsi128_ps(r02_lo), _mm_castsi128_ps(r13_lo),
        _MM_SHUFFLE(2, 0, 2, 0)));
    pack_lo = _mm_shuffle_epi32(pack_lo, _MM_SHUFFLE(3, 1, 2, 0));

    /* High half elements 4,6 and 5,7 */
    __m128i prod_02_hi = _mm_mul_epi32(k_hi, o_hi);
    __m128i prod_13_hi = _mm_mul_epi32(
        _mm_srli_si128(k_hi, 4), _mm_srli_si128(o_hi, 4));

    __m128i r02_hi = srai_epi64_15(_mm_add_epi64(prod_02_hi, v_16384_128));
    __m128i r13_hi = srai_epi64_15(_mm_add_epi64(prod_13_hi, v_16384_128));

    __m128i pack_hi = _mm_castps_si128(_mm_shuffle_ps(
        _mm_castsi128_ps(r02_hi), _mm_castsi128_ps(r13_hi),
        _MM_SHUFFLE(2, 0, 2, 0)));
    pack_hi = _mm_shuffle_epi32(pack_hi, _MM_SHUFFLE(3, 1, 2, 0));

    return _mm256_setr_m128i(pack_lo, pack_hi);
}

/*
 * Apply enhancement gain limit for one band (s123 variant).
 *
 * C reference pattern:
 *   rst_f = (k / 32768.0f) * (o / 64.0f)
 *   if (angle_flag && rst_f > 0)  rst = MIN(rst * gain_limit, t)
 *   if (angle_flag && rst_f < 0)  rst = MAX(rst * gain_limit, t)
 *
 * k, o, t are int32; rst is int32 (modified in place).
 */
static inline __m256i apply_gain_limit_s123(
    __m256i rst, __m256i k, __m256i o, __m256i t,
    __m256 angle_mask_f, __m256 v_gain)
{
    __m256 k_f = _mm256_mul_ps(_mm256_cvtepi32_ps(k), _mm256_set1_ps(1.0f / 32768.0f));
    __m256 o_f = _mm256_mul_ps(_mm256_cvtepi32_ps(o), _mm256_set1_ps(1.0f / 64.0f));
    __m256 rst_f = _mm256_mul_ps(k_f, o_f);

    __m256 rst_ps = _mm256_cvtepi32_ps(rst);
    __m256 rst_scaled = _mm256_mul_ps(rst_ps, v_gain);
    __m256 t_f = _mm256_cvtepi32_ps(t);

    __m256 v_zero_f = _mm256_setzero_ps();
    __m256 pos_mask = _mm256_cmp_ps(rst_f, v_zero_f, _CMP_GT_OQ);
    __m256 neg_mask = _mm256_cmp_ps(rst_f, v_zero_f, _CMP_LT_OQ);

    __m256 min_val = _mm256_min_ps(rst_scaled, t_f);
    __m256 max_val = _mm256_max_ps(rst_scaled, t_f);

    __m256 current = rst_ps;
    current = _mm256_blendv_ps(current, min_val,
        _mm256_and_ps(angle_mask_f, pos_mask));
    current = _mm256_blendv_ps(current, max_val,
        _mm256_and_ps(angle_mask_f, neg_mask));

    return _mm256_cvttps_epi32(current);
}

void adm_decouple_s123_avx2(AdmBuffer *buf, int w, int h, int stride,
                             double adm_enhn_gain_limit)
{
    const float cos_1deg_sq = (float)(cos(1.0 * M_PI / 180.0) *
                                      cos(1.0 * M_PI / 180.0));

    const i4_adm_dwt_band_t *ref = &buf->i4_ref_dwt2;
    const i4_adm_dwt_band_t *dis = &buf->i4_dis_dwt2;
    const i4_adm_dwt_band_t *r   = &buf->i4_decouple_r;
    const i4_adm_dwt_band_t *a   = &buf->i4_decouple_a;

    int left   = w * ADM_BORDER_FACTOR - 0.5 - 1;
    int top    = h * ADM_BORDER_FACTOR - 0.5 - 1;
    int right  = w - left + 2;
    int bottom = h - top + 2;

    if (left < 0)    left = 0;
    if (right > w)   right = w;
    if (top < 0)     top = 0;
    if (bottom > h)  bottom = h;

    /* Pointer to the file-scope div_lookup table.
     * div_lookup is declared static in the header and generated once at init. */
    const int32_t *div_lookup_ptr = div_lookup;

    for (int i = top; i < bottom; ++i) {
        int j = left;

        /* Process 8 int32 elements per iteration */
        for (; j + 7 < right; j += 8) {
            const int idx = i * stride + j;

            /* Load 8 x int32 for each band */
            __m256i oh = _mm256_loadu_si256((const __m256i *)(ref->band_h + idx));
            __m256i ov = _mm256_loadu_si256((const __m256i *)(ref->band_v + idx));
            __m256i od = _mm256_loadu_si256((const __m256i *)(ref->band_d + idx));
            __m256i th = _mm256_loadu_si256((const __m256i *)(dis->band_h + idx));
            __m256i tv = _mm256_loadu_si256((const __m256i *)(dis->band_v + idx));
            __m256i td = _mm256_loadu_si256((const __m256i *)(dis->band_d + idx));

            /*
             * Angle flag computation.
             *
             * C reference (s123):
             *   ot_dp    = (int64_t)oh*th + (int64_t)ov*tv
             *   o_mag_sq = (int64_t)oh*oh + (int64_t)ov*ov
             *   t_mag_sq = (int64_t)th*th + (int64_t)tv*tv
             *
             *   angle_flag = ((float)ot_dp / 4096.0 >= 0.0f) &&
             *     ((float)ot_dp/4096.0 * (float)ot_dp/4096.0 >=
             *      cos_1deg_sq * (float)o_mag_sq/4096.0 * (float)t_mag_sq/4096.0)
             *
             * Note: (float)int64_val loses precision for large values, then
             * / 4096.0 promotes to double. We must match this exactly.
             *
             * Since int32*int32 can overflow int32, we must use 64-bit products.
             * We process element-by-element in batches of 2 using _mm_mul_epi32,
             * convert to float (matching C's (float) cast of int64), then to double.
             *
             * However, for exact matching of the C cast sequence
             * (int64 -> float -> double -> /4096.0), we need to be careful.
             * We compute the int64 sums, convert to float (losing precision just
             * like C does), then to double.
             *
             * We process 4 elements at a time (two batches of 4 for the 8-wide vector).
             */

            /* For the angle flag, we need per-element int64 dot products.
             * Process in 4 groups of 2 elements each. */
            __m128i oh_lo = _mm256_castsi256_si128(oh);
            __m128i oh_hi = _mm256_extracti128_si256(oh, 1);
            __m128i ov_lo = _mm256_castsi256_si128(ov);
            __m128i ov_hi = _mm256_extracti128_si256(ov, 1);
            __m128i th_lo = _mm256_castsi256_si128(th);
            __m128i th_hi = _mm256_extracti128_si256(th, 1);
            __m128i tv_lo = _mm256_castsi256_si128(tv);
            __m128i tv_hi = _mm256_extracti128_si256(tv, 1);

            /*
             * _mm_mul_epi32 multiplies elements at positions 0 and 2 (the low
             * 32 bits of each 64-bit lane), producing two int64 results.
             * To get elements 1 and 3, we shift the register right by 4 bytes.
             */

            /* ot_dp for elements 0,2 of low half */
            __m128i dp_02_lo = _mm_add_epi64(
                _mm_mul_epi32(oh_lo, th_lo),
                _mm_mul_epi32(ov_lo, tv_lo));
            /* ot_dp for elements 1,3 of low half */
            __m128i dp_13_lo = _mm_add_epi64(
                _mm_mul_epi32(_mm_srli_si128(oh_lo, 4), _mm_srli_si128(th_lo, 4)),
                _mm_mul_epi32(_mm_srli_si128(ov_lo, 4), _mm_srli_si128(tv_lo, 4)));
            /* ot_dp for elements 0,2 of high half */
            __m128i dp_02_hi = _mm_add_epi64(
                _mm_mul_epi32(oh_hi, th_hi),
                _mm_mul_epi32(ov_hi, tv_hi));
            /* ot_dp for elements 1,3 of high half */
            __m128i dp_13_hi = _mm_add_epi64(
                _mm_mul_epi32(_mm_srli_si128(oh_hi, 4), _mm_srli_si128(th_hi, 4)),
                _mm_mul_epi32(_mm_srli_si128(ov_hi, 4), _mm_srli_si128(tv_hi, 4)));

            /* o_mag_sq */
            __m128i omag_02_lo = _mm_add_epi64(
                _mm_mul_epi32(oh_lo, oh_lo),
                _mm_mul_epi32(ov_lo, ov_lo));
            __m128i omag_13_lo = _mm_add_epi64(
                _mm_mul_epi32(_mm_srli_si128(oh_lo, 4), _mm_srli_si128(oh_lo, 4)),
                _mm_mul_epi32(_mm_srli_si128(ov_lo, 4), _mm_srli_si128(ov_lo, 4)));
            __m128i omag_02_hi = _mm_add_epi64(
                _mm_mul_epi32(oh_hi, oh_hi),
                _mm_mul_epi32(ov_hi, ov_hi));
            __m128i omag_13_hi = _mm_add_epi64(
                _mm_mul_epi32(_mm_srli_si128(oh_hi, 4), _mm_srli_si128(oh_hi, 4)),
                _mm_mul_epi32(_mm_srli_si128(ov_hi, 4), _mm_srli_si128(ov_hi, 4)));

            /* t_mag_sq */
            __m128i tmag_02_lo = _mm_add_epi64(
                _mm_mul_epi32(th_lo, th_lo),
                _mm_mul_epi32(tv_lo, tv_lo));
            __m128i tmag_13_lo = _mm_add_epi64(
                _mm_mul_epi32(_mm_srli_si128(th_lo, 4), _mm_srli_si128(th_lo, 4)),
                _mm_mul_epi32(_mm_srli_si128(tv_lo, 4), _mm_srli_si128(tv_lo, 4)));
            __m128i tmag_02_hi = _mm_add_epi64(
                _mm_mul_epi32(th_hi, th_hi),
                _mm_mul_epi32(tv_hi, tv_hi));
            __m128i tmag_13_hi = _mm_add_epi64(
                _mm_mul_epi32(_mm_srli_si128(th_hi, 4), _mm_srli_si128(th_hi, 4)),
                _mm_mul_epi32(_mm_srli_si128(tv_hi, 4), _mm_srli_si128(tv_hi, 4)));

            /*
             * Now we have 8 sets of (ot_dp, o_mag_sq, t_mag_sq) as int64 pairs.
             * The C code does: (float)ot_dp / 4096.0
             * We need to convert int64 -> float (matching C's cast), then -> double.
             *
             * SIMD int64→double via magic number trick: for signed int64 in
             * [-2^51, 2^51), add 3×2^51 as int64, reinterpret as double,
             * subtract 3×2^51 as double. This gives an exact conversion.
             * Then double→float via _mm_cvtpd_ps matches C's (float) cast rounding.
             * Then float→double via _mm256_cvtps_pd matches C's implicit promotion.
             *
             * Values here are products of ~14-bit DWT coefficients: max ~2^30,
             * well within the 2^51 safe range.
             */
            __m128i magic_i = _mm_set1_epi64x(0x4338000000000000LL);
            __m128d magic_d = _mm_set1_pd(6755399441055744.0); /* 3 * 2^51 */

            /* dp: int64 → double (exact) → float (rounded) → ordered __m128 */
            __m128 dp_f_lo = _mm_unpacklo_ps(
                _mm_cvtpd_ps(_mm_sub_pd(_mm_castsi128_pd(_mm_add_epi64(dp_02_lo, magic_i)), magic_d)),
                _mm_cvtpd_ps(_mm_sub_pd(_mm_castsi128_pd(_mm_add_epi64(dp_13_lo, magic_i)), magic_d)));
            __m128 dp_f_hi = _mm_unpacklo_ps(
                _mm_cvtpd_ps(_mm_sub_pd(_mm_castsi128_pd(_mm_add_epi64(dp_02_hi, magic_i)), magic_d)),
                _mm_cvtpd_ps(_mm_sub_pd(_mm_castsi128_pd(_mm_add_epi64(dp_13_hi, magic_i)), magic_d)));

            /* omag: same conversion */
            __m128 omag_f_lo = _mm_unpacklo_ps(
                _mm_cvtpd_ps(_mm_sub_pd(_mm_castsi128_pd(_mm_add_epi64(omag_02_lo, magic_i)), magic_d)),
                _mm_cvtpd_ps(_mm_sub_pd(_mm_castsi128_pd(_mm_add_epi64(omag_13_lo, magic_i)), magic_d)));
            __m128 omag_f_hi = _mm_unpacklo_ps(
                _mm_cvtpd_ps(_mm_sub_pd(_mm_castsi128_pd(_mm_add_epi64(omag_02_hi, magic_i)), magic_d)),
                _mm_cvtpd_ps(_mm_sub_pd(_mm_castsi128_pd(_mm_add_epi64(omag_13_hi, magic_i)), magic_d)));

            /* tmag: same conversion */
            __m128 tmag_f_lo = _mm_unpacklo_ps(
                _mm_cvtpd_ps(_mm_sub_pd(_mm_castsi128_pd(_mm_add_epi64(tmag_02_lo, magic_i)), magic_d)),
                _mm_cvtpd_ps(_mm_sub_pd(_mm_castsi128_pd(_mm_add_epi64(tmag_13_lo, magic_i)), magic_d)));
            __m128 tmag_f_hi = _mm_unpacklo_ps(
                _mm_cvtpd_ps(_mm_sub_pd(_mm_castsi128_pd(_mm_add_epi64(tmag_02_hi, magic_i)), magic_d)),
                _mm_cvtpd_ps(_mm_sub_pd(_mm_castsi128_pd(_mm_add_epi64(tmag_13_hi, magic_i)), magic_d)));

            /* float → double (matching C's implicit promotion for /4096.0) */
            __m256d dp_d_lo   = _mm256_cvtps_pd(dp_f_lo);
            __m256d dp_d_hi   = _mm256_cvtps_pd(dp_f_hi);
            __m256d omag_d_lo = _mm256_cvtps_pd(omag_f_lo);
            __m256d omag_d_hi = _mm256_cvtps_pd(omag_f_hi);
            __m256d tmag_d_lo = _mm256_cvtps_pd(tmag_f_lo);
            __m256d tmag_d_hi = _mm256_cvtps_pd(tmag_f_hi);

            __m256d v_inv_4096_d = _mm256_set1_pd(1.0 / 4096.0);
            __m256d v_zero_d = _mm256_setzero_pd();
            __m256d v_cos_d  = _mm256_set1_pd((double)cos_1deg_sq);

            __m256d dp_lo = _mm256_mul_pd(dp_d_lo, v_inv_4096_d);
            __m256d dp_hi = _mm256_mul_pd(dp_d_hi, v_inv_4096_d);
            __m256d om_lo = _mm256_mul_pd(omag_d_lo, v_inv_4096_d);
            __m256d om_hi = _mm256_mul_pd(omag_d_hi, v_inv_4096_d);
            __m256d tm_lo = _mm256_mul_pd(tmag_d_lo, v_inv_4096_d);
            __m256d tm_hi = _mm256_mul_pd(tmag_d_hi, v_inv_4096_d);

            /* cond1: dp >= 0 */
            __m256d c1_lo = _mm256_cmp_pd(dp_lo, v_zero_d, _CMP_GE_OQ);
            __m256d c1_hi = _mm256_cmp_pd(dp_hi, v_zero_d, _CMP_GE_OQ);

            /* cond2: dp^2 >= cos_1deg_sq * omag * tmag */
            __m256d dp_sq_lo = _mm256_mul_pd(dp_lo, dp_lo);
            __m256d dp_sq_hi = _mm256_mul_pd(dp_hi, dp_hi);
            __m256d mp_lo = _mm256_mul_pd(_mm256_mul_pd(v_cos_d, om_lo), tm_lo);
            __m256d mp_hi = _mm256_mul_pd(_mm256_mul_pd(v_cos_d, om_hi), tm_hi);
            __m256d c2_lo = _mm256_cmp_pd(dp_sq_lo, mp_lo, _CMP_GE_OQ);
            __m256d c2_hi = _mm256_cmp_pd(dp_sq_hi, mp_hi, _CMP_GE_OQ);

            __m256d angle_lo = _mm256_and_pd(c1_lo, c2_lo);
            __m256d angle_hi = _mm256_and_pd(c1_hi, c2_hi);

            int mask_lo = _mm256_movemask_pd(angle_lo);
            int mask_hi_val = _mm256_movemask_pd(angle_hi);
            int full_mask = mask_lo | (mask_hi_val << 4);

            __m256i angle_mask = _mm256_set_epi32(
                (full_mask & 0x80) ? -1 : 0,
                (full_mask & 0x40) ? -1 : 0,
                (full_mask & 0x20) ? -1 : 0,
                (full_mask & 0x10) ? -1 : 0,
                (full_mask & 0x08) ? -1 : 0,
                (full_mask & 0x04) ? -1 : 0,
                (full_mask & 0x02) ? -1 : 0,
                (full_mask & 0x01) ? -1 : 0
            );
            __m256 angle_mask_f = _mm256_castsi256_ps(angle_mask);

            /*
             * Division via get_best15_from32 + table lookup.
             */
            __m256i v_zero  = _mm256_setzero_si256();
            __m256i v_32768 = _mm256_set1_epi32(32768);

            /* --- band_h --- */
            __m256i div_h, shift_h, neg_h;
            get_best15_gather_avx2(oh, div_lookup_ptr, &div_h, &shift_h, &neg_h);
            __m256i tmp_kh = compute_tmp_k_s123(div_h, th, neg_h, shift_h);

            /* Handle oh == 0: use 32768 */
            __m256i oh_zero = _mm256_cmpeq_epi32(oh, v_zero);
            tmp_kh = _mm256_blendv_epi8(tmp_kh, v_32768, oh_zero);

            /* Clamp to [0, 32768] */
            __m256i kh = _mm256_max_epi32(tmp_kh, v_zero);
            kh = _mm256_min_epi32(kh, v_32768);

            /* rst_h = (kh * oh + 16384) >> 15 (64-bit) */
            __m256i rst_h = compute_rst_s123(kh, oh);

            /* --- band_v --- */
            __m256i div_v, shift_v, neg_v;
            get_best15_gather_avx2(ov, div_lookup_ptr, &div_v, &shift_v, &neg_v);
            __m256i tmp_kv = compute_tmp_k_s123(div_v, tv, neg_v, shift_v);

            __m256i ov_zero = _mm256_cmpeq_epi32(ov, v_zero);
            tmp_kv = _mm256_blendv_epi8(tmp_kv, v_32768, ov_zero);

            __m256i kv = _mm256_max_epi32(tmp_kv, v_zero);
            kv = _mm256_min_epi32(kv, v_32768);

            __m256i rst_v = compute_rst_s123(kv, ov);

            /* --- band_d --- */
            __m256i div_d, shift_d, neg_d;
            get_best15_gather_avx2(od, div_lookup_ptr, &div_d, &shift_d, &neg_d);
            __m256i tmp_kd = compute_tmp_k_s123(div_d, td, neg_d, shift_d);

            __m256i od_zero = _mm256_cmpeq_epi32(od, v_zero);
            tmp_kd = _mm256_blendv_epi8(tmp_kd, v_32768, od_zero);

            __m256i kd = _mm256_max_epi32(tmp_kd, v_zero);
            kd = _mm256_min_epi32(kd, v_32768);

            __m256i rst_d = compute_rst_s123(kd, od);

            /*
             * Enhancement gain limit.
             */
            __m256 v_gain = _mm256_set1_ps((float)adm_enhn_gain_limit);

            rst_h = apply_gain_limit_s123(rst_h, kh, oh, th,
                angle_mask_f, v_gain);
            rst_v = apply_gain_limit_s123(rst_v, kv, ov, tv,
                angle_mask_f, v_gain);
            rst_d = apply_gain_limit_s123(rst_d, kd, od, td,
                angle_mask_f, v_gain);

            /* Store int32 results directly (no packing needed for i4 path) */
            _mm256_storeu_si256((__m256i *)(r->band_h + idx), rst_h);
            _mm256_storeu_si256((__m256i *)(r->band_v + idx), rst_v);
            _mm256_storeu_si256((__m256i *)(r->band_d + idx), rst_d);

            _mm256_storeu_si256((__m256i *)(a->band_h + idx),
                _mm256_sub_epi32(th, rst_h));
            _mm256_storeu_si256((__m256i *)(a->band_v + idx),
                _mm256_sub_epi32(tv, rst_v));
            _mm256_storeu_si256((__m256i *)(a->band_d + idx),
                _mm256_sub_epi32(td, rst_d));
        }

        /* Scalar tail for remaining elements */
        for (; j < right; ++j) {
            const int idx = i * stride + j;

            int32_t oh_s = ref->band_h[idx];
            int32_t ov_s = ref->band_v[idx];
            int32_t od_s = ref->band_d[idx];
            int32_t th_s = dis->band_h[idx];
            int32_t tv_s = dis->band_v[idx];
            int32_t td_s = dis->band_d[idx];
            int32_t rst_h_s, rst_v_s, rst_d_s;

            int64_t ot_dp    = (int64_t)oh_s * th_s + (int64_t)ov_s * tv_s;
            int64_t o_mag_sq = (int64_t)oh_s * oh_s + (int64_t)ov_s * ov_s;
            int64_t t_mag_sq = (int64_t)th_s * th_s + (int64_t)tv_s * tv_s;

            int angle_flag = (((float)ot_dp / 4096.0) >= 0.0f) &&
                (((float)ot_dp / 4096.0) * ((float)ot_dp / 4096.0) >=
                    cos_1deg_sq * ((float)o_mag_sq / 4096.0) *
                    ((float)t_mag_sq / 4096.0));

            int32_t kh_shift = 0, kv_shift = 0, kd_shift = 0;

            uint32_t abs_oh = abs(oh_s);
            uint32_t abs_ov = abs(ov_s);
            uint32_t abs_od = abs(od_s);

            int8_t kh_sign = (oh_s < 0 ? -1 : 1);
            int8_t kv_sign = (ov_s < 0 ? -1 : 1);
            int8_t kd_sign = (od_s < 0 ? -1 : 1);

            uint16_t kh_msb, kv_msb, kd_msb;
            if (abs_oh < 32768) { kh_msb = abs_oh; }
            else { int k = __builtin_clz(abs_oh); k = 17 - k;
                   abs_oh = (abs_oh + (1 << (k - 1))) >> k; kh_shift = k; kh_msb = abs_oh; }
            if (abs_ov < 32768) { kv_msb = abs_ov; }
            else { int k = __builtin_clz(abs_ov); k = 17 - k;
                   abs_ov = (abs_ov + (1 << (k - 1))) >> k; kv_shift = k; kv_msb = abs_ov; }
            if (abs_od < 32768) { kd_msb = abs_od; }
            else { int k = __builtin_clz(abs_od); k = 17 - k;
                   abs_od = (abs_od + (1 << (k - 1))) >> k; kd_shift = k; kd_msb = abs_od; }

            int64_t tmp_kh_s = (oh_s == 0) ? 32768 :
                (((int64_t)div_lookup_ptr[kh_msb + 32768] * th_s) * kh_sign +
                 (1 << (14 + kh_shift))) >> (15 + kh_shift);
            int64_t tmp_kv_s = (ov_s == 0) ? 32768 :
                (((int64_t)div_lookup_ptr[kv_msb + 32768] * tv_s) * kv_sign +
                 (1 << (14 + kv_shift))) >> (15 + kv_shift);
            int64_t tmp_kd_s = (od_s == 0) ? 32768 :
                (((int64_t)div_lookup_ptr[kd_msb + 32768] * td_s) * kd_sign +
                 (1 << (14 + kd_shift))) >> (15 + kd_shift);

            int64_t kh_s = tmp_kh_s < 0 ? 0 : (tmp_kh_s > 32768 ? 32768 : tmp_kh_s);
            int64_t kv_s = tmp_kv_s < 0 ? 0 : (tmp_kv_s > 32768 ? 32768 : tmp_kv_s);
            int64_t kd_s = tmp_kd_s < 0 ? 0 : (tmp_kd_s > 32768 ? 32768 : tmp_kd_s);

            rst_h_s = ((kh_s * oh_s) + 16384) >> 15;
            rst_v_s = ((kv_s * ov_s) + 16384) >> 15;
            rst_d_s = ((kd_s * od_s) + 16384) >> 15;

            const float rst_h_f = ((float)kh_s / 32768) * ((float)oh_s / 64);
            const float rst_v_f = ((float)kv_s / 32768) * ((float)ov_s / 64);
            const float rst_d_f = ((float)kd_s / 32768) * ((float)od_s / 64);

            if (angle_flag && (rst_h_f > 0.))
                rst_h_s = MIN((rst_h_s * adm_enhn_gain_limit), th_s);
            if (angle_flag && (rst_h_f < 0.))
                rst_h_s = MAX((rst_h_s * adm_enhn_gain_limit), th_s);

            if (angle_flag && (rst_v_f > 0.))
                rst_v_s = MIN(rst_v_s * adm_enhn_gain_limit, tv_s);
            if (angle_flag && (rst_v_f < 0.))
                rst_v_s = MAX(rst_v_s * adm_enhn_gain_limit, tv_s);

            if (angle_flag && (rst_d_f > 0.))
                rst_d_s = MIN(rst_d_s * adm_enhn_gain_limit, td_s);
            if (angle_flag && (rst_d_f < 0.))
                rst_d_s = MAX(rst_d_s * adm_enhn_gain_limit, td_s);

            r->band_h[idx] = rst_h_s;
            r->band_v[idx] = rst_v_s;
            r->band_d[idx] = rst_d_s;

            a->band_h[idx] = th_s - rst_h_s;
            a->band_v[idx] = tv_s - rst_v_s;
            a->band_d[idx] = td_s - rst_d_s;
        }
    }
}

/* ================================================================
 * Scalar threshold helper for adm_cm boundary pixels (int16 path).
 * Computes the 3x3 neighborhood sum across 3 orientations.
 * For interior pixels (1<=i<=h-2, 1<=j<=w-2).
 * ================================================================ */
static inline int32_t adm_cm_thresh_s_i_j_scalar(
    int16_t *angles[3], int16_t *flt_angles[3],
    int src_stride, int i, int j)
{
    int32_t accum = 0;
    for (int theta = 0; theta < 3; ++theta) {
        int32_t sum = 0;
        int16_t *src_ptr = angles[theta] + src_stride * (i - 1);
        int16_t *flt_ptr = flt_angles[theta] + src_stride * (i - 1);
        sum += flt_ptr[j - 1] + flt_ptr[j] + flt_ptr[j + 1];
        src_ptr += src_stride;
        flt_ptr += src_stride;
        sum += flt_ptr[j - 1] + flt_ptr[j + 1];
        sum += (int16_t)(((ONE_BY_15 * abs((int32_t)src_ptr[j])) + 2048) >> 12);
        flt_ptr += src_stride;
        sum += flt_ptr[j - 1] + flt_ptr[j] + flt_ptr[j + 1];
        accum += sum;
    }
    return accum;
}

/* Scalar boundary threshold: i=0, j=0 */
static inline int32_t adm_cm_thresh_s_0_0_scalar(
    int16_t *angles[3], int16_t *flt_angles[3], int src_stride)
{
    int32_t accum = 0;
    for (int theta = 0; theta < 3; ++theta) {
        int32_t sum = 0;
        int16_t *src_ptr = angles[theta];
        int16_t *flt_ptr = flt_angles[theta];
        sum += flt_ptr[src_stride + 1] + flt_ptr[src_stride] + flt_ptr[src_stride + 1];
        sum += flt_ptr[1];
        sum += (int16_t)(((ONE_BY_15 * abs((int32_t)src_ptr[0])) + 2048) >> 12);
        sum += flt_ptr[1];
        sum += flt_ptr[src_stride + 1] + flt_ptr[src_stride] + flt_ptr[src_stride + 1];
        accum += sum;
    }
    return accum;
}

/* Scalar boundary threshold: i=0, j=1..w-2 */
static inline int32_t adm_cm_thresh_s_0_j_scalar(
    int16_t *angles[3], int16_t *flt_angles[3], int src_stride, int j)
{
    int32_t accum = 0;
    for (int theta = 0; theta < 3; ++theta) {
        int32_t sum = 0;
        int16_t *src_ptr = angles[theta];
        int16_t *flt_ptr = flt_angles[theta];
        sum += flt_ptr[src_stride + j - 1] + flt_ptr[src_stride + j] + flt_ptr[src_stride + j + 1];
        sum += flt_ptr[j - 1];
        sum += (int16_t)(((ONE_BY_15 * abs((int32_t)src_ptr[j])) + 2048) >> 12);
        sum += flt_ptr[j + 1];
        sum += flt_ptr[src_stride + j - 1] + flt_ptr[src_stride + j] + flt_ptr[src_stride + j + 1];
        accum += sum;
    }
    return accum;
}

/* Scalar boundary threshold: i=0, j=w-1 */
static inline int32_t adm_cm_thresh_s_0_wm1_scalar(
    int16_t *angles[3], int16_t *flt_angles[3], int src_stride, int w)
{
    int32_t accum = 0;
    for (int theta = 0; theta < 3; ++theta) {
        int32_t sum = 0;
        int16_t *src_ptr = angles[theta];
        int16_t *flt_ptr = flt_angles[theta];
        sum += flt_ptr[src_stride + w - 2] + flt_ptr[src_stride + w - 1] + flt_ptr[src_stride + w - 1];
        sum += flt_ptr[w - 2];
        sum += (int16_t)(((ONE_BY_15 * abs((int32_t)src_ptr[w - 1])) + 2048) >> 12);
        sum += flt_ptr[w - 1];
        sum += flt_ptr[src_stride + w - 2] + flt_ptr[src_stride + w - 1] + flt_ptr[src_stride + w - 1];
        accum += sum;
    }
    return accum;
}

/* Scalar boundary threshold: i=1..h-2, j=0 */
static inline int32_t adm_cm_thresh_s_i_0_scalar(
    int16_t *angles[3], int16_t *flt_angles[3], int src_stride, int i)
{
    int32_t accum = 0;
    for (int theta = 0; theta < 3; ++theta) {
        int32_t sum = 0;
        int16_t *src_ptr = angles[theta] + src_stride * (i - 1);
        int16_t *flt_ptr = flt_angles[theta] + src_stride * (i - 1);
        sum += flt_ptr[1] + flt_ptr[0] + flt_ptr[1];
        src_ptr += src_stride; flt_ptr += src_stride;
        sum += flt_ptr[1];
        sum += (int16_t)(((ONE_BY_15 * abs((int32_t)src_ptr[0])) + 2048) >> 12);
        sum += flt_ptr[1];
        flt_ptr += src_stride;
        sum += flt_ptr[1] + flt_ptr[0] + flt_ptr[1];
        accum += sum;
    }
    return accum;
}

/* Scalar boundary threshold: i=1..h-2, j=w-1 */
static inline int32_t adm_cm_thresh_s_i_wm1_scalar(
    int16_t *angles[3], int16_t *flt_angles[3], int src_stride, int w, int i)
{
    int32_t accum = 0;
    for (int theta = 0; theta < 3; ++theta) {
        int32_t sum = 0;
        int16_t *src_ptr = angles[theta] + src_stride * (i - 1);
        int16_t *flt_ptr = flt_angles[theta] + src_stride * (i - 1);
        sum += flt_ptr[w - 2] + flt_ptr[w - 1] + flt_ptr[w - 1];
        src_ptr += src_stride; flt_ptr += src_stride;
        sum += flt_ptr[w - 2];
        sum += (int16_t)(((ONE_BY_15 * abs((int32_t)src_ptr[w - 1])) + 2048) >> 12);
        sum += flt_ptr[w - 1];
        flt_ptr += src_stride;
        sum += flt_ptr[w - 2] + flt_ptr[w - 1] + flt_ptr[w - 1];
        accum += sum;
    }
    return accum;
}

/* Scalar boundary threshold: i=h-1, j=0 */
static inline int32_t adm_cm_thresh_s_hm1_0_scalar(
    int16_t *angles[3], int16_t *flt_angles[3], int src_stride, int h)
{
    int32_t accum = 0;
    for (int theta = 0; theta < 3; ++theta) {
        int32_t sum = 0;
        int16_t *src_ptr = angles[theta] + src_stride * (h - 2);
        int16_t *flt_ptr = flt_angles[theta] + src_stride * (h - 2);
        sum += flt_ptr[1] + flt_ptr[0] + flt_ptr[1];
        src_ptr += src_stride; flt_ptr += src_stride;
        sum += flt_ptr[1];
        sum += (int16_t)(((ONE_BY_15 * abs((int32_t)src_ptr[0])) + 2048) >> 12);
        sum += flt_ptr[1];
        sum += flt_ptr[1] + flt_ptr[0] + flt_ptr[1];
        accum += sum;
    }
    return accum;
}

/* Scalar boundary threshold: i=h-1, j=1..w-2 */
static inline int32_t adm_cm_thresh_s_hm1_j_scalar(
    int16_t *angles[3], int16_t *flt_angles[3], int src_stride, int h, int j)
{
    int32_t accum = 0;
    for (int theta = 0; theta < 3; ++theta) {
        int32_t sum = 0;
        int16_t *src_ptr = angles[theta] + src_stride * (h - 2);
        int16_t *flt_ptr = flt_angles[theta] + src_stride * (h - 2);
        sum += flt_ptr[j - 1] + flt_ptr[j] + flt_ptr[j + 1];
        src_ptr += src_stride; flt_ptr += src_stride;
        sum += flt_ptr[j - 1];
        sum += (int16_t)(((ONE_BY_15 * abs((int32_t)src_ptr[j])) + 2048) >> 12);
        sum += flt_ptr[j + 1];
        sum += flt_ptr[j - 1] + flt_ptr[j] + flt_ptr[j + 1];
        accum += sum;
    }
    return accum;
}

/* Scalar boundary threshold: i=h-1, j=w-1 */
static inline int32_t adm_cm_thresh_s_hm1_wm1_scalar(
    int16_t *angles[3], int16_t *flt_angles[3], int src_stride, int w, int h)
{
    int32_t accum = 0;
    for (int theta = 0; theta < 3; ++theta) {
        int32_t sum = 0;
        int16_t *src_ptr = angles[theta] + src_stride * (h - 2);
        int16_t *flt_ptr = flt_angles[theta] + src_stride * (h - 2);
        sum += flt_ptr[w - 2] + flt_ptr[w - 1] + flt_ptr[w - 1];
        src_ptr += src_stride; flt_ptr += src_stride;
        sum += flt_ptr[w - 2];
        sum += (int16_t)(((ONE_BY_15 * abs((int32_t)src_ptr[w - 1])) + 2048) >> 12);
        sum += flt_ptr[w - 1];
        sum += flt_ptr[w - 2] + flt_ptr[w - 1] + flt_ptr[w - 1];
        accum += sum;
    }
    return accum;
}

/* Scalar accum_round for adm_cm (int16 path) */
static inline void adm_cm_accum_round_scalar(
    int32_t x, int32_t thr, int32_t shift_sub,
    int32_t add_shift_sq, int32_t shift_sq,
    int32_t add_shift_cub, uint32_t shift_cub,
    int64_t *accum)
{
    x = abs(x) - ((int32_t)(thr) << shift_sub);
    x = x < 0 ? 0 : x;
    int32_t x_sq = (int32_t)((((int64_t)x * x) + add_shift_sq) >> shift_sq);
    int64_t val = (((int64_t)x_sq * x) + add_shift_cub) >> shift_cub;
    *accum += val;
}

/*
 * AVX2 helper: accumulate cubed values for 8 int32 pixels.
 * x_vec contains 8 int32 values already max(abs(x) - thr_shifted, 0).
 * Computes x^2 >> shift_sq, then (x^2 >> shift_sq) * x >> shift_cub,
 * and accumulates into accum (int64).
 */
static inline void adm_cm_accum_round_avx2_8(
    __m256i x_vec, int32_t add_shift_sq, int32_t shift_sq,
    int32_t add_shift_cub, uint32_t shift_cub, int64_t *accum)
{
    /* x_vec: 8 x int32 values, all non-negative */
    __m256i add_sq = _mm256_set1_epi32(add_shift_sq);

    /* Process low 4 and high 4 elements separately for int64 arithmetic */
    __m128i x_lo128 = _mm256_castsi256_si128(x_vec);
    __m128i x_hi128 = _mm256_extracti128_si256(x_vec, 1);

    /* x_sq = ((int64)x * x + add) >> shift for low 4 elements */
    /* _mm256_mul_epi32 multiplies elements 0,2,4,6 (even positions) producing 4 int64 results */
    /* So we need two passes per 4 elements: even and odd */

    /* Low 4 elements: x_lo128 contains elements [0,1,2,3] */
    /* Even elements (0,2): */
    __m256i x_lo256 = _mm256_cvtepi32_epi64(x_lo128);  /* 4 x int64 from low 4 */
    __m256i x_hi256 = _mm256_cvtepi32_epi64(x_hi128);  /* 4 x int64 from high 4 */

    __m256i add_sq64 = _mm256_set1_epi64x(add_shift_sq);
    __m256i add_cub64 = _mm256_set1_epi64x(add_shift_cub);

    /* x_sq_lo = (x_lo * x_lo + add_sq) >> shift_sq */
    /* We need 64-bit multiply. Use _mm256_mul_epi32 which takes 32-bit inputs from even positions */
    /* Since our values are in full 64-bit, and we know they fit in 32 bits, we can use mullo approach */
    /* Actually x values fit in int32 (they are non-negative clipped values), so we can do: */
    /* Recompute: x_sq per element using 64-bit math */

    /* x_lo256 has 4 int64 values. We need x*x as int64. */
    /* _mm256_mul_epi32 takes low 32 bits of each 64-bit lane and multiplies to give 64-bit results */
    /* That's exactly what we want since x fits in 32 bits */
    __m256i xsq_lo = _mm256_add_epi64(
        _mm256_mul_epi32(x_lo256, x_lo256), add_sq64);
    xsq_lo = _mm256_srli_epi64(xsq_lo, shift_sq);

    __m256i xsq_hi = _mm256_add_epi64(
        _mm256_mul_epi32(x_hi256, x_hi256), add_sq64);
    xsq_hi = _mm256_srli_epi64(xsq_hi, shift_sq);

    /* x_sq is now int32-ranged (shifted down). Truncate back to 32 bits for the cube step */
    /* val = (x_sq * x + add_cub) >> shift_cub (result is int64) */
    /* x_sq fits in int32, x fits in int32, product fits in int64 */
    __m256i val_lo = _mm256_add_epi64(
        _mm256_mul_epi32(xsq_lo, x_lo256), add_cub64);
    /* Arithmetic right shift for int64 - use variable shift or manual */
    /* Since shift_cub is small and values are non-negative, logical shift is fine */
    val_lo = _mm256_srli_epi64(val_lo, shift_cub);

    __m256i val_hi = _mm256_add_epi64(
        _mm256_mul_epi32(xsq_hi, x_hi256), add_cub64);
    val_hi = _mm256_srli_epi64(val_hi, shift_cub);

    /* Horizontal sum of all 8 int64 values */
    __m256i sum_8 = _mm256_add_epi64(val_lo, val_hi); /* 4 int64 */
    __m128i sum_lo = _mm256_castsi256_si128(sum_8);
    __m128i sum_hi = _mm256_extracti128_si256(sum_8, 1);
    __m128i sum_4 = _mm_add_epi64(sum_lo, sum_hi);   /* 2 int64 */
    __m128i sum_2 = _mm_add_epi64(sum_4, _mm_srli_si128(sum_4, 8)); /* 1 int64 */
    *accum += _mm_cvtsi128_si64(sum_2);
}

/*
 * Load 8 int16 values from ptr, sign-extend to 8 int32 in a __m256i.
 */
static inline __m256i load_i16_to_i32_8(const int16_t *ptr)
{
    __m128i v16 = _mm_loadu_si128((const __m128i *)ptr);
    return _mm256_cvtepi16_epi32(v16);
}

/*
 * Compute threshold for 8 pixels at positions j..j+7 (interior, int16 path).
 * angles[3] and flt_angles[3] point to the h/v/d subbands of csf_a and csf_f.
 * Returns __m256i with 8 int32 threshold values.
 */
static inline __m256i adm_cm_thresh_avx2_8(
    int16_t *angles[3], int16_t *flt_angles[3],
    int src_stride, int i, int j)
{
    __m256i thr = _mm256_setzero_si256();
    __m256i v_one_by_15 = _mm256_set1_epi32(ONE_BY_15);
    __m256i v_round = _mm256_set1_epi32(2048);

    for (int theta = 0; theta < 3; ++theta) {
        const int16_t *src_row = angles[theta] + src_stride * i;
        const int16_t *flt_above = flt_angles[theta] + src_stride * (i - 1);
        const int16_t *flt_center = flt_angles[theta] + src_stride * i;
        const int16_t *flt_below = flt_angles[theta] + src_stride * (i + 1);

        /* Row above: flt[i-1][j-1] + flt[i-1][j] + flt[i-1][j+1] */
        __m256i a_l = load_i16_to_i32_8(flt_above + j - 1);
        __m256i a_c = load_i16_to_i32_8(flt_above + j);
        __m256i a_r = load_i16_to_i32_8(flt_above + j + 1);
        __m256i row_above = _mm256_add_epi32(_mm256_add_epi32(a_l, a_c), a_r);

        /* Center row: flt[i][j-1] + flt[i][j+1] + center_term */
        __m256i c_l = load_i16_to_i32_8(flt_center + j - 1);
        __m256i c_r = load_i16_to_i32_8(flt_center + j + 1);

        /* Center term: (int16_t)(((ONE_BY_15 * abs(src[i][j])) + 2048) >> 12) */
        __m256i src_c = load_i16_to_i32_8(src_row + j);
        __m256i src_abs = _mm256_abs_epi32(src_c);
        __m256i center_term = _mm256_srli_epi32(
            _mm256_add_epi32(_mm256_mullo_epi32(v_one_by_15, src_abs), v_round), 12);
        /* Truncate to int16 range to match scalar (int16_t) cast */
        center_term = _mm256_slli_epi32(center_term, 16);
        center_term = _mm256_srai_epi32(center_term, 16);

        __m256i row_center = _mm256_add_epi32(_mm256_add_epi32(c_l, c_r), center_term);

        /* Row below: flt[i+1][j-1] + flt[i+1][j] + flt[i+1][j+1] */
        __m256i b_l = load_i16_to_i32_8(flt_below + j - 1);
        __m256i b_c = load_i16_to_i32_8(flt_below + j);
        __m256i b_r = load_i16_to_i32_8(flt_below + j + 1);
        __m256i row_below = _mm256_add_epi32(_mm256_add_epi32(b_l, b_c), b_r);

        __m256i theta_sum = _mm256_add_epi32(_mm256_add_epi32(row_above, row_center), row_below);
        thr = _mm256_add_epi32(thr, theta_sum);
    }
    return thr;
}

float adm_cm_avx2(AdmBuffer *buf, int w, int h, int src_stride,
                   int csf_a_stride,
                   const float csf_factors[4][2],
                   double adm_norm_view_dist, int adm_ref_display_height)
{
    const adm_dwt_band_t *src   = &buf->decouple_r;
    const adm_dwt_band_t *csf_f = &buf->csf_f;
    const adm_dwt_band_t *csf_a = &buf->csf_a;

    const float factor1 = csf_factors[0][0];
    const float factor2 = csf_factors[0][1];

    uint16_t i_rfactor[3];
    if (fabs(adm_norm_view_dist * adm_ref_display_height - DEFAULT_ADM_NORM_VIEW_DIST * DEFAULT_ADM_REF_DISPLAY_HEIGHT) < 1.0e-8) {
        i_rfactor[0] = 36453;
        i_rfactor[1] = 36453;
        i_rfactor[2] = 49417;
    } else {
        const float rfactor1[3] = { 1.0f / factor1, 1.0f / factor1, 1.0f / factor2 };
        const double pow2_21 = pow(2, 21);
        const double pow2_23 = pow(2, 23);
        i_rfactor[0] = (uint16_t)(rfactor1[0] * pow2_21);
        i_rfactor[1] = (uint16_t)(rfactor1[1] * pow2_21);
        i_rfactor[2] = (uint16_t)(rfactor1[2] * pow2_23);
    }

    const int32_t shift_xhsq = 29, shift_xvsq = 29, shift_xdsq = 30;
    const int32_t add_shift_xhsq = 268435456, add_shift_xvsq = 268435456, add_shift_xdsq = 536870912;

    const uint32_t shift_xhcub = (uint32_t)ceil(log2(w) - 4);
    const uint32_t add_shift_xhcub = (uint32_t)pow(2, (shift_xhcub - 1));
    const uint32_t shift_xvcub = (uint32_t)ceil(log2(w) - 4);
    const uint32_t add_shift_xvcub = (uint32_t)pow(2, (shift_xvcub - 1));
    const uint32_t shift_xdcub = (uint32_t)ceil(log2(w) - 3);
    const uint32_t add_shift_xdcub = (uint32_t)pow(2, (shift_xdcub - 1));

    const uint32_t shift_inner_accum = (uint32_t)ceil(log2(h));
    const uint32_t add_shift_inner_accum = (uint32_t)pow(2, (shift_inner_accum - 1));

    const int32_t shift_xhsub = 10, shift_xvsub = 10, shift_xdsub = 12;

    int16_t *angles[3] = { csf_a->band_h, csf_a->band_v, csf_a->band_d };
    int16_t *flt_angles[3] = { csf_f->band_h, csf_f->band_v, csf_f->band_d };

    int left = w * ADM_BORDER_FACTOR - 0.5;
    int top = h * ADM_BORDER_FACTOR - 0.5;
    int right = w - left;
    int bottom = h - top;

    const int start_col = (left > 1) ? left : 1;
    const int end_col = (right < (w - 1)) ? right : (w - 1);
    const int start_row = (top > 1) ? top : 1;
    const int end_row = (bottom < (h - 1)) ? bottom : (h - 1);

    int i, j;
    int64_t accum_h = 0, accum_v = 0, accum_d = 0;
    int64_t accum_inner_h = 0, accum_inner_v = 0, accum_inner_d = 0;

    /* i=0 row (boundary) */
    if (top <= 0) {
        if (left <= 0) {
            int32_t thr = adm_cm_thresh_s_0_0_scalar(angles, flt_angles, csf_a_stride);
            int32_t xh = (int32_t)src->band_h[0] * i_rfactor[0];
            int32_t xv = (int32_t)src->band_v[0] * i_rfactor[1];
            int32_t xd = (int32_t)src->band_d[0] * i_rfactor[2];
            adm_cm_accum_round_scalar(xh, thr, shift_xhsub, add_shift_xhsq, shift_xhsq, add_shift_xhcub, shift_xhcub, &accum_inner_h);
            adm_cm_accum_round_scalar(xv, thr, shift_xvsub, add_shift_xvsq, shift_xvsq, add_shift_xvcub, shift_xvcub, &accum_inner_v);
            adm_cm_accum_round_scalar(xd, thr, shift_xdsub, add_shift_xdsq, shift_xdsq, add_shift_xdcub, shift_xdcub, &accum_inner_d);
        }
        for (j = start_col; j < end_col; ++j) {
            int32_t thr = adm_cm_thresh_s_0_j_scalar(angles, flt_angles, csf_a_stride, j);
            int32_t xh = src->band_h[j] * i_rfactor[0];
            int32_t xv = src->band_v[j] * i_rfactor[1];
            int32_t xd = src->band_d[j] * i_rfactor[2];
            adm_cm_accum_round_scalar(xh, thr, shift_xhsub, add_shift_xhsq, shift_xhsq, add_shift_xhcub, shift_xhcub, &accum_inner_h);
            adm_cm_accum_round_scalar(xv, thr, shift_xvsub, add_shift_xvsq, shift_xvsq, add_shift_xvcub, shift_xvcub, &accum_inner_v);
            adm_cm_accum_round_scalar(xd, thr, shift_xdsub, add_shift_xdsq, shift_xdsq, add_shift_xdcub, shift_xdcub, &accum_inner_d);
        }
        if (right > (w - 1)) {
            int32_t thr = adm_cm_thresh_s_0_wm1_scalar(angles, flt_angles, csf_a_stride, w);
            int32_t xh = src->band_h[w - 1] * i_rfactor[0];
            int32_t xv = src->band_v[w - 1] * i_rfactor[1];
            int32_t xd = src->band_d[w - 1] * i_rfactor[2];
            adm_cm_accum_round_scalar(xh, thr, shift_xhsub, add_shift_xhsq, shift_xhsq, add_shift_xhcub, shift_xhcub, &accum_inner_h);
            adm_cm_accum_round_scalar(xv, thr, shift_xvsub, add_shift_xvsq, shift_xvsq, add_shift_xvcub, shift_xvcub, &accum_inner_v);
            adm_cm_accum_round_scalar(xd, thr, shift_xdsub, add_shift_xdsq, shift_xdsq, add_shift_xdcub, shift_xdcub, &accum_inner_d);
        }
    }
    accum_h += (accum_inner_h + add_shift_inner_accum) >> shift_inner_accum;
    accum_v += (accum_inner_v + add_shift_inner_accum) >> shift_inner_accum;
    accum_d += (accum_inner_d + add_shift_inner_accum) >> shift_inner_accum;

    /* Interior rows: i = start_row..end_row-1 */
    /* Determine if j=0 and/or j=w-1 boundaries are included */
    const int do_left_boundary = (left <= 0);
    const int do_right_boundary = (right > (w - 1));

    /* AVX2 vectorized interior j-loop constants */
    const __m256i v_rfactor_h = _mm256_set1_epi32((int32_t)i_rfactor[0]);
    const __m256i v_rfactor_v = _mm256_set1_epi32((int32_t)i_rfactor[1]);
    const __m256i v_rfactor_d = _mm256_set1_epi32((int32_t)i_rfactor[2]);
    const __m256i v_zero = _mm256_setzero_si256();

    for (i = start_row; i < end_row; ++i) {
        const int i_offset = i * src_stride;
        accum_inner_h = 0;
        accum_inner_v = 0;
        accum_inner_d = 0;

        /* Left boundary scalar */
        if (do_left_boundary) {
            int32_t thr = adm_cm_thresh_s_i_0_scalar(angles, flt_angles, csf_a_stride, i);
            int32_t xh = src->band_h[i_offset] * i_rfactor[0];
            int32_t xv = src->band_v[i_offset] * i_rfactor[1];
            int32_t xd = src->band_d[i_offset] * i_rfactor[2];
            adm_cm_accum_round_scalar(xh, thr, shift_xhsub, add_shift_xhsq, shift_xhsq, add_shift_xhcub, shift_xhcub, &accum_inner_h);
            adm_cm_accum_round_scalar(xv, thr, shift_xvsub, add_shift_xvsq, shift_xvsq, add_shift_xvcub, shift_xvcub, &accum_inner_v);
            adm_cm_accum_round_scalar(xd, thr, shift_xdsub, add_shift_xdsq, shift_xdsq, add_shift_xdcub, shift_xdcub, &accum_inner_d);
        }

        /* AVX2 vectorized interior j-loop: process 8 pixels at a time */
        j = start_col;
        for (; j + 8 <= end_col; j += 8) {
            /* Compute threshold for 8 pixels */
            __m256i thr_vec = adm_cm_thresh_avx2_8(angles, flt_angles, csf_a_stride, i, j);

            /* Band H: x = src->band_h[i_offset + j] * i_rfactor[0] */
            __m256i src_h = load_i16_to_i32_8(src->band_h + i_offset + j);
            __m256i xh_vec = _mm256_mullo_epi32(src_h, v_rfactor_h);
            /* abs(x) - (thr << shift_xhsub) */
            __m256i xh_abs = _mm256_abs_epi32(xh_vec);
            __m256i thr_shifted_h = _mm256_slli_epi32(thr_vec, shift_xhsub);
            __m256i xh_sub = _mm256_sub_epi32(xh_abs, thr_shifted_h);
            xh_sub = _mm256_max_epi32(xh_sub, v_zero);
            adm_cm_accum_round_avx2_8(xh_sub, add_shift_xhsq, shift_xhsq, add_shift_xhcub, shift_xhcub, &accum_inner_h);

            /* Band V */
            __m256i src_v = load_i16_to_i32_8(src->band_v + i_offset + j);
            __m256i xv_vec = _mm256_mullo_epi32(src_v, v_rfactor_v);
            __m256i xv_abs = _mm256_abs_epi32(xv_vec);
            __m256i thr_shifted_v = _mm256_slli_epi32(thr_vec, shift_xvsub);
            __m256i xv_sub = _mm256_sub_epi32(xv_abs, thr_shifted_v);
            xv_sub = _mm256_max_epi32(xv_sub, v_zero);
            adm_cm_accum_round_avx2_8(xv_sub, add_shift_xvsq, shift_xvsq, add_shift_xvcub, shift_xvcub, &accum_inner_v);

            /* Band D */
            __m256i src_d = load_i16_to_i32_8(src->band_d + i_offset + j);
            __m256i xd_vec = _mm256_mullo_epi32(src_d, v_rfactor_d);
            __m256i xd_abs = _mm256_abs_epi32(xd_vec);
            __m256i thr_shifted_d = _mm256_slli_epi32(thr_vec, shift_xdsub);
            __m256i xd_sub = _mm256_sub_epi32(xd_abs, thr_shifted_d);
            xd_sub = _mm256_max_epi32(xd_sub, v_zero);
            adm_cm_accum_round_avx2_8(xd_sub, add_shift_xdsq, shift_xdsq, add_shift_xdcub, shift_xdcub, &accum_inner_d);
        }

        /* Scalar remainder for interior */
        for (; j < end_col; ++j) {
            int32_t thr = adm_cm_thresh_s_i_j_scalar(angles, flt_angles, csf_a_stride, i, j);
            int32_t xh = src->band_h[i_offset + j] * i_rfactor[0];
            int32_t xv = src->band_v[i_offset + j] * i_rfactor[1];
            int32_t xd = src->band_d[i_offset + j] * i_rfactor[2];
            adm_cm_accum_round_scalar(xh, thr, shift_xhsub, add_shift_xhsq, shift_xhsq, add_shift_xhcub, shift_xhcub, &accum_inner_h);
            adm_cm_accum_round_scalar(xv, thr, shift_xvsub, add_shift_xvsq, shift_xvsq, add_shift_xvcub, shift_xvcub, &accum_inner_v);
            adm_cm_accum_round_scalar(xd, thr, shift_xdsub, add_shift_xdsq, shift_xdsq, add_shift_xdcub, shift_xdcub, &accum_inner_d);
        }

        /* Right boundary scalar */
        if (do_right_boundary) {
            int32_t thr = adm_cm_thresh_s_i_wm1_scalar(angles, flt_angles, csf_a_stride, w, i);
            int32_t xh = src->band_h[i_offset + w - 1] * i_rfactor[0];
            int32_t xv = src->band_v[i_offset + w - 1] * i_rfactor[1];
            int32_t xd = src->band_d[i_offset + w - 1] * i_rfactor[2];
            adm_cm_accum_round_scalar(xh, thr, shift_xhsub, add_shift_xhsq, shift_xhsq, add_shift_xhcub, shift_xhcub, &accum_inner_h);
            adm_cm_accum_round_scalar(xv, thr, shift_xvsub, add_shift_xvsq, shift_xvsq, add_shift_xvcub, shift_xvcub, &accum_inner_v);
            adm_cm_accum_round_scalar(xd, thr, shift_xdsub, add_shift_xdsq, shift_xdsq, add_shift_xdcub, shift_xdcub, &accum_inner_d);
        }

        accum_h += (accum_inner_h + add_shift_inner_accum) >> shift_inner_accum;
        accum_v += (accum_inner_v + add_shift_inner_accum) >> shift_inner_accum;
        accum_d += (accum_inner_d + add_shift_inner_accum) >> shift_inner_accum;
    }

    /* i=h-1 row (boundary) */
    accum_inner_h = 0;
    accum_inner_v = 0;
    accum_inner_d = 0;
    if (bottom > (h - 1)) {
        const int hm1_src_offset = (h - 1) * src_stride;
        if (left <= 0) {
            int32_t thr = adm_cm_thresh_s_hm1_0_scalar(angles, flt_angles, csf_a_stride, h);
            int32_t xh = src->band_h[hm1_src_offset] * i_rfactor[0];
            int32_t xv = src->band_v[hm1_src_offset] * i_rfactor[1];
            int32_t xd = src->band_d[hm1_src_offset] * i_rfactor[2];
            adm_cm_accum_round_scalar(xh, thr, shift_xhsub, add_shift_xhsq, shift_xhsq, add_shift_xhcub, shift_xhcub, &accum_inner_h);
            adm_cm_accum_round_scalar(xv, thr, shift_xvsub, add_shift_xvsq, shift_xvsq, add_shift_xvcub, shift_xvcub, &accum_inner_v);
            adm_cm_accum_round_scalar(xd, thr, shift_xdsub, add_shift_xdsq, shift_xdsq, add_shift_xdcub, shift_xdcub, &accum_inner_d);
        }
        for (j = start_col; j < end_col; ++j) {
            int32_t thr = adm_cm_thresh_s_hm1_j_scalar(angles, flt_angles, csf_a_stride, h, j);
            int32_t xh = src->band_h[hm1_src_offset + j] * i_rfactor[0];
            int32_t xv = src->band_v[hm1_src_offset + j] * i_rfactor[1];
            int32_t xd = src->band_d[hm1_src_offset + j] * i_rfactor[2];
            adm_cm_accum_round_scalar(xh, thr, shift_xhsub, add_shift_xhsq, shift_xhsq, add_shift_xhcub, shift_xhcub, &accum_inner_h);
            adm_cm_accum_round_scalar(xv, thr, shift_xvsub, add_shift_xvsq, shift_xvsq, add_shift_xvcub, shift_xvcub, &accum_inner_v);
            adm_cm_accum_round_scalar(xd, thr, shift_xdsub, add_shift_xdsq, shift_xdsq, add_shift_xdcub, shift_xdcub, &accum_inner_d);
        }
        if (right > (w - 1)) {
            int32_t thr = adm_cm_thresh_s_hm1_wm1_scalar(angles, flt_angles, csf_a_stride, w, h);
            int32_t xh = src->band_h[hm1_src_offset + w - 1] * i_rfactor[0];
            int32_t xv = src->band_v[hm1_src_offset + w - 1] * i_rfactor[1];
            int32_t xd = src->band_d[hm1_src_offset + w - 1] * i_rfactor[2];
            adm_cm_accum_round_scalar(xh, thr, shift_xhsub, add_shift_xhsq, shift_xhsq, add_shift_xhcub, shift_xhcub, &accum_inner_h);
            adm_cm_accum_round_scalar(xv, thr, shift_xvsub, add_shift_xvsq, shift_xvsq, add_shift_xvcub, shift_xvcub, &accum_inner_v);
            adm_cm_accum_round_scalar(xd, thr, shift_xdsub, add_shift_xdsq, shift_xdsq, add_shift_xdcub, shift_xdcub, &accum_inner_d);
        }
    }
    accum_h += (accum_inner_h + add_shift_inner_accum) >> shift_inner_accum;
    accum_v += (accum_inner_v + add_shift_inner_accum) >> shift_inner_accum;
    accum_d += (accum_inner_d + add_shift_inner_accum) >> shift_inner_accum;

    float f_accum_h = (float)(accum_h / pow(2, (52 - shift_xhcub - shift_inner_accum)));
    float f_accum_v = (float)(accum_v / pow(2, (52 - shift_xvcub - shift_inner_accum)));
    float f_accum_d = (float)(accum_d / pow(2, (57 - shift_xdcub - shift_inner_accum)));

    const float powf_add = powf((bottom - top) * (right - left) / 32.0f, 1.0f / 3.0f);
    float num_scale_h = powf(f_accum_h, 1.0f / 3.0f) + powf_add;
    float num_scale_v = powf(f_accum_v, 1.0f / 3.0f) + powf_add;
    float num_scale_d = powf(f_accum_d, 1.0f / 3.0f) + powf_add;

    return (num_scale_h + num_scale_v + num_scale_d);
}

/* ================================================================
 * i4 (int32) path scalar helpers for boundary pixels
 * ================================================================ */
static inline int32_t i4_adm_cm_thresh_s_i_j_scalar(
    int32_t *angles[3], int32_t *flt_angles[3],
    int src_stride, int i, int j,
    int32_t add_bef_shift, uint32_t shift)
{
    int32_t accum = 0;
    for (int theta = 0; theta < 3; ++theta) {
        int32_t sum = 0;
        int32_t *src_ptr = angles[theta] + src_stride * (i - 1);
        int32_t *flt_ptr = flt_angles[theta] + src_stride * (i - 1);
        sum += flt_ptr[j - 1] + flt_ptr[j] + flt_ptr[j + 1];
        src_ptr += src_stride; flt_ptr += src_stride;
        sum += flt_ptr[j - 1] + flt_ptr[j + 1];
        sum += (int32_t)((((int64_t)I4_ONE_BY_15 * abs(src_ptr[j])) + add_bef_shift) >> shift);
        flt_ptr += src_stride;
        sum += flt_ptr[j - 1] + flt_ptr[j] + flt_ptr[j + 1];
        accum += sum;
    }
    return accum;
}

static inline int32_t i4_adm_cm_thresh_s_0_0_scalar(
    int32_t *angles[3], int32_t *flt_angles[3], int src_stride,
    int32_t add_bef_shift, uint32_t shift)
{
    int32_t accum = 0;
    for (int theta = 0; theta < 3; ++theta) {
        int32_t sum = 0;
        int32_t *src_ptr = angles[theta];
        int32_t *flt_ptr = flt_angles[theta];
        sum += flt_ptr[src_stride + 1] + flt_ptr[src_stride] + flt_ptr[src_stride + 1];
        sum += flt_ptr[1];
        sum += (int32_t)((((int64_t)I4_ONE_BY_15 * abs(src_ptr[0])) + add_bef_shift) >> shift);
        sum += flt_ptr[1];
        sum += flt_ptr[src_stride + 1] + flt_ptr[src_stride] + flt_ptr[src_stride + 1];
        accum += sum;
    }
    return accum;
}

static inline int32_t i4_adm_cm_thresh_s_0_j_scalar(
    int32_t *angles[3], int32_t *flt_angles[3], int src_stride, int j,
    int32_t add_bef_shift, uint32_t shift)
{
    int32_t accum = 0;
    for (int theta = 0; theta < 3; ++theta) {
        int32_t sum = 0;
        int32_t *src_ptr = angles[theta];
        int32_t *flt_ptr = flt_angles[theta];
        sum += flt_ptr[src_stride + j - 1] + flt_ptr[src_stride + j] + flt_ptr[src_stride + j + 1];
        sum += flt_ptr[j - 1];
        sum += (int32_t)((((int64_t)I4_ONE_BY_15 * abs(src_ptr[j])) + add_bef_shift) >> shift);
        sum += flt_ptr[j + 1];
        sum += flt_ptr[src_stride + j - 1] + flt_ptr[src_stride + j] + flt_ptr[src_stride + j + 1];
        accum += sum;
    }
    return accum;
}

static inline int32_t i4_adm_cm_thresh_s_0_wm1_scalar(
    int32_t *angles[3], int32_t *flt_angles[3], int src_stride, int w,
    int32_t add_bef_shift, uint32_t shift)
{
    int32_t accum = 0;
    for (int theta = 0; theta < 3; ++theta) {
        int32_t sum = 0;
        int32_t *src_ptr = angles[theta];
        int32_t *flt_ptr = flt_angles[theta];
        sum += flt_ptr[src_stride + w - 2] + flt_ptr[src_stride + w - 1] + flt_ptr[src_stride + w - 1];
        sum += flt_ptr[w - 2];
        sum += (int32_t)((((int64_t)I4_ONE_BY_15 * abs(src_ptr[w - 1])) + add_bef_shift) >> shift);
        sum += flt_ptr[w - 1];
        sum += flt_ptr[src_stride + w - 2] + flt_ptr[src_stride + w - 1] + flt_ptr[src_stride + w - 1];
        accum += sum;
    }
    return accum;
}

static inline int32_t i4_adm_cm_thresh_s_i_0_scalar(
    int32_t *angles[3], int32_t *flt_angles[3], int src_stride, int i,
    int32_t add_bef_shift, uint32_t shift)
{
    int32_t accum = 0;
    for (int theta = 0; theta < 3; ++theta) {
        int32_t sum = 0;
        int32_t *src_ptr = angles[theta] + src_stride * (i - 1);
        int32_t *flt_ptr = flt_angles[theta] + src_stride * (i - 1);
        sum += flt_ptr[1] + flt_ptr[0] + flt_ptr[1];
        src_ptr += src_stride; flt_ptr += src_stride;
        sum += flt_ptr[1];
        sum += (int32_t)((((int64_t)I4_ONE_BY_15 * abs(src_ptr[0])) + add_bef_shift) >> shift);
        sum += flt_ptr[1];
        flt_ptr += src_stride;
        sum += flt_ptr[1] + flt_ptr[0] + flt_ptr[1];
        accum += sum;
    }
    return accum;
}

static inline int32_t i4_adm_cm_thresh_s_i_wm1_scalar(
    int32_t *angles[3], int32_t *flt_angles[3], int src_stride, int w, int i,
    int32_t add_bef_shift, uint32_t shift)
{
    int32_t accum = 0;
    for (int theta = 0; theta < 3; ++theta) {
        int32_t sum = 0;
        int32_t *src_ptr = angles[theta] + src_stride * (i - 1);
        int32_t *flt_ptr = flt_angles[theta] + src_stride * (i - 1);
        sum += flt_ptr[w - 2] + flt_ptr[w - 1] + flt_ptr[w - 1];
        src_ptr += src_stride; flt_ptr += src_stride;
        sum += flt_ptr[w - 2];
        sum += (int32_t)((((int64_t)I4_ONE_BY_15 * abs(src_ptr[w - 1])) + add_bef_shift) >> shift);
        sum += flt_ptr[w - 1];
        flt_ptr += src_stride;
        sum += flt_ptr[w - 2] + flt_ptr[w - 1] + flt_ptr[w - 1];
        accum += sum;
    }
    return accum;
}

static inline int32_t i4_adm_cm_thresh_s_hm1_0_scalar(
    int32_t *angles[3], int32_t *flt_angles[3], int src_stride, int h,
    int32_t add_bef_shift, uint32_t shift)
{
    int32_t accum = 0;
    for (int theta = 0; theta < 3; ++theta) {
        int32_t sum = 0;
        int32_t *src_ptr = angles[theta] + src_stride * (h - 2);
        int32_t *flt_ptr = flt_angles[theta] + src_stride * (h - 2);
        sum += flt_ptr[1] + flt_ptr[0] + flt_ptr[1];
        src_ptr += src_stride; flt_ptr += src_stride;
        sum += flt_ptr[1];
        sum += (int32_t)((((int64_t)I4_ONE_BY_15 * abs(src_ptr[0])) + add_bef_shift) >> shift);
        sum += flt_ptr[1];
        sum += flt_ptr[1] + flt_ptr[0] + flt_ptr[1];
        accum += sum;
    }
    return accum;
}

static inline int32_t i4_adm_cm_thresh_s_hm1_j_scalar(
    int32_t *angles[3], int32_t *flt_angles[3], int src_stride, int h, int j,
    int32_t add_bef_shift, uint32_t shift)
{
    int32_t accum = 0;
    for (int theta = 0; theta < 3; ++theta) {
        int32_t sum = 0;
        int32_t *src_ptr = angles[theta] + src_stride * (h - 2);
        int32_t *flt_ptr = flt_angles[theta] + src_stride * (h - 2);
        sum += flt_ptr[j - 1] + flt_ptr[j] + flt_ptr[j + 1];
        src_ptr += src_stride; flt_ptr += src_stride;
        sum += flt_ptr[j - 1];
        sum += (int32_t)((((int64_t)I4_ONE_BY_15 * abs(src_ptr[j])) + add_bef_shift) >> shift);
        sum += flt_ptr[j + 1];
        sum += flt_ptr[j - 1] + flt_ptr[j] + flt_ptr[j + 1];
        accum += sum;
    }
    return accum;
}

static inline int32_t i4_adm_cm_thresh_s_hm1_wm1_scalar(
    int32_t *angles[3], int32_t *flt_angles[3], int src_stride, int w, int h,
    int32_t add_bef_shift, uint32_t shift)
{
    int32_t accum = 0;
    for (int theta = 0; theta < 3; ++theta) {
        int32_t sum = 0;
        int32_t *src_ptr = angles[theta] + src_stride * (h - 2);
        int32_t *flt_ptr = flt_angles[theta] + src_stride * (h - 2);
        sum += flt_ptr[w - 2] + flt_ptr[w - 1] + flt_ptr[w - 1];
        src_ptr += src_stride; flt_ptr += src_stride;
        sum += flt_ptr[w - 2];
        sum += (int32_t)((((int64_t)I4_ONE_BY_15 * abs(src_ptr[w - 1])) + add_bef_shift) >> shift);
        sum += flt_ptr[w - 1];
        sum += flt_ptr[w - 2] + flt_ptr[w - 1] + flt_ptr[w - 1];
        accum += sum;
    }
    return accum;
}

/* i4 scalar accum_round (int32 path): x = abs(x) - (thr >> shift_sub) */
static inline void i4_adm_cm_accum_round_scalar(
    int32_t x, int32_t thr, int32_t shift_sub,
    int32_t add_shift_sq, int32_t shift_sq,
    int32_t add_shift_cub, uint32_t shift_cub,
    int64_t *accum)
{
    x = abs(x) - (thr >> shift_sub);
    x = x < 0 ? 0 : x;
    int32_t x_sq = (int32_t)((((int64_t)x * x) + add_shift_sq) >> shift_sq);
    int64_t val = (((int64_t)x_sq * x) + add_shift_cub) >> shift_cub;
    *accum += val;
}

/*
 * AVX2 helper: compute threshold for 8 int32 pixels in i4 path.
 */
static inline __m256i i4_adm_cm_thresh_avx2_8(
    int32_t *angles[3], int32_t *flt_angles[3],
    int src_stride, int i, int j,
    int32_t add_bef_shift, uint32_t shift)
{
    __m256i thr = _mm256_setzero_si256();
    __m256i v_add = _mm256_set1_epi64x(add_bef_shift);

    for (int theta = 0; theta < 3; ++theta) {
        const int32_t *src_row = angles[theta] + src_stride * i;
        const int32_t *flt_above = flt_angles[theta] + src_stride * (i - 1);
        const int32_t *flt_center = flt_angles[theta] + src_stride * i;
        const int32_t *flt_below = flt_angles[theta] + src_stride * (i + 1);

        /* Row above: flt[i-1][j-1] + flt[i-1][j] + flt[i-1][j+1] */
        __m256i a_l = _mm256_loadu_si256((const __m256i *)(flt_above + j - 1));
        __m256i a_c = _mm256_loadu_si256((const __m256i *)(flt_above + j));
        __m256i a_r = _mm256_loadu_si256((const __m256i *)(flt_above + j + 1));
        __m256i row_above = _mm256_add_epi32(_mm256_add_epi32(a_l, a_c), a_r);

        /* Center row: flt[i][j-1] + flt[i][j+1] + center_term */
        __m256i c_l = _mm256_loadu_si256((const __m256i *)(flt_center + j - 1));
        __m256i c_r = _mm256_loadu_si256((const __m256i *)(flt_center + j + 1));

        /* Center term: (int32_t)(((int64_t)I4_ONE_BY_15 * abs(src[j])) + add) >> shift */
        __m256i src_c = _mm256_loadu_si256((const __m256i *)(src_row + j));
        __m256i src_abs = _mm256_abs_epi32(src_c);
        /* Need 64-bit multiply: I4_ONE_BY_15 * abs(src) */
        /* Process elements 0,2,4,6 and 1,3,5,7 separately */
        __m256i v_one_by_15 = _mm256_set1_epi64x(I4_ONE_BY_15);
        /* Even elements: already in low 32 bits of each 64-bit lane after appropriate shuffle */
        __m256i src_abs_even = _mm256_and_si256(src_abs, _mm256_set1_epi64x(0xFFFFFFFF));
        __m256i prod_even = _mm256_mul_epi32(src_abs_even, v_one_by_15);
        __m256i ct_even = _mm256_srli_epi64(_mm256_add_epi64(prod_even, v_add), shift);

        /* Odd elements: shift right by 32 bits to position them */
        __m256i src_abs_odd = _mm256_srli_epi64(src_abs, 32);
        __m256i v_one_by_15_32 = _mm256_set1_epi32(I4_ONE_BY_15);
        __m256i prod_odd = _mm256_mul_epi32(src_abs_odd, v_one_by_15);
        __m256i ct_odd = _mm256_srli_epi64(_mm256_add_epi64(prod_odd, v_add), shift);

        /* Recombine: even results in low 32 of each 64-bit, odd in low 32 of each 64-bit */
        /* Pack back to 8 x int32 */
        /* ct_even has results at positions 0,_,2,_,4,_,6,_ (as 64-bit in low 32) */
        /* ct_odd has results at positions 1,_,3,_,5,_,7,_ (as 64-bit in low 32) */
        /* Shift odd left by 32, then OR */
        __m256i ct_odd_shifted = _mm256_slli_epi64(ct_odd, 32);
        __m256i center_term = _mm256_or_si256(
            _mm256_and_si256(ct_even, _mm256_set1_epi64x(0xFFFFFFFF)),
            ct_odd_shifted);

        __m256i row_center = _mm256_add_epi32(_mm256_add_epi32(c_l, c_r), center_term);

        /* Row below */
        __m256i b_l = _mm256_loadu_si256((const __m256i *)(flt_below + j - 1));
        __m256i b_c = _mm256_loadu_si256((const __m256i *)(flt_below + j));
        __m256i b_r = _mm256_loadu_si256((const __m256i *)(flt_below + j + 1));
        __m256i row_below = _mm256_add_epi32(_mm256_add_epi32(b_l, b_c), b_r);

        __m256i theta_sum = _mm256_add_epi32(_mm256_add_epi32(row_above, row_center), row_below);
        thr = _mm256_add_epi32(thr, theta_sum);
    }
    return thr;
}

/*
 * i4 AVX2 accumulation: process 8 int32 pixels, accumulate cubed values.
 * For i4 path: I4_ADM_CM_ACCUM_ROUND with shift_sub=0 (thr >> 0 = thr).
 */
static inline void i4_adm_cm_accum_round_avx2_8(
    __m256i x_vec, __m256i thr_vec,
    int32_t add_shift_sq, int32_t shift_sq,
    int32_t add_shift_cub, uint32_t shift_cub, int64_t *accum)
{
    __m256i v_zero = _mm256_setzero_si256();
    /* x = abs(x) - thr; x = max(x, 0) -- shift_sub is 0 for i4 */
    __m256i x_abs = _mm256_abs_epi32(x_vec);
    __m256i x_sub = _mm256_sub_epi32(x_abs, thr_vec);
    x_sub = _mm256_max_epi32(x_sub, v_zero);

    /* Process in two halves of 4 elements each for int64 precision */
    __m128i x_lo128 = _mm256_castsi256_si128(x_sub);
    __m128i x_hi128 = _mm256_extracti128_si256(x_sub, 1);
    __m256i x_lo = _mm256_cvtepi32_epi64(x_lo128);
    __m256i x_hi = _mm256_cvtepi32_epi64(x_hi128);

    __m256i add_sq64 = _mm256_set1_epi64x(add_shift_sq);
    __m256i add_cub64 = _mm256_set1_epi64x(add_shift_cub);

    /* x_sq = ((int64)x * x + add_sq) >> shift_sq */
    __m256i xsq_lo = _mm256_srli_epi64(
        _mm256_add_epi64(_mm256_mul_epi32(x_lo, x_lo), add_sq64), shift_sq);
    __m256i xsq_hi = _mm256_srli_epi64(
        _mm256_add_epi64(_mm256_mul_epi32(x_hi, x_hi), add_sq64), shift_sq);

    /* val = (x_sq * x + add_cub) >> shift_cub */
    __m256i val_lo = _mm256_srli_epi64(
        _mm256_add_epi64(_mm256_mul_epi32(xsq_lo, x_lo), add_cub64), shift_cub);
    __m256i val_hi = _mm256_srli_epi64(
        _mm256_add_epi64(_mm256_mul_epi32(xsq_hi, x_hi), add_cub64), shift_cub);

    /* Horizontal sum of 8 int64 values */
    __m256i sum_8 = _mm256_add_epi64(val_lo, val_hi);
    __m128i sum_lo = _mm256_castsi256_si128(sum_8);
    __m128i sum_hi = _mm256_extracti128_si256(sum_8, 1);
    __m128i sum_4 = _mm_add_epi64(sum_lo, sum_hi);
    __m128i sum_2 = _mm_add_epi64(sum_4, _mm_srli_si128(sum_4, 8));
    *accum += _mm_cvtsi128_si64(sum_2);
}

float i4_adm_cm_avx2(AdmBuffer *buf, int w, int h, int src_stride,
                      int csf_a_stride, int scale,
                      const float csf_factors[4][2])
{
    const i4_adm_dwt_band_t *src = &buf->i4_decouple_r;
    const i4_adm_dwt_band_t *csf_f = &buf->i4_csf_f;
    const i4_adm_dwt_band_t *csf_a = &buf->i4_csf_a;

    float factor1 = csf_factors[scale][0];
    float factor2 = csf_factors[scale][1];
    float rfactor1[3] = { 1.0f / factor1, 1.0f / factor1, 1.0f / factor2 };

    const uint32_t rfactor[3] = { (uint32_t)(rfactor1[0] * pow(2, 32)),
                                  (uint32_t)(rfactor1[1] * pow(2, 32)),
                                  (uint32_t)(rfactor1[2] * pow(2, 32)) };

    const uint32_t shift_dst[3] = { 28, 28, 28 };
    const uint32_t shift_flt[3] = { 32, 32, 32 };
    int32_t add_bef_shift_dst[3], add_bef_shift_flt[3];
    for (unsigned idx = 0; idx < 3; ++idx) {
        add_bef_shift_dst[idx] = (1u << (shift_dst[idx] - 1));
        add_bef_shift_flt[idx] = (1u << (shift_flt[idx] - 1));
    }

    uint32_t shift_cub = (uint32_t)ceil(log2(w));
    uint32_t add_shift_cub = (uint32_t)pow(2, (shift_cub - 1));
    uint32_t shift_inner_accum = (uint32_t)ceil(log2(h));
    uint32_t add_shift_inner_accum = (uint32_t)pow(2, (shift_inner_accum - 1));

    float final_shift[3] = { (float)pow(2,(45 - shift_cub - shift_inner_accum)),
                             (float)pow(2,(39 - shift_cub - shift_inner_accum)),
                             (float)pow(2,(36 - shift_cub - shift_inner_accum)) };

    const int32_t shift_sq = 30;
    const int32_t add_shift_sq = 536870912;
    const int32_t shift_sub = 0;

    int32_t *angles[3] = { csf_a->band_h, csf_a->band_v, csf_a->band_d };
    int32_t *flt_angles[3] = { csf_f->band_h, csf_f->band_v, csf_f->band_d };

    const int left = w * ADM_BORDER_FACTOR - 0.5;
    const int top = h * ADM_BORDER_FACTOR - 0.5;
    const int right = w - left;
    const int bottom = h - top;

    const int start_col = (left > 1) ? left : 1;
    const int end_col = (right < (w - 1)) ? right : (w - 1);
    const int start_row = (top > 1) ? top : 1;
    const int end_row = (bottom < (h - 1)) ? bottom : (h - 1);

    const int scale_idx = scale - 1;
    const int32_t add_bef_shift_dst_s = add_bef_shift_dst[scale_idx];
    const uint32_t shift_dst_s = shift_dst[scale_idx];
    const int32_t add_bef_shift_flt_s = add_bef_shift_flt[scale_idx];
    const uint32_t shift_flt_s = shift_flt[scale_idx];

    int i, j;
    int64_t accum_h = 0, accum_v = 0, accum_d = 0;
    int64_t accum_inner_h = 0, accum_inner_v = 0, accum_inner_d = 0;

    /* i=0 row */
    if (top <= 0) {
        if (left <= 0) {
            int32_t thr = i4_adm_cm_thresh_s_0_0_scalar(angles, flt_angles, csf_a_stride,
                                                         add_bef_shift_flt_s, shift_flt_s);
            int32_t xh = (int32_t)((((int64_t)src->band_h[0] * rfactor[0]) + add_bef_shift_dst_s) >> shift_dst_s);
            int32_t xv = (int32_t)((((int64_t)src->band_v[0] * rfactor[1]) + add_bef_shift_dst_s) >> shift_dst_s);
            int32_t xd = (int32_t)((((int64_t)src->band_d[0] * rfactor[2]) + add_bef_shift_dst_s) >> shift_dst_s);
            i4_adm_cm_accum_round_scalar(xh, thr, shift_sub, add_shift_sq, shift_sq, add_shift_cub, shift_cub, &accum_inner_h);
            i4_adm_cm_accum_round_scalar(xv, thr, shift_sub, add_shift_sq, shift_sq, add_shift_cub, shift_cub, &accum_inner_v);
            i4_adm_cm_accum_round_scalar(xd, thr, shift_sub, add_shift_sq, shift_sq, add_shift_cub, shift_cub, &accum_inner_d);
        }
        for (j = start_col; j < end_col; ++j) {
            int32_t thr = i4_adm_cm_thresh_s_0_j_scalar(angles, flt_angles, csf_a_stride, j,
                                                         add_bef_shift_flt_s, shift_flt_s);
            int32_t xh = (int32_t)((((int64_t)src->band_h[j] * rfactor[0]) + add_bef_shift_dst_s) >> shift_dst_s);
            int32_t xv = (int32_t)((((int64_t)src->band_v[j] * rfactor[1]) + add_bef_shift_dst_s) >> shift_dst_s);
            int32_t xd = (int32_t)((((int64_t)src->band_d[j] * rfactor[2]) + add_bef_shift_dst_s) >> shift_dst_s);
            i4_adm_cm_accum_round_scalar(xh, thr, shift_sub, add_shift_sq, shift_sq, add_shift_cub, shift_cub, &accum_inner_h);
            i4_adm_cm_accum_round_scalar(xv, thr, shift_sub, add_shift_sq, shift_sq, add_shift_cub, shift_cub, &accum_inner_v);
            i4_adm_cm_accum_round_scalar(xd, thr, shift_sub, add_shift_sq, shift_sq, add_shift_cub, shift_cub, &accum_inner_d);
        }
        if (right > (w - 1)) {
            int32_t thr = i4_adm_cm_thresh_s_0_wm1_scalar(angles, flt_angles, csf_a_stride, w,
                                                           add_bef_shift_flt_s, shift_flt_s);
            int32_t xh = (int32_t)((((int64_t)src->band_h[w - 1] * rfactor[0]) + add_bef_shift_dst_s) >> shift_dst_s);
            int32_t xv = (int32_t)((((int64_t)src->band_v[w - 1] * rfactor[1]) + add_bef_shift_dst_s) >> shift_dst_s);
            int32_t xd = (int32_t)((((int64_t)src->band_d[w - 1] * rfactor[2]) + add_bef_shift_dst_s) >> shift_dst_s);
            i4_adm_cm_accum_round_scalar(xh, thr, shift_sub, add_shift_sq, shift_sq, add_shift_cub, shift_cub, &accum_inner_h);
            i4_adm_cm_accum_round_scalar(xv, thr, shift_sub, add_shift_sq, shift_sq, add_shift_cub, shift_cub, &accum_inner_v);
            i4_adm_cm_accum_round_scalar(xd, thr, shift_sub, add_shift_sq, shift_sq, add_shift_cub, shift_cub, &accum_inner_d);
        }
    }
    accum_h += (accum_inner_h + add_shift_inner_accum) >> shift_inner_accum;
    accum_v += (accum_inner_v + add_shift_inner_accum) >> shift_inner_accum;
    accum_d += (accum_inner_d + add_shift_inner_accum) >> shift_inner_accum;

    /* Interior rows */
    const int do_left_boundary = (left <= 0);
    const int do_right_boundary = (right > (w - 1));

    for (i = start_row; i < end_row; ++i) {
        const int i_offset = i * src_stride;
        accum_inner_h = 0;
        accum_inner_v = 0;
        accum_inner_d = 0;

        if (do_left_boundary) {
            int32_t thr = i4_adm_cm_thresh_s_i_0_scalar(angles, flt_angles, csf_a_stride, i,
                                                         add_bef_shift_flt_s, shift_flt_s);
            int32_t xh = (int32_t)((((int64_t)src->band_h[i_offset] * rfactor[0]) + add_bef_shift_dst_s) >> shift_dst_s);
            int32_t xv = (int32_t)((((int64_t)src->band_v[i_offset] * rfactor[1]) + add_bef_shift_dst_s) >> shift_dst_s);
            int32_t xd = (int32_t)((((int64_t)src->band_d[i_offset] * rfactor[2]) + add_bef_shift_dst_s) >> shift_dst_s);
            i4_adm_cm_accum_round_scalar(xh, thr, shift_sub, add_shift_sq, shift_sq, add_shift_cub, shift_cub, &accum_inner_h);
            i4_adm_cm_accum_round_scalar(xv, thr, shift_sub, add_shift_sq, shift_sq, add_shift_cub, shift_cub, &accum_inner_v);
            i4_adm_cm_accum_round_scalar(xd, thr, shift_sub, add_shift_sq, shift_sq, add_shift_cub, shift_cub, &accum_inner_d);
        }

        /* AVX2 interior j-loop: 8 int32 pixels at a time */
        j = start_col;
        for (; j + 8 <= end_col; j += 8) {
            __m256i thr_vec = i4_adm_cm_thresh_avx2_8(angles, flt_angles, csf_a_stride, i, j,
                                                       add_bef_shift_flt_s, shift_flt_s);

            /* For each band: x = (int32_t)(((int64_t)src * rfactor + add) >> shift_dst)
             * Then do the accum_round with shift_sub=0, so threshold is thr directly.
             */
            /* Band H: compute x = (src * rfactor + add) >> shift scalar, then vectorize accum */
            {
                int32_t x_arr[8];
                for (int k = 0; k < 8; ++k) {
                    x_arr[k] = (int32_t)((((int64_t)src->band_h[i_offset + j + k] * rfactor[0]) +
                                           add_bef_shift_dst_s) >> shift_dst_s);
                }
                __m256i x_h = _mm256_loadu_si256((const __m256i *)x_arr);
                i4_adm_cm_accum_round_avx2_8(x_h, thr_vec, add_shift_sq, shift_sq, add_shift_cub, shift_cub, &accum_inner_h);
            }

            /* Band V */
            {
                int32_t x_arr[8];
                for (int k = 0; k < 8; ++k) {
                    x_arr[k] = (int32_t)((((int64_t)src->band_v[i_offset + j + k] * rfactor[1]) +
                                           add_bef_shift_dst_s) >> shift_dst_s);
                }
                __m256i x_v = _mm256_loadu_si256((const __m256i *)x_arr);
                i4_adm_cm_accum_round_avx2_8(x_v, thr_vec, add_shift_sq, shift_sq, add_shift_cub, shift_cub, &accum_inner_v);
            }

            /* Band D */
            {
                int32_t x_arr[8];
                for (int k = 0; k < 8; ++k) {
                    x_arr[k] = (int32_t)((((int64_t)src->band_d[i_offset + j + k] * rfactor[2]) +
                                           add_bef_shift_dst_s) >> shift_dst_s);
                }
                __m256i x_d = _mm256_loadu_si256((const __m256i *)x_arr);
                i4_adm_cm_accum_round_avx2_8(x_d, thr_vec, add_shift_sq, shift_sq, add_shift_cub, shift_cub, &accum_inner_d);
            }
        }

        /* Scalar remainder */
        for (; j < end_col; ++j) {
            int32_t thr = i4_adm_cm_thresh_s_i_j_scalar(angles, flt_angles, csf_a_stride, i, j,
                                                         add_bef_shift_flt_s, shift_flt_s);
            int32_t xh = (int32_t)((((int64_t)src->band_h[i_offset + j] * rfactor[0]) + add_bef_shift_dst_s) >> shift_dst_s);
            int32_t xv = (int32_t)((((int64_t)src->band_v[i_offset + j] * rfactor[1]) + add_bef_shift_dst_s) >> shift_dst_s);
            int32_t xd = (int32_t)((((int64_t)src->band_d[i_offset + j] * rfactor[2]) + add_bef_shift_dst_s) >> shift_dst_s);
            i4_adm_cm_accum_round_scalar(xh, thr, shift_sub, add_shift_sq, shift_sq, add_shift_cub, shift_cub, &accum_inner_h);
            i4_adm_cm_accum_round_scalar(xv, thr, shift_sub, add_shift_sq, shift_sq, add_shift_cub, shift_cub, &accum_inner_v);
            i4_adm_cm_accum_round_scalar(xd, thr, shift_sub, add_shift_sq, shift_sq, add_shift_cub, shift_cub, &accum_inner_d);
        }

        if (do_right_boundary) {
            int32_t thr = i4_adm_cm_thresh_s_i_wm1_scalar(angles, flt_angles, csf_a_stride, w, i,
                                                           add_bef_shift_flt_s, shift_flt_s);
            int32_t xh = (int32_t)((((int64_t)src->band_h[i_offset + w - 1] * rfactor[0]) + add_bef_shift_dst_s) >> shift_dst_s);
            int32_t xv = (int32_t)((((int64_t)src->band_v[i_offset + w - 1] * rfactor[1]) + add_bef_shift_dst_s) >> shift_dst_s);
            int32_t xd = (int32_t)((((int64_t)src->band_d[i_offset + w - 1] * rfactor[2]) + add_bef_shift_dst_s) >> shift_dst_s);
            i4_adm_cm_accum_round_scalar(xh, thr, shift_sub, add_shift_sq, shift_sq, add_shift_cub, shift_cub, &accum_inner_h);
            i4_adm_cm_accum_round_scalar(xv, thr, shift_sub, add_shift_sq, shift_sq, add_shift_cub, shift_cub, &accum_inner_v);
            i4_adm_cm_accum_round_scalar(xd, thr, shift_sub, add_shift_sq, shift_sq, add_shift_cub, shift_cub, &accum_inner_d);
        }

        accum_h += (accum_inner_h + add_shift_inner_accum) >> shift_inner_accum;
        accum_v += (accum_inner_v + add_shift_inner_accum) >> shift_inner_accum;
        accum_d += (accum_inner_d + add_shift_inner_accum) >> shift_inner_accum;
    }

    /* i=h-1 row */
    accum_inner_h = 0;
    accum_inner_v = 0;
    accum_inner_d = 0;
    if (bottom > (h - 1)) {
        const int hm1_offset = (h - 1) * src_stride;
        if (left <= 0) {
            int32_t thr = i4_adm_cm_thresh_s_hm1_0_scalar(angles, flt_angles, csf_a_stride, h,
                                                           add_bef_shift_flt_s, shift_flt_s);
            int32_t xh = (int32_t)((((int64_t)src->band_h[hm1_offset] * rfactor[0]) + add_bef_shift_dst_s) >> shift_dst_s);
            int32_t xv = (int32_t)((((int64_t)src->band_v[hm1_offset] * rfactor[1]) + add_bef_shift_dst_s) >> shift_dst_s);
            int32_t xd = (int32_t)((((int64_t)src->band_d[hm1_offset] * rfactor[2]) + add_bef_shift_dst_s) >> shift_dst_s);
            i4_adm_cm_accum_round_scalar(xh, thr, shift_sub, add_shift_sq, shift_sq, add_shift_cub, shift_cub, &accum_inner_h);
            i4_adm_cm_accum_round_scalar(xv, thr, shift_sub, add_shift_sq, shift_sq, add_shift_cub, shift_cub, &accum_inner_v);
            i4_adm_cm_accum_round_scalar(xd, thr, shift_sub, add_shift_sq, shift_sq, add_shift_cub, shift_cub, &accum_inner_d);
        }
        for (j = start_col; j < end_col; ++j) {
            int32_t thr = i4_adm_cm_thresh_s_hm1_j_scalar(angles, flt_angles, csf_a_stride, h, j,
                                                           add_bef_shift_flt_s, shift_flt_s);
            int32_t xh = (int32_t)((((int64_t)src->band_h[hm1_offset + j] * rfactor[0]) + add_bef_shift_dst_s) >> shift_dst_s);
            int32_t xv = (int32_t)((((int64_t)src->band_v[hm1_offset + j] * rfactor[1]) + add_bef_shift_dst_s) >> shift_dst_s);
            int32_t xd = (int32_t)((((int64_t)src->band_d[hm1_offset + j] * rfactor[2]) + add_bef_shift_dst_s) >> shift_dst_s);
            i4_adm_cm_accum_round_scalar(xh, thr, shift_sub, add_shift_sq, shift_sq, add_shift_cub, shift_cub, &accum_inner_h);
            i4_adm_cm_accum_round_scalar(xv, thr, shift_sub, add_shift_sq, shift_sq, add_shift_cub, shift_cub, &accum_inner_v);
            i4_adm_cm_accum_round_scalar(xd, thr, shift_sub, add_shift_sq, shift_sq, add_shift_cub, shift_cub, &accum_inner_d);
        }
        if (right > (w - 1)) {
            int32_t thr = i4_adm_cm_thresh_s_hm1_wm1_scalar(angles, flt_angles, csf_a_stride, w, h,
                                                             add_bef_shift_flt_s, shift_flt_s);
            int32_t xh = (int32_t)((((int64_t)src->band_h[hm1_offset + w - 1] * rfactor[0]) + add_bef_shift_dst_s) >> shift_dst_s);
            int32_t xv = (int32_t)((((int64_t)src->band_v[hm1_offset + w - 1] * rfactor[1]) + add_bef_shift_dst_s) >> shift_dst_s);
            int32_t xd = (int32_t)((((int64_t)src->band_d[hm1_offset + w - 1] * rfactor[2]) + add_bef_shift_dst_s) >> shift_dst_s);
            i4_adm_cm_accum_round_scalar(xh, thr, shift_sub, add_shift_sq, shift_sq, add_shift_cub, shift_cub, &accum_inner_h);
            i4_adm_cm_accum_round_scalar(xv, thr, shift_sub, add_shift_sq, shift_sq, add_shift_cub, shift_cub, &accum_inner_v);
            i4_adm_cm_accum_round_scalar(xd, thr, shift_sub, add_shift_sq, shift_sq, add_shift_cub, shift_cub, &accum_inner_d);
        }
    }
    accum_h += (accum_inner_h + add_shift_inner_accum) >> shift_inner_accum;
    accum_v += (accum_inner_v + add_shift_inner_accum) >> shift_inner_accum;
    accum_d += (accum_inner_d + add_shift_inner_accum) >> shift_inner_accum;

    float f_accum_h = (float)(accum_h / final_shift[scale_idx]);
    float f_accum_v = (float)(accum_v / final_shift[scale_idx]);
    float f_accum_d = (float)(accum_d / final_shift[scale_idx]);

    const float powf_add = powf((bottom - top) * (right - left) / 32.0f, 1.0f / 3.0f);
    float num_scale_h = powf(f_accum_h, 1.0f / 3.0f) + powf_add;
    float num_scale_v = powf(f_accum_v, 1.0f / 3.0f) + powf_add;
    float num_scale_d = powf(f_accum_d, 1.0f / 3.0f) + powf_add;

    return (num_scale_h + num_scale_v + num_scale_d);
}

void i4_adm_csf_avx2(AdmBuffer *RESTRICT buf, int scale, int w, int h,
                      int stride, const float csf_factors[4][2])
{
    const i4_adm_dwt_band_t *src = &buf->i4_decouple_a;
    const i4_adm_dwt_band_t *dst = &buf->i4_csf_a;
    const i4_adm_dwt_band_t *flt = &buf->i4_csf_f;

    const int32_t *src_angles[3] = { src->band_h, src->band_v, src->band_d };
    int32_t *dst_angles[3] = { dst->band_h, dst->band_v, dst->band_d };
    int32_t *flt_angles[3] = { flt->band_h, flt->band_v, flt->band_d };

    const float factor1 = csf_factors[scale][0];
    const float factor2 = csf_factors[scale][1];
    const float rfactor1[3] = { 1.0f / factor1, 1.0f / factor1, 1.0f / factor2 };

    const double pow2_32 = pow(2, 32);
    const uint32_t i_rfactor[3] = { (uint32_t)(rfactor1[0] * pow2_32),
                                    (uint32_t)(rfactor1[1] * pow2_32),
                                    (uint32_t)(rfactor1[2] * pow2_32) };

    const uint32_t FIX_ONE_BY_30 = 143165577;
    /* shift_dst is always 28, shift_flt is always 32 for all indices */
    const int32_t add_bef_shift_dst = (1u << 27); /* 1 << (28-1) */
    const int64_t add_bef_shift_flt = (1LL << 31);

    int left = w * ADM_BORDER_FACTOR - 0.5 - 1;
    int top = h * ADM_BORDER_FACTOR - 0.5 - 1;
    int right = w - left + 2;
    int bottom = h - top + 2;

    if (left < 0)   left = 0;
    if (right > w)   right = w;
    if (top < 0)     top = 0;
    if (bottom > h)  bottom = h;

    for (int theta = 0; theta < 3; ++theta)
    {
        const int32_t *src_ptr = src_angles[theta];
        int32_t *dst_ptr = dst_angles[theta];
        int32_t *flt_ptr = flt_angles[theta];

        const __m256i v_rf = _mm256_set1_epi32((int32_t)i_rfactor[theta]);
        const __m256i v_add_dst64 = _mm256_set1_epi64x((int64_t)add_bef_shift_dst);
        const __m256i v_fix30 = _mm256_set1_epi32((int32_t)FIX_ONE_BY_30);
        const __m256i v_add_flt64 = _mm256_set1_epi64x(add_bef_shift_flt);

        for (int i = top; i < bottom; ++i)
        {
            const int offset = i * stride;
            int j = left;

            /* AVX2 loop: process 8 int32 values per iteration.
             * We need uint32 * int32 -> int64 for the rfactor multiply.
             * Since i_rfactor may exceed INT32_MAX, we use unsigned multiply
             * on abs(src) and then conditionally negate. */
            for (; j + 7 < right; j += 8)
            {
                __m256i sv = _mm256_loadu_si256(
                    (const __m256i *)(src_ptr + offset + j));

                /* Compute absolute value and sign for conditional negation */
                __m256i abs_sv = _mm256_abs_epi32(sv);
                /* sign_mask: 0xFFFFFFFF for negative, 0x00000000 for non-negative */
                __m256i sign32 = _mm256_srai_epi32(sv, 31);

                /* Group A: elements 0,2,4,6 - unsigned multiply */
                __m256i mag_a = _mm256_mul_epu32(v_rf, abs_sv);

                /* Group B: elements 1,3,5,7 */
                __m256i abs_odd = _mm256_srli_epi64(abs_sv, 32);
                __m256i mag_b = _mm256_mul_epu32(v_rf, abs_odd);

                /* Expand sign to 64-bit lanes for conditional negation.
                 * sign32 has {s0,s1,s2,s3,s4,s5,s6,s7} where each is 0 or -1.
                 * For even elements (0,2,4,6): duplicate each into 64-bit lane
                 * using shuffle 0xA0 = {0,0,2,2} within each 128-bit half.
                 * For odd elements (1,3,5,7): use shuffle 0xF5 = {1,1,3,3}. */
                __m256i sign64_a = _mm256_shuffle_epi32(sign32, 0xA0);
                __m256i sign64_b = _mm256_shuffle_epi32(sign32, 0xF5);

                /* Conditional negate: (mag ^ sign) - sign */
                __m256i prod_a = _mm256_sub_epi64(
                    _mm256_xor_si256(mag_a, sign64_a), sign64_a);
                __m256i prod_b = _mm256_sub_epi64(
                    _mm256_xor_si256(mag_b, sign64_b), sign64_b);

                /* Add rounding constant */
                prod_a = _mm256_add_epi64(prod_a, v_add_dst64);
                prod_b = _mm256_add_epi64(prod_b, v_add_dst64);

                /* Arithmetic right shift by 28 (no native epi64 srai in AVX2) */
                __m256i srl_a = _mm256_srli_epi64(prod_a, 28);
                __m256i sa = _mm256_shuffle_epi32(
                    _mm256_srai_epi32(prod_a, 31), 0xF5);
                prod_a = _mm256_or_si256(srl_a,
                    _mm256_slli_epi64(sa, 36));

                __m256i srl_b = _mm256_srli_epi64(prod_b, 28);
                __m256i sb = _mm256_shuffle_epi32(
                    _mm256_srai_epi32(prod_b, 31), 0xF5);
                prod_b = _mm256_or_si256(srl_b,
                    _mm256_slli_epi64(sb, 36));

                /* Interleave even/odd results back into 8x int32 */
                __m256i b_shifted = _mm256_slli_epi64(prod_b, 32);
                __m256i dst_val = _mm256_blend_epi32(prod_a, b_shifted, 0xAA);

                _mm256_storeu_si256((__m256i *)(dst_ptr + offset + j), dst_val);

                /* Compute flt = (FIX_ONE_BY_30 * abs(dst_val) + 2^31) >> 32 */
                __m256i abs_dst = _mm256_abs_epi32(dst_val);

                __m256i flt_a = _mm256_mul_epu32(v_fix30, abs_dst);
                __m256i abs_dst_odd = _mm256_srli_epi64(abs_dst, 32);
                __m256i flt_b = _mm256_mul_epu32(v_fix30, abs_dst_odd);

                flt_a = _mm256_add_epi64(flt_a, v_add_flt64);
                flt_b = _mm256_add_epi64(flt_b, v_add_flt64);
                flt_a = _mm256_srli_epi64(flt_a, 32);
                flt_b = _mm256_srli_epi64(flt_b, 32);

                __m256i fb_shifted = _mm256_slli_epi64(flt_b, 32);
                __m256i flt_val = _mm256_blend_epi32(flt_a, fb_shifted, 0xAA);

                _mm256_storeu_si256((__m256i *)(flt_ptr + offset + j), flt_val);
            }

            /* Scalar tail */
            for (; j < right; ++j)
            {
                int32_t dst_val = (int32_t)(((i_rfactor[theta] * (int64_t)src_ptr[offset + j]) +
                    add_bef_shift_dst) >> 28);
                dst_ptr[offset + j] = dst_val;
                flt_ptr[offset + j] = (int32_t)((((int64_t)FIX_ONE_BY_30 * abs(dst_val)) +
                    add_bef_shift_flt) >> 32);
            }
        }
    }
}

float adm_csf_den_s123_avx2(const i4_adm_dwt_band_t *RESTRICT src, int scale,
                             int w, int h, int src_stride,
                             const float csf_factors[4][2])
{
    float factor1 = csf_factors[scale][0];
    float factor2 = csf_factors[scale][1];
    const float rfactor[3] = { 1.0f / factor1, 1.0f / factor1, 1.0f / factor2 };

    uint64_t accum_h = 0, accum_v = 0, accum_d = 0;
    const uint32_t shift_sq[3] = { 31, 30, 31 };
    const uint32_t accum_convert_float[3] = { 32, 27, 23 };
    const uint32_t add_shift_sq[3] =
        { 1u << shift_sq[0], 1u << shift_sq[1], 1u << shift_sq[2] };

    const int left = w * ADM_BORDER_FACTOR - 0.5;
    const int top = h * ADM_BORDER_FACTOR - 0.5;
    const int right = w - left;
    const int bottom = h - top;

    uint32_t shift_cub = (uint32_t)ceil(log2(right - left));
    uint32_t add_shift_cub = (uint32_t)pow(2, (shift_cub - 1));
    uint32_t shift_accum = (uint32_t)ceil(log2(bottom - top));
    uint32_t add_shift_accum = (uint32_t)pow(2, (shift_accum - 1));

    const int scale_idx = scale - 1;
    const uint32_t sq_shift = shift_sq[scale_idx];
    const uint64_t sq_add = add_shift_sq[scale_idx];

    int32_t *src_h_ptr = src->band_h + top * src_stride;
    int32_t *src_v_ptr = src->band_v + top * src_stride;
    int32_t *src_d_ptr = src->band_d + top * src_stride;

    const __m256i v_sq_add = _mm256_set1_epi64x((int64_t)sq_add);
    const __m256i v_cub_add = _mm256_set1_epi64x((int64_t)add_shift_cub);

    for (int i = top; i < bottom; ++i)
    {
        uint64_t accum_inner_h = 0;
        uint64_t accum_inner_v = 0;
        uint64_t accum_inner_d = 0;
        int j = left;

        /* AVX2: process 4 int32 elements at a time using 32x32->64 multiply */
        for (; j + 3 < right; j += 4)
        {
            /* band_h */
            {
                __m256i raw = _mm256_loadu_si256((const __m256i *)(src_h_ptr + j));
                __m256i absv = _mm256_abs_epi32(raw);
                __m128i abs128 = _mm256_castsi256_si128(absv);
                __m256i abs64 = _mm256_cvtepu32_epi64(abs128);
                __m256i sq = _mm256_mul_epu32(abs64, abs64);
                sq = _mm256_add_epi64(sq, v_sq_add);
                sq = _mm256_srli_epi64(sq, sq_shift);
                __m256i cube = _mm256_mul_epu32(sq, abs64);
                cube = _mm256_add_epi64(cube, v_cub_add);
                cube = _mm256_srli_epi64(cube, shift_cub);
                __m128i lo = _mm256_castsi256_si128(cube);
                __m128i hi = _mm256_extracti128_si256(cube, 1);
                __m128i sum128 = _mm_add_epi64(lo, hi);
                __m128i sum_hi = _mm_unpackhi_epi64(sum128, sum128);
                accum_inner_h += (uint64_t)_mm_cvtsi128_si64(
                    _mm_add_epi64(sum128, sum_hi));
            }
            /* band_v */
            {
                __m256i raw = _mm256_loadu_si256((const __m256i *)(src_v_ptr + j));
                __m256i absv = _mm256_abs_epi32(raw);
                __m128i abs128 = _mm256_castsi256_si128(absv);
                __m256i abs64 = _mm256_cvtepu32_epi64(abs128);
                __m256i sq = _mm256_mul_epu32(abs64, abs64);
                sq = _mm256_add_epi64(sq, v_sq_add);
                sq = _mm256_srli_epi64(sq, sq_shift);
                __m256i cube = _mm256_mul_epu32(sq, abs64);
                cube = _mm256_add_epi64(cube, v_cub_add);
                cube = _mm256_srli_epi64(cube, shift_cub);
                __m128i lo = _mm256_castsi256_si128(cube);
                __m128i hi = _mm256_extracti128_si256(cube, 1);
                __m128i sum128 = _mm_add_epi64(lo, hi);
                __m128i sum_hi = _mm_unpackhi_epi64(sum128, sum128);
                accum_inner_v += (uint64_t)_mm_cvtsi128_si64(
                    _mm_add_epi64(sum128, sum_hi));
            }
            /* band_d */
            {
                __m256i raw = _mm256_loadu_si256((const __m256i *)(src_d_ptr + j));
                __m256i absv = _mm256_abs_epi32(raw);
                __m128i abs128 = _mm256_castsi256_si128(absv);
                __m256i abs64 = _mm256_cvtepu32_epi64(abs128);
                __m256i sq = _mm256_mul_epu32(abs64, abs64);
                sq = _mm256_add_epi64(sq, v_sq_add);
                sq = _mm256_srli_epi64(sq, sq_shift);
                __m256i cube = _mm256_mul_epu32(sq, abs64);
                cube = _mm256_add_epi64(cube, v_cub_add);
                cube = _mm256_srli_epi64(cube, shift_cub);
                __m128i lo = _mm256_castsi256_si128(cube);
                __m128i hi = _mm256_extracti128_si256(cube, 1);
                __m128i sum128 = _mm_add_epi64(lo, hi);
                __m128i sum_hi = _mm_unpackhi_epi64(sum128, sum128);
                accum_inner_d += (uint64_t)_mm_cvtsi128_si64(
                    _mm_add_epi64(sum128, sum_hi));
            }
        }

        /* Scalar tail */
        for (; j < right; ++j)
        {
            uint32_t hv = (uint32_t)abs(src_h_ptr[j]);
            uint32_t vv = (uint32_t)abs(src_v_ptr[j]);
            uint32_t dv = (uint32_t)abs(src_d_ptr[j]);
            uint64_t val;
            val = ((((((uint64_t)hv * hv) + sq_add) >> sq_shift) * hv)
                + add_shift_cub) >> shift_cub;
            accum_inner_h += val;
            val = ((((((uint64_t)vv * vv) + sq_add) >> sq_shift) * vv)
                + add_shift_cub) >> shift_cub;
            accum_inner_v += val;
            val = ((((((uint64_t)dv * dv) + sq_add) >> sq_shift) * dv)
                + add_shift_cub) >> shift_cub;
            accum_inner_d += val;
        }

        accum_h += (accum_inner_h + add_shift_accum) >> shift_accum;
        accum_v += (accum_inner_v + add_shift_accum) >> shift_accum;
        accum_d += (accum_inner_d + add_shift_accum) >> shift_accum;

        src_h_ptr += src_stride;
        src_v_ptr += src_stride;
        src_d_ptr += src_stride;
    }

    double shift_csf = pow(2, (accum_convert_float[scale_idx] - shift_accum - shift_cub));
    double csf_h = (double)(accum_h / shift_csf) * pow(rfactor[0], 3);
    double csf_v = (double)(accum_v / shift_csf) * pow(rfactor[1], 3);
    double csf_d = (double)(accum_d / shift_csf) * pow(rfactor[2], 3);

    float powf_add = powf((bottom - top) * (right - left) / 32.0f, 1.0f / 3.0f);
    float den_scale_h = powf(csf_h, 1.0f / 3.0f) + powf_add;
    float den_scale_v = powf(csf_v, 1.0f / 3.0f) + powf_add;
    float den_scale_d = powf(csf_d, 1.0f / 3.0f) + powf_add;

    return (den_scale_h + den_scale_v + den_scale_d);
}

float adm_csf_den_scale_avx2(const adm_dwt_band_t *RESTRICT src, int w, int h,
                              int src_stride, const float csf_factors[4][2])
{
    const float factor1 = csf_factors[0][0];
    const float factor2 = csf_factors[0][1];
    const float rfactor[3] = { 1.0f / factor1, 1.0f / factor1, 1.0f / factor2 };

    uint64_t accum_h = 0, accum_v = 0, accum_d = 0;

    const int left = w * ADM_BORDER_FACTOR - 0.5;
    const int top = h * ADM_BORDER_FACTOR - 0.5;
    const int right = w - left;
    const int bottom = h - top;

    int32_t shift_accum = (int32_t)ceil(log2((bottom - top)*(right - left)) - 20);
    shift_accum = shift_accum > 0 ? shift_accum : 0;
    int32_t add_shift_accum =
        shift_accum > 0 ? (1 << (shift_accum - 1)) : 0;

    int16_t *src_hb = src->band_h + top * src_stride;
    int16_t *src_vb = src->band_v + top * src_stride;
    int16_t *src_db = src->band_d + top * src_stride;

    for (int i = top; i < bottom; ++i) {
        uint64_t accum_inner_h = 0;
        uint64_t accum_inner_v = 0;
        uint64_t accum_inner_d = 0;
        int j = left;

        /* AVX2: process 16 int16 elements at a time.
         * abs(x)^3 where abs(x) is uint16 (max 32768).
         * Square via 32-bit multiply, then cube via 32x32->64. */
        for (; j + 15 < right; j += 16)
        {
            __m256i h_raw = _mm256_loadu_si256((const __m256i *)(src_hb + j));
            __m256i v_raw = _mm256_loadu_si256((const __m256i *)(src_vb + j));
            __m256i d_raw = _mm256_loadu_si256((const __m256i *)(src_db + j));

            __m256i h_abs = _mm256_abs_epi16(h_raw);
            __m256i v_abs = _mm256_abs_epi16(v_raw);
            __m256i d_abs = _mm256_abs_epi16(d_raw);

            __m256i zero = _mm256_setzero_si256();
            __m256i h_lo16 = _mm256_unpacklo_epi16(h_abs, zero);
            __m256i h_hi16 = _mm256_unpackhi_epi16(h_abs, zero);
            __m256i h_sq_lo = _mm256_mullo_epi32(h_lo16, h_lo16);
            __m256i h_sq_hi = _mm256_mullo_epi32(h_hi16, h_hi16);

            /* band_h cube via even/odd 32x32->64 */
            {
                __m256i cube_a = _mm256_mul_epu32(h_sq_lo, h_lo16);
                __m256i cube_b = _mm256_mul_epu32(
                    _mm256_srli_epi64(h_sq_lo, 32),
                    _mm256_srli_epi64(h_lo16, 32));
                __m256i sum_ab = _mm256_add_epi64(cube_a, cube_b);
                __m256i cube_c = _mm256_mul_epu32(h_sq_hi, h_hi16);
                __m256i cube_d = _mm256_mul_epu32(
                    _mm256_srli_epi64(h_sq_hi, 32),
                    _mm256_srli_epi64(h_hi16, 32));
                __m256i sum_cd = _mm256_add_epi64(cube_c, cube_d);
                __m256i total = _mm256_add_epi64(sum_ab, sum_cd);
                __m128i lo128 = _mm256_castsi256_si128(total);
                __m128i hi128 = _mm256_extracti128_si256(total, 1);
                __m128i s128 = _mm_add_epi64(lo128, hi128);
                accum_inner_h += (uint64_t)_mm_cvtsi128_si64(s128) +
                    (uint64_t)_mm_cvtsi128_si64(_mm_unpackhi_epi64(s128, s128));
            }

            /* band_v */
            {
                __m256i v_lo16 = _mm256_unpacklo_epi16(v_abs, zero);
                __m256i v_hi16 = _mm256_unpackhi_epi16(v_abs, zero);
                __m256i v_sq_lo = _mm256_mullo_epi32(v_lo16, v_lo16);
                __m256i v_sq_hi = _mm256_mullo_epi32(v_hi16, v_hi16);
                __m256i cube_a = _mm256_mul_epu32(v_sq_lo, v_lo16);
                __m256i cube_b = _mm256_mul_epu32(
                    _mm256_srli_epi64(v_sq_lo, 32),
                    _mm256_srli_epi64(v_lo16, 32));
                __m256i sum_ab = _mm256_add_epi64(cube_a, cube_b);
                __m256i cube_c = _mm256_mul_epu32(v_sq_hi, v_hi16);
                __m256i cube_d = _mm256_mul_epu32(
                    _mm256_srli_epi64(v_sq_hi, 32),
                    _mm256_srli_epi64(v_hi16, 32));
                __m256i sum_cd = _mm256_add_epi64(cube_c, cube_d);
                __m256i total = _mm256_add_epi64(sum_ab, sum_cd);
                __m128i lo128 = _mm256_castsi256_si128(total);
                __m128i hi128 = _mm256_extracti128_si256(total, 1);
                __m128i s128 = _mm_add_epi64(lo128, hi128);
                accum_inner_v += (uint64_t)_mm_cvtsi128_si64(s128) +
                    (uint64_t)_mm_cvtsi128_si64(_mm_unpackhi_epi64(s128, s128));
            }

            /* band_d */
            {
                __m256i d_lo16 = _mm256_unpacklo_epi16(d_abs, zero);
                __m256i d_hi16 = _mm256_unpackhi_epi16(d_abs, zero);
                __m256i d_sq_lo = _mm256_mullo_epi32(d_lo16, d_lo16);
                __m256i d_sq_hi = _mm256_mullo_epi32(d_hi16, d_hi16);
                __m256i cube_a = _mm256_mul_epu32(d_sq_lo, d_lo16);
                __m256i cube_b = _mm256_mul_epu32(
                    _mm256_srli_epi64(d_sq_lo, 32),
                    _mm256_srli_epi64(d_lo16, 32));
                __m256i sum_ab = _mm256_add_epi64(cube_a, cube_b);
                __m256i cube_c = _mm256_mul_epu32(d_sq_hi, d_hi16);
                __m256i cube_d = _mm256_mul_epu32(
                    _mm256_srli_epi64(d_sq_hi, 32),
                    _mm256_srli_epi64(d_hi16, 32));
                __m256i sum_cd = _mm256_add_epi64(cube_c, cube_d);
                __m256i total = _mm256_add_epi64(sum_ab, sum_cd);
                __m128i lo128 = _mm256_castsi256_si128(total);
                __m128i hi128 = _mm256_extracti128_si256(total, 1);
                __m128i s128 = _mm_add_epi64(lo128, hi128);
                accum_inner_d += (uint64_t)_mm_cvtsi128_si64(s128) +
                    (uint64_t)_mm_cvtsi128_si64(_mm_unpackhi_epi64(s128, s128));
            }
        }

        /* Scalar tail */
        for (; j < right; ++j) {
            uint16_t hv = (uint16_t)abs(src_hb[j]);
            uint16_t vv = (uint16_t)abs(src_vb[j]);
            uint16_t dv = (uint16_t)abs(src_db[j]);
            accum_inner_h += ((uint64_t)hv * hv) * hv;
            accum_inner_v += ((uint64_t)vv * vv) * vv;
            accum_inner_d += ((uint64_t)dv * dv) * dv;
        }

        accum_h += (accum_inner_h + add_shift_accum) >> shift_accum;
        accum_v += (accum_inner_v + add_shift_accum) >> shift_accum;
        accum_d += (accum_inner_d + add_shift_accum) >> shift_accum;
        src_hb += src_stride;
        src_vb += src_stride;
        src_db += src_stride;
    }

    double shift_csf = pow(2, (18 - shift_accum));
    double csf_h = (double)(accum_h / shift_csf) * pow(rfactor[0], 3);
    double csf_v = (double)(accum_v / shift_csf) * pow(rfactor[1], 3);
    double csf_d = (double)(accum_d / shift_csf) * pow(rfactor[2], 3);

    float powf_add = powf((bottom - top) * (right - left) / 32.0f, 1.0f / 3.0f);
    float den_scale_h = powf(csf_h, 1.0f / 3.0f) + powf_add;
    float den_scale_v = powf(csf_v, 1.0f / 3.0f) + powf_add;
    float den_scale_d = powf(csf_d, 1.0f / 3.0f) + powf_add;

    return(den_scale_h + den_scale_v + den_scale_d);
}

/*
 * Scalar horizontal pass helper for s1_combined and s123_combined.
 * Processes one output pixel for one stream (ref or dis), writing to
 * band_a, band_v, band_h, band_d at the given position.
 */
static inline void dwt2_horz_scalar_one(
    const int32_t *tmplo, const int32_t *tmphi,
    const int *j0p, const int *j1p, const int *j2p, const int *j3p,
    const int16_t *filter_lo, const int16_t *filter_hi,
    int32_t *band_a, int32_t *band_v, int32_t *band_h, int32_t *band_d,
    int out_idx, int32_t add_round, int shift)
{
    int j0 = *j0p, j1 = *j1p, j2 = *j2p, j3 = *j3p;
    int32_t s0, s1, s2, s3;
    int64_t acc;

    s0 = tmplo[j0]; s1 = tmplo[j1]; s2 = tmplo[j2]; s3 = tmplo[j3];
    acc  = (int64_t)filter_lo[0] * s0;
    acc += (int64_t)filter_lo[1] * s1;
    acc += (int64_t)filter_lo[2] * s2;
    acc += (int64_t)filter_lo[3] * s3;
    band_a[out_idx] = (int32_t)((acc + add_round) >> shift);

    acc  = (int64_t)filter_hi[0] * s0;
    acc += (int64_t)filter_hi[1] * s1;
    acc += (int64_t)filter_hi[2] * s2;
    acc += (int64_t)filter_hi[3] * s3;
    band_v[out_idx] = (int32_t)((acc + add_round) >> shift);

    s0 = tmphi[j0]; s1 = tmphi[j1]; s2 = tmphi[j2]; s3 = tmphi[j3];
    acc  = (int64_t)filter_lo[0] * s0;
    acc += (int64_t)filter_lo[1] * s1;
    acc += (int64_t)filter_lo[2] * s2;
    acc += (int64_t)filter_lo[3] * s3;
    band_h[out_idx] = (int32_t)((acc + add_round) >> shift);

    acc  = (int64_t)filter_hi[0] * s0;
    acc += (int64_t)filter_hi[1] * s1;
    acc += (int64_t)filter_hi[2] * s2;
    acc += (int64_t)filter_hi[3] * s3;
    band_d[out_idx] = (int32_t)((acc + add_round) >> shift);
}

/*
 * AVX2 horizontal pass for int32 intermediate -> int32 output.
 * Processes 4 output elements at a time for one stream.
 * Uses gather for stride-2 access pattern of the FIR filter.
 *
 * For interior pixels (j_start=1 to w_half-3), ind_x[k][j] = 2j + k - 1.
 * For 4 output positions j, j+1, j+2, j+3:
 *   tap0[n] = tmp[2(j+n) - 1]  (stride-2 starting at 2j-1)
 *   tap1[n] = tmp[2(j+n)]      (stride-2 starting at 2j)
 *   tap2[n] = tmp[2(j+n) + 1]  (stride-2 starting at 2j+1)
 *   tap3[n] = tmp[2(j+n) + 2]  (stride-2 starting at 2j+2)
 */
static inline void dwt2_horz_avx2_4(
    const int32_t *tmp, int base,
    __m128i coeff0, __m128i coeff1, __m128i coeff2, __m128i coeff3,
    __m256i rounding, int shift,
    int32_t *out, int out_offset)
{
    /* Load contiguous int32 values and extract stride-2 taps via permute.
     * For 4 outputs at positions j..j+3 (base = 2j-1):
     *   t0 = [base+0, base+2, base+4, base+6] (even of block0)
     *   t1 = [base+1, base+3, base+5, base+7] (odd  of block0)
     *   t2 = [base+2, base+4, base+6, base+8] (even of block1)
     *   t3 = [base+3, base+5, base+7, base+9] (odd  of block1)
     * Two 256-bit loads cover the needed range. */
    __m256i block0 = _mm256_loadu_si256((const __m256i *)(tmp + base));      /* [0..7] */
    __m256i block1 = _mm256_loadu_si256((const __m256i *)(tmp + base + 2));  /* [2..9] */

    /* Permute indices to extract even/odd elements into low 128 bits */
    __m256i perm_even = _mm256_set_epi32(0, 0, 0, 0, 6, 4, 2, 0);
    __m256i perm_odd  = _mm256_set_epi32(0, 0, 0, 0, 7, 5, 3, 1);

    __m128i t0 = _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(block0, perm_even));
    __m128i t1 = _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(block0, perm_odd));
    __m128i t2 = _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(block1, perm_even));
    __m128i t3 = _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(block1, perm_odd));

    /* Multiply each tap by its coefficient and accumulate in int64.
     * _mm256_mul_epi32 multiplies even 32-bit lanes producing 64-bit results.
     * For 4 elements, promote __m128i to __m256i and use _mm256_mul_epi32
     * for even lanes, then shuffle for odd lanes. */

    /* Widen tap values and coefficients to 256-bit for int64 multiplication */
    __m256i w_t0 = _mm256_cvtepi32_epi64(t0);
    __m256i w_t1 = _mm256_cvtepi32_epi64(t1);
    __m256i w_t2 = _mm256_cvtepi32_epi64(t2);
    __m256i w_t3 = _mm256_cvtepi32_epi64(t3);

    __m256i w_c0 = _mm256_cvtepi32_epi64(coeff0);
    __m256i w_c1 = _mm256_cvtepi32_epi64(coeff1);
    __m256i w_c2 = _mm256_cvtepi32_epi64(coeff2);
    __m256i w_c3 = _mm256_cvtepi32_epi64(coeff3);

    /* Multiply-accumulate in int64: acc = c0*t0 + c1*t1 + c2*t2 + c3*t3 */
    __m256i acc = _mm256_mul_epi32(w_t0, w_c0);  /* 4 int64 products */
    acc = _mm256_add_epi64(acc, _mm256_mul_epi32(w_t1, w_c1));
    acc = _mm256_add_epi64(acc, _mm256_mul_epi32(w_t2, w_c2));
    acc = _mm256_add_epi64(acc, _mm256_mul_epi32(w_t3, w_c3));

    /* Add rounding and shift */
    acc = _mm256_add_epi64(acc, rounding);
    acc = srai_epi64_256(acc, shift);

    /* Truncate int64 -> int32: extract low 32 bits of each 64-bit lane */
    /* Shuffle to pack: we need elements at positions 0,2,4,6 (32-bit view) */
    __m256i shuffled = _mm256_shuffle_epi32(acc, 0x08); /* 00 00 10 00: pack pairs */
    /* Extract 128-bit halves and combine */
    __m128i lo = _mm256_castsi256_si128(shuffled);
    __m128i hi = _mm256_extracti128_si256(shuffled, 1);
    __m128i result = _mm_unpacklo_epi64(lo, hi);

    _mm_storeu_si128((__m128i *)(out + out_offset), result);
}

void adm_dwt2_s1_combined_avx2(const int16_t *i2_ref_scale,
                                const int16_t *i2_dis_scale,
                                AdmBuffer *buf, int w, int h,
                                int ref_stride, int dis_stride,
                                int dst_stride)
{
    const i4_adm_dwt_band_t *i4_ref_dwt2 = &buf->i4_ref_dwt2;
    const i4_adm_dwt_band_t *i4_dis_dwt2 = &buf->i4_dis_dwt2;
    int **ind_y = buf->ind_y;
    int **ind_x = buf->ind_x;

    const int16_t *filter_lo = dwt2_db2_coeffs_lo;
    const int16_t *filter_hi = dwt2_db2_coeffs_hi;

    /* Scale 1 constants: VP shift=0, HP shift=15, HP round=16384 */
    const int32_t add_bef_shift_round_HP = 16384;
    const int16_t shift_HP = 15;

    int32_t *tmplo_ref = buf->tmp_ref;
    int32_t *tmphi_ref = tmplo_ref + w;
    int32_t *tmplo_dis = tmphi_ref + w;
    int32_t *tmphi_dis = tmplo_dis + w;

    /* Broadcast filter coefficients as int32 for vertical pass mullo */
    __m256i vf_lo0 = _mm256_set1_epi32((int32_t)filter_lo[0]);
    __m256i vf_lo1 = _mm256_set1_epi32((int32_t)filter_lo[1]);
    __m256i vf_lo2 = _mm256_set1_epi32((int32_t)filter_lo[2]);
    __m256i vf_lo3 = _mm256_set1_epi32((int32_t)filter_lo[3]);
    __m256i vf_hi0 = _mm256_set1_epi32((int32_t)filter_hi[0]);
    __m256i vf_hi1 = _mm256_set1_epi32((int32_t)filter_hi[1]);
    __m256i vf_hi2 = _mm256_set1_epi32((int32_t)filter_hi[2]);
    __m256i vf_hi3 = _mm256_set1_epi32((int32_t)filter_hi[3]);

    /* Coefficients for horizontal pass as 128-bit broadcasts (for 4-element processing) */
    __m128i hc_lo0 = _mm_set1_epi32((int32_t)filter_lo[0]);
    __m128i hc_lo1 = _mm_set1_epi32((int32_t)filter_lo[1]);
    __m128i hc_lo2 = _mm_set1_epi32((int32_t)filter_lo[2]);
    __m128i hc_lo3 = _mm_set1_epi32((int32_t)filter_lo[3]);
    __m128i hc_hi0 = _mm_set1_epi32((int32_t)filter_hi[0]);
    __m128i hc_hi1 = _mm_set1_epi32((int32_t)filter_hi[1]);
    __m128i hc_hi2 = _mm_set1_epi32((int32_t)filter_hi[2]);
    __m128i hc_hi3 = _mm_set1_epi32((int32_t)filter_hi[3]);

    __m256i h_rounding = _mm256_set1_epi64x(add_bef_shift_round_HP);

    int w_half = (w + 1) / 2;

    for (int i = 0; i < (h + 1) / 2; ++i)
    {
        /* --- Vertical pass: int16 input -> int32 output, no shift --- */
        const int16_t *ref_row0 = i2_ref_scale + ind_y[0][i] * ref_stride;
        const int16_t *ref_row1 = i2_ref_scale + ind_y[1][i] * ref_stride;
        const int16_t *ref_row2 = i2_ref_scale + ind_y[2][i] * ref_stride;
        const int16_t *ref_row3 = i2_ref_scale + ind_y[3][i] * ref_stride;

        const int16_t *dis_row0 = i2_dis_scale + ind_y[0][i] * dis_stride;
        const int16_t *dis_row1 = i2_dis_scale + ind_y[1][i] * dis_stride;
        const int16_t *dis_row2 = i2_dis_scale + ind_y[2][i] * dis_stride;
        const int16_t *dis_row3 = i2_dis_scale + ind_y[3][i] * dis_stride;

        int j;
        for (j = 0; j + 7 < w; j += 8)
        {
            /* Load 8 int16 from each row, widen to int32 */
            __m256i r0 = _mm256_cvtepi16_epi32(_mm_loadu_si128((__m128i *)(ref_row0 + j)));
            __m256i r1 = _mm256_cvtepi16_epi32(_mm_loadu_si128((__m128i *)(ref_row1 + j)));
            __m256i r2 = _mm256_cvtepi16_epi32(_mm_loadu_si128((__m128i *)(ref_row2 + j)));
            __m256i r3 = _mm256_cvtepi16_epi32(_mm_loadu_si128((__m128i *)(ref_row3 + j)));

            /* Lo filter: lo0*r0 + lo1*r1 + lo2*r2 + lo3*r3 */
            __m256i lo_acc = _mm256_mullo_epi32(vf_lo0, r0);
            lo_acc = _mm256_add_epi32(lo_acc, _mm256_mullo_epi32(vf_lo1, r1));
            lo_acc = _mm256_add_epi32(lo_acc, _mm256_mullo_epi32(vf_lo2, r2));
            lo_acc = _mm256_add_epi32(lo_acc, _mm256_mullo_epi32(vf_lo3, r3));
            /* No shift for scale 1 vertical pass */
            _mm256_storeu_si256((__m256i *)(tmplo_ref + j), lo_acc);

            /* Hi filter */
            __m256i hi_acc = _mm256_mullo_epi32(vf_hi0, r0);
            hi_acc = _mm256_add_epi32(hi_acc, _mm256_mullo_epi32(vf_hi1, r1));
            hi_acc = _mm256_add_epi32(hi_acc, _mm256_mullo_epi32(vf_hi2, r2));
            hi_acc = _mm256_add_epi32(hi_acc, _mm256_mullo_epi32(vf_hi3, r3));
            _mm256_storeu_si256((__m256i *)(tmphi_ref + j), hi_acc);

            /* Distorted */
            __m256i d0 = _mm256_cvtepi16_epi32(_mm_loadu_si128((__m128i *)(dis_row0 + j)));
            __m256i d1 = _mm256_cvtepi16_epi32(_mm_loadu_si128((__m128i *)(dis_row1 + j)));
            __m256i d2 = _mm256_cvtepi16_epi32(_mm_loadu_si128((__m128i *)(dis_row2 + j)));
            __m256i d3 = _mm256_cvtepi16_epi32(_mm_loadu_si128((__m128i *)(dis_row3 + j)));

            lo_acc = _mm256_mullo_epi32(vf_lo0, d0);
            lo_acc = _mm256_add_epi32(lo_acc, _mm256_mullo_epi32(vf_lo1, d1));
            lo_acc = _mm256_add_epi32(lo_acc, _mm256_mullo_epi32(vf_lo2, d2));
            lo_acc = _mm256_add_epi32(lo_acc, _mm256_mullo_epi32(vf_lo3, d3));
            _mm256_storeu_si256((__m256i *)(tmplo_dis + j), lo_acc);

            hi_acc = _mm256_mullo_epi32(vf_hi0, d0);
            hi_acc = _mm256_add_epi32(hi_acc, _mm256_mullo_epi32(vf_hi1, d1));
            hi_acc = _mm256_add_epi32(hi_acc, _mm256_mullo_epi32(vf_hi2, d2));
            hi_acc = _mm256_add_epi32(hi_acc, _mm256_mullo_epi32(vf_hi3, d3));
            _mm256_storeu_si256((__m256i *)(tmphi_dis + j), hi_acc);
        }
        /* Scalar tail for vertical pass */
        for (; j < w; ++j)
        {
            int32_t s0, s1, s2, s3;
            int64_t acc;

            s0 = (int32_t)ref_row0[j];
            s1 = (int32_t)ref_row1[j];
            s2 = (int32_t)ref_row2[j];
            s3 = (int32_t)ref_row3[j];
            acc = (int64_t)filter_lo[0]*s0 + (int64_t)filter_lo[1]*s1 +
                  (int64_t)filter_lo[2]*s2 + (int64_t)filter_lo[3]*s3;
            tmplo_ref[j] = (int32_t)acc;
            acc = (int64_t)filter_hi[0]*s0 + (int64_t)filter_hi[1]*s1 +
                  (int64_t)filter_hi[2]*s2 + (int64_t)filter_hi[3]*s3;
            tmphi_ref[j] = (int32_t)acc;

            s0 = (int32_t)dis_row0[j];
            s1 = (int32_t)dis_row1[j];
            s2 = (int32_t)dis_row2[j];
            s3 = (int32_t)dis_row3[j];
            acc = (int64_t)filter_lo[0]*s0 + (int64_t)filter_lo[1]*s1 +
                  (int64_t)filter_lo[2]*s2 + (int64_t)filter_lo[3]*s3;
            tmplo_dis[j] = (int32_t)acc;
            acc = (int64_t)filter_hi[0]*s0 + (int64_t)filter_hi[1]*s1 +
                  (int64_t)filter_hi[2]*s2 + (int64_t)filter_hi[3]*s3;
            tmphi_dis[j] = (int32_t)acc;
        }

        /* --- Horizontal pass: int32 -> int32 with shift --- */

        /* Boundary pixel j=0 (scalar) */
        int out_off = i * dst_stride;
        dwt2_horz_scalar_one(tmplo_ref, tmphi_ref,
                             &ind_x[0][0], &ind_x[1][0], &ind_x[2][0], &ind_x[3][0],
                             filter_lo, filter_hi,
                             i4_ref_dwt2->band_a, i4_ref_dwt2->band_v,
                             i4_ref_dwt2->band_h, i4_ref_dwt2->band_d,
                             out_off, add_bef_shift_round_HP, shift_HP);
        dwt2_horz_scalar_one(tmplo_dis, tmphi_dis,
                             &ind_x[0][0], &ind_x[1][0], &ind_x[2][0], &ind_x[3][0],
                             filter_lo, filter_hi,
                             i4_dis_dwt2->band_a, i4_dis_dwt2->band_v,
                             i4_dis_dwt2->band_h, i4_dis_dwt2->band_d,
                             out_off, add_bef_shift_round_HP, shift_HP);

        /* Interior pixels: SIMD 4 at a time */
        int j_simd_end = w_half - 2;  /* exclusive; last 2 are boundary */
        /* Ensure we don't go past the interior region */
        int j_simd_stop = j_simd_end - ((j_simd_end - 1) % 4);
        /* Process 4 outputs per iteration starting from j=1 */
        for (j = 1; j + 3 < j_simd_end; j += 4)
        {
            int base = 2 * j - 1;  /* ind_x[0][j] = 2j-1 for interior */
            /* Ref lo -> band_a */
            dwt2_horz_avx2_4(tmplo_ref, base,
                              hc_lo0, hc_lo1, hc_lo2, hc_lo3,
                              h_rounding, shift_HP,
                              i4_ref_dwt2->band_a, out_off + j);
            /* Ref lo -> band_v */
            dwt2_horz_avx2_4(tmplo_ref, base,
                              hc_hi0, hc_hi1, hc_hi2, hc_hi3,
                              h_rounding, shift_HP,
                              i4_ref_dwt2->band_v, out_off + j);
            /* Ref hi -> band_h */
            dwt2_horz_avx2_4(tmphi_ref, base,
                              hc_lo0, hc_lo1, hc_lo2, hc_lo3,
                              h_rounding, shift_HP,
                              i4_ref_dwt2->band_h, out_off + j);
            /* Ref hi -> band_d */
            dwt2_horz_avx2_4(tmphi_ref, base,
                              hc_hi0, hc_hi1, hc_hi2, hc_hi3,
                              h_rounding, shift_HP,
                              i4_ref_dwt2->band_d, out_off + j);

            /* Dis lo -> band_a */
            dwt2_horz_avx2_4(tmplo_dis, base,
                              hc_lo0, hc_lo1, hc_lo2, hc_lo3,
                              h_rounding, shift_HP,
                              i4_dis_dwt2->band_a, out_off + j);
            /* Dis lo -> band_v */
            dwt2_horz_avx2_4(tmplo_dis, base,
                              hc_hi0, hc_hi1, hc_hi2, hc_hi3,
                              h_rounding, shift_HP,
                              i4_dis_dwt2->band_v, out_off + j);
            /* Dis hi -> band_h */
            dwt2_horz_avx2_4(tmphi_dis, base,
                              hc_lo0, hc_lo1, hc_lo2, hc_lo3,
                              h_rounding, shift_HP,
                              i4_dis_dwt2->band_h, out_off + j);
            /* Dis hi -> band_d */
            dwt2_horz_avx2_4(tmphi_dis, base,
                              hc_hi0, hc_hi1, hc_hi2, hc_hi3,
                              h_rounding, shift_HP,
                              i4_dis_dwt2->band_d, out_off + j);
        }

        /* Remaining interior + boundary pixels (scalar) */
        for (; j < w_half; ++j)
        {
            dwt2_horz_scalar_one(tmplo_ref, tmphi_ref,
                                 &ind_x[0][j], &ind_x[1][j], &ind_x[2][j], &ind_x[3][j],
                                 filter_lo, filter_hi,
                                 i4_ref_dwt2->band_a, i4_ref_dwt2->band_v,
                                 i4_ref_dwt2->band_h, i4_ref_dwt2->band_d,
                                 out_off + j, add_bef_shift_round_HP, shift_HP);
            dwt2_horz_scalar_one(tmplo_dis, tmphi_dis,
                                 &ind_x[0][j], &ind_x[1][j], &ind_x[2][j], &ind_x[3][j],
                                 filter_lo, filter_hi,
                                 i4_dis_dwt2->band_a, i4_dis_dwt2->band_v,
                                 i4_dis_dwt2->band_h, i4_dis_dwt2->band_d,
                                 out_off + j, add_bef_shift_round_HP, shift_HP);
        }
    }
}

void adm_dwt2_s123_combined_avx2(const int32_t *i4_ref_scale,
                                  const int32_t *i4_curr_dis,
                                  AdmBuffer *buf, int w, int h,
                                  int ref_stride, int dis_stride,
                                  int dst_stride, int scale)
{
    const i4_adm_dwt_band_t *i4_ref_dwt2 = &buf->i4_ref_dwt2;
    const i4_adm_dwt_band_t *i4_dis_dwt2 = &buf->i4_dis_dwt2;
    int **ind_y = buf->ind_y;
    int **ind_x = buf->ind_x;

    const int16_t *filter_lo = dwt2_db2_coeffs_lo;
    const int16_t *filter_hi = dwt2_db2_coeffs_hi;

    const int32_t add_bef_shift_round_VP[3] = { 0, 32768, 32768 };
    const int32_t add_bef_shift_round_HP[3] = { 16384, 32768, 16384 };
    const int16_t shift_VerticalPass[3] = { 0, 16, 16 };
    const int16_t shift_HorizontalPass[3] = { 15, 16, 15 };

    const int32_t add_VP = add_bef_shift_round_VP[scale - 1];
    const int shift_VP = shift_VerticalPass[scale - 1];
    const int32_t add_HP = add_bef_shift_round_HP[scale - 1];
    const int shift_HP = shift_HorizontalPass[scale - 1];

    int32_t *tmplo_ref = buf->tmp_ref;
    int32_t *tmphi_ref = tmplo_ref + w;
    int32_t *tmplo_dis = tmphi_ref + w;
    int32_t *tmphi_dis = tmplo_dis + w;

    /* Broadcast filter coefficients as int32 for the vertical pass */
    __m256i vf_lo0_256 = _mm256_set1_epi32((int32_t)filter_lo[0]);
    __m256i vf_lo1_256 = _mm256_set1_epi32((int32_t)filter_lo[1]);
    __m256i vf_lo2_256 = _mm256_set1_epi32((int32_t)filter_lo[2]);
    __m256i vf_lo3_256 = _mm256_set1_epi32((int32_t)filter_lo[3]);
    __m256i vf_hi0_256 = _mm256_set1_epi32((int32_t)filter_hi[0]);
    __m256i vf_hi1_256 = _mm256_set1_epi32((int32_t)filter_hi[1]);
    __m256i vf_hi2_256 = _mm256_set1_epi32((int32_t)filter_hi[2]);
    __m256i vf_hi3_256 = _mm256_set1_epi32((int32_t)filter_hi[3]);

    __m256i v_rounding_vp = _mm256_set1_epi64x(add_VP);

    /* Coefficients for horizontal pass as 128-bit broadcasts (for 4-element processing) */
    __m128i hc_lo0 = _mm_set1_epi32((int32_t)filter_lo[0]);
    __m128i hc_lo1 = _mm_set1_epi32((int32_t)filter_lo[1]);
    __m128i hc_lo2 = _mm_set1_epi32((int32_t)filter_lo[2]);
    __m128i hc_lo3 = _mm_set1_epi32((int32_t)filter_lo[3]);
    __m128i hc_hi0 = _mm_set1_epi32((int32_t)filter_hi[0]);
    __m128i hc_hi1 = _mm_set1_epi32((int32_t)filter_hi[1]);
    __m128i hc_hi2 = _mm_set1_epi32((int32_t)filter_hi[2]);
    __m128i hc_hi3 = _mm_set1_epi32((int32_t)filter_hi[3]);

    __m256i h_rounding = _mm256_set1_epi64x(add_HP);

    int w_half = (w + 1) / 2;

    for (int i = 0; i < (h + 1) / 2; ++i)
    {
        /* --- Vertical pass: int32 input -> int32 output --- */
        const int32_t *ref_row0 = i4_ref_scale + ind_y[0][i] * ref_stride;
        const int32_t *ref_row1 = i4_ref_scale + ind_y[1][i] * ref_stride;
        const int32_t *ref_row2 = i4_ref_scale + ind_y[2][i] * ref_stride;
        const int32_t *ref_row3 = i4_ref_scale + ind_y[3][i] * ref_stride;

        const int32_t *dis_row0 = i4_curr_dis + ind_y[0][i] * dis_stride;
        const int32_t *dis_row1 = i4_curr_dis + ind_y[1][i] * dis_stride;
        const int32_t *dis_row2 = i4_curr_dis + ind_y[2][i] * dis_stride;
        const int32_t *dis_row3 = i4_curr_dis + ind_y[3][i] * dis_stride;

        int j;
        for (j = 0; j + 7 < w; j += 8)
        {
            /* Load 8 int32 from each row */
            __m256i r0 = _mm256_loadu_si256((__m256i *)(ref_row0 + j));
            __m256i r1 = _mm256_loadu_si256((__m256i *)(ref_row1 + j));
            __m256i r2 = _mm256_loadu_si256((__m256i *)(ref_row2 + j));
            __m256i r3 = _mm256_loadu_si256((__m256i *)(ref_row3 + j));

            /* Lo filter: need int64 accumulation.
             * Process even and odd lanes separately. */
            __m256i acc_even = _mm256_setzero_si256();
            __m256i acc_odd  = _mm256_setzero_si256();
            mul_acc_i32_coeff(r0, vf_lo0_256, &acc_even, &acc_odd);
            mul_acc_i32_coeff(r1, vf_lo1_256, &acc_even, &acc_odd);
            mul_acc_i32_coeff(r2, vf_lo2_256, &acc_even, &acc_odd);
            mul_acc_i32_coeff(r3, vf_lo3_256, &acc_even, &acc_odd);

            __m256i lo_result;
            if (shift_VP == 0) {
                lo_result = merge_noshift_i64_to_i32(acc_even, acc_odd);
            } else {
                lo_result = merge_shift_i64_to_i32(acc_even, acc_odd, v_rounding_vp, shift_VP);
            }
            _mm256_storeu_si256((__m256i *)(tmplo_ref + j), lo_result);

            /* Hi filter */
            acc_even = _mm256_setzero_si256();
            acc_odd  = _mm256_setzero_si256();
            mul_acc_i32_coeff(r0, vf_hi0_256, &acc_even, &acc_odd);
            mul_acc_i32_coeff(r1, vf_hi1_256, &acc_even, &acc_odd);
            mul_acc_i32_coeff(r2, vf_hi2_256, &acc_even, &acc_odd);
            mul_acc_i32_coeff(r3, vf_hi3_256, &acc_even, &acc_odd);

            __m256i hi_result;
            if (shift_VP == 0) {
                hi_result = merge_noshift_i64_to_i32(acc_even, acc_odd);
            } else {
                hi_result = merge_shift_i64_to_i32(acc_even, acc_odd, v_rounding_vp, shift_VP);
            }
            _mm256_storeu_si256((__m256i *)(tmphi_ref + j), hi_result);

            /* Distorted */
            __m256i d0 = _mm256_loadu_si256((__m256i *)(dis_row0 + j));
            __m256i d1 = _mm256_loadu_si256((__m256i *)(dis_row1 + j));
            __m256i d2 = _mm256_loadu_si256((__m256i *)(dis_row2 + j));
            __m256i d3 = _mm256_loadu_si256((__m256i *)(dis_row3 + j));

            acc_even = _mm256_setzero_si256();
            acc_odd  = _mm256_setzero_si256();
            mul_acc_i32_coeff(d0, vf_lo0_256, &acc_even, &acc_odd);
            mul_acc_i32_coeff(d1, vf_lo1_256, &acc_even, &acc_odd);
            mul_acc_i32_coeff(d2, vf_lo2_256, &acc_even, &acc_odd);
            mul_acc_i32_coeff(d3, vf_lo3_256, &acc_even, &acc_odd);

            if (shift_VP == 0) {
                lo_result = merge_noshift_i64_to_i32(acc_even, acc_odd);
            } else {
                lo_result = merge_shift_i64_to_i32(acc_even, acc_odd, v_rounding_vp, shift_VP);
            }
            _mm256_storeu_si256((__m256i *)(tmplo_dis + j), lo_result);

            acc_even = _mm256_setzero_si256();
            acc_odd  = _mm256_setzero_si256();
            mul_acc_i32_coeff(d0, vf_hi0_256, &acc_even, &acc_odd);
            mul_acc_i32_coeff(d1, vf_hi1_256, &acc_even, &acc_odd);
            mul_acc_i32_coeff(d2, vf_hi2_256, &acc_even, &acc_odd);
            mul_acc_i32_coeff(d3, vf_hi3_256, &acc_even, &acc_odd);

            if (shift_VP == 0) {
                hi_result = merge_noshift_i64_to_i32(acc_even, acc_odd);
            } else {
                hi_result = merge_shift_i64_to_i32(acc_even, acc_odd, v_rounding_vp, shift_VP);
            }
            _mm256_storeu_si256((__m256i *)(tmphi_dis + j), hi_result);
        }
        /* Scalar tail for vertical pass */
        for (; j < w; ++j)
        {
            int32_t s0, s1, s2, s3;
            int64_t acc;

            s0 = ref_row0[j]; s1 = ref_row1[j];
            s2 = ref_row2[j]; s3 = ref_row3[j];
            acc = (int64_t)filter_lo[0]*s0 + (int64_t)filter_lo[1]*s1 +
                  (int64_t)filter_lo[2]*s2 + (int64_t)filter_lo[3]*s3;
            tmplo_ref[j] = (int32_t)((acc + add_VP) >> shift_VP);
            acc = (int64_t)filter_hi[0]*s0 + (int64_t)filter_hi[1]*s1 +
                  (int64_t)filter_hi[2]*s2 + (int64_t)filter_hi[3]*s3;
            tmphi_ref[j] = (int32_t)((acc + add_VP) >> shift_VP);

            s0 = dis_row0[j]; s1 = dis_row1[j];
            s2 = dis_row2[j]; s3 = dis_row3[j];
            acc = (int64_t)filter_lo[0]*s0 + (int64_t)filter_lo[1]*s1 +
                  (int64_t)filter_lo[2]*s2 + (int64_t)filter_lo[3]*s3;
            tmplo_dis[j] = (int32_t)((acc + add_VP) >> shift_VP);
            acc = (int64_t)filter_hi[0]*s0 + (int64_t)filter_hi[1]*s1 +
                  (int64_t)filter_hi[2]*s2 + (int64_t)filter_hi[3]*s3;
            tmphi_dis[j] = (int32_t)((acc + add_VP) >> shift_VP);
        }

        /* --- Horizontal pass: int32 -> int32 with shift --- */

        /* Boundary pixel j=0 (scalar) */
        int out_off = i * dst_stride;
        dwt2_horz_scalar_one(tmplo_ref, tmphi_ref,
                             &ind_x[0][0], &ind_x[1][0], &ind_x[2][0], &ind_x[3][0],
                             filter_lo, filter_hi,
                             i4_ref_dwt2->band_a, i4_ref_dwt2->band_v,
                             i4_ref_dwt2->band_h, i4_ref_dwt2->band_d,
                             out_off, add_HP, shift_HP);
        dwt2_horz_scalar_one(tmplo_dis, tmphi_dis,
                             &ind_x[0][0], &ind_x[1][0], &ind_x[2][0], &ind_x[3][0],
                             filter_lo, filter_hi,
                             i4_dis_dwt2->band_a, i4_dis_dwt2->band_v,
                             i4_dis_dwt2->band_h, i4_dis_dwt2->band_d,
                             out_off, add_HP, shift_HP);

        /* Interior pixels: SIMD 4 at a time */
        int j_simd_end = w_half - 2;

        for (j = 1; j + 3 < j_simd_end; j += 4)
        {
            int base = 2 * j - 1;
            /* Ref */
            dwt2_horz_avx2_4(tmplo_ref, base,
                              hc_lo0, hc_lo1, hc_lo2, hc_lo3,
                              h_rounding, shift_HP,
                              i4_ref_dwt2->band_a, out_off + j);
            dwt2_horz_avx2_4(tmplo_ref, base,
                              hc_hi0, hc_hi1, hc_hi2, hc_hi3,
                              h_rounding, shift_HP,
                              i4_ref_dwt2->band_v, out_off + j);
            dwt2_horz_avx2_4(tmphi_ref, base,
                              hc_lo0, hc_lo1, hc_lo2, hc_lo3,
                              h_rounding, shift_HP,
                              i4_ref_dwt2->band_h, out_off + j);
            dwt2_horz_avx2_4(tmphi_ref, base,
                              hc_hi0, hc_hi1, hc_hi2, hc_hi3,
                              h_rounding, shift_HP,
                              i4_ref_dwt2->band_d, out_off + j);

            /* Dis */
            dwt2_horz_avx2_4(tmplo_dis, base,
                              hc_lo0, hc_lo1, hc_lo2, hc_lo3,
                              h_rounding, shift_HP,
                              i4_dis_dwt2->band_a, out_off + j);
            dwt2_horz_avx2_4(tmplo_dis, base,
                              hc_hi0, hc_hi1, hc_hi2, hc_hi3,
                              h_rounding, shift_HP,
                              i4_dis_dwt2->band_v, out_off + j);
            dwt2_horz_avx2_4(tmphi_dis, base,
                              hc_lo0, hc_lo1, hc_lo2, hc_lo3,
                              h_rounding, shift_HP,
                              i4_dis_dwt2->band_h, out_off + j);
            dwt2_horz_avx2_4(tmphi_dis, base,
                              hc_hi0, hc_hi1, hc_hi2, hc_hi3,
                              h_rounding, shift_HP,
                              i4_dis_dwt2->band_d, out_off + j);
        }

        /* Remaining interior + boundary pixels (scalar) */
        for (; j < w_half; ++j)
        {
            dwt2_horz_scalar_one(tmplo_ref, tmphi_ref,
                                 &ind_x[0][j], &ind_x[1][j], &ind_x[2][j], &ind_x[3][j],
                                 filter_lo, filter_hi,
                                 i4_ref_dwt2->band_a, i4_ref_dwt2->band_v,
                                 i4_ref_dwt2->band_h, i4_ref_dwt2->band_d,
                                 out_off + j, add_HP, shift_HP);
            dwt2_horz_scalar_one(tmplo_dis, tmphi_dis,
                                 &ind_x[0][j], &ind_x[1][j], &ind_x[2][j], &ind_x[3][j],
                                 filter_lo, filter_hi,
                                 i4_dis_dwt2->band_a, i4_dis_dwt2->band_v,
                                 i4_dis_dwt2->band_h, i4_dis_dwt2->band_d,
                                 out_off + j, add_HP, shift_HP);
        }
    }
}
