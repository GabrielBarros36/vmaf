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

            __m256d v_4096_d = _mm256_set1_pd(4096.0);
            __m256d v_zero_d = _mm256_setzero_pd();
            __m256d v_cos_d = _mm256_set1_pd((double)cos_1deg_sq);

            /* ot_dp / 4096.0 */
            __m256d dp_lo = _mm256_div_pd(ot_dp_d_lo, v_4096_d);
            __m256d dp_hi = _mm256_div_pd(ot_dp_d_hi, v_4096_d);
            __m256d omag_lo = _mm256_div_pd(o_mag_d_lo, v_4096_d);
            __m256d omag_hi = _mm256_div_pd(o_mag_d_hi, v_4096_d);
            __m256d tmag_lo = _mm256_div_pd(t_mag_d_lo, v_4096_d);
            __m256d tmag_hi = _mm256_div_pd(t_mag_d_hi, v_4096_d);

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
            __m256 v_zero_f = _mm256_setzero_ps();

            /*
             * Division via lookup table:
             * tmp_k = (o == 0) ? 32768 : ((div_lookup[o + 32768] * t) + 16384) >> 15
             * k = clamp(tmp_k, 0, 32768)
             *
             * For the gather: index = o + 32768 (o is int16, so index is [0, 65536])
             * div_lookup is int32_t[65537]
             */
            __m256i v_32768 = _mm256_set1_epi32(32768);
            __m256i v_16384 = _mm256_set1_epi32(16384);
            __m256i v_zero = _mm256_setzero_si256();

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
                __m256 v_gain = _mm256_set1_ps((float)adm_enhn_gain_limit);
                __m256 v_32768_f = _mm256_set1_ps(32768.0f);
                __m256 v_64_f = _mm256_set1_ps(64.0f);

                /* Helper macro-like: apply enhancement gain limit */
                /* band_h */
                {
                    __m256 kh_f = _mm256_div_ps(_mm256_cvtepi32_ps(kh), v_32768_f);
                    __m256 oh_f = _mm256_div_ps(_mm256_cvtepi32_ps(oh), v_64_f);
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
                    __m256 kv_f = _mm256_div_ps(_mm256_cvtepi32_ps(kv), v_32768_f);
                    __m256 ov_f = _mm256_div_ps(_mm256_cvtepi32_ps(ov), v_64_f);
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
                    __m256 kd_f = _mm256_div_ps(_mm256_cvtepi32_ps(kd), v_32768_f);
                    __m256 od_f = _mm256_div_ps(_mm256_cvtepi32_ps(od), v_64_f);
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
