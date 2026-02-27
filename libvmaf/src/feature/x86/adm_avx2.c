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
        /* Vertical pass. */

        for (int j = 0; j < w; j = j + 16) {

            __m256i accum_mu2_lo, accum_mu2_hi, accum_mu1_lo, accum_mu1_hi;
            accum_mu2_lo = accum_mu2_hi = accum_mu1_lo = accum_mu1_hi =
                _mm256_setzero_si256();
            __m256i s0, s1, s2, s3;

            s0 = _mm256_cvtepu8_epi16(_mm_loadu_si128(
                (__m128i *)(src + (ind_y[0][i] * src_stride) + j)));
            s1 = _mm256_cvtepu8_epi16(_mm_loadu_si128(
                (__m128i *)(src + (ind_y[1][i] * src_stride) + j)));
            s2 = _mm256_cvtepu8_epi16(_mm_loadu_si128(
                (__m128i *)(src + (ind_y[2][i] * src_stride) + j)));
            s3 = _mm256_cvtepu8_epi16(_mm_loadu_si128(
                (__m128i *)(src + (ind_y[3][i] * src_stride) + j)));

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
            // for( int k =0; k<16;k++){
            //     fprintf(stderr, "actual value hi tmp is %d \n",tmphi[j +k]);
            // }
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

        for (int j = 1; j < (w + 1) / 2; j = j + 16) {
            {
                __m256i accum_mu2_lo, accum_mu2_hi, accum_mu1_lo, accum_mu1_hi;
                accum_mu2_lo = accum_mu2_hi = accum_mu1_lo = accum_mu1_hi =
                    _mm256_setzero_si256();

                __m256i s00, s22, s33, s44;

                s00 = _mm256_loadu_si256((__m256i *)(tmplo + ind_x[0][j]));
                s22 = _mm256_loadu_si256((__m256i *)(tmplo + ind_x[2][j]));
                s33 = _mm256_loadu_si256((__m256i *)(tmplo + 16 + ind_x[0][j]));
                s44 = _mm256_loadu_si256((__m256i *)(tmplo + 16 + ind_x[2][j]));

                accum_mu2_lo =
                    _mm256_add_epi32(accum_mu2_lo, _mm256_madd_epi16(s00, fl0));
                accum_mu2_hi =
                    _mm256_add_epi32(accum_mu2_hi, _mm256_madd_epi16(s33, fl0));
                accum_mu2_lo =
                    _mm256_add_epi32(accum_mu2_lo, _mm256_madd_epi16(s22, fl1));
                accum_mu2_hi =
                    _mm256_add_epi32(accum_mu2_hi, _mm256_madd_epi16(s44, fl1));

                accum_mu2_lo = _mm256_add_epi32(accum_mu2_lo, add_shift_HP_vex);
                accum_mu2_lo = _mm256_srli_epi32(accum_mu2_lo, 0x10);
                accum_mu2_hi = _mm256_add_epi32(accum_mu2_hi, add_shift_HP_vex);
                accum_mu2_hi = _mm256_srli_epi32(accum_mu2_hi, 0x10);

                accum_mu2_hi = _mm256_packus_epi32(accum_mu2_lo, accum_mu2_hi);
                accum_mu2_hi = _mm256_permute4x64_epi64(accum_mu2_hi, 0xD8);
                _mm256_storeu_si256(
                    (__m256i *)(dst->band_a + i * dst_stride + j),
                    accum_mu2_hi);

                accum_mu1_lo =
                    _mm256_add_epi32(accum_mu1_lo, _mm256_madd_epi16(s00, fh0));
                accum_mu1_hi =
                    _mm256_add_epi32(accum_mu1_hi, _mm256_madd_epi16(s33, fh0));
                accum_mu1_lo =
                    _mm256_add_epi32(accum_mu1_lo, _mm256_madd_epi16(s22, fh1));
                accum_mu1_hi =
                    _mm256_add_epi32(accum_mu1_hi, _mm256_madd_epi16(s44, fh1));

                accum_mu1_lo = _mm256_add_epi32(accum_mu1_lo, add_shift_HP_vex);
                accum_mu1_lo = _mm256_srli_epi32(accum_mu1_lo, 0x10);
                accum_mu1_hi = _mm256_add_epi32(accum_mu1_hi, add_shift_HP_vex);
                accum_mu1_hi = _mm256_srli_epi32(accum_mu1_hi, 0x10);

                accum_mu1_hi = _mm256_packus_epi32(accum_mu1_lo, accum_mu1_hi);
                accum_mu1_hi = _mm256_permute4x64_epi64(accum_mu1_hi, 0xD8);
                _mm256_storeu_si256(
                    (__m256i *)(dst->band_v + i * dst_stride + j),
                    accum_mu1_hi);
            }

            {
                __m256i accum_mu2_lo, accum_mu2_hi, accum_mu1_lo, accum_mu1_hi;
                accum_mu2_lo = accum_mu2_hi = accum_mu1_lo = accum_mu1_hi =
                    _mm256_setzero_si256();

                __m256i s00, s22, s33, s44;

                __m256i add_shift_HP_vex = _mm256_set1_epi32(32768);

                s00 = _mm256_loadu_si256((__m256i *)(tmphi + ind_x[0][j]));
                s22 = _mm256_loadu_si256((__m256i *)(tmphi + ind_x[2][j]));
                s33 = _mm256_loadu_si256((__m256i *)(tmphi + 16 + ind_x[0][j]));
                s44 = _mm256_loadu_si256((__m256i *)(tmphi + 16 + ind_x[2][j]));

                accum_mu2_lo =
                    _mm256_add_epi32(accum_mu2_lo, _mm256_madd_epi16(s00, fl0));
                accum_mu2_hi =
                    _mm256_add_epi32(accum_mu2_hi, _mm256_madd_epi16(s33, fl0));
                accum_mu2_lo =
                    _mm256_add_epi32(accum_mu2_lo, _mm256_madd_epi16(s22, fl1));
                accum_mu2_hi =
                    _mm256_add_epi32(accum_mu2_hi, _mm256_madd_epi16(s44, fl1));

                accum_mu2_lo = _mm256_add_epi32(accum_mu2_lo, add_shift_HP_vex);
                accum_mu2_lo = _mm256_srli_epi32(accum_mu2_lo, 0x10);
                accum_mu2_hi = _mm256_add_epi32(accum_mu2_hi, add_shift_HP_vex);
                accum_mu2_hi = _mm256_srli_epi32(accum_mu2_hi, 0x10);

                accum_mu2_hi = _mm256_packus_epi32(accum_mu2_lo, accum_mu2_hi);
                accum_mu2_hi = _mm256_permute4x64_epi64(accum_mu2_hi, 0xD8);
                _mm256_storeu_si256(
                    (__m256i *)(dst->band_h + i * dst_stride + j),
                    accum_mu2_hi);

                accum_mu1_lo =
                    _mm256_add_epi32(accum_mu1_lo, _mm256_madd_epi16(s00, fh0));
                accum_mu1_hi =
                    _mm256_add_epi32(accum_mu1_hi, _mm256_madd_epi16(s33, fh0));
                accum_mu1_lo =
                    _mm256_add_epi32(accum_mu1_lo, _mm256_madd_epi16(s22, fh1));
                accum_mu1_hi =
                    _mm256_add_epi32(accum_mu1_hi, _mm256_madd_epi16(s44, fh1));

                accum_mu1_lo = _mm256_add_epi32(accum_mu1_lo, add_shift_HP_vex);
                accum_mu1_lo = _mm256_srli_epi32(accum_mu1_lo, 0x10);
                accum_mu1_hi = _mm256_add_epi32(accum_mu1_hi, add_shift_HP_vex);
                accum_mu1_hi = _mm256_srli_epi32(accum_mu1_hi, 0x10);

                accum_mu1_hi = _mm256_packus_epi32(accum_mu1_lo, accum_mu1_hi);
                accum_mu1_hi = _mm256_permute4x64_epi64(accum_mu1_hi, 0xD8);
                _mm256_storeu_si256(
                    (__m256i *)(dst->band_d + i * dst_stride + j),
                    accum_mu1_hi);
            }
        }
    }
}

void adm_csf_den_s0_avx2(const adm_dwt_band_t *src, int w, int h,
                         int src_stride, int left, int top,
                         int right, int bottom,
                         int32_t shift_accum, int32_t add_shift_accum,
                         uint64_t *out_h, uint64_t *out_v, uint64_t *out_d)
{
    uint64_t accum_h = 0, accum_v = 0, accum_d = 0;

    int16_t *src_h = src->band_h + top * src_stride;
    int16_t *src_v = src->band_v + top * src_stride;
    int16_t *src_d = src->band_d + top * src_stride;

    for (int i = top; i < bottom; ++i) {
        if (i + 1 < bottom) {
            __builtin_prefetch(src_h + src_stride, 0, 1);
            __builtin_prefetch(src_v + src_stride, 0, 1);
            __builtin_prefetch(src_d + src_stride, 0, 1);
        }

        __m256i accum_h_vec = _mm256_setzero_si256();
        __m256i accum_v_vec = _mm256_setzero_si256();
        __m256i accum_d_vec = _mm256_setzero_si256();
        int j = left;

        for (; j + 16 <= right; j += 16) {
            /* Load 16 int16, take abs */
            __m256i sh = _mm256_abs_epi16(_mm256_loadu_si256((const __m256i *)(src_h + j)));
            __m256i sv = _mm256_abs_epi16(_mm256_loadu_si256((const __m256i *)(src_v + j)));
            __m256i sd = _mm256_abs_epi16(_mm256_loadu_si256((const __m256i *)(src_d + j)));

            /* Process H band: first 8 elements */
            {
                __m256i vals = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(sh));
                __m256i sq = _mm256_mullo_epi32(vals, vals);
                /* Even elements cube */
                accum_h_vec = _mm256_add_epi64(accum_h_vec, _mm256_mul_epu32(sq, vals));
                /* Odd elements cube */
                accum_h_vec = _mm256_add_epi64(accum_h_vec,
                    _mm256_mul_epu32(_mm256_srli_epi64(sq, 32), _mm256_srli_epi64(vals, 32)));
            }
            /* H band: next 8 elements */
            {
                __m256i vals = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(sh, 1));
                __m256i sq = _mm256_mullo_epi32(vals, vals);
                accum_h_vec = _mm256_add_epi64(accum_h_vec, _mm256_mul_epu32(sq, vals));
                accum_h_vec = _mm256_add_epi64(accum_h_vec,
                    _mm256_mul_epu32(_mm256_srli_epi64(sq, 32), _mm256_srli_epi64(vals, 32)));
            }

            /* V band */
            {
                __m256i vals = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(sv));
                __m256i sq = _mm256_mullo_epi32(vals, vals);
                accum_v_vec = _mm256_add_epi64(accum_v_vec, _mm256_mul_epu32(sq, vals));
                accum_v_vec = _mm256_add_epi64(accum_v_vec,
                    _mm256_mul_epu32(_mm256_srli_epi64(sq, 32), _mm256_srli_epi64(vals, 32)));
            }
            {
                __m256i vals = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(sv, 1));
                __m256i sq = _mm256_mullo_epi32(vals, vals);
                accum_v_vec = _mm256_add_epi64(accum_v_vec, _mm256_mul_epu32(sq, vals));
                accum_v_vec = _mm256_add_epi64(accum_v_vec,
                    _mm256_mul_epu32(_mm256_srli_epi64(sq, 32), _mm256_srli_epi64(vals, 32)));
            }

            /* D band */
            {
                __m256i vals = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(sd));
                __m256i sq = _mm256_mullo_epi32(vals, vals);
                accum_d_vec = _mm256_add_epi64(accum_d_vec, _mm256_mul_epu32(sq, vals));
                accum_d_vec = _mm256_add_epi64(accum_d_vec,
                    _mm256_mul_epu32(_mm256_srli_epi64(sq, 32), _mm256_srli_epi64(vals, 32)));
            }
            {
                __m256i vals = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(sd, 1));
                __m256i sq = _mm256_mullo_epi32(vals, vals);
                accum_d_vec = _mm256_add_epi64(accum_d_vec, _mm256_mul_epu32(sq, vals));
                accum_d_vec = _mm256_add_epi64(accum_d_vec,
                    _mm256_mul_epu32(_mm256_srli_epi64(sq, 32), _mm256_srli_epi64(vals, 32)));
            }
        }

        /* Horizontal reduce 4 × uint64 → 1 */
        uint64_t inner_h = 0, inner_v = 0, inner_d = 0;
        {
            __m128i lo, hi;
            lo = _mm256_castsi256_si128(accum_h_vec);
            hi = _mm256_extracti128_si256(accum_h_vec, 1);
            __m128i sum = _mm_add_epi64(lo, hi);
            inner_h = (uint64_t)_mm_extract_epi64(sum, 0) + (uint64_t)_mm_extract_epi64(sum, 1);

            lo = _mm256_castsi256_si128(accum_v_vec);
            hi = _mm256_extracti128_si256(accum_v_vec, 1);
            sum = _mm_add_epi64(lo, hi);
            inner_v = (uint64_t)_mm_extract_epi64(sum, 0) + (uint64_t)_mm_extract_epi64(sum, 1);

            lo = _mm256_castsi256_si128(accum_d_vec);
            hi = _mm256_extracti128_si256(accum_d_vec, 1);
            sum = _mm_add_epi64(lo, hi);
            inner_d = (uint64_t)_mm_extract_epi64(sum, 0) + (uint64_t)_mm_extract_epi64(sum, 1);
        }

        /* Scalar remainder */
        for (; j < right; ++j) {
            uint16_t hv = (uint16_t)abs(src_h[j]);
            uint16_t vv = (uint16_t)abs(src_v[j]);
            uint16_t dv = (uint16_t)abs(src_d[j]);
            inner_h += ((uint64_t)hv * hv) * hv;
            inner_v += ((uint64_t)vv * vv) * vv;
            inner_d += ((uint64_t)dv * dv) * dv;
        }

        accum_h += (inner_h + add_shift_accum) >> shift_accum;
        accum_v += (inner_v + add_shift_accum) >> shift_accum;
        accum_d += (inner_d + add_shift_accum) >> shift_accum;
        src_h += src_stride;
        src_v += src_stride;
        src_d += src_stride;
    }

    *out_h = accum_h;
    *out_v = accum_v;
    *out_d = accum_d;
}

void adm_csf_s0_avx2(AdmBuffer *buf, int w, int h, int stride,
                      const uint16_t i_rfactor[3],
                      const uint8_t i_shifts[3],
                      const uint16_t i_shiftsadd[3])
{
    const adm_dwt_band_t *src = &buf->decouple_a;
    const adm_dwt_band_t *dst = &buf->csf_a;
    const adm_dwt_band_t *flt = &buf->csf_f;

    const int16_t *src_angles[3] = { src->band_h, src->band_v, src->band_d };
    int16_t *dst_angles[3] = { dst->band_h, dst->band_v, dst->band_d };
    int16_t *flt_angles[3] = { flt->band_h, flt->band_v, flt->band_d };

    const uint16_t FIX_ONE_BY_30 = 4369;

    int left = w * ADM_BORDER_FACTOR - 0.5 - 1;
    int top = h * ADM_BORDER_FACTOR - 0.5 - 1;
    int right = w - left + 2;
    int bottom = h - top + 2;

    if (left < 0) left = 0;
    if (right > w) right = w;
    if (top < 0) top = 0;
    if (bottom > h) bottom = h;

    for (int theta = 0; theta < 3; ++theta) {
        const int16_t *src_ptr = src_angles[theta];
        int16_t *dst_ptr = dst_angles[theta];
        int16_t *flt_ptr = flt_angles[theta];

        const __m256i rfact = _mm256_set1_epi32((int32_t)i_rfactor[theta]);
        const __m128i shift_cnt = _mm_cvtsi32_si128(i_shifts[theta]);
        const __m256i shift_add = _mm256_set1_epi32(i_shiftsadd[theta]);
        const __m256i fix30 = _mm256_set1_epi32(FIX_ONE_BY_30);
        const __m256i round_2048 = _mm256_set1_epi32(2048);

        for (int i = top; i < bottom; ++i) {
            int offset = i * stride;
            int j = left;

            /* AVX2: process 16 int16 values per iteration */
            for (; j + 16 <= right; j += 16) {
                __m256i s = _mm256_loadu_si256((const __m256i *)(src_ptr + offset + j));

                /* Sign-extend int16 to int32 (two halves) */
                __m256i s_lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(s));
                __m256i s_hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(s, 1));

                /* Multiply by i_rfactor (int32 * int32 → int32, no overflow) */
                __m256i d_lo = _mm256_mullo_epi32(s_lo, rfact);
                __m256i d_hi = _mm256_mullo_epi32(s_hi, rfact);

                /* Add rounding and arithmetic shift right */
                d_lo = _mm256_sra_epi32(_mm256_add_epi32(d_lo, shift_add), shift_cnt);
                d_hi = _mm256_sra_epi32(_mm256_add_epi32(d_hi, shift_add), shift_cnt);

                /* Pack int32 back to int16 (saturating, within 128-bit lanes) */
                __m256i d_packed = _mm256_packs_epi32(d_lo, d_hi);
                /* Fix lane order: packs interleaves within 128-bit lanes */
                d_packed = _mm256_permute4x64_epi64(d_packed, 0xD8);
                _mm256_storeu_si256((__m256i *)(dst_ptr + offset + j), d_packed);

                /* Compute flt = (FIX_ONE_BY_30 * abs(dst_val) + 2048) >> 12 */
                __m256i abs_lo = _mm256_abs_epi32(d_lo);
                __m256i abs_hi = _mm256_abs_epi32(d_hi);
                __m256i f_lo = _mm256_srli_epi32(
                    _mm256_add_epi32(_mm256_mullo_epi32(abs_lo, fix30), round_2048), 12);
                __m256i f_hi = _mm256_srli_epi32(
                    _mm256_add_epi32(_mm256_mullo_epi32(abs_hi, fix30), round_2048), 12);
                __m256i f_packed = _mm256_packs_epi32(f_lo, f_hi);
                f_packed = _mm256_permute4x64_epi64(f_packed, 0xD8);
                _mm256_storeu_si256((__m256i *)(flt_ptr + offset + j), f_packed);
            }

            /* Scalar remainder */
            for (; j < right; ++j) {
                int32_t dst_val = i_rfactor[theta] * (int32_t)src_ptr[offset + j];
                int16_t i16_dst_val = (int16_t)((dst_val + i_shiftsadd[theta]) >> i_shifts[theta]);
                dst_ptr[offset + j] = i16_dst_val;
                flt_ptr[offset + j] = (int16_t)(((FIX_ONE_BY_30 * abs((int32_t)i16_dst_val))
                    + 2048) >> 12);
            }
        }
    }
}
