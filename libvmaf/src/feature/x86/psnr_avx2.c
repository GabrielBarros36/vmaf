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
#include <stddef.h>

void psnr_sse_8_avx2(const uint8_t *ref, const uint8_t *dis,
                      uint64_t *sse, unsigned w, unsigned h,
                      ptrdiff_t ref_stride, ptrdiff_t dis_stride)
{
    uint64_t sse_acc = 0;

    for (unsigned i = 0; i < h; i++) {
        __m256i row_acc = _mm256_setzero_si256();
        unsigned j = 0;

        /* Process 32 uint8 per iteration (split into two groups of 16).
         * Each group of 16 uint8 is widened to int16, subtracted, then
         * _mm256_madd_epi16(e, e) produces 8 int32 squared-error sums
         * (adjacent pairs are added). We accumulate these int32 sums
         * in row_acc. Per-row accumulation is safe because the maximum
         * row contribution is w * 255^2. For w <= 65535 this fits in
         * uint32 (65535 * 65025 ~ 4.26G > 2^32), but we only need
         * w <= 33025 for uint32 safety. Practically, video widths are
         * well under 33025, so uint32 per-row accumulation is fine.
         * We drain to uint64 at the end of each row regardless. */
        for (; j + 31 < w; j += 32) {
            __m256i r = _mm256_loadu_si256((const __m256i *)(ref + j));
            __m256i d = _mm256_loadu_si256((const __m256i *)(dis + j));

            /* Low 16 bytes -> 16 int16 */
            __m256i r_lo = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(r));
            __m256i d_lo = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(d));
            __m256i e_lo = _mm256_sub_epi16(r_lo, d_lo);
            __m256i sq_lo = _mm256_madd_epi16(e_lo, e_lo);

            /* High 16 bytes -> 16 int16 */
            __m256i r_hi = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(r, 1));
            __m256i d_hi = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(d, 1));
            __m256i e_hi = _mm256_sub_epi16(r_hi, d_hi);
            __m256i sq_hi = _mm256_madd_epi16(e_hi, e_hi);

            /* Accumulate 8+8 = 16 int32 values into row_acc */
            row_acc = _mm256_add_epi32(row_acc, sq_lo);
            row_acc = _mm256_add_epi32(row_acc, sq_hi);
        }

        /* Handle 16-byte chunks */
        for (; j + 15 < w; j += 16) {
            __m128i r128 = _mm_loadu_si128((const __m128i *)(ref + j));
            __m128i d128 = _mm_loadu_si128((const __m128i *)(dis + j));

            __m256i r16 = _mm256_cvtepu8_epi16(r128);
            __m256i d16 = _mm256_cvtepu8_epi16(d128);
            __m256i e = _mm256_sub_epi16(r16, d16);
            __m256i sq = _mm256_madd_epi16(e, e);

            row_acc = _mm256_add_epi32(row_acc, sq);
        }

        /* Horizontal sum of row_acc (8 x int32) -> uint32_t */
        __m128i lo128 = _mm256_castsi256_si128(row_acc);
        __m128i hi128 = _mm256_extracti128_si256(row_acc, 1);
        __m128i sum128 = _mm_add_epi32(lo128, hi128);
        sum128 = _mm_add_epi32(sum128, _mm_srli_si128(sum128, 8));
        sum128 = _mm_add_epi32(sum128, _mm_srli_si128(sum128, 4));
        uint32_t row_sse = (uint32_t)_mm_cvtsi128_si32(sum128);

        /* Scalar tail */
        for (; j < w; j++) {
            const int16_t e = (int16_t)ref[j] - (int16_t)dis[j];
            row_sse += (uint32_t)(e * e);
        }

        sse_acc += row_sse;
        ref += ref_stride;
        dis += dis_stride;
    }

    *sse = sse_acc;
}

void psnr_sse_hbd_avx2(const uint16_t *ref, const uint16_t *dis,
                        uint64_t *sse, unsigned w, unsigned h,
                        ptrdiff_t ref_stride, ptrdiff_t dis_stride)
{
    const ptrdiff_t ref_px_stride = ref_stride / sizeof(uint16_t);
    const ptrdiff_t dis_px_stride = dis_stride / sizeof(uint16_t);
    uint64_t sse_acc = 0;

    for (unsigned i = 0; i < h; i++) {
        /* For HBD, pixel values can be up to 65535 (16-bit), so the
         * difference can be up to 65535 which doesn't fit in int16.
         * We widen to int32 for the subtraction and squaring.
         *
         * Process 8 uint16 at a time:
         * - Load 8 uint16 (128 bits) from ref and dis
         * - Widen to 8 int32 (256 bits) using _mm256_cvtepu16_epi32
         * - Subtract in int32
         * - Square with _mm256_mullo_epi32
         * - Accumulate squared diffs into 64-bit running sum
         *
         * For the 64-bit accumulation, we widen the 8 int32 squared
         * values to 4+4 int64 values and add to a 64-bit accumulator.
         */
        __m256i row_acc_lo = _mm256_setzero_si256(); /* 4 x int64 */
        __m256i row_acc_hi = _mm256_setzero_si256(); /* 4 x int64 */
        unsigned j = 0;

        for (; j + 7 < w; j += 8) {
            __m128i r128 = _mm_loadu_si128((const __m128i *)(ref + j));
            __m128i d128 = _mm_loadu_si128((const __m128i *)(dis + j));

            /* Widen uint16 -> int32 */
            __m256i r32 = _mm256_cvtepu16_epi32(r128);
            __m256i d32 = _mm256_cvtepu16_epi32(d128);

            /* Subtract in int32 (result fits: -65535 to +65535) */
            __m256i diff = _mm256_sub_epi32(r32, d32);

            /* Square: result fits in uint32 for 10/12-bit, but can
             * overflow for 16-bit (65535^2 = 4,294,836,225 > 2^31).
             * Use _mm256_mullo_epi32 which gives low 32 bits.
             * For 16-bit content, 65535^2 = 0xFFFE0001 which fits
             * in uint32 (< 2^32), so mullo_epi32 is correct. */
            __m256i sq = _mm256_mullo_epi32(diff, diff);

            /* Widen int32 -> int64 and accumulate.
             * Split 8 x int32 into two groups of 4 x int64. */
            __m128i sq_lo128 = _mm256_castsi256_si128(sq);
            __m128i sq_hi128 = _mm256_extracti128_si256(sq, 1);
            __m256i sq_64_lo = _mm256_cvtepu32_epi64(sq_lo128);
            __m256i sq_64_hi = _mm256_cvtepu32_epi64(sq_hi128);

            row_acc_lo = _mm256_add_epi64(row_acc_lo, sq_64_lo);
            row_acc_hi = _mm256_add_epi64(row_acc_hi, sq_64_hi);
        }

        /* Horizontal sum of 8 x int64 accumulators -> uint64_t */
        __m256i row_sum = _mm256_add_epi64(row_acc_lo, row_acc_hi);
        __m128i rlo = _mm256_castsi256_si128(row_sum);
        __m128i rhi = _mm256_extracti128_si256(row_sum, 1);
        __m128i rsum = _mm_add_epi64(rlo, rhi);
        rsum = _mm_add_epi64(rsum, _mm_srli_si128(rsum, 8));
        uint64_t row_sse = (uint64_t)_mm_cvtsi128_si64(rsum);

        /* Scalar tail */
        for (; j < w; j++) {
            const int32_t e = (int32_t)ref[j] - (int32_t)dis[j];
            row_sse += (uint64_t)((uint32_t)(e * e));
        }

        sse_acc += row_sse;
        ref += ref_px_stride;
        dis += dis_px_stride;
    }

    *sse = sse_acc;
}
