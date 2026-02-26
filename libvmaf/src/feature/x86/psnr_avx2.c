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
#include <stdlib.h>

void psnr_sse_8_avx2(const uint8_t *ref, const uint8_t *dis,
                     uint64_t *sse, unsigned width, unsigned height,
                     ptrdiff_t ref_stride, ptrdiff_t dis_stride)
{
    uint64_t total_sse = 0;

    for (unsigned i = 0; i < height; i++) {
        __m256i row_sse = _mm256_setzero_si256(); /* 8x 32-bit accumulators */
        unsigned j = 0;

        /* Process 16 pixels at a time: zero-extend uint8 to int16,
         * subtract, then madd to get paired squared errors in 32-bit. */
        for (; j + 16 <= width; j += 16) {
            __m256i r = _mm256_cvtepu8_epi16(
                            _mm_loadu_si128((const __m128i *)(ref + j)));
            __m256i d = _mm256_cvtepu8_epi16(
                            _mm_loadu_si128((const __m128i *)(dis + j)));
            __m256i diff = _mm256_sub_epi16(r, d);
            /* madd(diff, diff) = sum of pairs of squared diffs as 32-bit */
            __m256i sq = _mm256_madd_epi16(diff, diff);
            row_sse = _mm256_add_epi32(row_sse, sq);
        }

        /* Horizontal sum of 8x 32-bit lanes into a scalar uint32_t */
        __m128i lo = _mm256_castsi256_si128(row_sse);
        __m128i hi = _mm256_extracti128_si256(row_sse, 1);
        __m128i sum128 = _mm_add_epi32(lo, hi);
        sum128 = _mm_add_epi32(sum128,
                     _mm_shuffle_epi32(sum128, _MM_SHUFFLE(1, 0, 3, 2)));
        sum128 = _mm_add_epi32(sum128,
                     _mm_shuffle_epi32(sum128, _MM_SHUFFLE(0, 1, 0, 1)));
        uint32_t row_total = (uint32_t)_mm_cvtsi128_si32(sum128);

        /* Scalar tail for remaining pixels */
        for (; j < width; j++) {
            int16_t e = (int16_t)ref[j] - (int16_t)dis[j];
            row_total += (uint32_t)(e * e);
        }

        total_sse += row_total;
        ref += ref_stride;
        dis += dis_stride;
    }

    *sse = total_sse;
}

void psnr_sse_16_avx2(const uint16_t *ref, const uint16_t *dis,
                      uint64_t *sse, unsigned width, unsigned height,
                      ptrdiff_t ref_stride, ptrdiff_t dis_stride)
{
    uint64_t total_sse = 0;

    for (unsigned i = 0; i < height; i++) {
        /* 4x 64-bit accumulators — needed because (65535)^2 exceeds 32 bits */
        __m256i row_sse64 = _mm256_setzero_si256();
        unsigned j = 0;

        /* Process 8 uint16 pixels at a time: widen to 32-bit, subtract,
         * then use 64-bit multiply to get squared errors. */
        for (; j + 8 <= width; j += 8) {
            __m256i r = _mm256_cvtepu16_epi32(
                            _mm_loadu_si128((const __m128i *)(ref + j)));
            __m256i d = _mm256_cvtepu16_epi32(
                            _mm_loadu_si128((const __m128i *)(dis + j)));
            __m256i diff = _mm256_sub_epi32(r, d);

            /* _mm256_mul_epi32 multiplies the low 32-bit lane of each
             * 64-bit element, producing 4x 64-bit results.  We need to
             * handle even-index and odd-index elements separately. */
            __m256i diff_odd = _mm256_srli_epi64(diff, 32);

            __m256i sq_even = _mm256_mul_epi32(diff, diff);
            __m256i sq_odd  = _mm256_mul_epi32(diff_odd, diff_odd);

            row_sse64 = _mm256_add_epi64(row_sse64, sq_even);
            row_sse64 = _mm256_add_epi64(row_sse64, sq_odd);
        }

        /* Horizontal sum of 4x 64-bit lanes into a scalar uint64_t */
        __m128i lo = _mm256_castsi256_si128(row_sse64);
        __m128i hi = _mm256_extracti128_si256(row_sse64, 1);
        __m128i sum128 = _mm_add_epi64(lo, hi);
        sum128 = _mm_add_epi64(sum128,
                     _mm_shuffle_epi32(sum128, _MM_SHUFFLE(1, 0, 3, 2)));
        uint64_t row_total = (uint64_t)_mm_cvtsi128_si64(sum128);

        /* Scalar tail for remaining pixels */
        for (; j < width; j++) {
            uint32_t e = abs((int32_t)ref[j] - (int32_t)dis[j]);
            row_total += (uint64_t)e * e;
        }

        total_sse += row_total;
        ref += ref_stride;
        dis += dis_stride;
    }

    *sse = total_sse;
}
