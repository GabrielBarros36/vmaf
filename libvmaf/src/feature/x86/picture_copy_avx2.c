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

void picture_copy_8bit_avx2(float *dst, ptrdiff_t dst_stride,
                            const uint8_t *src, ptrdiff_t src_stride,
                            int offset, unsigned w, unsigned h)
{
    const __m256 offset_vec = _mm256_set1_ps((float)offset);
    const unsigned w8 = w & ~7u;

    for (unsigned i = 0; i < h; i++) {
        unsigned j = 0;
        for (; j < w8; j += 8) {
            /* Load 8 uint8 values, zero-extend to 32-bit, convert to float */
            __m128i u8 = _mm_loadl_epi64((const __m128i *)(src + j));
            __m256i i32 = _mm256_cvtepu8_epi32(u8);
            __m256 f32 = _mm256_cvtepi32_ps(i32);
            f32 = _mm256_add_ps(f32, offset_vec);
            _mm256_storeu_ps(dst + j, f32);
        }
        /* Scalar tail */
        for (; j < w; j++) {
            dst[j] = (float)src[j] + offset;
        }
        dst += dst_stride / sizeof(float);
        src += src_stride;
    }
}

void picture_copy_hbd_avx2(float *dst, ptrdiff_t dst_stride,
                           const uint16_t *src, ptrdiff_t src_stride,
                           int offset, float inv_scaler,
                           unsigned w, unsigned h)
{
    const __m256 offset_vec = _mm256_set1_ps((float)offset);
    const __m256 inv_scaler_vec = _mm256_set1_ps(inv_scaler);
    const unsigned w8 = w & ~7u;

    for (unsigned i = 0; i < h; i++) {
        unsigned j = 0;
        for (; j < w8; j += 8) {
            /* Load 8 uint16 values, zero-extend to 32-bit, convert to float */
            __m128i u16 = _mm_loadu_si128((const __m128i *)(src + j));
            __m256i i32 = _mm256_cvtepu16_epi32(u16);
            __m256 f32 = _mm256_cvtepi32_ps(i32);
            /* Use separate mul + add (not fmadd) to match scalar rounding */
            f32 = _mm256_mul_ps(f32, inv_scaler_vec);
            f32 = _mm256_add_ps(f32, offset_vec);
            _mm256_storeu_ps(dst + j, f32);
        }
        /* Scalar tail */
        for (; j < w; j++) {
            dst[j] = (float)src[j] * inv_scaler + offset;
        }
        dst += dst_stride / sizeof(float);
        src += src_stride / 2;
    }
}
