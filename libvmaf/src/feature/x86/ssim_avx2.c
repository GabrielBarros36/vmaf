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
#include <string.h>

/*
 * AVX2-vectorized horizontal Gaussian moment accumulation for 8-bit SSIM.
 *
 * For each output pixel x, the scalar code computes:
 *   for each kernel tap k in [k_min, k_max):
 *     s = src[x - hkernel_offs + k]
 *     d = dst[x - hkernel_offs + k]
 *     window = hkernel[k]
 *     mux += window * s
 *     muy += window * d
 *     x2  += window * s * s
 *     xy  += window * s * d
 *     y2  += window * d * d
 *     w   += window
 *
 * For 8-bit data with a small kernel (typically 5 taps, weights summing to
 * 256), we can safely accumulate in 32-bit integers:
 *   max window*s*s = 256 * 255 * 255 = 16,646,400  (fits int32)
 *   5 taps: 5 * 16,646,400 = 83,232,000  (fits int32)
 *
 * Strategy: Process 8 output pixels at a time. For each kernel tap, load
 * 8 src and 8 dst values (as 16-bit in 256-bit registers), multiply by
 * the window coefficient, and accumulate the six moments.
 *
 * We store results into int64_t output arrays because that's what the
 * caller (ssim_moments struct) expects, but we accumulate in int32 and
 * convert at the end.
 */
void ssim_hconv_8_avx2(const uint8_t *src, const uint8_t *dst,
                        int64_t *buf_mux, int64_t *buf_muy,
                        int64_t *buf_x2, int64_t *buf_xy,
                        int64_t *buf_y2, int64_t *buf_w,
                        int w, const unsigned *hkernel, int hkernel_sz)
{
    const int hkernel_offs = hkernel_sz >> 1;

    /* Region where the full kernel fits without boundary issues */
    const int inner_start = hkernel_offs;
    const int inner_end = w - hkernel_offs;

    /* Handle left boundary pixels with scalar code */
    for (int x = 0; x < inner_start && x < w; x++) {
        int k_min = hkernel_offs - x > 0 ? hkernel_offs - x : 0;
        int k_max = x + hkernel_offs - w + 1 > 0
                        ? hkernel_sz - (x + hkernel_offs - w + 1)
                        : hkernel_sz;
        int64_t mux = 0, muy = 0, x2 = 0, xy = 0, y2 = 0, wt = 0;
        for (int k = k_min; k < k_max; k++) {
            int s = src[x - hkernel_offs + k];
            int d = dst[x - hkernel_offs + k];
            int window = (int)hkernel[k];
            mux += window * s;
            muy += window * d;
            x2  += window * s * s;
            xy  += window * s * d;
            y2  += window * d * d;
            wt  += window;
        }
        buf_mux[x] = mux;
        buf_muy[x] = muy;
        buf_x2[x]  = x2;
        buf_xy[x]  = xy;
        buf_y2[x]  = y2;
        buf_w[x]   = wt;
    }

    /* Vectorized inner region: process 8 pixels at a time */
    if (inner_end > inner_start) {
        int x = inner_start;

        for (; x + 7 < inner_end; x += 8) {
            __m256i acc_mux = _mm256_setzero_si256();
            __m256i acc_muy = _mm256_setzero_si256();
            __m256i acc_x2  = _mm256_setzero_si256();
            __m256i acc_xy  = _mm256_setzero_si256();
            __m256i acc_y2  = _mm256_setzero_si256();
            __m256i acc_w   = _mm256_setzero_si256();

            for (int k = 0; k < hkernel_sz; k++) {
                int tap_idx = x - hkernel_offs + k;
                __m256i win = _mm256_set1_epi32((int)hkernel[k]);

                /*
                 * Load 8 uint8 pixels, zero-extend to 32-bit.
                 * We load into a 64-bit GPR-width via _mm_loadl_epi64
                 * then use _mm256_cvtepu8_epi32 to get 8x32-bit in YMM.
                 */
                __m128i s8 = _mm_loadl_epi64((const __m128i *)(src + tap_idx));
                __m256i s32 = _mm256_cvtepu8_epi32(s8);

                __m128i d8 = _mm_loadl_epi64((const __m128i *)(dst + tap_idx));
                __m256i d32 = _mm256_cvtepu8_epi32(d8);

                /* window * s, window * d */
                __m256i ws = _mm256_mullo_epi32(win, s32);
                __m256i wd = _mm256_mullo_epi32(win, d32);

                acc_mux = _mm256_add_epi32(acc_mux, ws);
                acc_muy = _mm256_add_epi32(acc_muy, wd);

                /* window * s * s = ws * s */
                __m256i wss = _mm256_mullo_epi32(ws, s32);
                acc_x2 = _mm256_add_epi32(acc_x2, wss);

                /* window * s * d = ws * d */
                __m256i wsd = _mm256_mullo_epi32(ws, d32);
                acc_xy = _mm256_add_epi32(acc_xy, wsd);

                /* window * d * d = wd * d */
                __m256i wdd = _mm256_mullo_epi32(wd, d32);
                acc_y2 = _mm256_add_epi32(acc_y2, wdd);

                acc_w = _mm256_add_epi32(acc_w, win);
            }

            /*
             * Convert 8x int32 accumulators to int64 and store.
             * Split each 256-bit register into low 4 and high 4 int32 values,
             * then sign-extend to int64.
             */
            __m128i mux_lo128 = _mm256_castsi256_si128(acc_mux);
            __m128i mux_hi128 = _mm256_extracti128_si256(acc_mux, 1);
            _mm256_storeu_si256((__m256i *)(buf_mux + x),
                                _mm256_cvtepi32_epi64(mux_lo128));
            _mm256_storeu_si256((__m256i *)(buf_mux + x + 4),
                                _mm256_cvtepi32_epi64(mux_hi128));

            __m128i muy_lo128 = _mm256_castsi256_si128(acc_muy);
            __m128i muy_hi128 = _mm256_extracti128_si256(acc_muy, 1);
            _mm256_storeu_si256((__m256i *)(buf_muy + x),
                                _mm256_cvtepi32_epi64(muy_lo128));
            _mm256_storeu_si256((__m256i *)(buf_muy + x + 4),
                                _mm256_cvtepi32_epi64(muy_hi128));

            __m128i x2_lo128 = _mm256_castsi256_si128(acc_x2);
            __m128i x2_hi128 = _mm256_extracti128_si256(acc_x2, 1);
            _mm256_storeu_si256((__m256i *)(buf_x2 + x),
                                _mm256_cvtepi32_epi64(x2_lo128));
            _mm256_storeu_si256((__m256i *)(buf_x2 + x + 4),
                                _mm256_cvtepi32_epi64(x2_hi128));

            __m128i xy_lo128 = _mm256_castsi256_si128(acc_xy);
            __m128i xy_hi128 = _mm256_extracti128_si256(acc_xy, 1);
            _mm256_storeu_si256((__m256i *)(buf_xy + x),
                                _mm256_cvtepi32_epi64(xy_lo128));
            _mm256_storeu_si256((__m256i *)(buf_xy + x + 4),
                                _mm256_cvtepi32_epi64(xy_hi128));

            __m128i y2_lo128 = _mm256_castsi256_si128(acc_y2);
            __m128i y2_hi128 = _mm256_extracti128_si256(acc_y2, 1);
            _mm256_storeu_si256((__m256i *)(buf_y2 + x),
                                _mm256_cvtepi32_epi64(y2_lo128));
            _mm256_storeu_si256((__m256i *)(buf_y2 + x + 4),
                                _mm256_cvtepi32_epi64(y2_hi128));

            __m128i w_lo128 = _mm256_castsi256_si128(acc_w);
            __m128i w_hi128 = _mm256_extracti128_si256(acc_w, 1);
            _mm256_storeu_si256((__m256i *)(buf_w + x),
                                _mm256_cvtepi32_epi64(w_lo128));
            _mm256_storeu_si256((__m256i *)(buf_w + x + 4),
                                _mm256_cvtepi32_epi64(w_hi128));
        }

        /* Scalar tail for remaining inner pixels */
        for (; x < inner_end; x++) {
            int64_t mux = 0, muy = 0, x2 = 0, xy = 0, y2 = 0, wt = 0;
            for (int k = 0; k < hkernel_sz; k++) {
                int s = src[x - hkernel_offs + k];
                int d = dst[x - hkernel_offs + k];
                int window = (int)hkernel[k];
                mux += window * s;
                muy += window * d;
                x2  += window * s * s;
                xy  += window * s * d;
                y2  += window * d * d;
                wt  += window;
            }
            buf_mux[x] = mux;
            buf_muy[x] = muy;
            buf_x2[x]  = x2;
            buf_xy[x]  = xy;
            buf_y2[x]  = y2;
            buf_w[x]   = wt;
        }
    }

    /* Handle right boundary pixels with scalar code */
    for (int x = (inner_end > inner_start ? inner_end : inner_start); x < w; x++) {
        int k_min = hkernel_offs - x > 0 ? hkernel_offs - x : 0;
        int k_max = x + hkernel_offs - w + 1 > 0
                        ? hkernel_sz - (x + hkernel_offs - w + 1)
                        : hkernel_sz;
        int64_t mux = 0, muy = 0, x2 = 0, xy = 0, y2 = 0, wt = 0;
        for (int k = k_min; k < k_max; k++) {
            int s = src[x - hkernel_offs + k];
            int d = dst[x - hkernel_offs + k];
            int window = (int)hkernel[k];
            mux += window * s;
            muy += window * d;
            x2  += window * s * s;
            xy  += window * s * d;
            y2  += window * d * d;
            wt  += window;
        }
        buf_mux[x] = mux;
        buf_muy[x] = muy;
        buf_x2[x]  = x2;
        buf_xy[x]  = xy;
        buf_y2[x]  = y2;
        buf_w[x]   = wt;
    }
}
