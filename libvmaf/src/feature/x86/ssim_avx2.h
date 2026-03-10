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

#ifndef X86_AVX2_SSIM_H_
#define X86_AVX2_SSIM_H_

#include <stdint.h>

/*
 * Compute horizontal Gaussian-windowed SSIM moments for one row of 8-bit data.
 *
 * For each output pixel x in [0, w), this accumulates the windowed moments
 * (mux, muy, x2, xy, y2, w) using the horizontal kernel, reading from
 * the src/dst rows with appropriate clamping at boundaries.
 *
 * Parameters:
 *   src         - pointer to current row of reference image (uint8_t)
 *   dst         - pointer to current row of distorted image (uint8_t)
 *   buf_mux     - output array for mux moments (int64_t, length w)
 *   buf_muy     - output array for muy moments (int64_t, length w)
 *   buf_x2      - output array for x2 moments (int64_t, length w)
 *   buf_xy      - output array for xy moments (int64_t, length w)
 *   buf_y2      - output array for y2 moments (int64_t, length w)
 *   buf_w       - output array for w (weight) moments (int64_t, length w)
 *   w           - image width
 *   hkernel     - horizontal Gaussian kernel coefficients (unsigned)
 *   hkernel_sz  - number of kernel coefficients
 */
void ssim_hconv_8_avx2(const uint8_t *src, const uint8_t *dst,
                        int64_t *buf_mux, int64_t *buf_muy,
                        int64_t *buf_x2, int64_t *buf_xy,
                        int64_t *buf_y2, int64_t *buf_w,
                        int w, const unsigned *hkernel, int hkernel_sz);

#endif /* X86_AVX2_SSIM_H_ */
