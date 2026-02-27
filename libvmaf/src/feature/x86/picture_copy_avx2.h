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

#ifndef PICTURE_COPY_AVX2_H_
#define PICTURE_COPY_AVX2_H_

#include <stddef.h>
#include <stdint.h>

void picture_copy_8bit_avx2(float *dst, ptrdiff_t dst_stride,
                            const uint8_t *src, ptrdiff_t src_stride,
                            int offset, unsigned w, unsigned h);

void picture_copy_hbd_avx2(float *dst, ptrdiff_t dst_stride,
                           const uint16_t *src, ptrdiff_t src_stride,
                           int offset, float inv_scaler,
                           unsigned w, unsigned h);

#endif /* PICTURE_COPY_AVX2_H_ */
