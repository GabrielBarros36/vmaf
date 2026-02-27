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

#ifndef X86_AVX2_ADM_H_
#define X86_AVX2_ADM_H_

#include "feature/integer_adm.h"

void adm_dwt2_8_avx2(const uint8_t *src, const adm_dwt_band_t *dst,
                     AdmBuffer *buf, int w, int h, int src_stride,
                     int dst_stride);

void adm_csf_s0_avx2(AdmBuffer *buf, int w, int h, int stride,
                      const uint16_t i_rfactor[3],
                      const uint8_t i_shifts[3],
                      const uint16_t i_shiftsadd[3]);

#endif /* X86_AVX2_ADM_H_ */
