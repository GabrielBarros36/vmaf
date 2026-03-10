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

#include <stdint.h>

#include <libvmaf/picture.h>
#include "common/macros.h"

void picture_copy_hbd(float *RESTRICT dst, ptrdiff_t dst_stride,
                      VmafPicture *RESTRICT src, int offset, float scaler)
{
    float *float_data = dst;
    uint16_t *data = src->data[0];
    const unsigned w = src->w[0];
    const unsigned h = src->h[0];

    /*
     * When both source and destination planes are contiguous (stride equals
     * the actual row width), flatten into a single pass over all pixels.
     * This eliminates per-row pointer arithmetic and allows the compiler to
     * better vectorize the conversion loop.
     */
    if ((ptrdiff_t)(w * sizeof(uint16_t)) == src->stride[0] &&
        (ptrdiff_t)(w * sizeof(float)) == dst_stride) {
        const unsigned total = w * h;
        for (unsigned j = 0; j < total; j++) {
            float_data[j] = (float) data[j] / scaler + offset;
        }
        return;
    }

    for (unsigned i = 0; i < h; i++) {
        for (unsigned j = 0; j < w; j++) {
            float_data[j] = (float) data[j] / scaler + offset;
        }
        float_data += dst_stride / sizeof(float);
        data += src->stride[0] / 2;
    }
    return;
}

void picture_copy(float *RESTRICT dst, ptrdiff_t dst_stride,
                  VmafPicture *RESTRICT src, int offset, unsigned bpc)
{
    if (bpc == 10) {
        picture_copy_hbd(dst, dst_stride, src, offset, 4.0f);
        return;
    } else if (bpc == 12) {
        picture_copy_hbd(dst, dst_stride, src, offset, 16.0f);
        return;
    } else if (bpc == 16) {
        picture_copy_hbd(dst, dst_stride, src, offset, 256.0f);
        return;
    }

    float *float_data = dst;
    uint8_t *data = src->data[0];
    const unsigned w = src->w[0];
    const unsigned h = src->h[0];

    /*
     * When both source and destination planes are contiguous (stride equals
     * the actual row width), flatten into a single pass over all pixels.
     * This eliminates per-row pointer arithmetic and allows the compiler to
     * better vectorize the conversion loop.
     */
    if ((ptrdiff_t)(w * sizeof(uint8_t)) == src->stride[0] &&
        (ptrdiff_t)(w * sizeof(float)) == dst_stride) {
        const unsigned total = w * h;
        for (unsigned j = 0; j < total; j++) {
            float_data[j] = (float) data[j] + offset;
        }
        return;
    }

    for (unsigned i = 0; i < h; i++) {
        for (unsigned j = 0; j < w; j++) {
            float_data[j] = (float) data[j] + offset;
        }
        float_data += dst_stride / sizeof(float);
        data += src->stride[0];
    }

    return;
}
