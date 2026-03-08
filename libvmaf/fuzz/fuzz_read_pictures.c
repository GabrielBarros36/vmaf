/**
 * Fuzz target F3: Picture read pipeline
 *
 * Structured fuzzing: the first 2 bytes select width/height and bit depth
 * from a constrained set, and the remaining bytes fill pixel data. This
 * avoids spending fuzzer cycles on invalid picture configurations.
 */

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include "libvmaf/libvmaf.h"
#include "libvmaf/picture.h"
#include "libvmaf/model.h"

static const unsigned WIDTHS[]  = { 64, 128, 192, 576 };
static const unsigned HEIGHTS[] = { 64, 128, 108, 324 };
#define N_DIMS 4

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    if (size < 8)
        return 0;

    /* Use first 2 bytes to select dimensions and bit depth. */
    unsigned dim_idx = data[0] % N_DIMS;
    unsigned w = WIDTHS[dim_idx];
    unsigned h = HEIGHTS[dim_idx];
    unsigned bpc = (data[1] & 1) ? 10 : 8;

    data += 2;
    size -= 2;

    /* Compute minimum pixel data needed for YUV420P. */
    unsigned bytes_per_sample = (bpc > 8) ? 2 : 1;
    size_t luma_sz = (size_t)w * h * bytes_per_sample;
    size_t chroma_sz = (size_t)(w / 2) * (h / 2) * bytes_per_sample;
    size_t frame_sz = luma_sz + 2 * chroma_sz;

    if (size < frame_sz)
        return 0;

    /* Initialize VMAF context. */
    VmafContext *vmaf = NULL;
    VmafConfiguration cfg = {
        .log_level = VMAF_LOG_LEVEL_NONE,
        .n_threads = 1,
        .n_subsample = 0,
        .cpumask = 0,
        .gpumask = 0,
    };
    if (vmaf_init(&vmaf, cfg) < 0)
        return 0;

    /* Load built-in model. */
    VmafModel *model = NULL;
    VmafModelConfig model_cfg = {
        .name = "fuzz",
        .flags = VMAF_MODEL_FLAGS_DEFAULT,
    };
    if (vmaf_model_load(&model, &model_cfg, "vmaf_v0.6.1") < 0) {
        vmaf_close(vmaf);
        return 0;
    }
    if (vmaf_use_features_from_model(vmaf, model) < 0) {
        vmaf_model_destroy(model);
        vmaf_close(vmaf);
        return 0;
    }

    /* Allocate reference and distorted pictures. */
    VmafPicture ref, dist;
    if (vmaf_picture_alloc(&ref, VMAF_PIX_FMT_YUV420P, bpc, w, h) < 0) {
        vmaf_model_destroy(model);
        vmaf_close(vmaf);
        return 0;
    }
    if (vmaf_picture_alloc(&dist, VMAF_PIX_FMT_YUV420P, bpc, w, h) < 0) {
        vmaf_picture_unref(&ref);
        vmaf_model_destroy(model);
        vmaf_close(vmaf);
        return 0;
    }

    /* Fill reference with constant mid-gray. */
    unsigned mid = (bpc > 8) ? 512 : 128;
    for (int p = 0; p < 3; p++) {
        size_t plane_sz = (size_t)ref.w[p] * ref.h[p] * bytes_per_sample;
        memset(ref.data[p], mid & 0xFF, plane_sz);
    }

    /* Fill distorted with fuzz data. */
    const uint8_t *src = data;
    for (int p = 0; p < 3; p++) {
        size_t row_bytes = (size_t)dist.w[p] * bytes_per_sample;
        for (unsigned y = 0; y < dist.h[p]; y++) {
            uint8_t *row = (uint8_t *)dist.data[p] + y * dist.stride[p];
            size_t to_copy = row_bytes;
            if (src + to_copy > data + size)
                to_copy = (size_t)((data + size) - src);
            if (to_copy > 0)
                memcpy(row, src, to_copy);
            src += row_bytes;
        }
    }

    /* Process the frame pair. */
    vmaf_read_pictures(vmaf, &ref, &dist, 0);

    /* Flush. */
    vmaf_read_pictures(vmaf, NULL, NULL, 0);

    /* Read score (exercises predict path). */
    double score;
    vmaf_score_at_index(vmaf, model, &score, 0);

    /* Cleanup. */
    vmaf_model_destroy(model);
    vmaf_close(vmaf);

    return 0;
}
