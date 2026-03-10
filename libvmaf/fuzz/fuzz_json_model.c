/**
 * Fuzz target F2: Model JSON loader
 *
 * Feeds arbitrary data to vmaf_read_json_model_from_buffer to exercise
 * the pdjson streaming parser and SVM model parsing paths.
 */

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include "libvmaf/model.h"

/* Forward declaration of the buffer-based loader. */
int vmaf_read_json_model_from_buffer(VmafModel **model, VmafModelConfig *cfg,
                                     const char *data, const int data_len);

#define MAX_INPUT_SIZE (1 << 20)  /* 1 MiB */

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    if (size == 0 || size > MAX_INPUT_SIZE)
        return 0;

    VmafModel *model = NULL;
    VmafModelConfig cfg = {
        .name = "fuzz",
        .flags = VMAF_MODEL_FLAGS_DEFAULT,
    };

    int err = vmaf_read_json_model_from_buffer(&model, &cfg,
                                                (const char *)data,
                                                (int)size);
    if (model) {
        vmaf_model_destroy(model);
    }

    return 0;
}
