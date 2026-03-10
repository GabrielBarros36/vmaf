/**
 * Fuzz target F1: Y4M header parser
 *
 * Feeds arbitrary data to the Y4M parser via fmemopen to exercise
 * y4m_parse_tags and y4m_input_open_impl, including all chroma format
 * branches, dimension computation, and buffer allocation.
 */

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "vidinput.h"

/* Limit maximum input size to prevent OOM on huge dimension values.
   The fuzzer should explore header parsing, not exhaust memory. */
#define MAX_INPUT_SIZE (1 << 20)  /* 1 MiB */

extern const video_input_vtbl Y4M_INPUT_VTBL;

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    if (size < 10 || size > MAX_INPUT_SIZE)
        return 0;

    /* Create a FILE* from the fuzzer-provided buffer. */
    FILE *f = fmemopen((void *)data, size, "rb");
    if (!f)
        return 0;

    /* Attempt to open as Y4M. This exercises y4m_parse_tags and
       y4m_input_open_impl, including all chroma format branches,
       dimension computation, and buffer allocation. */
    void *ctx = Y4M_INPUT_VTBL.open(f);
    if (ctx) {
        video_input_info info;
        Y4M_INPUT_VTBL.get_info(ctx, &info);

        /* Attempt to read one frame if the header parsed successfully.
           This exercises y4m_input_fetch_frame and convert functions. */
        video_input_ycbcr ycbcr;
        char tag[5];
        Y4M_INPUT_VTBL.fetch_frame(ctx, f, ycbcr, tag);

        Y4M_INPUT_VTBL.close(ctx);
        free(ctx);
    }

    fclose(f);
    return 0;
}
