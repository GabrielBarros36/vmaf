/**
 * Format and Codec Coverage Tests -- VmafPicture Format Handling
 *
 * Tests VmafPicture allocation across all pixel formats, bit depths,
 * stride alignment verification, and non-power-of-2 dimension handling.
 */

#include <stdint.h>
#include <string.h>
#include <errno.h>

#include "test.h"
#include "libvmaf/picture.h"

/* ====================================================================
 * Section 1: Pixel Format Allocation Tests
 *
 * Test vmaf_picture_alloc with all VmafPixelFormat values at various
 * bit depths.
 * ==================================================================== */

typedef struct {
    enum VmafPixelFormat pix_fmt;
    unsigned bpc;
    unsigned w;
    unsigned h;
    const char *label;
} alloc_test_case;

static const alloc_test_case alloc_cases[] = {
    /* YUV420P at all bit depths */
    { VMAF_PIX_FMT_YUV420P,  8,  320,  240, "420p_8bit_320x240" },
    { VMAF_PIX_FMT_YUV420P, 10,  320,  240, "420p_10bit_320x240" },
    { VMAF_PIX_FMT_YUV420P, 12,  320,  240, "420p_12bit_320x240" },
    { VMAF_PIX_FMT_YUV420P, 16,  320,  240, "420p_16bit_320x240" },
    { VMAF_PIX_FMT_YUV420P,  8, 1920, 1080, "420p_8bit_1920x1080" },
    { VMAF_PIX_FMT_YUV420P, 10, 1920, 1080, "420p_10bit_1920x1080" },

    /* YUV422P at all bit depths */
    { VMAF_PIX_FMT_YUV422P,  8,  320,  240, "422p_8bit_320x240" },
    { VMAF_PIX_FMT_YUV422P, 10,  320,  240, "422p_10bit_320x240" },
    { VMAF_PIX_FMT_YUV422P, 12,  320,  240, "422p_12bit_320x240" },
    { VMAF_PIX_FMT_YUV422P,  8, 1920, 1080, "422p_8bit_1920x1080" },

    /* YUV444P at all bit depths */
    { VMAF_PIX_FMT_YUV444P,  8,  320,  240, "444p_8bit_320x240" },
    { VMAF_PIX_FMT_YUV444P, 10,  320,  240, "444p_10bit_320x240" },
    { VMAF_PIX_FMT_YUV444P, 12,  320,  240, "444p_12bit_320x240" },
    { VMAF_PIX_FMT_YUV444P,  8, 1920, 1080, "444p_8bit_1920x1080" },

    /* YUV400P (luma only) */
    { VMAF_PIX_FMT_YUV400P,  8,  320,  240, "400p_8bit_320x240" },
    { VMAF_PIX_FMT_YUV400P, 10,  320,  240, "400p_10bit_320x240" },
    { VMAF_PIX_FMT_YUV400P, 12,  320,  240, "400p_12bit_320x240" },
    { VMAF_PIX_FMT_YUV400P,  8, 1920, 1080, "400p_8bit_1920x1080" },
};
#define NUM_ALLOC_CASES (sizeof(alloc_cases) / sizeof(alloc_cases[0]))

static char *test_pixel_format_allocation(void) {
    static char errbuf[256];

    for (unsigned i = 0; i < NUM_ALLOC_CASES; i++) {
        const alloc_test_case *tc = &alloc_cases[i];
        VmafPicture pic;
        int err = vmaf_picture_alloc(&pic, tc->pix_fmt, tc->bpc, tc->w, tc->h);
        if (err) {
            snprintf(errbuf, sizeof(errbuf),
                     "alloc failed for %s", tc->label);
            return errbuf;
        }

        /* Verify basic properties */
        if (pic.pix_fmt != tc->pix_fmt) {
            vmaf_picture_unref(&pic);
            snprintf(errbuf, sizeof(errbuf),
                     "%s: pix_fmt mismatch", tc->label);
            return errbuf;
        }
        if (pic.bpc != tc->bpc) {
            vmaf_picture_unref(&pic);
            snprintf(errbuf, sizeof(errbuf),
                     "%s: bpc mismatch", tc->label);
            return errbuf;
        }
        if (pic.w[0] != tc->w || pic.h[0] != tc->h) {
            vmaf_picture_unref(&pic);
            snprintf(errbuf, sizeof(errbuf),
                     "%s: luma dimensions mismatch", tc->label);
            return errbuf;
        }

        /* Verify chroma plane dimensions */
        if (tc->pix_fmt == VMAF_PIX_FMT_YUV444P) {
            if (pic.w[1] != tc->w || pic.h[1] != tc->h) {
                vmaf_picture_unref(&pic);
                snprintf(errbuf, sizeof(errbuf),
                         "%s: 444 chroma dims should equal luma", tc->label);
                return errbuf;
            }
        } else if (tc->pix_fmt == VMAF_PIX_FMT_YUV420P) {
            if (pic.w[1] != tc->w / 2 || pic.h[1] != tc->h / 2) {
                vmaf_picture_unref(&pic);
                snprintf(errbuf, sizeof(errbuf),
                         "%s: 420 chroma dims mismatch", tc->label);
                return errbuf;
            }
        } else if (tc->pix_fmt == VMAF_PIX_FMT_YUV422P) {
            if (pic.w[1] != tc->w / 2 || pic.h[1] != tc->h) {
                vmaf_picture_unref(&pic);
                snprintf(errbuf, sizeof(errbuf),
                         "%s: 422 chroma dims mismatch", tc->label);
                return errbuf;
            }
        } else if (tc->pix_fmt == VMAF_PIX_FMT_YUV400P) {
            if (pic.w[1] != 0 || pic.h[1] != 0) {
                vmaf_picture_unref(&pic);
                snprintf(errbuf, sizeof(errbuf),
                         "%s: 400 chroma dims should be 0", tc->label);
                return errbuf;
            }
            if (pic.data[1] != NULL || pic.data[2] != NULL) {
                vmaf_picture_unref(&pic);
                snprintf(errbuf, sizeof(errbuf),
                         "%s: 400 chroma data should be NULL", tc->label);
                return errbuf;
            }
        }

        /* Verify data pointers are non-NULL for luma */
        if (!pic.data[0]) {
            vmaf_picture_unref(&pic);
            snprintf(errbuf, sizeof(errbuf),
                     "%s: luma data pointer is NULL", tc->label);
            return errbuf;
        }

        vmaf_picture_unref(&pic);
    }

    return NULL;
}


/* ====================================================================
 * Section 2: Stride Alignment Tests
 *
 * Verify that all allocated pictures have 32-byte aligned strides
 * and data pointers.
 * ==================================================================== */

typedef struct {
    enum VmafPixelFormat pix_fmt;
    unsigned bpc;
    unsigned w;
    unsigned h;
    const char *label;
} stride_test_case;

static const stride_test_case stride_cases[] = {
    /* Various widths that test alignment edge cases */
    { VMAF_PIX_FMT_YUV420P,  8,    1,    1, "stride_1x1_420_8" },
    { VMAF_PIX_FMT_YUV420P,  8,   31,   16, "stride_31x16_420_8" },
    { VMAF_PIX_FMT_YUV420P,  8,   32,   16, "stride_32x16_420_8" },
    { VMAF_PIX_FMT_YUV420P,  8,   33,   16, "stride_33x16_420_8" },
    { VMAF_PIX_FMT_YUV420P, 10,   33,   16, "stride_33x16_420_10" },
    { VMAF_PIX_FMT_YUV420P,  8,  321,  241, "stride_321x241_420_8" },
    { VMAF_PIX_FMT_YUV420P, 10,  321,  241, "stride_321x241_420_10" },
    { VMAF_PIX_FMT_YUV422P,  8,  321,  241, "stride_321x241_422_8" },
    { VMAF_PIX_FMT_YUV422P, 10,  321,  241, "stride_321x241_422_10" },
    { VMAF_PIX_FMT_YUV444P,  8,  321,  241, "stride_321x241_444_8" },
    { VMAF_PIX_FMT_YUV444P, 10,  321,  241, "stride_321x241_444_10" },
    { VMAF_PIX_FMT_YUV444P, 12,  321,  241, "stride_321x241_444_12" },
    { VMAF_PIX_FMT_YUV400P,  8,  321,  241, "stride_321x241_400_8" },
    { VMAF_PIX_FMT_YUV420P,  8, 1920, 1080, "stride_1920x1080_420_8" },
    { VMAF_PIX_FMT_YUV420P, 10, 1921, 1080, "stride_1921x1080_420_10" },
    { VMAF_PIX_FMT_YUV444P,  8, 7680, 4320, "stride_7680x4320_444_8" },
};
#define NUM_STRIDE_CASES (sizeof(stride_cases) / sizeof(stride_cases[0]))

static char *test_stride_alignment(void) {
    static char errbuf[256];

    for (unsigned i = 0; i < NUM_STRIDE_CASES; i++) {
        const stride_test_case *tc = &stride_cases[i];
        VmafPicture pic;
        int err = vmaf_picture_alloc(&pic, tc->pix_fmt, tc->bpc, tc->w, tc->h);
        if (err) {
            snprintf(errbuf, sizeof(errbuf),
                     "%s: alloc failed", tc->label);
            return errbuf;
        }

        /* Check luma stride alignment */
        if (pic.stride[0] % 32 != 0) {
            vmaf_picture_unref(&pic);
            snprintf(errbuf, sizeof(errbuf),
                     "%s: luma stride %td not 32-byte aligned",
                     tc->label, pic.stride[0]);
            return errbuf;
        }

        /* Check luma data pointer alignment */
        if (((uintptr_t)pic.data[0]) % 32 != 0) {
            vmaf_picture_unref(&pic);
            snprintf(errbuf, sizeof(errbuf),
                     "%s: luma data pointer not 32-byte aligned",
                     tc->label);
            return errbuf;
        }

        /* Verify luma stride is at least as wide as needed */
        unsigned bytes_per_sample = (tc->bpc > 8) ? 2 : 1;
        if ((unsigned)pic.stride[0] < tc->w * bytes_per_sample) {
            vmaf_picture_unref(&pic);
            snprintf(errbuf, sizeof(errbuf),
                     "%s: luma stride too small", tc->label);
            return errbuf;
        }

        /* Check chroma stride and data alignment (if not 400) */
        if (tc->pix_fmt != VMAF_PIX_FMT_YUV400P) {
            if (pic.stride[1] % 32 != 0) {
                vmaf_picture_unref(&pic);
                snprintf(errbuf, sizeof(errbuf),
                         "%s: chroma stride %td not 32-byte aligned",
                         tc->label, pic.stride[1]);
                return errbuf;
            }
            if (((uintptr_t)pic.data[1]) % 32 != 0) {
                vmaf_picture_unref(&pic);
                snprintf(errbuf, sizeof(errbuf),
                         "%s: Cb data pointer not 32-byte aligned",
                         tc->label);
                return errbuf;
            }
            if (((uintptr_t)pic.data[2]) % 32 != 0) {
                vmaf_picture_unref(&pic);
                snprintf(errbuf, sizeof(errbuf),
                         "%s: Cr data pointer not 32-byte aligned",
                         tc->label);
                return errbuf;
            }
        }

        vmaf_picture_unref(&pic);
    }

    return NULL;
}


/* ====================================================================
 * Section 3: Non-Power-of-2 Dimensions Tests
 *
 * Test allocation with odd widths and heights to verify proper
 * chroma plane rounding.
 * ==================================================================== */

typedef struct {
    unsigned w, h;
    enum VmafPixelFormat pix_fmt;
    unsigned bpc;
    unsigned exp_cw, exp_ch; /* expected chroma width/height */
    const char *label;
} odd_dim_test_case;

static const odd_dim_test_case odd_dim_cases[] = {
    /* 420: chroma = w>>1, h>>1 */
    {   3,   4, VMAF_PIX_FMT_YUV420P,  8,   1,   2, "odd_3x4_420" },
    {   4,   3, VMAF_PIX_FMT_YUV420P,  8,   2,   1, "odd_4x3_420" },
    {   3,   3, VMAF_PIX_FMT_YUV420P,  8,   1,   1, "odd_3x3_420" },
    {   5,   5, VMAF_PIX_FMT_YUV420P,  8,   2,   2, "odd_5x5_420" },
    {  15,  17, VMAF_PIX_FMT_YUV420P,  8,   7,   8, "odd_15x17_420" },
    { 321, 241, VMAF_PIX_FMT_YUV420P,  8, 160, 120, "odd_321x241_420" },
    { 321, 241, VMAF_PIX_FMT_YUV420P, 10, 160, 120, "odd_321x241_420_10" },
    { 321, 241, VMAF_PIX_FMT_YUV420P, 12, 160, 120, "odd_321x241_420_12" },

    /* 422: chroma = w>>1, h */
    {   3,   4, VMAF_PIX_FMT_YUV422P,  8,   1,   4, "odd_3x4_422" },
    {   5,   7, VMAF_PIX_FMT_YUV422P,  8,   2,   7, "odd_5x7_422" },
    { 321, 241, VMAF_PIX_FMT_YUV422P,  8, 160, 241, "odd_321x241_422" },
    { 321, 241, VMAF_PIX_FMT_YUV422P, 10, 160, 241, "odd_321x241_422_10" },

    /* 444: chroma = w, h */
    {   3,   3, VMAF_PIX_FMT_YUV444P,  8,   3,   3, "odd_3x3_444" },
    { 321, 241, VMAF_PIX_FMT_YUV444P,  8, 321, 241, "odd_321x241_444" },
    { 321, 241, VMAF_PIX_FMT_YUV444P, 10, 321, 241, "odd_321x241_444_10" },

    /* 400: chroma = 0, 0 */
    {   3,   3, VMAF_PIX_FMT_YUV400P,  8,   0,   0, "odd_3x3_400" },
    { 321, 241, VMAF_PIX_FMT_YUV400P,  8,   0,   0, "odd_321x241_400" },

    /* Very small dimensions */
    {   1,   1, VMAF_PIX_FMT_YUV420P,  8,   0,   0, "odd_1x1_420" },
    {   2,   1, VMAF_PIX_FMT_YUV420P,  8,   1,   0, "odd_2x1_420" },
    {   1,   2, VMAF_PIX_FMT_YUV420P,  8,   0,   1, "odd_1x2_420" },
    {   2,   2, VMAF_PIX_FMT_YUV420P,  8,   1,   1, "odd_2x2_420" },
};
#define NUM_ODD_DIM_CASES (sizeof(odd_dim_cases) / sizeof(odd_dim_cases[0]))

static char *test_odd_dimensions(void) {
    static char errbuf[256];

    for (unsigned i = 0; i < NUM_ODD_DIM_CASES; i++) {
        const odd_dim_test_case *tc = &odd_dim_cases[i];
        VmafPicture pic;
        int err = vmaf_picture_alloc(&pic, tc->pix_fmt, tc->bpc, tc->w, tc->h);
        if (err) {
            snprintf(errbuf, sizeof(errbuf),
                     "%s: alloc failed", tc->label);
            return errbuf;
        }

        /* Verify luma dimensions */
        if (pic.w[0] != tc->w || pic.h[0] != tc->h) {
            vmaf_picture_unref(&pic);
            snprintf(errbuf, sizeof(errbuf),
                     "%s: luma dims mismatch (got %ux%u, want %ux%u)",
                     tc->label, pic.w[0], pic.h[0], tc->w, tc->h);
            return errbuf;
        }

        /* Verify chroma dimensions */
        if (pic.w[1] != tc->exp_cw || pic.h[1] != tc->exp_ch) {
            vmaf_picture_unref(&pic);
            snprintf(errbuf, sizeof(errbuf),
                     "%s: chroma dims mismatch (got %ux%u, want %ux%u)",
                     tc->label, pic.w[1], pic.h[1],
                     tc->exp_cw, tc->exp_ch);
            return errbuf;
        }

        /* Verify stride alignment */
        if (pic.stride[0] % 32 != 0) {
            vmaf_picture_unref(&pic);
            snprintf(errbuf, sizeof(errbuf),
                     "%s: luma stride not 32-byte aligned", tc->label);
            return errbuf;
        }

        vmaf_picture_unref(&pic);
    }

    return NULL;
}


/* ====================================================================
 * Section 4: Stride Correctness Tests
 *
 * Verify that writing to the picture buffer through stride-based
 * access works correctly (no overlap between planes, padding is zero).
 * ==================================================================== */

static char *test_stride_write_read(void) {
    static char errbuf[256];

    /* Test with several format/dimension combos */
    struct {
        enum VmafPixelFormat pf;
        unsigned bpc, w, h;
        const char *label;
    } cases[] = {
        { VMAF_PIX_FMT_YUV420P,  8, 320, 240, "write_read_420_8" },
        { VMAF_PIX_FMT_YUV420P, 10, 320, 240, "write_read_420_10" },
        { VMAF_PIX_FMT_YUV422P,  8, 321, 241, "write_read_422_8_odd" },
        { VMAF_PIX_FMT_YUV444P,  8, 321, 241, "write_read_444_8_odd" },
        { VMAF_PIX_FMT_YUV444P, 12, 320, 240, "write_read_444_12" },
    };

    for (unsigned ci = 0; ci < sizeof(cases)/sizeof(cases[0]); ci++) {
        VmafPicture pic;
        int err = vmaf_picture_alloc(&pic, cases[ci].pf, cases[ci].bpc,
                                     cases[ci].w, cases[ci].h);
        if (err) {
            snprintf(errbuf, sizeof(errbuf),
                     "%s: alloc failed", cases[ci].label);
            return errbuf;
        }

        /* Write a gradient pattern to luma plane using stride-based access */
        unsigned bps = (cases[ci].bpc > 8) ? 2 : 1;
        for (unsigned y = 0; y < pic.h[0]; y++) {
            uint8_t *row = (uint8_t *)pic.data[0] + y * pic.stride[0];
            for (unsigned x = 0; x < pic.w[0]; x++) {
                uint16_t val = (uint16_t)((x + y) % ((1u << cases[ci].bpc) - 1));
                if (bps == 2) {
                    row[x * 2]     = val & 0xFF;
                    row[x * 2 + 1] = (val >> 8) & 0xFF;
                } else {
                    row[x] = (uint8_t)val;
                }
            }
        }

        /* Read back and verify */
        for (unsigned y = 0; y < pic.h[0]; y++) {
            uint8_t *row = (uint8_t *)pic.data[0] + y * pic.stride[0];
            for (unsigned x = 0; x < pic.w[0]; x++) {
                uint16_t expected = (uint16_t)((x + y) % ((1u << cases[ci].bpc) - 1));
                uint16_t got;
                if (bps == 2) {
                    got = (uint16_t)row[x * 2] | ((uint16_t)row[x * 2 + 1] << 8);
                } else {
                    got = row[x];
                }
                if (got != expected) {
                    vmaf_picture_unref(&pic);
                    snprintf(errbuf, sizeof(errbuf),
                             "%s: pixel[%u,%u] mismatch: got %u want %u",
                             cases[ci].label, y, x, got, expected);
                    return errbuf;
                }
            }
        }

        /* Verify padding bytes are still zero (vmaf_picture_alloc memsets to 0) */
        for (unsigned y = 0; y < pic.h[0]; y++) {
            uint8_t *row = (uint8_t *)pic.data[0] + y * pic.stride[0];
            unsigned data_bytes = pic.w[0] * bps;
            for (ptrdiff_t b = data_bytes; b < pic.stride[0]; b++) {
                if (row[b] != 0) {
                    vmaf_picture_unref(&pic);
                    snprintf(errbuf, sizeof(errbuf),
                             "%s: padding byte at row %u offset %td is %u (want 0)",
                             cases[ci].label, y, b, row[b]);
                    return errbuf;
                }
            }
        }

        vmaf_picture_unref(&pic);
    }

    return NULL;
}


/* ====================================================================
 * Section 5: High Bit Depth Stride Tests
 *
 * Verify stride calculations for high bit depth (10/12/16-bit).
 * ==================================================================== */

static char *test_hbd_stride(void) {
    static char errbuf[256];

    struct {
        enum VmafPixelFormat pf;
        unsigned bpc, w, h;
        const char *label;
    } cases[] = {
        { VMAF_PIX_FMT_YUV420P, 10, 320, 240, "hbd_420_10_320" },
        { VMAF_PIX_FMT_YUV420P, 12, 320, 240, "hbd_420_12_320" },
        { VMAF_PIX_FMT_YUV420P, 10, 321, 241, "hbd_420_10_321" },
        { VMAF_PIX_FMT_YUV422P, 10, 321, 241, "hbd_422_10_321" },
        { VMAF_PIX_FMT_YUV444P, 10, 321, 241, "hbd_444_10_321" },
        { VMAF_PIX_FMT_YUV444P, 12, 321, 241, "hbd_444_12_321" },
    };

    for (unsigned ci = 0; ci < sizeof(cases)/sizeof(cases[0]); ci++) {
        VmafPicture pic;
        int err = vmaf_picture_alloc(&pic, cases[ci].pf, cases[ci].bpc,
                                     cases[ci].w, cases[ci].h);
        if (err) {
            snprintf(errbuf, sizeof(errbuf),
                     "%s: alloc failed", cases[ci].label);
            return errbuf;
        }

        /* For HBD, stride should be >= width * 2 and 32-byte aligned */
        if ((unsigned)pic.stride[0] < cases[ci].w * 2) {
            vmaf_picture_unref(&pic);
            snprintf(errbuf, sizeof(errbuf),
                     "%s: luma stride %td too small for HBD (need >= %u)",
                     cases[ci].label, pic.stride[0], cases[ci].w * 2);
            return errbuf;
        }

        if (pic.stride[0] % 32 != 0) {
            vmaf_picture_unref(&pic);
            snprintf(errbuf, sizeof(errbuf),
                     "%s: luma stride not 32-byte aligned", cases[ci].label);
            return errbuf;
        }

        /* Verify stride is computed as: ((w + 31) & ~31) << 1 */
        unsigned aligned_w = (cases[ci].w + 31) & ~31u;
        ptrdiff_t expected_stride = (ptrdiff_t)aligned_w << 1;
        if (pic.stride[0] != expected_stride) {
            vmaf_picture_unref(&pic);
            snprintf(errbuf, sizeof(errbuf),
                     "%s: luma stride %td != expected %td",
                     cases[ci].label, pic.stride[0], expected_stride);
            return errbuf;
        }

        vmaf_picture_unref(&pic);
    }

    return NULL;
}


/* ====================================================================
 * Section 6: Error Handling for Picture Allocation
 * ==================================================================== */

static char *test_alloc_errors(void) {
    VmafPicture pic;
    int err;

    /* NULL pointer */
    err = vmaf_picture_alloc(NULL, VMAF_PIX_FMT_YUV420P, 8, 320, 240);
    mu_assert("alloc_errors: NULL pic must fail", err == -EINVAL);

    /* UNKNOWN format */
    err = vmaf_picture_alloc(&pic, VMAF_PIX_FMT_UNKNOWN, 8, 320, 240);
    mu_assert("alloc_errors: UNKNOWN fmt must fail", err == -EINVAL);

    /* bpc too low */
    err = vmaf_picture_alloc(&pic, VMAF_PIX_FMT_YUV420P, 7, 320, 240);
    mu_assert("alloc_errors: bpc=7 must fail", err == -EINVAL);

    /* bpc too high */
    err = vmaf_picture_alloc(&pic, VMAF_PIX_FMT_YUV420P, 17, 320, 240);
    mu_assert("alloc_errors: bpc=17 must fail", err == -EINVAL);

    /* bpc = 0 */
    err = vmaf_picture_alloc(&pic, VMAF_PIX_FMT_YUV420P, 0, 320, 240);
    mu_assert("alloc_errors: bpc=0 must fail", err == -EINVAL);

    return NULL;
}


/* ====================================================================
 * Section 7: 400P (Mono) Specific Tests
 * ==================================================================== */

static char *test_yuv400p_specifics(void) {
    VmafPicture pic;
    int err = vmaf_picture_alloc(&pic, VMAF_PIX_FMT_YUV400P, 8, 320, 240);
    mu_assert("400p: alloc succeeds", err == 0);

    mu_assert("400p: luma w", pic.w[0] == 320);
    mu_assert("400p: luma h", pic.h[0] == 240);
    mu_assert("400p: chroma w is 0", pic.w[1] == 0);
    mu_assert("400p: chroma h is 0", pic.h[1] == 0);
    mu_assert("400p: Cb data is NULL", pic.data[1] == NULL);
    mu_assert("400p: Cr data is NULL", pic.data[2] == NULL);
    mu_assert("400p: luma stride aligned", (pic.stride[0] % 32) == 0);

    /* Write to luma plane -- should not crash */
    memset(pic.data[0], 128, pic.stride[0] * pic.h[0]);

    vmaf_picture_unref(&pic);
    return NULL;
}

static char *test_yuv400p_10bit(void) {
    VmafPicture pic;
    int err = vmaf_picture_alloc(&pic, VMAF_PIX_FMT_YUV400P, 10, 320, 240);
    mu_assert("400p_10: alloc succeeds", err == 0);
    mu_assert("400p_10: Cb data is NULL", pic.data[1] == NULL);
    mu_assert("400p_10: Cr data is NULL", pic.data[2] == NULL);
    mu_assert("400p_10: stride >= width*2", (unsigned)pic.stride[0] >= 320 * 2);
    vmaf_picture_unref(&pic);
    return NULL;
}


/* ====================================================================
 * Test Runner
 * ==================================================================== */

char *run_tests(void) {
    /* Pixel format allocation across all formats and bit depths */
    mu_run_test(test_pixel_format_allocation);

    /* Stride alignment verification */
    mu_run_test(test_stride_alignment);

    /* Non-power-of-2 / odd dimension tests */
    mu_run_test(test_odd_dimensions);

    /* Stride write/read correctness */
    mu_run_test(test_stride_write_read);

    /* High bit depth stride calculations */
    mu_run_test(test_hbd_stride);

    /* Error handling */
    mu_run_test(test_alloc_errors);

    /* YUV400P specifics */
    mu_run_test(test_yuv400p_specifics);
    mu_run_test(test_yuv400p_10bit);

    return NULL;
}
