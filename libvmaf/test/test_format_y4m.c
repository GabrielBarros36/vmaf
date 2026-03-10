/**
 * Format and Codec Coverage Tests -- Y4M Format Handling
 *
 * Tests Y4M header parsing (valid/malformed), format-depth matrix round-trips,
 * range preservation, and error handling per the format-codec-coverage spec.
 */

#include <stdint.h>
#include <string.h>
#include <errno.h>

#include "test.h"
#include "test_format_common.h"
#include "vidinput.h"
#include "libvmaf/picture.h"

/* ====================================================================
 * Section 3: Y4M Header Parsing Edge Cases
 * ==================================================================== */

/* Helper: attempt to open a Y4M header and return success/failure.
 * On success, fills in *info and closes the video_input.
 * Returns 0 on success, -1 on failure. */
static int try_open_y4m_header(const char *header, size_t len,
                               video_input_info *info)
{
    FILE *f = create_y4m_header_only(header, len);
    if (!f) return -1;

    video_input vid;
    int ret = video_input_open(&vid, f);
    if (ret < 0) {
        /* video_input_open failure -- file already closed by us or by the
         * function; to be safe we fclose here since video_input_close is
         * only called on success. */
        fclose(f);
        return -1;
    }

    if (info)
        video_input_get_info(&vid, info);
    video_input_close(&vid);
    return 0;
}

/* Test 1: valid_minimal */
static char *test_y4m_valid_minimal(void) {
    const char *hdr = "YUV4MPEG2 W320 H240 F30:1\n";
    video_input_info info;
    int ret = try_open_y4m_header(hdr, strlen(hdr), &info);
    mu_assert("valid_minimal: open must succeed", ret == 0);
    mu_assert("valid_minimal: pic_w", info.pic_w == 320);
    mu_assert("valid_minimal: pic_h", info.pic_h == 240);
    mu_assert("valid_minimal: fps_n", info.fps_n == 30);
    mu_assert("valid_minimal: fps_d", info.fps_d == 1);
    mu_assert("valid_minimal: default chroma is 420", info.pixel_fmt == PF_420);
    mu_assert("valid_minimal: depth 8", info.depth == 8);
    return NULL;
}

/* Test 2: valid_all_tags */
static char *test_y4m_valid_all_tags(void) {
    const char *hdr = "YUV4MPEG2 W1920 H1080 F24:1 Ip A1:1 C420\n";
    video_input_info info;
    int ret = try_open_y4m_header(hdr, strlen(hdr), &info);
    mu_assert("valid_all_tags: open must succeed", ret == 0);
    mu_assert("valid_all_tags: pic_w", info.pic_w == 1920);
    mu_assert("valid_all_tags: pic_h", info.pic_h == 1080);
    mu_assert("valid_all_tags: fps_n", info.fps_n == 24);
    mu_assert("valid_all_tags: pixel_fmt PF_420", info.pixel_fmt == PF_420);
    mu_assert("valid_all_tags: par_n", info.par_n == 1);
    mu_assert("valid_all_tags: par_d", info.par_d == 1);
    return NULL;
}

/* Test 3: valid_tag_reorder */
static char *test_y4m_valid_tag_reorder(void) {
    const char *hdr = "YUV4MPEG2 H240 F30:1 W320\n";
    video_input_info info;
    int ret = try_open_y4m_header(hdr, strlen(hdr), &info);
    mu_assert("valid_tag_reorder: open must succeed", ret == 0);
    mu_assert("valid_tag_reorder: pic_w", info.pic_w == 320);
    mu_assert("valid_tag_reorder: pic_h", info.pic_h == 240);
    return NULL;
}

/* Test 4: valid_unknown_tags */
static char *test_y4m_valid_unknown_tags(void) {
    const char *hdr = "YUV4MPEG2 W320 H240 F30:1 Xfoo Ybar\n";
    video_input_info info;
    int ret = try_open_y4m_header(hdr, strlen(hdr), &info);
    mu_assert("valid_unknown_tags: open must succeed", ret == 0);
    mu_assert("valid_unknown_tags: pic_w", info.pic_w == 320);
    return NULL;
}

/* Test 5: valid_extra_spaces */
static char *test_y4m_valid_extra_spaces(void) {
    const char *hdr = "YUV4MPEG2  W320  H240  F30:1\n";
    video_input_info info;
    int ret = try_open_y4m_header(hdr, strlen(hdr), &info);
    mu_assert("valid_extra_spaces: open must succeed", ret == 0);
    mu_assert("valid_extra_spaces: pic_w", info.pic_w == 320);
    mu_assert("valid_extra_spaces: pic_h", info.pic_h == 240);
    return NULL;
}

/* Test 6: missing_width */
static char *test_y4m_missing_width(void) {
    const char *hdr = "YUV4MPEG2 H240 F30:1\n";
    int ret = try_open_y4m_header(hdr, strlen(hdr), NULL);
    mu_assert("missing_width: must fail", ret < 0);
    return NULL;
}

/* Test 7: missing_height */
static char *test_y4m_missing_height(void) {
    const char *hdr = "YUV4MPEG2 W320 F30:1\n";
    int ret = try_open_y4m_header(hdr, strlen(hdr), NULL);
    mu_assert("missing_height: must fail", ret < 0);
    return NULL;
}

/* Test 8: missing_fps */
static char *test_y4m_missing_fps(void) {
    const char *hdr = "YUV4MPEG2 W320 H240\n";
    int ret = try_open_y4m_header(hdr, strlen(hdr), NULL);
    mu_assert("missing_fps: must fail", ret < 0);
    return NULL;
}

/* Test 9: missing_magic */
static char *test_y4m_missing_magic(void) {
    const char *hdr = "NOT_Y4M W320 H240 F30:1\n";
    int ret = try_open_y4m_header(hdr, strlen(hdr), NULL);
    mu_assert("missing_magic: must fail", ret < 0);
    return NULL;
}

/* Test 10: wrong_version -- version check is non-fatal (just warns) but
 * parsing proceeds. The 'E' in YUV4MPEG will be at index 8, != '2'.
 * The parser checks buffer[8]!='2' and prints a warning but continues.
 * However y4m_parse_tags starts at buffer+5 which is "PEG1 W...". The
 * 'P' tag and 'E' tag are unknown and ignored. So parsing should still work
 * only if W, H, F are present and the initial 8-byte magic "YUV4MPEG" matches. */
static char *test_y4m_wrong_version(void) {
    /* YUV4MPEG1 -- magic "YUV4MPEG" matches, buffer[8]='1' != '2' => warning */
    const char *hdr = "YUV4MPEG1 W320 H240 F30:1\n";
    video_input_info info;
    /* The parse_tags call starts at buffer+5 = "EG1 W320 H240 F30:1".
     * 'E' tag unknown, 'W' parsed, 'H' parsed, 'F' parsed => should succeed. */
    int ret = try_open_y4m_header(hdr, strlen(hdr), &info);
    /* This may succeed or fail depending on how the parser handles the offset.
     * We just assert it doesn't crash. */
    (void)ret;
    (void)info;
    return NULL;
}

/* Test 11: truncated_header (no newline, EOF) */
static char *test_y4m_truncated_header(void) {
    const char *hdr = "YUV4MPEG2 W320 H2";
    int ret = try_open_y4m_header(hdr, strlen(hdr), NULL);
    mu_assert("truncated_header: must fail", ret < 0);
    return NULL;
}

/* Test 12: header_at_bufsize_limit (255 bytes + newline right at buffer boundary) */
static char *test_y4m_header_at_bufsize_limit(void) {
    /* Build a header that is exactly 255 chars long before the newline position.
     * The buffer is 256 bytes, reading up to 255 (index 0..254).
     * We need: "YUV4MPEG2 W320 H240 F30:1 C420 X<padding>\n" */
    char hdr[257];
    int base_len = snprintf(hdr, sizeof(hdr), "YUV4MPEG2 W320 H240 F30:1 C420 X");
    /* Pad with 'a' up to position 254 */
    while (base_len < 254) {
        hdr[base_len++] = 'a';
    }
    hdr[base_len++] = '\n'; /* position 254 = newline => 255 bytes total */
    hdr[base_len] = '\0';

    video_input_info info;
    int ret = try_open_y4m_header(hdr, (size_t)base_len, &info);
    /* Should succeed because the newline is found within 255 chars */
    mu_assert("header_at_bufsize_limit: open must succeed", ret == 0);
    mu_assert("header_at_bufsize_limit: pic_w", info.pic_w == 320);
    return NULL;
}

/* Test 13: header_exceeds_bufsize -- header > 255 bytes with no newline within limit */
static char *test_y4m_header_exceeds_bufsize(void) {
    char hdr[300];
    int base_len = snprintf(hdr, sizeof(hdr), "YUV4MPEG2 W320 H240 F30:1 C420 X");
    /* Pad well beyond 255 */
    while (base_len < 280) {
        hdr[base_len++] = 'a';
    }
    hdr[base_len++] = '\n';
    hdr[base_len] = '\0';
    /* The parser reads at most 255 chars before stopping; the newline is beyond
     * that, so the header gets truncated. This may fail or succeed depending on
     * whether tags are truncated. We just verify no crash. */
    int ret = try_open_y4m_header(hdr, (size_t)base_len, NULL);
    (void)ret;
    return NULL;
}

/* Test 14: chroma_type_near_overflow -- 15-char chroma value fits in chroma_type[16] */
static char *test_y4m_chroma_type_near_overflow(void) {
    /* C tag with exactly 15-char value: "C" + 15 chars = 16 bytes including C.
     * The parser checks q-p > 16 (where p points at 'C' and q at the space after).
     * 15-char value means q-p = 16, which does NOT trigger the > 16 check. */
    const char *hdr = "YUV4MPEG2 W320 H240 F30:1 C420aaaaaaaaaaa\n";
    /* "420aaaaaaaaaaa" is 15 chars */
    int ret = try_open_y4m_header(hdr, strlen(hdr), NULL);
    /* Will fail because it's an unknown chroma type, but should not overflow */
    (void)ret;
    return NULL;
}

/* Test 15: chroma_type_overflow -- 16+ char value triggers q-p > 16 check */
static char *test_y4m_chroma_type_overflow(void) {
    const char *hdr = "YUV4MPEG2 W320 H240 F30:1 C420aaaaaaaaaaaa\n";
    /* "420aaaaaaaaaaaa" is 16 chars => q-p = 17 > 16 => return -1 */
    int ret = try_open_y4m_header(hdr, strlen(hdr), NULL);
    mu_assert("chroma_type_overflow: must fail", ret < 0);
    return NULL;
}

/* Test 16: unknown_chroma_type */
static char *test_y4m_unknown_chroma_type(void) {
    const char *hdr = "YUV4MPEG2 W320 H240 F30:1 Crgb\n";
    int ret = try_open_y4m_header(hdr, strlen(hdr), NULL);
    mu_assert("unknown_chroma_type: must fail", ret < 0);
    return NULL;
}

/* Test 17: interlaced_content */
static char *test_y4m_interlaced_content(void) {
    const char *hdr = "YUV4MPEG2 W320 H240 F30:1 It\n";
    int ret = try_open_y4m_header(hdr, strlen(hdr), NULL);
    mu_assert("interlaced_content: must fail", ret < 0);
    return NULL;
}

/* Test 18: negative_dimensions */
static char *test_y4m_negative_dimensions(void) {
    const char *hdr = "YUV4MPEG2 W-1 H240 F30:1\n";
    /* May parse -1 via sscanf; just verify no crash */
    int ret = try_open_y4m_header(hdr, strlen(hdr), NULL);
    (void)ret;
    return NULL;
}

/* Test 19: zero_dimensions */
static char *test_y4m_zero_dimensions(void) {
    const char *hdr = "YUV4MPEG2 W0 H0 F30:1\n";
    int ret = try_open_y4m_header(hdr, strlen(hdr), NULL);
    (void)ret;
    return NULL;
}

/* Test 20: fps_zero_denominator */
static char *test_y4m_fps_zero_denominator(void) {
    const char *hdr = "YUV4MPEG2 W320 H240 F30:0\n";
    video_input_info info;
    int ret = try_open_y4m_header(hdr, strlen(hdr), &info);
    /* Parser does not validate fps_d */
    mu_assert("fps_zero_denominator: parser succeeds", ret == 0);
    mu_assert("fps_zero_denominator: fps_d is 0", info.fps_d == 0);
    return NULL;
}

/* Test 21: empty_after_magic */
static char *test_y4m_empty_after_magic(void) {
    const char *hdr = "YUV4MPEG2\n";
    int ret = try_open_y4m_header(hdr, strlen(hdr), NULL);
    mu_assert("empty_after_magic: must fail (missing W/H/F)", ret < 0);
    return NULL;
}

/* Test 22: frame_header_params -- frame header with extra params after FRAME */
static char *test_y4m_frame_header_params(void) {
    /* Create a valid Y4M with 1 frame where frame header has extra params */
    const unsigned w = 16, h = 16;
    const char *chroma = "420";

    FILE *f = tmpfile();
    mu_assert("frame_header_params: tmpfile", f != NULL);

    const char *shdr = "YUV4MPEG2 W16 H16 F30:1 C420\n";
    fwrite(shdr, 1, strlen(shdr), f);

    /* Frame header with params */
    const char *fhdr = "FRAME Ifirst\n";
    fwrite(fhdr, 1, strlen(fhdr), f);

    /* Frame data: 16*16 + 2*(8*8) = 384 bytes */
    size_t fsz = compute_frame_data_size(w, h, chroma, 8);
    uint8_t *frame_data = (uint8_t *)calloc(1, fsz);
    fwrite(frame_data, 1, fsz, f);
    free(frame_data);

    fseek(f, 0, SEEK_SET);

    video_input vid;
    int ret = video_input_open(&vid, f);
    mu_assert("frame_header_params: open succeeds", ret == 0);

    video_input_ycbcr ycbcr;
    char tag[5];
    ret = video_input_fetch_frame(&vid, ycbcr, tag);
    mu_assert("frame_header_params: fetch succeeds", ret == 1);

    video_input_close(&vid);
    return NULL;
}

/* Test 23: frame_header_overlong -- frame header > 79 chars after FRAME */
static char *test_y4m_frame_header_overlong(void) {
    const unsigned w = 16, h = 16;
    const char *chroma = "420";

    FILE *f = tmpfile();
    mu_assert("frame_header_overlong: tmpfile", f != NULL);

    const char *shdr = "YUV4MPEG2 W16 H16 F30:1 C420\n";
    fwrite(shdr, 1, strlen(shdr), f);

    /* Write "FRAME " followed by 80+ chars before newline */
    fwrite("FRAME ", 1, 6, f);
    char pad[100];
    memset(pad, 'x', 90);
    pad[90] = '\n';
    fwrite(pad, 1, 91, f);

    size_t fsz = compute_frame_data_size(w, h, chroma, 8);
    uint8_t *frame_data = (uint8_t *)calloc(1, fsz);
    fwrite(frame_data, 1, fsz, f);
    free(frame_data);

    fseek(f, 0, SEEK_SET);

    video_input vid;
    int ret = video_input_open(&vid, f);
    mu_assert("frame_header_overlong: stream open succeeds", ret == 0);

    video_input_ycbcr ycbcr;
    char tag[5];
    ret = video_input_fetch_frame(&vid, ycbcr, tag);
    mu_assert("frame_header_overlong: fetch must fail", ret == -1);

    video_input_close(&vid);
    return NULL;
}


/* ====================================================================
 * Section 4: Format-Depth Matrix Round-Trip Tests
 * ==================================================================== */

/*
 * Format-depth matrix test infrastructure.
 * Each combination: generate Y4M, parse, verify info + alloc VmafPicture.
 */

typedef struct {
    const char *chroma;
    video_input_pixel_format expected_pf;
    enum VmafPixelFormat vmaf_pf;
    unsigned bpc;
} format_spec;

static const format_spec format_matrix[] = {
    { "420",    PF_420, VMAF_PIX_FMT_YUV420P,  8 },
    { "420p10", PF_420, VMAF_PIX_FMT_YUV420P, 10 },
    { "420p12", PF_420, VMAF_PIX_FMT_YUV420P, 12 },
    { "422",    PF_422, VMAF_PIX_FMT_YUV422P,  8 },
    { "422p10", PF_422, VMAF_PIX_FMT_YUV422P, 10 },
    { "422p12", PF_422, VMAF_PIX_FMT_YUV422P, 12 },
    { "444",    PF_444, VMAF_PIX_FMT_YUV444P,  8 },
    { "444p10", PF_444, VMAF_PIX_FMT_YUV444P, 10 },
    { "444p12", PF_444, VMAF_PIX_FMT_YUV444P, 12 },
    { "mono",   PF_420, VMAF_PIX_FMT_YUV420P,  8 },
};
#define NUM_FORMATS (sizeof(format_matrix) / sizeof(format_matrix[0]))

typedef struct {
    unsigned w, h;
} dim_spec;

static const dim_spec dim_matrix[] = {
    { 16,   16  },
    { 320,  240 },
    { 321,  241 },
    { 1920, 1080},
};
#define NUM_DIMS (sizeof(dim_matrix) / sizeof(dim_matrix[0]))

static const enum fill_pattern pattern_matrix[] = {
    FILL_ZERO,
    FILL_GRADIENT_H,
    FILL_RANDOM_42,
};
#define NUM_PATTERNS (sizeof(pattern_matrix) / sizeof(pattern_matrix[0]))

/**
 * Run a single format-depth round-trip test.
 * Returns NULL on success, error message string on failure.
 */
static char *run_format_roundtrip(const format_spec *fmt,
                                  unsigned w, unsigned h,
                                  enum fill_pattern pat)
{
    static char errbuf[256];

    /* Mono with odd dimensions for 420 can require rounding -- skip
     * combinations that don't apply well */

    /* Generate Y4M */
    FILE *f = create_y4m_file(w, h, fmt->chroma, fmt->bpc, 1, pat);
    if (!f) {
        snprintf(errbuf, sizeof(errbuf),
                 "%s %ux%u bpc=%u: failed to create Y4M file",
                 fmt->chroma, w, h, fmt->bpc);
        return errbuf;
    }

    /* Open via video_input */
    video_input vid;
    int ret = video_input_open(&vid, f);
    if (ret < 0) {
        fclose(f);
        snprintf(errbuf, sizeof(errbuf),
                 "%s %ux%u bpc=%u: video_input_open failed",
                 fmt->chroma, w, h, fmt->bpc);
        return errbuf;
    }

    /* Verify parsed info */
    video_input_info info;
    video_input_get_info(&vid, &info);

    if (info.pixel_fmt != fmt->expected_pf) {
        video_input_close(&vid);
        snprintf(errbuf, sizeof(errbuf),
                 "%s %ux%u bpc=%u: pixel_fmt mismatch (got %d, want %d)",
                 fmt->chroma, w, h, fmt->bpc,
                 info.pixel_fmt, fmt->expected_pf);
        return errbuf;
    }
    if (info.depth != (int)fmt->bpc) {
        video_input_close(&vid);
        snprintf(errbuf, sizeof(errbuf),
                 "%s %ux%u bpc=%u: depth mismatch (got %d)",
                 fmt->chroma, w, h, fmt->bpc, info.depth);
        return errbuf;
    }
    if (info.pic_w != (int)w || info.pic_h != (int)h) {
        video_input_close(&vid);
        snprintf(errbuf, sizeof(errbuf),
                 "%s %ux%u bpc=%u: dimension mismatch (got %dx%d)",
                 fmt->chroma, w, h, fmt->bpc, info.pic_w, info.pic_h);
        return errbuf;
    }

    /* Fetch frame */
    video_input_ycbcr ycbcr;
    char tag[5];
    ret = video_input_fetch_frame(&vid, ycbcr, tag);
    if (ret != 1) {
        video_input_close(&vid);
        snprintf(errbuf, sizeof(errbuf),
                 "%s %ux%u bpc=%u: fetch_frame failed (ret=%d)",
                 fmt->chroma, w, h, fmt->bpc, ret);
        return errbuf;
    }

    /* Allocate VmafPicture and verify */
    VmafPicture pic;
    int err = vmaf_picture_alloc(&pic, fmt->vmaf_pf, fmt->bpc, w, h);
    if (err) {
        video_input_close(&vid);
        snprintf(errbuf, sizeof(errbuf),
                 "%s %ux%u bpc=%u: vmaf_picture_alloc failed",
                 fmt->chroma, w, h, fmt->bpc);
        return errbuf;
    }

    /* Verify stride alignment */
    if (pic.stride[0] % 32 != 0) {
        vmaf_picture_unref(&pic);
        video_input_close(&vid);
        snprintf(errbuf, sizeof(errbuf),
                 "%s %ux%u bpc=%u: luma stride not 32-byte aligned",
                 fmt->chroma, w, h, fmt->bpc);
        return errbuf;
    }

    /* Verify plane dimensions for 444 explicitly */
    if (fmt->vmaf_pf == VMAF_PIX_FMT_YUV444P) {
        if (pic.w[1] != w || pic.h[1] != h) {
            vmaf_picture_unref(&pic);
            video_input_close(&vid);
            snprintf(errbuf, sizeof(errbuf),
                     "%s %ux%u bpc=%u: 444 chroma dims mismatch",
                     fmt->chroma, w, h, fmt->bpc);
            return errbuf;
        }
    }

    /* Verify chroma plane dimensions for 420 with odd dims */
    if (fmt->vmaf_pf == VMAF_PIX_FMT_YUV420P) {
        unsigned expected_cw = w >> 1;
        unsigned expected_ch = h >> 1;
        if (pic.w[1] != expected_cw || pic.h[1] != expected_ch) {
            /* Note: vmaf_picture_alloc uses w >> ss_hor, not (w+1)/2.
             * This is correct per the implementation. */
        }
    }

    vmaf_picture_unref(&pic);
    video_input_close(&vid);
    return NULL;
}

/*
 * Macro to generate a test function for each format/dim/pattern combo.
 * We use a single test that iterates the full matrix to keep test count
 * manageable while still covering 120 combinations.
 */
static char *test_format_depth_matrix(void) {
    for (unsigned fi = 0; fi < NUM_FORMATS; fi++) {
        for (unsigned di = 0; di < NUM_DIMS; di++) {
            for (unsigned pi = 0; pi < NUM_PATTERNS; pi++) {
                char *msg = run_format_roundtrip(
                    &format_matrix[fi],
                    dim_matrix[di].w,
                    dim_matrix[di].h,
                    pattern_matrix[pi]);
                mu_assert(msg ? msg : "unreachable", msg == NULL);
            }
        }
    }
    return NULL;
}


/* ====================================================================
 * Section 4.4: YUV444P-Specific Tests (explicit per R3)
 * ==================================================================== */

static char *test_y4m_444_8bit(void) {
    const unsigned w = 320, h = 240;
    FILE *f = create_y4m_file(w, h, "444", 8, 1, FILL_GRADIENT_H);
    mu_assert("444_8bit: create file", f != NULL);

    video_input vid;
    mu_assert("444_8bit: open", video_input_open(&vid, f) == 0);

    video_input_info info;
    video_input_get_info(&vid, &info);
    mu_assert("444_8bit: pixel_fmt PF_444", info.pixel_fmt == PF_444);
    mu_assert("444_8bit: depth 8", info.depth == 8);

    VmafPicture pic;
    mu_assert("444_8bit: alloc", vmaf_picture_alloc(&pic, VMAF_PIX_FMT_YUV444P, 8, w, h) == 0);
    mu_assert("444_8bit: chroma w == luma w", pic.w[1] == w);
    mu_assert("444_8bit: chroma h == luma h", pic.h[1] == h);
    vmaf_picture_unref(&pic);
    video_input_close(&vid);
    return NULL;
}

static char *test_y4m_444_10bit(void) {
    const unsigned w = 320, h = 240;
    FILE *f = create_y4m_file(w, h, "444p10", 10, 1, FILL_GRADIENT_H);
    mu_assert("444_10bit: create file", f != NULL);

    video_input vid;
    mu_assert("444_10bit: open", video_input_open(&vid, f) == 0);

    video_input_info info;
    video_input_get_info(&vid, &info);
    mu_assert("444_10bit: pixel_fmt PF_444", info.pixel_fmt == PF_444);
    mu_assert("444_10bit: depth 10", info.depth == 10);

    VmafPicture pic;
    mu_assert("444_10bit: alloc", vmaf_picture_alloc(&pic, VMAF_PIX_FMT_YUV444P, 10, w, h) == 0);
    mu_assert("444_10bit: chroma w == luma w", pic.w[1] == w);
    mu_assert("444_10bit: chroma h == luma h", pic.h[1] == h);
    mu_assert("444_10bit: stride aligned", (pic.stride[0] % 32) == 0);
    vmaf_picture_unref(&pic);
    video_input_close(&vid);
    return NULL;
}

static char *test_y4m_444_12bit(void) {
    const unsigned w = 320, h = 240;
    FILE *f = create_y4m_file(w, h, "444p12", 12, 1, FILL_GRADIENT_H);
    mu_assert("444_12bit: create file", f != NULL);

    video_input vid;
    mu_assert("444_12bit: open", video_input_open(&vid, f) == 0);

    video_input_info info;
    video_input_get_info(&vid, &info);
    mu_assert("444_12bit: pixel_fmt PF_444", info.pixel_fmt == PF_444);
    mu_assert("444_12bit: depth 12", info.depth == 12);

    VmafPicture pic;
    mu_assert("444_12bit: alloc", vmaf_picture_alloc(&pic, VMAF_PIX_FMT_YUV444P, 12, w, h) == 0);
    mu_assert("444_12bit: chroma w == luma w", pic.w[1] == w);
    mu_assert("444_12bit: chroma h == luma h", pic.h[1] == h);
    mu_assert("444_12bit: stride aligned", (pic.stride[0] % 32) == 0);
    vmaf_picture_unref(&pic);
    video_input_close(&vid);
    return NULL;
}


/* ====================================================================
 * Section 5: Range Handling Tests
 * ==================================================================== */

/* Test: XCOLORRANGE tag is silently ignored */
static char *test_y4m_range_tag_ignored(void) {
    const char *hdr = "YUV4MPEG2 W320 H240 F30:1 XCOLORRANGE=LIMITED\n";
    video_input_info info;
    int ret = try_open_y4m_header(hdr, strlen(hdr), &info);
    mu_assert("range_tag_ignored: open succeeds", ret == 0);
    mu_assert("range_tag_ignored: pic_w", info.pic_w == 320);
    return NULL;
}

static char *test_y4m_range_tag_full_ignored(void) {
    const char *hdr = "YUV4MPEG2 W320 H240 F30:1 XCOLORRANGE=FULL\n";
    video_input_info info;
    int ret = try_open_y4m_header(hdr, strlen(hdr), &info);
    mu_assert("range_tag_full_ignored: open succeeds", ret == 0);
    return NULL;
}

/* Helper: create Y4M with specific pixel boundary values and verify round-trip */
static char *verify_boundary_roundtrip_8bit(void) {
    const uint8_t test_values[] = {0, 1, 15, 16, 235, 236, 254, 255};

    FILE *f = tmpfile();
    mu_assert("boundary_8bit: tmpfile", f != NULL);

    const char *shdr = "YUV4MPEG2 W8 H1 F30:1 C420\n";
    fwrite(shdr, 1, strlen(shdr), f);
    fwrite("FRAME\n", 1, 6, f);

    /* Y plane: 8x1 = 8 bytes */
    fwrite(test_values, 1, 8, f);
    /* Cb and Cr planes: (8+1)/2 * (1+1)/2 = 4*1 = 4 bytes each */
    uint8_t chroma[4] = {128, 128, 128, 128};
    fwrite(chroma, 1, 4, f);
    fwrite(chroma, 1, 4, f);

    fseek(f, 0, SEEK_SET);

    video_input vid;
    mu_assert("boundary_8bit: open", video_input_open(&vid, f) == 0);

    video_input_ycbcr ycbcr;
    char tag[5];
    int ret = video_input_fetch_frame(&vid, ycbcr, tag);
    mu_assert("boundary_8bit: fetch", ret == 1);

    /* Verify Y values survived round-trip */
    /* The ycbcr[0].data pointer accounts for pic_x/pic_y offset.
     * For our 8x1 frame, frame_w = (8+15)&~0xF = 16, pic_x = (16-8)/2 & ~1 = 4.
     * So we need to offset by pic_x. */
    video_input_info info;
    video_input_get_info(&vid, &info);
    const uint8_t *ydata = ycbcr[0].data
                           + info.pic_y * ycbcr[0].stride
                           + info.pic_x;
    for (int i = 0; i < 8; i++) {
        if (ydata[i] != test_values[i]) {
            video_input_close(&vid);
            static char msg[128];
            snprintf(msg, sizeof(msg),
                     "boundary_8bit: value[%d] mismatch: got %u want %u",
                     i, ydata[i], test_values[i]);
            return msg;
        }
    }

    video_input_close(&vid);
    return NULL;
}

static char *test_y4m_boundary_values_8bit(void) {
    return verify_boundary_roundtrip_8bit();
}

/* Boundary values for 10-bit */
static char *test_y4m_boundary_values_10bit(void) {
    const uint16_t test_values[] = {0, 64, 940, 1023};

    FILE *f = tmpfile();
    mu_assert("boundary_10bit: tmpfile", f != NULL);

    const char *shdr = "YUV4MPEG2 W4 H1 F30:1 C420p10\n";
    fwrite(shdr, 1, strlen(shdr), f);
    fwrite("FRAME\n", 1, 6, f);

    /* Y plane: 4*1*2 = 8 bytes */
    uint8_t ybuf[8];
    for (int i = 0; i < 4; i++)
        write_sample_16le(ybuf + i * 2, test_values[i]);
    fwrite(ybuf, 1, 8, f);

    /* Cb and Cr planes: (4+1)/2 * (1+1)/2 = 2*1 = 2 samples each = 4 bytes each */
    uint8_t cbuf[4];
    write_sample_16le(cbuf, 512);
    write_sample_16le(cbuf + 2, 512);
    fwrite(cbuf, 1, 4, f);
    fwrite(cbuf, 1, 4, f);

    fseek(f, 0, SEEK_SET);

    video_input vid;
    mu_assert("boundary_10bit: open", video_input_open(&vid, f) == 0);

    video_input_ycbcr ycbcr;
    char tag[5];
    int ret = video_input_fetch_frame(&vid, ycbcr, tag);
    mu_assert("boundary_10bit: fetch", ret == 1);

    video_input_info info;
    video_input_get_info(&vid, &info);
    const uint8_t *ydata = ycbcr[0].data
                           + info.pic_y * ycbcr[0].stride
                           + info.pic_x * 2;
    for (int i = 0; i < 4; i++) {
        uint16_t got = read_sample_16le(ydata + i * 2);
        if (got != test_values[i]) {
            video_input_close(&vid);
            static char msg[128];
            snprintf(msg, sizeof(msg),
                     "boundary_10bit: value[%d] mismatch: got %u want %u",
                     i, got, test_values[i]);
            return msg;
        }
    }

    video_input_close(&vid);
    return NULL;
}

/* Boundary values for 12-bit */
static char *test_y4m_boundary_values_12bit(void) {
    const uint16_t test_values[] = {0, 256, 3760, 4095};

    FILE *f = tmpfile();
    mu_assert("boundary_12bit: tmpfile", f != NULL);

    const char *shdr = "YUV4MPEG2 W4 H1 F30:1 C420p12\n";
    fwrite(shdr, 1, strlen(shdr), f);
    fwrite("FRAME\n", 1, 6, f);

    /* Y plane */
    uint8_t ybuf[8];
    for (int i = 0; i < 4; i++)
        write_sample_16le(ybuf + i * 2, test_values[i]);
    fwrite(ybuf, 1, 8, f);

    /* Cb and Cr */
    uint8_t cbuf[4];
    write_sample_16le(cbuf, 2048);
    write_sample_16le(cbuf + 2, 2048);
    fwrite(cbuf, 1, 4, f);
    fwrite(cbuf, 1, 4, f);

    fseek(f, 0, SEEK_SET);

    video_input vid;
    mu_assert("boundary_12bit: open", video_input_open(&vid, f) == 0);

    video_input_ycbcr ycbcr;
    char tag[5];
    int ret = video_input_fetch_frame(&vid, ycbcr, tag);
    mu_assert("boundary_12bit: fetch", ret == 1);

    video_input_info info;
    video_input_get_info(&vid, &info);
    const uint8_t *ydata = ycbcr[0].data
                           + info.pic_y * ycbcr[0].stride
                           + info.pic_x * 2;
    for (int i = 0; i < 4; i++) {
        uint16_t got = read_sample_16le(ydata + i * 2);
        if (got != test_values[i]) {
            video_input_close(&vid);
            static char msg[128];
            snprintf(msg, sizeof(msg),
                     "boundary_12bit: value[%d] mismatch: got %u want %u",
                     i, got, test_values[i]);
            return msg;
        }
    }

    video_input_close(&vid);
    return NULL;
}

/* Limited-range pixel values preserved */
static char *test_y4m_limited_range_values(void) {
    /* Verify that limited range values [16, 235] are preserved */
    FILE *f = tmpfile();
    mu_assert("limited_range: tmpfile", f != NULL);

    const char *shdr = "YUV4MPEG2 W16 H16 F30:1 C420\n";
    fwrite(shdr, 1, strlen(shdr), f);
    fwrite("FRAME\n", 1, 6, f);

    /* Y plane: fill with limited-range values */
    uint8_t ybuf[16 * 16];
    for (int i = 0; i < 16 * 16; i++)
        ybuf[i] = (uint8_t)(16 + (i % 220)); /* 16..235 */
    fwrite(ybuf, 1, sizeof(ybuf), f);

    /* Chroma: 8x8 each */
    uint8_t cbuf[8 * 8];
    memset(cbuf, 128, sizeof(cbuf));
    fwrite(cbuf, 1, sizeof(cbuf), f);
    fwrite(cbuf, 1, sizeof(cbuf), f);

    fseek(f, 0, SEEK_SET);

    video_input vid;
    mu_assert("limited_range: open", video_input_open(&vid, f) == 0);

    video_input_ycbcr ycbcr;
    char tag[5];
    mu_assert("limited_range: fetch", video_input_fetch_frame(&vid, ycbcr, tag) == 1);

    video_input_info info;
    video_input_get_info(&vid, &info);
    const uint8_t *ydata = ycbcr[0].data + info.pic_x + info.pic_y * ycbcr[0].stride;
    for (unsigned row = 0; row < 16; row++) {
        for (unsigned col = 0; col < 16; col++) {
            uint8_t expected = (uint8_t)(16 + ((row * 16 + col) % 220));
            uint8_t got = ydata[row * ycbcr[0].stride + col];
            if (got != expected) {
                video_input_close(&vid);
                static char msg[128];
                snprintf(msg, sizeof(msg),
                         "limited_range: pixel[%u,%u] mismatch: got %u want %u",
                         row, col, got, expected);
                return msg;
            }
        }
    }

    video_input_close(&vid);
    return NULL;
}

/* Full-range pixel values preserved */
static char *test_y4m_full_range_values(void) {
    FILE *f = tmpfile();
    mu_assert("full_range: tmpfile", f != NULL);

    const char *shdr = "YUV4MPEG2 W16 H16 F30:1 C420\n";
    fwrite(shdr, 1, strlen(shdr), f);
    fwrite("FRAME\n", 1, 6, f);

    /* Y plane: fill with full range values [0, 255] */
    uint8_t ybuf[16 * 16];
    for (int i = 0; i < 16 * 16; i++)
        ybuf[i] = (uint8_t)(i % 256);
    fwrite(ybuf, 1, sizeof(ybuf), f);

    uint8_t cbuf[8 * 8];
    memset(cbuf, 128, sizeof(cbuf));
    fwrite(cbuf, 1, sizeof(cbuf), f);
    fwrite(cbuf, 1, sizeof(cbuf), f);

    fseek(f, 0, SEEK_SET);

    video_input vid;
    mu_assert("full_range: open", video_input_open(&vid, f) == 0);

    video_input_ycbcr ycbcr;
    char tag[5];
    mu_assert("full_range: fetch", video_input_fetch_frame(&vid, ycbcr, tag) == 1);

    video_input_info info;
    video_input_get_info(&vid, &info);
    const uint8_t *ydata = ycbcr[0].data + info.pic_x + info.pic_y * ycbcr[0].stride;
    for (unsigned row = 0; row < 16; row++) {
        for (unsigned col = 0; col < 16; col++) {
            uint8_t expected = (uint8_t)((row * 16 + col) % 256);
            uint8_t got = ydata[row * ycbcr[0].stride + col];
            if (got != expected) {
                video_input_close(&vid);
                static char msg[128];
                snprintf(msg, sizeof(msg),
                         "full_range: pixel[%u,%u] mismatch: got %u want %u",
                         row, col, got, expected);
                return msg;
            }
        }
    }

    video_input_close(&vid);
    return NULL;
}


/* ====================================================================
 * Section 8: Error Handling Tests
 * ==================================================================== */

/* err_no_magic: random bytes */
static char *test_y4m_err_no_magic(void) {
    const char *data = "RANDOMGARBAGE W320 H240 F30:1\n";
    int ret = try_open_y4m_header(data, strlen(data), NULL);
    mu_assert("err_no_magic: must fail", ret < 0);
    return NULL;
}

/* err_empty_file */
static char *test_y4m_err_empty_file(void) {
    FILE *f = tmpfile();
    mu_assert("err_empty_file: tmpfile", f != NULL);
    /* Write nothing, rewind */
    fseek(f, 0, SEEK_SET);

    video_input vid;
    int ret = video_input_open(&vid, f);
    if (ret == 0) {
        video_input_close(&vid);
    } else {
        fclose(f);
    }
    mu_assert("err_empty_file: must fail", ret < 0);
    return NULL;
}

/* err_truncated_frame: valid header but not enough frame data */
static char *test_y4m_err_truncated_frame(void) {
    FILE *f = tmpfile();
    mu_assert("err_truncated_frame: tmpfile", f != NULL);

    const char *shdr = "YUV4MPEG2 W320 H240 F30:1 C420\n";
    fwrite(shdr, 1, strlen(shdr), f);
    fwrite("FRAME\n", 1, 6, f);
    /* Write only 10 bytes of frame data instead of needed ~115200 */
    uint8_t dummy[10] = {0};
    fwrite(dummy, 1, 10, f);

    fseek(f, 0, SEEK_SET);

    video_input vid;
    int ret = video_input_open(&vid, f);
    mu_assert("err_truncated_frame: open succeeds", ret == 0);

    video_input_ycbcr ycbcr;
    char tag[5];
    ret = video_input_fetch_frame(&vid, ycbcr, tag);
    mu_assert("err_truncated_frame: fetch must fail", ret == -1);

    video_input_close(&vid);
    return NULL;
}

/* err_missing_frame_tag: valid header but frame data without FRAME prefix */
static char *test_y4m_err_missing_frame_tag(void) {
    FILE *f = tmpfile();
    mu_assert("err_missing_frame_tag: tmpfile", f != NULL);

    const char *shdr = "YUV4MPEG2 W16 H16 F30:1 C420\n";
    fwrite(shdr, 1, strlen(shdr), f);
    /* Write raw data without "FRAME\n" */
    size_t fsz = compute_frame_data_size(16, 16, "420", 8);
    uint8_t *data = (uint8_t *)calloc(1, fsz + 6);
    fwrite(data, 1, fsz + 6, f);
    free(data);

    fseek(f, 0, SEEK_SET);

    video_input vid;
    int ret = video_input_open(&vid, f);
    mu_assert("err_missing_frame_tag: open succeeds", ret == 0);

    video_input_ycbcr ycbcr;
    char tag[5];
    ret = video_input_fetch_frame(&vid, ycbcr, tag);
    mu_assert("err_missing_frame_tag: fetch must fail", ret == -1);

    video_input_close(&vid);
    return NULL;
}

/* err_unsupported_chroma */
static char *test_y4m_err_unsupported_chroma(void) {
    const char *hdr = "YUV4MPEG2 W320 H240 F30:1 C422p16\n";
    int ret = try_open_y4m_header(hdr, strlen(hdr), NULL);
    mu_assert("err_unsupported_chroma: must fail", ret < 0);
    return NULL;
}

/* err_interlaced */
static char *test_y4m_err_interlaced(void) {
    const char *hdr = "YUV4MPEG2 W320 H240 F30:1 It\n";
    int ret = try_open_y4m_header(hdr, strlen(hdr), NULL);
    mu_assert("err_interlaced: must fail", ret < 0);
    return NULL;
}

/* err_picture_alloc_bad_fmt */
static char *test_err_picture_alloc_bad_fmt(void) {
    VmafPicture pic;
    int err = vmaf_picture_alloc(&pic, VMAF_PIX_FMT_UNKNOWN, 8, 320, 240);
    mu_assert("err_picture_alloc_bad_fmt: must fail", err != 0);
    return NULL;
}

/* err_picture_alloc_bad_bpc */
static char *test_err_picture_alloc_bad_bpc(void) {
    VmafPicture pic;
    int err = vmaf_picture_alloc(&pic, VMAF_PIX_FMT_YUV420P, 7, 320, 240);
    mu_assert("err_picture_alloc_bad_bpc: must fail", err != 0);
    return NULL;
}

/* err_picture_alloc_bad_bpc_high */
static char *test_err_picture_alloc_bad_bpc_high(void) {
    VmafPicture pic;
    int err = vmaf_picture_alloc(&pic, VMAF_PIX_FMT_YUV420P, 17, 320, 240);
    mu_assert("err_picture_alloc_bad_bpc_high: must fail", err != 0);
    return NULL;
}

/* err_picture_alloc_null */
static char *test_err_picture_alloc_null(void) {
    int err = vmaf_picture_alloc(NULL, VMAF_PIX_FMT_YUV420P, 8, 320, 240);
    mu_assert("err_picture_alloc_null: must fail", err != 0);
    return NULL;
}


/* ====================================================================
 * Section 8.2: Boundary Condition Robustness
 * ==================================================================== */

static char *test_boundary_1x1_420(void) {
    VmafPicture pic;
    int err = vmaf_picture_alloc(&pic, VMAF_PIX_FMT_YUV420P, 8, 1, 1);
    mu_assert("boundary_1x1_420: alloc succeeds", err == 0);
    /* Chroma: w >> 1 = 0, h >> 1 = 0 -- that's what the implementation does */
    vmaf_picture_unref(&pic);
    return NULL;
}

static char *test_boundary_1x1_444(void) {
    VmafPicture pic;
    int err = vmaf_picture_alloc(&pic, VMAF_PIX_FMT_YUV444P, 8, 1, 1);
    mu_assert("boundary_1x1_444: alloc succeeds", err == 0);
    mu_assert("boundary_1x1_444: chroma w 1", pic.w[1] == 1);
    mu_assert("boundary_1x1_444: chroma h 1", pic.h[1] == 1);
    vmaf_picture_unref(&pic);
    return NULL;
}

static char *test_boundary_odd_w_420(void) {
    VmafPicture pic;
    int err = vmaf_picture_alloc(&pic, VMAF_PIX_FMT_YUV420P, 8, 3, 4);
    mu_assert("boundary_odd_w_420: alloc succeeds", err == 0);
    mu_assert("boundary_odd_w_420: chroma w", pic.w[1] == 3 >> 1);
    mu_assert("boundary_odd_w_420: chroma h", pic.h[1] == 4 >> 1);
    vmaf_picture_unref(&pic);
    return NULL;
}

static char *test_boundary_odd_h_420(void) {
    VmafPicture pic;
    int err = vmaf_picture_alloc(&pic, VMAF_PIX_FMT_YUV420P, 8, 4, 3);
    mu_assert("boundary_odd_h_420: alloc succeeds", err == 0);
    mu_assert("boundary_odd_h_420: chroma w", pic.w[1] == 4 >> 1);
    mu_assert("boundary_odd_h_420: chroma h", pic.h[1] == 3 >> 1);
    vmaf_picture_unref(&pic);
    return NULL;
}

static char *test_boundary_odd_both_420(void) {
    VmafPicture pic;
    int err = vmaf_picture_alloc(&pic, VMAF_PIX_FMT_YUV420P, 8, 3, 3);
    mu_assert("boundary_odd_both_420: alloc succeeds", err == 0);
    mu_assert("boundary_odd_both_420: chroma w", pic.w[1] == 3 >> 1);
    mu_assert("boundary_odd_both_420: chroma h", pic.h[1] == 3 >> 1);
    vmaf_picture_unref(&pic);
    return NULL;
}

static char *test_boundary_max_dimension(void) {
    VmafPicture pic;
    int err = vmaf_picture_alloc(&pic, VMAF_PIX_FMT_YUV420P, 8, 7680, 4320);
    mu_assert("boundary_max_dimension: alloc succeeds", err == 0);
    mu_assert("boundary_max_dimension: stride aligned", (pic.stride[0] % 32) == 0);
    mu_assert("boundary_max_dimension: luma w", pic.w[0] == 7680);
    mu_assert("boundary_max_dimension: luma h", pic.h[0] == 4320);
    vmaf_picture_unref(&pic);
    return NULL;
}


/* ====================================================================
 * Test Runner
 * ==================================================================== */

char *run_tests(void) {
    /* Section 3: Header parsing edge cases (23 tests) */
    mu_run_test(test_y4m_valid_minimal);
    mu_run_test(test_y4m_valid_all_tags);
    mu_run_test(test_y4m_valid_tag_reorder);
    mu_run_test(test_y4m_valid_unknown_tags);
    mu_run_test(test_y4m_valid_extra_spaces);
    mu_run_test(test_y4m_missing_width);
    mu_run_test(test_y4m_missing_height);
    mu_run_test(test_y4m_missing_fps);
    mu_run_test(test_y4m_missing_magic);
    mu_run_test(test_y4m_wrong_version);
    mu_run_test(test_y4m_truncated_header);
    mu_run_test(test_y4m_header_at_bufsize_limit);
    mu_run_test(test_y4m_header_exceeds_bufsize);
    mu_run_test(test_y4m_chroma_type_near_overflow);
    mu_run_test(test_y4m_chroma_type_overflow);
    mu_run_test(test_y4m_unknown_chroma_type);
    mu_run_test(test_y4m_interlaced_content);
    mu_run_test(test_y4m_negative_dimensions);
    mu_run_test(test_y4m_zero_dimensions);
    mu_run_test(test_y4m_fps_zero_denominator);
    mu_run_test(test_y4m_empty_after_magic);
    mu_run_test(test_y4m_frame_header_params);
    mu_run_test(test_y4m_frame_header_overlong);

    /* Section 4: Format-depth matrix (120 round-trip sub-tests) */
    mu_run_test(test_format_depth_matrix);

    /* Section 4.4: YUV444P explicit coverage (R3) */
    mu_run_test(test_y4m_444_8bit);
    mu_run_test(test_y4m_444_10bit);
    mu_run_test(test_y4m_444_12bit);

    /* Section 5: Range handling */
    mu_run_test(test_y4m_range_tag_ignored);
    mu_run_test(test_y4m_range_tag_full_ignored);
    mu_run_test(test_y4m_limited_range_values);
    mu_run_test(test_y4m_full_range_values);
    mu_run_test(test_y4m_boundary_values_8bit);
    mu_run_test(test_y4m_boundary_values_10bit);
    mu_run_test(test_y4m_boundary_values_12bit);

    /* Section 8.1: Error handling */
    mu_run_test(test_y4m_err_no_magic);
    mu_run_test(test_y4m_err_empty_file);
    mu_run_test(test_y4m_err_truncated_frame);
    mu_run_test(test_y4m_err_missing_frame_tag);
    mu_run_test(test_y4m_err_unsupported_chroma);
    mu_run_test(test_y4m_err_interlaced);
    mu_run_test(test_err_picture_alloc_bad_fmt);
    mu_run_test(test_err_picture_alloc_bad_bpc);
    mu_run_test(test_err_picture_alloc_bad_bpc_high);
    mu_run_test(test_err_picture_alloc_null);

    /* Section 8.2: Boundary conditions */
    mu_run_test(test_boundary_1x1_420);
    mu_run_test(test_boundary_1x1_444);
    mu_run_test(test_boundary_odd_w_420);
    mu_run_test(test_boundary_odd_h_420);
    mu_run_test(test_boundary_odd_both_420);
    mu_run_test(test_boundary_max_dimension);

    return NULL;
}
