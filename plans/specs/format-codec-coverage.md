# Format and Codec Coverage Tests — Specification

## 1. Objective

Validate that libvmaf's Y4M parser, raw YUV input path, and `VmafPicture` allocation correctly handle the full matrix of pixel formats, bit depths, chroma subsampling modes, Y4M header variations, YUV range semantics, and non-standard memory strides, including proper rejection of malformed or unsupported inputs.

This test suite must catch regressions in format parsing, buffer sizing, and pixel data layout before they reach production.

---

## 2. Scope — Components Under Test

There are **4 components** spanning 3 source files and 1 public header.

### 2.1 Y4M Header Parser

**Source:** `libvmaf/tools/y4m_input.c`

The parser reads the `YUV4MPEG2` stream header and extracts:
- `W` (width), `H` (height) — required
- `F` (framerate as `num:den`) — required
- `C` (chroma type) — optional, defaults to `420`
- `I` (interlace mode) — optional, defaults to `?` (treated as progressive)
- `A` (pixel aspect ratio as `num:den`) — optional, defaults to `0:0`

Unknown tags are silently ignored. The header buffer is fixed at 256 bytes (`Y4M_HEADER_BUFSIZE`).

### 2.2 Y4M Chroma Type Dispatch

**Source:** `libvmaf/tools/y4m_input.c`, function `y4m_input_open_impl`

The parser recognizes these chroma type strings and maps them to internal format parameters:

| Chroma String | Subsampling | Bit Depth | Conversion Required |
|---------------|-------------|-----------|---------------------|
| `420` | 4:2:0 | 8 | None |
| `420jpeg` | 4:2:0 | 8 | None |
| `420mpeg2` | 4:2:0 | 8 | None |
| `420p10` | 4:2:0 | 10 | None |
| `420p12` | 4:2:0 | 12 | None |
| `422` | 4:2:2 | 8 | mpeg2-to-jpeg chroma shift |
| `422p10` | 4:2:2 | 10 | None |
| `422p12` | 4:2:2 | 12 | None |
| `444` | 4:4:4 | 8 | None |
| `444p10` | 4:4:4 | 10 | None |
| `444p12` | 4:4:4 | 12 | None |
| `444alpha` | 4:4:4 | 8 | Alpha plane discarded |
| `420paldv` | 4:2:0 | 8 | PAL DV chroma resampling |
| `411` | 4:1:1 -> 4:2:2 | 8 | Horizontal upsampling |
| `mono` | Luma only | 8 | Chroma filled with 128 |

### 2.3 VmafPicture Allocation and Stride

**Source:** `libvmaf/src/picture.c`, **Header:** `libvmaf/include/libvmaf/picture.h`

`vmaf_picture_alloc` computes per-plane widths, heights, and strides:
- Stride is aligned to 32-byte boundaries (`DATA_ALIGN`).
- For high bit depth (`bpc > 8`), stride is doubled.
- Chroma plane dimensions are halved according to subsampling.
- All data pointers are 32-byte aligned.

Supported pixel formats:
```c
enum VmafPixelFormat {
    VMAF_PIX_FMT_UNKNOWN,   // 0 — rejected
    VMAF_PIX_FMT_YUV420P,   // 1
    VMAF_PIX_FMT_YUV422P,   // 2
    VMAF_PIX_FMT_YUV444P,   // 3
    VMAF_PIX_FMT_YUV400P,   // 4 — luma only
};
```

### 2.4 Frame Data Fetch and Copy

**Source:** `libvmaf/tools/vmaf.c`, function `fetch_picture`

Copies per-plane pixel data from Y4M/YUV frame buffers into `VmafPicture`, handling:
- 8-bit vs. high-bit-depth paths
- Bit-depth upscaling via left-shift when reference and distorted depths differ
- Chroma decimation factors derived from `pixel_fmt` bit encoding

---

## 3. Y4M Header Parsing Edge Cases

### 3.1 Test Cases

Each test constructs a Y4M header as an in-memory byte buffer, writes it (optionally followed by frame data) to a temporary file, then opens it via `y4m_input_open` (through `video_input_open`) and verifies the resulting `video_input_info` fields.

| # | Test Name | Header Content | Expected Result |
|---|-----------|---------------|-----------------|
| 1 | `valid_minimal` | `YUV4MPEG2 W320 H240 F30:1\n` | Success; w=320, h=240, fps=30:1, chroma defaults to 420 |
| 2 | `valid_all_tags` | `YUV4MPEG2 W1920 H1080 F24:1 Ip A1:1 C420\n` | Success; all fields parsed correctly |
| 3 | `valid_tag_reorder` | `YUV4MPEG2 H240 F30:1 W320\n` | Success; tag order should not matter |
| 4 | `valid_unknown_tags` | `YUV4MPEG2 W320 H240 F30:1 Xfoo Ybar\n` | Success; unknown tags silently ignored |
| 5 | `valid_extra_spaces` | `YUV4MPEG2  W320  H240  F30:1\n` | Success; multiple spaces between tags |
| 6 | `missing_width` | `YUV4MPEG2 H240 F30:1\n` | Failure; `y4m_parse_tags` returns -1 |
| 7 | `missing_height` | `YUV4MPEG2 W320 F30:1\n` | Failure; returns -1 |
| 8 | `missing_fps` | `YUV4MPEG2 W320 H240\n` | Failure; returns -1 |
| 9 | `missing_magic` | `NOT_Y4M W320 H240 F30:1\n` | Failure; magic check fails |
| 10 | `wrong_version` | `YUV4MPEG1 W320 H240 F30:1\n` | Warning printed but parsing continues (version check is non-fatal) |
| 11 | `truncated_header` | `YUV4MPEG2 W320 H2` (no newline, EOF) | Failure; fread loop exhausts 256 bytes |
| 12 | `header_at_bufsize_limit` | 255-byte header ending with `\n` | Success; exactly fits buffer |
| 13 | `header_exceeds_bufsize` | 256+ byte header (no newline within 255 chars) | Parsed with truncation; may fail if required tags are truncated |
| 14 | `chroma_type_near_overflow` | `C` tag with 15-char value (max for `chroma_type[16]`) | Success; fits in buffer with null terminator |
| 15 | `chroma_type_overflow` | `C` tag with 16+ char value | Failure; `q-p > 16` check triggers return -1 |
| 16 | `unknown_chroma_type` | `YUV4MPEG2 W320 H240 F30:1 Crgb\n` | Failure; unrecognized chroma type |
| 17 | `interlaced_content` | `YUV4MPEG2 W320 H240 F30:1 It\n` | Failure; only progressive (`Ip`) is accepted |
| 18 | `negative_dimensions` | `YUV4MPEG2 W-1 H240 F30:1\n` | Failure or undefined; sscanf may parse -1 |
| 19 | `zero_dimensions` | `YUV4MPEG2 W0 H0 F30:1\n` | Parser succeeds but buffer size computations degenerate |
| 20 | `fps_zero_denominator` | `YUV4MPEG2 W320 H240 F30:0\n` | Parser succeeds (no validation on fps_d) |
| 21 | `empty_after_magic` | `YUV4MPEG2\n` | Failure; missing W, H, F |
| 22 | `frame_header_params` | Frame header `FRAME Ifirst\n` (with parameters after FRAME) | Success; frame header params are skipped |
| 23 | `frame_header_overlong` | Frame header with 80+ chars after FRAME | Failure; frame header parsing rejects > 79 chars |

### 3.2 Implementation Approach

Because `y4m_input_open` reads from a `FILE *`, tests must create temporary files containing crafted headers. Use `tmpfile()` or `mkstemp()` to create the file, write the header bytes, rewind with `fseek`, then pass to `video_input_open`.

```c
static FILE *create_y4m_tmpfile(const char *header, size_t header_len,
                                 const uint8_t *frame_data, size_t frame_len) {
    FILE *f = tmpfile();
    if (!f) return NULL;
    fwrite(header, 1, header_len, f);
    if (frame_data && frame_len > 0)
        fwrite(frame_data, 1, frame_len, f);
    fseek(f, 0, SEEK_SET);
    return f;
}
```

---

## 4. Color Space and Bit Depth Coverage

### 4.1 Format-Depth Matrix

Every combination of pixel format and bit depth that libvmaf supports must be tested end-to-end: Y4M parse, `VmafPicture` allocation, and pixel data round-trip.

| # | Pixel Format | Bit Depth | Y4M Chroma String | `VmafPixelFormat` | Status |
|---|-------------|-----------|-------------------|-------------------|--------|
| 1 | YUV420P | 8 | `420` | `VMAF_PIX_FMT_YUV420P` | Tested in existing suite |
| 2 | YUV420P | 10 | `420p10` | `VMAF_PIX_FMT_YUV420P` | Tested in existing suite |
| 3 | YUV420P | 12 | `420p12` | `VMAF_PIX_FMT_YUV420P` | Needs test |
| 4 | YUV422P | 8 | `422` | `VMAF_PIX_FMT_YUV422P` | Needs test |
| 5 | YUV422P | 10 | `422p10` | `VMAF_PIX_FMT_YUV422P` | Needs test |
| 6 | YUV422P | 12 | `422p12` | `VMAF_PIX_FMT_YUV422P` | Needs test |
| 7 | YUV444P | 8 | `444` | `VMAF_PIX_FMT_YUV444P` | **Currently untested** |
| 8 | YUV444P | 10 | `444p10` | `VMAF_PIX_FMT_YUV444P` | **Currently untested** |
| 9 | YUV444P | 12 | `444p12` | `VMAF_PIX_FMT_YUV444P` | **Currently untested** |
| 10 | YUV400P (mono) | 8 | `mono` | mapped to 420 internally | Needs test |

### 4.2 Per-Format Verification Points

For each format-depth combination, verify:

1. **Y4M parsing:** `video_input_get_info` returns correct `pixel_fmt`, `depth`, `pic_w`, `pic_h`.
2. **`VmafPicture` allocation:** `vmaf_picture_alloc` succeeds; plane widths, heights, and strides are correct.
3. **Plane dimensions:** Chroma planes have correct subsampled dimensions.

| Format | Luma W x H | Chroma W x H | Bytes per sample |
|--------|-----------|--------------|------------------|
| YUV420P | W x H | (W/2) x (H/2) | 1 (8-bit), 2 (10/12-bit) |
| YUV422P | W x H | (W/2) x H | 1 (8-bit), 2 (10/12-bit) |
| YUV444P | W x H | W x H | 1 (8-bit), 2 (10/12-bit) |
| YUV400P | W x H | 0 x 0 | 1 (8-bit) |

4. **Pixel data round-trip:** Write known pixel values into a Y4M file, read them back through the parser, copy into `VmafPicture`, and verify the values match.

### 4.3 Test Dimensions

Each format-depth combination must be tested at:

| Width | Height | Rationale |
|-------|--------|-----------|
| 16 | 16 | Minimum practical size, power of 2 |
| 320 | 240 | Standard small resolution |
| 321 | 241 | Odd dimensions (chroma rounding edge case) |
| 1920 | 1080 | Full HD |

Odd dimensions are particularly important for 4:2:0 and 4:2:2 where chroma plane sizes involve rounding: `(W+1)/2` and `(H+1)/2`.

### 4.4 YUV444P-Specific Tests

YUV444P is flagged as **currently untested** in the project plan. The following must be explicitly verified:

```c
static char *test_y4m_444_8bit(void) {
    /* Generate Y4M with C444 header and 3 full-resolution planes */
    const unsigned w = 320, h = 240;
    const size_t plane_sz = w * h;
    const size_t frame_sz = plane_sz * 3;  /* Y + Cb + Cr, no subsampling */

    /* Verify parsed info */
    mu_assert("pixel_fmt must be PF_444", info.pixel_fmt == PF_444);
    mu_assert("depth must be 8", info.depth == 8);

    /* Verify VmafPicture plane dimensions */
    VmafPicture pic;
    int err = vmaf_picture_alloc(&pic, VMAF_PIX_FMT_YUV444P, 8, w, h);
    mu_assert("alloc must succeed", !err);
    mu_assert("chroma width must equal luma width", pic.w[1] == w);
    mu_assert("chroma height must equal luma height", pic.h[1] == h);
    /* ... copy data and verify pixel values ... */
    vmaf_picture_unref(&pic);
    return NULL;
}
```

---

## 5. Limited vs. Full Range YUV Handling

### 5.1 Background

The Y4M format does not natively carry a color range flag. However:

- The Y4M spec allows arbitrary `X`-prefixed tags (extension tags), and tools like FFmpeg emit `XCOLORRANGE=FULL` or `XCOLORRANGE=LIMITED`.
- libvmaf's current Y4M parser **silently ignores** unknown tags, including `XCOLORRANGE`.
- libvmaf features use `OPT_RANGE_PIXEL_OFFSET` (defined as -128 in `offset.h`) to shift pixel values, assuming limited-range input.

### 5.2 Current Behavior

The parser has no range awareness. All Y4M input is treated identically regardless of whether pixel values span [16, 235] (limited) or [0, 255] (full). This is a known limitation.

### 5.3 Test Cases

| # | Test Name | Description | Expectation |
|---|-----------|-------------|-------------|
| 1 | `range_tag_ignored` | Y4M header with `XCOLORRANGE=LIMITED` | Parser succeeds; tag is silently ignored; `video_input_info` has no range field |
| 2 | `range_tag_full_ignored` | Y4M header with `XCOLORRANGE=FULL` | Parser succeeds; tag is silently ignored |
| 3 | `limited_range_pixel_values` | Frame with luma values in [16, 235] | All values preserved in `VmafPicture` without clipping |
| 4 | `full_range_pixel_values` | Frame with luma values in [0, 255] | All values preserved in `VmafPicture` without clipping |
| 5 | `boundary_values_8bit` | Frame with pixels = 0, 1, 15, 16, 235, 236, 254, 255 | All values survive round-trip without modification |
| 6 | `boundary_values_10bit` | Frame with pixels = 0, 64, 940, 1023 | All values survive round-trip |
| 7 | `boundary_values_12bit` | Frame with pixels = 0, 256, 3760, 4095 | All values survive round-trip |
| 8 | `vmaf_score_limited_vs_full` | Integration test: compute VMAF on identical content encoded as limited-range vs. full-range | Scores differ (documents current behavior); test records the delta |

### 5.4 Implementation Note

Tests 1-7 are unit tests that verify the parser and `VmafPicture` faithfully preserve input pixel values without range clipping. Test 8 is an integration test that documents the impact of range mismatch on VMAF scores (not a pass/fail test, but a regression anchor).

---

## 6. Non-Standard Strides and Padded Rows

### 6.1 Background

`vmaf_picture_alloc` computes strides as `(width + 31) & ~31` (aligned to 32-byte boundaries), then doubles for high bit depth. External callers using `vmaf_picture_alloc` always get aligned strides. However:

- The Y4M parser uses **tightly packed** rows (stride = width * bytes_per_sample) in its internal buffers.
- The `fetch_picture` function in `vmaf.c` copies row-by-row from tight Y4M buffers to aligned `VmafPicture` strides.
- External API users may supply `VmafPicture` structures with custom strides (e.g., from GPU memory or framework-allocated buffers).

### 6.2 Test Cases

| # | Test Name | Description | Expectation |
|---|-----------|-------------|-------------|
| 1 | `stride_equals_width` | `VmafPicture` with stride == width (tight packing) | Feature extractors handle correctly |
| 2 | `stride_aligned_32` | `VmafPicture` from `vmaf_picture_alloc` (default) | Baseline; must work |
| 3 | `stride_aligned_64` | `VmafPicture` with stride manually set to 64-byte alignment | Feature extractors handle correctly |
| 4 | `stride_aligned_128` | `VmafPicture` with stride at 128-byte alignment | Feature extractors handle correctly |
| 5 | `stride_odd_padding` | `VmafPicture` with stride = width + 7 (non-power-of-2 padding) | Feature extractors handle correctly |
| 6 | `stride_large_gap` | `VmafPicture` with stride = width * 2 (50% padding per row) | Feature extractors handle correctly |
| 7 | `stride_hbd_tight` | 10-bit `VmafPicture` with stride = width * 2 (tight 16-bit packing) | Correct pixel access |
| 8 | `stride_hbd_padded` | 10-bit `VmafPicture` with stride = (width * 2 + 63) & ~63 | Correct pixel access |
| 9 | `stride_mismatch_planes` | `VmafPicture` where luma stride != 2 * chroma stride | Feature extractors handle independently |
| 10 | `y4m_to_picture_stride_copy` | Parse Y4M (tight rows) into `VmafPicture` (aligned rows) | Pixel values identical; padding bytes are zero |

### 6.3 Implementation Approach

Tests 1-9 bypass the Y4M parser and construct `VmafPicture` structures directly using manual allocation (not `vmaf_picture_alloc`) to control stride values:

```c
static int alloc_picture_custom_stride(VmafPicture *pic,
                                        enum VmafPixelFormat pix_fmt,
                                        unsigned bpc, unsigned w, unsigned h,
                                        ptrdiff_t stride_y, ptrdiff_t stride_c) {
    memset(pic, 0, sizeof(*pic));
    pic->pix_fmt = pix_fmt;
    pic->bpc = bpc;

    const int ss_hor = pix_fmt != VMAF_PIX_FMT_YUV444P;
    const int ss_ver = pix_fmt == VMAF_PIX_FMT_YUV420P;
    pic->w[0] = w;
    pic->w[1] = pic->w[2] = w >> ss_hor;
    pic->h[0] = h;
    pic->h[1] = pic->h[2] = h >> ss_ver;
    pic->stride[0] = stride_y;
    pic->stride[1] = pic->stride[2] = stride_c;

    const size_t y_sz = stride_y * h;
    const size_t uv_sz = stride_c * (h >> ss_ver);
    uint8_t *data = aligned_malloc(y_sz + 2 * uv_sz, 32);
    if (!data) return -1;
    memset(data, 0, y_sz + 2 * uv_sz);
    pic->data[0] = data;
    pic->data[1] = data + y_sz;
    pic->data[2] = data + y_sz + uv_sz;

    /* Initialize ref counting */
    vmaf_picture_priv_init(pic);
    vmaf_ref_init(&pic->ref);
    return 0;
}
```

Test 10 creates a Y4M file with known pixel data, parses it through the full `video_input` + `fetch_picture` path, and verifies that the resulting `VmafPicture` has correct aligned strides with zeroed padding bytes and correct pixel values.

### 6.4 Verifying Stride Correctness

For each stride test, fill the luma plane with a horizontal gradient pattern:

```c
/* Fill luma plane with row-major gradient */
for (unsigned y = 0; y < h; y++) {
    uint8_t *row = (uint8_t *)pic->data[0] + y * pic->stride[0];
    for (unsigned x = 0; x < w; x++) {
        row[x] = (uint8_t)((x + y * w) & 0xFF);
    }
}
```

Then run a feature extractor (e.g., PSNR on two identical pictures) and verify the score is the expected perfect value. If the feature extractor mishandles strides, the score will be wrong because it reads padding bytes as pixel data.

---

## 7. Test Data Generation

### 7.1 Programmatic Y4M File Creation

All test Y4M files are generated programmatically at test time. No external test media files are required.

```c
/**
 * Generate a minimal Y4M file in memory.
 *
 * @param w          Frame width
 * @param h          Frame height
 * @param chroma     Chroma type string (e.g., "420", "444p10")
 * @param bpc        Bits per component (8, 10, or 12)
 * @param n_frames   Number of frames to generate
 * @param pattern    Fill pattern for pixel data
 * @param out_size   Output: total file size
 * @return           Allocated buffer containing the Y4M file, or NULL on error.
 *                   Caller must free.
 */
static uint8_t *generate_y4m(unsigned w, unsigned h, const char *chroma,
                              unsigned bpc, unsigned n_frames,
                              enum fill_pattern pattern, size_t *out_size);
```

### 7.2 Fill Patterns

| Pattern | Description | Purpose |
|---------|-------------|---------|
| `FILL_ZERO` | All samples = 0 | Lower bound of full range |
| `FILL_MAX` | All samples = (1 << bpc) - 1 | Upper bound of full range |
| `FILL_MID` | All samples = 1 << (bpc - 1) | Mid-range constant |
| `FILL_LIMITED_BLACK` | Y=16, Cb=Cr=128 (scaled for bpc) | Limited range black |
| `FILL_LIMITED_WHITE` | Y=235, Cb=Cr=128 (scaled for bpc) | Limited range white |
| `FILL_GRADIENT_H` | Horizontal ramp 0..max | Detects byte-order or stride issues |
| `FILL_GRADIENT_V` | Vertical ramp 0..max | Detects stride handling issues |
| `FILL_RANDOM_42` | Deterministic PRNG (seed=42) | General correctness |
| `FILL_CHECKERBOARD` | Alternating 0 and max | High-frequency pattern |

### 7.3 Y4M Header Construction

```c
static size_t write_y4m_header(uint8_t *buf, size_t buf_sz,
                                unsigned w, unsigned h, const char *chroma) {
    return snprintf((char *)buf, buf_sz,
                    "YUV4MPEG2 W%u H%u F30:1 Ip C%s\n",
                    w, h, chroma);
}
```

### 7.4 Frame Data Layout

For each frame, write `FRAME\n` followed by raw planar pixel data:

```c
static size_t compute_frame_size(unsigned w, unsigned h,
                                  const char *chroma, unsigned bpc) {
    unsigned xstride = (bpc > 8) ? 2 : 1;
    size_t y_sz = w * h * xstride;
    size_t c_w, c_h;

    if (strncmp(chroma, "444", 3) == 0) {
        c_w = w; c_h = h;
    } else if (strncmp(chroma, "422", 3) == 0) {
        c_w = (w + 1) / 2; c_h = h;
    } else if (strcmp(chroma, "mono") == 0) {
        c_w = 0; c_h = 0;
    } else { /* 420 variants */
        c_w = (w + 1) / 2; c_h = (h + 1) / 2;
    }

    return y_sz + 2 * c_w * c_h * xstride;
}
```

### 7.5 High Bit Depth Byte Order

Y4M files store high-bit-depth samples in **little-endian** 16-bit words. The test data generator must match this:

```c
static void write_sample_16le(uint8_t *dst, uint16_t val) {
    dst[0] = val & 0xFF;
    dst[1] = (val >> 8) & 0xFF;
}
```

---

## 8. Error Handling Tests

### 8.1 Malformed Input Rejection

| # | Test Name | Input | Expected |
|---|-----------|-------|----------|
| 1 | `err_no_magic` | File starting with random bytes | `video_input_open` returns error |
| 2 | `err_empty_file` | Zero-length file | `video_input_open` returns error |
| 3 | `err_truncated_frame` | Valid header but frame data shorter than expected | `video_input_fetch_frame` returns -1 |
| 4 | `err_missing_frame_tag` | Valid header but frame data without `FRAME` prefix | `video_input_fetch_frame` returns -1 |
| 5 | `err_unsupported_chroma` | `C422p16` (not in supported set) | `y4m_input_open` returns error |
| 6 | `err_interlaced` | `It` interlace tag | `y4m_input_open` returns error |
| 7 | `err_picture_alloc_bad_fmt` | `vmaf_picture_alloc` with `VMAF_PIX_FMT_UNKNOWN` | Returns `-EINVAL` |
| 8 | `err_picture_alloc_bad_bpc` | `vmaf_picture_alloc` with `bpc=7` | Returns `-EINVAL` |
| 9 | `err_picture_alloc_bad_bpc_high` | `vmaf_picture_alloc` with `bpc=17` | Returns `-EINVAL` |
| 10 | `err_picture_alloc_null` | `vmaf_picture_alloc(NULL, ...)` | Returns `-EINVAL` |

### 8.2 Boundary Condition Robustness

| # | Test Name | Input | Expected |
|---|-----------|-------|----------|
| 1 | `boundary_1x1_420` | 1x1 pixel YUV420P | Allocation succeeds; chroma planes are 1x1 (due to `(1+1)/2 = 1`) |
| 2 | `boundary_1x1_444` | 1x1 pixel YUV444P | Allocation succeeds; all planes 1x1 |
| 3 | `boundary_odd_w_420` | 3x4 pixel YUV420P | Chroma width = `(3+1)/2 = 2`; Y4M and VmafPicture agree |
| 4 | `boundary_odd_h_420` | 4x3 pixel YUV420P | Chroma height = `(3+1)/2 = 2`; agreement |
| 5 | `boundary_odd_both_420` | 3x3 pixel YUV420P | Chroma = 2x2; agreement |
| 6 | `boundary_max_dimension` | 7680x4320 (8K) | Allocation succeeds; correct stride alignment |

---

## 9. Test Implementation

### 9.1 Test File Organization

```
libvmaf/test/
  test_y4m_format.c          # Y4M header parsing and format coverage
  test_picture_stride.c      # Non-standard stride and padding tests
  test_format_common.h       # Shared Y4M generation utilities
```

Each test file is a standalone executable following the existing Minunit pattern (`test.h`).

### 9.2 test_format_common.h

This shared header provides:

```c
#ifndef TEST_FORMAT_COMMON_H
#define TEST_FORMAT_COMMON_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

enum fill_pattern {
    FILL_ZERO,
    FILL_MAX,
    FILL_MID,
    FILL_LIMITED_BLACK,
    FILL_LIMITED_WHITE,
    FILL_GRADIENT_H,
    FILL_GRADIENT_V,
    FILL_RANDOM_42,
    FILL_CHECKERBOARD,
};

/* Create a temporary file containing a Y4M stream with the given parameters.
 * Returns a FILE* rewound to the beginning, or NULL on failure.
 * The caller must fclose() the returned file. */
static FILE *create_y4m_file(unsigned w, unsigned h, const char *chroma,
                              unsigned bpc, unsigned n_frames,
                              enum fill_pattern pattern);

/* Create a temporary file containing only the raw Y4M header (no frame data).
 * Useful for testing header parsing in isolation.
 * The header string is written verbatim; the caller controls exact byte content. */
static FILE *create_y4m_header_only(const char *header, size_t len);

/* Compute expected frame data size for a given format. */
static size_t compute_frame_data_size(unsigned w, unsigned h,
                                       const char *chroma, unsigned bpc);

/* Fill a buffer with the specified pattern for a given plane. */
static void fill_plane(uint8_t *buf, unsigned w, unsigned h,
                        unsigned bpc, enum fill_pattern pattern, int plane_idx);

#endif /* TEST_FORMAT_COMMON_H */
```

### 9.3 test_y4m_format.c Structure

```c
#include "test.h"
#include "test_format_common.h"
#include "vidinput.h"
#include "libvmaf/picture.h"

/* === Section 3: Header parsing edge cases === */
static char *test_y4m_valid_minimal(void) { ... }
static char *test_y4m_valid_all_tags(void) { ... }
static char *test_y4m_missing_width(void) { ... }
/* ... 20+ header tests from Section 3.1 ... */

/* === Section 4: Format-depth matrix === */
static char *test_y4m_420_8bit(void) { ... }
static char *test_y4m_420_10bit(void) { ... }
static char *test_y4m_420_12bit(void) { ... }
static char *test_y4m_422_8bit(void) { ... }
static char *test_y4m_422_10bit(void) { ... }
static char *test_y4m_422_12bit(void) { ... }
static char *test_y4m_444_8bit(void) { ... }
static char *test_y4m_444_10bit(void) { ... }
static char *test_y4m_444_12bit(void) { ... }
static char *test_y4m_mono_8bit(void) { ... }

/* === Section 5: Range handling === */
static char *test_y4m_range_tag_ignored(void) { ... }
static char *test_y4m_full_range_values(void) { ... }
static char *test_y4m_limited_range_values(void) { ... }
static char *test_y4m_boundary_values_8bit(void) { ... }
static char *test_y4m_boundary_values_10bit(void) { ... }
static char *test_y4m_boundary_values_12bit(void) { ... }

/* === Section 8: Error handling === */
static char *test_y4m_err_no_magic(void) { ... }
static char *test_y4m_err_empty_file(void) { ... }
static char *test_y4m_err_truncated_frame(void) { ... }
/* ... remaining error tests ... */

char *run_tests(void) {
    /* Header parsing */
    mu_run_test(test_y4m_valid_minimal);
    mu_run_test(test_y4m_valid_all_tags);
    mu_run_test(test_y4m_missing_width);
    /* ... */

    /* Format-depth matrix */
    mu_run_test(test_y4m_420_8bit);
    mu_run_test(test_y4m_420_10bit);
    /* ... */

    /* Range handling */
    mu_run_test(test_y4m_range_tag_ignored);
    /* ... */

    /* Error handling */
    mu_run_test(test_y4m_err_no_magic);
    /* ... */

    return NULL;
}
```

### 9.4 Format Round-Trip Test Pattern

Each format-depth test follows this pattern:

```c
static char *test_y4m_444_10bit(void) {
    const unsigned w = 320, h = 240;
    const unsigned bpc = 10;
    const char *chroma = "444p10";

    /* 1. Generate Y4M file with known pixel data */
    FILE *f = create_y4m_file(w, h, chroma, bpc, 1, FILL_GRADIENT_H);
    mu_assert("failed to create test Y4M file", f != NULL);

    /* 2. Open via video_input */
    video_input vid;
    int err = video_input_open(&vid, f);
    mu_assert("video_input_open must succeed", err == 0);

    /* 3. Verify parsed info */
    video_input_info info;
    video_input_get_info(&vid, &info);
    mu_assert("pixel_fmt must be PF_444", info.pixel_fmt == PF_444);
    mu_assert("depth must be 10", info.depth == 10);
    mu_assert("pic_w must be 320", info.pic_w == 320);
    mu_assert("pic_h must be 240", info.pic_h == 240);

    /* 4. Fetch frame and verify plane data */
    video_input_ycbcr ycbcr;
    int ret = video_input_fetch_frame(&vid, ycbcr, NULL);
    mu_assert("fetch_frame must return 1", ret == 1);

    /* 5. Verify chroma planes are full resolution */
    mu_assert("Cb plane width must equal luma", ycbcr[1].width == ycbcr[0].width);
    mu_assert("Cr plane width must equal luma", ycbcr[2].width == ycbcr[0].width);

    /* 6. Allocate VmafPicture and copy */
    VmafPicture pic;
    err = vmaf_picture_alloc(&pic, VMAF_PIX_FMT_YUV444P, bpc, w, h);
    mu_assert("vmaf_picture_alloc must succeed", !err);
    mu_assert("VmafPicture chroma width must equal luma", pic.w[1] == w);
    mu_assert("VmafPicture chroma height must equal luma", pic.h[1] == h);

    /* 7. Verify stride alignment */
    mu_assert("luma stride must be 32-byte aligned", (pic.stride[0] % 32) == 0);
    mu_assert("chroma stride must be 32-byte aligned", (pic.stride[1] % 32) == 0);

    /* 8. Cleanup */
    vmaf_picture_unref(&pic);
    video_input_close(&vid);
    return NULL;
}
```

---

## 10. Integration with Build System

### 10.1 Meson Configuration

Add to `libvmaf/test/meson.build`:

```meson
# Format and Codec Coverage Tests
format_test_inc = include_directories('../tools/')

test_y4m_format = executable('test_y4m_format',
    ['test.c', 'test_y4m_format.c',
     '../tools/vidinput.c', '../tools/y4m_input.c', '../tools/yuv_input.c',
     '../src/picture.c', '../src/mem.c', '../src/ref.c'],
    include_directories : [libvmaf_inc, test_inc,
                           include_directories('../src/'),
                           format_test_inc],
    dependencies : [stdatomic_dependency, thread_lib, cuda_dependency],
    link_with : get_option('default_library') == 'both' ? libvmaf.get_static_lib() : libvmaf,
)

test_picture_stride = executable('test_picture_stride',
    ['test.c', 'test_picture_stride.c',
     '../src/picture.c', '../src/mem.c', '../src/ref.c'],
    include_directories : [libvmaf_inc, test_inc,
                           include_directories('../src/')],
    dependencies : [stdatomic_dependency, thread_lib, math_lib, cuda_dependency],
    link_with : get_option('default_library') == 'both' ? libvmaf.get_static_lib() : libvmaf,
)

test('test_y4m_format', test_y4m_format)
test('test_picture_stride', test_picture_stride)
```

### 10.2 Dependencies

- `test_y4m_format` links against the Y4M parser (`y4m_input.c`), the raw YUV parser (`yuv_input.c`), the video input dispatcher (`vidinput.c`), and the picture allocation code. It needs include paths for both `libvmaf/tools/` and `libvmaf/src/`.
- `test_picture_stride` only needs picture allocation and feature extraction. It links against the main `libvmaf` library to access feature extractors for stride verification.

### 10.3 CI Integration

Both test executables run as part of `ninja test` on all CI platforms. No external test media files are required; all test data is generated programmatically.

---

## 11. Test Execution Matrix

### 11.1 Y4M Header Parsing Tests

```
for each header_test_case in [valid_minimal, valid_all_tags, ..., frame_header_overlong]:
    1. Create tmpfile with crafted header
    2. Attempt video_input_open
    3. Assert success or failure as specified in Section 3.1
    4. If success, verify video_input_info fields
    5. Close and cleanup
```

**Total: 23 header parsing tests.**

### 11.2 Format-Depth Round-Trip Tests

```
for each (format, bpc, chroma) in format_depth_matrix:
    for each (w, h) in [(16,16), (320,240), (321,241), (1920,1080)]:
        for each pattern in [FILL_ZERO, FILL_GRADIENT_H, FILL_RANDOM_42]:
            1. Generate Y4M file with pattern
            2. Open and parse
            3. Verify info fields
            4. Fetch frame
            5. Allocate VmafPicture
            6. Copy pixels
            7. Verify pixel values match input pattern
            8. Verify stride alignment
```

**Total: 10 formats x 4 dimensions x 3 patterns = 120 round-trip tests.**

### 11.3 Range Tests

**Total: 8 range tests** (Section 5.3).

### 11.4 Stride Tests

**Total: 10 stride tests** (Section 6.2).

### 11.5 Error Handling Tests

**Total: 16 error/boundary tests** (Sections 8.1 and 8.2).

### 11.6 Grand Total

**177 test points** across 2 test executables.

---

## 12. Completion Requirements

The format and codec coverage test harness is **complete** when ALL of the following requirements are met:

### R1. Y4M Header Parsing Coverage

All 23 header parsing test cases from Section 3.1 exist and produce the expected results (success or specified failure mode).

**Verification:** Each test case in Section 3.1 has a corresponding `test_y4m_*` function. Running the test binary produces 23 pass results for header tests.

### R2. Full Format-Depth Matrix

Every format-depth combination in Section 4.1 (10 combinations) has round-trip tests at all 4 dimension sets and at least 3 fill patterns.

**Verification:** Count the `mu_run_test` calls for format tests. There must be at least 120 test points covering the full cross-product.

### R3. YUV444P Explicit Coverage

YUV444P is tested at 8-bit, 10-bit, and 12-bit depths with explicit verification that chroma plane dimensions equal luma plane dimensions.

**Verification:** Tests `test_y4m_444_8bit`, `test_y4m_444_10bit`, `test_y4m_444_12bit` exist and assert `pic.w[1] == pic.w[0]` and `pic.h[1] == pic.h[0]`.

### R4. Range Preservation

Tests verify that pixel values at the boundaries of both limited range ([16, 235] for 8-bit) and full range ([0, 255] for 8-bit) survive the Y4M parse + `VmafPicture` copy round-trip without modification.

**Verification:** Tests from Section 5.3 (#3 through #7) explicitly write boundary pixel values and read them back, asserting exact equality.

### R5. Non-Standard Stride Handling

At least 10 stride configurations from Section 6.2 are tested. Tests verify that feature extractors produce correct results with non-default strides.

**Verification:** Test functions for each stride case in Section 6.2 exist. At least one test verifies PSNR output on stride-varied pictures.

### R6. Odd Dimension Chroma Rounding

Tests with odd width (321) and/or odd height (241) verify that chroma plane dimensions are computed as `(w+1)/2` and `(h+1)/2` for 4:2:0, matching between the Y4M parser and `VmafPicture`.

**Verification:** At least 3 tests use odd dimensions and assert chroma plane sizes.

### R7. Error Rejection

All 10 malformed-input tests from Section 8.1 verify that the parser returns an error (non-zero return code or NULL context) without crashing or leaking memory.

**Verification:** Each error test asserts the expected failure return value. Running under valgrind or ASan shows no leaks or undefined behavior.

### R8. Programmatic Test Data

No test depends on external media files. All Y4M test data is generated at runtime by the `test_format_common.h` utilities.

**Verification:** The test executables pass on a clean build with no pre-existing test data files.

### R9. Build Integration

Both `test_y4m_format` and `test_picture_stride` compile and run as part of `ninja test` on all CI platforms (Ubuntu x86_64, Ubuntu ARM64, macOS, Windows).

**Verification:** CI pipeline passes with the new tests on all platforms.

### R10. High Bit Depth Byte Order

10-bit and 12-bit Y4M round-trip tests verify correct little-endian sample encoding and decoding. Tests write 16-bit samples in LE order and verify the parsed values match.

**Verification:** Tests at 10-bit and 12-bit explicitly check multi-byte pixel values.

### R11. Deterministic Reproducibility

All tests are fully deterministic. PRNG seeds are hardcoded. No dependency on wall-clock time, system load, or filesystem ordering.

**Verification:** Run the test suite twice and confirm identical pass/fail results.

### R12. Zero Production Code Regression

The test infrastructure does not modify any production source files. Tests link against existing production code without changes. If any header exposure is needed, it is done via include path configuration in meson, not by modifying source files.

**Verification:** `git diff` of production source files shows no changes.

---

## 13. Acceptance Criteria Summary

| Req | Description | Pass Condition |
|-----|-------------|----------------|
| R1 | Header parsing coverage | 23/23 header test cases implemented and passing |
| R2 | Format-depth matrix | 10 formats x 4 dimensions x 3 patterns = 120+ round-trip tests |
| R3 | YUV444P explicit coverage | 3 tests (8/10/12-bit) asserting chroma == luma dimensions |
| R4 | Range preservation | Boundary pixel values survive round-trip at 8/10/12-bit |
| R5 | Non-standard strides | 10 stride configurations tested with feature extractors |
| R6 | Odd dimension rounding | Odd W and H tested for 4:2:0 chroma rounding |
| R7 | Error rejection | 10 malformed inputs rejected without crash or leak |
| R8 | Programmatic test data | Zero external media file dependencies |
| R9 | Build integration | Tests pass in CI on all platforms |
| R10 | HBD byte order | 10-bit and 12-bit LE round-trip verified |
| R11 | Deterministic | Repeated runs produce identical results |
| R12 | Zero production impact | No production source files modified |

All 12 requirements must be met for the format and codec coverage test harness to be considered complete.
