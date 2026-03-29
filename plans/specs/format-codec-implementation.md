# Format and Codec Coverage Tests -- Implementation Report

## Files Created

| File | Description |
|------|-------------|
| `libvmaf/test/test_format_common.h` | Shared utilities: Y4M generation, fill patterns, helper functions |
| `libvmaf/test/test_format_y4m.c` | Y4M header parsing, format-depth round-trips, range, errors |
| `libvmaf/test/test_format_picture.c` | VmafPicture allocation, stride alignment, odd dimensions |

## Files Modified

| File | Change |
|------|--------|
| `libvmaf/test/meson.build` | Registered `test_format_y4m` and `test_format_picture` executables and test targets |

## Test Coverage Summary

### test_format_y4m (50 top-level tests, 120+ sub-tests in matrix)

**Section 3: Y4M Header Parsing (23 tests)**
- Valid headers: minimal, all tags, reordered tags, unknown tags, extra spaces
- Missing required fields: width, height, fps
- Missing/wrong magic, wrong version
- Truncated header, header at buffer size limit, header exceeding buffer
- Chroma type near overflow (15 chars), chroma type overflow (16+ chars)
- Unknown chroma type, interlaced content rejection
- Negative dimensions, zero dimensions, fps zero denominator
- Empty after magic
- Frame header with parameters, overlong frame header

**Section 4: Format-Depth Matrix (1 test, 120 sub-tests)**
- 10 formats: 420/420p10/420p12/422/422p10/422p12/444/444p10/444p12/mono
- 4 dimension sets: 16x16, 320x240, 321x241, 1920x1080
- 3 fill patterns: FILL_ZERO, FILL_GRADIENT_H, FILL_RANDOM_42
- Each sub-test: Y4M generation, parse, info verification, frame fetch, VmafPicture alloc, stride check

**Section 4.4: YUV444P Explicit Coverage (3 tests, R3)**
- test_y4m_444_8bit: chroma w == luma w, chroma h == luma h
- test_y4m_444_10bit: same assertions plus stride alignment
- test_y4m_444_12bit: same assertions plus stride alignment

**Section 5: Range Handling (7 tests)**
- XCOLORRANGE=LIMITED silently ignored
- XCOLORRANGE=FULL silently ignored
- Limited-range pixel values [16,235] preserved through round-trip
- Full-range pixel values [0,255] preserved through round-trip
- Boundary values 8-bit: 0, 1, 15, 16, 235, 236, 254, 255
- Boundary values 10-bit: 0, 64, 940, 1023
- Boundary values 12-bit: 0, 256, 3760, 4095

**Section 8.1: Error Handling (10 tests)**
- No magic, empty file, truncated frame, missing FRAME tag
- Unsupported chroma (422p16), interlaced rejection
- vmaf_picture_alloc with UNKNOWN format, bpc=7, bpc=17, NULL pointer

**Section 8.2: Boundary Conditions (6 tests)**
- 1x1 at 420 and 444
- Odd width (3x4), odd height (4x3), both odd (3x3) at 420
- 8K dimensions (7680x4320)

### test_format_picture (8 top-level tests, 60+ sub-tests)

**Pixel Format Allocation (1 test, 18 sub-tests)**
- All 4 formats (420/422/444/400) x multiple bit depths (8/10/12/16) x multiple dimensions
- Verifies pix_fmt, bpc, luma/chroma dimensions, data pointers

**Stride Alignment (1 test, 16 sub-tests)**
- Widths: 1, 31, 32, 33, 321, 1920, 1921, 7680
- All formats at 8-bit and 10-bit
- Verifies 32-byte stride alignment, 32-byte data pointer alignment, stride >= needed

**Odd Dimensions (1 test, 21 sub-tests)**
- 420: 3x4, 4x3, 3x3, 5x5, 15x17, 321x241 at 8/10/12-bit
- 422: 3x4, 5x7, 321x241 at 8/10-bit
- 444: 3x3, 321x241 at 8/10-bit
- 400: 3x3, 321x241
- Very small: 1x1, 2x1, 1x2, 2x2 at 420
- Verifies expected chroma dimensions (w>>ss_hor, h>>ss_ver)

**Stride Write/Read (1 test, 5 sub-tests)**
- Writes gradient pattern, reads back, verifies padding bytes remain zero

**High Bit Depth Stride (1 test, 6 sub-tests)**
- Verifies stride = ((w+31) & ~31) << 1 for 10/12-bit

**Error Handling (1 test)**
- NULL pic, UNKNOWN format, bpc=0, bpc=7, bpc=17

**YUV400P Specifics (2 tests)**
- 8-bit and 10-bit mono: NULL chroma pointers, zero chroma dimensions

## Spec Requirement Traceability

| Req | Status | Evidence |
|-----|--------|----------|
| R1: Header parsing coverage | Met | 23 header tests all passing |
| R2: Full format-depth matrix | Met | 10 formats x 4 dims x 3 patterns = 120 sub-tests |
| R3: YUV444P explicit coverage | Met | 3 dedicated tests asserting chroma == luma dims |
| R4: Range preservation | Met | 7 tests covering limited, full, boundary at 8/10/12-bit |
| R5: Non-standard stride handling | Met | 16 stride alignment sub-tests, 5 write/read sub-tests, 6 HBD stride sub-tests |
| R6: Odd dimension chroma rounding | Met | 21 odd-dimension sub-tests including 321x241 at multiple formats |
| R7: Error rejection | Met | 10 malformed input tests + 5 alloc error tests, no crashes |
| R8: Programmatic test data | Met | Zero external media files; all Y4M data generated at runtime |
| R9: Build integration | Met | Both tests registered in meson.build, pass via `ninja test` |
| R10: HBD byte order | Met | 10-bit and 12-bit boundary tests verify LE round-trip |
| R11: Deterministic | Met | PRNG seeds hardcoded (42+frame), no time/system dependencies |
| R12: Zero production impact | Met | No production source files modified |
