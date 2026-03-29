# Degenerate Input Test Suite -- Specification

## 1. Objective

Validate that every metric and feature extractor in libvmaf handles **degenerate, edge-case, and boundary-condition inputs** without crashing, producing undefined behavior, or returning silently wrong results. This test suite must verify that:

1. No input pattern causes a segfault, SIGFPE, or heap corruption.
2. Metrics return mathematically correct scores (or well-defined sentinel values) for all degenerate inputs.
3. Minimum and maximum image dimensions are handled gracefully, either by producing correct output or by returning a documented error code.
4. All bit depths (8, 10, 12, 16) are exercised.

This suite complements the SIMD Correctness Oracle (Section 3.2) by operating at the **feature-extractor API level** rather than at the individual SIMD function level.

---

## 2. Scope -- Metrics Under Test

The following 7 feature extractors are tested with degenerate inputs:

| # | Feature Extractor | Registration Name | Key Output Features |
|---|-------------------|-------------------|---------------------|
| 1 | Integer ADM | `"adm"` | `VMAF_integer_feature_adm2_score`, `integer_adm`, `integer_adm_num`, `integer_adm_den` |
| 2 | Integer VIF | `"vif"` | `VMAF_integer_feature_vif_scale0_score` .. `scale3`, `integer_vif`, `integer_vif_num`, `integer_vif_den` |
| 3 | Integer PSNR | `"psnr"` | `psnr_y`, `psnr_cb`, `psnr_cr` |
| 4 | Integer SSIM | `"ssim"` | `ssim` |
| 5 | Integer Motion | `"motion"` | `VMAF_integer_feature_motion2_score`, `VMAF_integer_feature_motion_score` |
| 6 | CAMBI | `"cambi"` | `cambi` |
| 7 | VMAF (composite) | via `VmafModel` | `vmaf` (final fused score) |

---

## 3. Degenerate Input Patterns

### 3.1 Synthetic Test Data Patterns

| # | Pattern Name | Description | Primary Purpose |
|---|-------------|-------------|-----------------|
| P1 | All-zero (black) | Every pixel = 0 | Division-by-zero in VIF/ADM denominators |
| P2 | All-max (white) | Every pixel = `(1 << bpc) - 1` | Saturation handling, integer overflow |
| P3 | Identical ref=dis | `ref` and `dis` are byte-identical | Perfect-score boundary (VMAF=100, PSNR=inf, SSIM=1.0) |
| P4 | Single-pixel difference | `ref` = all-128; `dis` = all-128 except one pixel = 129 | Minimum detectable distortion |
| P5 | Gradient ramp (0-max) | Horizontal ramp from 0 to `(1 << bpc) - 1` | Full dynamic range coverage |
| P6 | Random noise (seeded) | PRNG-filled (seed=42), independent ref and dis | Statistical stability, general correctness |
| P7 | Checkerboard at Nyquist | Alternating 0 and max in a checkerboard pattern | Worst-case for downsampling/filtering stages |
| P8 | DC constant (mid-gray) | Every pixel = `(1 << bpc) / 2` | Mid-range constant; filter DC response |
| P9 | Single-hot impulse | One pixel = max, rest = 0 | Impulse response; off-by-one indexing |
| P10 | Boundary-stripe | First and last row/column = max, rest = 0 | Edge/mirror handling in convolution filters |

### 3.2 Dimension Variants

| # | Width | Height | Rationale |
|---|-------|--------|-----------|
| D1 | 1 | 1 | Absolute minimum dimension |
| D2 | 2 | 2 | Minimum for subsampled chroma (YUV420) |
| D3 | 3 | 3 | Odd, non-power-of-2 minimum |
| D4 | 7 | 7 | Below SIMD threshold (`w % 8 != 0`) |
| D5 | 8 | 8 | Exact SIMD boundary |
| D6 | 9 | 9 | Just above SIMD boundary |
| D7 | 15 | 15 | Odd, near SIMD boundary |
| D8 | 17 | 17 | Just above 16, non-power-of-2 |
| D9 | 64 | 64 | Standard small resolution |
| D10 | 120 | 68 | Non-power-of-2, not multiple of 16 but multiple of 8 |
| D11 | 576 | 324 | Matches primary test video resolution |
| D12 | 1920 | 1080 | Full HD |
| D13 | 4096 | 2160 | 4K stress test |
| D14 | 7680 | 4320 | 8K stress test |

### 3.3 Bit Depth Variants

All tests must be repeated for each supported bit depth:

| Bit Depth | Pixel Format | Max Pixel Value | Bytes Per Sample |
|-----------|-------------|-----------------|------------------|
| 8 | `VMAF_PIX_FMT_YUV420P` | 255 | 1 |
| 10 | `VMAF_PIX_FMT_YUV420P` | 1023 | 2 |
| 12 | `VMAF_PIX_FMT_YUV420P` | 4095 | 2 |
| 16 | `VMAF_PIX_FMT_YUV420P` | 65535 | 2 |

### 3.4 Temporal Edge Cases (Motion-Specific)

| # | Scenario | Frame Count | Purpose |
|---|----------|-------------|---------|
| T1 | Single frame | 1 | Motion requires >= 2 frames; index 0 should emit score = 0.0 |
| T2 | Two identical frames | 2 | Motion score = 0.0 at both indices |
| T3 | Two maximally different frames | 2 | Frame 0 = all-zero, Frame 1 = all-max |

---

## 4. Synthetic Data Generator API

### 4.1 Header: `test_degenerate_common.h`

All generator functions operate on `VmafPicture` and are shared across test files.

```c
#ifndef TEST_DEGENERATE_COMMON_H
#define TEST_DEGENERATE_COMMON_H

#include <stdint.h>
#include <string.h>
#include "libvmaf/picture.h"

/*
 * Fill a pre-allocated VmafPicture with a degenerate pattern.
 * The picture must already be allocated via vmaf_picture_alloc().
 * Only the Y (luma) plane is filled for YUV400P; all 3 planes for YUV420P.
 */

/** P1: All pixels = 0 */
void degen_fill_black(VmafPicture *pic);

/** P2: All pixels = (1 << bpc) - 1 */
void degen_fill_white(VmafPicture *pic);

/** P3: All pixels = (1 << bpc) / 2  (mid-gray, also used for identical ref=dis) */
void degen_fill_midgray(VmafPicture *pic);

/** P4: All pixels = (1 << bpc) / 2, then set pixel at (0,0) to (1 << bpc) / 2 + 1 */
void degen_fill_single_pixel_diff(VmafPicture *pic);

/** P5: Horizontal gradient ramp from 0 to max across each row */
void degen_fill_gradient(VmafPicture *pic);

/** P6: Deterministic PRNG fill (xorshift32, given seed) */
void degen_fill_random(VmafPicture *pic, uint32_t seed);

/** P7: Checkerboard: even positions = 0, odd positions = max */
void degen_fill_checkerboard(VmafPicture *pic);

/** P8: DC constant (same as midgray but kept as distinct semantic entry) */
void degen_fill_constant(VmafPicture *pic, unsigned value);

/** P9: Single pixel at (h/2, w/2) = max, rest = 0 */
void degen_fill_impulse(VmafPicture *pic);

/** P10: First/last row and first/last column = max, rest = 0 */
void degen_fill_boundary_stripe(VmafPicture *pic);

/**
 * Allocate a VmafPicture pair (ref, dis) for a given pattern and dimension.
 * Returns 0 on success, negative errno on failure.
 *
 * For P3 (identical), ref and dis receive the same data.
 * For all other patterns, ref gets the pattern and dis gets a second pattern
 * (or the same pattern, depending on test intent).
 */
int degen_alloc_pair(VmafPicture *ref, VmafPicture *dis,
                     enum VmafPixelFormat pix_fmt, unsigned bpc,
                     unsigned w, unsigned h);

/** Helper: get the maximum pixel value for a given bit depth */
static inline unsigned degen_max_val(unsigned bpc) {
    return (1u << bpc) - 1;
}

/** Helper: get the mid-gray value for a given bit depth */
static inline unsigned degen_mid_val(unsigned bpc) {
    return 1u << (bpc - 1);
}

#endif /* TEST_DEGENERATE_COMMON_H */
```

### 4.2 PRNG Specification

The random fill uses a **xorshift32** generator for speed and reproducibility:

```c
static inline uint32_t xorshift32(uint32_t *state) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

void degen_fill_random(VmafPicture *pic, uint32_t seed) {
    uint32_t state = seed;
    const unsigned max = degen_max_val(pic->bpc);
    const unsigned n_planes = (pic->pix_fmt == VMAF_PIX_FMT_YUV400P) ? 1 : 3;
    const unsigned sample_size = (pic->bpc > 8) ? 2 : 1;

    for (unsigned p = 0; p < n_planes; p++) {
        for (unsigned i = 0; i < pic->h[p]; i++) {
            if (sample_size == 1) {
                uint8_t *row = (uint8_t *)pic->data[p] + i * pic->stride[p];
                for (unsigned j = 0; j < pic->w[p]; j++)
                    row[j] = xorshift32(&state) % (max + 1);
            } else {
                uint16_t *row = (uint16_t *)((uint8_t *)pic->data[p] +
                                              i * pic->stride[p]);
                for (unsigned j = 0; j < pic->w[p]; j++)
                    row[j] = xorshift32(&state) % (max + 1);
            }
        }
    }
}
```

### 4.3 Gradient Fill Specification

```c
void degen_fill_gradient(VmafPicture *pic) {
    const unsigned max = degen_max_val(pic->bpc);
    const unsigned n_planes = (pic->pix_fmt == VMAF_PIX_FMT_YUV400P) ? 1 : 3;

    for (unsigned p = 0; p < n_planes; p++) {
        const unsigned w = pic->w[p];
        for (unsigned i = 0; i < pic->h[p]; i++) {
            if (pic->bpc <= 8) {
                uint8_t *row = (uint8_t *)pic->data[p] + i * pic->stride[p];
                for (unsigned j = 0; j < w; j++)
                    row[j] = (uint8_t)((uint64_t)j * max / (w > 1 ? w - 1 : 1));
            } else {
                uint16_t *row = (uint16_t *)((uint8_t *)pic->data[p] +
                                              i * pic->stride[p]);
                for (unsigned j = 0; j < w; j++)
                    row[j] = (uint16_t)((uint64_t)j * max / (w > 1 ? w - 1 : 1));
            }
        }
    }
}
```

---

## 5. Expected Behavior Matrix

### 5.1 PSNR Expected Scores

| Pattern Pair | Expected `psnr_y` | Rationale |
|---|---|---|
| P1 ref vs P1 dis (black=black) | `psnr_max` (capped, per bit depth: 60 for 8-bit, 72 for 10-bit, 84 for 12-bit, 108 for 16-bit) | MSE=0; PSNR capped at `(6 * bpc) + 12` when `min_sse` is 0 |
| P2 ref vs P2 dis (white=white) | `psnr_max` (same as above) | MSE=0 |
| P3 identical ref=dis | `psnr_max` | MSE=0 |
| P1 ref vs P2 dis (black vs white) | 0.0 | MSE = max^2; PSNR = 10*log10(1) = 0 |
| P4 single-pixel diff | > 40 (8-bit, 64x64), exact value depends on dimensions | MSE = 1 / (w*h); PSNR = 10*log10(max^2 * w * h) |
| P6 random ref vs random dis | > 0, finite | Non-zero MSE for different seeds |

**Required behavior:** PSNR must never return NaN, negative infinity, or crash. MSE of zero must yield the capped `psnr_max`.

### 5.2 VIF Expected Scores

| Pattern Pair | Expected `integer_vif` | Rationale |
|---|---|---|
| P3 identical ref=dis | 1.0 (within tolerance 1e-4) | Perfect fidelity |
| P1 ref=dis (both black) | 1.0 or well-defined value; no crash | Zero signal; denominator protection must activate |
| P1 ref vs P2 dis | 0.0 or near 0.0; no crash | Maximum distortion |
| P2 ref=dis (both white) | 1.0 or well-defined value; no crash | Constant signal; denominator may be zero |
| P5 gradient ref=dis | 1.0 (within tolerance 1e-4) | Identical content |
| P7 checkerboard ref=dis | 1.0 (within tolerance 1e-4) | Identical content |

**Required behavior:** VIF must never produce NaN, infinity, segfault, or SIGFPE. When both ref and dis are constant (zero or max), the implementation must handle the zero-denominator case gracefully. The score must be in the range `[0.0, +inf)` (VIF can exceed 1.0 when dis has higher energy than ref, but must be finite).

### 5.3 ADM Expected Scores

| Pattern Pair | Expected `VMAF_integer_feature_adm2_score` | Rationale |
|---|---|---|
| P3 identical ref=dis | 1.0 (within tolerance 1e-4) | Perfect fidelity |
| P1 ref=dis (both black) | 1.0 or well-defined value; no crash | Zero wavelet coefficients; denominator protection |
| P1 ref vs P2 dis | 0.0 or near 0.0; no crash | Maximum distortion in all subbands |
| P7 checkerboard ref=dis | 1.0 (within tolerance 1e-4) | Identical content |

**Required behavior:** ADM must never produce NaN, infinity, segfault, or SIGFPE. Score must be in `[0.0, 1.0]` for valid inputs. The DWT requires a minimum image dimension; images that are too small for the 4-level wavelet decomposition (`w < 8` or `h < 8` after DWT) may return an error code rather than a score, and this is acceptable.

### 5.4 SSIM Expected Scores

| Pattern Pair | Expected `ssim` | Rationale |
|---|---|---|
| P3 identical ref=dis | 1.0 (exact or within 1e-6) | SSIM definition |
| P1 ref=dis (both black) | 1.0 (C1, C2 stabilization constants prevent 0/0) | SSIM uses (2*mu_x*mu_y + C1) stabilization |
| P2 ref=dis (both white) | 1.0 | Same stabilization |
| P1 ref vs P2 dis | Near -1.0 to 0.0 | Maximum difference; exact value depends on C1, C2 |
| P4 single-pixel diff | Near 1.0 | Minimal distortion |

**Required behavior:** SSIM must never crash. Score must be in `[-1.0, 1.0]`. The stabilization constants C1, C2 (derived from `SSIM_K1` and `SSIM_K2`) prevent division by zero for constant images.

### 5.5 Motion Expected Scores

| Scenario | Expected `VMAF_integer_feature_motion2_score` | Rationale |
|---|---|---|
| T1: Single frame | 0.0 at index 0 | No previous frame to compare; code emits 0.0 for index 0 |
| T2: Two identical frames | 0.0 at all indices | SAD between blurred frames = 0 |
| T3: Two maximally different frames | > 0.0, finite | Non-zero SAD between all-zero and all-max blurred frames |
| Any pattern, single frame | 0.0 | Motion is undefined for a single frame |

**Required behavior:** Motion must never crash. Score must be >= 0.0 and finite. The flush function must be called correctly after the last frame.

### 5.6 CAMBI Expected Scores

| Pattern Pair | Expected `cambi` | Rationale |
|---|---|---|
| P1 black frame | 0.0 or near 0.0 | No banding in a constant image |
| P2 white frame | 0.0 or near 0.0 | No banding in a constant image |
| P5 gradient | > 0.0 | Gradients can exhibit banding after quantization |
| P7 checkerboard | >= 0.0 | High-frequency pattern; implementation-specific |

**CAMBI minimum dimension constraint:** CAMBI requires `enc_width >= 216` OR `enc_height >= 216` (see `CAMBI_MIN_WIDTH_HEIGHT`). Images smaller than this must return `-EINVAL` from `init()`. Degenerate dimension tests D1-D8 (1x1 through 17x17) must confirm this error code rather than attempting extraction.

**Required behavior:** CAMBI must never crash. For images meeting the minimum dimension constraint, score must be >= 0.0 and finite.

### 5.7 VMAF Composite Score

| Pattern Pair | Expected `vmaf` | Rationale |
|---|---|---|
| P3 identical ref=dis (large enough for all features) | Near 100.0 (within model clip range) | All constituent features at perfect score |
| P1 ref=dis (both black) | Finite, no crash | Depends on model handling of edge-case sub-scores |
| P1 ref vs P2 dis | Near 0.0 (within model clip range) | All features indicate maximum distortion |

**Required behavior:** VMAF must never crash. Score must be finite. Score clipping to `[0, 100]` depends on model flags; with default clipping, score must be in `[0, 100]`.

---

## 6. Expected Error Handling

### 6.1 Acceptable vs. Unacceptable Outcomes

| Outcome | Acceptable? | Notes |
|---|---|---|
| Correct finite score | Yes | Preferred outcome |
| Finite score with documented reduced accuracy | Yes | e.g., VIF on constant frames may return 0.0 instead of 1.0 |
| `NaN` or `Inf` score stored in feature collector | **Conditional** | Must be documented; must not propagate as a crash |
| Error code from `init()` (e.g., `-EINVAL`) | Yes | For dimensions below feature minimum |
| Error code from `extract()` | Yes | For unsupported configurations |
| Segmentation fault | **No** | Never acceptable |
| SIGFPE (floating-point exception) | **No** | Never acceptable |
| Heap buffer overflow | **No** | Never acceptable |
| Infinite loop / hang | **No** | Never acceptable |

### 6.2 Crash Detection Strategy

Each test runs the feature extractor inside the normal test process. Since Minunit tests abort on the first failure, crash detection is inherent: if the process exits with a signal, the test fails. For additional safety:

1. **Guard allocations:** Allocate `VmafPicture` buffers with extra guard bytes (via `vmaf_picture_alloc` which already provides 32-byte-aligned buffers) and verify that the allocated picture's metadata (w, h, stride) is consistent after extraction.
2. **Score range checks:** After each extraction, verify the score is finite (not NaN, not Inf) using `isfinite()`.
3. **Return code checks:** Verify that `extract()` returns 0 or a documented negative error code, never an undocumented positive value.

```c
static int score_is_valid(double score) {
    return isfinite(score);
}

#define mu_assert_no_crash(msg, err) \
    mu_assert(msg " returned unexpected error", (err) == 0 || (err) < 0)

#define mu_assert_score_finite(msg, score) \
    mu_assert(msg " produced non-finite score", score_is_valid(score))
```

---

## 7. Test Architecture

### 7.1 Test Level

Tests operate at the **feature-extractor API level**, using the public `vmaf_read_pictures()` / `vmaf_feature_score_at_index()` API and/or the internal `VmafFeatureExtractor` init/extract/flush/close interface. This ensures that:

1. The full feature-extractor pipeline is exercised (init, extract, flush, close).
2. Memory allocation/deallocation within the extractor is tested.
3. CPU dispatch (SIMD selection) occurs normally.

### 7.2 Test File Organization

```
libvmaf/test/
  test_degenerate_common.h       # Synthetic data generators (Section 4)
  test_degenerate_common.c       # Generator implementations
  test_degenerate_psnr.c         # PSNR degenerate input tests
  test_degenerate_vif.c          # VIF degenerate input tests
  test_degenerate_adm.c          # ADM degenerate input tests
  test_degenerate_ssim.c         # SSIM degenerate input tests
  test_degenerate_motion.c       # Motion degenerate input tests
  test_degenerate_cambi.c        # CAMBI degenerate input tests
  test_degenerate_vmaf.c         # VMAF composite score tests
  test_degenerate_dimensions.c   # Dimension edge cases across all metrics
```

Each test file is a standalone executable using the Minunit framework (`test.h`).

### 7.3 Test Structure Pattern

Each per-metric test file follows this structure:

```c
#include "test.h"
#include "test_degenerate_common.h"
#include "libvmaf/libvmaf.h"

static char *test_psnr_black_vs_black(void) {
    int err = 0;
    VmafContext *vmaf;
    VmafConfiguration cfg = { .log_level = VMAF_LOG_LEVEL_NONE };
    err = vmaf_init(&vmaf, cfg);
    mu_assert("vmaf_init failed", !err);

    err = vmaf_use_feature(vmaf, "psnr", NULL);
    mu_assert("vmaf_use_feature(psnr) failed", !err);

    VmafPicture ref, dis;
    err = vmaf_picture_alloc(&ref, VMAF_PIX_FMT_YUV420P, 8, 64, 64);
    mu_assert("ref alloc failed", !err);
    err = vmaf_picture_alloc(&dis, VMAF_PIX_FMT_YUV420P, 8, 64, 64);
    mu_assert("dis alloc failed", !err);

    degen_fill_black(&ref);
    degen_fill_black(&dis);

    err = vmaf_read_pictures(vmaf, &ref, &dis, 0);
    mu_assert("vmaf_read_pictures failed", !err);
    err = vmaf_read_pictures(vmaf, NULL, NULL, 0);  /* flush */
    mu_assert("vmaf_read_pictures flush failed", !err);

    double score;
    err = vmaf_feature_score_at_index(vmaf, "psnr_y", &score, 0);
    mu_assert("vmaf_feature_score_at_index failed", !err);
    mu_assert_score_finite("psnr_y black vs black", score);
    /* PSNR for identical frames should be psnr_max = (6 * 8) + 12 = 60 */
    mu_assert("psnr_y should be capped psnr_max for identical black frames",
              score >= 59.0 && score <= 61.0);

    err = vmaf_close(vmaf);
    mu_assert("vmaf_close failed", !err);
    return NULL;
}

/* ... additional test functions ... */

char *run_tests(void) {
    mu_run_test(test_psnr_black_vs_black);
    /* ... */
    return NULL;
}
```

### 7.4 Parameterized Test Loops

For comprehensive coverage, tests that need to run across multiple dimensions and bit depths use a loop pattern:

```c
typedef struct {
    unsigned w;
    unsigned h;
    const char *label;
} DimensionEntry;

static const DimensionEntry dimensions[] = {
    {   1,    1, "1x1"     },
    {   2,    2, "2x2"     },
    {   3,    3, "3x3"     },
    {   7,    7, "7x7"     },
    {   8,    8, "8x8"     },
    {   9,    9, "9x9"     },
    {  15,   15, "15x15"   },
    {  17,   17, "17x17"   },
    {  64,   64, "64x64"   },
    { 120,   68, "120x68"  },
    { 576,  324, "576x324" },
};

static const unsigned bit_depths[] = { 8, 10, 12, 16 };

static char *test_psnr_identical_all_dimensions(void) {
    for (unsigned d = 0; d < sizeof(dimensions)/sizeof(dimensions[0]); d++) {
        for (unsigned b = 0; b < sizeof(bit_depths)/sizeof(bit_depths[0]); b++) {
            unsigned w = dimensions[d].w;
            unsigned h = dimensions[d].h;
            unsigned bpc = bit_depths[b];

            VmafPicture ref, dis;
            int err = vmaf_picture_alloc(&ref, VMAF_PIX_FMT_YUV420P, bpc, w, h);
            if (err) continue;  /* alloc may fail for invalid dimensions */
            err = vmaf_picture_alloc(&dis, VMAF_PIX_FMT_YUV420P, bpc, w, h);
            if (err) { vmaf_picture_unref(&ref); continue; }

            degen_fill_midgray(&ref);
            degen_fill_midgray(&dis);

            /* Run through VmafContext ... */
            /* Assert score is psnr_max ... */

            vmaf_picture_unref(&ref);
            vmaf_picture_unref(&dis);
        }
    }
    return NULL;
}
```

### 7.5 Large Dimension Tests (4K/8K)

Tests for D13 (4096x2160) and D14 (7680x4320) are separated into their own test functions and marked with a naming convention (e.g., `test_*_stress_4k`, `test_*_stress_8k`). These tests:

1. Allocate large buffers (4K luma at 16-bit = ~67 MB; 8K = ~265 MB).
2. Run only a subset of patterns (P1 black, P3 identical, P6 random) to keep runtime reasonable.
3. May be conditionally compiled or skipped if the system has insufficient memory.

```c
static char *test_vif_identical_4k(void) {
    VmafPicture ref, dis;
    int err = vmaf_picture_alloc(&ref, VMAF_PIX_FMT_YUV420P, 10, 4096, 2160);
    if (err) return NULL;  /* Skip if alloc fails (insufficient memory) */
    err = vmaf_picture_alloc(&dis, VMAF_PIX_FMT_YUV420P, 10, 4096, 2160);
    if (err) { vmaf_picture_unref(&ref); return NULL; }

    degen_fill_midgray(&ref);
    degen_fill_midgray(&dis);

    /* ... run VIF, assert score ~ 1.0 ... */

    vmaf_picture_unref(&ref);
    vmaf_picture_unref(&dis);
    return NULL;
}
```

---

## 8. Dimension Edge Cases

### 8.1 Minimum Dimension Matrix

Not all metrics support all dimensions. The test suite must verify correct behavior at each boundary:

| Dimension | PSNR | VIF | ADM | SSIM | Motion | CAMBI | VMAF |
|-----------|------|-----|-----|------|--------|-------|------|
| 1x1 | Score OK | Error or degenerate score | Error (DWT needs >= 2) | Error or degenerate | Score OK (single frame = 0) | `-EINVAL` | Depends on sub-features |
| 2x2 | Score OK | Degenerate score | Error (DWT needs >= 4 for 2 levels) | Degenerate score | Score OK | `-EINVAL` | Depends |
| 3x3 | Score OK | Degenerate score | Error | Degenerate score | Score OK | `-EINVAL` | Depends |
| 7x7 | Score OK | Score OK | May use C fallback (non-multiple of 8) | Score OK | Score OK | `-EINVAL` | Depends |
| 8x8 | Score OK | Score OK | Score OK (minimum for DWT) | Score OK | Score OK | `-EINVAL` | Depends |
| 576x324 | Score OK | Score OK | Score OK | Score OK | Score OK | Score OK (>= 216) | Score OK |

**For each cell marked "Error":** The test asserts that `init()` or `extract()` returns a negative error code, and no crash occurs.

**For each cell marked "Degenerate score":** The test asserts the score is finite and within the metric's valid range, but does not assert a specific value.

### 8.2 Pixel Format Edge Cases

Additional tests verify behavior with the `YUV400P` (luma-only) pixel format:

- PSNR with `YUV400P`: should produce `psnr_y` only, no chroma scores.
- VIF/ADM/SSIM: operate on luma only; should produce correct scores.
- CAMBI: operates on luma only; should produce correct scores.

---

## 9. Memory Edge Cases

### 9.1 Stride Alignment

`vmaf_picture_alloc()` guarantees 32-byte-aligned strides. Tests must verify that:

1. Odd-width images have padded strides (stride > w * bytes_per_sample).
2. The padding bytes do not affect metric output.
3. Metric extractors do not read beyond the valid pixel region within a row.

### 9.2 Allocation Failure Handling

Tests for very large dimensions (8K at 16-bit) must gracefully handle `vmaf_picture_alloc()` failure:

```c
int err = vmaf_picture_alloc(&pic, VMAF_PIX_FMT_YUV420P, 16, 7680, 4320);
if (err) {
    /* Skip this test -- not enough memory */
    return NULL;
}
```

### 9.3 Double-Free Prevention

After each test, `vmaf_picture_unref()` must be called exactly once per picture. If `vmaf_read_pictures()` takes ownership, the test must not call `vmaf_picture_unref()` again. The test framework must track ownership correctly.

---

## 10. Integration with Build System

### 10.1 Meson Configuration

Add to `libvmaf/test/meson.build`:

```meson
# Degenerate Input Tests
degen_common_src = files('test_degenerate_common.c')

degen_tests = {
    'test_degenerate_psnr':       'test_degenerate_psnr.c',
    'test_degenerate_vif':        'test_degenerate_vif.c',
    'test_degenerate_adm':        'test_degenerate_adm.c',
    'test_degenerate_ssim':       'test_degenerate_ssim.c',
    'test_degenerate_motion':     'test_degenerate_motion.c',
    'test_degenerate_cambi':      'test_degenerate_cambi.c',
    'test_degenerate_vmaf':       'test_degenerate_vmaf.c',
    'test_degenerate_dimensions': 'test_degenerate_dimensions.c',
}

foreach name, src : degen_tests
    exe = executable(name,
        ['test.c', src, degen_common_src],
        include_directories : [libvmaf_inc, test_inc, include_directories('../src/')],
        link_with : get_option('default_library') == 'both' ?
                    libvmaf.get_static_lib() : libvmaf,
        dependencies : [math_lib, cuda_dependency],
    )
    test(name, exe, timeout : 300)
endforeach
```

### 10.2 Timeout Configuration

- Standard dimension tests (up to 576x324): default timeout (30 seconds).
- Large dimension tests (1920x1080): 120-second timeout.
- Stress tests (4K, 8K): 300-second timeout, configured via `test(..., timeout: 300)`.

### 10.3 CI Configuration

The degenerate input tests run as part of `ninja test` on all CI platforms. No additional workflow changes are required beyond adding the meson.build entries.

For memory-constrained CI runners, the 8K stress tests will be skipped automatically when `vmaf_picture_alloc()` fails (see Section 9.2).

---

## 11. Test Execution Matrix

### 11.1 Per-Metric Test Count

Each metric is tested with the applicable cross-product of patterns, dimensions, and bit depths:

| Metric | Patterns | Dimensions | Bit Depths | Temporal Cases | Total Test Points |
|--------|----------|------------|------------|----------------|-------------------|
| PSNR | 10 (P1-P10 as ref=dis) + 3 (cross-pattern pairs) | 11 (D1-D11) | 4 | -- | ~572 |
| VIF | 10 + 3 | 9 (D4-D12, skip D1-D3) | 4 | -- | ~468 |
| ADM | 10 + 3 | 8 (D5-D12) | 4 | -- | ~416 |
| SSIM | 10 + 3 | 11 (D1-D11) | 4 | -- | ~572 |
| Motion | 6 (subset of patterns) | 6 (D5-D10) | 4 | 3 (T1-T3) | ~216 |
| CAMBI | 6 (subset) | 4 (D9-D12) | 4 | -- | ~96 |
| VMAF | 4 (P1, P3, P4, P6) | 3 (D9-D11) | 2 (8, 10) | -- | ~24 |
| Dimensions | 2 (P1, P3) | 14 (D1-D14) | 4 | -- | ~112 |

**Total test points: ~2,476.** The dimension edge-case tests (Section 8) contribute an additional ~112 targeted checks.

### 11.2 Cross-Pattern Pairs

In addition to ref=dis (identical pattern) tests, the following cross-pattern pairs are tested for each metric:

| Pair | Ref Pattern | Dis Pattern | Purpose |
|------|------------|-------------|---------|
| X1 | P1 (black) | P2 (white) | Maximum possible distortion |
| X2 | P5 (gradient) | P6 (random) | Structured vs. unstructured |
| X3 | P8 (mid-gray) | P4 (single-pixel diff) | Minimum detectable distortion |

---

## 12. Completion Requirements

The degenerate input test suite is **complete** when ALL of the following requirements are met:

### R1. Generator Coverage
All 10 synthetic data patterns (P1-P10) from Section 3.1 are implemented in `test_degenerate_common.h` / `test_degenerate_common.c` with the function signatures specified in Section 4.1.

**Verification:** Code review confirms all 10 `degen_fill_*` functions exist and produce the specified patterns.

### R2. Metric Coverage
All 7 feature extractors listed in Section 2 (`psnr`, `vif`, `adm`, `ssim`, `motion`, `cambi`, and composite `vmaf`) have dedicated test files with at least one test per pattern from Section 3.1.

**Verification:** Each test file contains `mu_run_test()` calls covering all applicable patterns for that metric. No metric is left untested.

### R3. No-Crash Guarantee
Every degenerate input test verifies that the feature extractor does not crash (segfault, SIGFPE, abort). The test process must exit cleanly for all pattern/dimension/bit-depth combinations.

**Verification:** Full test suite passes on all CI platforms with exit code 0. No test triggers a signal-based termination.

### R4. Score Validity Checks
Every test that successfully extracts a score verifies the score is finite (not NaN, not Inf) using `isfinite()`. Tests for known-value patterns (P3 identical, P1 vs P2 max distortion) verify the score is within the expected range documented in Section 5.

**Verification:** Code review confirms every `vmaf_feature_score_at_index()` call is followed by a `mu_assert_score_finite()` or equivalent check.

### R5. Dimension Edge Cases
Tests for dimensions D1-D8 (1x1 through 17x17) exist for every metric. For metrics that cannot handle small dimensions (ADM below 8x8, CAMBI below 216x216), the test asserts a negative error code from `init()` or `extract()`.

**Verification:** The dimension edge-case test file (`test_degenerate_dimensions.c`) contains a parameterized test that exercises all 14 dimensions from Section 3.2 against all 7 metrics.

### R6. Bit Depth Coverage
All tests run across 4 bit depths (8, 10, 12, 16). The generator functions correctly produce values clamped to `[0, (1 << bpc) - 1]` for each bit depth.

**Verification:** Each parameterized test loop iterates over all 4 bit depths. Generator output is verified by spot-checking pixel values against expected ranges.

### R7. Temporal Edge Cases
The Motion feature extractor is tested with single-frame input (T1), two identical frames (T2), and two maximally different frames (T3) as specified in Section 3.4.

**Verification:** `test_degenerate_motion.c` contains explicit test functions for T1, T2, and T3.

### R8. Error Code Consistency
When a feature extractor rejects a degenerate input (e.g., CAMBI with a 3x3 image), it returns a negative errno code (e.g., `-EINVAL`). Tests assert the specific error code, not just "non-zero."

**Verification:** Code review confirms tests check `err == -EINVAL` (or equivalent documented error) for known rejection cases.

### R9. Memory Safety
No test leaks memory. Every `vmaf_picture_alloc()` is paired with exactly one `vmaf_picture_unref()` (or ownership transfer to `vmaf_read_pictures()`). Every `vmaf_init()` is paired with `vmaf_close()`.

**Verification:** Run the full test suite under Valgrind (or ASan) with zero reported leaks or errors.

### R10. Build Integration
All test executables compile and link correctly as part of `ninja test` on all CI platforms (Ubuntu x86_64, Ubuntu ARM64, macOS x86_64, Windows x86_64).

**Verification:** CI pipeline passes with the new tests on all platforms.

### R11. Large Dimension Tests
At least one test exercises 4K (4096x2160) and one exercises 8K (7680x4320) with the PSNR and VIF extractors. These tests gracefully skip when memory allocation fails.

**Verification:** On CI runners with sufficient memory, the 4K and 8K tests pass. On memory-constrained runners, they skip without failure.

### R12. Deterministic Reproducibility
All tests are fully deterministic. PRNG seeds are hardcoded (seed=42). Running the same test twice produces identical results. No dependency on wall-clock time, system load, or thread scheduling.

**Verification:** Run the test suite twice and confirm identical pass/fail results.

### R13. Graceful NaN/Inf Documentation
If any metric produces NaN or Inf for a specific degenerate input (e.g., VIF denominator = 0 for constant frames), this behavior is explicitly documented in the test with a comment explaining why it is acceptable, and the test asserts the specific non-finite value rather than silently accepting it.

**Verification:** Code review confirms all non-finite score assertions are accompanied by explanatory comments.

---

## 13. Acceptance Criteria Summary

| Req | Description | Pass Condition |
|-----|-------------|----------------|
| R1 | Generator coverage | 10/10 `degen_fill_*` functions implemented |
| R2 | Metric coverage | 7/7 feature extractors have test files with full pattern coverage |
| R3 | No-crash guarantee | Zero signal-based terminations across all tests |
| R4 | Score validity | Every extracted score checked with `isfinite()`; known values in expected ranges |
| R5 | Dimension edge cases | All 14 dimensions tested against all 7 metrics |
| R6 | Bit depth coverage | All tests run at 8, 10, 12, and 16 bpc |
| R7 | Temporal edge cases | Motion tested with T1, T2, T3 scenarios |
| R8 | Error code consistency | Rejection cases assert specific negative errno codes |
| R9 | Memory safety | Zero leaks/errors under Valgrind/ASan |
| R10 | Build integration | Tests pass on all 4 CI platforms |
| R11 | Large dimension tests | 4K and 8K tests exist and pass or gracefully skip |
| R12 | Deterministic | Repeated runs produce identical results |
| R13 | NaN/Inf documentation | All non-finite outcomes documented and explicitly asserted |

All 13 requirements must be met for the degenerate input test suite to be considered complete.
