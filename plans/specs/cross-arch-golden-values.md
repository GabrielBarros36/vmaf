# Cross-Architecture Golden Value Tests — Specification

## 1. Objective

Establish a set of **canonical golden values** — metric scores computed using the C reference code path (no SIMD) — and validate on every CI platform that SIMD-enabled builds reproduce these scores within defined tolerances. Any platform that diverges fails the test explicitly. No silent drift is permitted.

This complements the SIMD Correctness Oracle (Section 3.3) which tests individual dispatch functions in isolation. Golden value tests operate at the **full metric pipeline** level: they exercise model loading, feature extraction, score aggregation, and prediction end-to-end, catching integration-level regressions that per-function oracle tests would miss.

---

## 2. Metrics Under Test

| # | Metric | Feature Extractor Name | Integer Path | Float Path | Score Key |
|---|--------|----------------------|--------------|------------|-----------|
| 1 | VMAF | `vmaf` (integer features) | Yes | — | `vmaf` |
| 2 | VIF (per-scale) | `vif` | Yes | — | `vif_scale0` .. `vif_scale3` |
| 3 | ADM2 | `adm` | Yes | — | `adm2` |
| 4 | Motion2 | `motion` | Yes | — | `motion2` |
| 5 | PSNR | `psnr` | Yes | — | `psnr_y`, `psnr_cb`, `psnr_cr` |
| 6 | Float SSIM | `float_ssim` | — | Yes | `float_ssim` |
| 7 | Float MS-SSIM | `float_ms_ssim` | — | Yes | `float_ms_ssim` |
| 8 | Float PSNR | `float_psnr` | — | Yes | `float_psnr` |
| 9 | CAMBI | `cambi` | Yes | — | `cambi` |

**Note:** Integer VIF/ADM/Motion paths use integer arithmetic internally and produce scores via `float` division at the end. Float SSIM/MS-SSIM/PSNR paths use `float`/`double` arithmetic throughout. The tolerance regime differs accordingly (see Section 6).

---

## 3. Test Videos

Golden value computation uses the existing test videos already committed to the repository. These are small enough for CI yet exercise real codec artifacts and content diversity.

### 3.1 Primary Test Pair (576x324, 8-bit YUV420)

| Role | File | Resolution | Bit Depth | Format | Frames |
|------|------|-----------|-----------|--------|--------|
| Reference | `src01_hrc00_576x324.yuv` | 576x324 | 8 | YUV420P | 48 |
| Distorted | `src01_hrc01_576x324.yuv` | 576x324 | 8 | YUV420P | 48 |

**Location:** `python/test/resource/yuv/`

This is the primary test pair used by the existing Python test suite. It exercises the standard integer feature extraction pipeline and produces well-known reference scores.

### 3.2 10-bit Test Pair (576x324, 10-bit YUV420)

| Role | File | Resolution | Bit Depth | Format |
|------|------|-----------|-----------|--------|
| Reference | `src01_hrc00_576x324.yuv420p10le.yuv` | 576x324 | 10 | YUV420P10LE |
| Distorted | `src01_hrc01_576x324.yuv420p10le.yuv` | 576x324 | 10 | YUV420P10LE |

### 3.3 12-bit Test Pair (576x324, 12-bit YUV420)

| Role | File | Resolution | Bit Depth | Format |
|------|------|-----------|-----------|--------|
| Reference | `src01_hrc00_576x324.yuv420p12le.yuv` | 576x324 | 12 | YUV420P12LE |
| Distorted | `src01_hrc01_576x324.yuv420p12le.yuv` | 576x324 | 12 | YUV420P12LE |

### 3.4 Identity Test (reference vs. itself)

The 8-bit reference (`src01_hrc00_576x324.yuv`) compared against itself. This validates perfect-score edge cases: VIF should be ~1.0, ADM2 should be 1.0, PSNR should be the maximum (60.0 for 8-bit capped), SSIM should be 1.0.

### 3.5 CAMBI Test Video

| Role | File | Resolution | Bit Depth | Format |
|------|------|-----------|-----------|--------|
| No-ref input | `src01_hrc01_576x324.yuv` | 576x324 | 8 | YUV420P |

CAMBI is a no-reference metric; only the distorted video is needed.

---

## 4. Models Under Test

| # | Model File | Description | Used With Test Pair |
|---|-----------|-------------|---------------------|
| 1 | `vmaf_v0.6.1.json` | Standard VMAF (integer features) | 8-bit, 10-bit, 12-bit |
| 2 | `vmaf_v0.6.1neg.json` | VMAF NEG (enhancement gain limiting) | 8-bit |
| 3 | `vmaf_4k_v0.6.1.json` | VMAF 4K model | 8-bit |
| 4 | `vmaf_float_v0.6.1.json` | VMAF using float features | 8-bit |
| 5 | `vmaf_b_v0.6.3.json` | VMAF v0.6.3 (integer features) | 8-bit |

**Location:** `model/`

All models listed are already shipped in the repository and used by existing Python tests. Adding them to the golden value set locks down their expected output across architectures.

---

## 5. Canonical Golden Value Computation

### 5.1 Build Configuration

Golden values are computed using a build that **disables all SIMD assembly**:

```bash
meson setup libvmaf build_ref \
    --buildtype release \
    -Denable_float=true \
    -Denable_asm=false

ninja -C build_ref
```

The `-Denable_asm=false` flag forces all dispatch function pointers to resolve to C reference implementations. This ensures the golden values are architecture-independent — the same C code produces the same results on x86-64, ARM64, and any future target.

### 5.2 Score Generation

A dedicated tool `generate_golden_values` (built only in the test target) invokes the libvmaf API for each test-pair/model combination and emits a JSON file.

```c
// Pseudocode for golden value generation
VmafContext *vmaf;
VmafConfiguration cfg = {
    .log_level = VMAF_LOG_LEVEL_NONE,
    .n_threads = 1,  // Single-threaded for determinism
    .n_subsample = 1,
};
vmaf_init(&vmaf, cfg);

// Load model
VmafModel *model;
VmafModelConfig model_cfg = { .name = "vmaf", .flags = VMAF_MODEL_FLAGS_DEFAULT };
vmaf_model_load(&model, &model_cfg, "vmaf_v0.6.1");
vmaf_use_features_from_model(vmaf, model);

// Also register standalone feature extractors for metrics not in the model
vmaf_use_feature(vmaf, "psnr", NULL);
vmaf_use_feature(vmaf, "float_ssim", NULL);
vmaf_use_feature(vmaf, "float_ms_ssim", NULL);
vmaf_use_feature(vmaf, "float_psnr", NULL);
vmaf_use_feature(vmaf, "cambi", NULL);

// Read frames, call vmaf_read_pictures, then extract scores
// Write to JSON
```

### 5.3 Determinism Requirements

- **Single-threaded:** `n_threads = 1` eliminates thread-scheduling non-determinism.
- **No subsampling:** `n_subsample = 1` processes every frame.
- **Stable libvmaf version:** Golden values are regenerated only when the C reference code or model files change. The regeneration commit message must state "Regenerate golden values" and explain the reason.

---

## 6. Tolerance Definitions

### 6.1 Tolerance Table

| Metric Category | Path Type | Tolerance | Rationale |
|-----------------|-----------|-----------|-----------|
| VIF (integer) | Integer arithmetic with float division at end | Absolute: 1e-6 | Integer accumulation is deterministic; only the final float division can differ by rounding |
| ADM2 (integer) | Integer arithmetic with float division at end | Absolute: 1e-6 | Same as VIF |
| Motion2 (integer) | Integer convolution + float normalization | Absolute: 1e-6 | Same as VIF |
| PSNR (integer) | Integer MSE + float log10 | Absolute: 1e-4 | Log10 amplifies small differences in the denominator |
| VMAF (composite) | SVM prediction over integer features | Absolute: 1e-4 | Combines multiple features through SVM; error accumulates |
| Float SSIM | Float throughout | Absolute: 1e-4 | Non-associative float addition in SIMD reductions |
| Float MS-SSIM | Float throughout | Absolute: 1e-4 | Same as SSIM, compounded across scales |
| Float PSNR | Float throughout | Absolute: 1e-4 | Float accumulation differences |
| CAMBI | Integer + float pooling | Absolute: 1e-4 | Integer banding detection with float spatial pooling |

### 6.2 Comparison Function

```c
#include <math.h>

/* Returns 1 if values match within tolerance, 0 otherwise. */
static int score_within_tolerance(double golden, double actual, double atol) {
    if (golden == actual) return 1;  /* Exact match, handles +/-inf */
    return fabs(golden - actual) <= atol;
}

/* For per-frame comparisons: use relative tolerance when values are large. */
static int score_within_tolerance_rel(double golden, double actual,
                                       double atol, double rtol) {
    if (golden == actual) return 1;
    double abs_diff = fabs(golden - actual);
    if (abs_diff <= atol) return 1;
    double max_abs = fmax(fabs(golden), fabs(actual));
    if (max_abs < 1e-15) return 1;  /* Both effectively zero */
    return (abs_diff / max_abs) <= rtol;
}
```

### 6.3 Zero-Tolerance Aspirational Target

For integer-path metrics (VIF, ADM2, Motion2), the long-term goal is **exact match** (tolerance = 0) between the C reference and all SIMD paths. The 1e-6 tolerance exists only to accommodate the final float division. If a platform-specific divergence exceeds 1e-6 for these metrics, it indicates a bug in the SIMD implementation, not an expected floating-point artifact.

---

## 7. Golden Value File Format

### 7.1 File Location

```
libvmaf/test/golden_values/
    golden_values.json
```

A single JSON file containing all golden values, checked into the repository.

### 7.2 JSON Schema

```json
{
    "version": 1,
    "generated_with": {
        "build_flags": "-Denable_asm=false -Denable_float=true",
        "commit": "abc123def456",
        "date": "2026-03-08"
    },
    "test_cases": [
        {
            "id": "vmaf_8bit_576x324",
            "description": "VMAF v0.6.1 on 8-bit 576x324 src01 ref vs. dis",
            "ref_video": "src01_hrc00_576x324.yuv",
            "dis_video": "src01_hrc01_576x324.yuv",
            "width": 576,
            "height": 324,
            "pix_fmt": "yuv420p",
            "bit_depth": 8,
            "model": "vmaf_v0.6.1",
            "n_frames": 48,
            "scores": {
                "vmaf": {
                    "mean": 76.66890519623612,
                    "per_frame": [75.06, 75.45, ...],
                    "tolerance": 1e-4
                },
                "vif_scale0": {
                    "mean": 0.3636620710647402,
                    "tolerance": 1e-6
                },
                "vif_scale1": {
                    "mean": 0.7674952820232231,
                    "tolerance": 1e-6
                },
                "vif_scale2": {
                    "mean": 0.8631077727416296,
                    "tolerance": 1e-6
                },
                "vif_scale3": {
                    "mean": 0.9157200890843669,
                    "tolerance": 1e-6
                },
                "adm2": {
                    "mean": 0.9345149030293786,
                    "tolerance": 1e-6
                },
                "motion2": {
                    "mean": 3.8953518541666665,
                    "tolerance": 1e-6
                },
                "psnr_y": {
                    "mean": 30.755063979166668,
                    "tolerance": 1e-4
                },
                "psnr_cb": {
                    "mean": 38.4494410625,
                    "tolerance": 1e-4
                },
                "psnr_cr": {
                    "mean": 40.99191027083334,
                    "tolerance": 1e-4
                },
                "float_ssim": {
                    "mean": 0.86322654166666657,
                    "tolerance": 1e-4
                },
                "float_ms_ssim": {
                    "mean": 0.9632406874999999,
                    "tolerance": 1e-4
                },
                "float_psnr": {
                    "mean": 30.7550666667,
                    "tolerance": 1e-4
                }
            }
        },
        {
            "id": "vmaf_8bit_576x324_identity",
            "description": "VMAF v0.6.1 on 8-bit 576x324 ref vs. ref (identity)",
            "ref_video": "src01_hrc00_576x324.yuv",
            "dis_video": "src01_hrc00_576x324.yuv",
            "width": 576,
            "height": 324,
            "pix_fmt": "yuv420p",
            "bit_depth": 8,
            "model": "vmaf_v0.6.1",
            "n_frames": 48,
            "scores": {
                "vmaf": {
                    "mean": 99.946416604585025,
                    "tolerance": 1e-4
                },
                "adm2": {
                    "mean": 1.0,
                    "tolerance": 1e-6
                },
                "float_ssim": {
                    "mean": 1.0,
                    "tolerance": 1e-6
                },
                "float_ms_ssim": {
                    "mean": 1.0,
                    "tolerance": 1e-6
                },
                "float_psnr": {
                    "mean": 60.0,
                    "tolerance": 1e-4
                }
            }
        },
        {
            "id": "vmaf_10bit_576x324",
            "description": "VMAF v0.6.1 on 10-bit 576x324 src01 ref vs. dis",
            "ref_video": "src01_hrc00_576x324.yuv420p10le.yuv",
            "dis_video": "src01_hrc01_576x324.yuv420p10le.yuv",
            "width": 576,
            "height": 324,
            "pix_fmt": "yuv420p10le",
            "bit_depth": 10,
            "model": "vmaf_v0.6.1",
            "n_frames": 3,
            "scores": {
                "vmaf": {
                    "mean": 82.56523033333333,
                    "tolerance": 1e-4
                },
                "adm2": {
                    "mean": 0.9517763333333334,
                    "tolerance": 1e-6
                },
                "motion2": {
                    "mean": 2.8104600000000004,
                    "tolerance": 1e-6
                }
            }
        },
        {
            "id": "vmaf_12bit_576x324",
            "description": "VMAF v0.6.1 on 12-bit 576x324 src01 ref vs. dis",
            "ref_video": "src01_hrc00_576x324.yuv420p12le.yuv",
            "dis_video": "src01_hrc01_576x324.yuv420p12le.yuv",
            "width": 576,
            "height": 324,
            "pix_fmt": "yuv420p12le",
            "bit_depth": 12,
            "model": "vmaf_v0.6.1",
            "n_frames": 3,
            "scores": {
                "vmaf": {
                    "mean": 82.56523033333333,
                    "tolerance": 1e-4
                },
                "adm2": {
                    "mean": 0.9517763333333334,
                    "tolerance": 1e-6
                },
                "motion2": {
                    "mean": 2.8104600000000004,
                    "tolerance": 1e-6
                }
            }
        },
        {
            "id": "vmaf_neg_8bit_576x324",
            "description": "VMAF NEG (v0.6.1neg) on 8-bit 576x324 src01 ref vs. dis",
            "ref_video": "src01_hrc00_576x324.yuv",
            "dis_video": "src01_hrc01_576x324.yuv",
            "width": 576,
            "height": 324,
            "pix_fmt": "yuv420p",
            "bit_depth": 8,
            "model": "vmaf_v0.6.1neg",
            "n_frames": 48,
            "scores": {
                "vmaf": {
                    "mean": 0.0,
                    "tolerance": 1e-4,
                    "comment": "Placeholder: generate from C reference build"
                }
            }
        },
        {
            "id": "vmaf_4k_8bit_576x324",
            "description": "VMAF 4K on 8-bit 576x324 src01 ref vs. dis",
            "ref_video": "src01_hrc00_576x324.yuv",
            "dis_video": "src01_hrc01_576x324.yuv",
            "width": 576,
            "height": 324,
            "pix_fmt": "yuv420p",
            "bit_depth": 8,
            "model": "vmaf_4k_v0.6.1",
            "n_frames": 48,
            "scores": {
                "vmaf": {
                    "mean": 84.95064735416668,
                    "tolerance": 1e-4
                }
            }
        },
        {
            "id": "cambi_8bit_576x324",
            "description": "CAMBI on 8-bit 576x324 src01 distorted",
            "ref_video": null,
            "dis_video": "src01_hrc01_576x324.yuv",
            "width": 576,
            "height": 324,
            "pix_fmt": "yuv420p",
            "bit_depth": 8,
            "model": null,
            "n_frames": 48,
            "scores": {
                "cambi": {
                    "mean": 0.25968416666666666,
                    "tolerance": 1e-4
                }
            }
        }
    ]
}
```

### 7.3 Per-Frame vs. Mean Scores

The golden value file stores **mean scores** (averaged across all frames) as the primary comparison target. Optionally, per-frame scores may be included for debugging but are not required for the pass/fail check. Mean-score comparison is sufficient because:

1. It matches the existing Python test methodology (`assertAlmostEqual` on mean scores).
2. Per-frame deviations that cancel out in the mean would still be caught by the SIMD oracle tests (Section 3.3 of the plan), which test individual functions.
3. If per-frame validation is desired later, it can be added without changing the file format (the `per_frame` array is already in the schema).

---

## 8. Test Implementation

### 8.1 File Organization

```
libvmaf/test/
    test_golden_values.c         # Golden value validation test
    golden_values/
        golden_values.json       # Canonical reference scores
    tools/
        generate_golden.c        # Golden value generator (build tool, not test)
```

### 8.2 Test Program Structure

The test program `test_golden_values.c` follows the existing Minunit pattern:

```c
#include "test.h"
#include "libvmaf/libvmaf.h"
#include "libvmaf/model.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Tolerance constants */
#define TOL_INTEGER_FEAT  1e-6
#define TOL_FLOAT_FEAT    1e-4
#define TOL_VMAF_SCORE    1e-4
#define TOL_PSNR          1e-4
#define TOL_CAMBI         1e-4

/* Embedded golden values (compiled in, no runtime JSON parsing needed) */
typedef struct {
    const char *name;
    double golden_mean;
    double tolerance;
} GoldenScore;

typedef struct {
    const char *id;
    const char *ref_path;
    const char *dis_path;
    unsigned width;
    unsigned height;
    int bit_depth;
    enum VmafPixelFormat pix_fmt;
    const char *model_name;
    unsigned n_frames;
    const GoldenScore *scores;
    unsigned n_scores;
} GoldenTestCase;

static int score_within_tolerance(double golden, double actual, double atol) {
    if (golden == actual) return 1;
    return fabs(golden - actual) <= atol;
}

static char *run_golden_test(const GoldenTestCase *tc) {
    int err = 0;
    VmafContext *vmaf;
    VmafConfiguration cfg = {
        .log_level = VMAF_LOG_LEVEL_NONE,
        .n_threads = 1,
        .n_subsample = 1,
    };
    err = vmaf_init(&vmaf, cfg);
    mu_assert("vmaf_init failed", !err);

    /* Load model if specified */
    VmafModel *model = NULL;
    if (tc->model_name) {
        VmafModelConfig model_cfg = {
            .name = "golden_test",
            .flags = VMAF_MODEL_FLAGS_DEFAULT,
        };
        err = vmaf_model_load(&model, &model_cfg, tc->model_name);
        mu_assert("vmaf_model_load failed", !err);
        err = vmaf_use_features_from_model(vmaf, model);
        mu_assert("vmaf_use_features_from_model failed", !err);
    }

    /* Register additional feature extractors */
    vmaf_use_feature(vmaf, "psnr", NULL);
    vmaf_use_feature(vmaf, "float_ssim", NULL);
    vmaf_use_feature(vmaf, "float_ms_ssim", NULL);
    vmaf_use_feature(vmaf, "float_psnr", NULL);
    if (!tc->ref_path) {
        vmaf_use_feature(vmaf, "cambi", NULL);
    }

    /* Read YUV frames and feed to vmaf_read_pictures() */
    /* ... (YUV reader implementation) ... */

    /* Extract and compare scores */
    for (unsigned i = 0; i < tc->n_scores; i++) {
        double score;
        err = vmaf_score_pooled(vmaf, model, VMAF_POOL_METHOD_MEAN,
                                &score, 0, tc->n_frames - 1);
        /* Or use vmaf_feature_score_pooled for individual features */

        if (!score_within_tolerance(tc->scores[i].golden_mean,
                                     score, tc->scores[i].tolerance)) {
            fprintf(stderr,
                "\n  GOLDEN VALUE MISMATCH: %s\n"
                "    Test case:  %s\n"
                "    Metric:     %s\n"
                "    Expected:   %.15g\n"
                "    Actual:     %.15g\n"
                "    Difference: %.15g\n"
                "    Tolerance:  %.15g\n",
                tc->id, tc->id, tc->scores[i].name,
                tc->scores[i].golden_mean, score,
                fabs(tc->scores[i].golden_mean - score),
                tc->scores[i].tolerance);
            mu_assert("golden value mismatch", 0);
        }
    }

    if (model) vmaf_model_destroy(model);
    vmaf_close(vmaf);
    return NULL;
}

/* Test functions — one per test case */
static char *test_golden_vmaf_8bit(void) {
    static const GoldenScore scores[] = {
        { "vmaf",       76.66890519623612,     TOL_VMAF_SCORE },
        { "vif_scale0", 0.3636620710647402,    TOL_INTEGER_FEAT },
        { "vif_scale1", 0.7674952820232231,    TOL_INTEGER_FEAT },
        { "vif_scale2", 0.8631077727416296,    TOL_INTEGER_FEAT },
        { "vif_scale3", 0.9157200890843669,    TOL_INTEGER_FEAT },
        { "adm2",       0.9345149030293786,    TOL_INTEGER_FEAT },
        { "motion2",    3.8953518541666665,    TOL_INTEGER_FEAT },
        { "psnr_y",     30.755063979166668,    TOL_PSNR },
        { "float_ssim", 0.86322654166666657,   TOL_FLOAT_FEAT },
        { "float_ms_ssim", 0.9632406874999999, TOL_FLOAT_FEAT },
    };
    static const GoldenTestCase tc = {
        .id = "vmaf_8bit_576x324",
        .ref_path = "src01_hrc00_576x324.yuv",
        .dis_path = "src01_hrc01_576x324.yuv",
        .width = 576, .height = 324,
        .bit_depth = 8,
        .pix_fmt = VMAF_PIX_FMT_YUV420P,
        .model_name = "vmaf_v0.6.1",
        .n_frames = 48,
        .scores = scores,
        .n_scores = sizeof(scores) / sizeof(scores[0]),
    };
    return run_golden_test(&tc);
}

char *run_tests(void) {
    mu_run_test(test_golden_vmaf_8bit);
    /* mu_run_test(test_golden_vmaf_8bit_identity); */
    /* mu_run_test(test_golden_vmaf_10bit); */
    /* mu_run_test(test_golden_vmaf_12bit); */
    /* mu_run_test(test_golden_vmaf_neg); */
    /* mu_run_test(test_golden_vmaf_4k); */
    /* mu_run_test(test_golden_cambi); */
    return NULL;
}
```

### 8.3 YUV Reader

The test program needs a minimal YUV file reader to feed frames to the libvmaf API. Two implementation options:

**Option A (preferred): Use the vmaf CLI's existing y4m/yuv reader.** Link against the `y4m_input` and YUV reader code already in `libvmaf/tools/`. This avoids duplicating frame-reading logic.

**Option B: Use vmaf_read_pictures with pre-allocated VmafPicture buffers.** Read raw YUV data from disk, populate `VmafPicture` structs, and pass them via `vmaf_read_pictures()`. This approach is self-contained but requires handling pixel format and stride details.

### 8.4 Alternative: Reuse vmaf CLI as a Subprocess

Instead of a C test program, the golden value test can invoke the `vmaf` CLI binary and parse its JSON output:

```bash
vmaf --reference ref.yuv --distorted dis.yuv \
     --width 576 --height 324 --pixel_format 420 --bitdepth 8 \
     --model vmaf_v0.6.1 \
     --feature psnr --feature float_ssim --feature float_ms_ssim \
     --json --output /dev/stdout
```

The test script (or C program) then parses the JSON and compares each score to the golden value. This approach has the advantage of testing the full CLI pipeline, but introduces a dependency on the CLI binary being built and available at test time.

**Recommendation:** Implement as a C test program (Option A/B) for consistency with the existing Minunit test suite, and add a separate Python-based CLI validation test for additional coverage.

---

## 9. Build System Integration

### 9.1 Meson Configuration

Add to `libvmaf/test/meson.build`:

```meson
# Golden Value Tests
test_video_dir = join_paths(meson.project_source_root(), '../python/test/resource/yuv/')
model_dir = join_paths(meson.project_source_root(), '../model/')
golden_values_dir = join_paths(meson.current_source_dir(), 'golden_values/')

test_golden_values = executable('test_golden_values',
    ['test.c', 'test_golden_values.c'],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/')],
    link_with : get_option('default_library') == 'both' ? libvmaf.get_static_lib() : libvmaf,
    c_args : [
        '-DTEST_VIDEO_DIR="' + test_video_dir + '"',
        '-DMODEL_DIR="' + model_dir + '"',
        '-DGOLDEN_VALUES_DIR="' + golden_values_dir + '"',
    ],
    dependencies : [math_lib, thread_lib, cuda_dependency],
)

test('test_golden_values', test_golden_values, timeout: 300)
```

The 300-second timeout accommodates processing 48 frames across multiple models. On CI runners this typically completes in under 60 seconds.

### 9.2 Test Data Access

Test videos are accessed via compile-time path constants (`TEST_VIDEO_DIR`). The paths point to the existing `python/test/resource/yuv/` directory. This avoids duplicating test video files.

Models are accessed through the standard `vmaf_model_load()` API which searches the built-in model path. The `MODEL_DIR` compile-time constant is provided as a fallback.

---

## 10. CI Integration

### 10.1 Validation on Every Build

The `test_golden_values` test runs as part of `ninja test` on every CI platform:

| Platform | Architecture | SIMD Available | Expected Behavior |
|----------|-------------|---------------|-------------------|
| Ubuntu x86_64 | x86-64 | AVX2 (+AVX-512 if available) | Scores must match golden values within tolerance |
| Ubuntu ARM64 | AArch64 | NEON | Scores must match golden values within tolerance |
| macOS x86_64 | x86-64 | AVX2 | Scores must match golden values within tolerance |
| macOS ARM64 | AArch64 | NEON | Scores must match golden values within tolerance |
| Windows x86_64 | x86-64 | AVX2 | Scores must match golden values within tolerance |

### 10.2 Forced-Fallback CI Job

The existing `simd-oracle-fallback` job in `.github/workflows/simd-oracle.yml` already builds with `-Denable_asm=false` and runs the full test suite. The golden value test is included automatically.

When run with `-Denable_asm=false`, the golden value test serves as a **tautological check**: the C reference path is computing scores and comparing them against golden values that were computed by the same C reference path. This verifies that:

1. The golden values are consistent with the current C reference code.
2. No compiler optimization has introduced a divergence (e.g., `-ffast-math` was not inadvertently enabled).
3. The test infrastructure itself is working correctly.

### 10.3 Golden Value Regeneration

**When to regenerate:**
- A C reference implementation is intentionally modified (bug fix, algorithm change).
- A model file is updated.
- A new metric or test video is added.

**How to regenerate:**

```bash
# Build with ASM disabled
meson setup libvmaf build_ref --buildtype release \
    -Denable_float=true -Denable_asm=false
ninja -C build_ref

# Run the generator
build_ref/test/generate_golden_values > libvmaf/test/golden_values/golden_values.json

# Verify the new golden values pass on the C reference build
ninja -C build_ref test

# Commit with explanation
git add libvmaf/test/golden_values/golden_values.json
git commit -m "Regenerate golden values: <reason>"
```

**Who can regenerate:** Any contributor, but the PR must be reviewed by at least one maintainer who verifies that the score changes are expected and explained.

### 10.4 Workflow Addition

Add the golden value test to the existing `simd-oracle.yml` workflow:

```yaml
  golden-values-x86:
    name: Golden Value Tests (x86_64)
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v6
      - name: Install dependencies
        run: |
          pip install meson
          sudo apt-get update
          sudo apt-get -yq install ninja-build nasm gcc g++

      - name: Configure and build
        run: |
          meson setup libvmaf libvmaf/build --buildtype release \
              -Denable_float=true
          ninja -C libvmaf/build

      - name: Run golden value tests
        run: |
          libvmaf/build/test/test_golden_values

  golden-values-arm:
    name: Golden Value Tests (ARM64)
    runs-on: ubuntu-24.04-arm
    steps:
      - uses: actions/checkout@v6
      - name: Install dependencies
        run: |
          pip install meson
          sudo apt-get update
          sudo apt-get -yq install ninja-build gcc g++

      - name: Configure and build
        run: |
          meson setup libvmaf libvmaf/build --buildtype release \
              -Denable_float=true
          ninja -C libvmaf/build

      - name: Run golden value tests
        run: |
          libvmaf/build/test/test_golden_values
```

---

## 11. Failure Diagnostics

### 11.1 Failure Output Format

When a golden value test fails, the output must include all information needed to diagnose the issue without requiring a local reproduction:

```
test_golden_vmaf_8bit: fail
  GOLDEN VALUE MISMATCH: vmaf_8bit_576x324
    Test case:  vmaf_8bit_576x324
    Metric:     vif_scale0
    Expected:   0.363662071064740
    Actual:     0.363662082173651
    Difference: 1.1108911e-08
    Tolerance:  1.0000000e-06
    Platform:   x86_64 AVX2
    Build:      release, enable_asm=true
```

### 11.2 Debugging Workflow

When a golden value test fails on a specific platform:

1. **Check if the SIMD oracle tests also fail.** If yes, the bug is in a specific SIMD function — fix it there.
2. **Check if the failure is within 10x of the tolerance.** If yes, the tolerance may need adjustment, but investigate first.
3. **Run the same test with `-Denable_asm=false` on the failing platform.** If it passes, the issue is SIMD-specific. If it also fails, the issue is in the C reference code or the golden values are stale.
4. **Compare per-frame scores.** Enable per-frame output to identify which frame(s) diverge.
5. **Check compiler version and flags.** Different compilers may produce different rounding behavior for the same C code.

---

## 12. Test Case Matrix

### 12.1 Complete Test Case List

| # | Test Case ID | Ref Video | Dis Video | Model | Bit Depth | Metrics Checked |
|---|-------------|-----------|-----------|-------|-----------|-----------------|
| 1 | `vmaf_8bit_576x324` | src01_hrc00 | src01_hrc01 | vmaf_v0.6.1 | 8 | VMAF, VIF x4, ADM2, Motion2, PSNR x3, SSIM, MS-SSIM |
| 2 | `vmaf_8bit_576x324_identity` | src01_hrc00 | src01_hrc00 | vmaf_v0.6.1 | 8 | VMAF, VIF x4, ADM2, PSNR, SSIM, MS-SSIM |
| 3 | `vmaf_10bit_576x324` | src01_hrc00_10 | src01_hrc01_10 | vmaf_v0.6.1 | 10 | VMAF, VIF x4, ADM2, Motion2 |
| 4 | `vmaf_12bit_576x324` | src01_hrc00_12 | src01_hrc01_12 | vmaf_v0.6.1 | 12 | VMAF, VIF x4, ADM2, Motion2 |
| 5 | `vmaf_neg_8bit_576x324` | src01_hrc00 | src01_hrc01 | vmaf_v0.6.1neg | 8 | VMAF |
| 6 | `vmaf_4k_8bit_576x324` | src01_hrc00 | src01_hrc01 | vmaf_4k_v0.6.1 | 8 | VMAF |
| 7 | `vmaf_float_8bit_576x324` | src01_hrc00 | src01_hrc01 | vmaf_float_v0.6.1 | 8 | VMAF |
| 8 | `cambi_8bit_576x324` | (none) | src01_hrc01 | (none) | 8 | CAMBI |

### 12.2 Total Score Comparisons

| Test Case | Scores | Bit Depths | Total |
|-----------|--------|-----------|-------|
| 1 | 13 | 1 | 13 |
| 2 | 10 | 1 | 10 |
| 3 | 7 | 1 | 7 |
| 4 | 7 | 1 | 7 |
| 5 | 1 | 1 | 1 |
| 6 | 1 | 1 | 1 |
| 7 | 1 | 1 | 1 |
| 8 | 1 | 1 | 1 |
| **Total** | | | **41** |

Each of these 41 comparisons is performed on every CI platform, giving a total of **41 x N_platforms** cross-architecture validation points.

---

## 13. Relationship to Other Test Components

| Component | Scope | Catches |
|-----------|-------|---------|
| **SIMD Oracle Tests (3.3)** | Individual dispatch functions | Bugs in SIMD function implementations |
| **Golden Value Tests (this spec)** | Full metric pipeline | Integration-level regressions, model loading errors, score aggregation bugs, cross-architecture drift |
| **Python Integration Tests** | Full CLI pipeline | End-to-end regressions including I/O, CLI parsing, output formatting |

Golden value tests fill the gap between per-function SIMD validation and full CLI integration testing. They ensure that even if individual functions are correct, the way they are composed into complete metrics produces consistent results.

---

## 14. Completion Requirements

The cross-architecture golden value test harness is **complete** when ALL of the following requirements are met:

### R1. Canonical Golden Values Generated

A `golden_values.json` file exists at `libvmaf/test/golden_values/golden_values.json`, was generated with `-Denable_asm=false`, and contains golden scores for all 8 test cases listed in Section 12.1.

**Verification:** The file exists, is valid JSON, and contains entries for all test case IDs.

### R2. All Metrics Covered

Every metric in Section 2 (VMAF, VIF x4 scales, ADM2, Motion2, PSNR x3 channels, Float SSIM, Float MS-SSIM, Float PSNR, CAMBI) has at least one golden value entry.

**Verification:** Parse the golden values file and confirm each metric name appears at least once.

### R3. All Bit Depths Tested

Golden values exist for 8-bit, 10-bit, and 12-bit inputs (using the test videos in Section 3).

**Verification:** Test case entries exist with `bit_depth` values of 8, 10, and 12.

### R4. All Models Tested

Golden values exist for all models listed in Section 4: `vmaf_v0.6.1`, `vmaf_v0.6.1neg`, `vmaf_4k_v0.6.1`, `vmaf_float_v0.6.1`, and `vmaf_b_v0.6.3`.

**Verification:** Test case entries exist referencing each model name.

### R5. Tolerance Enforcement

The test program applies the tolerance values from Section 6.1. Integer-path features use 1e-6. Float-path features and composite scores use 1e-4. No tolerance is set looser than what Section 6.1 specifies.

**Verification:** Code review confirms the tolerance constants match Section 6.1.

### R6. Diagnostic Failure Messages

When a golden value comparison fails, the output includes: (a) the test case ID, (b) the metric name, (c) the expected value, (d) the actual value, (e) the absolute difference, and (f) the tolerance.

**Verification:** Intentionally perturb a golden value and confirm the failure message contains all required fields.

### R7. Build Integration

The test compiles and runs as part of `ninja test` on all CI platforms without manual intervention.

**Verification:** CI pipeline passes on all platforms with the golden value test included.

### R8. C Reference Fallback Passes

The golden value test passes when built with `-Denable_asm=false`, confirming that the golden values are consistent with the current C reference code.

**Verification:** The `simd-oracle-fallback` CI job passes with the golden value test.

### R9. Single-Threaded Determinism

The test uses `n_threads = 1` and `n_subsample = 1` to ensure deterministic results across runs.

**Verification:** Run the test twice on the same platform and confirm identical pass/fail results and identical scores.

### R10. Regeneration Procedure Documented and Working

A `generate_golden_values` tool or script exists, is buildable from the Meson configuration, and produces output that matches the `golden_values.json` format. The regeneration procedure in Section 10.3 is verified to work end-to-end.

**Verification:** Follow the regeneration procedure from scratch and confirm it produces a valid golden values file that passes all tests.

### R11. Identity Test Case

A test case exists that compares a video against itself and verifies perfect scores (ADM2 = 1.0, SSIM = 1.0, etc.).

**Verification:** The identity test case exists and passes on all platforms.

### R12. No Test Video Duplication

The test accesses existing video files in `python/test/resource/yuv/` rather than duplicating them into a separate directory.

**Verification:** No new `.yuv` files are added to the repository as part of this implementation.

---

## 15. Acceptance Criteria Summary

| Req | Description | Pass Condition |
|-----|-------------|----------------|
| R1 | Golden values file exists | Valid JSON with all 8 test cases |
| R2 | All metrics covered | VMAF, VIF, ADM2, Motion2, PSNR, SSIM, MS-SSIM, CAMBI present |
| R3 | All bit depths tested | 8-bit, 10-bit, 12-bit entries exist |
| R4 | All models tested | 5 models referenced in test cases |
| R5 | Tolerance enforcement | Integer: 1e-6, Float/composite: 1e-4 |
| R6 | Diagnostic messages | Failing tests report case ID, metric, expected, actual, diff, tolerance |
| R7 | Build integration | `ninja test` passes on all CI platforms |
| R8 | C reference fallback | `-Denable_asm=false` build passes golden value tests |
| R9 | Deterministic | Single-threaded, repeated runs produce identical results |
| R10 | Regeneration works | `generate_golden_values` tool produces valid output |
| R11 | Identity test | Ref-vs-ref test case with perfect scores |
| R12 | No video duplication | Reuses existing `python/test/resource/yuv/` files |

All 12 requirements must be met for the cross-architecture golden value test harness to be considered complete.
