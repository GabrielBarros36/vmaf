# Numerical Precision Differential Tests — Specification

## 1. Objective

Validate that the **float** and **integer** implementations of each VMAF metric produce scores that agree within documented, justified tolerance bounds when given identical input. Track the maximum observed delta across the test corpus and alert if it exceeds historical bounds, ensuring that no implementation change silently degrades cross-implementation consistency.

This test suite guards the contract that switching between float and integer feature extractors does not materially change VMAF scoring behavior.

---

## 2. Scope — Metrics with Dual Implementations

There are **5 metrics** with both a float-domain and an integer-domain feature extractor. Each produces one or more per-frame scores that must be compared.

### 2.1 Implementation Inventory

| # | Metric | Float Extractor Name | Integer Extractor Name | Float Source | Integer Source | Float Feature Guard |
|---|--------|---------------------|----------------------|--------------|----------------|---------------------|
| 1 | ADM | `float_adm` | `adm` | `float_adm.c` | `integer_adm.c` | `VMAF_FLOAT_FEATURES` |
| 2 | VIF | `float_vif` | `vif` | `float_vif.c` | `integer_vif.c` | `VMAF_FLOAT_FEATURES` |
| 3 | Motion | `float_motion` | `motion` | `float_motion.c` | `integer_motion.c` | `VMAF_FLOAT_FEATURES` |
| 4 | PSNR | `float_psnr` | `psnr` | `float_psnr.c` | `integer_psnr.c` | `VMAF_FLOAT_FEATURES` |
| 5 | SSIM | `float_ssim` | `ssim` | `float_ssim.c` | `integer_ssim.c` | Always compiled |

**Key architectural difference:** The float implementations operate on `float` pixel buffers (converted via `picture_copy()` with a bias of -128 for ADM/VIF/Motion, 0 for PSNR/SSIM) and use floating-point arithmetic throughout. The integer implementations operate directly on the original pixel data (`uint8_t` or `uint16_t`) and use fixed-point arithmetic with defined shift/rounding constants. The two pipelines are algorithmically equivalent but numerically distinct.

### 2.2 Comparable Score Pairs

For each metric, the following score pairs are compared between float and integer:

| Metric | Float Feature Name | Integer Feature Name | Score Type |
|--------|--------------------|---------------------|------------|
| ADM | `VMAF_feature_adm2_score` | `VMAF_integer_feature_adm2_score` | Primary score |
| ADM | `VMAF_feature_adm_scale0_score` | `integer_adm_scale0` | Per-scale ratio |
| ADM | `VMAF_feature_adm_scale1_score` | `integer_adm_scale1` | Per-scale ratio |
| ADM | `VMAF_feature_adm_scale2_score` | `integer_adm_scale2` | Per-scale ratio |
| ADM | `VMAF_feature_adm_scale3_score` | `integer_adm_scale3` | Per-scale ratio |
| VIF | `VMAF_feature_vif_scale0_score` | `VMAF_integer_feature_vif_scale0_score` | Per-scale ratio |
| VIF | `VMAF_feature_vif_scale1_score` | `VMAF_integer_feature_vif_scale1_score` | Per-scale ratio |
| VIF | `VMAF_feature_vif_scale2_score` | `VMAF_integer_feature_vif_scale2_score` | Per-scale ratio |
| VIF | `VMAF_feature_vif_scale3_score` | `VMAF_integer_feature_vif_scale3_score` | Per-scale ratio |
| Motion | `VMAF_feature_motion2_score` | `VMAF_integer_feature_motion2_score` | Primary score |
| Motion | `VMAF_feature_motion_score` | `VMAF_integer_feature_motion_score` | Per-frame score |
| PSNR | `float_psnr` | `psnr_y` | Luma PSNR |
| SSIM | `float_ssim` | `ssim` | Primary score |

**Note on PSNR:** The float PSNR (`float_psnr`) and integer PSNR (`psnr`) use fundamentally different peak definitions. `float_psnr` normalizes pixels to float and uses peak values like 255.0 / 255.75 / 255.9375, while `psnr` operates on native integer samples with peak = `(1 << bpc) - 1`. For 8-bit content both compute identical MSE (and thus identical PSNR when peaks match), but for higher bit depths the peak definitions diverge. The precision test for PSNR at 8-bit validates arithmetic equivalence; higher bit depth tests document the expected divergence.

---

## 3. API Invocation — Selecting Float vs. Integer

### 3.1 C API

Both implementations are registered as `VmafFeatureExtractor` instances and are resolved by name through `vmaf_get_feature_extractor_by_name()`:

```c
#include "feature/feature_extractor.h"

// Float ADM
VmafFeatureExtractor *fex_float = vmaf_get_feature_extractor_by_name("float_adm");

// Integer ADM
VmafFeatureExtractor *fex_int = vmaf_get_feature_extractor_by_name("adm");
```

The full name mapping:

| Metric | Float Name String | Integer Name String |
|--------|-------------------|---------------------|
| ADM | `"float_adm"` | `"adm"` |
| VIF | `"float_vif"` | `"vif"` |
| Motion | `"float_motion"` | `"motion"` |
| PSNR | `"float_psnr"` | `"psnr"` |
| SSIM | `"float_ssim"` | `"ssim"` |

### 3.2 Public API (`vmaf_use_feature`)

From the public `libvmaf` API, feature extractors are registered by name:

```c
vmaf_use_feature(vmaf, "float_adm", NULL);  // float ADM
vmaf_use_feature(vmaf, "adm", NULL);        // integer ADM
```

After `vmaf_read_pictures()` completes, scores are fetched by feature name:

```c
double float_score, int_score;
vmaf_feature_score_at_index(vmaf, "VMAF_feature_adm2_score", &float_score, idx);
vmaf_feature_score_at_index(vmaf, "VMAF_integer_feature_adm2_score", &int_score, idx);
```

### 3.3 Build Prerequisite

The float feature extractors for ADM, VIF, Motion, and PSNR are gated behind the `enable_float` meson option (compiled only when `VMAF_FLOAT_FEATURES` is defined). The precision differential tests therefore require the build to be configured with:

```
meson setup build -Denable_float=true -Denable_tests=true
```

SSIM and float_ssim are always compiled and do not require `enable_float`.

---

## 4. Test Input Corpus

### 4.1 Video Test Inputs

The tests use the existing VMAF test corpus already checked into the repository at `python/test/resource/yuv/`. These files exercise real-world video content rather than synthetic patterns, which is critical for validating that precision bounds hold on content the models were trained on.

| # | File(s) | Resolution | Bit Depth | Format | Purpose |
|---|---------|-----------|-----------|--------|---------|
| 1 | `src01_hrc00_576x324.yuv` / `src01_hrc01_576x324.yuv` | 576x324 | 8 | YUV420P | Primary distorted pair — standard test content |
| 2 | `src01_hrc00_576x324.yuv` / `src01_hrc00_576x324.yuv` | 576x324 | 8 | YUV420P | Identical pair — tests perfect-score agreement |
| 3 | `src01_hrc00_576x324.yuv420p10le.yuv` / `src01_hrc01_576x324.yuv422p10le.yuv` | 576x324 | 10 | YUV422P10LE | 10-bit content |
| 4 | `src01_hrc00_576x324.yuv420p12le.yuv` / `src01_hrc01_576x324.yuv420p12le.yuv` | 576x324 | 12 | YUV420P12LE | 12-bit content |
| 5 | `src01_hrc00_576x324.yuv420p16le.yuv` / (self) | 576x324 | 16 | YUV420P16LE | 16-bit content, identical pair |
| 6 | `ref_test_..._160x90.yuv` / `dis_test_..._160x90.yuv` | 160x90 | 8 | YUV420P | Small resolution — edge-case for multi-scale metrics |
| 7 | `checkerboard_1920_1080_10_3_0_0.yuv` | 1920x1080 | 10 | YUV420P10LE | Synthetic high-frequency pattern |
| 8 | `flat_1920_1080_0.yuv` | 1920x1080 | 8 | YUV420P | Flat-field (constant pixel value) |

### 4.2 Synthetic Per-Frame Inputs (C-Level Tests)

For C unit tests that bypass the full feature extractor pipeline, synthetic `VmafPicture` frames are generated programmatically:

| Category | Description | Purpose |
|----------|-------------|---------|
| **Constant-128** | All luma pixels = 128 | Mid-range DC — verifies baseline agreement |
| **Gradient** | Horizontal ramp 0..255 | Covers full dynamic range |
| **Random (seed=42)** | Deterministic PRNG fill | General coverage |
| **Flat-zero** | All pixels = 0 | Division-by-zero edge case |
| **Flat-max** | All pixels = 255 (8-bit) | Overflow edge case |
| **Checkerboard** | Alternating 0/255 | High-frequency worst case |

Each category is tested at the following dimensions:

| Width | Height | Rationale |
|-------|--------|-----------|
| 64 | 64 | Minimum practical for multi-scale (4 DWT levels) |
| 576 | 324 | Matches primary test video |
| 1920 | 1080 | Full HD |

---

## 5. Tolerance Bounds

### 5.1 Tolerance Derivation Methodology

Tolerances are derived from three sources:

1. **Empirical measurement** from the existing Python golden-value tests, which record both float and integer scores for the same content.
2. **Algorithmic analysis** of where float-vs-integer divergence enters (quantization of filter taps, fixed-point rounding, accumulator bit width).
3. **Safety margin** — the test tolerance is set to 2x the maximum historically observed delta, providing headroom for platform-dependent floating-point behavior.

### 5.2 Observed Deltas from Golden Values

The following deltas are computed from the Python test suite golden values (`vmafexec_feature_extractor_test.py`) on the 576x324 8-bit test pair:

| Metric | Score | Float Value | Integer Value | Absolute Delta | Relative Delta |
|--------|-------|-------------|---------------|----------------|----------------|
| ADM | adm2 | 0.934515 | 0.934506 | 9.1e-06 | 9.7e-06 |
| ADM | scale0 | 0.907887 | 0.907887 | <1e-07 | <1e-07 |
| ADM | scale1 | 0.893871 | (same places) | ~1e-04 | ~1e-04 |
| ADM | scale2 | 0.930012 | (same places) | ~1e-04 | ~1e-04 |
| ADM | scale3 | 0.964966 | (same places) | ~1e-04 | ~1e-04 |
| VIF | scale0 | 0.363421 | 0.363662 | 2.4e-04 | 6.6e-04 |
| VIF | scale1 | 0.766647 | 0.767495 | 8.5e-04 | 1.1e-03 |
| VIF | scale2 | 0.862853 | 0.863108 | 2.5e-04 | 2.9e-04 |
| VIF | scale3 | 0.915972 | 0.915720 | 2.5e-04 | 2.7e-04 |
| Motion | motion2 | 3.895352 | 3.895345 | 6.6e-06 | 1.7e-06 |
| Motion | motion | 4.049825 | 4.049818 | 7.2e-06 | 1.8e-06 |
| SSIM | ssim | (float_ssim) | (ssim) | ~1e-04 | ~1e-04 |

### 5.3 Tolerance Table

Based on the observed deltas (with 2x safety margin) and algorithmic understanding:

| Metric | Tolerance Type | Tolerance Value | Justification |
|--------|---------------|-----------------|---------------|
| **ADM** | Absolute | **5e-04** | Integer ADM uses 16-bit fixed-point DWT with rounding; float ADM uses native `float`. Max observed delta ~1e-04 at per-scale level. Tolerance = ~5x observed max. |
| **VIF** | Absolute | **2e-03** | Integer VIF uses fixed-point multiply-accumulate for statistics; float VIF uses native `float` convolutions. Max observed delta ~1.1e-03 at scale1. Tolerance = ~2x observed max. |
| **Motion** | Absolute | **1e-04** | Integer motion uses fixed-point Gaussian blur + integer SAD; float motion uses `float` convolution + `float` subtraction. Max observed delta ~7e-06. Tolerance provides generous headroom. |
| **PSNR (8-bit)** | Absolute | **1e-04** | Both implementations compute identical MSE for 8-bit content; divergence arises only from peak definition and `log10` precision. |
| **PSNR (>8-bit)** | Informational | **N/A** | Different peak definitions make scores non-comparable. Deltas are logged but not asserted. |
| **SSIM** | Absolute | **1e-03** | `float_ssim` and `ssim` use entirely different computational approaches (float windowed convolution vs. integer Gaussian kernel accumulation). |

### 5.4 Near-Zero Handling

When both float and integer scores are near zero (absolute value < 1e-9), the tolerance check passes unconditionally. This avoids false alarms from division-by-zero in relative-error calculations (e.g., motion score on the first frame is defined as 0.0 for both implementations).

---

## 6. Delta Tracking Mechanism

### 6.1 Per-Test-Run Delta Log

Each test invocation records the maximum observed delta across all frames and all score types into a structured log:

```
PRECISION_DELTA metric=ADM score=adm2 max_abs_delta=9.1e-06 max_rel_delta=9.7e-06 frames=48 input=src01_576x324_8bit
PRECISION_DELTA metric=VIF score=scale1 max_abs_delta=8.5e-04 max_rel_delta=1.1e-03 frames=48 input=src01_576x324_8bit
```

### 6.2 Historical Bounds File

A JSON baseline file stores the historical maximum deltas observed in CI:

```
libvmaf/test/precision_baselines.json
```

Format:
```json
{
  "version": 1,
  "baselines": {
    "ADM.adm2": {
      "max_abs_delta": 9.1e-06,
      "tolerance": 5e-04,
      "updated": "2026-03-08",
      "input": "src01_576x324_8bit"
    },
    "VIF.scale1": {
      "max_abs_delta": 8.5e-04,
      "tolerance": 2e-03,
      "updated": "2026-03-08",
      "input": "src01_576x324_8bit"
    }
  }
}
```

### 6.3 Alert Mechanism

The test produces a **warning** (not a failure) when the observed delta exceeds the historical baseline by more than 10%, even if it remains within tolerance. This early-warning system catches gradual drift:

```c
if (observed_delta > baseline_delta * 1.1) {
    fprintf(stderr, "WARNING: %s delta %.2e exceeds historical baseline %.2e by %.1f%%\n",
            score_name, observed_delta, baseline_delta,
            100.0 * (observed_delta - baseline_delta) / baseline_delta);
}
```

The test **fails** only when the observed delta exceeds the tolerance from Section 5.3.

### 6.4 Baseline Update Procedure

When a code change intentionally alters float-vs-integer agreement (e.g., improved integer rounding), update the baseline:

1. Run the precision tests with `--update-baselines` (environment variable `VMAF_UPDATE_PRECISION_BASELINES=1`).
2. The test writes the new observed maxima to `precision_baselines.json`.
3. Review the diff in the baseline file during code review.
4. Commit the updated baseline with the code change.

---

## 7. Test Architecture

### 7.1 Design: Two-Layer Testing

The precision differential tests use two complementary approaches:

#### Layer 1: C Unit Tests (per-frame granularity)

C tests create `VmafPicture` pairs, run both the float and integer feature extractor `extract()` functions on them, and compare the resulting scores from the `VmafFeatureCollector`. This tests the full feature extractor pipeline including initialization, pixel format handling, and score normalization.

```c
static char *test_adm_precision_8bit(void)
{
    VmafFeatureExtractor *fex_float = vmaf_get_feature_extractor_by_name("float_adm");
    VmafFeatureExtractor *fex_int   = vmaf_get_feature_extractor_by_name("adm");

    VmafFeatureExtractorContext *ctx_float, *ctx_int;
    vmaf_feature_extractor_context_create(&ctx_float, fex_float, NULL);
    vmaf_feature_extractor_context_create(&ctx_int, fex_int, NULL);

    VmafFeatureCollector *fc_float, *fc_int;
    vmaf_feature_collector_init(&fc_float);
    vmaf_feature_collector_init(&fc_int);

    // Create identical ref/dis VmafPicture pairs
    VmafPicture ref, dis;
    vmaf_picture_alloc(&ref, VMAF_PIX_FMT_YUV420P, 8, 576, 324);
    vmaf_picture_alloc(&dis, VMAF_PIX_FMT_YUV420P, 8, 576, 324);
    fill_gradient(&ref); fill_gradient(&dis);  // deterministic fill

    vmaf_feature_extractor_context_extract(ctx_float, &ref, NULL, &dis, NULL, 0, fc_float);
    vmaf_feature_extractor_context_extract(ctx_int,   &ref, NULL, &dis, NULL, 0, fc_int);

    double score_float, score_int;
    vmaf_feature_collector_get_score(fc_float, "VMAF_feature_adm2_score", &score_float, 0);
    vmaf_feature_collector_get_score(fc_int, "VMAF_integer_feature_adm2_score", &score_int, 0);

    double delta = fabs(score_float - score_int);
    mu_assert("ADM adm2 delta exceeds tolerance", delta < ADM_TOLERANCE);

    // Track max delta for reporting
    track_max_delta("ADM.adm2", delta);

    // Cleanup...
    return NULL;
}
```

#### Layer 2: Python Integration Tests (full-pipeline, multi-frame)

Python tests use `vmafexec` to run both float and integer feature extractors on real video files and compare the per-frame score arrays:

```python
def test_adm_precision_576x324_8bit(self):
    ref_path, dis_path, asset, _ = set_default_576_324_videos_for_testing()

    float_fex = FloatAdmFeatureExtractor([asset], None, result_store=None,
                                          optional_dict={'debug': True})
    float_fex.run()

    int_fex = IntegerAdmFeatureExtractor([asset], None, result_store=None,
                                          optional_dict={'debug': True})
    int_fex.run()

    float_scores = float_fex.results[0]['float_ADM_feature_adm2_scores']
    int_scores   = int_fex.results[0]['integer_ADM_feature_adm2_scores']

    max_delta = max(abs(f - i) for f, i in zip(float_scores, int_scores))
    self.assertLess(max_delta, ADM_TOLERANCE)

    # Log max delta for historical tracking
    print(f"PRECISION_DELTA metric=ADM score=adm2 max_abs_delta={max_delta:.2e} "
          f"frames={len(float_scores)} input=src01_576x324_8bit")
```

### 7.2 Test File Organization

```
libvmaf/test/
  test_precision_differential.c     # C unit tests for all 5 metrics
  precision_baselines.json          # Historical delta baselines

python/test/
  precision_differential_test.py    # Python integration tests
```

### 7.3 C Test Structure

The C test file follows the existing Minunit pattern (`test.h`) and is structured as one test function per metric-input combination:

```c
// Per-metric test functions
static char *test_adm_precision_8bit(void);
static char *test_adm_precision_identical(void);
static char *test_vif_precision_8bit(void);
static char *test_vif_precision_identical(void);
static char *test_motion_precision_8bit(void);
static char *test_motion_precision_identical(void);
static char *test_psnr_precision_8bit(void);
static char *test_ssim_precision_8bit(void);
static char *test_ssim_precision_identical(void);

// Synthetic input tests
static char *test_adm_precision_synthetic_gradient(void);
static char *test_adm_precision_synthetic_flat(void);
static char *test_vif_precision_synthetic_gradient(void);
static char *test_motion_precision_synthetic_random(void);

// Delta reporting
static char *test_precision_report(void);  // prints summary of all max deltas

char *run_tests(void) {
#if VMAF_FLOAT_FEATURES
    mu_run_test(test_adm_precision_8bit);
    mu_run_test(test_adm_precision_identical);
    mu_run_test(test_vif_precision_8bit);
    mu_run_test(test_vif_precision_identical);
    mu_run_test(test_motion_precision_8bit);
    mu_run_test(test_motion_precision_identical);
    mu_run_test(test_psnr_precision_8bit);
    mu_run_test(test_adm_precision_synthetic_gradient);
    mu_run_test(test_adm_precision_synthetic_flat);
    mu_run_test(test_vif_precision_synthetic_gradient);
    mu_run_test(test_motion_precision_synthetic_random);
#endif
    mu_run_test(test_ssim_precision_8bit);
    mu_run_test(test_ssim_precision_identical);
    mu_run_test(test_precision_report);
    return NULL;
}
```

ADM/VIF/Motion/PSNR precision tests are guarded by `#if VMAF_FLOAT_FEATURES` because the float implementations require the `enable_float` build option. SSIM precision tests are always compiled because both `float_ssim` and `ssim` are unconditionally built.

---

## 8. Comparison Function

### 8.1 Core Comparison Logic

```c
#define ADM_TOLERANCE    5e-04
#define VIF_TOLERANCE    2e-03
#define MOTION_TOLERANCE 1e-04
#define PSNR_TOLERANCE   1e-04
#define SSIM_TOLERANCE   1e-03
#define NEAR_ZERO_THRESH 1e-09

typedef struct {
    const char *name;
    double max_abs_delta;
    double max_rel_delta;
    unsigned count;
} DeltaTracker;

static int scores_agree(double float_val, double int_val, double tolerance,
                         DeltaTracker *tracker)
{
    double abs_delta = fabs(float_val - int_val);
    double denom = fmax(fabs(float_val), fabs(int_val));
    double rel_delta = (denom > NEAR_ZERO_THRESH) ? abs_delta / denom : 0.0;

    tracker->count++;
    if (abs_delta > tracker->max_abs_delta) tracker->max_abs_delta = abs_delta;
    if (rel_delta > tracker->max_rel_delta) tracker->max_rel_delta = rel_delta;

    if (fabs(float_val) < NEAR_ZERO_THRESH && fabs(int_val) < NEAR_ZERO_THRESH)
        return 1;  // both near zero — pass

    return abs_delta < tolerance;
}
```

### 8.2 Failure Reporting

When a comparison fails, the test reports:

```
PRECISION FAIL: ADM.adm2
  float_value  = 0.934514854166667
  int_value    = 0.935014854166667
  abs_delta    = 5.00e-04  (tolerance: 5.00e-04)
  rel_delta    = 5.35e-04
  frame_index  = 17
  input        = gradient_576x324_8bit
```

---

## 9. Reporting Format

### 9.1 Console Output (Test Pass)

```
test_adm_precision_8bit: pass
  ADM.adm2:   max_abs=9.1e-06  max_rel=9.7e-06  frames=48  status=WITHIN_BOUNDS
  ADM.scale0: max_abs=2.3e-07  max_rel=2.5e-07  frames=48  status=WITHIN_BOUNDS
  ADM.scale1: max_abs=8.1e-05  max_rel=9.1e-05  frames=48  status=WITHIN_BOUNDS
  ADM.scale2: max_abs=6.2e-05  max_rel=6.7e-05  frames=48  status=WITHIN_BOUNDS
  ADM.scale3: max_abs=3.1e-05  max_rel=3.2e-05  frames=48  status=WITHIN_BOUNDS
```

### 9.2 Console Output (Historical Drift Warning)

```
test_vif_precision_8bit: pass
  VIF.scale1: max_abs=9.2e-04  max_rel=1.2e-03  frames=48  status=DRIFT_WARNING
    WARNING: delta 9.2e-04 exceeds historical baseline 8.5e-04 by 8.2%
```

### 9.3 Summary Report

At the end of the test run, `test_precision_report` prints a consolidated table:

```
========== Numerical Precision Summary ==========
Metric      Score     MaxAbsDelta  MaxRelDelta  Tolerance  Status
ADM         adm2      9.1e-06      9.7e-06      5.0e-04    OK
ADM         scale0    2.3e-07      2.5e-07      5.0e-04    OK
ADM         scale1    8.1e-05      9.1e-05      5.0e-04    OK
ADM         scale2    6.2e-05      6.7e-05      5.0e-04    OK
ADM         scale3    3.1e-05      3.2e-05      5.0e-04    OK
VIF         scale0    2.4e-04      6.6e-04      2.0e-03    OK
VIF         scale1    8.5e-04      1.1e-03      2.0e-03    OK
VIF         scale2    2.5e-04      2.9e-04      2.0e-03    OK
VIF         scale3    2.5e-04      2.7e-04      2.0e-03    OK
Motion      motion2   6.6e-06      1.7e-06      1.0e-04    OK
Motion      motion    7.2e-06      1.8e-06      1.0e-04    OK
PSNR        psnr_y    <1.0e-07     <1.0e-07     1.0e-04    OK
SSIM        ssim      4.2e-05      4.5e-05      1.0e-03    OK
=================================================
```

### 9.4 Machine-Readable Output

When the environment variable `VMAF_PRECISION_JSON=1` is set, the test also writes a JSON report to `precision_report.json`:

```json
{
  "timestamp": "2026-03-08T12:00:00Z",
  "results": [
    {
      "metric": "ADM",
      "score": "adm2",
      "max_abs_delta": 9.1e-06,
      "max_rel_delta": 9.7e-06,
      "tolerance": 5e-04,
      "frames": 48,
      "input": "src01_576x324_8bit",
      "status": "OK"
    }
  ]
}
```

---

## 10. Integration with Build System

### 10.1 Meson Configuration

Add to `libvmaf/test/meson.build`:

```meson
# Numerical Precision Differential Tests
if get_option('enable_float')
    test_precision = executable('test_precision_differential',
        ['test.c', 'test_precision_differential.c'],
        include_directories : [libvmaf_inc, test_inc, include_directories('../src/')],
        link_with : get_option('default_library') == 'both' ? libvmaf.get_static_lib() : libvmaf,
        c_args : vmaf_cflags_common,
        dependencies : [math_lib, stdatomic_dependency, thread_lib, cuda_dependency],
    )
    test('test_precision_differential', test_precision,
         timeout : 300,
         env : ['VMAF_TEST_RESOURCE_PATH=' +
                join_paths(meson.project_source_root(), '..', 'python', 'test', 'resource')]
    )
endif
```

The test executable links against the full `libvmaf` static library, which includes both float and integer feature extractors (when `enable_float=true`).

### 10.2 CI Configuration

#### Dedicated CI Job

Add a CI job that specifically runs precision differential tests with float enabled:

```yaml
precision-tests:
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v6
    - name: Build with float features
      run: |
        cd libvmaf
        meson setup build -Denable_float=true -Denable_tests=true
        ninja -C build
    - name: Run precision tests
      run: |
        cd libvmaf
        VMAF_PRECISION_JSON=1 ninja -C build test
    - name: Upload precision report
      uses: actions/upload-artifact@v5
      with:
        name: precision-report
        path: libvmaf/build/precision_report.json
```

#### Integration with Existing CI

The precision tests also run as part of the standard `ninja test` on any CI job that builds with `enable_float=true`. On builds without `enable_float`, only the SSIM precision test compiles and runs.

### 10.3 Python Integration Test Runner

```bash
cd python
python -m pytest test/precision_differential_test.py -v --tb=short
```

The Python tests require `vmaf` to be built with `enable_float=true` and installed in the Python environment.

---

## 11. Test Execution Matrix

### 11.1 C Unit Tests

| Metric | Input Categories | Dimensions | Bit Depths | Score Comparisons | Total Points |
|--------|-----------------|------------|------------|-------------------|--------------|
| ADM | 6 synthetic + 2 video | 3 + 1 | 8 | 5 (adm2 + 4 scales) | 5 x 20 = 100 |
| VIF | 6 synthetic + 2 video | 3 + 1 | 8 | 4 (4 scales) | 4 x 20 = 80 |
| Motion | 6 synthetic + 2 video | 3 + 1 | 8 | 2 (motion + motion2) | 2 x 20 = 40 |
| PSNR | 6 synthetic + 2 video | 3 + 1 | 8 | 1 (psnr_y) | 1 x 20 = 20 |
| SSIM | 6 synthetic + 2 video | 3 + 1 | 8 | 1 (ssim) | 1 x 20 = 20 |

**Total per C test run: ~260 comparison points.**

### 11.2 Python Integration Tests

| Metric | Video Pairs | Bit Depths | Frames/Pair | Score Types | Total Points |
|--------|------------|------------|-------------|-------------|--------------|
| ADM | 3 | 8, 10, 12 | 48 | 5 | 3 x 48 x 5 = 720 |
| VIF | 3 | 8, 10, 12 | 48 | 4 | 3 x 48 x 4 = 576 |
| Motion | 2 | 8, 12 | 48 | 2 | 2 x 48 x 2 = 192 |
| PSNR | 1 | 8 | 48 | 1 | 48 |
| SSIM | 2 | 8, 10 | 48 | 1 | 96 |

**Total per Python test run: ~1,632 comparison points.**

---

## 12. Completion Requirements

The numerical precision differential test suite is **complete** when ALL of the following requirements are met:

### R1. All Five Metrics Covered

Every metric with dual float/integer implementations (ADM, VIF, Motion, PSNR, SSIM) has precision differential tests that exercise both implementations on identical inputs and compare scores.

**Verification:** For each of the 5 metrics in Section 2.1, there exists at least one test function that calls both the float and integer feature extractors and compares their output scores.

### R2. All Score Types Compared

For each metric, all score types listed in Section 2.2 are compared between float and integer outputs. No score type is left unverified.

**Verification:** Count the distinct score names compared across all test functions. The count must match the 13 score pairs listed in Section 2.2.

### R3. Documented Tolerance Bounds

Every tolerance value in Section 5.3 is:
- Defined as a named constant in the test source code.
- Justified in code comments referencing the observed delta and safety margin.
- Traceable to the empirical data in Section 5.2.

**Verification:** Code review confirms each tolerance constant has a comment explaining its derivation.

### R4. Real Video Content Tested

The tests exercise at least the primary 576x324 8-bit distorted pair (`src01_hrc00` vs `src01_hrc01`) and the identical pair (`src01_hrc00` vs itself). These are the same inputs used in the existing Python golden-value tests.

**Verification:** Test code references the standard test video paths and processes at least 2 frames from each.

### R5. Multiple Bit Depths Tested

The Python integration tests exercise 8-bit, 10-bit, and 12-bit content for ADM and VIF (the metrics most sensitive to bit-depth-dependent fixed-point behavior).

**Verification:** Test functions for 10-bit and 12-bit content exist for ADM and VIF.

### R6. Delta Tracking Operational

Every test function records the maximum observed delta via the `DeltaTracker` mechanism. The summary report (Section 9.3) is printed at the end of every test run.

**Verification:** Run the test suite and confirm the summary table appears in stdout.

### R7. Historical Baseline File Exists

The file `libvmaf/test/precision_baselines.json` is checked into the repository, contains baseline deltas for all 13 score pairs, and the drift-warning mechanism from Section 6.3 is implemented.

**Verification:** The JSON file parses correctly and the warning logic is present in the test source.

### R8. Build Integration

The C precision test compiles and runs as part of `ninja test` when configured with `meson setup build -Denable_float=true -Denable_tests=true`.

**Verification:** CI pipeline with `enable_float=true` passes with the precision tests included.

### R9. SSIM Always-On

The SSIM precision test compiles and runs even without `enable_float=true`, since both `float_ssim` and `ssim` are unconditionally compiled.

**Verification:** Build with `enable_float=false` and confirm the SSIM precision test is present in the test list and passes.

### R10. Near-Zero Edge Cases

Tests include at least one scenario where both float and integer scores are near zero (e.g., motion score on frame 0, or identical ref/dis pair for VIF numerator). The near-zero handling from Section 5.4 prevents false failures.

**Verification:** Test functions for identical-pair inputs exist and pass.

### R11. Failure Diagnostics

When a tolerance check fails, the test output includes: (a) the metric name, (b) the score name, (c) the float value, (d) the integer value, (e) the absolute delta, (f) the tolerance, (g) the frame index, and (h) the input description.

**Verification:** Temporarily set a tolerance to 0.0 and confirm all 8 fields appear in the failure message.

### R12. Deterministic Reproducibility

All tests are fully deterministic. PRNG seeds are hardcoded. Running the same test twice on the same platform produces identical delta values. No dependency on wall-clock time, system load, or thread scheduling.

**Verification:** Run the test suite twice and confirm identical delta values in the summary report.

### R13. No Production Code Changes

The precision tests operate entirely through the public feature extractor API (`vmaf_get_feature_extractor_by_name`, `vmaf_feature_extractor_context_create`, `vmaf_feature_extractor_context_extract`). No modifications to production source files are required.

**Verification:** `git diff` of `libvmaf/src/` shows no changes after adding the precision tests.

---

## 13. Acceptance Criteria Summary

| Req | Description | Pass Condition |
|-----|-------------|----------------|
| R1 | All five metrics covered | 5/5 metrics have precision differential tests |
| R2 | All score types compared | 13/13 score pairs verified |
| R3 | Documented tolerances | Each tolerance has derivation comment |
| R4 | Real video content | 576x324 distorted + identical pairs tested |
| R5 | Multiple bit depths | ADM and VIF tested at 8, 10, and 12 bit |
| R6 | Delta tracking | Summary report printed on every run |
| R7 | Historical baselines | `precision_baselines.json` checked in |
| R8 | Build integration | Tests pass in CI with `enable_float=true` |
| R9 | SSIM always-on | SSIM test runs without `enable_float` |
| R10 | Near-zero handling | Identical-pair tests pass |
| R11 | Failure diagnostics | 8 fields reported on failure |
| R12 | Deterministic | Repeated runs produce identical results |
| R13 | No production changes | Zero modifications to `libvmaf/src/` |

All 13 requirements must be met for the numerical precision differential test suite to be considered complete.
