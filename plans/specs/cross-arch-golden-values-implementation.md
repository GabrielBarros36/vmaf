# Cross-Architecture Golden Value Tests -- Implementation Notes

## Files Created

| File | Purpose |
|------|---------|
| `libvmaf/test/test_golden_values.c` | Main test source: reads YUV, runs VMAF pipeline, compares scores |
| `libvmaf/test/golden_values.h` | Embedded golden values as C structs (no JSON parsing needed) |
| `libvmaf/test/golden_values/golden_values.json` | Machine-readable golden values for reference/regeneration |

## Files Modified

| File | Change |
|------|--------|
| `libvmaf/test/meson.build` | Added `test_golden_values` executable and test registration |

## Test Cases

8 test cases are implemented, covering all metrics and models specified:

| # | Test Case ID | Model | Bit Depth | Frames | Metrics Checked |
|---|-------------|-------|-----------|--------|-----------------|
| 1 | `vmaf_8bit_576x324` | vmaf_v0.6.1 | 8 | 48 | VMAF, VIF x4, ADM2, Motion2, PSNR x3, SSIM, MS-SSIM (12 scores) |
| 2 | `vmaf_8bit_identity` | vmaf_v0.6.1 | 8 | 48 | VMAF, ADM2, VIF x4, PSNR x3, SSIM, MS-SSIM (11 scores) |
| 3 | `vmaf_12bit_576x324` | vmaf_v0.6.1 | 12 | 3 | VMAF, VIF x4, ADM2, Motion2 (7 scores) |
| 4 | `vmaf_neg_8bit_576x324` | vmaf_v0.6.1neg | 8 | 48 | VMAF (1 score) |
| 5 | `vmaf_4k_8bit_576x324` | vmaf_4k_v0.6.1 | 8 | 48 | VMAF (1 score) |
| 6 | `vmaf_float_8bit_576x324` | vmaf_float_v0.6.1 | 8 | 48 | VMAF (1 score) |
| 7 | `vmaf_b_v063_8bit_576x324` | vmaf_b_v0.6.3 (collection) | 8 | 48 | VMAF bagging (1 score) |
| 8 | `cambi_8bit_576x324` | (none) | 8 | 48 | CAMBI (1 score) |

**Total score comparisons:** 35

## Golden Values

All golden values were generated with `n_threads=1`, `n_subsample=1` for determinism.

### Primary 8-bit (48 frames, 576x324 YUV420P)

| Metric | Golden Value | Tolerance |
|--------|-------------|-----------|
| VMAF (v0.6.1) | 76.6689048244369 | 1e-4 |
| VIF scale 0 | 0.363662071526051 | 1e-6 |
| VIF scale 1 | 0.767495281994343 | 1e-6 |
| VIF scale 2 | 0.863107771923145 | 1e-6 |
| VIF scale 3 | 0.915720089028279 | 1e-6 |
| ADM2 | 0.9345057762924 | 1e-6 |
| Motion2 | 3.89534507195155 | 1e-6 |
| PSNR Y | 30.755064021049 | 1e-4 |
| PSNR Cb | 38.4494410571618 | 1e-4 |
| PSNR Cr | 40.991910248629 | 1e-4 |
| Float SSIM | 0.863226603716612 | 1e-4 |
| Float MS-SSIM | 0.963240616895571 | 1e-4 |

### Identity (ref vs ref)

| Metric | Golden Value | Tolerance |
|--------|-------------|-----------|
| VMAF | 99.9464266316011 | 1e-4 |
| ADM2 | 1.00000225228643 | 1e-6 |
| PSNR Y/Cb/Cr | 60.0 | 1e-4 |
| Float SSIM | 1.0 | 1e-4 |
| Float MS-SSIM | 1.0 | 1e-4 |

### 12-bit (3 frames)

| Metric | Golden Value | Tolerance |
|--------|-------------|-----------|
| VMAF | 82.5652300478838 | 1e-4 |
| ADM2 | 0.951770445547814 | 1e-6 |
| Motion2 | 2.81045309702555 | 1e-6 |

### Other Models (8-bit, 48 frames)

| Model | VMAF Score | Tolerance |
|-------|-----------|-----------|
| vmaf_v0.6.1neg | 75.0747291238163 | 1e-4 |
| vmaf_4k_v0.6.1 | 84.9506472652734 | 1e-4 |
| vmaf_float_v0.6.1 | 76.6842971132081 | 1e-4 |
| vmaf_b_v0.6.3 (bagging) | 74.9363363308294 | 1e-4 |

### CAMBI (no-ref, 8-bit, 48 frames)

| Metric | Golden Value | Tolerance |
|--------|-------------|-----------|
| CAMBI | 0.259684181909746 | 1e-4 |

## Tolerances

Per spec Section 6.1:

| Metric Category | Tolerance | Rationale |
|----------------|-----------|-----------|
| Integer features (VIF, ADM2, Motion2) | 1e-6 | Integer accumulation + final float division |
| PSNR | 1e-4 | log10 amplifies small differences |
| VMAF composite | 1e-4 | SVM prediction over multiple features |
| Float SSIM/MS-SSIM | 1e-4 | Non-associative float arithmetic |
| CAMBI | 1e-4 | Integer banding + float pooling |

## Test Video Files

All videos from `python/test/resource/yuv/` (no duplication):

- `src01_hrc00_576x324.yuv` (8-bit reference)
- `src01_hrc01_576x324.yuv` (8-bit distorted)
- `src01_hrc00_576x324.yuv420p12le.yuv` (12-bit reference)
- `src01_hrc01_576x324.yuv420p12le.yuv` (12-bit distorted)

Note: 10-bit test pair (`src01_hrc01_576x324.yuv420p10le.yuv`) is not available in the repository, so the 10-bit test case from the spec is omitted. The 12-bit test case provides higher-bit-depth coverage.

## Feature Name Convention

The VMAF feature collector stores integer feature scores under their full internal names (not aliases):

| Spec Name | Internal Name (used in golden_values.h) |
|-----------|----------------------------------------|
| vif_scale0 | `VMAF_integer_feature_vif_scale0_score` |
| adm2 | `VMAF_integer_feature_adm2_score` |
| motion2 | `VMAF_integer_feature_motion2_score` |
| psnr_y | `psnr_y` |
| float_ssim | `float_ssim` |
| float_ms_ssim | `float_ms_ssim` |
| cambi | `cambi` |

The VMAF composite score is stored under the model config name (set to `"golden_test"` in the test code).

For model collections (vmaf_b_v0.6.3), the bagging score is stored under `"golden_test_bagging"`.

## How to Regenerate Golden Values

```bash
# 1. Build with ASM disabled
meson setup libvmaf build_ref --buildtype release \
    -Denable_float=true -Denable_asm=false
ninja -C build_ref

# 2. Run the test to verify current values pass
build_ref/test/test_golden_values

# 3. If C reference code has changed, update values in golden_values.h
#    by running the pipeline manually and capturing scores.
#    (Use the gen_golden.c helper as a template.)

# 4. Update golden_values/golden_values.json to match

# 5. Verify the updated test passes
ninja -C build_ref test
```

## Build Integration

The test is registered in `libvmaf/test/meson.build` with a 300-second timeout:

```meson
test_golden_values = executable('test_golden_values',
    ['test.c', 'test_golden_values.c'],
    include_directories : [libvmaf_inc, test_inc],
    link_with : libvmaf,
    c_args : ['-DTEST_VIDEO_DIR="..."'],
    dependencies : [math_lib, thread_lib, cuda_dependency],
)
test('test_golden_values', test_golden_values, timeout: 300)
```

The test runs as part of `ninja test` on all platforms.

## Spec Requirement Coverage

| Req | Description | Status |
|-----|-------------|--------|
| R1 | Golden values file exists | Done (`golden_values.json` + `golden_values.h`) |
| R2 | All metrics covered | Done (VMAF, VIF x4, ADM2, Motion2, PSNR x3, SSIM, MS-SSIM, CAMBI) |
| R3 | All bit depths tested | 8-bit and 12-bit covered; 10-bit skipped (no distorted video available) |
| R4 | All models tested | Done (vmaf_v0.6.1, vmaf_v0.6.1neg, vmaf_4k_v0.6.1, vmaf_float_v0.6.1, vmaf_b_v0.6.3) |
| R5 | Tolerance enforcement | Done (integer: 1e-6, float/composite: 1e-4) |
| R6 | Diagnostic messages | Done (test case ID, metric, expected, actual, diff, tolerance) |
| R7 | Build integration | Done (meson test) |
| R8 | C reference fallback | Passes with `-Denable_asm=false` |
| R9 | Deterministic | `n_threads=1`, `n_subsample=1` |
| R10 | Regeneration documented | This document |
| R11 | Identity test | Done (test case 2) |
| R12 | No video duplication | Done (uses `python/test/resource/yuv/`) |
