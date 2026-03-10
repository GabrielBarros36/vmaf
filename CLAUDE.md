# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VMAF (Video Multi-Method Assessment Fusion) is Netflix's video quality metric library. The core is **libvmaf** (C, with SIMD assembly), wrapped by Python bindings. It computes perceptual quality scores (VMAF, VIF, ADM, PSNR, SSIM, CAMBI, etc.) by comparing reference and distorted video frames.

## Build Commands

```bash
# Standard build (from repo root)
cd libvmaf && meson setup build --buildtype release && ninja -C build

# With float feature extractors (needed for precision differential tests)
meson setup build --buildtype release -Denable_float=true

# Reconfigure existing build
meson configure build -Denable_float=true

# Common options: -Denable_asm=false (C-only), -Denable_avx512=false, -Denable_cuda=true
# All options listed in libvmaf/meson_options.txt
```

## Running Tests

```bash
# Master test harness (from repo root)
./scripts/run_tests.sh                          # Tier 1: quick smoke tests (<1s)
./scripts/run_tests.sh standard                 # Tier 1+2: adds SIMD oracle, threading, edge cases (~1-2min)
./scripts/run_tests.sh extended                 # Tier 1+2+3: adds golden values, stress tests (~5-15min)
./scripts/run_tests.sh benchmark                # Tier 4: SIMD performance benchmarks (informational)
./scripts/run_tests.sh all                      # All local tiers
./scripts/run_tests.sh --build-dir libvmaf/build  # Specify build dir

# Direct meson commands (from libvmaf/)
ninja -C build test                             # All registered tests
meson test -C build test_model_validation       # Single test by name
meson test -C build --suite stress              # Run stress suite only
ninja -C build benchmark                        # Performance benchmarks (not in `ninja test`)

# Python tests (from repo root)
tox -c python
```

## Test Tiers

| Tier | Tests | What's Covered |
|------|-------|----------------|
| 1 Quick | 18 core unit tests + model validation + format/picture | API, data structures, model loading |
| 2 Standard | + SIMD oracle, degenerate inputs, thread safety, precision, Y4M format, framesync | SIMD correctness, edge cases, concurrency, float vs int |
| 3 Extended | + golden values, thread stress | Cross-architecture score stability |
| 4 Benchmark | bench_simd_perf | SIMD kernel throughput vs C reference |
| 5 CI-only | sanitizers.yml, build-matrix.yml, fuzz.yml | ASan/UBSan/MSan/TSan, build variants, fuzzing |

## Writing C Tests

Tests use a Minunit-style framework (`libvmaf/test/test.h`). Each test file must:
1. Include `test.h`
2. Define test functions returning `char *` (NULL = pass, string = failure message)
3. Define `char *run_tests(void)` calling `mu_run_test(test_fn)` for each test
4. Link with `test.c` which provides `main()`

```c
#include "test.h"
static char *test_something(void) {
    mu_assert("values should match", actual == expected);
    return NULL;
}
char *run_tests(void) {
    mu_run_test(test_something);
    return NULL;
}
```

Register in `libvmaf/test/meson.build`:
```meson
exe = executable('test_foo', ['test.c', 'test_foo.c'],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/')],
    link_with : get_option('default_library') == 'both' ? libvmaf.get_static_lib() : libvmaf,
    dependencies : [thread_lib, cuda_dependency],
)
test('test_foo', exe, timeout : 300)
```

## Architecture

**Feature extractor pipeline:** `VmafContext` orchestrates scoring. Feature extractors (ADM, VIF, PSNR, etc.) are registered in a static array in `libvmaf/src/feature/feature_extractor.c` and discovered by name string. Each extractor has `init/extract/close` callbacks and declares its output feature names.

**Dual implementations:** Most metrics have both fixed-point (`integer_*.c`) and floating-point (`float_*.c`) versions. Integer is default and faster; float requires `-Denable_float=true`. Feature names: `"adm"` vs `"float_adm"`, `"vif"` vs `"float_vif"`, etc.

**SIMD dispatch:** Each module has a dispatch struct (e.g., `VifResiduals`) with function pointers set at init time based on CPU capabilities. SIMD variants live in `src/feature/x86/` (AVX2, AVX-512 via NASM) and `src/feature/arm/` (NEON). The C reference is the fallback. Width constraints exist (e.g., ADM DWT requires `w % 8 == 0` for SIMD).

**Threading:** `VmafContext.n_threads` controls the thread pool size. Feature extractors run in parallel per-frame. Temporal features (motion) use `VmafFrameSyncContext` for ordering. Thread count should not affect scores.

**Model system:** SVM-based models in JSON format, optionally compiled into the binary (`built_in_models=true`). Loaded via `vmaf_model_load("vmaf_v0.6.1")` (built-in) or `vmaf_model_load_from_path()` (file). Bootstrap/bagging models use `vmaf_model_collection_load()`.

**Scoring flow:** `vmaf_init()` -> `vmaf_use_features_from_model()` -> loop `vmaf_read_pictures(ref, dist, idx)` -> `vmaf_read_pictures(NULL, NULL, 0)` to flush -> `vmaf_score_at_index()` / `vmaf_score_pooled()`.

## Key Paths

- Public API: `libvmaf/include/libvmaf/libvmaf.h`, `picture.h`, `model.h`, `feature.h`
- Feature extractors: `libvmaf/src/feature/integer_*.c`, `float_*.c`
- SIMD: `libvmaf/src/feature/x86/`, `libvmaf/src/feature/arm/`
- Models: `model/*.json` (17 JSON model files)
- Test harness specs: `plans/specs/` (one spec + implementation doc per testing component)
- CI workflows: `.github/workflows/` (sanitizers, build-matrix, fuzz, simd-oracle, perf-regression)

## Test Data

Test videos are in `python/test/resource/yuv/`. The primary pair is `src01_hrc00_576x324.yuv` (reference) and `src01_hrc01_576x324.yuv` (distorted), 576x324, 8-bit YUV420p, 48 frames. Higher bit-depth variants (10-bit, 12-bit) exist with fewer frames. Tests reference these via compile-time path defines (e.g., `-DTEST_VIDEO_DIR=...`).
