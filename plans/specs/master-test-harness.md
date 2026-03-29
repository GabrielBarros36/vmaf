# Master Test Harness

## Overview

The VMAF master test harness (`scripts/run_tests.sh`) provides a unified entry
point for running all VMAF tests, organized into six tiers by execution cost
and infrastructure requirements. The tier system lets developers run an
appropriate subset locally while reserving expensive or specialized tests for
CI.

## Tier System

### Tier 1: Quick Local Tests

**Time:** <30 seconds
**Command:** `./scripts/run_tests.sh` or `./scripts/run_tests.sh quick`
**Purpose:** Fast pre-commit smoke test. Every developer should run this before
pushing.

| Test | Component |
|------|-----------|
| test_picture | Picture allocation and lifecycle |
| test_feature_collector | Feature score collection |
| test_thread_pool | Thread pool creation and task dispatch |
| test_model | Model loading and JSON parsing |
| test_predict | Score prediction pipeline |
| test_dict | Dictionary (key-value) operations |
| test_feature_extractor | Feature extractor registration and dispatch |
| test_cpu | CPU feature detection |
| test_ref | Reference counting |
| test_feature | Feature name/alias resolution |
| test_ciede | CIEDE2000 color difference |
| test_cambi | CAMBI banding detection |
| test_luminance_tools | Luminance conversion utilities |
| test_cli_parse | CLI argument parsing |
| test_psnr | PSNR computation |
| test_propagate_metadata | Metadata propagation |
| test_format_picture | Picture format coverage (3.11) |
| test_model_validation | Model integrity checks (3.10) |

### Tier 2: Standard Local Tests

**Time:** ~1-2 minutes (cumulative with Tier 1)
**Command:** `./scripts/run_tests.sh standard`
**Purpose:** Validates SIMD correctness, thread safety, edge-case inputs, and
numerical precision. Recommended before merging feature branches.

| Test | Component |
|------|-----------|
| test_degenerate | Degenerate/edge-case inputs (3.3) |
| test_thread_safety | Concurrent API usage (3.8) |
| test_format_y4m | Y4M format parsing and pixel formats (3.11) |
| test_simd_adm | ADM SIMD vs C-reference oracle (3.1) |
| test_simd_vif | VIF SIMD vs C-reference oracle (3.1) |
| test_simd_motion | Motion SIMD vs C-reference oracle (3.1) |
| test_simd_cambi | CAMBI SIMD vs C-reference oracle (3.1) |
| test_precision_differential | Float vs integer precision bounds (3.6) |
| test_framesync | Frame synchronization |

**Note:** `test_precision_differential` requires `enable_float=true` at build
time. The runner automatically skips it when float is disabled.

### Tier 3: Extended Local Tests

**Time:** ~5-15 minutes (cumulative with Tiers 1+2)
**Command:** `./scripts/run_tests.sh extended`
**Purpose:** Full local validation including cross-architecture golden values
and thread-safety stress tests. Run before releases or after significant
algorithmic changes.

| Test | Component |
|------|-----------|
| test_golden_values | Cross-architecture golden value comparison (3.4) |
| test_thread_safety_stress | Extended thread-safety stress suite (3.8) |

### Tier 4: Benchmarks

**Time:** ~1-5 minutes
**Command:** `./scripts/run_tests.sh benchmark`
**Purpose:** Collects SIMD performance data. Not pass/fail -- results are
informational and can be compared against baselines.

| Benchmark | Component |
|-----------|-----------|
| bench_simd_perf | SIMD kernel throughput measurement (3.9) |

### Tier 5: CI-Only

**Command:** `./scripts/run_tests.sh ci-status` (shows inventory)
**Purpose:** Tests that require GitHub Actions infrastructure -- sanitizer
builds, build configuration matrix, and fuzz testing.

| Workflow | File | Purpose |
|----------|------|---------|
| Sanitizers (3.2) | `.github/workflows/sanitizers.yml` | ASan, UBSan, MSan, TSan builds |
| Build Matrix (3.7) | `.github/workflows/build-matrix.yml` | Debug, LTO, ASM-off, AVX-512-off, float toggle, -O0 |
| Fuzzing (3.5) | `.github/workflows/fuzz.yml` | libFuzzer: Y4M parser, JSON model, picture reader |
| SIMD Oracle CI | `.github/workflows/simd-oracle.yml` | Multi-arch SIMD oracle (x86, ARM64, C-ref fallback) |
| Perf Regression | `.github/workflows/perf-regression.yml` | Benchmark on x86_64 and ARM64 runners |

### Tier 6: Manual/Specialized

**Purpose:** Tests requiring specific tooling that cannot be auto-detected.

| Test | Location | Requirements |
|------|----------|--------------|
| Property-based tests (3.5) | `python/test/test_property_fuzz.py` | Python + hypothesis |
| Fuzz targets (3.5) | `libvmaf/fuzz/` | clang with `-fsanitize=fuzzer` |

## Rationale

The tier system is designed around two axes:

1. **Execution time** -- Developers need fast feedback during edit-compile-test
   cycles. Tier 1 tests complete in under 30 seconds. Tiers 2 and 3 add
   progressively more thorough checks for pre-merge validation.

2. **Infrastructure requirements** -- Some tests need sanitizer-instrumented
   builds, multi-architecture runners, or fuzzing engines that are
   impractical to set up on every developer machine. These are relegated to
   Tiers 5 and 6.

The mapping between the ten test harness components (3.1 through 3.11) and
tiers:

| Component | Tier(s) |
|-----------|---------|
| 3.1 SIMD Correctness Oracle | 2 (local), 5 (multi-arch CI) |
| 3.2 Sanitizer CI Jobs | 5 |
| 3.3 Degenerate Input Tests | 2 |
| 3.4 Cross-Arch Golden Values | 3 |
| 3.5 Fuzzing Infrastructure | 5, 6 |
| 3.6 Numerical Precision | 2 |
| 3.7 Build Config Matrix | 5 |
| 3.8 Thread Safety | 2 (basic), 3 (stress) |
| 3.9 Performance Regression | 4 |
| 3.10 Model Validation | 1 |
| 3.11 Format/Codec Coverage | 1 (picture), 2 (Y4M) |

## Script Options

```
./scripts/run_tests.sh [TIER] [OPTIONS]

Tiers:
  quick       (default) Tier 1 only
  standard    Tiers 1+2
  extended    Tiers 1+2+3
  benchmark   Tier 4 only
  all         Tiers 1+2+3+4
  ci-status   Show CI workflow inventory

Options:
  --build-dir DIR   Specify meson build directory
  --no-color        Disable colored output
  --verbose         Show full test output on failure
  --help            Show usage
```

## Build Directory Detection

The script searches for a build directory in this order:

1. User-specified `--build-dir`
2. `libvmaf/build`
3. `libvmaf/build_verify`
4. `libvmaf/builddir`

If none exist, it offers to create one with recommended settings
(`debugoptimized`, `enable_float=true`).

## CI Workflow Overview

All CI workflows trigger on push and pull_request events unless noted
otherwise.

- **sanitizers.yml** -- Four parallel jobs (ASan, UBSan, MSan, TSan) each
  build with clang and run the full meson test suite under instrumentation.
  Timeout multipliers are set to 10-15x to accommodate sanitizer overhead.

- **build-matrix.yml** -- Six configurations: debug build (C1), LTO with
  GCC and Clang ThinLTO (C2), ASM disabled (C3), AVX-512 disabled (C4),
  float enabled/disabled toggle (C5), and -O0 build (C8). Two
  configurations (32-bit cross-compile C6, older compilers C7) are deferred.

- **fuzz.yml** -- Three libFuzzer targets (Y4M parser, JSON model, picture
  reader) run for 120s on PR/push and 3600s on nightly schedule. Crash
  artifacts are uploaded on failure.

- **simd-oracle.yml** -- Runs SIMD oracle tests on x86_64 (with best-effort
  AVX-512), ARM64 NEON, and C-reference fallback (ASM disabled).

- **perf-regression.yml** -- Runs bench_simd_perf on x86_64 and ARM64
  runners, compares against baseline files, and uploads results as
  artifacts.

## Relationship Between Tiers

```
Tier 1 (quick)      -- Always run. Foundation for all other tiers.
  |
  v
Tier 2 (standard)   -- Adds SIMD, concurrency, edge-case, and precision tests.
  |
  v
Tier 3 (extended)   -- Adds cross-arch golden values and stress tests.
  |
  v
Tier 4 (benchmark)  -- Performance data collection (orthogonal to correctness).

Tier 5 (CI-only)    -- Sanitizers, build variants, fuzzing (GitHub Actions).
Tier 6 (manual)     -- Property-based tests, fuzz targets (special tooling).
```

Tiers 1-3 are cumulative: `standard` includes `quick`, and `extended` includes
both. Tier 4 is independent (benchmarks are not correctness tests). Tiers 5
and 6 are not runnable through the script -- they exist in CI or require manual
setup.
