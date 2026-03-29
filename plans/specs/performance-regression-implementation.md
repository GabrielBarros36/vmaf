# Performance Regression Tests -- Implementation Notes

## Files Created

### Benchmark Framework

| File | Purpose |
|------|---------|
| `libvmaf/test/bench_simd_perf.h` | Benchmark timing utilities: clock abstraction, statistics computation (median/min/max/stddev/CV%), JSON output formatting, benchmark runner with warmup + calibration + measurement |
| `libvmaf/test/bench_simd_perf.c` | Main benchmark executable covering all 9 SIMD dispatch points with C reference and SIMD variant benchmarks |
| `libvmaf/test/baselines/README` | Documentation of baseline file format and update procedures |

### Regression Detection

| File | Purpose |
|------|---------|
| `scripts/check_perf_regression.py` | Python script comparing benchmark JSON results against stored baselines with configurable thresholds (10% regression, 25% warning, 3x fallback detection) |

### CI Integration

| File | Purpose |
|------|---------|
| `.github/workflows/perf-regression.yml` | GitHub Actions workflow running benchmarks on x86_64 and ARM64 for every PR/push to master, with optional baseline generation via workflow_dispatch |

### Build System

| File | Modification |
|------|-------------|
| `libvmaf/test/meson.build` | Added `bench_simd_perf` executable and registered it with `benchmark()` (not `test()`) |

## Functions Benchmarked

All 9 dispatch points from the SIMD correctness oracle:

| # | Function | Benchmark ID | C | AVX2 | AVX-512 | NEON |
|---|----------|-------------|---|------|---------|------|
| 1 | `adm_dwt2_8` | `adm_dwt2_8` | Y | Y | - | Y |
| 2 | `subsample_rd_8` | `vif_subsample_rd_8` | Y | Y | Y | Y |
| 3 | `subsample_rd_16` | `vif_subsample_rd_16` | Y | Y | Y | Y |
| 4 | `vif_statistic_8` | `vif_statistic_8` | Y | Y | Y | Y |
| 5 | `vif_statistic_16` | `vif_statistic_16` | Y | Y | Y | Y |
| 6 | `x_convolution_16` | `motion_x_convolution_16` | Y | Y | Y | - |
| 7 | `increment_range` | `cambi_increment_range` | Y | Y | - | - |
| 8 | `decrement_range` | `cambi_decrement_range` | Y | Y | - | - |
| 9 | `get_derivative_data_for_row` | `cambi_derivative_row` | Y | Y | - | - |

Benchmark counts per platform:
- x86-64 with AVX-512: 9 C + 9 AVX2 + 5 AVX-512 = 23
- x86-64 without AVX-512: 9 C + 9 AVX2 = 18
- ARM64: 9 C + 5 NEON = 14

## How to Run Benchmarks

### Build

```bash
cd libvmaf
meson setup build --buildtype release -Denable_float=true
ninja -C build
```

### Run All Benchmarks

```bash
# Print summary to stderr, JSON to stdout
./build/test/bench_simd_perf

# Save JSON results to a file
./build/test/bench_simd_perf --output results.json

# More repetitions for stability
./build/test/bench_simd_perf --output results.json --reps 15
```

### Run via Meson

```bash
# This runs benchmarks (NOT tests)
ninja -C build benchmark
```

### Compare Against Baseline

```bash
python3 scripts/check_perf_regression.py \
    libvmaf/test/baselines/perf_baseline_linux_x86_64.json results.json
```

### A/B Comparison

```bash
# Branch A
git checkout main
meson setup build_a --buildtype release -Denable_float=true
ninja -C build_a
./build_a/test/bench_simd_perf --output results_a.json --reps 15

# Branch B
git checkout feature-branch
meson setup build_b --buildtype release -Denable_float=true
ninja -C build_b
./build_b/test/bench_simd_perf --output results_b.json --reps 15

# Compare
python3 scripts/check_perf_regression.py results_a.json results_b.json
```

## Output Format

### Console (stderr)

Human-readable table showing:
- Function name and variant
- Median, min, max time in nanoseconds
- Coefficient of variation (CV%)
- Speedup ratios for SIMD variants vs C reference

### JSON (stdout or --output file)

```json
{
  "benchmarks": [
    {
      "name": "adm_dwt2_8",
      "variant": "c",
      "iters_per_rep": 21,
      "num_reps": 7,
      "median_ns": 4681710.29,
      "min_ns": 4679060.71,
      "max_ns": 4697653.38,
      "mean_ns": 4686141.46,
      "stddev_ns": 8211.71,
      "cv_pct": 0.1752
    }
  ]
}
```

## Measurement Methodology

1. **Warm-up:** 3 iterations (discarded)
2. **Calibration:** Single iteration timed to compute iteration count targeting 100ms total measurement time (min 10, max 10000 iterations)
3. **Measurement:** N repetitions (default 7), each consisting of the calibrated iteration count
4. **Statistics:** Median time per iteration is the primary metric; also reports min, max, mean, stddev, CV%

Timer: `clock_gettime(CLOCK_MONOTONIC)` on Linux, `mach_absolute_time()` on macOS.

## Regression Detection Thresholds

| Ratio | Interpretation | Action |
|-------|---------------|--------|
| <= 1.10 | Within tolerance | PASS |
| 1.10 - 1.25 | Possible regression | FAIL if min > baseline median, else WARNING |
| 1.25 - 3.00 | Likely regression | FAIL if min > baseline median, else WARNING |
| > 3.00 | Likely C fallback | FAIL (critical) |

Additional checks:
- SIMD speedup < 1.5x vs C: WARNING
- SIMD speedup < 1.1x vs C: FAIL
- CV% > 5%: Measurement skipped for pass/fail (noisy)

## Design Decisions

- **Custom C harness vs google/benchmark:** Chose custom C harness to match the pure-C project with zero external dependencies.
- **benchmark() vs test():** Registered as `benchmark()` so `ninja test` is not slowed by benchmark runs.
- **Median vs mean:** Median is more robust to outliers in noisy CI environments.
- **Buffer reset between iterations:** VIF functions modify their buffers in-place, so buffers are restored from a saved copy before each call.
- **1920x1080 input:** Full HD is representative of production workloads and gives SIMD paths enough data to demonstrate their advantage.
