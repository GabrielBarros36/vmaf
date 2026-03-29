# Performance Regression Tests — Specification

## 1. Objective

Detect performance regressions in SIMD-optimized code paths by benchmarking each SIMD function against the C reference implementation on fixed, deterministic inputs and comparing wall-clock timings to stored baselines.

A >10% slowdown in any SIMD function relative to its baseline signals either an accidental fallback to the C path, a code change that defeated vectorization, or an unintended algorithmic regression. This test suite must catch such regressions before they reach production.

---

## 2. Scope — Functions Under Benchmark

The same **9 dispatch points** from the SIMD correctness oracle (Section 2 of `SIMD-correctness-oracle.md`) are benchmarked. Each function is measured in both its C reference form and every available SIMD variant for the build target.

### 2.1 ADM Module

| # | Function | Variants | Benchmark ID |
|---|----------|----------|--------------|
| 1 | `adm_dwt2_8` | C, AVX2, NEON | `adm_dwt2_8` |

### 2.2 VIF Module

| # | Function | Variants | Benchmark ID |
|---|----------|----------|--------------|
| 2 | `subsample_rd_8` | C, AVX2, AVX-512, NEON | `vif_subsample_rd_8` |
| 3 | `subsample_rd_16` | C, AVX2, AVX-512, NEON | `vif_subsample_rd_16` |
| 4 | `vif_statistic_8` | C, AVX2, AVX-512, NEON | `vif_statistic_8` |
| 5 | `vif_statistic_16` | C, AVX2, AVX-512, NEON | `vif_statistic_16` |

### 2.3 Motion Module

| # | Function | Variants | Benchmark ID |
|---|----------|----------|--------------|
| 6 | `x_convolution_16` | C, AVX2, AVX-512 | `motion_x_convolution_16` |

### 2.4 CAMBI Module

| # | Function | Variants | Benchmark ID |
|---|----------|----------|--------------|
| 7 | `increment_range` | C, AVX2 | `cambi_increment_range` |
| 8 | `decrement_range` | C, AVX2 | `cambi_decrement_range` |
| 9 | `get_derivative_data_for_row` | C, AVX2 | `cambi_derivative_row` |

**Total benchmark cases per platform:**
- x86-64 with AVX-512: 9 C + 9 AVX2 + 5 AVX-512 = **23 benchmarks**
- x86-64 without AVX-512: 9 C + 9 AVX2 = **18 benchmarks**
- ARM64: 9 C + 5 NEON = **14 benchmarks**

---

## 3. Benchmark Framework Choice

### 3.1 Decision: Custom Lightweight C Harness (Not google/benchmark)

The test harness plan suggests `google/benchmark` or similar. After analysis, a **custom C micro-benchmark harness** is the correct choice for this project, for the following reasons:

| Factor | google/benchmark | Custom C Harness |
|--------|-----------------|------------------|
| Language | C++ library | Pure C (matches libvmaf) |
| Build integration | Requires C++ dependency, subproject, or wrap | Zero dependencies beyond libc |
| Meson integration | Needs `cmake` or `wrap` dependency | Compiles directly as a C executable |
| Output format | Console + JSON | JSON (custom, purpose-built) |
| Warm-up / iterations | Automatic | Manual (simple, transparent) |
| Statistical analysis | Built-in | Custom (median, MAD, percentiles) |
| Binary size | Large (~500KB) | Trivial (~10KB) |
| Existing pattern | No C++ in test suite | Matches existing Minunit-style tests |

**Justification:** libvmaf is a pure C project. Its test suite uses a minimal Minunit-style framework with zero external test dependencies. Introducing a C++ dependency solely for benchmarking would be inconsistent and add build complexity. A lightweight custom harness of ~200 lines provides the same core functionality (warm-up, iteration count, statistical aggregation, JSON output) without any external dependency.

### 3.2 Measurement Methodology

Each benchmark follows this protocol:

1. **Warm-up phase:** Run the function 3 times to prime instruction caches and branch predictors. Discard results.
2. **Calibration phase:** Time a single iteration to estimate per-call cost. Use this to compute the iteration count needed for at least 100ms of total measurement time per benchmark (minimum 10 iterations, maximum 10,000).
3. **Measurement phase:** Execute `N` repetitions, each consisting of `iters` iterations. Record the wall-clock time for each repetition. Default: `N = 7` repetitions.
4. **Reporting:** Compute median, minimum, maximum, mean, standard deviation, and coefficient of variation (CV%) across the `N` repetitions. Report the **median time per iteration** as the primary metric.

**Timer selection (platform-dependent):**

| Platform | Timer | Resolution |
|----------|-------|------------|
| Linux | `clock_gettime(CLOCK_MONOTONIC)` | ~1 ns |
| macOS | `mach_absolute_time()` | ~1 ns |
| Windows | `QueryPerformanceCounter()` | ~100 ns |

---

## 4. Benchmark Harness Implementation

### 4.1 File Organization

```
libvmaf/test/
  bench_simd_perf.c          # Benchmark harness main (all 9 functions)
  bench_simd_perf.h          # Benchmark framework utilities
  test_simd_common.h         # Shared input generators (reused from oracle tests)
```

### 4.2 Benchmark Framework Header (`bench_simd_perf.h`)

```c
#ifndef BENCH_SIMD_PERF_H_
#define BENCH_SIMD_PERF_H_

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#ifdef __APPLE__
#include <mach/mach_time.h>
#endif

/* ========== High-Resolution Timer ========== */

static inline uint64_t bench_now_ns(void) {
#if defined(__linux__)
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
#elif defined(__APPLE__)
    static mach_timebase_info_data_t tb;
    if (tb.denom == 0) mach_timebase_info(&tb);
    return mach_absolute_time() * tb.numer / tb.denom;
#elif defined(_WIN32)
    LARGE_INTEGER freq, ctr;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&ctr);
    return (uint64_t)(ctr.QuadPart * 1000000000ULL / freq.QuadPart);
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
#endif
}

/* ========== Statistics ========== */

#define BENCH_MAX_REPS 31

typedef struct {
    char name[128];           /* e.g., "vif_subsample_rd_8_avx2" */
    char variant[32];         /* e.g., "avx2", "c", "avx512", "neon" */
    uint64_t times_ns[BENCH_MAX_REPS]; /* per-repetition times (total, not per-iter) */
    int num_reps;
    int iters_per_rep;
    double median_ns;         /* median time per single iteration */
    double min_ns;
    double max_ns;
    double mean_ns;
    double stddev_ns;
    double cv_pct;            /* coefficient of variation as percentage */
} BenchResult;

static int cmp_uint64(const void *a, const void *b) {
    uint64_t va = *(const uint64_t *)a;
    uint64_t vb = *(const uint64_t *)b;
    return (va > vb) - (va < vb);
}

static inline void bench_compute_stats(BenchResult *r) {
    int n = r->num_reps;
    int iters = r->iters_per_rep;

    /* Sort times for median */
    uint64_t sorted[BENCH_MAX_REPS];
    memcpy(sorted, r->times_ns, n * sizeof(uint64_t));
    qsort(sorted, n, sizeof(uint64_t), cmp_uint64);

    r->median_ns = (double)sorted[n / 2] / iters;
    r->min_ns    = (double)sorted[0] / iters;
    r->max_ns    = (double)sorted[n - 1] / iters;

    double sum = 0;
    for (int i = 0; i < n; i++)
        sum += (double)r->times_ns[i] / iters;
    r->mean_ns = sum / n;

    double var = 0;
    for (int i = 0; i < n; i++) {
        double v = (double)r->times_ns[i] / iters - r->mean_ns;
        var += v * v;
    }
    r->stddev_ns = sqrt(var / n);
    r->cv_pct = (r->mean_ns > 0) ? 100.0 * r->stddev_ns / r->mean_ns : 0;
}

/* ========== Benchmark Runner ========== */

typedef void (*BenchFunc)(void *ctx);

static inline void bench_run(BenchResult *r, BenchFunc fn, void *ctx,
                              const char *name, const char *variant,
                              int num_reps)
{
    snprintf(r->name, sizeof(r->name), "%s", name);
    snprintf(r->variant, sizeof(r->variant), "%s", variant);

    if (num_reps > BENCH_MAX_REPS) num_reps = BENCH_MAX_REPS;

    /* Warm-up: 3 iterations */
    for (int i = 0; i < 3; i++) fn(ctx);

    /* Calibrate: determine iterations for ~100ms of work */
    uint64_t t0 = bench_now_ns();
    fn(ctx);
    uint64_t single_ns = bench_now_ns() - t0;
    if (single_ns == 0) single_ns = 1;

    int iters = (int)(100000000ULL / single_ns); /* target 100ms */
    if (iters < 10) iters = 10;
    if (iters > 10000) iters = 10000;
    r->iters_per_rep = iters;

    /* Measurement */
    r->num_reps = num_reps;
    for (int rep = 0; rep < num_reps; rep++) {
        uint64_t start = bench_now_ns();
        for (int i = 0; i < iters; i++) {
            fn(ctx);
        }
        r->times_ns[rep] = bench_now_ns() - start;
    }

    bench_compute_stats(r);
}

/* ========== JSON Output ========== */

static inline void bench_result_to_json(const BenchResult *r, FILE *fp) {
    fprintf(fp,
        "    {\n"
        "      \"name\": \"%s\",\n"
        "      \"variant\": \"%s\",\n"
        "      \"iters_per_rep\": %d,\n"
        "      \"num_reps\": %d,\n"
        "      \"median_ns\": %.2f,\n"
        "      \"min_ns\": %.2f,\n"
        "      \"max_ns\": %.2f,\n"
        "      \"mean_ns\": %.2f,\n"
        "      \"stddev_ns\": %.2f,\n"
        "      \"cv_pct\": %.4f\n"
        "    }",
        r->name, r->variant,
        r->iters_per_rep, r->num_reps,
        r->median_ns, r->min_ns, r->max_ns,
        r->mean_ns, r->stddev_ns, r->cv_pct);
}

#endif /* BENCH_SIMD_PERF_H_ */
```

### 4.3 Benchmark Main File (`bench_simd_perf.c`)

The benchmark executable benchmarks all 9 dispatch points. Each function gets one benchmark per variant (C, AVX2, AVX-512, NEON as applicable).

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bench_simd_perf.h"
#include "test_simd_common.h"
#include "config.h"

/* Feature module headers */
#include "feature/integer_adm.h"
#include "feature/integer_vif.h"

#if ARCH_X86
#include "feature/x86/adm_avx2.h"
#include "feature/x86/vif_avx2.h"
#include "feature/x86/motion_avx2.h"
#include "feature/x86/cambi_avx2.h"
#if HAVE_AVX512
#include "feature/x86/vif_avx512.h"
#include "feature/x86/motion_avx512.h"
#endif
#endif

#if ARCH_AARCH64
#include "feature/arm64/adm_neon.h"
#include "feature/arm64/vif_neon.h"
#endif

/* ========== Benchmark Input Dimensions ========== */
/* Use 1920x1080 as the standard benchmark dimension.
 * This represents a realistic production workload and
 * ensures SIMD paths have enough data to show their advantage. */
#define BENCH_W 1920
#define BENCH_H 1080

/* ========== Per-Function Benchmark Contexts ========== */

typedef struct {
    uint8_t  *src;
    adm_dwt_band_t dst;
    AdmBuffer *buf;
    int w, h, src_stride, dst_stride;
} AdmDwt2Ctx;

static void bench_adm_dwt2_8_c(void *ctx) {
    AdmDwt2Ctx *c = (AdmDwt2Ctx *)ctx;
    adm_dwt2_8(c->src, &c->dst, c->buf, c->w, c->h,
               c->src_stride, c->dst_stride);
}

#if ARCH_X86
static void bench_adm_dwt2_8_avx2(void *ctx) {
    AdmDwt2Ctx *c = (AdmDwt2Ctx *)ctx;
    adm_dwt2_8_avx2(c->src, &c->dst, c->buf, c->w, c->h,
                     c->src_stride, c->dst_stride);
}
#endif

#if ARCH_AARCH64
static void bench_adm_dwt2_8_neon(void *ctx) {
    AdmDwt2Ctx *c = (AdmDwt2Ctx *)ctx;
    adm_dwt2_8_neon(c->src, &c->dst, c->buf, c->w, c->h,
                     c->src_stride, c->dst_stride);
}
#endif

/* ... (similar wrappers for all 9 functions and their SIMD variants) ... */

/* ========== Main ========== */

int main(int argc, char **argv) {
    const char *output_file = NULL;
    int num_reps = 7;

    /* Parse arguments */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--output") == 0 && i + 1 < argc)
            output_file = argv[++i];
        else if (strcmp(argv[i], "--reps") == 0 && i + 1 < argc)
            num_reps = atoi(argv[++i]);
        else if (strcmp(argv[i], "--help") == 0) {
            fprintf(stderr,
                "Usage: bench_simd_perf [--output FILE] [--reps N]\n"
                "  --output FILE   Write JSON results to FILE (default: stdout)\n"
                "  --reps N        Number of repetitions per benchmark (default: 7)\n");
            return 0;
        }
    }

    FILE *out = output_file ? fopen(output_file, "w") : stdout;
    if (!out) { perror("fopen"); return 1; }

    /* Allocate and fill benchmark input buffers with random_42 pattern */
    /* ... (setup code for each function's context) ... */

    BenchResult results[64];
    int n_results = 0;

    /* --- ADM dwt2_8 --- */
    bench_run(&results[n_results++], bench_adm_dwt2_8_c, &adm_ctx,
              "adm_dwt2_8", "c", num_reps);
#if ARCH_X86
    bench_run(&results[n_results++], bench_adm_dwt2_8_avx2, &adm_ctx,
              "adm_dwt2_8", "avx2", num_reps);
#endif
#if ARCH_AARCH64
    bench_run(&results[n_results++], bench_adm_dwt2_8_neon, &adm_ctx,
              "adm_dwt2_8", "neon", num_reps);
#endif

    /* ... (repeat for all 9 functions) ... */

    /* Write JSON output */
    fprintf(out, "{\n  \"benchmarks\": [\n");
    for (int i = 0; i < n_results; i++) {
        if (i > 0) fprintf(out, ",\n");
        bench_result_to_json(&results[i], out);
    }
    fprintf(out, "\n  ]\n}\n");

    if (output_file) fclose(out);

    /* Print summary table to stderr */
    fprintf(stderr, "\n%-35s %12s %12s %12s %8s\n",
            "Benchmark", "Median(ns)", "Min(ns)", "Max(ns)", "CV%");
    fprintf(stderr, "%-35s %12s %12s %12s %8s\n",
            "-----------------------------------",
            "------------", "------------", "------------", "--------");
    for (int i = 0; i < n_results; i++) {
        char label[80];
        snprintf(label, sizeof(label), "%s [%s]", results[i].name, results[i].variant);
        fprintf(stderr, "%-35s %12.1f %12.1f %12.1f %7.2f%%\n",
                label, results[i].median_ns, results[i].min_ns,
                results[i].max_ns, results[i].cv_pct);
    }

    /* Compute and print speedup ratios */
    fprintf(stderr, "\nSpeedup vs. C reference:\n");
    for (int i = 0; i < n_results; i++) {
        if (strcmp(results[i].variant, "c") == 0) continue;
        /* Find matching C baseline */
        for (int j = 0; j < n_results; j++) {
            if (strcmp(results[j].variant, "c") == 0 &&
                strcmp(results[j].name, results[i].name) == 0) {
                double speedup = results[j].median_ns / results[i].median_ns;
                fprintf(stderr, "  %s [%s]: %.2fx\n",
                        results[i].name, results[i].variant, speedup);
                break;
            }
        }
    }

    /* Free benchmark contexts */
    /* ... */

    return 0;
}
```

---

## 5. Benchmark Input Data

### 5.1 Fixed Input Specification

All benchmarks use a single, deterministic input configuration to ensure reproducibility:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Width | 1920 | Full HD — representative of production workloads |
| Height | 1080 | Full HD |
| Input pattern | `INPUT_RANDOM_42` (xorshift32, seed=42) | Deterministic; exercises all code paths; avoids zero-shortcut optimizations |
| Bit depth (16-bit functions) | 10 | Most common HDR bit depth |
| Scale (VIF) | 0 | First scale level (largest data) |
| Stride alignment | 32-byte | Matches `VMAF_ALIGNMENT` |

### 5.2 Input Reuse

The same input buffer is reused across iterations within a benchmark. This is intentional: we are measuring **compute throughput**, not memory allocation. The input is allocated once during setup, filled once, and the same data is processed on every iteration.

### 5.3 Output Buffer Management

Output buffers are pre-allocated before benchmarking and reused across iterations. Output data is overwritten each iteration but never read between iterations (preventing the compiler from optimizing away the computation is handled by marking the output pointer as `volatile` in the benchmark wrapper or using a `benchmark_clobber()` barrier).

```c
/* Prevent dead code elimination */
static inline void bench_clobber(void *p) {
    __asm__ volatile("" : : "r"(p) : "memory");
}
```

---

## 6. Baseline Storage Format

### 6.1 Baseline File Location

```
libvmaf/test/baselines/
  perf_baseline_linux_x86_64.json
  perf_baseline_linux_aarch64.json
  perf_baseline_macos_x86_64.json
  perf_baseline_macos_aarch64.json
  perf_baseline_windows_x86_64.json
```

Baselines are stored **in the repository** and updated via a deliberate PR workflow (Section 8.4). They are not auto-updated by CI.

### 6.2 Baseline JSON Schema

```json
{
  "schema_version": 1,
  "platform": {
    "os": "linux",
    "arch": "x86_64",
    "cpu_model": "AMD EPYC 7763 64-Core Processor",
    "runner_label": "ubuntu-latest"
  },
  "generated_at": "2026-03-08T12:00:00Z",
  "commit": "332dde62abc...",
  "benchmarks": [
    {
      "name": "adm_dwt2_8",
      "variant": "c",
      "median_ns": 1523400.00,
      "min_ns": 1498200.00,
      "max_ns": 1571800.00,
      "mean_ns": 1530100.00,
      "stddev_ns": 22100.00,
      "cv_pct": 1.44,
      "iters_per_rep": 65,
      "num_reps": 7
    },
    {
      "name": "adm_dwt2_8",
      "variant": "avx2",
      "median_ns": 312700.00,
      "min_ns": 308100.00,
      "max_ns": 321400.00,
      "mean_ns": 314200.00,
      "stddev_ns": 4100.00,
      "cv_pct": 1.30,
      "iters_per_rep": 320,
      "num_reps": 7
    }
  ]
}
```

### 6.3 Field Definitions

| Field | Type | Description |
|-------|------|-------------|
| `schema_version` | int | Schema version for forward compatibility. Current: `1`. |
| `platform.os` | string | Operating system: `linux`, `macos`, `windows`. |
| `platform.arch` | string | CPU architecture: `x86_64`, `aarch64`. |
| `platform.cpu_model` | string | CPU model string from `/proc/cpuinfo` or equivalent. |
| `platform.runner_label` | string | GitHub Actions runner label (e.g., `ubuntu-latest`). |
| `generated_at` | string | ISO 8601 timestamp of baseline generation. |
| `commit` | string | Git commit hash the baseline was generated from. |
| `benchmarks[].name` | string | Function name (matches the Benchmark ID from Section 2). |
| `benchmarks[].variant` | string | Implementation variant: `c`, `avx2`, `avx512`, `neon`. |
| `benchmarks[].median_ns` | float | Median time per iteration in nanoseconds (primary comparison metric). |
| `benchmarks[].min_ns` | float | Minimum time per iteration across repetitions. |
| `benchmarks[].max_ns` | float | Maximum time per iteration across repetitions. |
| `benchmarks[].mean_ns` | float | Mean time per iteration. |
| `benchmarks[].stddev_ns` | float | Standard deviation of time per iteration. |
| `benchmarks[].cv_pct` | float | Coefficient of variation as a percentage. |
| `benchmarks[].iters_per_rep` | int | Number of iterations in each repetition. |
| `benchmarks[].num_reps` | int | Number of repetitions measured. |

---

## 7. Regression Detection Logic

### 7.1 Primary Detection: Median Comparison

For each benchmark function+variant pair, compare the current run's median to the stored baseline's median:

```
regression_ratio = current_median_ns / baseline_median_ns
```

| Condition | Interpretation | Action |
|-----------|---------------|--------|
| `regression_ratio <= 1.10` | Within tolerance | PASS |
| `1.10 < regression_ratio <= 1.25` | Possible regression | WARNING |
| `regression_ratio > 1.25` | Clear regression | FAIL |
| `regression_ratio > 3.00` | Likely fallback to C path | FAIL (critical) |

The primary threshold is **10%** (`regression_ratio > 1.10`). The 25% and 3x thresholds provide additional diagnostic context.

### 7.2 Statistical Significance Guard

A raw median comparison can produce false positives when measurement noise is high. To guard against this, a regression is only flagged as a FAIL if **both** conditions are met:

1. `regression_ratio > 1.10` (median exceeded threshold)
2. `current_min_ns > baseline_median_ns` (even the best run is slower than the baseline median)

If condition 1 is met but condition 2 is not, the result is downgraded to WARNING (noisy measurement, not a confirmed regression).

### 7.3 SIMD-vs-C Speedup Floor

In addition to baseline comparison, the harness checks that each SIMD variant is **at least 1.5x faster** than its corresponding C reference in the same run. This catches the pathological case where both baseline and current are slow (e.g., both accidentally using the C path):

```
speedup = c_median_ns / simd_median_ns
if speedup < 1.5:
    WARNING: SIMD variant may not be using vectorized path
if speedup < 1.1:
    FAIL: SIMD variant shows no speedup over C reference
```

### 7.4 CV% Quality Gate

If any benchmark's coefficient of variation exceeds **5%**, the harness emits a WARNING that the measurement environment is too noisy for reliable regression detection. The benchmark result is still reported but not used for pass/fail determination.

### 7.5 Regression Detection Script

A Python script `scripts/check_perf_regression.py` performs the comparison:

```python
#!/usr/bin/env python3
"""Compare benchmark results against stored baselines."""

import json
import sys
import os

REGRESSION_THRESHOLD = 1.10   # 10%
WARNING_THRESHOLD = 1.25      # 25%
FALLBACK_THRESHOLD = 3.00     # 3x (likely C fallback)
MIN_SPEEDUP = 1.5             # SIMD must be at least 1.5x faster than C
CV_QUALITY_GATE = 5.0         # CV% above this = noisy measurement

def load_json(path):
    with open(path) as f:
        return json.load(f)

def find_bench(benchmarks, name, variant):
    for b in benchmarks:
        if b["name"] == name and b["variant"] == variant:
            return b
    return None

def check_regression(baseline_path, current_path):
    baseline = load_json(baseline_path)
    current = load_json(current_path)

    exit_code = 0
    warnings = []
    failures = []

    current_benchmarks = current["benchmarks"]
    baseline_benchmarks = baseline["benchmarks"]

    for cb in current_benchmarks:
        name, variant = cb["name"], cb["variant"]
        bb = find_bench(baseline_benchmarks, name, variant)
        if bb is None:
            warnings.append(f"{name} [{variant}]: no baseline found (new benchmark?)")
            continue

        ratio = cb["median_ns"] / bb["median_ns"]
        cv = cb.get("cv_pct", 0)

        if cv > CV_QUALITY_GATE:
            warnings.append(
                f"{name} [{variant}]: CV={cv:.1f}% exceeds {CV_QUALITY_GATE}% "
                f"quality gate (noisy measurement, skipping pass/fail)")
            continue

        if ratio > FALLBACK_THRESHOLD:
            failures.append(
                f"{name} [{variant}]: {ratio:.2f}x slower than baseline "
                f"(CRITICAL: likely fallback to C path)")
            exit_code = 1
        elif ratio > WARNING_THRESHOLD:
            # Check statistical significance
            if cb["min_ns"] > bb["median_ns"]:
                failures.append(
                    f"{name} [{variant}]: {ratio:.2f}x slower than baseline "
                    f"(confirmed: min > baseline median)")
                exit_code = 1
            else:
                warnings.append(
                    f"{name} [{variant}]: {ratio:.2f}x slower than baseline "
                    f"(not confirmed: min <= baseline median)")
        elif ratio > REGRESSION_THRESHOLD:
            if cb["min_ns"] > bb["median_ns"]:
                failures.append(
                    f"{name} [{variant}]: {ratio:.2f}x slower than baseline")
                exit_code = 1
            else:
                warnings.append(
                    f"{name} [{variant}]: {ratio:.2f}x slower but within noise "
                    f"(min <= baseline median)")

    # Check SIMD-vs-C speedup floor
    for cb in current_benchmarks:
        if cb["variant"] == "c":
            continue
        c_bench = find_bench(current_benchmarks, cb["name"], "c")
        if c_bench is None:
            continue
        speedup = c_bench["median_ns"] / cb["median_ns"]
        if speedup < 1.1:
            failures.append(
                f"{cb['name']} [{cb['variant']}]: speedup vs C = {speedup:.2f}x "
                f"(FAIL: no measurable speedup)")
            exit_code = 1
        elif speedup < MIN_SPEEDUP:
            warnings.append(
                f"{cb['name']} [{cb['variant']}]: speedup vs C = {speedup:.2f}x "
                f"(below expected {MIN_SPEEDUP}x)")

    # Report
    for w in warnings:
        print(f"WARNING: {w}")
    for f in failures:
        print(f"FAIL: {f}")
    if not warnings and not failures:
        print("PASS: All benchmarks within tolerance")

    return exit_code

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <baseline.json> <current.json>")
        sys.exit(2)
    sys.exit(check_regression(sys.argv[1], sys.argv[2]))
```

---

## 8. CI Integration

### 8.1 Workflow: Performance Regression Checks

A dedicated GitHub Actions workflow `.github/workflows/perf-regression.yml` runs benchmarks and compares against baselines.

```yaml
name: Performance Regression

on:
  pull_request:
  push:
    branches: [master]

env:
  DEBIAN_FRONTEND: noninteractive

jobs:
  perf-bench-x86:
    name: Benchmark (x86_64)
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v6

      - name: Install dependencies
        run: |
          sudo apt-get update
          sudo -E apt-get -yq install ninja-build nasm gcc g++
          pip install meson

      - name: Check AVX-512 support
        id: avx512
        run: |
          if grep -q avx512 /proc/cpuinfo; then
            echo "supported=true" >> $GITHUB_OUTPUT
          else
            echo "supported=false" >> $GITHUB_OUTPUT
          fi

      - name: Configure
        run: |
          AVX512_FLAG=""
          if [ "${{ steps.avx512.outputs.supported }}" = "true" ]; then
            AVX512_FLAG="-Denable_avx512=true"
          fi
          meson setup libvmaf libvmaf/build --buildtype release \
            -Denable_float=true $AVX512_FLAG

      - name: Build
        run: ninja -vC libvmaf/build

      - name: Run benchmarks
        run: |
          libvmaf/build/test/bench_simd_perf \
            --output bench_current.json --reps 7

      - name: Check for regressions
        run: |
          BASELINE="libvmaf/test/baselines/perf_baseline_linux_x86_64.json"
          if [ -f "$BASELINE" ]; then
            python3 scripts/check_perf_regression.py \
              "$BASELINE" bench_current.json
          else
            echo "WARNING: No baseline file found. Skipping regression check."
            echo "Run with --update-baseline to generate one."
          fi

      - name: Upload benchmark results
        uses: actions/upload-artifact@v5
        if: always()
        with:
          name: bench-results-linux-x86_64
          path: bench_current.json

  perf-bench-arm:
    name: Benchmark (ARM64)
    runs-on: ubuntu-24.04-arm
    steps:
      - uses: actions/checkout@v6

      - name: Install dependencies
        run: |
          sudo apt-get update
          sudo -E apt-get -yq install ninja-build gcc g++
          pip install meson

      - name: Configure
        run: |
          meson setup libvmaf libvmaf/build --buildtype release \
            -Denable_float=true

      - name: Build
        run: ninja -vC libvmaf/build

      - name: Run benchmarks
        run: |
          libvmaf/build/test/bench_simd_perf \
            --output bench_current.json --reps 7

      - name: Check for regressions
        run: |
          BASELINE="libvmaf/test/baselines/perf_baseline_linux_aarch64.json"
          if [ -f "$BASELINE" ]; then
            python3 scripts/check_perf_regression.py \
              "$BASELINE" bench_current.json
          else
            echo "WARNING: No baseline file found. Skipping regression check."
          fi

      - name: Upload benchmark results
        uses: actions/upload-artifact@v5
        if: always()
        with:
          name: bench-results-linux-aarch64
          path: bench_current.json
```

### 8.2 CI Runner Considerations

| Concern | Mitigation |
|---------|------------|
| Runner hardware varies | Baselines are per-runner-label; the 10% threshold accounts for cloud VM variability |
| Other jobs on the same runner | Use `CLOCK_MONOTONIC` (not `CLOCK_PROCESS_CPUTIME_ID`) to measure wall time; the warm-up and multi-rep median absorb transient load |
| Runner migration (GitHub changes hardware) | When runner hardware changes, baselines must be regenerated (Section 8.4) |
| Benchmark too slow for CI | At 1920x1080 with calibrated iteration counts, total benchmark time is ~2-5 minutes per platform |

### 8.3 Non-Blocking CI

The performance regression workflow is **not** a required check for PR merge. It runs on every PR and push to `master` but failures produce **warnings in the PR comment**, not merge-blocking failures. This avoids blocking PRs due to transient CI noise.

Exception: a `regression_ratio > 3.0` (likely C fallback) is always a blocking failure.

### 8.4 Baseline Update Procedure

Baselines are updated through a deliberate process, not automatically:

1. A maintainer runs the benchmark on the CI runner (or triggers a dedicated workflow dispatch).
2. The workflow uploads the JSON result as an artifact.
3. The maintainer downloads the artifact, reviews it, and commits it to `libvmaf/test/baselines/` in a dedicated PR.
4. The PR description must explain why the baseline is being updated (e.g., "Runner hardware changed", "Intentional algorithm change", "Initial baseline generation").

**Workflow dispatch for baseline generation:**

```yaml
  update-baseline:
    if: github.event_name == 'workflow_dispatch'
    runs-on: ubuntu-latest
    steps:
      # ... (same build steps) ...
      - name: Generate baseline
        run: |
          libvmaf/build/test/bench_simd_perf \
            --output perf_baseline_linux_x86_64.json --reps 15
      - name: Upload baseline
        uses: actions/upload-artifact@v5
        with:
          name: perf-baseline-linux-x86_64
          path: perf_baseline_linux_x86_64.json
```

---

## 9. Build System Integration

### 9.1 Meson Configuration

Add to `libvmaf/test/meson.build`:

```meson
# Performance Regression Benchmarks
bench_simd_perf = executable('bench_simd_perf',
    ['bench_simd_perf.c', '../src/mem.c'],
    include_directories : [libvmaf_inc, test_inc, simd_oracle_src_inc],
    link_with : get_option('default_library') == 'both' ? libvmaf.get_static_lib() : libvmaf,
    dependencies : [math_lib, cuda_dependency],
    objects : [
        platform_specific_cpu_objects,
    ],
    c_args : ['-O2'],
)

# Register as a benchmark (not a test — does not run during `ninja test`)
benchmark('bench_simd_perf', bench_simd_perf,
    timeout: 600,
    args: ['--reps', '7'],
)
```

**Key design decision:** The benchmark is registered with Meson's `benchmark()` function, not `test()`. This means:
- `ninja test` does **not** run benchmarks (benchmarks are slow and noisy).
- `ninja benchmark` runs benchmarks explicitly.
- CI workflows invoke the benchmark executable directly for full control over arguments and output.

### 9.2 Meson Options

No new Meson options are required. The benchmark executable is always built when `enable_tests=true`. It simply does not run automatically.

---

## 10. Local Development Usage

### 10.1 Running Benchmarks Locally

```bash
# Build
cd libvmaf
meson setup build --buildtype release -Denable_float=true
ninja -C build

# Run all benchmarks, print summary to stderr, JSON to stdout
./build/test/bench_simd_perf

# Save results to a file
./build/test/bench_simd_perf --output my_results.json

# Increase repetitions for more stable results
./build/test/bench_simd_perf --output my_results.json --reps 15

# Compare against baseline
python3 scripts/check_perf_regression.py \
    test/baselines/perf_baseline_linux_x86_64.json my_results.json
```

### 10.2 A/B Comparison Workflow

To compare performance between two branches:

```bash
# Benchmark branch A
git checkout main
meson setup build_a --buildtype release -Denable_float=true
ninja -C build_a
./build_a/test/bench_simd_perf --output results_a.json --reps 15

# Benchmark branch B
git checkout feature-branch
meson setup build_b --buildtype release -Denable_float=true
ninja -C build_b
./build_b/test/bench_simd_perf --output results_b.json --reps 15

# Compare (use branch A as "baseline")
python3 scripts/check_perf_regression.py results_a.json results_b.json
```

### 10.3 Best Practices for Local Benchmarking

| Practice | Rationale |
|----------|-----------|
| Build with `--buildtype release` | Debug builds are 5-10x slower; not representative |
| Close other applications | Reduces measurement noise |
| Use `--reps 15` or higher locally | More repetitions yield more stable medians |
| Pin CPU frequency if possible | `cpupower frequency-set -g performance` on Linux |
| Disable turbo boost | `echo 1 > /sys/devices/system/cpu/intel_pstate/no_turbo` |
| Run multiple times | If CV% > 3%, results are suspect |

---

## 11. Reporting Format

### 11.1 Console Output (stderr)

The benchmark harness prints a human-readable table to stderr:

```
Benchmark                            Median(ns)    Min(ns)     Max(ns)      CV%
-----------------------------------  ------------ ------------ ------------ --------
adm_dwt2_8 [c]                       1523400.0    1498200.0    1571800.0    1.44%
adm_dwt2_8 [avx2]                     312700.0     308100.0     321400.0    1.30%
vif_subsample_rd_8 [c]                 891200.0     885400.0     903100.0    0.72%
vif_subsample_rd_8 [avx2]              198400.0     195100.0     204300.0    1.61%
vif_subsample_rd_8 [avx512]            142100.0     139800.0     147200.0    1.82%
...

Speedup vs. C reference:
  adm_dwt2_8 [avx2]: 4.87x
  vif_subsample_rd_8 [avx2]: 4.49x
  vif_subsample_rd_8 [avx512]: 6.27x
  ...
```

### 11.2 JSON Output (file/stdout)

See Section 6.2 for the JSON schema. The JSON output is the machine-readable format consumed by the regression detection script.

### 11.3 Regression Check Output

The regression detection script (`check_perf_regression.py`) prints pass/fail/warning messages:

```
PASS: All benchmarks within tolerance
```

or:

```
WARNING: motion_x_convolution_16 [avx2]: 1.12x slower but within noise (min <= baseline median)
FAIL: vif_subsample_rd_8 [avx2]: 3.41x slower than baseline (CRITICAL: likely fallback to C path)
FAIL: vif_subsample_rd_8 [avx2]: speedup vs C = 1.02x (FAIL: no measurable speedup)
```

---

## 12. Completion Requirements

The performance regression test suite is **complete** when ALL of the following requirements are met:

### R1. Full Function Coverage
All 9 dispatch points from Section 2 have benchmarks for the C reference and every SIMD variant available on the build target.

**Verification:** On x86-64 with AVX-512, the benchmark produces 23 result entries. On x86-64 without AVX-512, 18 entries. On ARM64, 14 entries.

### R2. Deterministic Input
All benchmarks use the same fixed input (1920x1080, `INPUT_RANDOM_42`, seed=42). Running the benchmark twice on the same hardware produces results with CV% < 5%.

**Verification:** Run the benchmark 3 times. All 3 runs use identical input buffers (verified by checksumming the input). CV% for each benchmark is below 5% on a quiescent machine.

### R3. Statistical Robustness
Each benchmark performs warm-up (3 iterations), calibrated iteration counts (targeting 100ms per repetition), and multiple repetitions (default 7). The primary metric is the median, not the mean.

**Verification:** Code review confirms warm-up, calibration, and median computation. The `iters_per_rep` field in the output is at least 10 for every benchmark.

### R4. Baseline Files Exist
Baseline JSON files exist for at least two platforms: `linux_x86_64` and `linux_aarch64`. Each baseline contains entries for all functions and variants available on that platform.

**Verification:** Baseline files are committed to `libvmaf/test/baselines/` and parse successfully with `check_perf_regression.py`.

### R5. Regression Detection Logic
The regression detection script correctly identifies:
- PASS: current median within 10% of baseline median
- WARNING: current median 10-25% above baseline but min below baseline median
- FAIL: current median >10% above baseline and min above baseline median
- FAIL (critical): current median >3x baseline (likely C fallback)

**Verification:** Create synthetic baseline and current JSON files with known ratios. Verify the script produces the correct exit code and messages for each case.

### R6. SIMD Speedup Floor
The harness checks that each SIMD variant is at least 1.5x faster than the C reference in the same run. A speedup below 1.1x is a FAIL.

**Verification:** Run the benchmark. Every SIMD variant shows a speedup of at least 1.5x over C. Artificially force a function to use the C path and verify the speedup check detects it.

### R7. Build Integration
The benchmark executable compiles as part of the standard build (`ninja`), is registered as a Meson `benchmark()` target (not `test()`), and does not run during `ninja test`.

**Verification:** `ninja test` completes without running benchmarks. `ninja benchmark` or direct invocation runs benchmarks successfully.

### R8. CI Integration
A GitHub Actions workflow runs benchmarks on `ubuntu-latest` (x86_64) and `ubuntu-24.04-arm` (ARM64) for every PR and push to master. Regression results are reported as workflow output. Benchmark artifacts are uploaded.

**Verification:** The workflow runs successfully on a test PR, produces artifacts, and the regression check step produces meaningful output.

### R9. Local Usability
A developer can run benchmarks locally with a single command (`./build/test/bench_simd_perf`) and compare against baselines with a second command (`python3 scripts/check_perf_regression.py baseline.json current.json`).

**Verification:** Follow the instructions in Section 10.1 on a fresh build. The commands work without additional setup.

### R10. Console Reporting
The benchmark prints a human-readable summary table (with function names, median times, and speedup ratios) to stderr, in addition to the machine-readable JSON output.

**Verification:** Run the benchmark and visually inspect stderr output. The table includes all benchmarked functions with their timings and speedup ratios.

### R11. Non-Blocking CI (Except Critical)
The performance regression workflow is not a required PR check. Only `regression_ratio > 3.0` (likely C fallback) is flagged as a merge-blocking failure.

**Verification:** A PR with a 15% regression in one function can still be merged. A PR with a 4x regression blocks merge.

### R12. Zero Production Code Changes
The benchmark harness does not modify any production code. It links against the same library objects as the oracle tests and calls the same function symbols.

**Verification:** `git diff` of production source files (`libvmaf/src/`) shows no changes attributable to the benchmark harness.

---

## 13. Acceptance Criteria Summary

| Req | Description | Pass Condition |
|-----|-------------|----------------|
| R1 | Full function coverage | 23 benchmarks on x86-64+AVX512; 14 on ARM64 |
| R2 | Deterministic input | Same input every run; CV% < 5% on quiescent machine |
| R3 | Statistical robustness | Warm-up, calibration, 7+ reps, median-based |
| R4 | Baseline files | Committed baselines for linux_x86_64 and linux_aarch64 |
| R5 | Regression detection | Correct PASS/WARNING/FAIL for known ratios |
| R6 | SIMD speedup floor | All SIMD variants >= 1.5x faster than C reference |
| R7 | Build integration | `benchmark()` in Meson; not in `ninja test` |
| R8 | CI integration | Workflow runs on x86_64 and ARM64; uploads artifacts |
| R9 | Local usability | Single-command run and comparison |
| R10 | Console reporting | Human-readable table with timings and speedups |
| R11 | Non-blocking CI | Only 3x+ regression blocks merge |
| R12 | Zero production impact | No changes to production source files |

All 12 requirements must be met for the performance regression test suite to be considered complete.
