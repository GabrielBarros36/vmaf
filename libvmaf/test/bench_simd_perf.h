/**
 * SIMD Performance Benchmark Framework
 *
 * Lightweight C benchmark harness for measuring SIMD dispatch function
 * performance. Provides high-resolution timing, statistical analysis,
 * and JSON output formatting.
 */

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

/* ========== Compiler Barrier ========== */

/* Prevent dead code elimination */
static inline void bench_clobber(void *p) {
    __asm__ volatile("" : : "r"(p) : "memory");
}

/* ========== Statistics ========== */

#define BENCH_MAX_REPS 31

typedef struct {
    char name[128];           /* e.g., "vif_subsample_rd_8" */
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
    for (int i = 0; i < 3; i++) {
        fn(ctx);
        bench_clobber(ctx);
    }

    /* Calibrate: determine iterations for ~100ms of work */
    uint64_t t0 = bench_now_ns();
    fn(ctx);
    bench_clobber(ctx);
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
            bench_clobber(ctx);
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

/* ========== Console Table Output ========== */

static inline void bench_print_table(const BenchResult *results, int n_results,
                                      FILE *fp)
{
    fprintf(fp, "\n%-40s %12s %12s %12s %8s\n",
            "Benchmark", "Median(ns)", "Min(ns)", "Max(ns)", "CV%");
    fprintf(fp, "%-40s %12s %12s %12s %8s\n",
            "----------------------------------------",
            "------------", "------------", "------------", "--------");
    for (int i = 0; i < n_results; i++) {
        char label[164];
        snprintf(label, sizeof(label), "%.127s [%.31s]",
                 results[i].name, results[i].variant);
        fprintf(fp, "%-40s %12.1f %12.1f %12.1f %7.2f%%\n",
                label, results[i].median_ns, results[i].min_ns,
                results[i].max_ns, results[i].cv_pct);
    }

    /* Compute and print speedup ratios */
    fprintf(fp, "\nSpeedup vs. C reference:\n");
    for (int i = 0; i < n_results; i++) {
        if (strcmp(results[i].variant, "c") == 0) continue;
        /* Find matching C baseline */
        for (int j = 0; j < n_results; j++) {
            if (strcmp(results[j].variant, "c") == 0 &&
                strcmp(results[j].name, results[i].name) == 0) {
                double speedup = results[j].median_ns / results[i].median_ns;
                fprintf(fp, "  %-36s [%s]: %.2fx\n",
                        results[i].name, results[i].variant, speedup);
                break;
            }
        }
    }
    fprintf(fp, "\n");
}

#endif /* BENCH_SIMD_PERF_H_ */
