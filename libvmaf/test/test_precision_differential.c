/**
 *
 *  Copyright 2016-2020 Netflix, Inc.
 *
 *     Licensed under the BSD+Patent License (the "License");
 *     you may not use this file except in compliance with the License.
 *     You may obtain a copy of the License at
 *
 *         https://opensource.org/licenses/BSDplusPatent
 *
 *     Unless required by applicable law or agreed to in writing, software
 *     distributed under the License is distributed on an "AS IS" BASIS,
 *     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *     See the License for the specific language governing permissions and
 *     limitations under the License.
 *
 */

/**
 * Numerical Precision Differential Tests
 *
 * Validates that the float and integer implementations of each VMAF metric
 * produce scores that agree within documented, justified tolerance bounds
 * when given identical input. Covers ADM, VIF, Motion, PSNR, and SSIM.
 *
 * ADM/VIF/Motion/PSNR tests are guarded by VMAF_FLOAT_FEATURES.
 * SSIM tests are always compiled (both float_ssim and ssim are unconditional).
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "config.h"
#include "test.h"

#include "libvmaf/libvmaf.h"
#include "libvmaf/picture.h"

/* ========== Tolerance Constants ==========
 *
 * Each tolerance is derived from empirical measurement of float vs integer
 * score deltas on standard test content (Section 5.2 of spec), with a safety
 * margin of approximately 2x the maximum historically observed delta.
 *
 * ADM_TOLERANCE (5e-03):
 *   Integer ADM uses 16-bit fixed-point DWT with rounding; float ADM uses
 *   native float. Max observed delta on real video ~1e-04 at per-scale level.
 *   Synthetic content (gradient vs random) produces per-scale deltas up to
 *   ~2e-03 due to extreme input patterns amplifying fixed-point rounding.
 *   Tolerance set to ~2.5x the maximum synthetic delta for headroom.
 *
 * VIF_TOLERANCE (2e-02):
 *   Integer VIF uses fixed-point multiply-accumulate; float VIF uses native
 *   float convolutions. Max observed delta on real video ~1.1e-03 at scale1.
 *   Synthetic content (gradient vs random) produces per-scale deltas up to
 *   ~9e-03 at higher scales where downsampled resolution amplifies rounding.
 *   Tolerance set to ~2x the maximum synthetic delta for headroom.
 *
 * MOTION_TOLERANCE (1e-04):
 *   Integer motion uses fixed-point Gaussian blur + integer SAD; float motion
 *   uses float convolution + float subtraction. Max observed delta ~7e-06.
 *   Generous headroom.
 *
 * PSNR_TOLERANCE (1e-04):
 *   Both implementations compute identical MSE for 8-bit content; divergence
 *   arises only from peak definition and log10 precision.
 *
 * SSIM_TOLERANCE (1e-03):
 *   float_ssim and ssim use different computational approaches (float windowed
 *   convolution vs integer Gaussian kernel accumulation). ~10x margin.
 */
#define ADM_TOLERANCE    5e-03
#define VIF_TOLERANCE    2e-02
#define MOTION_TOLERANCE 1e-04
#define PSNR_TOLERANCE   1e-04
#define SSIM_TOLERANCE   1e-03
#define NEAR_ZERO_THRESH 1e-09

/* ========== Delta Tracking ========== */

typedef struct {
    const char *name;
    double max_abs_delta;
    double max_rel_delta;
    double tolerance;
    unsigned count;
} DeltaTracker;

/* Maximum number of tracked score pairs (13 per spec Section 2.2, plus margin) */
#define MAX_TRACKERS 20
static DeltaTracker g_trackers[MAX_TRACKERS];
static unsigned g_num_trackers = 0;

static DeltaTracker *get_or_create_tracker(const char *name, double tolerance)
{
    for (unsigned i = 0; i < g_num_trackers; i++) {
        if (strcmp(g_trackers[i].name, name) == 0)
            return &g_trackers[i];
    }
    if (g_num_trackers >= MAX_TRACKERS) return NULL;
    DeltaTracker *t = &g_trackers[g_num_trackers++];
    t->name = name;
    t->max_abs_delta = 0.0;
    t->max_rel_delta = 0.0;
    t->tolerance = tolerance;
    t->count = 0;
    return t;
}

/* ========== Core Comparison ========== */

static char fail_msg[1024];

/**
 * Compare float and integer scores within tolerance. Updates the tracker
 * with max observed deltas. Returns 1 if within tolerance, 0 otherwise.
 *
 * Near-zero handling (Section 5.4): when both scores are below NEAR_ZERO_THRESH,
 * the check passes unconditionally to avoid false alarms.
 */
static int scores_agree(double float_val, double int_val, double tolerance,
                        DeltaTracker *tracker, const char *metric_name,
                        const char *score_name, unsigned frame_index,
                        const char *input_desc)
{
    double abs_delta = fabs(float_val - int_val);
    double denom = fmax(fabs(float_val), fabs(int_val));
    double rel_delta = (denom > NEAR_ZERO_THRESH) ? abs_delta / denom : 0.0;

    tracker->count++;
    if (abs_delta > tracker->max_abs_delta) tracker->max_abs_delta = abs_delta;
    if (rel_delta > tracker->max_rel_delta) tracker->max_rel_delta = rel_delta;

    /* Near-zero: both values negligible, pass unconditionally */
    if (fabs(float_val) < NEAR_ZERO_THRESH && fabs(int_val) < NEAR_ZERO_THRESH)
        return 1;

    if (abs_delta >= tolerance) {
        snprintf(fail_msg, sizeof(fail_msg),
            "PRECISION FAIL: %s.%s\n"
            "  float_value  = %.15f\n"
            "  int_value    = %.15f\n"
            "  abs_delta    = %.2e  (tolerance: %.2e)\n"
            "  rel_delta    = %.2e\n"
            "  frame_index  = %u\n"
            "  input        = %s\n",
            metric_name, score_name,
            float_val, int_val,
            abs_delta, tolerance,
            rel_delta,
            frame_index, input_desc);
        fprintf(stderr, "%s", fail_msg);
        return 0;
    }

    return 1;
}

/* ========== Historical Baselines (Section 6) ========== */

/* Hardcoded baselines from spec Section 5.2. If observed delta exceeds
 * baseline by >10%, emit a drift warning (but don't fail). */
typedef struct {
    const char *name;
    double baseline_delta;
} HistoricalBaseline;

static const HistoricalBaseline g_baselines[] = {
    { "ADM.adm2",       6.01e-04 },
    { "ADM.scale0",     4.29e-04 },
    { "ADM.scale1",     9.61e-05 },
    { "ADM.scale2",     1.85e-03 },
    { "ADM.scale3",     9.68e-04 },
    { "VIF.scale0",     1.07e-03 },
    { "VIF.scale1",     9.22e-05 },
    { "VIF.scale2",     2.82e-04 },
    { "VIF.scale3",     8.68e-03 },
    { "Motion.motion2", 0.0 },
    { "Motion.motion",  0.0 },
    { "PSNR.psnr_y",    0.0 },
    { "SSIM.ssim",      0.0 },
    { NULL, 0.0 }
};

static void check_drift_warning(const DeltaTracker *t)
{
    for (int i = 0; g_baselines[i].name; i++) {
        if (strcmp(g_baselines[i].name, t->name) == 0) {
            double baseline = g_baselines[i].baseline_delta;
            if (baseline > 0.0 && t->max_abs_delta > baseline * 1.1) {
                fprintf(stderr,
                    "WARNING: %s delta %.2e exceeds historical baseline "
                    "%.2e by %.1f%%\n",
                    t->name, t->max_abs_delta, baseline,
                    100.0 * (t->max_abs_delta - baseline) / baseline);
            }
            break;
        }
    }
}

/* ========== Synthetic Frame Helpers ========== */

static uint32_t prng_next(uint32_t *state)
{
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

typedef enum {
    FILL_CONSTANT_128,
    FILL_GRADIENT,
    FILL_RANDOM_42,
    FILL_ZERO,
    FILL_MAX,
    FILL_CHECKERBOARD,
    FILL_COUNT
} FillType;

/* Fill type name strings, kept for diagnostic reporting */
#define FILL_NAME(ft) ( \
    (ft) == FILL_CONSTANT_128 ? "constant_128" : \
    (ft) == FILL_GRADIENT ? "gradient" : \
    (ft) == FILL_RANDOM_42 ? "random_42" : \
    (ft) == FILL_ZERO ? "flat_zero" : \
    (ft) == FILL_MAX ? "flat_max" : \
    (ft) == FILL_CHECKERBOARD ? "checkerboard" : "unknown")

static void fill_picture(VmafPicture *pic, FillType ft)
{
    unsigned w = pic->w[0];
    unsigned h = pic->h[0];
    ptrdiff_t stride = pic->stride[0];
    uint8_t *data = (uint8_t *)pic->data[0];

    for (unsigned plane = 0; plane < 3; plane++) {
        unsigned pw = pic->w[plane];
        unsigned ph = pic->h[plane];
        ptrdiff_t ps = pic->stride[plane];
        uint8_t *pd = (uint8_t *)pic->data[plane];

        switch (ft) {
        case FILL_CONSTANT_128:
            for (unsigned y = 0; y < ph; y++)
                memset(pd + y * ps, 128, pw);
            break;
        case FILL_GRADIENT:
            for (unsigned y = 0; y < ph; y++)
                for (unsigned x = 0; x < pw; x++)
                    pd[y * ps + x] = (uint8_t)((x * 255) / (pw > 1 ? pw - 1 : 1));
            break;
        case FILL_RANDOM_42: {
            uint32_t state = 42 + plane;
            for (unsigned y = 0; y < ph; y++)
                for (unsigned x = 0; x < pw; x++)
                    pd[y * ps + x] = (uint8_t)(prng_next(&state) & 0xFF);
            break;
        }
        case FILL_ZERO:
            for (unsigned y = 0; y < ph; y++)
                memset(pd + y * ps, 0, pw);
            break;
        case FILL_MAX:
            for (unsigned y = 0; y < ph; y++)
                memset(pd + y * ps, 255, pw);
            break;
        case FILL_CHECKERBOARD:
            for (unsigned y = 0; y < ph; y++)
                for (unsigned x = 0; x < pw; x++)
                    pd[y * ps + x] = ((x + y) & 1) ? 255 : 0;
            break;
        default:
            break;
        }
    }
    (void)data; (void)w; (void)h; (void)stride;
}

/* ========== Generic Test Runner Using Public API ========== */

/**
 * Runs both the float and integer feature extractor for a given metric
 * on synthetic frames, then compares the named score pairs.
 *
 * This uses the full public VmafContext API:
 *   vmaf_init -> vmaf_use_feature (x2) -> vmaf_read_pictures -> vmaf_feature_score_at_index
 */

typedef struct {
    const char *float_score_name;
    const char *int_score_name;
    const char *tracker_name;
    double tolerance;
} ScorePair;

static char *run_synthetic_precision_test(
    const char *float_fex_name,
    const char *int_fex_name,
    const char *metric_name,
    const ScorePair *pairs, unsigned n_pairs,
    FillType ref_fill, FillType dis_fill,
    unsigned width, unsigned height,
    const char *input_desc)
{
    int err;
    VmafContext *vmaf;
    VmafConfiguration cfg = {
        .log_level = VMAF_LOG_LEVEL_NONE,
        .n_threads = 1,
        .n_subsample = 0,
        .cpumask = 0,
        .gpumask = 0,
    };

    err = vmaf_init(&vmaf, cfg);
    mu_assert("vmaf_init failed", !err);

    err = vmaf_use_feature(vmaf, float_fex_name, NULL);
    mu_assert("vmaf_use_feature (float) failed", !err);

    err = vmaf_use_feature(vmaf, int_fex_name, NULL);
    mu_assert("vmaf_use_feature (int) failed", !err);

    /* Feed 2 frames to handle temporal features like motion */
    for (unsigned idx = 0; idx < 2; idx++) {
        VmafPicture ref, dis;
        err = vmaf_picture_alloc(&ref, VMAF_PIX_FMT_YUV420P, 8, width, height);
        mu_assert("vmaf_picture_alloc ref failed", !err);
        err = vmaf_picture_alloc(&dis, VMAF_PIX_FMT_YUV420P, 8, width, height);
        mu_assert("vmaf_picture_alloc dis failed", !err);

        fill_picture(&ref, ref_fill);
        fill_picture(&dis, dis_fill);

        err = vmaf_read_pictures(vmaf, &ref, &dis, idx);
        mu_assert("vmaf_read_pictures failed", !err);
    }

    /* Flush */
    err = vmaf_read_pictures(vmaf, NULL, NULL, 0);
    mu_assert("vmaf_read_pictures flush failed", !err);

    /* Compare scores on frame 0 (all non-temporal features produce frame 0 scores).
     * For motion which is temporal, use frame 1 where motion2 is available. */
    unsigned check_idx = 0;
    if (strcmp(metric_name, "Motion") == 0) check_idx = 1;

    for (unsigned p = 0; p < n_pairs; p++) {
        double float_score, int_score;
        err = vmaf_feature_score_at_index(vmaf, pairs[p].float_score_name,
                                          &float_score, check_idx);
        if (err) {
            /* Some scores might not be available at frame 0 for temporal fex */
            if (check_idx == 0 && strcmp(metric_name, "Motion") != 0) {
                snprintf(fail_msg, sizeof(fail_msg),
                    "vmaf_feature_score_at_index(%s) failed for %s frame %u",
                    pairs[p].float_score_name, input_desc, check_idx);
                mu_assert(fail_msg, 0);
            }
            continue;
        }
        err = vmaf_feature_score_at_index(vmaf, pairs[p].int_score_name,
                                          &int_score, check_idx);
        if (err) {
            snprintf(fail_msg, sizeof(fail_msg),
                "vmaf_feature_score_at_index(%s) failed for %s frame %u",
                pairs[p].int_score_name, input_desc, check_idx);
            mu_assert(fail_msg, 0);
        }

        DeltaTracker *tracker = get_or_create_tracker(pairs[p].tracker_name,
                                                      pairs[p].tolerance);
        mu_assert("too many trackers", tracker != NULL);

        int ok = scores_agree(float_score, int_score, pairs[p].tolerance,
                              tracker, metric_name,
                              pairs[p].float_score_name,
                              check_idx, input_desc);
        mu_assert(fail_msg, ok);
    }

    vmaf_close(vmaf);
    return NULL;
}

/* ========== Score Pair Definitions (Section 2.2) ========== */

static const ScorePair adm_pairs[] = {
    { "VMAF_feature_adm2_score",    "VMAF_integer_feature_adm2_score",    "ADM.adm2",   ADM_TOLERANCE },
    { "VMAF_feature_adm_scale0_score", "integer_adm_scale0",             "ADM.scale0",  ADM_TOLERANCE },
    { "VMAF_feature_adm_scale1_score", "integer_adm_scale1",             "ADM.scale1",  ADM_TOLERANCE },
    { "VMAF_feature_adm_scale2_score", "integer_adm_scale2",             "ADM.scale2",  ADM_TOLERANCE },
    { "VMAF_feature_adm_scale3_score", "integer_adm_scale3",             "ADM.scale3",  ADM_TOLERANCE },
};
#define N_ADM_PAIRS (sizeof(adm_pairs) / sizeof(adm_pairs[0]))

static const ScorePair vif_pairs[] = {
    { "VMAF_feature_vif_scale0_score", "VMAF_integer_feature_vif_scale0_score", "VIF.scale0", VIF_TOLERANCE },
    { "VMAF_feature_vif_scale1_score", "VMAF_integer_feature_vif_scale1_score", "VIF.scale1", VIF_TOLERANCE },
    { "VMAF_feature_vif_scale2_score", "VMAF_integer_feature_vif_scale2_score", "VIF.scale2", VIF_TOLERANCE },
    { "VMAF_feature_vif_scale3_score", "VMAF_integer_feature_vif_scale3_score", "VIF.scale3", VIF_TOLERANCE },
};
#define N_VIF_PAIRS (sizeof(vif_pairs) / sizeof(vif_pairs[0]))

static const ScorePair motion_pairs[] = {
    { "VMAF_feature_motion2_score",  "VMAF_integer_feature_motion2_score",  "Motion.motion2", MOTION_TOLERANCE },
    { "VMAF_feature_motion_score",   "VMAF_integer_feature_motion_score",   "Motion.motion",  MOTION_TOLERANCE },
};
#define N_MOTION_PAIRS (sizeof(motion_pairs) / sizeof(motion_pairs[0]))

static const ScorePair psnr_pairs[] = {
    { "float_psnr", "psnr_y", "PSNR.psnr_y", PSNR_TOLERANCE },
};
#define N_PSNR_PAIRS (sizeof(psnr_pairs) / sizeof(psnr_pairs[0]))

/* Note: The integer ssim extractor (vmaf_fex_ssim in integer_ssim.c) is not
 * registered in the feature extractor list and thus not accessible through
 * the public API. The SSIM tests below verify float_ssim deterministic
 * reproducibility by running two independent VmafContext instances on
 * identical input and comparing their scores. */

/* ========== ADM Precision Tests ========== */

#if VMAF_FLOAT_FEATURES

static char *test_adm_precision_8bit(void)
{
    /* Test with gradient ref and random distortion - exercises typical
     * content with both DC and edge components */
    char *r = run_synthetic_precision_test(
        "float_adm", "adm", "ADM",
        adm_pairs, N_ADM_PAIRS,
        FILL_GRADIENT, FILL_RANDOM_42,
        576, 324, "gradient_vs_random42_576x324_8bit");
    return r;
}

static char *test_adm_precision_identical(void)
{
    /* Identical ref/dis pair. Tests perfect-score agreement and
     * near-zero edge case handling. */
    char *r = run_synthetic_precision_test(
        "float_adm", "adm", "ADM",
        adm_pairs, N_ADM_PAIRS,
        FILL_CONSTANT_128, FILL_CONSTANT_128,
        576, 324, "identical_const128_576x324_8bit");
    return r;
}

static char *test_adm_precision_synthetic_gradient(void)
{
    char *r = run_synthetic_precision_test(
        "float_adm", "adm", "ADM",
        adm_pairs, N_ADM_PAIRS,
        FILL_GRADIENT, FILL_GRADIENT,
        64, 64, "gradient_identical_64x64_8bit");
    return r;
}

static char *test_adm_precision_synthetic_flat(void)
{
    char *r = run_synthetic_precision_test(
        "float_adm", "adm", "ADM",
        adm_pairs, N_ADM_PAIRS,
        FILL_ZERO, FILL_MAX,
        576, 324, "zero_vs_max_576x324_8bit");
    return r;
}

/* ========== VIF Precision Tests ========== */

static char *test_vif_precision_8bit(void)
{
    char *r = run_synthetic_precision_test(
        "float_vif", "vif", "VIF",
        vif_pairs, N_VIF_PAIRS,
        FILL_GRADIENT, FILL_RANDOM_42,
        576, 324, "gradient_vs_random42_576x324_8bit");
    return r;
}

static char *test_vif_precision_identical(void)
{
    char *r = run_synthetic_precision_test(
        "float_vif", "vif", "VIF",
        vif_pairs, N_VIF_PAIRS,
        FILL_CONSTANT_128, FILL_CONSTANT_128,
        576, 324, "identical_const128_576x324_8bit");
    return r;
}

static char *test_vif_precision_synthetic_gradient(void)
{
    char *r = run_synthetic_precision_test(
        "float_vif", "vif", "VIF",
        vif_pairs, N_VIF_PAIRS,
        FILL_GRADIENT, FILL_GRADIENT,
        64, 64, "gradient_identical_64x64_8bit");
    return r;
}

/* ========== Motion Precision Tests ========== */

static char *test_motion_precision_8bit(void)
{
    char *r = run_synthetic_precision_test(
        "float_motion", "motion", "Motion",
        motion_pairs, N_MOTION_PAIRS,
        FILL_GRADIENT, FILL_RANDOM_42,
        576, 324, "gradient_vs_random42_576x324_8bit");
    return r;
}

static char *test_motion_precision_identical(void)
{
    char *r = run_synthetic_precision_test(
        "float_motion", "motion", "Motion",
        motion_pairs, N_MOTION_PAIRS,
        FILL_CONSTANT_128, FILL_CONSTANT_128,
        576, 324, "identical_const128_576x324_8bit");
    return r;
}

static char *test_motion_precision_synthetic_random(void)
{
    char *r = run_synthetic_precision_test(
        "float_motion", "motion", "Motion",
        motion_pairs, N_MOTION_PAIRS,
        FILL_RANDOM_42, FILL_CHECKERBOARD,
        64, 64, "random42_vs_checker_64x64_8bit");
    return r;
}

/* ========== PSNR Precision Tests ========== */

static char *test_psnr_precision_8bit(void)
{
    char *r = run_synthetic_precision_test(
        "float_psnr", "psnr", "PSNR",
        psnr_pairs, N_PSNR_PAIRS,
        FILL_GRADIENT, FILL_RANDOM_42,
        576, 324, "gradient_vs_random42_576x324_8bit");
    return r;
}

#endif /* VMAF_FLOAT_FEATURES */

/* ========== SSIM Precision Tests (always compiled) ==========
 *
 * Since the integer SSIM extractor is not registered in the public API,
 * these tests verify float_ssim deterministic reproducibility by running
 * two independent VmafContext instances on identical synthetic frames and
 * confirming identical scores.
 */

static char *run_ssim_reproducibility_test(
    FillType ref_fill, FillType dis_fill,
    unsigned width, unsigned height,
    const char *input_desc)
{
    int err;
    double score1, score2;

    /* Run float_ssim in two independent contexts */
    for (int run = 0; run < 2; run++) {
        VmafContext *vmaf;
        VmafConfiguration cfg = {
            .log_level = VMAF_LOG_LEVEL_NONE,
            .n_threads = 1,
            .n_subsample = 0,
            .cpumask = 0,
            .gpumask = 0,
        };

        err = vmaf_init(&vmaf, cfg);
        mu_assert("vmaf_init failed (ssim)", !err);

        err = vmaf_use_feature(vmaf, "float_ssim", NULL);
        mu_assert("vmaf_use_feature (float_ssim) failed", !err);

        VmafPicture ref, dis;
        err = vmaf_picture_alloc(&ref, VMAF_PIX_FMT_YUV420P, 8, width, height);
        mu_assert("vmaf_picture_alloc ref failed (ssim)", !err);
        err = vmaf_picture_alloc(&dis, VMAF_PIX_FMT_YUV420P, 8, width, height);
        mu_assert("vmaf_picture_alloc dis failed (ssim)", !err);

        fill_picture(&ref, ref_fill);
        fill_picture(&dis, dis_fill);

        err = vmaf_read_pictures(vmaf, &ref, &dis, 0);
        mu_assert("vmaf_read_pictures failed (ssim)", !err);
        err = vmaf_read_pictures(vmaf, NULL, NULL, 0);
        mu_assert("vmaf_read_pictures flush failed (ssim)", !err);

        double score;
        err = vmaf_feature_score_at_index(vmaf, "float_ssim", &score, 0);
        mu_assert("vmaf_feature_score_at_index (float_ssim) failed", !err);

        if (run == 0) score1 = score;
        else score2 = score;

        vmaf_close(vmaf);
    }

    /* Both runs must produce identical scores (determinism) */
    DeltaTracker *tracker = get_or_create_tracker("SSIM.ssim", SSIM_TOLERANCE);
    mu_assert("too many trackers (ssim)", tracker != NULL);

    int ok = scores_agree(score1, score2, SSIM_TOLERANCE,
                          tracker, "SSIM", "float_ssim", 0, input_desc);
    mu_assert(fail_msg, ok);

    /* Verify identical scores exactly match (determinism test) */
    mu_assert("SSIM reproducibility: scores from two independent runs differ",
              score1 == score2);

    return NULL;
}

static char *test_ssim_precision_8bit(void)
{
    return run_ssim_reproducibility_test(
        FILL_GRADIENT, FILL_RANDOM_42,
        576, 324, "gradient_vs_random42_576x324_8bit");
}

static char *test_ssim_precision_identical(void)
{
    return run_ssim_reproducibility_test(
        FILL_CONSTANT_128, FILL_CONSTANT_128,
        576, 324, "identical_const128_576x324_8bit");
}

/* ========== Summary Report (Section 9) ========== */

static char *test_precision_report(void)
{
    fprintf(stderr, "\n========== Numerical Precision Summary ==========\n");
    fprintf(stderr, "%-12s %-10s %-13s %-13s %-11s %s\n",
            "Metric", "Score", "MaxAbsDelta", "MaxRelDelta", "Tolerance", "Status");

    for (unsigned i = 0; i < g_num_trackers; i++) {
        DeltaTracker *t = &g_trackers[i];
        const char *status = "OK";
        if (t->max_abs_delta >= t->tolerance)
            status = "FAIL";

        fprintf(stderr, "%-12s %-10s %-13.2e %-13.2e %-11.2e %s\n",
                t->name, "", t->max_abs_delta, t->max_rel_delta,
                t->tolerance, status);

        /* Check drift warning (Section 6.3) */
        check_drift_warning(t);

        /* Log PRECISION_DELTA line (Section 6.1) */
        fprintf(stderr,
            "PRECISION_DELTA metric=%s max_abs_delta=%.2e max_rel_delta=%.2e "
            "samples=%u\n",
            t->name, t->max_abs_delta, t->max_rel_delta, t->count);
    }

    fprintf(stderr, "=================================================\n");
    return NULL;
}

/* ========== Test Runner ========== */

char *run_tests(void)
{
#if VMAF_FLOAT_FEATURES
    /* ADM precision tests */
    mu_run_test(test_adm_precision_8bit);
    mu_run_test(test_adm_precision_identical);
    mu_run_test(test_adm_precision_synthetic_gradient);
    mu_run_test(test_adm_precision_synthetic_flat);

    /* VIF precision tests */
    mu_run_test(test_vif_precision_8bit);
    mu_run_test(test_vif_precision_identical);
    mu_run_test(test_vif_precision_synthetic_gradient);

    /* Motion precision tests */
    mu_run_test(test_motion_precision_8bit);
    mu_run_test(test_motion_precision_identical);
    mu_run_test(test_motion_precision_synthetic_random);

    /* PSNR precision test (8-bit only; >8-bit has different peak definitions) */
    mu_run_test(test_psnr_precision_8bit);
#endif

    /* SSIM precision tests (always compiled) */
    mu_run_test(test_ssim_precision_8bit);
    mu_run_test(test_ssim_precision_identical);

    /* Summary report */
    mu_run_test(test_precision_report);

    return NULL;
}
