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
 * Cross-Architecture Golden Value Tests
 *
 * Validates that the full VMAF metric pipeline produces scores matching
 * canonical reference values (computed with -Denable_asm=false) within
 * defined tolerances. Catches integration-level regressions and
 * cross-architecture SIMD divergence.
 *
 * See plans/specs/cross-arch-golden-values.md for the full specification.
 */

#include "test.h"
#include "golden_values.h"

#include "libvmaf/libvmaf.h"
#include "libvmaf/model.h"
#include "libvmaf/picture.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* Compile-time path to test video directory */
#ifndef TEST_VIDEO_DIR
#error "TEST_VIDEO_DIR must be defined at compile time"
#endif

/* ---- Comparison function (spec Section 6.2) ---- */
static int score_within_tolerance(double golden, double actual, double atol)
{
    if (golden == actual) return 1;  /* Exact match, handles +/-inf */
    return fabs(golden - actual) <= atol;
}

/* ---- Minimal YUV reader ---- */
static int read_yuv_frame(FILE *f, VmafPicture *pic,
                          unsigned w, unsigned h, unsigned bit_depth,
                          enum VmafPixelFormat pix_fmt)
{
    int err = vmaf_picture_alloc(pic, pix_fmt, bit_depth, w, h);
    if (err) return -1;

    unsigned bytes_per_sample = bit_depth > 8 ? 2 : 1;

    for (unsigned p = 0; p < 3; p++) {
        uint8_t *dst = (uint8_t *)pic->data[p];
        unsigned pw = pic->w[p];
        unsigned ph = pic->h[p];
        for (unsigned row = 0; row < ph; row++) {
            size_t nread = fread(dst, bytes_per_sample, pw, f);
            if (nread != pw) {
                vmaf_picture_unref(pic);
                return 1; /* EOF */
            }
            dst += pic->stride[p];
        }
    }
    return 0;
}

/* ---- Run a single golden value test case ---- */

/* Static buffer for failure message (mu_assert requires a string literal
 * or static storage). */
static char fail_msg[4096];

static char *run_golden_test(const GoldenTestCase *tc)
{
    int err = 0;

    /* Build full paths */
    char ref_path[2048], dis_path[2048];
    snprintf(ref_path, sizeof(ref_path), "%s%s", TEST_VIDEO_DIR, tc->ref_file);
    snprintf(dis_path, sizeof(dis_path), "%s%s", TEST_VIDEO_DIR, tc->dis_file);

    /* Initialize VMAF context: single-threaded, no subsampling */
    VmafConfiguration cfg = {
        .log_level = VMAF_LOG_LEVEL_NONE,
        .n_threads = 1,
        .n_subsample = 1,
    };

    VmafContext *vmaf = NULL;
    err = vmaf_init(&vmaf, cfg);
    mu_assert("vmaf_init failed", !err);

    /* Load model if specified */
    VmafModel *model = NULL;
    VmafModelCollection *model_collection = NULL;
    if (tc->model_version) {
        VmafModelConfig model_cfg = {
            .name = "golden_test",
            .flags = VMAF_MODEL_FLAGS_DEFAULT,
        };

        if (tc->is_model_collection) {
            err = vmaf_model_collection_load(&model, &model_collection,
                                             &model_cfg, tc->model_version);
            if (err) {
                snprintf(fail_msg, sizeof(fail_msg),
                         "vmaf_model_collection_load failed for %s: model=%s err=%d",
                         tc->id, tc->model_version, err);
                vmaf_close(vmaf);
                mu_assert(fail_msg, 0);
            }
            err = vmaf_use_features_from_model_collection(vmaf, model_collection);
            if (err) {
                snprintf(fail_msg, sizeof(fail_msg),
                         "vmaf_use_features_from_model_collection failed for %s err=%d",
                         tc->id, err);
                vmaf_model_destroy(model);
                vmaf_model_collection_destroy(model_collection);
                vmaf_close(vmaf);
                mu_assert(fail_msg, 0);
            }
        } else {
            err = vmaf_model_load(&model, &model_cfg, tc->model_version);
            if (err) {
                snprintf(fail_msg, sizeof(fail_msg),
                         "vmaf_model_load failed for %s: model=%s err=%d",
                         tc->id, tc->model_version, err);
                vmaf_close(vmaf);
                mu_assert(fail_msg, 0);
            }
            err = vmaf_use_features_from_model(vmaf, model);
            if (err) {
                snprintf(fail_msg, sizeof(fail_msg),
                         "vmaf_use_features_from_model failed for %s err=%d",
                         tc->id, err);
                vmaf_model_destroy(model);
                vmaf_close(vmaf);
                mu_assert(fail_msg, 0);
            }
        }
    }

    /* Register additional feature extractors */
    for (unsigned i = 0; i < tc->n_extra_features; i++) {
        err = vmaf_use_feature(vmaf, tc->extra_features[i], NULL);
        if (err) {
            snprintf(fail_msg, sizeof(fail_msg),
                     "vmaf_use_feature '%s' failed for %s err=%d",
                     tc->extra_features[i], tc->id, err);
            if (model) vmaf_model_destroy(model);
            if (model_collection) vmaf_model_collection_destroy(model_collection);
            vmaf_close(vmaf);
            mu_assert(fail_msg, 0);
        }
    }

    /* Open video files */
    FILE *f_ref = fopen(ref_path, "rb");
    if (!f_ref) {
        snprintf(fail_msg, sizeof(fail_msg),
                 "Cannot open ref video: %s", ref_path);
        if (model) vmaf_model_destroy(model);
        vmaf_close(vmaf);
        mu_assert(fail_msg, 0);
    }

    FILE *f_dis = fopen(dis_path, "rb");
    if (!f_dis) {
        snprintf(fail_msg, sizeof(fail_msg),
                 "Cannot open dis video: %s", dis_path);
        fclose(f_ref);
        if (model) vmaf_model_destroy(model);
        vmaf_close(vmaf);
        mu_assert(fail_msg, 0);
    }

    /* Read frames and feed to VMAF */
    unsigned frame_idx = 0;
    while (1) {
        VmafPicture pic_ref, pic_dis;
        int ret_ref = read_yuv_frame(f_ref, &pic_ref,
                                     tc->width, tc->height,
                                     tc->bit_depth, tc->pix_fmt);
        int ret_dis = read_yuv_frame(f_dis, &pic_dis,
                                     tc->width, tc->height,
                                     tc->bit_depth, tc->pix_fmt);

        if (ret_ref || ret_dis) {
            if (!ret_ref) vmaf_picture_unref(&pic_ref);
            if (!ret_dis) vmaf_picture_unref(&pic_dis);
            break;
        }

        err = vmaf_read_pictures(vmaf, &pic_ref, &pic_dis, frame_idx);
        if (err) {
            snprintf(fail_msg, sizeof(fail_msg),
                     "vmaf_read_pictures failed at frame %u for %s",
                     frame_idx, tc->id);
            fclose(f_ref);
            fclose(f_dis);
            if (model) vmaf_model_destroy(model);
            if (model_collection) vmaf_model_collection_destroy(model_collection);
            vmaf_close(vmaf);
            mu_assert(fail_msg, 0);
        }
        frame_idx++;
    }

    /* Flush the pipeline */
    err = vmaf_read_pictures(vmaf, NULL, NULL, 0);
    if (err) {
        snprintf(fail_msg, sizeof(fail_msg),
                 "vmaf_read_pictures flush failed for %s err=%d",
                 tc->id, err);
        fclose(f_ref);
        fclose(f_dis);
        if (model) vmaf_model_destroy(model);
        vmaf_close(vmaf);
        mu_assert(fail_msg, 0);
    }

    /* Verify frame count */
    if (frame_idx != tc->n_frames) {
        snprintf(fail_msg, sizeof(fail_msg),
                 "Frame count mismatch for %s: expected %u, got %u",
                 tc->id, tc->n_frames, frame_idx);
        fclose(f_ref);
        fclose(f_dis);
        if (model) vmaf_model_destroy(model);
        vmaf_close(vmaf);
        mu_assert(fail_msg, 0);
    }

    /* Trigger VMAF score prediction if a model is loaded.
     * This must be done before vmaf_feature_score_pooled can retrieve the
     * composite VMAF score, because vmaf_score_pooled triggers the SVM
     * prediction and stores the result in the feature collector. */
    if (model) {
        if (model_collection) {
            VmafModelCollectionScore coll_score = { 0 };
            err = vmaf_score_pooled_model_collection(
                vmaf, model_collection, VMAF_POOL_METHOD_MEAN,
                &coll_score, 0, frame_idx - 1);
        } else {
            double vmaf_pooled_score = 0.0;
            err = vmaf_score_pooled(vmaf, model, VMAF_POOL_METHOD_MEAN,
                                    &vmaf_pooled_score, 0, frame_idx - 1);
        }
        if (err) {
            snprintf(fail_msg, sizeof(fail_msg),
                     "vmaf_score_pooled failed for %s err=%d",
                     tc->id, err);
            fclose(f_ref);
            fclose(f_dis);
            vmaf_model_destroy(model);
            if (model_collection) vmaf_model_collection_destroy(model_collection);
            vmaf_close(vmaf);
            mu_assert(fail_msg, 0);
        }
    }

    /* Compare each golden score */
    for (unsigned i = 0; i < tc->n_scores; i++) {
        double actual;
        err = vmaf_feature_score_pooled(vmaf, tc->scores[i].name,
                                        VMAF_POOL_METHOD_MEAN,
                                        &actual, 0, frame_idx - 1);
        if (err) {
            snprintf(fail_msg, sizeof(fail_msg),
                     "vmaf_feature_score_pooled failed for %s metric=%s err=%d",
                     tc->id, tc->scores[i].name, err);
            fclose(f_ref);
            fclose(f_dis);
            if (model) vmaf_model_destroy(model);
            if (model_collection) vmaf_model_collection_destroy(model_collection);
            vmaf_close(vmaf);
            mu_assert(fail_msg, 0);
        }

        if (!score_within_tolerance(tc->scores[i].golden_mean,
                                    actual, tc->scores[i].tolerance)) {
            snprintf(fail_msg, sizeof(fail_msg),
                     "\n  GOLDEN VALUE MISMATCH: %s\n"
                     "    Metric:     %s\n"
                     "    Expected:   %.15g\n"
                     "    Actual:     %.15g\n"
                     "    Difference: %.15g\n"
                     "    Tolerance:  %.15g",
                     tc->id, tc->scores[i].name,
                     tc->scores[i].golden_mean, actual,
                     fabs(tc->scores[i].golden_mean - actual),
                     tc->scores[i].tolerance);
            fclose(f_ref);
            fclose(f_dis);
            if (model) vmaf_model_destroy(model);
            if (model_collection) vmaf_model_collection_destroy(model_collection);
            vmaf_close(vmaf);
            mu_assert(fail_msg, 0);
        }
    }

    /* Cleanup */
    fclose(f_ref);
    fclose(f_dis);
    if (model) vmaf_model_destroy(model);
    if (model_collection) vmaf_model_collection_destroy(model_collection);
    vmaf_close(vmaf);

    return NULL;
}

/* ---- Individual test functions ---- */
static char *test_golden_vmaf_8bit(void)
{
    return run_golden_test(&golden_test_cases[0]);
}

static char *test_golden_vmaf_identity(void)
{
    return run_golden_test(&golden_test_cases[1]);
}

static char *test_golden_vmaf_12bit(void)
{
    return run_golden_test(&golden_test_cases[2]);
}

static char *test_golden_vmaf_neg(void)
{
    return run_golden_test(&golden_test_cases[3]);
}

static char *test_golden_vmaf_4k(void)
{
    return run_golden_test(&golden_test_cases[4]);
}

static char *test_golden_vmaf_float(void)
{
    return run_golden_test(&golden_test_cases[5]);
}

static char *test_golden_vmaf_b_v063(void)
{
    return run_golden_test(&golden_test_cases[6]);
}

static char *test_golden_cambi(void)
{
    return run_golden_test(&golden_test_cases[7]);
}

/* ---- Check for test video availability ---- */

/*
 * The golden value tests require YUV video files from python/test/resource/yuv/.
 * These files are NOT tracked in git (they are in .gitignore) and are downloaded
 * on demand by the Python test framework from the vmaf_resource repo.
 *
 * In CI, these files are typically not available for pure C test runs.
 * When the files are missing, the test exits with code 77 which meson
 * interprets as "skip" (Autotools convention).
 */
static int test_videos_available(void)
{
    char path[2048];
    snprintf(path, sizeof(path), "%s%s", TEST_VIDEO_DIR,
             golden_test_cases[0].ref_file);
    FILE *f = fopen(path, "rb");
    if (f) {
        fclose(f);
        return 1;
    }
    return 0;
}

/* ---- Test runner ---- */
char *run_tests(void)
{
    if (!test_videos_available()) {
        fprintf(stderr,
            "SKIP: test video files not found in %s\n"
            "These files are downloaded by the Python test framework.\n"
            "Run: python -c \"from vmaf.config import VmafConfig; "
            "VmafConfig.test_resource_path('yuv', 'src01_hrc00_576x324.yuv')\"\n",
            TEST_VIDEO_DIR);
        exit(77);  /* meson "skip" exit code */
    }

    mu_run_test(test_golden_vmaf_8bit);
    mu_run_test(test_golden_vmaf_identity);
    mu_run_test(test_golden_vmaf_12bit);
    mu_run_test(test_golden_vmaf_neg);
    mu_run_test(test_golden_vmaf_4k);
    mu_run_test(test_golden_vmaf_float);
    mu_run_test(test_golden_vmaf_b_v063);
    mu_run_test(test_golden_cambi);
    return NULL;
}
