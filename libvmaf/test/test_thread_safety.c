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

#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "test.h"
#include "libvmaf/libvmaf.h"
#include "libvmaf/model.h"
#include "libvmaf/picture.h"
#include "framesync.h"
#include "thread_pool.h"

/* ---------------------------------------------------------------------------
 * Configurable parameters — standard vs. extended stress
 * ---------------------------------------------------------------------------*/

#ifdef VMAF_THREAD_STRESS_EXTENDED
#define OOO_ITERATIONS      50
#define OOO_NUM_FRAMES      200
#define TEARDOWN_ITERATIONS  100
#define TEARDOWN_FRAMES      200
#else
#define OOO_ITERATIONS      5
#define OOO_NUM_FRAMES      50
#define TEARDOWN_ITERATIONS  10
#define TEARDOWN_FRAMES      50
#endif

#define NUM_THREAD_COUNTS       5
#define NUM_FRAMES              10
#define TEST_W                  64
#define TEST_H                  64

#define OOO_BUF_LEN             1024

#define NUM_CONCURRENT_CONTEXTS 4
#define CONCURRENT_FRAMES       10

/* ---------------------------------------------------------------------------
 * Helper: generate a synthetic VmafPicture filled with a flat value
 * ---------------------------------------------------------------------------*/

static int generate_test_picture(VmafPicture *pic, unsigned w, unsigned h,
                                 unsigned bpc, uint8_t fill_value)
{
    int err = vmaf_picture_alloc(pic, VMAF_PIX_FMT_YUV420P, bpc, w, h);
    if (err) return err;
    for (unsigned p = 0; p < 3; p++) {
        uint8_t *data = (uint8_t *)pic->data[p];
        for (unsigned y = 0; y < pic->h[p]; y++) {
            memset(data + y * pic->stride[p], fill_value, pic->w[p]);
        }
    }
    return 0;
}

/* ---------------------------------------------------------------------------
 * Helper: load the default vmaf_v0.6.1 model
 * ---------------------------------------------------------------------------*/

static int load_default_model(VmafModel **model)
{
    VmafModelConfig cfg = {
        .name = "vmaf",
        .flags = VMAF_MODEL_FLAGS_DEFAULT,
    };
    return vmaf_model_load(model, &cfg, "vmaf_v0.6.1");
}

/* ===========================================================================
 * TEST 1: Thread Count Determinism
 *
 * Run VMAF scoring with n_threads = 1, 2, 4, 8, 16 on the same synthetic
 * input and verify that all per-frame and pooled scores are bit-identical
 * to the single-threaded baseline.
 * ===========================================================================*/

static char *test_thread_count_determinism(void)
{
    unsigned thread_counts[NUM_THREAD_COUNTS] = { 1, 2, 4, 8, 16 };
    double baseline_scores[NUM_FRAMES];
    double baseline_pooled = 0.0;
    int err;

    for (unsigned t = 0; t < NUM_THREAD_COUNTS; t++) {
        VmafModel *model;
        err = load_default_model(&model);
        mu_assert("failed to load model", !err);

        VmafContext *vmaf;
        VmafConfiguration cfg = {
            .log_level = VMAF_LOG_LEVEL_NONE,
            .n_threads = thread_counts[t],
        };
        err = vmaf_init(&vmaf, cfg);
        mu_assert("failed to init vmaf context", !err);

        err = vmaf_use_features_from_model(vmaf, model);
        mu_assert("failed to register model features", !err);

        for (unsigned i = 0; i < NUM_FRAMES; i++) {
            VmafPicture ref, dist;
            err = generate_test_picture(&ref, TEST_W, TEST_H, 8,
                                        (i * 17) & 0xFF);
            mu_assert("failed to allocate ref picture", !err);
            err = generate_test_picture(&dist, TEST_W, TEST_H, 8,
                                        (i * 31 + 7) & 0xFF);
            mu_assert("failed to allocate dist picture", !err);
            err = vmaf_read_pictures(vmaf, &ref, &dist, i);
            mu_assert("failed to read pictures", !err);
        }

        /* Flush */
        err = vmaf_read_pictures(vmaf, NULL, NULL, 0);
        mu_assert("failed to flush", !err);

        double scores[NUM_FRAMES];
        for (unsigned i = 0; i < NUM_FRAMES; i++) {
            err = vmaf_score_at_index(vmaf, model, &scores[i], i);
            mu_assert("failed to get score", !err);
        }

        double pooled;
        err = vmaf_score_pooled(vmaf, model, VMAF_POOL_METHOD_MEAN,
                                &pooled, 0, NUM_FRAMES - 1);
        mu_assert("failed to get pooled score", !err);

        if (t == 0) {
            /* Store single-threaded baseline */
            memcpy(baseline_scores, scores, sizeof(scores));
            baseline_pooled = pooled;
        } else {
            /* Compare against baseline */
            for (unsigned i = 0; i < NUM_FRAMES; i++) {
                if (memcmp(&scores[i], &baseline_scores[i],
                           sizeof(double)) != 0) {
                    fprintf(stderr,
                        "\nThread count %u: per-frame score mismatch at "
                        "frame %u\n"
                        "  baseline (1-thread): %.17g\n"
                        "  actual (%u-threads): %.17g\n"
                        "  absolute diff:       %.17g\n",
                        thread_counts[t], i, baseline_scores[i],
                        thread_counts[t], scores[i],
                        scores[i] - baseline_scores[i]);
                    mu_assert("per-frame score differs from "
                              "single-threaded baseline", 0);
                }
            }
            if (memcmp(&pooled, &baseline_pooled, sizeof(double)) != 0) {
                fprintf(stderr,
                    "\nThread count %u: pooled score mismatch\n"
                    "  baseline (1-thread): %.17g\n"
                    "  actual (%u-threads): %.17g\n"
                    "  absolute diff:       %.17g\n",
                    thread_counts[t], baseline_pooled,
                    thread_counts[t], pooled,
                    pooled - baseline_pooled);
                mu_assert("pooled score differs from "
                          "single-threaded baseline", 0);
            }
        }

        vmaf_close(vmaf);
        vmaf_model_destroy(model);
    }

    return NULL;
}

/* ===========================================================================
 * TEST 2: Out-of-Order Frame Submission (framesync stress)
 *
 * Stress-test VmafFrameSyncContext by submitting frames from concurrent
 * threads via VmafThreadPool, verifying that inter-frame data dependencies
 * are correctly resolved regardless of scheduling order.
 * ===========================================================================*/

typedef struct OooThreadData {
    VmafFrameSyncContext *fs_ctx;
    unsigned index;
    int err;
} OooThreadData;

static void ooo_worker(void *data)
{
    OooThreadData *td = data;
    uint8_t *shared_buf;
    int err;

    err = vmaf_framesync_acquire_new_buf(td->fs_ctx, (void *)&shared_buf,
                                         OOO_BUF_LEN, td->index);
    if (err) { td->err = err; return; }

    /* Fill with deterministic pattern: each byte = (index * 7 + offset) & 0xFF */
    for (unsigned i = 0; i < OOO_BUF_LEN; i++)
        shared_buf[i] = (td->index * 7 + i) & 0xFF;

    err = vmaf_framesync_submit_filled_data(td->fs_ctx, shared_buf, td->index);
    if (err) { td->err = err; return; }

    if (td->index == 0) return;

    /* Retrieve and verify previous frame's data */
    uint8_t *dep_buf;
    err = vmaf_framesync_retrieve_filled_data(td->fs_ctx, (void *)&dep_buf,
                                              td->index - 1);
    if (err) { td->err = err; return; }

    for (unsigned i = 0; i < OOO_BUF_LEN; i++) {
        uint8_t expected = ((td->index - 1) * 7 + i) & 0xFF;
        if (dep_buf[i] != expected) {
            fprintf(stderr,
                "\nOut-of-order data corruption: frame %u depends on "
                "frame %u, byte %u: expected 0x%02x, got 0x%02x\n",
                td->index, td->index - 1, i, expected, dep_buf[i]);
            td->err = -1;
            vmaf_framesync_release_buf(td->fs_ctx, dep_buf, td->index - 1);
            return;
        }
    }

    vmaf_framesync_release_buf(td->fs_ctx, dep_buf, td->index - 1);
}

static char *test_framesync_out_of_order_stress(void)
{
    for (unsigned iter = 0; iter < OOO_ITERATIONS; iter++) {
        VmafThreadPool *pool;
        VmafFrameSyncContext *fs_ctx;
        int err;

        err = vmaf_thread_pool_create(&pool, 4);
        mu_assert("failed to create thread pool", !err);
        err = vmaf_framesync_init(&fs_ctx);
        mu_assert("failed to init framesync", !err);

        OooThreadData td_array[OOO_NUM_FRAMES];
        for (unsigned i = 0; i < OOO_NUM_FRAMES; i++) {
            td_array[i].fs_ctx = fs_ctx;
            td_array[i].index = i;
            td_array[i].err = 0;
            err = vmaf_thread_pool_enqueue(pool, ooo_worker, &td_array[i],
                                           sizeof(OooThreadData));
            mu_assert("failed to enqueue", !err);
        }

        err = vmaf_thread_pool_wait(pool);
        mu_assert("failed to wait on thread pool", !err);

        /* Check for errors reported by workers */
        for (unsigned i = 0; i < OOO_NUM_FRAMES; i++) {
            if (td_array[i].err) {
                fprintf(stderr,
                    "\nOut-of-order worker error at frame %u, iteration %u\n",
                    i, iter);
                mu_assert("out-of-order worker encountered error", 0);
            }
        }

        err = vmaf_thread_pool_destroy(pool);
        mu_assert("failed to destroy thread pool", !err);
        err = vmaf_framesync_destroy(fs_ctx);
        mu_assert("failed to destroy framesync", !err);
    }

    return NULL;
}

/* ===========================================================================
 * TEST 3: In-Flight Teardown
 *
 * Start scoring with many threads and frames, then call vmaf_close()
 * without flushing first. Verify no crash, deadlock, or memory corruption.
 * ===========================================================================*/

static char *test_inflight_teardown(void)
{
    for (unsigned iter = 0; iter < TEARDOWN_ITERATIONS; iter++) {
        VmafModel *model;
        int err = load_default_model(&model);
        mu_assert("failed to load model", !err);

        VmafContext *vmaf;
        VmafConfiguration cfg = {
            .log_level = VMAF_LOG_LEVEL_NONE,
            .n_threads = 8,
        };
        err = vmaf_init(&vmaf, cfg);
        mu_assert("failed to init context", !err);

        err = vmaf_use_features_from_model(vmaf, model);
        mu_assert("failed to register model", !err);

        /* Submit frames rapidly without flushing */
        for (unsigned i = 0; i < TEARDOWN_FRAMES; i++) {
            VmafPicture ref, dist;
            err = generate_test_picture(&ref, TEST_W, TEST_H, 8,
                                        (i * 13) & 0xFF);
            if (err) break;
            err = generate_test_picture(&dist, TEST_W, TEST_H, 8,
                                        (i * 29 + 3) & 0xFF);
            if (err) {
                vmaf_picture_unref(&ref);
                break;
            }
            err = vmaf_read_pictures(vmaf, &ref, &dist, i);
            if (err) break; /* context may reject frames during teardown */
        }

        /* Teardown while work is potentially in-flight */
        err = vmaf_close(vmaf);
        mu_assert("vmaf_close failed or hung during in-flight teardown", !err);

        vmaf_model_destroy(model);
    }

    return NULL;
}

/* ===========================================================================
 * TEST 4: Concurrent VmafContext Instances Sharing a Model
 *
 * Create multiple VmafContext instances, each in its own pthread, all
 * sharing the same VmafModel pointer. Run them concurrently and verify
 * all produce identical scores.
 * ===========================================================================*/

typedef struct ConcurrentCtxData {
    VmafModel *model;
    double scores[CONCURRENT_FRAMES];
    double pooled_score;
    int err;
    unsigned ctx_id;
} ConcurrentCtxData;

static void *concurrent_ctx_thread(void *arg)
{
    ConcurrentCtxData *d = arg;

    VmafContext *vmaf;
    VmafConfiguration cfg = {
        .log_level = VMAF_LOG_LEVEL_NONE,
        .n_threads = 4,
    };
    d->err = vmaf_init(&vmaf, cfg);
    if (d->err) return NULL;

    d->err = vmaf_use_features_from_model(vmaf, d->model);
    if (d->err) { vmaf_close(vmaf); return NULL; }

    for (unsigned i = 0; i < CONCURRENT_FRAMES; i++) {
        VmafPicture ref, dist;
        d->err = generate_test_picture(&ref, TEST_W, TEST_H, 8,
                                       (i * 17) & 0xFF);
        if (d->err) { vmaf_close(vmaf); return NULL; }
        d->err = generate_test_picture(&dist, TEST_W, TEST_H, 8,
                                       (i * 31 + 7) & 0xFF);
        if (d->err) {
            vmaf_picture_unref(&ref);
            vmaf_close(vmaf);
            return NULL;
        }
        d->err = vmaf_read_pictures(vmaf, &ref, &dist, i);
        if (d->err) { vmaf_close(vmaf); return NULL; }
    }

    d->err = vmaf_read_pictures(vmaf, NULL, NULL, 0);
    if (d->err) { vmaf_close(vmaf); return NULL; }

    for (unsigned i = 0; i < CONCURRENT_FRAMES; i++) {
        d->err = vmaf_score_at_index(vmaf, d->model, &d->scores[i], i);
        if (d->err) { vmaf_close(vmaf); return NULL; }
    }

    d->err = vmaf_score_pooled(vmaf, d->model, VMAF_POOL_METHOD_MEAN,
                               &d->pooled_score, 0, CONCURRENT_FRAMES - 1);

    vmaf_close(vmaf);
    return NULL;
}

static char *test_concurrent_contexts_shared_model(void)
{
    VmafModel *model;
    int err = load_default_model(&model);
    mu_assert("failed to load model", !err);

    pthread_t threads[NUM_CONCURRENT_CONTEXTS];
    ConcurrentCtxData ctx_data[NUM_CONCURRENT_CONTEXTS];

    for (unsigned i = 0; i < NUM_CONCURRENT_CONTEXTS; i++) {
        memset(&ctx_data[i], 0, sizeof(ctx_data[i]));
        ctx_data[i].model = model;
        ctx_data[i].ctx_id = i;
        err = pthread_create(&threads[i], NULL, concurrent_ctx_thread,
                             &ctx_data[i]);
        mu_assert("failed to create thread", !err);
    }

    for (unsigned i = 0; i < NUM_CONCURRENT_CONTEXTS; i++) {
        pthread_join(threads[i], NULL);
        if (ctx_data[i].err) {
            fprintf(stderr,
                "\nConcurrent context %u encountered error: %d\n",
                i, ctx_data[i].err);
        }
        mu_assert("concurrent context encountered error", !ctx_data[i].err);
    }

    /* All contexts should produce identical scores */
    for (unsigned i = 1; i < NUM_CONCURRENT_CONTEXTS; i++) {
        for (unsigned f = 0; f < CONCURRENT_FRAMES; f++) {
            if (memcmp(&ctx_data[0].scores[f], &ctx_data[i].scores[f],
                       sizeof(double)) != 0) {
                fprintf(stderr,
                    "\nConcurrent context score mismatch: ctx 0 vs ctx %u "
                    "at frame %u\n"
                    "  ctx 0: %.17g\n"
                    "  ctx %u: %.17g\n"
                    "  diff:  %.17g\n",
                    i, f, ctx_data[0].scores[f],
                    i, ctx_data[i].scores[f],
                    ctx_data[i].scores[f] - ctx_data[0].scores[f]);
                mu_assert("scores differ between concurrent contexts", 0);
            }
        }
        if (memcmp(&ctx_data[0].pooled_score, &ctx_data[i].pooled_score,
                   sizeof(double)) != 0) {
            fprintf(stderr,
                "\nConcurrent context pooled score mismatch: ctx 0 vs ctx %u\n"
                "  ctx 0: %.17g\n"
                "  ctx %u: %.17g\n"
                "  diff:  %.17g\n",
                i, ctx_data[0].pooled_score,
                i, ctx_data[i].pooled_score,
                ctx_data[i].pooled_score - ctx_data[0].pooled_score);
            mu_assert("pooled scores differ between concurrent contexts", 0);
        }
    }

    vmaf_model_destroy(model);
    return NULL;
}

/* ===========================================================================
 * Test runner
 * ===========================================================================*/

char *run_tests(void)
{
    mu_run_test(test_thread_count_determinism);
    mu_run_test(test_framesync_out_of_order_stress);
    mu_run_test(test_inflight_teardown);
    mu_run_test(test_concurrent_contexts_shared_model);
    return NULL;
}
