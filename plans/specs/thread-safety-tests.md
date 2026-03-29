# Thread Safety and Concurrency Tests — Specification

## 1. Objective

Validate that libvmaf produces **deterministic, identical VMAF scores** regardless of the number of worker threads, and that all concurrent access patterns — including out-of-order frame submission, in-flight teardown, and shared model instances across multiple `VmafContext` objects — are free from data races, deadlocks, and undefined behavior.

This test suite must catch any threading regression before it reaches production, and must be compatible with ThreadSanitizer (TSan) for automated race condition detection.

---

## 2. Scope — Concurrency Surfaces Under Test

There are **4 concurrency surfaces** in libvmaf that require testing.

### 2.1 Thread Count Determinism

The `VmafConfiguration.n_threads` parameter controls how many threads run feature extractors concurrently via `VmafThreadPool`. Changing the thread count must not change the final VMAF score for any given input pair.

| Component | Relevant Code | Synchronization Mechanism |
|-----------|--------------|---------------------------|
| `VmafThreadPool` | `thread_pool.c` | `pthread_mutex_t`, `pthread_cond_t` |
| `VmafFeatureExtractorContextPool` | `feature_extractor.c` | `pthread_mutex_t`, `atomic_int`, `pthread_cond_t` |
| `VmafFeatureCollector` | `feature_collector.c` | `pthread_mutex_t` |

### 2.2 Frame Synchronization

The `VmafFrameSyncContext` in `framesync.c` manages inter-frame data dependencies when feature extractors process frames out of order. It uses a linked list of buffers with states (`BUF_FREE`, `BUF_ACQUIRED`, `BUF_FILLED`, `BUF_RETRIEVED`) protected by `acquire_lock` and `retrieve_lock` mutexes.

### 2.3 Context Lifecycle

`vmaf_close()` calls `vmaf_thread_pool_wait()` followed by `vmaf_framesync_destroy()` and `vmaf_thread_pool_destroy()`. Tearing down a context while feature extraction is in-flight must complete gracefully without crashes, hangs, or memory corruption.

### 2.4 Concurrent Context Instances

Multiple `VmafContext` instances may coexist in the same process, each with its own thread pool. When they share the same `VmafModel` pointer (loaded once, used by multiple contexts), the read-only model data must remain safe for concurrent access.

---

## 3. Test Architecture

### 3.1 Design Principle: Public API Black-Box Testing

All tests exercise the **public libvmaf API** (`vmaf_init`, `vmaf_use_features_from_model`, `vmaf_read_pictures`, `vmaf_score_at_index`, `vmaf_close`). Tests do not call internal functions directly. This validates thread safety as experienced by real callers.

Exception: The `VmafFrameSyncContext` stress test (Section 5) calls the internal framesync API directly, following the precedent set by the existing `test_framesync.c`.

### 3.2 Synthetic Input Generation

Tests generate synthetic `VmafPicture` pairs deterministically rather than reading video files. This eliminates I/O as a variable and keeps tests self-contained.

```c
static int generate_test_picture(VmafPicture *pic, unsigned w, unsigned h,
                                  unsigned bpc, uint8_t fill_value)
{
    int err = vmaf_picture_alloc(pic, VMAF_PIX_FMT_YUV420P, bpc, w, h);
    if (err) return err;
    for (unsigned p = 0; p < 3; p++) {
        uint8_t *data = pic->data[p];
        for (unsigned y = 0; y < pic->h[p]; y++) {
            memset(data + y * pic->stride[p], fill_value, pic->w[p]);
        }
    }
    return 0;
}
```

### 3.3 Test File Organization

```
libvmaf/test/
  test_thread_safety.c    # All thread safety and concurrency tests
```

The test file is a standalone executable following the existing Minunit pattern (`test.h`).

### 3.4 Model Path Configuration

Tests that require a `VmafModel` use the built-in default model loaded via `vmaf_model_load()`. The model path is provided at compile time via `-DJSON_MODEL_PATH`:

```c
static int load_default_model(VmafModel **model)
{
    VmafModelConfig cfg = { .name = "vmaf", .flags = VMAF_MODEL_FLAGS_DEFAULT };
    return vmaf_model_load(model, &cfg, "vmaf_v0.6.1");
}
```

---

## 4. Thread Count Sweep Tests

### 4.1 Objective

Verify that VMAF scoring produces **bit-identical** pooled scores when run with 1, 2, 4, 8, and 16 threads.

### 4.2 Test Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Thread counts | 1, 2, 4, 8, 16 | Powers of 2 covering single-threaded through high contention |
| Frame count | 10 | Enough to exercise temporal features and frame dependencies |
| Resolution | 64x64 | Small enough for fast execution, large enough for feature extractors |
| Bit depth | 8 | Standard depth, sufficient for determinism verification |
| Pixel format | YUV420P | Most common format |
| Model | `vmaf_v0.6.1` | Default model exercising ADM, VIF, and motion extractors |

### 4.3 Test Procedure

For each thread count in `{1, 2, 4, 8, 16}`:
1. Call `vmaf_init()` with `cfg.n_threads = thread_count`.
2. Call `vmaf_use_features_from_model()` with the default model.
3. Generate and submit 10 synthetic frame pairs via `vmaf_read_pictures()`.
4. Flush by calling `vmaf_read_pictures()` with NULL.
5. Retrieve per-frame VMAF scores via `vmaf_score_at_index()`.
6. Retrieve pooled VMAF score via `vmaf_score_pooled()` with `VMAF_POOL_METHOD_MEAN`.
7. Call `vmaf_close()`.

Compare all per-frame and pooled scores against the single-threaded baseline. All scores must be **bit-identical** (`double` exact equality via `memcmp`).

### 4.4 Code Example

```c
#define NUM_THREAD_COUNTS 5
#define NUM_FRAMES 10
#define TEST_W 64
#define TEST_H 64

static char *test_thread_count_determinism(void)
{
    unsigned thread_counts[NUM_THREAD_COUNTS] = { 1, 2, 4, 8, 16 };
    double baseline_scores[NUM_FRAMES];
    double baseline_pooled;
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
            generate_test_picture(&ref, TEST_W, TEST_H, 8, (i * 17) & 0xFF);
            generate_test_picture(&dist, TEST_W, TEST_H, 8, (i * 31 + 7) & 0xFF);
            err = vmaf_read_pictures(vmaf, &ref, &dist, i);
            mu_assert("failed to read pictures", !err);
        }
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
            memcpy(baseline_scores, scores, sizeof(scores));
            baseline_pooled = pooled;
        } else {
            for (unsigned i = 0; i < NUM_FRAMES; i++) {
                mu_assert("per-frame score differs from single-threaded baseline",
                          memcmp(&scores[i], &baseline_scores[i],
                                 sizeof(double)) == 0);
            }
            mu_assert("pooled score differs from single-threaded baseline",
                      memcmp(&pooled, &baseline_pooled, sizeof(double)) == 0);
        }

        vmaf_close(vmaf);
        vmaf_model_destroy(model);
    }

    return NULL;
}
```

### 4.5 Failure Diagnostics

When a score mismatch is detected, the test reports:
- The thread count that diverged
- The frame index at which the first divergence occurred
- The baseline (1-thread) score value
- The divergent score value
- The absolute difference

---

## 5. Out-of-Order Frame Submission Stress Tests

### 5.1 Objective

Stress-test `VmafFrameSyncContext` by submitting frames in non-sequential order from concurrent threads, verifying that frame data dependencies are correctly resolved regardless of scheduling order.

### 5.2 Test Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Thread count | 4 | Multiple threads to induce reordering |
| Frame count | 50 | Large enough to create significant out-of-order pressure |
| Buffer size | 1024 bytes | Matches existing `test_framesync.c` pattern |
| Iterations | 5 | Repeat to catch intermittent race conditions |

### 5.3 Test Procedure

1. Create a `VmafThreadPool` and `VmafFrameSyncContext`.
2. Submit 50 frames to the thread pool. Each worker:
   a. Acquires a framesync buffer via `vmaf_framesync_acquire_new_buf()`.
   b. Fills the buffer with deterministic data derived from the frame index.
   c. Submits the filled buffer via `vmaf_framesync_submit_filled_data()`.
   d. Retrieves the **previous** frame's buffer via `vmaf_framesync_retrieve_filled_data()` (blocking until available).
   e. Verifies the retrieved data matches the expected pattern for the previous frame.
   f. Releases the dependent buffer via `vmaf_framesync_release_buf()`.
3. Wait for all workers and destroy the pool and framesync context.
4. Repeat the entire procedure for 5 iterations.

### 5.4 Code Example

```c
#define OOO_NUM_FRAMES 50
#define OOO_BUF_LEN 1024
#define OOO_ITERATIONS 5

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

    /* Retrieve and verify previous frame */
    uint8_t *dep_buf;
    err = vmaf_framesync_retrieve_filled_data(td->fs_ctx, (void *)&dep_buf,
                                               td->index - 1);
    if (err) { td->err = err; return; }

    for (unsigned i = 0; i < OOO_BUF_LEN; i++) {
        uint8_t expected = ((td->index - 1) * 7 + i) & 0xFF;
        if (dep_buf[i] != expected) {
            td->err = -1;
            break;
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

        for (unsigned i = 0; i < OOO_NUM_FRAMES; i++) {
            OooThreadData td = { .fs_ctx = fs_ctx, .index = i, .err = 0 };
            err = vmaf_thread_pool_enqueue(pool, ooo_worker, &td, sizeof(td));
            mu_assert("failed to enqueue", !err);
        }

        err = vmaf_thread_pool_wait(pool);
        mu_assert("failed to wait on thread pool", !err);
        err = vmaf_thread_pool_destroy(pool);
        mu_assert("failed to destroy thread pool", !err);
        err = vmaf_framesync_destroy(fs_ctx);
        mu_assert("failed to destroy framesync", !err);
    }

    return NULL;
}
```

---

## 6. In-Flight Teardown Tests

### 6.1 Objective

Verify that calling `vmaf_close()` while frame scoring is still in progress completes without crashes, deadlocks, or memory leaks. The `vmaf_close()` function calls `vmaf_thread_pool_wait()` before destroying resources, so this test validates that the wait-then-destroy sequence handles all in-flight work correctly.

### 6.2 Test Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Thread count | 8 | High parallelism increases in-flight work likelihood |
| Frame count | 50 | Enough frames that some will be in-flight at teardown time |
| Resolution | 64x64 | Fast per-frame computation |
| Iterations | 10 | Repeated teardown to catch intermittent issues |

### 6.3 Test Procedure

For each iteration:
1. Initialize `VmafContext` with 8 threads.
2. Register the default model.
3. Submit 50 frame pairs rapidly without waiting.
4. Immediately call `vmaf_close()` **without** first flushing (no `vmaf_read_pictures(NULL, NULL)`).
5. Verify `vmaf_close()` returns 0 (success) and does not hang.

The key behavior being tested: `vmaf_close()` must internally call `vmaf_thread_pool_wait()` to drain all queued and in-progress work, then clean up all resources safely.

### 6.4 Code Example

```c
#define TEARDOWN_ITERATIONS 10
#define TEARDOWN_FRAMES 50

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
            generate_test_picture(&ref, 64, 64, 8, (i * 13) & 0xFF);
            generate_test_picture(&dist, 64, 64, 8, (i * 29 + 3) & 0xFF);
            err = vmaf_read_pictures(vmaf, &ref, &dist, i);
            if (err) break;  /* context may reject frames during teardown */
        }

        /* Teardown while work is potentially in-flight */
        err = vmaf_close(vmaf);
        mu_assert("vmaf_close failed or hung during in-flight teardown", !err);

        vmaf_model_destroy(model);
    }

    return NULL;
}
```

### 6.5 Timeout Guard

The test executable is registered in meson with a timeout. If `vmaf_close()` deadlocks, the test framework will kill the process and report a failure:

```meson
test('test_thread_safety', test_thread_safety, timeout: 120)
```

---

## 7. Concurrent VmafContext Tests

### 7.1 Objective

Verify that multiple `VmafContext` instances, each running in its own set of threads, can safely share the same `VmafModel` pointer and coexist in the same process without interference.

### 7.2 Test Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Concurrent contexts | 4 | Moderate contention on shared model |
| Threads per context | 4 | Each context has its own thread pool |
| Frames per context | 10 | Enough to produce scores |
| Resolution | 64x64 | Fast execution |

### 7.3 Test Procedure

1. Load a single `VmafModel` via `vmaf_model_load()`.
2. Spawn 4 pthreads, each of which:
   a. Creates its own `VmafContext` with `n_threads = 4`.
   b. Registers the **shared** model via `vmaf_use_features_from_model()`.
   c. Submits 10 frame pairs (identical inputs across all contexts).
   d. Flushes via `vmaf_read_pictures(NULL, NULL)`.
   e. Retrieves all per-frame scores.
   f. Calls `vmaf_close()`.
3. Join all 4 pthreads.
4. Destroy the model (only after all contexts are closed).
5. Verify that all 4 contexts produced identical scores.

### 7.4 Code Example

```c
#define NUM_CONCURRENT_CONTEXTS 4
#define CONCURRENT_FRAMES 10

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
        generate_test_picture(&ref, 64, 64, 8, (i * 17) & 0xFF);
        generate_test_picture(&dist, 64, 64, 8, (i * 31 + 7) & 0xFF);
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
        mu_assert("concurrent context encountered error", !ctx_data[i].err);
    }

    /* All contexts should produce identical scores */
    for (unsigned i = 1; i < NUM_CONCURRENT_CONTEXTS; i++) {
        for (unsigned f = 0; f < CONCURRENT_FRAMES; f++) {
            mu_assert("scores differ between concurrent contexts",
                      memcmp(&ctx_data[0].scores[f], &ctx_data[i].scores[f],
                             sizeof(double)) == 0);
        }
        mu_assert("pooled scores differ between concurrent contexts",
                  memcmp(&ctx_data[0].pooled_score, &ctx_data[i].pooled_score,
                         sizeof(double)) == 0);
    }

    vmaf_model_destroy(model);
    return NULL;
}
```

---

## 8. Race Condition Detection Strategy

### 8.1 ThreadSanitizer (TSan) Integration

All thread safety tests must be runnable under Clang's ThreadSanitizer. TSan instruments memory accesses at compile time to detect:

- Data races (unsynchronized concurrent read/write to the same memory location)
- Lock order inversions (potential deadlocks)
- Use of destroyed mutexes or condition variables
- Thread leaks

### 8.2 TSan Build Configuration

Add a meson option and build configuration for TSan:

```meson
# In the test executable definition
if get_option('enable_tsan')
    tsan_args = ['-fsanitize=thread', '-g', '-O1']
    test_thread_safety_exe = executable('test_thread_safety',
        ['test.c', 'test_thread_safety.c'],
        c_args: tsan_args,
        link_args: tsan_args,
        ...
    )
endif
```

### 8.3 TSan Suppression File

If there are known benign races in third-party code (e.g., `svm.cpp`), a suppression file should be provided:

```
# tsan_suppressions.txt
race:svm_predict_values
```

### 8.4 CI Integration for TSan

A dedicated CI job compiles and runs the thread safety tests with TSan enabled:

```yaml
- name: Thread Safety (TSan)
  run: |
    CC=clang meson setup builddir_tsan \
      -Denable_tests=true \
      -Db_sanitize=thread
    ninja -C builddir_tsan
    meson test -C builddir_tsan test_thread_safety
```

TSan exit code is non-zero if any race is detected, causing CI to fail.

### 8.5 What TSan Covers vs. What Tests Cover

| Concern | TSan Detects | Tests Detect |
|---------|-------------|--------------|
| Data races on shared memory | Yes | No (silent corruption) |
| Deadlocks (lock order inversion) | Yes | Indirectly (timeout) |
| Score non-determinism from thread scheduling | No | Yes |
| Use-after-free in teardown | Yes (with ASan) | No (crash is non-deterministic) |
| Functional correctness under concurrency | No | Yes |

Both TSan instrumentation and functional test assertions are required for complete coverage.

---

## 9. Stress Test Parameters

### 9.1 Standard Test Run

The standard test run executes with parameters tuned for CI environments (completing within 60 seconds on a 4-core machine):

| Test | Iterations | Frames | Threads | Expected Duration |
|------|-----------|--------|---------|-------------------|
| Thread count sweep | 5 thread counts x 1 | 10 per run | 1-16 | ~15s |
| Out-of-order framesync | 5 | 50 per iteration | 4 | ~10s |
| In-flight teardown | 10 | 50 per iteration | 8 | ~10s |
| Concurrent contexts | 1 | 10 per context x 4 | 4 per ctx | ~5s |
| **Total** | | | | **~40s** |

### 9.2 Extended Stress Run

An optional extended stress configuration can be enabled via environment variable for nightly CI or manual investigation:

| Test | Iterations | Frames | Threads |
|------|-----------|--------|---------|
| Thread count sweep | 5 thread counts x 10 | 50 per run | 1-16 |
| Out-of-order framesync | 50 | 200 per iteration | 8 |
| In-flight teardown | 100 | 200 per iteration | 16 |
| Concurrent contexts | 10 | 50 per context x 8 | 8 per ctx |

The extended configuration is activated at compile time:

```c
#ifdef VMAF_THREAD_STRESS_EXTENDED
#define OOO_ITERATIONS 50
#define OOO_NUM_FRAMES 200
#define TEARDOWN_ITERATIONS 100
#define TEARDOWN_FRAMES 200
#else
#define OOO_ITERATIONS 5
#define OOO_NUM_FRAMES 50
#define TEARDOWN_ITERATIONS 10
#define TEARDOWN_FRAMES 50
#endif
```

---

## 10. Integration with Build System

### 10.1 Meson Configuration

Add to `libvmaf/test/meson.build`:

```meson
test_thread_safety = executable('test_thread_safety',
    ['test.c', 'test_thread_safety.c'],
    include_directories : [libvmaf_inc, test_inc],
    link_with : get_option('default_library') == 'both' ? libvmaf.get_static_lib() : libvmaf,
    c_args : vmaf_cflags_common,
    dependencies : [thread_lib, cuda_dependency],
)

test('test_thread_safety', test_thread_safety, timeout: 120)
```

The test links against the full `libvmaf` library (static or shared) to exercise the complete threading stack, including the thread pool, feature extractor context pool, framesync, and feature collector.

### 10.2 Extended Stress Build Target

For the extended stress configuration, add a separate executable:

```meson
test_thread_safety_stress = executable('test_thread_safety_stress',
    ['test.c', 'test_thread_safety.c'],
    include_directories : [libvmaf_inc, test_inc],
    link_with : get_option('default_library') == 'both' ? libvmaf.get_static_lib() : libvmaf,
    c_args : [vmaf_cflags_common, '-DVMAF_THREAD_STRESS_EXTENDED'],
    dependencies : [thread_lib, cuda_dependency],
)

test('test_thread_safety_stress', test_thread_safety_stress,
     timeout: 600, is_parallel: false, suite: 'stress')
```

The stress target is placed in a `stress` test suite so it can be run selectively:

```bash
meson test -C builddir --suite stress
```

### 10.3 CI Configuration

The thread safety tests run automatically as part of `ninja test` on every CI platform that supports pthreads:

- **Ubuntu x86_64:** Standard run + TSan run
- **Ubuntu ARM64:** Standard run
- **macOS x86_64 / ARM64:** Standard run
- **Windows x86_64:** Skipped (pthreads dependency; Windows threading model differs)

The TSan CI job requires a separate build directory with `-Db_sanitize=thread`.

---

## 11. Completion Requirements

The thread safety and concurrency test harness is **complete** when ALL of the following requirements are met:

### R1. Thread Count Sweep Coverage
The test exercises all 5 thread counts (1, 2, 4, 8, 16) and verifies bit-identical VMAF scores (both per-frame and pooled) against the single-threaded baseline.

**Verification:** Run the test and confirm 5 thread counts are tested with score comparison against the 1-thread result. Intentionally modify a score in one thread-count run and confirm the test fails.

### R2. Feature Extractor Coverage
The thread count sweep test registers a model that exercises at least 3 distinct feature extractors (ADM, VIF, motion) concurrently, since determinism bugs are most likely to appear in feature extraction parallelism.

**Verification:** Confirm the model used in the test requires `vmaf_fex_integer_adm`, `vmaf_fex_integer_vif`, and `vmaf_fex_integer_motion`.

### R3. Out-of-Order Frame Verification
The framesync stress test submits at least 50 frames with 4+ threads and verifies data integrity for every inter-frame dependency (frame N reads frame N-1's data).

**Verification:** Confirm the test loops over all frames, retrieves the previous frame's buffer, and asserts byte-level correctness of the retrieved data.

### R4. In-Flight Teardown Safety
The in-flight teardown test calls `vmaf_close()` without first flushing, at least 10 times, with 8+ threads and 50+ frames, and confirms `vmaf_close()` returns 0 without hanging.

**Verification:** Run the test. If `vmaf_close()` hangs, the 120-second test timeout triggers a failure. If it crashes, the test process exits with a signal. Both are detectable.

### R5. Concurrent Context Isolation
The concurrent context test runs 4+ `VmafContext` instances simultaneously, each with its own thread pool, sharing a single `VmafModel`, and verifies all produce identical scores.

**Verification:** Confirm 4 pthreads are created, each running a full vmaf_init/read/score/close cycle, and scores are compared across all contexts with `memcmp`.

### R6. TSan Clean
The entire test suite runs under ThreadSanitizer without any reported data races, lock inversions, or thread leaks in libvmaf code. (Third-party code may be suppressed.)

**Verification:** Build with `-Db_sanitize=thread`, run the tests, and confirm TSan exit code is 0 with no warnings in stderr.

### R7. Deterministic Reproducibility
All tests are fully deterministic. Synthetic input data is computed from frame index using fixed arithmetic (no PRNG seeds that vary, no wall-clock dependencies). Running the same test twice produces identical results.

**Verification:** Run the test suite twice and confirm identical pass/fail results.

### R8. No Deadlock Under Repeated Stress
The extended stress configuration (when enabled) runs the full suite at 10x iteration count without deadlocking or crashing. The 600-second timeout catches any hangs.

**Verification:** Build with `-DVMAF_THREAD_STRESS_EXTENDED` and run. All tests pass within the timeout.

### R9. Build Integration
The tests compile and run as part of `ninja test` on all supported CI platforms (Ubuntu x86_64, Ubuntu ARM64, macOS) without manual intervention. The test executable links against libvmaf and pthreads.

**Verification:** CI pipeline passes with the new test on all platforms.

### R10. Diagnostic Failure Messages
When a test fails, the output includes enough information to diagnose the issue:
- For score mismatches: thread count, frame index, expected vs. actual score
- For data corruption: frame index, byte offset, expected vs. actual value
- For hangs: the test timeout mechanism reports which test timed out

**Verification:** Intentionally introduce a score perturbation and confirm the failure message includes thread count, frame index, and score values.

### R11. Zero Production Code Changes
The test infrastructure does not modify any production source file. All test code is contained in `libvmaf/test/test_thread_safety.c` and build system additions to `libvmaf/test/meson.build`.

**Verification:** `git diff --stat` after implementing the tests shows only additions to `meson.build` and the new test file.

---

## 12. Acceptance Criteria Summary

| Req | Description | Pass Condition |
|-----|-------------|----------------|
| R1 | Thread count sweep | 5 thread counts tested, scores bit-identical to 1-thread baseline |
| R2 | Feature extractor coverage | Model exercises ADM + VIF + motion concurrently |
| R3 | Out-of-order frames | 50+ frames, 4+ threads, all inter-frame data verified |
| R4 | In-flight teardown | 10+ iterations, `vmaf_close()` returns 0, no hang/crash |
| R5 | Concurrent contexts | 4+ contexts sharing 1 model, identical scores |
| R6 | TSan clean | Zero TSan warnings on libvmaf code |
| R7 | Deterministic | Repeated runs produce identical results |
| R8 | Extended stress | 10x iterations pass within 600s timeout |
| R9 | Build integration | Tests pass in CI on all supported platforms |
| R10 | Diagnostics | Failures report thread count, frame index, score values |
| R11 | Zero production impact | No production source files modified |

All 11 requirements must be met for the thread safety and concurrency test harness to be considered complete.
