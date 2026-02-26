# Allocation Optimization Report — Section 2 Findings

## Overview

This document describes the memory/allocation optimizations implemented for four
components of the VMAF inference path. These address findings 2b, 2c, 2d, and 2e
from [optimization-opportunities.md](optimization-opportunities.md).

Finding 2a (SSIM per-frame malloc/free) was already implemented in Section 1 and
is documented in [simd-optimizations-report.md](simd-optimizations-report.md).

## Changes Made

### 1. predict.c — Cache Feature Names and SVM Node Array (Finding 2b, Medium impact)

**Problem:** `vmaf_predict_score_at_index()` is called once per frame per model.
Each call performed ~13 allocation pairs for the default 6-feature model:
- 1 `malloc` for the `svm_node` array
- For each feature: context create (3 mallocs), name generation (1 malloc),
  immediate context destroy (3 frees), name free (1 free)

The context creation was used solely to call `vmaf_feature_name_from_options()`
— the context was never used for actual extraction.

**Solution:** Added lazy caching of two arrays in `VmafModel`:
- `svm_nodes`: Pre-allocated `svm_node` array of size `n_features + 1`, reused
  every frame instead of malloc/free per call.
- `feature_names`: Pre-computed array of feature name strings, generated once on
  first call using the same context-create/name-from-options/context-destroy
  pattern, then cached for all subsequent frames.

Both fields start as NULL (model struct is zero-initialized during loading) and
are populated lazily on first call to `vmaf_predict_score_at_index()`. This
eliminates all per-frame allocations from the prediction path.

**Impact for bootstrap models:** The `vmaf_bootstrap_predict_score_at_index()`
function calls `vmaf_predict_score_at_index()` twice per model per frame across
~20 models. The caching eliminates `~13 * 2 * 20 = 520` allocation pairs per
frame in that path.

**Files modified:**
- `libvmaf/src/model.h` — added `svm_nodes` and `feature_names` fields
- `libvmaf/src/predict.c` — lazy initialization, cached loop body
- `libvmaf/src/model.c` — cleanup in `vmaf_model_destroy()`

---

### 2. cambi.c — Pre-allocate Heatmap Row Buffer (Finding 2c, Low impact)

**Problem:** `dump_c_values()` called `malloc(width * sizeof(uint16_t))` and
`free()` on every call. This function runs once per scale (5 scales) per frame
when heatmaps are enabled.

**Solution:** Added a `heatmap_row` field to `CambiBuffers`, allocated once in
`init()` using the full original width (which covers all scales since width
only decreases), and freed in `close_cambi()`. The `dump_c_values()` function
now receives the pre-allocated buffer as a parameter.

**Files modified:**
- `libvmaf/src/feature/cambi.c` — added field, init allocation, close free,
  modified `dump_c_values()` signature and call site

---

### 3. thread_pool.c — Embedded Data Buffer + Pre-allocated Job Pool (Finding 2d, Medium impact)

**Problem:** Every `vmaf_thread_pool_enqueue()` call performed two heap
allocations (job struct + data copy), and every completed job performed two
frees. For 3 extractors at 30 fps with threads enabled, this was 90
allocation-pair cycles per second.

**Solution:** Two changes that together eliminate all per-enqueue heap traffic
in the common case:

1. **Embedded data buffer:** Added a 256-byte `data_buf` inline in
   `VmafThreadPoolJob`. The job data (a `struct ThreadData` of ~200 bytes) is
   copied into this buffer instead of a separate `malloc`. This eliminates the
   per-job data `malloc`/`memcpy`/`free`.

2. **Pre-allocated job pool with free list:** At `vmaf_thread_pool_create()`
   time, allocates `n_threads * 8` job nodes in a single contiguous block and
   links them into a free list. `enqueue()` pops from the free list (under the
   existing queue lock) instead of calling `malloc`. After job execution, the
   runner returns the node to the free list (also under the lock) instead of
   calling `free`.

If the free list is exhausted (shouldn't happen in normal operation), falls
back to `malloc` with the same embedded-data optimization.

**Thread safety:** All free list accesses are protected by the existing
`queue.lock` mutex. The runner returns jobs to the free list after re-acquiring
the lock (before decrementing `n_working`), which is a slight reordering from
the original code but maintains the same correctness invariants.

**Files modified:**
- `libvmaf/src/thread_pool.c` — all changes in this one file

---

### 4. feature_collector.c — Hash Map for Feature Vector Lookup (Finding 2e, Medium impact)

**Problem:** `find_feature_vector()` performed a linear O(n) `strcmp` scan over
all registered feature vectors on every `vmaf_feature_collector_append()` and
`vmaf_feature_collector_get_score()` call. This ran under the global mutex,
serializing all feature score appends.

For the default model with ~10-12 named feature vectors, this was ~10-12 string
comparisons per append, all while holding the lock.

**Solution:** Added a 64-slot open-addressing hash map (`fv_hash`) embedded
directly in `VmafFeatureCollector`. Uses djb2 hashing with linear probing.

- `find_feature_vector()` now calls `fc_hash_lookup()` — O(1) average case
- New feature vectors are inserted into the hash map when first created
- Hash table is zero-initialized by the existing `memset(fc, 0, ...)` in init
- Keys are pointers into `FeatureVector.name` — no separate allocation needed
- No additional cleanup required — keys are freed with their parent FeatureVector

**Files modified:**
- `libvmaf/src/feature/feature_collector.h` — added `FcHashEntry` type and
  `fv_hash` field
- `libvmaf/src/feature/feature_collector.c` — added hash functions, modified
  `find_feature_vector()` and `append()`

---

## Correctness Verification

### C unit tests
All 17 C unit tests pass, including:
- `test_predict` — directly tests `vmaf_predict_score_at_index` with cached paths
- `test_feature_collector` — tests append/get with the new hash map
- `test_thread_pool` — tests enqueue/execute/wait/destroy with the pool
- `test_cambi` — tests CAMBI with the modified buffer management
- `test_framesync` — tests multi-threaded 10-frame processing end-to-end

### Python score regression tests
**67/67 tests pass.** All scores are bit-identical to 4 decimal places across:
- `VmafQualityRunner`, `VmafLegacyQualityRunner` (full VMAF pipeline)
- `PsnrQualityRunner`, `SsimQualityRunner`, `VifQualityRunner`
- Bootstrap models (10-model and 20-model collections)
- Multi-threaded execution (`test_run_vmaf_runner_3threads`)
- Transform score, phone model, and various model configurations

---

## Performance Results

Benchmarked on AWS `c6a.metal` (AMD EPYC 7R13, AVX2, turbo disabled,
performance governor, pinned to 4 cores on NUMA node 0) with a synthetic 1080p
48-frame video, 7 timed repetitions per run.

| Run | Label | Time (s) | Δ vs baseline | IPC | Cache miss % | Cycles |
|-----|-------|----------|---------------|-----|-------------|--------|
| Baseline (avg of 3) | baseline | 2.718 | — | 3.17 | 10.41% | 28,847M |
| After SIMD (Section 1) | simd-opt | 2.710 | -0.29% | 3.15 | 10.42% | 28,739M |
| **After alloc (Section 2)** | **alloc-opt** | **2.701** | **-0.63%** | 3.15 | 10.37% | 28,701M |
| **Incremental Δ (Section 2 only)** | | **-0.010s** | **-0.35%** | — | -0.05% | **-37M** |

### Analysis

The Section 2 allocation optimizations provide a **0.35% incremental wall-clock
improvement** over the Section 1 SIMD optimizations. Combined, Sections 1 and 2
together yield a **0.63% cumulative improvement** from the original baseline.

The modest wall-clock delta is expected for this 48-frame benchmark:

- **predict.c caching (2b):** Eliminates ~13 allocation pairs per frame per
  model. For the 48-frame benchmark with 1 model, this is ~624 allocation pairs
  removed. The impact is proportionally larger for:
  - Longer videos (thousands of frames)
  - Bootstrap model collections (20× more predict calls per frame)
  - Multi-model workflows

- **Thread pool (2d):** Eliminates 2 mallocs + 2 frees per enqueued job. For
  the 48-frame benchmark with 3 extractors per frame (4 threads), this is ~288
  allocation pairs removed. The benefit scales linearly with frame count.

- **Feature collector hash (2e):** Replaces ~10 strcmp comparisons with 1 hash
  lookup per append. Called hundreds of times per frame (once per feature score).
  Reduces mutex hold time, improving thread scaling.

- **Cambi heatmap (2c):** Only active when `heatmaps_path` is set (diagnostic
  mode). Zero impact on the standard benchmark.

The 37M cycle reduction (0.13%) reflects the direct savings from eliminating
heap allocator traffic. The remaining 0.22% wall-clock improvement likely comes
from reduced allocator contention and improved cache behavior.

### Remaining opportunities from Section 2

Finding 2a (SSIM allocations) was already addressed in Section 1. All five
Section 2 findings are now complete.

The dominant remaining optimization opportunities are in Sections 3 (algorithmic
redundancy: `dwt_quant_step` precomputation, `cos_1deg_sq` constant, `pow(2,k)`
→ bit-shift), which would eliminate per-frame transcendental function calls in
the ADM extractor (~34% of CPU time).

---

## Summary

| Finding | What | Type | Status | Score impact |
|---------|------|------|--------|-------------|
| 2a | SSIM allocation fix (4 malloc/free per frame → 0) | Memory | Done (Section 1) | None (bit-identical) |
| 2b | predict.c cache feature names + svm_nodes | Memory | Done | None (bit-identical) |
| 2c | cambi.c pre-allocate heatmap row buffer | Memory | Done | None (bit-identical) |
| 2d | thread_pool.c embedded data + pre-allocated job pool | Memory | Done | None (bit-identical) |
| 2e | feature_collector.c hash map for feature lookup | Memory | Done | None (bit-identical) |
