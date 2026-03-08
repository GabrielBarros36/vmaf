# VMAF Inference Performance Optimization Plan

## Environment
- Architecture: aarch64 (ARM Neoverse N1, 8 cores)
- SIMD: NEON (asimd, asimddp, asimdhp)
- Build: Release, O3, ASM enabled

## Optimization Ideas (Sorted by Expected Impact)

### High Impact

#### 1. Eliminate per-frame malloc in SVM prediction [PREDICT]
**File:** `libvmaf/src/predict.c:240, 2581-2594 (svm.cpp)`
**Problem:** `vmaf_predict_score_at_index()` calls `malloc/free` for `svm_node` array every frame. `svm_predict()` internally also mallocs `dec_values`. For VMAF's SVR model (NU_SVR), the node array is always `n_features+1` elements (typically 7).
**Fix:** Pre-allocate node buffer in prediction context or use stack allocation. For the SVM side, since VMAF always uses NU_SVR, the dec_values allocation in `svm_predict` is always size 1 - use stack variable.
**Expected Impact:** High - removes 2+ malloc/free cycles per frame per model.

#### 2. Cache feature extractor lookups in prediction [PREDICT]
**File:** `libvmaf/src/predict.c:244-276`
**Problem:** For every frame, `vmaf_predict_score_at_index()` calls `vmaf_get_feature_extractor_by_feature_name()`, then creates and destroys a `VmafFeatureExtractorContext` just to generate the feature name string. This involves string comparisons, malloc, dictionary copy, and cleanup - all to get the same feature name every single time.
**Fix:** Cache the resolved feature names the first time, reuse on subsequent frames.
**Expected Impact:** High - eliminates O(n_features * n_extractors) string comparisons + allocs per frame.

#### 3. Optimize feature_collector linear search with hash lookup [COLLECTOR]
**File:** `libvmaf/src/feature/feature_collector.c:293-305`
**Problem:** `find_feature_vector()` does linear search by `strcmp` over all registered features. Called from both `append` and `get_score`, both holding the mutex.
**Fix:** Use a simple hash table or sorted array with binary search for feature vector lookup.
**Expected Impact:** Medium-High - reduces lock hold time and per-frame overhead for scoring.

#### 4. Enable Link-Time Optimization (LTO) [BUILD]
**File:** `libvmaf/meson.build`
**Problem:** LTO is disabled (`b_lto = false`). LTO allows cross-translation-unit inlining, dead code elimination, and better optimization across the feature extractor pipeline.
**Fix:** Add LTO support as a build option, default enabled for release builds.
**Expected Impact:** Medium-High - typically 5-15% improvement for compute-heavy C code.

#### 5. ADM CM inner loop strength reduction [ADM]
**File:** `libvmaf/src/feature/integer_adm.c:1266-1647, 1647-2055`
**Problem:** The `adm_cm` and `i4_adm_cm` functions contain complex inner loops with divisions, conditional branches, and redundant index computations. These are the most compute-intensive functions in the ADM pipeline.
**Fix:** Hoist loop-invariant computations, reduce integer divisions, use branchless min/max where possible.
**Expected Impact:** Medium - ADM is ~40% of VMAF compute time.

#### 6. Reduce mutex contention in feature_collector [COLLECTOR]
**File:** `libvmaf/src/feature/feature_collector.c:314,418`
**Problem:** `vmaf_feature_collector_append` and `vmaf_feature_collector_get_score` both acquire a global mutex. In multi-threaded mode, all extractors contend on this single lock to write their features.
**Fix:** Use read-write lock (pthread_rwlock) - multiple readers can proceed concurrently. Only writes need exclusive access. Alternatively, use per-feature-vector locks.
**Expected Impact:** Medium - reduces thread contention, especially with n_threads > 2.

### Medium Impact

#### 7. SVM kernel: optimize for dense features [SVM]
**File:** `libvmaf/src/svm.cpp:320-377`
**Problem:** The RBF kernel `k_function` uses sparse vector format with branch-heavy index matching. VMAF features are always dense (all indices present and sequential).
**Fix:** Add a dense-vector fast path for the RBF kernel that uses a simple loop without index matching branches.
**Expected Impact:** Medium - fewer branches and better vectorization in the SVM kernel loop.

#### 8. VIF: reduce redundant buffer copies in pad operations [VIF]
**File:** `libvmaf/src/feature/integer_vif.c:77-94`
**Problem:** `pad_top_and_bottom` does full-stride memcpy for padding, including copying chroma data that may not be needed.
**Fix:** Only copy the actual used width bytes rather than full stride.
**Expected Impact:** Low-Medium - reduces memory bandwidth usage.

#### 9. Picture copy optimization with NEON [PICTURE]
**File:** `libvmaf/src/feature/picture_copy.c`
**Problem:** Frame data copying uses generic memcpy per row.
**Fix:** Use NEON-accelerated bulk copy for aligned data, or copy full buffer when strides match.
**Expected Impact:** Low-Medium - depends on how often copies happen.

#### 10. Precompute ADM CSF coefficients [ADM]
**File:** `libvmaf/src/feature/integer_adm.c:932-1100`
**Problem:** CSF computation involves floating-point operations that could be partially precomputed based on the scale/viewing distance parameters.
**Fix:** Precompute scale-dependent constants in init() rather than per-frame.
**Expected Impact:** Low-Medium.

### Low Impact

#### 11. Avoid redundant bootstrap SVM prediction [PREDICT]
**File:** `libvmaf/src/predict.c:362-382`
**Problem:** `vmaf_bootstrap_predict_score_at_index` calls `vmaf_predict_score_at_index` TWICE per model in the collection - once for unclipped score, once for clipped score. The feature extraction and SVM prediction are identical; only the transform/clip step differs.
**Fix:** Compute SVM prediction once, then apply transform/clip separately.
**Expected Impact:** Low (only affects bootstrap/bagging model collections, not standard VMAF).

#### 12. Use stack buffer for small allocations in feature_name [MISC]
**File:** Various locations creating feature name strings
**Problem:** Feature names are heap-allocated via malloc for short strings.
**Fix:** Use stack-allocated buffers with fixed size (feature names are typically < 128 chars).
**Expected Impact:** Low - micro-optimization reducing malloc pressure.

#### 13. Optimize log2 lookup table generation [VIF]
**File:** `libvmaf/src/feature/integer_vif.c`
**Problem:** `log_generate` is called in init() but the table is the same every time.
**Fix:** Use a compile-time constant table instead of runtime generation.
**Expected Impact:** Low - only affects init time, not per-frame.

#### 14. ADM: merge i16_to_i32 conversion with subsequent operation [ADM]
**File:** `libvmaf/src/feature/integer_adm.c:2055-2065`
**Problem:** `i16_to_i32` is a separate pass that converts data before further processing.
**Fix:** Merge the conversion into the consuming function to avoid an extra memory pass.
**Expected Impact:** Low - reduces memory bandwidth by one pass.

#### 15. Compiler hints: restrict pointers and alignment [BUILD]
**File:** Various feature extractors
**Problem:** The compiler may not be able to prove that input/output pointers don't alias, preventing vectorization.
**Fix:** Add `restrict` qualifiers to function parameters where appropriate.
**Expected Impact:** Low - compiler may already handle this well at O3.

## Implementation Status

| # | Optimization | Status | Commit | Perf Impact |
|---|---|---|---|---|
| 1 | Eliminate per-frame malloc in SVM prediction | DONE | 88bcd085 | Removes 2 malloc/free per frame |
| 2 | Cache feature extractor lookups | DONE | 88bcd085 | Eliminates O(n*m) string ops per frame |
| 3 | Hash-based feature_collector lookup | DONE | 1b4ee267 | O(1) avg vs O(n) linear search |
| 4 | Enable LTO | DONE | 40d52a2c | 5-15% typical improvement via cross-TU optimization |
| 5 | ADM CM inner loop optimization | DONE | aaaa754d | Hoisted loop-invariant index computations, deduplicated powf |
| 6 | Read-write lock for feature_collector | DONE | 1b4ee267 | Concurrent reads in multi-threaded mode |
| 7 | SVM branch prediction hint | DONE | eedd2544 | Better branch prediction for dense vectors |
| 8 | VIF pad optimization | DONE | 2a6038db | Copies only used width instead of full stride |
| 9 | Picture copy optimization | DONE | 01607b1f | Flat loop when strides are contiguous |
| 10 | Precompute ADM CSF coefficients | DONE | 4980e570 | Eliminates 24 dwt_quant_step calls per frame |
| 11 | Avoid redundant bootstrap prediction | DONE | ea766f35 | Computes SVM prediction once, applies transform/clip separately |
| 12 | Stack buffers for feature names | DONE | 31d5dfee | Stack buffer for per-frame feature name construction |
| 13 | Compile-time log2 table | DONE | 5078769a | Static const 32K-entry table replaces runtime generation |
| 14 | Merge i16_to_i32 with consumer | DONE | 05dcc3ee | Inline widening in DWT, eliminates separate conversion pass |
| 15 | Restrict pointer hints | DONE | 5ea71f4c | RESTRICT on buffer pointers in ADM, VIF, motion, picture_copy |
