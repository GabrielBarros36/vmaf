# VMAF Inference Path — Optimization Opportunities

## Executive Summary

This document catalogues concrete, actionable optimization opportunities found in the VMAF
inference path (the code that runs when scoring a video). Findings span the C library
(`libvmaf/src/`) and the Python orchestration layer (`python/vmaf/core/`). The most impactful
opportunities are in:

1. **SIMD coverage gaps** — `integer_ssim.c` and the 8-bit PSNR path have no SIMD at all.
   The `adm_decouple` hot loop (`integer_adm.c`) also has no SIMD despite being a large
   inner-loop over every pixel.
2. **Per-frame allocations** — `integer_ssim.c` does four `malloc`/`free` calls inside
   `calc_ssim()` on every frame. `predict.c` does `malloc`/`free` + context creation/
   destruction inside `vmaf_predict_score_at_index()` on every frame.
3. **Redundant per-frame floating-point computation** — `dwt_quant_step()` (involving
   `log10`, `pow`) and `cos_1deg_sq` (involving `cos`) are recomputed inside hot loops
   on every frame, for constant model parameters.
4. **Vertical filter loop structure in VIF** — The scalar `vif_statistic_8` / `vif_statistic_16`
   loops iterate column-by-column in the vertical pass, causing strided memory access
   that thrashes L1 cache.
5. **Python subprocess overhead** — Each `FeatureExtractor` subclass launches the vmaf
   binary as a separate subprocess, re-reading and re-decoding the YUV files for every
   feature type requested.

Impact ratings below reflect expected wall-time reduction on a standard 1080p stream when
the fix is applied in isolation. Without profiling data the ratings are estimates and are
marked accordingly.

---

## 1. SIMD Gaps

### 1a. `integer_ssim.c` — No SIMD at All (High)

**File:** `libvmaf/src/feature/integer_ssim.c`
**Lines:** 88–196 (`calc_ssim`), 199–238 (feature extractor descriptor)

`calc_ssim` contains nested loops over every pixel:

```c
for (y = 0; y < _h + vkernel_offs; y++) {
    for (x = 0; x < _w; x++) {
        for (k = k_min; k < k_max; k++) {   // horizontal kernel tap
            m.mux += window * s;
            m.muy += window * d;
            m.x2  += window * s * s;
            ...
        }
    }
    for (x = 0; x < _w; x++) {
        for (k = k_min; k < k_max; k++) {   // vertical reduction
            m.mux += window * buf->mux; ...
        }
        ssim += ...;
    }
}
```

There is no SIMD dispatch, no function pointer, no architecture-specific implementation —
compare with `integer_vif.c` and `integer_adm.c` which both have AVX2 and AVX-512
(and ARM NEON) back-ends. The `ssim_moments` struct is 48 bytes, which fits into two
AVX2 registers; the inner kernel (5 taps, constant weights) is well-suited to
`_mm256_madd_epi16`.

Additionally, `gaussian_filter_init` (lines 42–72) computes the Gaussian kernel with
`exp()` on every call, and is called **twice** per frame (once for the vertical kernel,
once for the horizontal kernel). The filter parameters are always `sigma=1.5, max_len=5`,
so the result is constant — it should be precomputed in `init()` and stored in the
extractor state.

**Suggested fix:**
- Add a `SsimState` struct with preallocated `hkernel`, `vkernel`, and a line buffer
  of `ssim_moments` (width × line_sz), computed once in `init()`.
- Add an AVX2/NEON back-end for the horizontal accumulation pass.

---

### 1b. `integer_psnr.c` — 8-bit SSE Loop Has No SIMD (Medium)

**File:** `libvmaf/src/feature/integer_psnr.c`
**Lines:** 116–159 (`psnr` function — 8-bit path)

The 8-bit path:
```c
for (unsigned i = 0; i < ref_pic->h[p]; i++) {
    uint32_t sse_inner = 0;
    for (unsigned j = 0; j < ref_pic->w[p]; j++) {
        const int16_t e = ref[j] - dis[j];
        sse_inner += e * e;
    }
    sse += sse_inner;
    ref += ref_pic->stride[p];
    dis += dist_pic->stride[p];
}
```

This is a textbook SAD/SSE computation with no SIMD. AVX2 can compute 32 SAD/SSE
values per cycle with `_mm256_sad_epu8` or a sequence of `_mm256_sub_epi16` /
`_mm256_madd_epi16`. The 16-bit path (`psnr_hbd`) also has no SIMD. Neither path has
a SIMD dispatch mechanism. There is no SIMD anywhere under `x86/` or `arm64/` for psnr.

**Suggested fix:**
Add `x86/psnr_avx2.c` with AVX2 implementation; dispatch via function pointer (similar
to how `integer_motion.c` dispatches `sad`).

---

### 1c. `integer_adm.c` — `adm_decouple` / `adm_decouple_s123` Inner Loops Have No SIMD (High)

**File:** `libvmaf/src/feature/integer_adm.c`
**Lines:** 660–778 (`adm_decouple`), 789–930 (`adm_decouple_s123`)

`adm_decouple` is called once per ADM scale (4 scales) per frame, and iterates over a
large fraction of the DWT-subband pixels:

```c
for (int i = top; i < bottom; ++i) {
    for (int j = left; j < right; ++j) {
        // Per-pixel: 3 int64 dot products, 3 div_lookup lookups, 6 clamps, 6 float ops
        ot_dp    = (int64_t)oh * th + (int64_t)ov * tv;
        o_mag_sq = (int64_t)oh * oh + (int64_t)ov * ov;
        t_mag_sq = (int64_t)th * th + (int64_t)tv * tv;
        int angle_flag = ...;
        ...
    }
}
```

The AVX2 file (`x86/adm_avx2.c`) only implements `adm_dwt2_8_avx2` (the wavelet
transform step). The decouple / CSF / CM scoring loops (the bulk of the ADM
computation) are pure scalar. Given that ADM operates on `int16_t` data, eight 16-bit
values fit in a 128-bit SSE register or sixteen in AVX2.

The same gap exists for `adm_csf` (lines 932–1022), `adm_csf_den_scale`
(lines 1101–1182), `adm_cm` (lines ~1260–1645), and their `s123` equivalents.

**Uncertainty note:** These loops have mixed integer and float operations
(`angle_flag` uses float promotion). Vectorising the angle check requires careful
treatment of the float comparisons. Impact should still be high — these are per-pixel
loops called 4 times per frame.

**Suggested fix:**
Add AVX2 back-ends for `adm_decouple`, `adm_csf`, and `adm_cm`, dispatched via
function pointers from `init()` (same pattern as `dwt2_8`).

---

### 1d. `integer_motion.c` — `sad_c` Has No SIMD Dispatch (Medium)

**File:** `libvmaf/src/feature/integer_motion.c`
**Lines:** 227–242 (`sad_c`), 318–319 (init — only `x_convolution` gets SIMD, not `sad`)

```c
static void sad_c(VmafPicture *pic_a, VmafPicture *pic_b, uint64_t *sad) {
    for (unsigned i = 0; i < pic_a->h[0]; i++) {
        uint32_t inner_sad = 0;
        for (unsigned j = 0; j < pic_a->w[0]; j++)
            inner_sad += abs(a[j] - b[j]);
        ...
    }
}
```

There is a `s->sad` function pointer in `MotionState`, but it is always set to `sad_c`
(line 318) and never updated from any AVX2/AVX-512 implementation. The motion_avx2.c
and motion_avx512.c files only implement `x_convolution_16`. A full-frame absolute
difference accumulation is a natural fit for `_mm256_sad_epu8`.

**Suggested fix:**
Implement `sad_avx2` in `x86/motion_avx2.c` and wire it up in `init()`.

---

### 1e. `adm_avx2.c` — Scalar Tail Loop in Horizontal Pass (Low)

**File:** `libvmaf/src/feature/x86/adm_avx2.c`
**Lines:** 120–260+ (horizontal pass of `adm_dwt2_8_avx2`)

After the SIMD vertical pass (chunks of 16), the horizontal pass iterates one output
column at a time with scalar code:
```c
// One output pixel at a time — 4 scalar multiplies, 1 shift, 1 store
accum = 0;
accum += (int32_t)filter_lo[0] * s0; ...
dst->band_a[i * dst_stride] = (accum + add_shift_HP) >> shift_HP;
```
This is only done for `(w+1)/2` columns but it is repeated for every pair of rows
(the outer `i` loop covers `(h+1)/2` iterations).

The filter is only 4 taps. Four 16-bit multiplies per output pixel is modest work, but
at HD/4K resolutions the column count is large. The horizontal pass could process 8
output pixels simultaneously with `_mm256_madd_epi16`.

---

## 2. Memory / Allocation

### 2a. `integer_ssim.c` — Per-Frame `malloc`/`free` (High)

**File:** `libvmaf/src/feature/integer_ssim.c`
**Lines:** 107–115 and 192–195 (inside `calc_ssim`)

`calc_ssim` is called once per frame and contains:
```c
vkernel_sz = gaussian_filter_init(&vkernel, 1.5, 5);  // malloc inside
// ...
lines[0] = line_buf = (ssim_moments *)malloc(line_sz * _w * sizeof(*line_buf));
lines    = (ssim_moments **)malloc(line_sz * sizeof(*lines));
hkernel_sz = gaussian_filter_init(&hkernel, 1.5, 5);  // malloc inside
// ...
free(line_buf); free(lines); free(vkernel); free(hkernel);
```

That is 4 allocations and 4 frees on every frame. For a 1080p, 30-fps stream this is
120 allocation pairs per second. The kernel sizes are fixed (sigma=1.5, max_len=5
always produces a 9-element kernel), and the line buffer size depends only on width and
height (frame geometry), which are fixed for the lifetime of an extractor.

**Suggested fix:**
Add a `SsimState` struct (the extractor currently has an empty state via `init` returning
0). Precompute `hkernel`, `vkernel`, and allocate `line_buf` / `lines` once in `init()`;
free them in `close()`.

---

### 2b. `predict.c` — Per-Frame `malloc` + Context Creation Inside `vmaf_predict_score_at_index` (Medium)

**File:** `libvmaf/src/predict.c`
**Lines:** 240–332 (`vmaf_predict_score_at_index`)

Called once per frame per model, this function:
- `malloc`s an `svm_node` array (line 240)
- Loops over all model features and, for each, calls `vmaf_feature_extractor_context_create`
  (line 262) which allocates a new context
- Calls `vmaf_feature_name_from_options` which `malloc`s a string (line 272)
- Destroys the context immediately (line 275)
- `free`s the string (line 299), `free`s the node (line 330)

For the default vmaf_v0.6.1 model with 6 features, this is at minimum 13 allocations per
frame (1 node + 6 contexts + 6 strings + 6 frees for strings + 1 free for node).

The context create/destroy pattern is used purely to call `vmaf_feature_name_from_options`;
the context is never used for actual extraction. This is unnecessarily expensive.

**Suggested fix:**
Pre-build the feature name array for a model once (during `vmaf_use_features_from_model`
or lazily on first prediction) and cache it in the `VmafModel` struct. The `svm_node`
array can also be pre-allocated and reused since its size equals `model->n_features`.

---

### 2c. `cambi.c` — Per-Frame `malloc` in `dump_c_values` (Low)

**File:** `libvmaf/src/feature/cambi.c`
**Lines:** 1079–1090 (`dump_c_values`)

```c
uint16_t *to_write = malloc(width * sizeof(uint16_t));
...
free(to_write);
```

This only runs when `heatmaps_path` is set (a debug/diagnostic feature), but it still
represents an unnecessary per-frame, per-scale allocation. The buffer could be pre-
allocated in `CambiBuffers` during `init()`.

---

### 2d. `thread_pool.c` — Per-Frame Job `malloc`/`memcpy`/`free` (Medium)

**File:** `libvmaf/src/thread_pool.c`
**Lines:** 124–132 (`vmaf_thread_pool_enqueue`), 58–63 (`vmaf_thread_pool_job_destroy`)

Every enqueued job triggers:
```c
VmafThreadPoolJob *job = malloc(sizeof(*job));
job->data = malloc(data_sz);
memcpy(job->data, data, data_sz);
```
and on completion:
```c
if (job->data) free(job->data);
free(job);
```

When `n_threads > 0`, `vmaf_read_pictures` (in `libvmaf.c`) enqueues one job per
registered feature extractor per frame (lines 449–455). For a standard run with 3
extractors and 30 fps, this is 90 allocation-pair cycles per second at steady state.

The thread pool uses a linked-list queue (`head`/`tail` pointers) with dynamic node
allocation. A fixed-size ring-buffer of pre-allocated job nodes would eliminate all
per-enqueue heap traffic.

**Suggested fix:**
Replace the linked-list queue with a bounded ring buffer of job nodes pre-allocated
at `vmaf_thread_pool_create` time, sized to `n_threads * max_features_per_frame`.
The `ThreadData` struct (`libvmaf.c`, line 386) is small enough to embed directly in
the job node.

---

### 2e. `feature_collector.c` — Linear Search on Every `append` (Medium)

**File:** `libvmaf/src/feature/feature_collector.c`
**Lines:** 293–305 (`find_feature_vector`), called from 307–396 (`vmaf_feature_collector_append`)

```c
static FeatureVector *find_feature_vector(VmafFeatureCollector *fc,
                                          const char *feature_name) {
    for (unsigned i = 0; i < fc->cnt; i++) {
        FeatureVector *fv = fc->feature_vector[i];
        if (!strcmp(fv->name, feature_name)) {
            feature_vector = fv; break;
        }
    }
}
```

Called under the global `feature_collector->lock` mutex on every feature score append
(once per feature per frame), this does a linear `strcmp` scan over all registered
feature vectors. For the default model there are ~10–12 named feature vectors; the scan
is short, but it is serialised and locks out concurrent appends from different threads.

**Suggested fix:**
Replace the array with a hash map (keyed by feature name string). This removes the
lock contention bottleneck for multi-threaded scoring (when `n_threads > 1`). An
alternative is to have each feature extractor cache its `FeatureVector *` pointer
after the first `append`, avoiding the lookup entirely on subsequent frames.

---

## 3. Algorithmic Redundancy

### 3a. `integer_adm.c` — `dwt_quant_step` Called Many Times Per Frame with the Same Arguments (High)

**File:** `libvmaf/src/feature/integer_adm.c`
**Lines:** 947–948, 1037–1038, 1107–1108, 1190–1191, 1656–1657 (and more)

`dwt_quant_step` is called inside every major ADM sub-function:
- `adm_csf` (lines 947–948) — 2 calls
- `i4_adm_csf` (lines 1037–1038) — 2 calls per scale (3 scales) = 6 calls
- `adm_csf_den_scale` (lines 1107–1108) — 2 calls
- `adm_csf_den_s123` (lines 1190–1191) — 2 calls per scale = 6 calls
- `adm_cm` (approx. line 1280) — 2 calls
- `i4_adm_cm` (lines 1656–1657) — 2 calls per scale = 6 calls

Each call to `dwt_quant_step` computes:
```c
float r = adm_norm_view_dist * adm_ref_display_height * M_PI / 180.0;
float temp = log10(pow(2.0, lambda + 1) * params->f0 * params->g[theta] / r);
float Q = 2.0 * params->a * pow(10.0, params->k * temp * temp)
          / dwt_7_9_basis_function_amplitudes[lambda][theta];
```

This includes `log10` and `pow(10, ...)` — expensive transcendental calls. Since
`adm_norm_view_dist` and `adm_ref_display_height` are fixed for the lifetime of the
extractor, all 15+ `dwt_quant_step` results are constant across all frames.

There is a partial mitigation for scale 0 in `adm_csf` (lines 958–969): if the
parameters equal the defaults, hard-coded constants are used. But for non-default
parameters, and for all other scales, the transcendental calls run on every frame.

**Suggested fix:**
Precompute all required `dwt_quant_step` values during `init()` and store them in
`AdmState`. There are at most 4 (lambda) × 3 (theta) = 12 unique combinations.
Pass the precomputed table into each sub-function.

---

### 3b. `integer_adm.c` — `cos_1deg_sq` Recomputed Per Frame in `adm_decouple` and `adm_decouple_s123` (Medium)

**File:** `libvmaf/src/feature/integer_adm.c`
**Lines:** 663 and 792

```c
const float cos_1deg_sq = cos(1.0 * M_PI / 180.0) * cos(1.0 * M_PI / 180.0);
```

This is evaluated at the top of both `adm_decouple` (line 663) and
`adm_decouple_s123` (line 792) — which are called once each per ADM scale per frame,
so 8 `cos()` calls per frame. Since 1 degree is a literal constant, `cos_1deg_sq` is
numerically constant and can be defined as a compile-time constant or a global.

**Suggested fix:**
Replace with `#define ADM_COS_1DEG_SQ (0.9996954135095487f)` (the pre-computed value)
or a `static const float` computed once.

---

### 3c. `integer_adm.c` — `div_lookup_generator` Per Extractor Instantiation (Low)

**File:** `libvmaf/src/feature/integer_adm.c`
**Lines:** 2643 (in `init()`), definition around line ~440

`div_lookup_generator()` fills a global 65536-entry lookup table. It is called every
time an ADM feature extractor is initialised. If two models share the ADM extractor
with different options (leading to separate instances), the table is regenerated
unnecessarily. The result is always the same (it is a reciprocal LUT independent of
parameters).

**Suggested fix:**
Add a `static bool div_lookup_initialized = false` guard so the table is generated at
most once per process lifetime.

---

### 3d. `integer_adm.c` — `pow(2, ...)` Calls in Hot Loops (Medium)

**File:** `libvmaf/src/feature/integer_adm.c`
**Lines:** 1042 (`i4_adm_csf`), 1170–1172 (`adm_csf_den_scale`), 1308–1318 (`adm_cm`),
1675–1682 (`i4_adm_cm`)

Multiple inner setup sections compute values like:
```c
const double pow2_32 = pow(2, 32);
const uint32_t shift_xhcub = (uint32_t)ceil(log2(w) - 4);
const uint32_t add_shift_xhcub = (uint32_t)pow(2, (shift_xhcub - 1));
const uint32_t shift_inner_accum = (uint32_t)ceil(log2(h));
```

These are placed inside functions that are called once per scale per frame. The values
depend only on `w`, `h`, and `scale` — all fixed for the lifetime of the extractor
(or at least constant per frame, since `w` and `h` are halved each scale but
deterministically so). The `ceil(log2(x))` is equivalent to `32 - __builtin_clz(x-1)`
and should not require a floating-point `log2` call.

**Suggested fix:**
Precompute these shift constants in `init()` (one set per scale), store in `AdmState`.
Replace `pow(2, k)` with `(1u << k)` or `(1ULL << k)` as appropriate.

---

### 3e. `vif_statistic_8` / `vif_statistic_16` — Redundant `ii`/`jj` Computation Inside Inner Filter Loop (Low)

**File:** `libvmaf/src/feature/integer_vif.c`
**Lines:** 122–135 (vertical pass of `vif_statistic_8`)

```c
for (unsigned fi = 0; fi < fwidth; ++fi) {
    int ii = i - fwidth / 2;        // ii is invariant — computed every iteration
    int ii_check = ii + fi;
    ...
}
```

`ii = i - fwidth / 2` is loop-invariant and is recomputed on every iteration of the
inner `fi` loop. Most compilers will hoist this, but the pattern appears in both the
scalar reference implementation and should be corrected for clarity and to ensure
reliable optimisation even at `-O1`.

---

### 3f. `integer_adm.c` — `adm_csf_den_scale` and `adm_cm` Both Call `dwt_quant_step` with the Same Arguments (Medium)

**File:** `libvmaf/src/feature/integer_adm.c`
**Lines:** 1107–1108 (`adm_csf_den_scale`), 1280–1281 (`adm_cm`, approx.)

For scale 0, both `adm_csf_den_scale` and `adm_cm` independently call:
```c
const float factor1 = dwt_quant_step(..., 0, 1, adm_norm_view_dist, adm_ref_display_height);
const float factor2 = dwt_quant_step(..., 0, 2, adm_norm_view_dist, adm_ref_display_height);
```

These two functions are called back-to-back from `integer_compute_adm` (lines 2480–2486)
with the same `adm_norm_view_dist` / `adm_ref_display_height`, so the transcendental
computation is duplicated. This is subsumed by finding 3a, but is called out separately
because it is a simpler, more targeted fix (pass the pre-computed factors from
`integer_compute_adm` directly).

---

## 4. Cache / Memory Access Patterns

### 4a. `vif_statistic_8` — Vertical Pass Has Strided Non-Sequential Access (Medium)

**File:** `libvmaf/src/feature/integer_vif.c`
**Lines:** 236–263 (vertical pass of `vif_statistic_8`)

The scalar vertical pass iterates `for j = 0..w-1` within `for i = 0..h-1`, and for
each `(i, j)` cell steps through `fwidth` (=17) rows:
```c
for (unsigned fi = 0; fi < fwidth; ++fi) {
    int ii_check = ii + fi;
    accum_ref += fcoeff * ref[ii_check * buf.stride + j];
```

This accesses `ref[ii_check * stride + j]` with stride steps of `buf.stride` bytes.
For 1080p (stride ~2048 bytes), the 17 tap positions are 34 KB apart in memory, far
exceeding L1 cache (32–64 KB). The AVX2 implementation in `vif_avx2.c` restructures
this into blocks of 16, but the scalar fallback suffers from it.

**Suggested fix:**
The SIMD path already handles this correctly. If falling back to scalar for width
remainders, consider transposing the `fwidth`-deep column into a local array before
the inner loop to improve locality.

---

### 4b. `cambi.c` — `c_values_histograms` Layout is `[value][col]` (Medium)

**File:** `libvmaf/src/feature/cambi.c`
**Lines:** 832–854 (`c_value_pixel`), 947 (histogram layout comment)

```c
// histograms[i * width + j] accesses the j'th histogram, i'th value
// This is done for cache optimization reasons
uint16_t p_0 = histograms[value * histogram_width + histogram_col];
...
uint16_t p_1 = histograms[(value + diffs[num_diffs + d + 1]) * histogram_width + histogram_col];
uint16_t p_2 = histograms[(value + diffs[num_diffs - d - 1]) * histogram_width + histogram_col];
```

The comment claims the layout `[value][col]` is a cache optimization. However,
`c_value_pixel` accesses `histograms[value * width + col]` for a fixed `col` and
varying `value` offsets — this is a strided access pattern over rows, which has poor
spatial locality (each `value + diff` step lands `width` uint16_t's = `2*width` bytes
away in memory). If the layout were `[col][value]` (`histograms[col * num_bins + value]`),
the inner loop over `d` would be sequential in memory.

**Uncertainty note:** The histogram update functions (`increment_range`,
`decrement_range`, and their AVX2 variants) access `histograms[val * width + left..right]`
— a horizontal range for a fixed `val`. With layout `[value][col]`, these updates are
sequential. A layout change would trade off update locality vs. query locality; which
is dominant depends on the ratio of updates to queries, which varies with mask density.
Profiling is required before acting on this.

---

### 4c. `integer_adm.c` — `adm_decouple` Reads Six Band Arrays With Interleaved Pattern (Low)

**File:** `libvmaf/src/feature/integer_adm.c`
**Lines:** 660–778

`adm_decouple` reads from 6 `int16_t` band arrays (`ref.band_h`, `ref.band_v`,
`ref.band_d`, `dis.band_h`, `dis.band_v`, `dis.band_d`) interleaved per pixel `(i,j)`.
These six arrays are in separate heap allocations. For a 1080p half-resolution subband
(960×540 `int16_t` = ~1 MB each), accessing 6 non-contiguous arrays per pixel is poor
for the hardware prefetcher.

A structure-of-arrays to array-of-structures layout change (one `[i*stride+j]`
access into a struct `{oh, ov, od, th, tv, td}` array) would improve locality at the
cost of a more invasive refactor.

---

## 5. Parallelism Opportunities

### 5a. `integer_vif.c` — Multiscale VIF Scales Are Computed Sequentially (Medium)

**File:** `libvmaf/src/feature/integer_vif.c`
**Lines:** 704–801 (the `extract` function)

VIF computes 4 scales sequentially: scale 1 feeds scale 2, which feeds scale 3. Each
scale is about ¼ the pixels of the previous. Scales 2, 3, and 4 could theoretically
be computed in parallel after scale 1 finishes, but in practice the sequential
dependency (each scale's input is the subsampled output of the previous) means the
scales must be serial.

However, the two independent passes within a scale — the reference-signal processing
and the distorted-signal processing — could be parallelised. Each calls the same filter
pipeline independently. Currently they share a single `VifBuffer`, which prevents
concurrent access.

**Uncertainty note:** The potential speedup from parallelising within a single frame is
limited because the thread pool works at the feature-extractor level (each extractor
runs in its own thread). Within a single extractor, additional parallelism would require
a second level of threading.

---

### 5b. `integer_adm.c` — Four ADM Scales Are Computed Sequentially, Some Could Overlap (Low)

**File:** `libvmaf/src/feature/integer_adm.c`
**Lines:** 2453–2519 (`integer_compute_adm`, the scale loop)

The four ADM scales are inherently dependent (scale N's DWT input is scale N-1's
lowband output). However, within each scale, the `adm_decouple`, `adm_csf`, and
`adm_cm` phases are sequential though they could pipeline across scales — computing
`adm_decouple` for scale 2 while `adm_cm` for scale 1 is still running.

This is a speculative and low-priority opportunity given the implementation complexity.

---

## 6. Python Orchestration Overhead

### 6a. Feature Extractor Subclasses Launch Separate `vmaf` Subprocess Per Feature Type (High)

**File:** `python/vmaf/core/vmafexec_feature_extractor.py`, `python/vmaf/__init__.py`
**Lines:** `__init__.py` lines 140–192 (`call_vmafexec_multi_features`)

Each `FeatureExtractor` subclass (e.g., `IntegerVifFeatureExtractor`,
`IntegerAdmFeatureExtractor`, etc.) calls `ExternalProgramCaller.call_vmafexec_single_feature`,
which in turn calls `call_vmafexec_multi_features` and launches a new subprocess:

```python
run_process(' '.join(cmd), shell=True)
```

Each subprocess independently opens and reads the full reference and distorted YUV files.
For a 1080p, 100-frame sequence, if 4 feature extractors are requested, the YUV files
are read **4 times** — once per feature type. This is the dominant I/O cost in the Python
orchestration path and is unrelated to the C-level scoring performance.

The `VmafQualityRunner` path avoids this by calling `vmaf` once with all features in a
single subprocess, but individual `FeatureExtractor` subclasses used in isolation (e.g.
`IntegerVifFeatureExtractor` or `IntegerAdmFeatureExtractor`) each re-read the files.

**Suggested fix:**
When the `FeatureAssembler` (`python/vmaf/core/feature_assembler.py`) discovers that
multiple feature types are needed, it should consolidate them into a single
`call_vmafexec_multi_features` call, rather than invoking one subprocess per extractor.
This is architecturally non-trivial but is the highest-impact Python-side improvement.

---

### 6b. `run_process` Uses `shell=True` (Low)

**File:** `python/vmaf/__init__.py`
**Lines:** 192

```python
run_process(' '.join(cmd), shell=True)
```

Using `shell=True` adds a shell interpreter process (e.g. `/bin/sh`) between the Python
process and the vmaf binary, adding a small but measurable overhead on each invocation.
When `cmd` is a list, passing it directly with `shell=False` is faster and safer.

---

### 6c. `VmafLegacyQualityRunner` Calls `svm_predict` Frame-by-Frame in Python (Low)

**File:** `python/vmaf/core/quality_runner.py`
**Lines:** 236–244

```python
for score_vector in zip(*ordered_scaled_scores_list):
    vif, adm, ansnr, motion = score_vector
    xs = [[vif, adm, ansnr, motion]]
    score = svmutil.svm_predict([0], xs, model)[0][0]
```

Each frame calls `svmutil.svm_predict` individually. A batch prediction over all frames
at once would reduce Python call overhead:
```python
all_xs = [[vif, adm, ansnr, motion] for vif, adm, ansnr, motion in ...]
scores = svmutil.svm_predict([0]*n, all_xs, model)[0]
```

This applies only to `VmafLegacyQualityRunner` (the Python-side SVM path).
`VmafQualityRunner` delegates to the C library which already handles this efficiently.

---

## 7. Additional SIMD Gaps (Default Integer Path)

### 7a. `integer_motion.c` — `y_convolution_8` / `y_convolution_16` Have No SIMD (Medium)

**File:** `libvmaf/src/feature/integer_motion.c`
**Lines:** 182–225 (`y_convolution_8`), 117–159 (`y_convolution_16`)

The vertical Gaussian filter pass (`y_convolution`) is entirely scalar. For non-edge
pixels (top_edge to bottom_edge), the inner loop applies a 5-tap vertical filter:

```c
for (unsigned i = top_edge; i < bottom_edge; i++) {
    for (unsigned j = 0; j < width; ++j) {
        uint32_t accum = 0;
        for (int k = 0; k < filter_width; ++k) {
            accum += filter[k] * (*src_p2);
            src_p2 += src_stride;
        }
        dst[i * dst_stride + j] = (accum + add_before_shift) >> shift_var;
    }
}
```

The `x_convolution_16` function already has AVX2/AVX-512 implementations
(`motion_avx2.c`, `motion_avx512.c`), but `y_convolution` has none. This is a 1D
vertical Gaussian filter — horizontally adjacent output pixels are independent and
can be computed 16 at a time with `_mm256_madd_epi16`.

**Suggested fix:**
Add `y_convolution_16_avx2` that processes 16 output pixels per iteration using
vertical tap multiply-accumulate with AVX2. Dispatch via function pointer in `init()`.

---

### 7b. VIF AVX2 — Per-Element Scalar Processing After SIMD Blocks (High)

**File:** `libvmaf/src/feature/x86/vif_avx2.c`
**Lines:** 485–537 (inside `vif_statistic_8_avx2`)

After computing 16 sigma values (xx, yy, xy) via AVX2, the code extracts them to
stack arrays (lines 138–140) and processes them one-by-one in a scalar loop with
branches (sigma_nsq threshold check + `log2` computation):

```c
// 16 sigma values computed via SIMD, then:
for (int s_idx = 0; s_idx < 16; s_idx++) {
    if (xx[s_idx] < sigma_nsq) { ... }
    else { log2_32(...); ... }
    num_log += ...;
    den_log += ...;
}
```

This defeats the SIMD parallelism for the most compute-intensive part of VIF
scoring. The threshold check can use `_mm256_cmpgt_epi32` and the `log2`
approximation can be vectorised with AVX2 bit-manipulation (CLZ via
`_mm256_lzcnt_epi32` on AVX-512 or via float-cast trick on AVX2).

**Suggested fix:**
Vectorise the sigma threshold + log2 accumulation using AVX2 masked operations
(`_mm256_blendv_epi8`). Accumulate `num_log` / `den_log` in SIMD registers and
reduce horizontally at the end of each row.

---

### 7c. VIF AVX2 — Filter Coefficient Constants Reloaded Inside Inner Loop (Medium)

**File:** `libvmaf/src/feature/x86/vif_avx2.c`
**Lines:** 259, 293, 354, 823, 861, 925

Filter coefficients are broadcast via `_mm256_set1_epi32(vif_filt_s0[fj])` inside
the `fwidth/2` loop instead of being hoisted before the `j` loop:

```c
for (unsigned fj = 0; fj < fwidth_half; ++fj) {
    __m256i fq = _mm256_set1_epi32(vif_filt_s0[fj]);  // reloaded every j-iter
    ...
}
```

The filter has at most 9 taps. Pre-loading all tap constants into registers before
the loop eliminates redundant `_mm256_set1_epi32` broadcasts.

**Suggested fix:**
Pre-load `vif_filt_s0[0..fwidth_half-1]` into an array of `__m256i` registers
before the `j` loop. Reference them by index inside the inner loop.

---

### 7d. VIF — Horizontal Pixel Padding Is Element-by-Element (Medium)

**File:** `libvmaf/src/feature/integer_vif.h`
**Lines:** 82–116 (`PADDING_SQ_DATA` and `PADDING_SQ_DATA_2`)

The padding macros reflect border values one element at a time for 5 arrays
(mu1, mu2, ref, dis, ref_dis):

```c
for (unsigned f = 1; f <= fwidth_half; ++f) {
    buf.tmp.mu1[-f] = buf.tmp.mu1[f];
    buf.tmp.mu2[-f] = buf.tmp.mu2[f];
    buf.tmp.ref[-f] = buf.tmp.ref[f];
    buf.tmp.dis[-f] = buf.tmp.dis[f];
    buf.tmp.ref_dis[-f] = buf.tmp.ref_dis[f];
    // ... right side similar
}
```

Called once per output row per VIF scale. For fwidth_half=8, this is 80 individual
32-bit copies per row. With AVX2 broadcast + store, the left/right padding zones
(8 elements × 4 bytes = 32 bytes each) fit in a single 256-bit register.

**Suggested fix:**
Use `_mm256_set1_epi32(edge_value)` + `_mm256_storeu_si256` to fill each padding
zone in one instruction per array.

---

### 7e. ADM — `adm_decouple` Angle Flag Uses Per-Pixel Float Division (High)

**File:** `libvmaf/src/feature/integer_adm.c`
**Lines:** 745–747 (`adm_decouple`), 882–884 (`adm_decouple_s123`)

The per-pixel angle flag computation promotes int64 dot products to float and
performs 4 float divisions:

```c
int angle_flag = (((float)ot_dp / 4096.0) >= 0.0f) &&
    (((float)ot_dp / 4096.0) * ((float)ot_dp / 4096.0) >=
        cos_1deg_sq * ((float)o_mag_sq / 4096.0) * ((float)t_mag_sq / 4096.0));
```

Since the divisor is a constant power-of-two (4096 = 2^12), these divisions are
equivalent to right-shifts in fixed-point. The comparison `ot_dp² ≥ cos_1deg_sq ×
o_mag_sq × t_mag_sq` can be done entirely in integer arithmetic (after scaling),
eliminating all float conversion. With AVX2, 8 pixels can be evaluated in parallel
using `_mm256_cmpgt_epi64`.

**Note:** This is a sub-finding of 1c but provides a concrete vectorisation strategy
for the angle flag, which is the most branch-heavy part of the inner loop.

**Suggested fix:**
Replace float comparison with fixed-point integer comparison. Vectorise with AVX2
64-bit compare and blend.

---

### 7f. ADM — Scalar Clamps in `adm_decouple` Use Nested Ternaries (Medium)

**File:** `libvmaf/src/feature/integer_adm.c`
**Lines:** 760–762 (`adm_decouple`)

Three values are clamped with nested ternary operators:

```c
int32_t kh = tmp_kh < 0 ? 0 : (tmp_kh > 32768 ? 32768 : tmp_kh);
int32_t kv = tmp_kv < 0 ? 0 : (tmp_kv > 32768 ? 32768 : tmp_kv);
int32_t kd = tmp_kd < 0 ? 0 : (tmp_kd > 32768 ? 32768 : tmp_kd);
```

AVX2 replaces each clamp with two instructions: `_mm256_max_epi32(x, zero)` +
`_mm256_min_epi32(x, limit)`, processing 8 values per register with zero branches.

**Note:** Sub-finding of 1c.

---

### 7g. ADM — `adm_csf` Triple-Angle Multiply-Shift-Store Loop Has No SIMD (Medium-High)

**File:** `libvmaf/src/feature/integer_adm.c`
**Lines:** 1025–1042 (`adm_csf`), 1097–1118 (`i4_adm_csf`)

The CSF loop iterates over 3 angles × height × width, applying a per-pixel
multiply, shift, abs, and second multiply:

```c
for (int theta = 0; theta < 3; ++theta) {
    for (int i = top; i < bottom; ++i) {
        for (int j = left; j < right; ++j) {
            int32_t dst_val = i_rfactor[theta] * (int32_t)src_ptr[j];
            int16_t i16_dst_val = (int16_t)((dst_val + add) >> shift);
            dst_ptr[j] = i16_dst_val;
            flt_ptr[j] = (int16_t)(((FIX_ONE_BY_30 * abs((int32_t)i16_dst_val)) + 2048) >> 12);
        }
    }
}
```

This is a natural fit for AVX2: `_mm256_mullo_epi16` for the factor multiply,
`_mm256_abs_epi16` + `_mm256_mullo_epi32` for the filtered output, processing
16 int16 values per iteration.

**Note:** Sub-finding of 1c.

---

### 7h. ADM — `adm_csf_den_scale` / `adm_csf_den_s123` Cubing Accumulation Has No SIMD (Medium)

**File:** `libvmaf/src/feature/integer_adm.c`
**Lines:** 1151–1185 (`adm_csf_den_scale`), 1207–1290 (`adm_csf_den_s123`)

Accumulates cubes of absolute values for all band pixels:

```c
for (int j = left; j < right; ++j) {
    uint16_t h = (uint16_t)abs(src_h[j]);
    uint64_t val = ((uint64_t)h * h) * h;
    accum_inner_h += val;
}
```

AVX2 can compute `abs` with `_mm256_abs_epi16`, square with `_mm256_mullo_epi16` +
widening, and accumulate the cube into 64-bit accumulators. The `s123` variant uses
int32 data (8 values per register instead of 16).

**Note:** Sub-finding of 1c.

---

### 7i. ADM — `adm_cm` Macro-Expanded Abs/Clamp/Cube Inner Loop Has No SIMD (High)

**File:** `libvmaf/src/feature/integer_adm.c`
**Lines:** 1376–1472 (center region of `adm_cm`)

The `ADM_CM_ACCUM_ROUND` macro expands to per-pixel: abs → subtract threshold →
clamp to zero → square → cube → accumulate:

```c
x = abs(x) - (thr << shift);
x = x < 0 ? 0 : x;
x_sq = (int32_t)((((int64_t)x * x) + add) >> shift);
val = (((int64_t)x_sq * x) + add) >> shift;
accum_inner += val;
```

The center region ("completely within frame", lines 1436–1472) is the dominant code
path and has no boundary-condition branches. AVX2 can process 8 int32 values per
iteration using `_mm256_abs_epi32`, `_mm256_sub_epi32`, `_mm256_max_epi32(x, 0)`,
and widening multiplies.

**Note:** Sub-finding of 1c. This is the most compute-intensive part of ADM.

---

## 8. Additional Memory / Allocation

### 8a. Float Extractors — Per-Frame Buffer Allocations (Medium)

**Files:**
- `libvmaf/src/feature/adm.c` line 128 (20 buffers)
- `libvmaf/src/feature/vif.c` line 129 (8 buffers)
- `libvmaf/src/feature/ansnr.c` line 69 (2 buffers)
- `libvmaf/src/feature/ms_ssim.c` line 105 (5 buffers per scale)

All float feature extractors allocate large working buffers inside `compute_*()`
on every frame and free them at the end. For a 1080p stream with all float
extractors enabled, this is ~35 `aligned_malloc`/`free` pairs per frame.

**Suggested fix:**
Move buffer allocation to each extractor's `init()` callback and store in the
extractor state struct. Free in `close()`.

---

### 8b. `picture.c` — Two Small `malloc` Calls Per Picture (Low)

**File:** `libvmaf/src/picture.c`
**Lines:** 50–57 (`vmaf_picture_priv_init`), 98–103 (`vmaf_picture_alloc`)

Every picture allocation triggers two separate small heap allocations:
- `VmafPicturePrivate` (~8 bytes)
- `VmafRef` (~8 bytes, atomic counter)

In the threaded path with `n_threads × n_features` concurrent pictures, this adds
up. Both structs are tiny and could be embedded directly in `VmafPicture` or
allocated as a single block with the pixel data.

**Suggested fix:**
Embed `VmafRef` directly in `VmafPicture`, or allocate priv + ref as one
contiguous block.

---

### 8c. `picture_copy.c` — uint8/uint16 to Float Conversion Has No SIMD (Low-Medium)

**File:** `libvmaf/src/feature/picture_copy.c`
**Lines:** 56–62 (8-bit path)

```c
for (unsigned j = 0; j < src->w[0]; j++) {
    float_data[j] = (float) data[j] + offset;
}
```

Called by every float extractor at the start of each frame. At 1080p, this is
2.07M scalar integer-to-float conversions. AVX2 can process 8 pixels per iteration
with `_mm_loadu_si64` → `_mm256_cvtepu8_epi32` → `_mm256_cvtepi32_ps` →
`_mm256_add_ps`.

**Suggested fix:**
Add AVX2 path for `picture_copy` with SIMD type conversion. Dispatch via CPU
flag check.

---

## 9. Additional Algorithmic Redundancy

### 9a. `psnr_hvs.c` — Constant Mask Table Recomputed Per Frame (Low)

**File:** `libvmaf/src/feature/third_party/xiph/psnr_hvs.c`
**Lines:** 236–239

```c
for (x = 0; x < 8; x++)
    for (y = 0; y < 8; y++)
        mask[x][y] = (_csf[x][y] * 0.3885746225901003) *
                     (_csf[x][y] * 0.3885746225901003);
```

`_csf` is a compile-time constant array. The `mask` table is therefore constant
and should be precomputed once (in `init()` or as a static const).

**Suggested fix:**
Replace with a `static const double mask[8][8] = { ... }` initialiser containing
the precomputed values.

---

### 9b. CIEDE2000 — Per-Pixel Transcendental Functions in YUV→LAB Conversion (High if used)

**File:** `libvmaf/src/feature/ciede.c`
**Lines:** 273–308 (`get_lab_color`), 201–237 (`ciede2000`), 334–367 (main loop)

The CIEDE2000 extractor performs two YUV→LAB conversions per pixel, each involving
`pow()`, `sqrt()`, and the CIEDE2000 delta itself uses `pow()`, `sqrt()`, `sin()`,
`cos()`, `atan2()`. At 1080p this is ~4M transcendental function calls per frame.

```c
for (unsigned i = 0; i < ref->h[0]; i++) {
    for (unsigned j = 0; j < ref->w[0]; j++) {
        const LABColor c1 = get_lab_color(r_y, r_u, r_v, bpc);  // pow, sqrt
        const LABColor c2 = get_lab_color(d_y, d_u, d_v, bpc);  // pow, sqrt
        de00_sum += ciede2000(c1, c2, ksub);                     // pow, sqrt, trig
    }
}
```

**Suggested fix:**
Build a 256-entry (8-bit) or 1024-entry (10-bit) lookup table for the YUV→LAB
conversion during `init()`. For CIEDE2000 delta, consider AVX2 vectorisation of
the arithmetic using approximate SIMD transcendentals (e.g., fast `rsqrt` +
Newton-Raphson refinement for `sqrt`, polynomial approximation for `atan2`).

---

### 9c. CIEDE2000 — Per-Frame YUV444 Buffer Allocation (Low)

**File:** `libvmaf/src/feature/ciede.c`
**Lines:** ~61–95

When the input is YUV420 or YUV422, `ciede.c` allocates a full YUV444 upsampled
buffer per frame. This should be pre-allocated in `init()` based on the known
picture format and dimensions.

---

## 10. Secondary Extractor SIMD Gaps

### 10a. `psnr_hvs.c` — Scalar 8×8 DCT (Medium-High if used)

**File:** `libvmaf/src/feature/third_party/xiph/psnr_hvs.c`
**Lines:** 155–163 (`od_bin_fdct8x8`), 241–332 (main block loop)

PSNR-HVS processes the frame in non-overlapping 8×8 blocks. Each block undergoes
a forward DCT via `od_bin_fdct8` (called 16 times: 8 rows + 8 columns). At 1080p
with step=7, this is ~39k blocks × 16 = ~624k scalar DCT-8 transforms per frame.

The DCT-8 is well-suited to AVX2: 8 butterfly stages can be computed using
`_mm256_add_epi32` / `_mm256_sub_epi32` with register permutations.

**Suggested fix:**
Replace `od_bin_fdct8` with an AVX2 DCT-8 that processes all 8 rows (or columns)
simultaneously. This reduces 8 scalar DCT calls to 1 SIMD call per direction.

---

### 10b. `psnr_hvs.c` — Scalar Error Weighting Loop (Low)

**File:** `libvmaf/src/feature/third_party/xiph/psnr_hvs.c`
**Lines:** 320–331

Per-block error weighting iterates over 64 DCT coefficients with a branch per
coefficient:

```c
for (i = 0; i < 8; i++) {
    for (j = 0; j < 8; j++) {
        err = abs(dct_s[i*8+j] - dct_d[i*8+j]);
        if (i != 0 || j != 0)
            err = err < s_mask / mask[i][j] ? 0 : err - s_mask / mask[i][j];
        ret += (err * _csf[i][j]) * (err * _csf[i][j]);
    }
}
```

64 values fit in 8 AVX2 registers. The conditional can be replaced with
`_mm256_max_ps(err - threshold, zero)` (branchless clamp).

---

### 10c. CAMBI — `filter_mode` (Mode-3 Median) Has No SIMD (Medium)

**File:** `libvmaf/src/feature/cambi.c`
**Lines:** 711–729

Applies a 3-tap mode filter horizontally then vertically using a 3-line sliding
window. Called once per scale (5 scales) on potentially large images:

```c
for (int j = 1; j < width - 1; j++) {
    buffer[j] = mode3(data[j-1], data[j], data[j+1]);
}
```

The `mode3` function (lines 705–709) uses two equality checks and a min-of-3
fallback — a natural fit for branchless AVX2 using `_mm256_cmpeq_epi16` +
`_mm256_blendv_epi8` + `_mm256_min_epi16`.

**Suggested fix:**
Vectorise `mode3` for 16 uint16 values per iteration. Use `cmpeq` masks to
select matching values, fall through to `min3` via blend.

---

### 10d. CAMBI — `calculate_c_values_row` Per-Pixel Loop (Medium)

**File:** `libvmaf/src/feature/cambi.c`
**Lines:** 919–930

The per-row C-value computation iterates pixel-by-pixel with a mask check:

```c
for (int col = 0; col < width; col++) {
    if (mask[row * stride + col]) {
        c_values[row * width + col] = c_value_pixel(...);
    }
}
```

The mask check is branch-per-pixel. AVX2 can load 16 mask values, compute a
bitmask with `_mm256_movemask_epi8`, and skip fully-zero 16-pixel blocks. For
blocks with mixed mask values, the inner `c_value_pixel` histogram queries can be
partially vectorised with `_mm256_i32gather_epi32` for histogram lookups.

---

### 10e. CAMBI — `decimate` / `decimate_generic_*` Scalar Pixel Copy (Low-Medium)

**File:** `libvmaf/src/feature/cambi.c`
**Lines:** 464–586 (`decimate_generic_*`), 689–697 (`decimate`)

The same-size path in `decimate_generic_uint16_and_convert_to_10b` (lines 560–564)
is a simple left-shift loop:

```c
for (unsigned j = 0; j < out_w; j++) {
    out_data[i * out_stride + j] = data[i * stride + j] << shift_factor;
}
```

Straightforward to vectorise with `_mm256_slli_epi16`. The subsampled `decimate`
function (lines 689–697) reads every 2nd row and column — harder to vectorise
efficiently but could use shuffled loads.

---

### 10f. Float Extractors — All Have Zero SIMD (Medium overall)

**Files:** `libvmaf/src/feature/float_adm.c`, `float_vif.c`, `float_psnr.c`,
`float_ssim.c`, `float_ms_ssim.c`, `float_motion.c`, `float_ansnr.c`,
`float_moment.c`

None of the eight float feature extractors have any SIMD implementation. Their
backing computation files (`adm.c`, `vif.c`, `psnr.c`, `ssim.c`, `ms_ssim.c`,
`motion.c`, `ansnr.c`, `moment.c`) contain only scalar C loops. By contrast, their
integer counterparts have AVX2/AVX-512 back-ends.

The most impactful individual loops (if float extractors are enabled):

| Extractor | Hot loop | AVX2 gain estimate |
|-----------|----------|--------------------|
| `float_psnr` (`psnr.c:42–51`) | SSE accumulation | 6–8× |
| `float_motion` (`motion.c:44–60`) | SAD with `fabs()` | 7–8× |
| `float_ssim` (`ssim.c` via `iqa/convolve.c`) | 11×11 Gaussian conv | 8–12× |
| `float_ms_ssim` (`ms_ssim.c` via IQA) | 9×9 LPF + decimation | 6–8× |
| `float_vif` (`vif.c:147–249`) | Multi-scale filter + stats | 6–8× |
| `float_adm` (`adm.c:71–292`) | DWT + decouple + CSF | 6–8× |

**Uncertainty note:** Float extractors are optional (`-Denable_float=true`) and not
used in the default inference path. Impact is only relevant when explicitly enabled.

---

### 10g. IQA Library — Scalar 2D Convolution (Low)

**File:** `libvmaf/src/feature/iqa/convolve.c`
**Lines:** 90–200

The IQA convolution library (used by `float_ssim` and `float_ms_ssim`) performs
non-separable 2D convolution — it applies the full 2D kernel (11×11 = 121 MACs per
pixel for SSIM, 9×9 = 81 for MS-SSIM) instead of separable 1D horizontal + 1D
vertical passes. Separable filtering reduces 121 MACs to 22 (11+11).

**Suggested fix:**
Replace `_iqa_filter_pixel` with separable horizontal + vertical 1D passes. Use
`_mm256_fmadd_ps` for the 1D filter taps. The integer VIF code already demonstrates
this pattern via `common/convolution_avx.c`.

---

## 11. Python Orchestration (Additional)

### 11a. `feature_extractor.py` — Triple-Nested Regex Loop in Log Parsing (Low)

**File:** `python/vmaf/core/feature_extractor.py`
**Lines:** 70–100

Frame-by-frame log file parsing uses a triple-nested loop: for each line in the
log file, for each atom feature, format and apply a regex. For N frames and M
features, this is O(N×M) regex compilations.

**Suggested fix:**
Pre-compile all regex patterns once at class level. Better: parse each line once
and extract all features in a single pass (e.g., a combined regex with named groups).

---

### 11b. `result_store.py` — `ast.literal_eval()` for Result Deserialization (Low)

**File:** `python/vmaf/core/result_store.py`
**Lines:** 78–85

Results are serialised as Python `repr()` strings and deserialised with
`ast.literal_eval()`. JSON parsing is 5–10× faster for structured data.

**Suggested fix:**
Switch to JSON serialisation (`json.dump` / `json.load`) for result storage.

---

### 11c. `quality_runner.py` — Model Loaded Twice Per Asset (Low)

**File:** `python/vmaf/core/quality_runner.py`
**Lines:** 284–293 (`_get_vmaf_feature_assembler_instance`), 472–473 (`_create_prediction_result_dict`)

`VmafQualityRunner` calls `_load_model(asset)` twice per asset — once to set up
the feature assembler and once to create the prediction result dict.

**Suggested fix:**
Cache the loaded model on the runner instance after the first load.

---

### 11d. `feature_assembler.py` — Exception-Based Wildcard Key Matching (Low)

**File:** `python/vmaf/core/feature_assembler.py`
**Lines:** 89–102

Feature score lookup uses `try/except KeyError` with a regex-based wildcard
fallback. For large feature sets with many mismatched keys, every miss triggers
the expensive fallback.

**Suggested fix:**
Pre-build a feature key → scores_key mapping on first access. Use `dict.get()`
instead of exception-based control flow.

---

## 12. Infrastructure

### 12a. `feature_collector.c` — Metadata Callback Releases Lock Repeatedly (Low)

**File:** `libvmaf/src/feature/feature_collector.c`
**Lines:** 383–423

On every feature score append with metadata callbacks registered, the code
releases and re-acquires `feature_collector->lock` in an inner loop over models
(lines 407, 410). This pattern causes lock thrashing and opens a race window.

**Suggested fix:**
Batch the prediction calls outside the lock, or defer metadata callbacks to a
post-append phase.

---

### 12b. `framesync.c` — Linear Buffer Queue Traversal (Low)

**File:** `libvmaf/src/framesync.c`
**Lines:** 69–110

Buffer acquire traverses the linked list from head on every call. For typical
buffer counts (2–4) this is fast, but maintaining a free-list pointer would
eliminate the traversal entirely.

---

### 12c. `libvmaf.c` — `validate_pic_params` Redundant After First Frame (Low)

**File:** `libvmaf/src/libvmaf.c`
**Lines:** 461–490

Frame dimension and format validation runs on every `vmaf_read_pictures` call.
After the first frame establishes `vmaf->pic_params`, subsequent checks are
redundant (dimensions cannot change mid-stream).

**Suggested fix:**
Set a `pic_params_validated` flag after the first frame and skip full validation
on subsequent frames.

---

## Summary Table

| # | Category | Finding | Impact | Uncertainty |
|---|----------|---------|--------|-------------|
| 1a | SIMD gap | `integer_ssim.c` — no SIMD, per-frame malloc for kernels | High | Low |
| 1b | SIMD gap | `integer_psnr.c` — 8-bit and HBD SSE loops, no SIMD | Medium | Low |
| 1c | SIMD gap | `adm_decouple` / `adm_csf` / `adm_cm` — no SIMD | High | Medium |
| 1d | SIMD gap | `sad_c` in motion extractor — `s->sad` never overridden | Medium | Low |
| 1e | SIMD gap | `adm_avx2.c` horizontal pass — scalar tail | Low | Low |
| 2a | Memory | `integer_ssim.c` — 4 malloc/free per frame in `calc_ssim` | High | Low |
| 2b | Memory | `predict.c` — malloc + context create/destroy per frame | Medium | Low |
| 2c | Memory | `cambi.c` — malloc in `dump_c_values` per scale per frame | Low | Low |
| 2d | Memory | `thread_pool.c` — per-job malloc/memcpy/free | Medium | Medium |
| 2e | Memory | `feature_collector.c` — O(n) linear search under lock | Medium | Low |
| 3a | Algorithmic | `dwt_quant_step` (log10/pow) called 15+ times per frame | High | Low |
| 3b | Algorithmic | `cos_1deg_sq` computed via `cos()` 8 times per frame | Medium | Low |
| 3c | Algorithmic | `div_lookup_generator` called per extractor init | Low | Low |
| 3d | Algorithmic | `pow(2, k)` / `ceil(log2(x))` in per-scale setup | Medium | Low |
| 3e | Algorithmic | Loop-invariant `ii` computed inside inner filter loop | Low | Low |
| 3f | Algorithmic | Duplicate `dwt_quant_step` between den_scale and cm | Medium | Low |
| 4a | Cache | VIF vertical pass — strided access (scalar path) | Medium | Low |
| 4b | Cache | CAMBI histogram layout may be suboptimal for query path | Medium | High |
| 4c | Cache | ADM decouple — 6 separate band arrays per pixel | Low | Medium |
| 5a | Parallelism | VIF ref/dis sub-paths could be independent | Medium | High |
| 5b | Parallelism | ADM scale pipeline could partially overlap | Low | High |
| 6a | Python | Each FeatureExtractor launches separate subprocess + re-reads YUV | High | Low |
| 6b | Python | `shell=True` adds interpreter overhead per subprocess | Low | Low |
| 6c | Python | Legacy SVM predict called frame-by-frame in Python | Low | Low |
| 7a | SIMD gap | `y_convolution_8/16` in motion — no SIMD | Medium | Low |
| 7b | SIMD gap | VIF AVX2 — per-element scalar after SIMD (sigma+log2) | High | Medium |
| 7c | SIMD gap | VIF AVX2 — filter coefficients reloaded inside inner loop | Medium | Low |
| 7d | SIMD gap | VIF horizontal pixel padding — element-by-element | Medium | Low |
| 7e | SIMD gap | ADM `adm_decouple` angle flag float division per pixel | High | Low |
| 7f | SIMD gap | ADM `adm_decouple` nested ternary clamps | Medium | Low |
| 7g | SIMD gap | ADM `adm_csf` triple-angle multiply-shift-store loop | Medium-High | Low |
| 7h | SIMD gap | ADM `adm_csf_den` cubing accumulation loop | Medium | Low |
| 7i | SIMD gap | ADM `adm_cm` abs/clamp/cube inner loop | High | Medium |
| 8a | Memory | Float extractors — 35+ per-frame buffer allocations | Medium | Low |
| 8b | Memory | `picture.c` — two small mallocs per picture | Low | Low |
| 8c | Memory | `picture_copy.c` — uint8/16→float with no SIMD | Low-Medium | Low |
| 9a | Algorithmic | `psnr_hvs.c` — constant mask table recomputed per frame | Low | Low |
| 9b | Algorithmic | CIEDE2000 — per-pixel transcendental funcs in YUV→LAB | High | Low |
| 9c | Algorithmic | CIEDE2000 — per-frame YUV444 buffer allocation | Low | Low |
| 10a | SIMD gap | `psnr_hvs.c` — scalar 8×8 DCT (~624k transforms/frame) | Medium-High | Low |
| 10b | SIMD gap | `psnr_hvs.c` — scalar error weighting (64 coeffs/block) | Low | Low |
| 10c | SIMD gap | CAMBI `filter_mode` — mode-3 median no SIMD | Medium | Low |
| 10d | SIMD gap | CAMBI `calculate_c_values_row` — per-pixel with mask branch | Medium | Medium |
| 10e | SIMD gap | CAMBI `decimate` / `decimate_generic` — scalar copy/shift | Low-Medium | Low |
| 10f | SIMD gap | All 8 float extractors — zero SIMD coverage | Medium | Low |
| 10g | SIMD gap | IQA library — non-separable 2D convolution | Low | Low |
| 11a | Python | Triple-nested regex loop in log parsing | Low | Low |
| 11b | Python | `ast.literal_eval()` for result deserialization | Low | Low |
| 11c | Python | Model loaded twice per asset in QualityRunner | Low | Low |
| 11d | Python | Exception-based wildcard key matching in assembler | Low | Low |
| 12a | Infra | Feature collector metadata callback lock thrashing | Low | Medium |
| 12b | Infra | Framesync buffer queue linear traversal | Low | Low |
| 12c | Infra | `validate_pic_params` redundant after first frame | Low | Low |
