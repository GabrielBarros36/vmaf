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
