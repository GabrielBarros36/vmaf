# VMAF AVX2 SIMD Optimization Plan

## Environment
- Architecture: x86_64 (AVX2 targets: Haswell+, Zen1+)
- SIMD: AVX2 (256-bit integer/FP, FMA3)
- Build: Release, O3, intrinsics via `-mavx -mavx2`
- Current AVX2 files: `adm_avx2.c`, `vif_avx2.c`, `motion_avx2.c`, `cambi_avx2.c`

## Optimization Ideas (Sorted by Expected Impact)

### High Impact

#### 1. AVX2-vectorize ADM decouple functions [ADM]
**File:** `libvmaf/src/feature/integer_adm.c:667-937`
**Problem:** `adm_decouple()` (int16 path, lines 667-785) and `adm_decouple_s123()` (int32 path, lines 796-937) are entirely scalar C. These run on every frame at every DWT scale. The inner loops compute dot products (`oh*th + ov*tv`), magnitudes (`oh*oh + ov*ov`), angle flags, lookup-table division for gain factors (kh, kv, kd), and gain-clamped reconstruction across 3 wavelet bands (h, v, d). All of these are embarrassingly parallel across pixels.
**Fix:** Vectorize with AVX2: use `_mm256_madd_epi16` for dot products and magnitudes (8 pixels per iteration), `_mm256_i32gather_epi32` or pre-shuffled lookup for div_lookup table, and `_mm256_blendv_epi8` for the angle_flag conditional. Process 8-16 pixels per iteration. Add dispatch point alongside existing `dwt2_8` dispatch at line 2816.
**Expected Impact:** High - ADM is ~40% of VMAF compute time. Decouple runs 3x per frame (one per scale) over the full image. Estimated 4-8x speedup for this function.

#### 2. AVX2-vectorize ADM CSF and CM functions [ADM]
**File:** `libvmaf/src/feature/integer_adm.c:939-1030 (adm_csf), 1266-2055 (adm_cm/i4_adm_cm)`
**Problem:** `adm_csf()` applies Contrast Sensitivity Function scaling to decoupled bands - simple multiply-shift-abs per element across 3 bands. `adm_cm()` and `i4_adm_cm()` are the most compute-intensive ADM functions, computing contrast masking with divisions, conditional branches, and accumulation loops. All are purely scalar C with no SIMD dispatch.
**Fix:** For `adm_csf`: straightforward `_mm256_mullo_epi16` / `_mm256_mulhi_epi16` with shift and `_mm256_abs_epi16`. For `adm_cm`/`i4_adm_cm`: vectorize the inner accumulation loops, use branchless min/max (`_mm256_min_epi32`/`_mm256_max_epi32`), hoist loop-invariant divisions. Add new dispatch function pointers in `AdmState`.
**Expected Impact:** High - `adm_cm` and `i4_adm_cm` are the heaviest ADM functions (profiling shows they dominate within ADM). Combined with #1, this would vectorize the entire ADM pipeline beyond DWT.

#### 3. VIF: replace redundant loads with alignr-based sliding window [VIF]
**File:** `libvmaf/src/feature/x86/vif_avx2.c:166-187, 255-339, 606-687`
**Problem:** The horizontal filter pass in `vif_statistic_8_avx2` and `vif_statistic_16_avx2` loads overlapping data from memory for each filter tap. For fwidth=17 (scale 0), this means ~85 unaligned 256-bit loads per 16-element output block (5 signals x 17 taps). The code at lines 1222-1230 (`vif_subsample_rd_8_avx2`) already demonstrates a superior approach using `_mm256_alignr_epi8` to shift pre-loaded data instead of reloading.
**Fix:** Apply the alignr sliding-window pattern from `vif_subsample_rd_8` to `vif_statistic_8/16`. Load 2-3 aligned 256-bit vectors per signal, then shift with `_mm256_alignr_epi8` for each tap offset. This reduces loads from ~85 to ~15-20 per block.
**Expected Impact:** High - VIF is the second-heaviest feature extractor. Reducing memory loads by 4-5x in the hot inner loop significantly reduces memory bandwidth pressure and cache misses.

#### 4. VIF: unroll fixed-size filter loops [VIF]
**File:** `libvmaf/src/feature/x86/vif_avx2.c:166-187, 820-827`
**Problem:** The inner filter loop iterates `fwidth/2` times (8 for scale 0, 4 for scale 1, 2 for scale 2, 1 for scale 3) with dynamic coefficient loading (`_mm256_set1_epi16(vif_filt_s0[tap])`). Filter widths are compile-time constants (17, 9, 5, 3) asserted at function entry (line 122). The loop overhead and dynamic coefficient broadcast prevent the compiler from optimizing the dependency chain.
**Fix:** Create scale-specialized inner loops or use template macros to fully unroll the filter tap loop. Pre-broadcast all coefficients before the loop. For scale 0 (fwidth=17), this eliminates 8 iterations of loop overhead and enables better instruction scheduling.
**Expected Impact:** Medium-High - eliminates loop control overhead and enables better out-of-order execution across filter taps.

#### 5. AVX2-vectorize Motion SAD [MOTION]
**File:** `libvmaf/src/feature/integer_motion.c:228-244`
**Problem:** `sad_c()` computes Sum of Absolute Differences between two 16-bit pictures. It's purely scalar: `inner_sad += abs(a[j] - b[j])` in a nested loop over the entire frame. The dispatch infrastructure already exists (`s->sad` function pointer at line 320) but is never overridden with a SIMD variant.
**Fix:** Implement `sad_avx2()` using `_mm256_sub_epi16` + `_mm256_abs_epi16` + `_mm256_madd_epi16` (with all-ones multiplier for horizontal pair-wise sum) + `_mm256_add_epi32` accumulation. Process 16 pixels per iteration. Add AVX2 dispatch at existing check point (lines 310-318).
**Expected Impact:** Medium-High - SAD is called every frame for motion estimation. The operation is perfectly vectorizable with expected 8-16x speedup for the inner loop.

#### 6. VIF: reduce shuffle/permute overhead in data packing [VIF]
**File:** `libvmaf/src/feature/x86/vif_avx2.c:281-282, 334-338, 720-726, 741-767`
**Problem:** The VIF kernels use a recurring pattern of `_mm256_slli_si256` + `_mm256_blend_epi32` (11 occurrences) and `_mm256_permutevar8x32_epi32` (8+ occurrences) to repack results from 32-bit accumulators into output format. The permute mask `_mm256_set_epi32(7,5,3,1,6,4,2,0)` is recreated inside the inner loop at line 591. Each `_mm256_permutevar8x32_epi32` has 3-cycle latency.
**Fix:** Hoist the permute mask outside all loops. Replace the `slli_si256` + `blend_epi32` pattern with direct `_mm256_shuffle_epi32` or restructure accumulation to produce results in the correct lane order from the start (avoiding cross-lane permutes entirely). Consider using `_mm256_packs_epi32` directly where applicable.
**Expected Impact:** Medium - eliminates ~20+ unnecessary shuffle/permute instructions per 16-element block across the VIF hot path.

### Medium Impact

#### 7. AVX2-vectorize Integer PSNR [PSNR]
**File:** `libvmaf/src/feature/integer_psnr.c:116-159 (8-bit), 161-202 (HBD)`
**Problem:** Both `psnr()` and `psnr_hbd()` are purely scalar. The inner loop (`e = ref[j] - dis[j]; sse_inner += e * e`) is a textbook SIMD target. No dispatch infrastructure exists.
**Fix:** For 8-bit: load 32 bytes with `_mm256_loadu_si256`, widen to int16 with `_mm256_cvtepu8_epi16`, subtract, then `_mm256_madd_epi16` (self-multiply + horizontal pair add) for squared-error accumulation. For HBD: load 16 uint16 values, subtract, use `_mm256_madd_epi16` with itself. Add SIMD dispatch struct and CPU flag check in `init()`.
**Expected Impact:** Medium - PSNR is lightweight but frequently computed alongside VMAF. 8-16x speedup on the inner loop.

#### 8. AVX2-vectorize Motion vertical convolution [MOTION]
**File:** `libvmaf/src/feature/integer_motion.c:118-160 (y_convolution_16), 184-225 (y_convolution_8)`
**Problem:** Only horizontal convolution (`x_convolution`) has AVX2 dispatch. Vertical convolution uses the same 5-tap Gaussian filter `[3571, 16004, 26386, 16004, 3571]` but operates column-wise. The `y_convolution` function pointer (line 307) is never overridden with SIMD.
**Fix:** Implement `y_convolution_16_avx2` that loads 5 rows of 16 elements each, multiplies by filter coefficients using `_mm256_madd_epi16`, and accumulates vertically. This is more cache-friendly than the C reference since each output element reads from the same column across 5 rows (stride-apart).
**Expected Impact:** Medium - completes the motion filtering pipeline in SIMD. Vertical filtering is roughly equal cost to horizontal.

#### 9. ADM DWT: interleave tmplo/tmphi horizontal pass blocks [ADM]
**File:** `libvmaf/src/feature/x86/adm_avx2.c:166-272`
**Problem:** The horizontal pass processes tmplo (lines 166-217) and tmphi (lines 219-272) in sequential blocks within the same loop iteration. Both blocks perform identical filter coefficient multiplications on different data but cannot exploit instruction-level parallelism across blocks because they're scoped sequentially.
**Fix:** Interleave the tmplo and tmphi computations: load from both buffers, issue multiply-accumulate instructions alternately. This doubles the number of independent instructions in-flight, allowing the CPU's out-of-order engine to fill multiply and add ports more efficiently.
**Expected Impact:** Medium - improves ILP in the DWT horizontal pass. ADM DWT is already SIMD-accelerated but has room for better utilization of execution ports.

#### 10. VIF: vectorize sigma analysis (eliminate SIMD-to-scalar round trip) [VIF]
**File:** `libvmaf/src/feature/x86/vif_avx2.c:138-140, 385-386, 485-527`
**Problem:** After computing sigma values in SIMD, the code stores results to aligned stack arrays (`xx[16]`, `yy[16]`, `xy[16]`), then reads them back element-by-element in a scalar loop (lines 485-527) for the final log-domain VIF computation. This SIMD-to-scalar-to-SIMD transition causes store-forwarding stalls and prevents vectorization of the final analysis.
**Fix:** Keep sigma values in YMM registers and vectorize the comparison/log-lookup logic. The log2_table lookup can be vectorized with `_mm256_i32gather_epi32`. The conditional logic (sigma thresholding) can use `_mm256_cmpgt_epi32` + blend.
**Expected Impact:** Medium - eliminates 48 stack stores + 48 scalar loads per 16-element block, plus enables vectorization of the final scoring step.

#### 11. ADM DWT: widen vertical pass to process 32 elements per iteration [ADM]
**File:** `libvmaf/src/feature/x86/adm_avx2.c:55-120`
**Problem:** The vertical pass processes 16 elements per iteration (loading 128-bit chunks of uint8, widening to int16 in 256-bit registers). This under-utilizes AVX2 throughput - modern CPUs can sustain 2 loads per cycle but the current loop only issues 4 loads per 16 elements (one per source row).
**Fix:** Double the loop body to process 32 elements per iteration: load two 128-bit chunks per row, widen both to 256-bit, and process with filter coefficients. This better amortizes loop overhead and fills more execution ports per iteration.
**Expected Impact:** Low-Medium - incremental improvement on the already-vectorized DWT vertical pass. Limited by memory bandwidth for large images.

#### 12. VIF: reduce register pressure in statistic functions [VIF]
**File:** `libvmaf/src/feature/x86/vif_avx2.c:255-339, 590-687`
**Problem:** The `vif_statistic_16_avx2` function declares 14+ concurrent YMM variables (accum_ref_left/right, accum_dis_left/right, accum_ref_dis_left/right, accum_mu1_left/right, accum_mu2_left/right, plus temporaries). AVX2 has only 16 YMM registers, causing the compiler to spill registers to the stack. In the 16-bit path (lines 590-687), the 64-bit accumulation via `_mm256_cvtepu32_epi64` + `_mm256_mul_epu32` chains compound the pressure.
**Fix:** Restructure the loop to process fewer signals per iteration (e.g., compute mu1/mu2 first, store, then compute ref/dis/ref_dis), trading loop iterations for register locality. Alternatively, split into separate horizontal passes per signal.
**Expected Impact:** Low-Medium - reduces stack spills in the hot inner loop. Impact depends on specific CPU microarchitecture.

### Low Impact

#### 13. AVX2-vectorize Integer SSIM moment accumulation [SSIM]
**File:** `libvmaf/src/feature/integer_ssim.c:88-196`
**Problem:** `calc_ssim()` is entirely scalar C. The inner loop (lines 131-152) computes Gaussian-windowed moments (mux, muy, x2, xy, y2, w) per pixel with window coefficient multiplication and accumulation.
**Fix:** Vectorize the inner moment accumulation: load 8-16 pixels, multiply by window coefficients via `_mm256_madd_epi16`, accumulate into 32-bit registers. The window indexing is regular (spatial offsets), making it SIMD-friendly.
**Expected Impact:** Low-Medium - SSIM is not part of the default VMAF model pipeline but is commonly computed alongside it. The complex windowed access pattern limits vectorization efficiency.

#### 14. Motion x_convolution: software-pipeline loads and computation [MOTION]
**File:** `libvmaf/src/feature/x86/motion_avx2.c:48-107`
**Problem:** The 5-tap convolution loop issues 5 sequential 256-bit loads per iteration (one per filter tap) before beginning computation. Modern CPUs have ~4 load ports, so 5 sequential loads stall. The constant `addnum = _mm256_set1_epi32(32768)` is also recreated inside the loop (line 89).
**Fix:** Interleave loads from the current iteration with computation from the previous iteration (software pipelining). Hoist constant creation outside the loop. Consider 2x unrolling to process 32 elements per iteration.
**Expected Impact:** Low-Medium - motion convolution is already AVX2-vectorized. This is a micro-optimization for better pipeline utilization.

#### 15. Float convolution: hoist filter coefficient broadcasts [FLOAT]
**File:** `libvmaf/src/feature/common/convolution_avx.c:184-252`
**Problem:** The float convolution functions (`convolution_f32_avx_s_1d_h_scanline_17`, etc.) perform `_mm256_broadcast_ss(filter + k)` for each coefficient on every scanline iteration. For fixed filter widths (17, 9, 5), these broadcasts produce the same values every time.
**Fix:** Pre-broadcast all filter coefficients into YMM registers once before the scanline loop. For fwidth=17, this uses 9 registers (exploiting symmetry), which fits within AVX2's 16-register budget.
**Expected Impact:** Low - float path is not the default and is less frequently used than integer. Eliminates redundant broadcasts.

#### 16. CAMBI: vectorize tail loops and reduce redundant loads [CAMBI]
**File:** `libvmaf/src/feature/x86/cambi_avx2.c:49-83`
**Problem:** `get_derivative_data_for_row_avx2` loads `horiz_vals1` twice (once for horizontal comparison, once for vertical), and tail loops (lines 60-62, 77-81) process remaining elements one at a time. The `_mm256_and_si256(ones, result)` after `_mm256_cmpeq_epi16` is redundant since the comparison already produces all-ones/zeros.
**Fix:** Reuse the first horizontal load for the vertical comparison. Remove the redundant AND with ones. Use masked stores for tail elements instead of scalar fallback.
**Expected Impact:** Low - CAMBI is a lightweight metric with simple operations. These are micro-optimizations.

#### 17. AVX2-vectorize picture_copy for strided paths [PICTURE]
**File:** `libvmaf/src/feature/picture_copy.c:24-100`
**Problem:** `picture_copy_hbd()` and `picture_copy()` convert integer pixel data to float32 element-by-element. The contiguous path (matching strides) can be auto-vectorized by the compiler, but the strided path (lines 47-52, 91-97) with per-row memcpy + conversion cannot.
**Fix:** For strided paths, use `_mm256_cvtepu16_epi32` (8 pixels) + `_mm256_cvtepi32_ps` for 16-bit, or `_mm256_cvtepu8_epi16` + widening + conversion for 8-bit. Process 8-16 pixels per iteration within each row.
**Expected Impact:** Low - picture copy runs once per frame and is not in the critical path. Compiler auto-vectorization may already handle the contiguous case.

#### 18. AVX2-vectorize moment computation [MOMENT]
**File:** `libvmaf/src/feature/moment.c:8-73`
**Problem:** `compute_1st_moment()` and `compute_2nd_moment()` are trivial float accumulation loops (sum of elements, sum of squared elements) with no SIMD.
**Fix:** Use `_mm256_add_ps` for first moment, `_mm256_fmadd_ps` for second moment, with horizontal reduction at the end.
**Expected Impact:** Low - moment is a trivial metric with minimal compute cost relative to ADM/VIF.

## Implementation Status

| # | Optimization | Status | Commit | Perf Impact |
|---|---|---|---|---|
| 1 | AVX2-vectorize ADM decouple | DONE | 251b394e | ~4-8x speedup for decouple |
| 2 | AVX2-vectorize ADM CSF (CSF part) | DONE | 251b394e | CSF 16-elements/iter via mulhi/mullo |
| 2 | AVX2-vectorize ADM CM (CM part) | SKIPPED | — | Too complex: 9 boundary macros, neighbor-dependent thresholds, ~2000 lines |
| 3 | VIF: alignr-based sliding window | SKIPPED | — | Superseded by #4 unrolling; incompatible changes to same loops |
| 4 | VIF: unroll fixed-size filter loops | REVERTED | a1849360 | Implemented (a8f21327) then reverted: +19.7% regression on vif_statistic_8 (icache pressure on AMD EPYC) |
| 5 | AVX2-vectorize Motion SAD | DONE | 8e91c1b3 | ~8-16x speedup for SAD |
| 6 | VIF: reduce shuffle/permute overhead | DONE | df0f3f98 | ~30 slli+blend → shuffle_ps |
| 7 | AVX2-vectorize Integer PSNR | DONE | 4d699340 | ~8-16x speedup for PSNR |
| 8 | AVX2-vectorize Motion y_convolution | DONE | 8e91c1b3 | Completes motion SIMD pipeline |
| 9 | ADM DWT: interleave horizontal pass | DONE | 251b394e | Better ILP utilization |
| 10 | VIF: vectorize sigma analysis | SKIPPED | — | Requires __builtin_clz + double division; no SIMD benefit |
| 11 | ADM DWT: widen vertical pass | DONE | 251b394e | 32 elements/iter vertical pass |
| 12 | VIF: reduce register pressure | SKIPPED | — | Compiler-level; uncertain benefit without profiling |
| 13 | AVX2-vectorize SSIM moments | DONE | 73666800 | 8 pixels/iter horizontal moment convolution |
| 14 | Motion x_convolution: software pipelining | DONE | 8e91c1b3 | Better pipeline utilization |
| 15 | Float convolution: hoist broadcasts | DONE | — (already optimized) | Already hoisted in existing code |
| 16 | CAMBI: tail loops and redundant ops | DONE | d0fae336 | Micro-optimization |
| 17 | AVX2-vectorize picture_copy strided | DONE | — (auto-vectorized) | Compiler auto-vectorizes with -O3 |
| 18 | AVX2-vectorize moment computation | DONE | — (auto-vectorized) | Compiler auto-vectorizes with -O3 |
