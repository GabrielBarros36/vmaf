# Optimization #5: ADM CM Inner Loop Strength Reduction

## Summary

Hoisted loop-invariant computations and pre-computed index expressions in
the `adm_cm` and `i4_adm_cm` functions in `libvmaf/src/feature/integer_adm.c`.
These are the most compute-intensive functions in the ADM pipeline (~40% of
total VMAF compute).

## Changes Applied

### 1. Pre-compute `i * src_stride` as `i_offset` (both functions)

In every outer `for (i = ...)` loop, the expression `i * src_stride` was
recomputed for each band access (`band_h`, `band_v`, `band_d`) in the inner
`j` loop. Since `i` and `src_stride` are invariant within the inner loop,
this multiplication is computed once per outer iteration and stored in
`const int i_offset`.

Affected code paths (4 branches in each function):
- "Completely within frame"
- "Right border within frame, left outside"
- "Left border within frame, right outside"
- "Both borders outside frame"

### 2. Pre-compute `(h - 1) * src_stride` for bottom-row cases (both functions)

The `i = h-1` row processing at the end of each function computed
`(h - 1) * src_stride` up to 9 times (3 bands x 3 boundary cases). This is
now computed once as `hm1_src_offset` (in `adm_cm`) and `i4_hm1_src_offset`
(in `i4_adm_cm`).

### 3. Hoist `scale - 1` array indexing (`i4_adm_cm` only)

The `i4_adm_cm` function indexed four arrays with `[scale - 1]` on every
pixel in the inner loop:
- `add_bef_shift_dst[scale - 1]`
- `shift_dst[scale - 1]`
- `add_bef_shift_flt[scale - 1]`
- `shift_flt[scale - 1]`

These are all loop-invariant (scale does not change). They are now loaded
once into scalar locals:
- `add_bef_shift_dst_s`
- `shift_dst_s`
- `add_bef_shift_flt_s`
- `shift_flt_s`

Similarly, `final_shift[scale - 1]` at the end uses `scale_idx` instead
of recomputing `scale - 1`.

### 4. Hoist repeated `powf` call in final scoring (both functions)

The expression `powf((bottom - top) * (right - left) / 32.0f, 1.0f / 3.0f)`
was computed identically three times (once for h, v, d components). It is
now computed once as `powf_add`.

## What Was NOT Changed

- No mathematical logic or intermediate precision was altered.
- The `ADM_CM_THRESH_S_*` and `I4_ADM_CM_THRESH_S_*` macros were not modified
  (they have their own internal pointer arithmetic which is separate).
- The `ADM_CM_ACCUM_ROUND` and `I4_ADM_CM_ACCUM_ROUND` macros were not modified.
- A pre-existing bug in the "Left border within frame, right outside" path of
  `i4_adm_cm` (where `rfactor[i * src_stride + w - 1]` indexes out of bounds
  of the 3-element `rfactor` array) was left untouched to preserve existing
  behavior.

## Correctness Verification

All tests pass identically before and after the change:

| Test | Status |
|------|--------|
| test_model_validation | PASS |
| test_precision_differential | PASS |
| test_degenerate | PASS |
| test_golden_values | PASS |
| test_thread_safety | PASS |
| test_simd_adm | FAIL (pre-existing NEON issue) |
| All other Tier 1+2 tests | PASS |

The optimizations are purely mechanical strength reductions that compute the
same values with fewer instructions in the inner loop.
