# Optimization #10: Precompute ADM CSF Coefficients

## Summary

Moved CSF (Contrast Sensitivity Function) coefficient computation from the
per-frame ADM path to the feature extractor's `init()` function. The CSF
coefficients depend only on `adm_norm_view_dist` and `adm_ref_display_height`,
both of which are constant across all frames, so computing them once at
initialization eliminates redundant floating-point work on every frame.

## Problem

The `dwt_quant_step()` function computes CSF coefficients using `log10()`,
`pow()`, and other transcendental math functions. It was called in six
different per-frame functions:

| Function           | Scale | Calls per frame |
|--------------------|-------|-----------------|
| `adm_csf`          | 0     | 2 (theta=1,2)   |
| `adm_csf_den_scale`| 0     | 2               |
| `adm_cm`           | 0     | 2               |
| `i4_adm_csf`       | 1-3   | 2 x 3 = 6       |
| `adm_csf_den_s123` | 1-3   | 2 x 3 = 6       |
| `i4_adm_cm`        | 1-3   | 2 x 3 = 6       |

Total: **24 calls to `dwt_quant_step()` per frame**, each involving `log10()`
and `pow()` -- but they only produce 8 distinct values (4 scales x 2 theta
values), all of which are constant across frames.

## Solution

Added a `float csf_factors[4][2]` array to the `AdmState` struct:
- `csf_factors[scale][0]` = `dwt_quant_step(scale, theta=1)`
- `csf_factors[scale][1]` = `dwt_quant_step(scale, theta=2)`

The array is populated once in `init()` after the option values are resolved.
Each per-frame function now reads from this precomputed array instead of
calling `dwt_quant_step()`.

## Files Changed

- `libvmaf/src/feature/integer_adm.c`
  - `AdmState` struct: added `csf_factors[4][2]`
  - `init()`: added precomputation loop
  - `adm_csf()`: replaced `dwt_quant_step` calls with `csf_factors` lookups
  - `i4_adm_csf()`: same
  - `adm_csf_den_scale()`: same
  - `adm_csf_den_s123()`: same
  - `adm_cm()`: same
  - `i4_adm_cm()`: same
  - `integer_compute_adm()`: updated signature and call sites
  - `extract()`: passes `s->csf_factors` to `integer_compute_adm()`

## Correctness

This is a pure code-motion optimization. The precomputed values are
identical to what `dwt_quant_step()` would compute per-frame because:

1. The function inputs (`dwt_7_9_YCbCr_threshold[0]`, `adm_norm_view_dist`,
   `adm_ref_display_height`) are all constant across frames.
2. The `dwt_quant_step()` function is deterministic (no state, no
   floating-point rounding mode changes).
3. The computed `float` values are stored and reused exactly, with no
   intermediate precision changes.

All downstream computations (`rfactor`, `i_rfactor`, etc.) are unchanged --
they still derive from `factor1` and `factor2` in the same way; only the
source of those two values changed from a function call to an array lookup.

## Verification

- All 17 meson unit tests pass.
- The `vmaf` CLI tool produces identical ADM and VMAF scores for the standard
  576x324 test pair (e.g., frame 0: `integer_adm2=0.962084`,
  `vmaf=83.856284`).
- SIMD paths (AVX2, NEON) are unaffected: they do not call `dwt_quant_step()`.
- CUDA path is unaffected: it uses its own `integer_compute_adm_cuda()`.
