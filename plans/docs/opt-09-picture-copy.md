# Optimization #9: Picture Copy Optimization

## Summary

Optimized `picture_copy.c` to flatten the 2D row-by-row pixel conversion
loop into a single contiguous pass when source and destination planes have
no padding (stride equals row width).

## Problem

The `picture_copy` and `picture_copy_hbd` functions convert integer pixel
data (uint8 or uint16) to float for use by the float-path feature
extractors (float_adm, float_vif, float_ssim, float_psnr, float_motion,
float_ansnr, float_moment, float_ms_ssim). The original implementation
always iterated row by row, advancing source and destination pointers by
their respective strides after each row. This per-row pointer arithmetic
adds overhead and can inhibit compiler auto-vectorization of the
conversion loop.

## Solution

Added a fast path that checks whether both the source and destination
planes are contiguous in memory:

- **8-bit case:** `src->stride[0] == w * sizeof(uint8_t)` and
  `dst_stride == w * sizeof(float)`
- **HBD case:** `src->stride[0] == w * sizeof(uint16_t)` and
  `dst_stride == w * sizeof(float)`

When both conditions hold, the entire plane is processed as a single
flat array of `w * h` elements, eliminating per-row pointer arithmetic.
The compiler can then vectorize the single loop more aggressively
(e.g., unroll, use NEON/SVE auto-vectorization on aarch64).

When strides differ (due to padding or alignment requirements), the
original per-row fallback path is used unchanged.

## Design Rationale

- No NEON intrinsics were added. Modern aarch64 libc `memcpy` is already
  NEON-optimized, and the compiler's auto-vectorizer handles the int-to-float
  conversion loop well when presented as a single contiguous pass.
- The check is a simple comparison of two integers at the start of each
  function call, adding negligible overhead to the non-contiguous path.
- The per-row fallback is preserved exactly as before for correctness
  when strides include padding.

## Files Changed

- `libvmaf/src/feature/picture_copy.c` -- Added contiguous-plane fast
  path to both `picture_copy` (8-bit) and `picture_copy_hbd` (10/12/16-bit).

## Testing

All standard-tier tests pass. The only failure is `test_simd_adm`, which
is a known pre-existing issue unrelated to this change.
