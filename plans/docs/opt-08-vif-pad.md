# Optimization #8: VIF Reduce Redundant Buffer Copies in Pad Operations

## Problem

The `pad_top_and_bottom` function in `integer_vif.c` used `buf.stride` as the
memcpy length when mirroring rows for vertical padding. Since `buf.stride` is
the aligned stride (`ALIGN_CEIL(w << hbd)`, rounded up to 32-byte boundary),
this copies up to 31 extra bytes per row beyond the actual pixel data width.
With `fwidth_half` iterations copying 4 rows each (ref top/bottom, dis
top/bottom), the wasted bandwidth adds up across all 4 VIF scales.

## Fix

Added a `data_width` parameter to `pad_top_and_bottom` specifying the actual
number of bytes of pixel data per row. The memcpy calls now use `data_width`
instead of `buf.stride`. Row address calculations still use `buf.stride` for
correct pointer arithmetic.

Call sites pass the appropriate width:
- Scale 0 (initial call in `extract`): `w << (bpc > 8)` -- raw pixel width in bytes
- Subsequent scales (via `decimate_and_pad`): `(w / 2) * sizeof(uint16_t)` -- decimated uint16_t data width

## Files Changed

- `libvmaf/src/feature/integer_vif.c`: Modified `pad_top_and_bottom` signature
  and both call sites.

## Impact

Reduces memory bandwidth in the C reference code path for VIF padding. The
savings are proportional to the stride alignment gap (up to 31 bytes per row
per memcpy, across all padding rows and scales). This only affects the C
fallback; SIMD variants (AVX2, AVX-512, NEON) have their own copies of
`pad_top_and_bottom` which are unchanged.
