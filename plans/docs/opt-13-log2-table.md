# Optimization #13: Compile-time log2 Lookup Table Generation

## Summary

Replaced the runtime-generated log2 lookup table in `integer_vif.c` with a
precomputed static const table, eliminating redundant `log2f` + `round`
computation on every VIF feature extractor initialization.

## Background

The integer VIF implementation uses a 32769-entry lookup table mapping indices
`[32767, 65535]` to `(uint16_t)round(log2f((float)i) * 2048)`. This table was
previously regenerated at runtime in `log_generate()` on every call to `init()`,
which occurs each time a VIF feature extractor is created.

Since the values depend only on compile-time constants (the formula is purely a
function of the index), this work is redundant.

## Changes

### New file: `libvmaf/src/feature/vif_log2_table.h`

A generated header containing the precomputed 32769-entry `uint16_t` table.
Values were generated using the exact same formula as the original `log_generate`
function: `(uint16_t)round(log2f((float)i) * 2048)` for `i` in `[32767, 65535]`.

Bit-exact equivalence was verified by comparing every entry against the
runtime-generated table.

### Modified: `libvmaf/src/feature/integer_vif.c`

- Added `#include "vif_log2_table.h"`
- Changed `log_generate()` from computing `log2f`+`round` in a loop to a single
  `memcpy` from the static const table into the per-instance `log2_table` array

## Design Decisions

**Why memcpy rather than a pointer?** The `VifPublicState` struct embeds the
`log2_table` as `uint16_t log2_table[65537]` (an inline array). Changing this to
a pointer would alter the struct layout, requiring changes in all SIMD
implementations (AVX2, AVX-512, NEON) and CUDA code that reference `s->log2_table`.
Using `memcpy` preserves full ABI compatibility with zero changes to any other file.

**Why not generate-once with a flag?** The table is per-instance (embedded in each
`VifPublicState`), so a static flag would not be correct in a multi-threaded
context where multiple extractors are created. The `memcpy` approach is simpler,
correct, and fast.

## Verification

- All 17 meson unit tests pass
- VIF scores on test videos (`src01_hrc00/hrc01_576x324.yuv`) are bit-for-bit
  identical to the original implementation
- Table values verified entry-by-entry against runtime-generated values

## Performance Impact

Replaces 32769 calls to `log2f` + `roundf` (floating-point math) with a single
64KB `memcpy` on each extractor init. The improvement is modest in absolute terms
(init is not on the hot path) but eliminates unnecessary work and reduces
dependency on floating-point math library behavior at init time.
