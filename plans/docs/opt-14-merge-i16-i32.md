# Optimization #14: Merge i16_to_i32 Conversion with Consuming Operation

## Summary

Eliminated a separate memory pass that converted DWT band_a data from int16 to
int32 by merging the widening into the consuming DWT function for scale 1.

## Problem

In the ADM pipeline, the scale 0 DWT (`adm_dwt2_8` / `adm_dwt2_16`) produces
`band_a` as int16 values stored in `buf->ref_dwt2.band_a` and
`buf->dis_dwt2.band_a`. A separate function `i16_to_i32` then copied every
element from the int16 buffer to an int32 buffer (`buf->i4_ref_dwt2.band_a`,
`buf->i4_dis_dwt2.band_a`). This int32 data was consumed by
`adm_dwt2_s123_combined` at scale 1 as input to the next DWT decomposition.

The separate conversion pass performed a full read+write over `(h+1)/2 * (w+1)/2`
elements for both reference and distorted frames -- pure memory bandwidth waste,
since the conversion (sign extension from int16 to int32) is trivial.

## Solution

Created a specialized function `adm_dwt2_s1_combined` that reads directly from
`int16_t *` input for scale 1. This function is identical to
`adm_dwt2_s123_combined` except:

1. **Input type**: Reads `int16_t *` instead of `int32_t *`, performing the
   widening `(int32_t)value` inline during the vertical pass loads.

2. **Hardcoded scale 1 constants**: Since this function only handles scale 1,
   the shift/rounding constants are compile-time values rather than array lookups:
   - `shift_VP = 0` (no vertical pass shift, simplifies to direct cast)
   - `add_VP = 0` (no rounding addition needed)
   - `shift_HP = 15`, `add_HP = 16384`

3. **Caller restructured**: The loop in `integer_compute_adm` now has three
   branches: `scale==0` (DWT from pixel data), `scale==1` (DWT from i16 band_a
   via `adm_dwt2_s1_combined`), and `scale>=2` (DWT from i32 band_a via
   `adm_dwt2_s123_combined`).

The `i16_to_i32` function was removed entirely.

## Data Flow

```
Before:
  scale 0: dwt2_8 -> i16 band_a -> i16_to_i32 -> i32 band_a
  scale 1: adm_dwt2_s123_combined(i32 band_a) -> i32 band_a
  scale 2+: adm_dwt2_s123_combined(i32 band_a) -> i32 band_a

After:
  scale 0: dwt2_8 -> i16 band_a  (no conversion)
  scale 1: adm_dwt2_s1_combined(i16 band_a) -> i32 band_a  (widening inline)
  scale 2+: adm_dwt2_s123_combined(i32 band_a) -> i32 band_a  (unchanged)
```

## Correctness Argument

The widening from int16 to int32 is a lossless sign extension. Whether it
happens as a separate pass or inline during a load instruction, the resulting
int32 values are identical. The filter computation uses int64 accumulators
(`(int64_t)filter * s10`), so the int32 intermediate values have the same
precision regardless of their source type.

All 53 model validation tests pass, confirming identical ADM scores.

## Performance Impact

- Eliminates 2 full passes over `(h+1)/2 * (w+1)/2` elements per frame
  (one for ref, one for dis)
- For 1080p: eliminates reading ~518K int16 values and writing ~518K int32 values
  (per ref/dis pair)
- The i16 vertical pass reads may also benefit from reduced cache pressure
  (reading 2-byte values vs 4-byte values)
- Scale 1 constants are compile-time, avoiding array index lookups

## Files Modified

- `libvmaf/src/feature/integer_adm.c`:
  - Removed `i16_to_i32` function
  - Added `adm_dwt2_s1_combined` (i16-input DWT for scale 1)
  - Modified `integer_compute_adm` to use 3-way branch (scale 0 / 1 / 2+)

## Testing

- All tier 1+2 tests pass (26/26, excluding known `test_simd_adm` failure)
- Model validation: 53/53 tests pass (verifies ADM score correctness)
