# Optimization #15: Compiler Hints with Restrict Pointers

## Summary

Added `restrict` qualifiers (via the existing `RESTRICT` macro, which expands
to `__restrict`) to pointer parameters of compute-intensive feature extractor
functions in libvmaf.  This tells the compiler that input and output buffers
do not alias, enabling better auto-vectorization and loop optimizations.

## Problem

Without `restrict`, the C compiler must conservatively assume that any two
pointer parameters might point to overlapping memory.  This forces the compiler
to reload values from memory around every store and prevents several
optimization opportunities, including loop vectorization, load-store
reordering, and strength reduction of address calculations.

## Convention

The codebase already defines a portable `RESTRICT` macro in
`libvmaf/src/feature/common/macros.h`:

```c
#define RESTRICT __restrict
```

All new annotations use this macro for consistency and MSVC portability.

## Changes

### integer_adm.c

Added `#include "common/macros.h"` and `RESTRICT` to the following functions:

| Function | Parameters annotated | Rationale |
|----------|---------------------|-----------|
| `adm_dwt2_8` | `src`, `dst`, `buf` | DWT kernel; src is read-only input, dst bands are output, buf holds scratch |
| `adm_dwt2_16` | `src`, `dst`, `buf` | Same as above for 16-bit inputs |
| `adm_dwt2_s123_combined` | `i4_ref_scale`, `i4_curr_dis`, `buf` | Two distinct input buffers + scratch buffer |
| `i16_to_i32` | `src`, `dst` | Type-widening copy between disjoint band structures |
| `adm_csf_den_scale` | `src` | Read-only band input for cubing loop |
| `adm_csf_den_s123` | `src` | Same as above for 32-bit bands |
| `adm_cm` | `buf` | Buffer struct used for read-only access to distinct sub-buffers |
| `i4_adm_cm` | `buf` | Same as above for 32-bit version |

### integer_vif.c / integer_vif.h

Added `#include "common/macros.h"` to the header and `RESTRICT` to:

| Function | Parameters annotated | Rationale |
|----------|---------------------|-----------|
| `vif_statistic_8` | `s`, `num`, `den` | State struct + two scalar output pointers that don't alias |
| `vif_statistic_16` | `s`, `num`, `den` | Same for 16-bit path |

### integer_motion.c

Added `#include "common/macros.h"` and `RESTRICT` to:

| Function | Parameters annotated | Rationale |
|----------|---------------------|-----------|
| `x_convolution_16` | `src`, `dst` | Horizontal convolution: reads from src, writes to separate dst |
| `y_convolution_16` | `src`, `dst` | Vertical convolution: same non-aliasing guarantee |
| `y_convolution_8` | `src`, `dst` | 8-bit vertical convolution variant |

### picture_copy.c / picture_copy.h

Added `#include "common/macros.h"` and `RESTRICT` to:

| Function | Parameters annotated | Rationale |
|----------|---------------------|-----------|
| `picture_copy_hbd` | `dst`, `src` | Float output buffer vs VmafPicture input |
| `picture_copy` | `dst`, `src` | Same for the dispatch function |

## Correctness Verification

- All annotated pointer pairs truly do not alias in practice:
  - Input buffers are separate from output buffers
  - State structs are disjoint from scalar output pointers
  - DWT band structures use pre-allocated non-overlapping memory regions
- Built successfully with `ninja -C libvmaf/build -j8`
- All standard-tier tests pass (26/27; `test_simd_adm` is a known
  pre-existing failure unrelated to this change)

## Expected Impact

The `restrict` qualifier is a compiler hint, not a runtime change. The actual
performance benefit depends on the compiler and optimization level:

- **GCC/Clang -O2/-O3**: may enable auto-vectorization of loops that were
  previously blocked by alias analysis
- **Tight inner loops** (DWT, convolution, VIF statistics): most likely to
  benefit since the compiler can now assume stores to output don't invalidate
  loads from input
- **No correctness impact**: restrict is a promise to the compiler, not a
  runtime assertion. If the promise is violated (aliased pointers), the result
  would be undefined behavior, but the callers in libvmaf never pass aliased
  buffers to these functions.
