# Optimization #12: Stack Buffers for Feature Name Construction

## Summary

Replace heap allocations (`malloc`/`free`) with stack buffers for short feature
name strings in the per-frame scoring hot path, reducing malloc pressure during
VMAF prediction.

## Problem

In `vmaf_predict_score_at_index()`, the function `vmaf_feature_name_from_options()`
is called once per feature per frame. It heap-allocates the resulting feature name
string (via `malloc` inside `vmaf_feature_name_from_opts_dict()`), which the caller
then immediately uses for a score lookup and frees. Feature names are short strings
(typically under 128 characters), so the repeated malloc/free overhead is wasteful
on a hot path.

## Solution

Added a new function `vmaf_feature_name_from_options_buf()` that writes the
constructed feature name into a caller-provided stack buffer instead of
heap-allocating. This function:

- Takes the same parameters as `vmaf_feature_name_from_options()` plus a buffer
  pointer and size.
- Returns an `int` error code (0 on success) rather than a pointer.
- Writes directly into the caller's buffer, eliminating the malloc/free pair.

Internally, the existing `vmaf_feature_name_from_opts_dict()` was refactored into
a buffer-writing core (`vmaf_feature_name_from_opts_dict_buf()`) that both the
old heap-allocating API and the new buffer API share.

## Changes

### `libvmaf/src/feature/feature_name.h`
- Added `#include <stddef.h>` for `size_t`.
- Declared new function `vmaf_feature_name_from_options_buf()`.

### `libvmaf/src/feature/feature_name.c`
- Extracted `vmaf_feature_name_from_opts_dict_buf()` as the core implementation
  that writes to a caller-provided buffer.
- Rewrote `vmaf_feature_name_from_opts_dict()` as a thin wrapper that calls the
  buffer variant and then copies into a heap allocation (preserving the existing
  API for non-hot-path callers).
- Added `vmaf_feature_name_from_options_buf()` public function that builds the
  options dictionary and delegates to the buffer-based core.

### `libvmaf/src/predict.c`
- In `vmaf_predict_score_at_index()`, replaced the `vmaf_feature_name_from_options()`
  call (which returned a heap-allocated string) with
  `vmaf_feature_name_from_options_buf()` writing into a local `char feature_name[256]`
  stack buffer.
- Removed the corresponding `free(feature_name)` calls.

## What was NOT changed

- **Init-time allocations**: `feature_extractor_vector_append()` in `fex_ctx_vector.c`
  still uses the heap-allocating `vmaf_feature_name_from_options()` since it runs
  only during feature extractor registration, not per-frame.
- **`feature_name_dict_from_provided_features()`**: Still uses the heap-allocating
  variant since the result is stored in a dictionary.
- **Ownership semantics**: The existing `vmaf_feature_name_from_options()` API is
  unchanged; callers that need a heap-allocated string still get one.
- **`feature_collector.c` and `feature_extractor.c`**: No feature-name heap
  allocations on the per-frame hot path were found in these files.

## Verification

- All 17 meson unit tests pass.
- End-to-end VMAF scoring produces identical results (vmaf mean = 76.668905 on the
  standard test pair).
