# Optimization #11: Avoid Redundant Bootstrap SVM Prediction

## Problem

In `vmaf_bootstrap_predict_score_at_index()` (libvmaf/src/predict.c), each
bootstrap model required **two full calls** to `vmaf_predict_score_at_index()`:

1. First call with `VMAF_MODEL_FLAG_DISABLE_CLIP | VMAF_MODEL_FLAG_DISABLE_TRANSFORM`
   to obtain the raw (untransformed, unclipped) score for statistical
   aggregation (mean, stddev, percentiles).
2. Second call with no flags to obtain the transformed/clipped score for
   writing to the feature collector.

Both calls performed identical work through the expensive stages: feature
gathering (name resolution, dictionary copy, context creation), normalization,
and SVM prediction. Only the final transform/clip step differed.

For a typical bootstrap model collection with ~20 models, this meant ~20
redundant SVM predictions per frame.

## Solution

Compute the SVM prediction once per model (via the first call with
transform/clip disabled), then derive the clipped score by applying
`transform()` and `clip()` to a copy of the raw prediction. The clipped score
is written to the feature collector via `vmaf_feature_collector_append()`.

### Before

```c
for (unsigned i = 0; i < model_collection->cnt; i++) {
    // Call 1: raw score (no transform/clip)
    err = vmaf_predict_score_at_index(model, fc, index, &scores[i],
                                      false, false,
                                      DISABLE_CLIP | DISABLE_TRANSFORM);
    // Call 2: clipped score (full pipeline again)
    double score;
    err = vmaf_predict_score_at_index(model, fc, index, &score,
                                      true, false, 0);
}
```

### After

```c
for (unsigned i = 0; i < model_collection->cnt; i++) {
    // Single SVM prediction (no transform/clip)
    err = vmaf_predict_score_at_index(model, fc, index, &scores[i],
                                      false, false,
                                      DISABLE_CLIP | DISABLE_TRANSFORM);
    // Derive clipped score from raw prediction
    double clipped_score = scores[i];
    transform(model, &clipped_score, 0);
    clip(model, &clipped_score, 0);
    vmaf_feature_collector_append(fc, model->name, clipped_score, index);
}
```

## Correctness Argument

The `vmaf_predict_score_at_index` pipeline is:

1. Feature gathering + normalization
2. `svm_predict()` + `denormalize()`
3. `transform()` (skipped if `DISABLE_TRANSFORM` flag)
4. `clip()` (skipped if `DISABLE_CLIP` flag)

With `DISABLE_CLIP | DISABLE_TRANSFORM`, the first call returns the result
after step 2. Applying `transform()` and `clip()` with flags=0 to that result
produces the same value as a full call with no flags. The operations are
deterministic and have no side effects on the model or feature collector
(transform and clip only modify their `double *` argument).

## Impact

- Eliminates N redundant SVM predictions per frame (where N = number of
  bootstrap models, typically ~20).
- Each avoided call saves: feature name resolution, dictionary copy/free,
  extractor context create/destroy, feature score lookups, normalization,
  SVM kernel evaluation, denormalization, and memory allocation/free of the
  svm_node array.
- No change to output scores or feature collector contents.

## Files Modified

- `libvmaf/src/predict.c`: `vmaf_bootstrap_predict_score_at_index()`

## Testing

All existing tests pass. The prediction path is exercised by `test_predict`
and `test_model` which verify score correctness for both single models and
model collections.
