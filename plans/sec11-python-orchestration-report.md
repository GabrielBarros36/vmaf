# Python Orchestration (Additional) Report — Section 11 Findings

## Overview

This document describes the Python orchestration optimizations implemented for
the VMAF Python wrapper library, addressing findings 11a, 11c, and 11d from
[optimization-opportunities.md](optimization-opportunities.md).

Finding 11b (JSON serialization in result_store.py) was **deferred** because
changing the serialization format would break backward compatibility with existing
cached results.

## Changes Made

### 1. Pre-compile Regex Patterns — Finding 11a (Low impact)

**Problem:** In `FeatureExtractor._get_feature_scores()`, regex patterns were
compiled inside a nested loop: for each line in the log file, for each atom
feature, `re.match()` compiles and applies the pattern. For N frames and M
features, this is O(N×M) regex compilations.

**Solution:** Pre-compile all regex patterns before the line-processing loop:
```python
compiled_patterns = [(atom_feature, re.compile(pattern)) for ...]
for line in log_file:
    for atom_feature, compiled_re in compiled_patterns:
        match = compiled_re.match(line)
```

This reduces regex compilation from O(N×M) to O(M) — one compilation per atom
feature, reused across all lines.

**Files modified:**
- `python/vmaf/core/feature_extractor.py` — pre-compile regex in
  `_get_feature_scores()`

---

### 2. Cache Loaded Model — Finding 11c (Low impact)

**Problem:** `VmafQualityRunnerModelMixin._load_model(asset)` was called twice
per asset — once in `_get_vmaf_feature_assembler_instance()` to set up the feature
assembler, and once in `_create_prediction_result_dict()` to create predictions.
Model loading involves reading and parsing the JSON model file from disk each time.

**Solution:** Cache the loaded model and its filepath on the runner instance:
```python
if hasattr(self, '_cached_model') and self._cached_model_filepath == model_filepath:
    return self._cached_model
model = ...  # load from disk
self._cached_model = model
self._cached_model_filepath = model_filepath
return model
```

The cache is keyed by filepath to handle the (unlikely) case where different
assets use different models.

**Files modified:**
- `python/vmaf/core/quality_runner.py` — added `_cached_model` /
  `_cached_model_filepath` to `_load_model()`

---

### 3. Pre-build Feature Key Mapping — Finding 11d (Low impact)

**Problem:** `FeatureAssembler._create_feature_result_dicts()` uses
`try/except KeyError` with a regex-based wildcard fallback for feature score
lookups. For large feature sets with many mismatched keys, every miss triggers
Python exception handling overhead.

**Solution:** Two optimizations:
1. Replaced `try/except KeyError` with `dict.__contains__` check (`if key in dict`)
   to avoid exception overhead on misses
2. Added a `wildcard_cache` dict that caches wildcard-resolved key mappings across
   results. Once a wildcard pattern is resolved for a given result dict, the mapping
   is reused for subsequent lookups with the same pattern.

**Files modified:**
- `python/vmaf/core/feature_assembler.py` — replaced exception-based control flow
  with membership check and wildcard cache

---

## Deferred

### 11b. JSON Serialization in result_store.py

**Reason:** The `save_result` method serializes result dicts using Python's `str()`
(repr format), and `load_result` deserializes with `ast.literal_eval()`. The dict
values can contain:
- NumPy scalar types (`numpy.float64`, `numpy.int64`)
- Tuple representations from `Asset.__str__()`
- Other non-JSON-serializable Python types

Switching to `json.dump()`/`json.load()` would require:
- Custom JSON encoders for all non-standard types
- A migration path for existing cached results
- Backward-compatible reading (try JSON, fall back to literal_eval)

The complexity and risk outweigh the benefit for a Low-priority item.

---

## Correctness Verification

### Python score regression tests
**67/67 tests pass** when Section 11 changes are applied in isolation. The tests
exercise all three modified code paths:

- Regex pre-compilation: tested by every runner test that parses log output
- Model caching: tested by `test_run_vmaf_runner` and all VMAF model tests
- Key mapping: tested by `test_run_vmaf_runner` (feature assembly step)

**All scores are bit-identical** — these are pure performance optimizations that
do not change any computation logic.

---

## Performance Results

The Python optimizations primarily affect the Python orchestration layer, not the
C library benchmark. The benchmark measures C library performance via the `vmaf`
CLI tool, which does not use the Python layer.

**Where these optimizations matter:**
- `VmafQualityRunner` and `VmafLegacyQualityRunner` in the Python test suite
- Batch processing workflows that run many assets through the Python API
- Model loading time is halved (1 load vs 2 per asset)
- Regex compilation is reduced from O(N×M) to O(M) per feature extraction

---

## Summary

| Finding | What | Status | Score impact |
|---------|------|--------|-------------|
| 11a | Pre-compile regex patterns in log parsing | Done | None (bit-identical) |
| 11b | JSON serialization in result_store | Deferred — backward compatibility risk | — |
| 11c | Cache loaded model (1 load vs 2 per asset) | Done | None (bit-identical) |
| 11d | Pre-build feature key mapping with wildcard cache | Done | None (bit-identical) |
