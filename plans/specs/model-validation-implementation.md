# Model Validation Tests -- Implementation Report

## Files Created

### Test Source
- `libvmaf/test/test_model_validation.c` -- All model validation tests in a single file following the existing Minunit pattern.

### Build Integration
- `libvmaf/test/meson.build` -- Modified to register the `test_model_validation` executable and test target with a 300-second timeout.

## Test Coverage Summary

All 53 tests pass. The test suite covers all 12 requirements from the specification.

### R1: Full Model Coverage (9 primary models)

All 9 primary built-in models are tested for loading, feature dependency resolution, and round-trip scoring:

| Model | Load | Features | Round-trip | Identity |
|-------|------|----------|------------|----------|
| vmaf_v0.6.1 | single | yes | yes | yes |
| vmaf_v0.6.1neg | single | yes | yes | yes |
| vmaf_4k_v0.6.1 | single | yes | yes | yes |
| vmaf_4k_v0.6.1neg | single | yes | yes | yes |
| vmaf_b_v0.6.3 | collection | yes | yes | yes |
| vmaf_float_v0.6.1 | single | yes | yes | yes |
| vmaf_float_v0.6.1neg | single | yes | yes | yes |
| vmaf_float_4k_v0.6.1 | single | yes | yes | yes |
| vmaf_float_b_v0.6.3 | collection | yes | yes | yes |

**Implementation note:** `vmaf_b_v0.6.3` and `vmaf_float_b_v0.6.3` are bootstrap models stored as multi-model JSON files. They must be loaded via `vmaf_model_collection_load()`, not `vmaf_model_load()`. The spec originally listed them as single-model loadable, but the actual API requires collection loading. The tests use the correct API.

### R2: Model Collection Coverage (5 collections)

| Collection | Expected sub-models | Tested |
|-----------|-------------------|--------|
| vmaf_b_v0.6.3.json | 20 (key "0" is base) | yes |
| vmaf_rb_v0.6.2/vmaf_rb_v0.6.2.json | 19 (key "0" is base) | yes |
| vmaf_rb_v0.6.3/vmaf_rb_v0.6.3.json | 20 (key "0" is base) | yes |
| vmaf_4k_rb_v0.6.2/vmaf_4k_rb_v0.6.2.json | 19 (key "0" is base) | yes |
| vmaf_float_b_v0.6.3/vmaf_float_b_v0.6.3.json | 20 (key "0" is base) | yes |

**Implementation note:** The spec listed total model counts (e.g., 21 for vmaf_b_v0.6.3), but the collection parser treats key "0" as the base model returned separately. The collection `cnt` field equals total keys minus 1.

### R3: Feature Dependency Completeness

`test_feature_dependency_resolution` and `test_feature_dependency_resolution_float` iterate over all features of every model and verify that `vmaf_get_feature_extractor_by_feature_name()` returns a non-NULL extractor for each.

### R4: Valid Score Range

- All round-trip tests assert `score >= 0.0 && score <= 100.0`.
- All identity tests (ref == dist) assert `score >= 95.0` (or `bagging_score >= 95.0` for collections).

### R5: Error Handling (7 tests)

1. `test_load_nonexistent_path` -- Non-existent file path returns error
2. `test_load_nonexistent_builtin` -- Non-existent version string returns error
3. `test_load_corrupt_json` -- Invalid JSON content returns error
4. `test_load_truncated_json` -- Truncated model file returns error
5. `test_load_empty_file` -- Empty file returns error
6. `test_load_pkl_rejected` -- PKL format is rejected
7. `test_null_arguments` -- NULL arguments handled without crash

### R6: Legacy Model Compatibility (4 models)

`test_load_legacy_json_models` loads all 4 legacy JSON models from `model/other_models/`:
- vmaf_v0.6.0.json
- vmaf_v0.6.1mfz.json
- nflxtrain_norm_type_none.json
- nflx_v1.json

### R7: Configuration Flag Tests

`test_model_flags` tests all 4 flag values:
- `VMAF_MODEL_FLAGS_DEFAULT` -- clip enabled, transform disabled
- `VMAF_MODEL_FLAG_DISABLE_CLIP` -- clip disabled
- `VMAF_MODEL_FLAG_ENABLE_TRANSFORM` -- transform enabled
- `VMAF_MODEL_FLAG_DISABLE_TRANSFORM` -- transform disabled

### R8: Neg Option Validation

`test_neg_model_opts_vmaf_v0_6_1neg` and `test_neg_model_opts_vmaf_4k_v0_6_1neg` verify:
- Feature 0 (adm2): `adm_enhn_gain_limit = "1"`
- Feature 1 (motion2): NULL opts_dict
- Features 2-5 (vif_scale0-3): `vif_enhn_gain_limit = "1"`

### R9: Build Integration

The test compiles and runs as part of `meson test` / `ninja test`. No additional CI workflow changes needed.

### R10: No External Dependencies

All test pictures are synthesized programmatically (64x64, YUV420P, 8-bit). Corrupt/truncated files are written to `/tmp/` and cleaned up after. No network access or external video files required.

### R11: Float Feature Guard

All float-feature model tests are guarded by `#if VMAF_FLOAT_FEATURES`. The test compiles with or without float features.

### R12: Deterministic Reproducibility

All synthetic picture data uses fixed constant values (128 for ref, 96 for dist). No PRNG, wall-clock time, or thread scheduling dependency.

## Additional Test Details

### Built-in vs. File Equivalence (Section 3.1.4)

`test_builtin_vs_file_equivalence` and `test_builtin_vs_file_equivalence_float` compare every single-model built-in with its corresponding JSON file using a full structural comparison (slopes, intercepts, feature names, score clip, normalization, score transform).

### Feature Registration (Section 3.2.2)

`test_feature_registration` and `test_feature_registration_float` verify that `vmaf_use_features_from_model()` succeeds for all single models.

### Expected Feature Names (Section 3.2.3)

`test_expected_feature_names_integer` and `test_expected_feature_names_float` verify exact feature name strings for all models.

### Model Type and Normalization Compatibility (Section 3.4)

- `test_model_type_backward_compat` verifies SVM_NUSVR, BOOTSTRAP_SVM_NUSVR, and RESIDUE_BOOTSTRAP_SVM_NUSVR model types.
- `test_normalization_type_compat` verifies LINEAR_RESCALE and NONE normalization types.

### Custom Model Name (Section 3.6.2)

`test_custom_model_name` verifies that `VmafModelConfig.name` is correctly applied.

### Feature Overload (Section 3.6.3)

`test_feature_overload` verifies that `vmaf_model_feature_overload()` correctly modifies a model's feature options. Note that `vmaf_feature_dictionary_set()` normalizes "1.0" to "1" via `%g` formatting.
