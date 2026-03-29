# Model Validation Tests -- Specification

## 1. Objective

Verify that every model file shipped with libvmaf loads successfully, resolves all required feature extractors, produces valid VMAF scores through a complete scoring pipeline, and handles error conditions gracefully. This test suite must catch model file corruption, feature dependency mismatches, and version compatibility regressions before they reach production.

---

## 2. Scope -- Shipped Model Inventory

### 2.1 Primary JSON Models (Built-in)

These models are compiled into the libvmaf binary via `vmaf_model_load()` and are also available as standalone JSON files for `vmaf_model_load_from_path()`.

| # | Version String | JSON File | Model Type | Feature Set | Score Clip |
|---|---------------|-----------|------------|-------------|------------|
| 1 | `vmaf_v0.6.1` | `vmaf_v0.6.1.json` | LIBSVMNUSVR | integer (adm2, motion2, vif_scale0-3) | [0, 100] |
| 2 | `vmaf_v0.6.1neg` | `vmaf_v0.6.1neg.json` | LIBSVMNUSVR | integer (adm2, motion2, vif_scale0-3) + neg opts | [0, 100] |
| 3 | `vmaf_4k_v0.6.1` | `vmaf_4k_v0.6.1.json` | LIBSVMNUSVR | integer (adm2, motion2, vif_scale0-3) | [0, 100] |
| 4 | `vmaf_4k_v0.6.1neg` | `vmaf_4k_v0.6.1neg.json` | LIBSVMNUSVR | integer (adm2, motion2, vif_scale0-3) + neg opts | [0, 100] |
| 5 | `vmaf_b_v0.6.3` | `vmaf_b_v0.6.3.json` | BOOTSTRAP_LIBSVMNUSVR | integer (adm2, motion2, vif_scale0-3) | [0, 100] |
| 6 | `vmaf_float_v0.6.1` | `vmaf_float_v0.6.1.json` | LIBSVMNUSVR | float (adm2, motion2, vif_scale0-3) | [0, 100] |
| 7 | `vmaf_float_v0.6.1neg` | `vmaf_float_v0.6.1neg.json` | LIBSVMNUSVR | float (adm2, motion2, vif_scale0-3) + neg opts | [0, 100] |
| 8 | `vmaf_float_4k_v0.6.1` | `vmaf_float_4k_v0.6.1.json` | LIBSVMNUSVR | float (adm2, motion2, vif_scale0-3) | [0, 100] |
| 9 | `vmaf_float_b_v0.6.3` | `vmaf_float_b_v0.6.3.json` | BOOTSTRAP_LIBSVMNUSVR | float (adm2, motion2, vif_scale0-3) | [0, 100] |

**Note:** Models 6-9 require `VMAF_FLOAT_FEATURES` to be enabled at compile time. Tests for these models must be guarded by `#if VMAF_FLOAT_FEATURES`.

### 2.2 Model Collection Files (Bootstrap / Residue-Bootstrap)

These models contain multiple sub-models (bagging) and are loaded via `vmaf_model_collection_load_from_path()`.

| # | Directory | JSON File | Model Type | Sub-models |
|---|-----------|-----------|------------|------------|
| 10 | `vmaf_b_v0.6.3.json` | (top-level) | BOOTSTRAP_LIBSVMNUSVR | 21 |
| 11 | `vmaf_rb_v0.6.2/` | `vmaf_rb_v0.6.2.json` | RESIDUE_BOOTSTRAP_LIBSVMNUSVR | 20 |
| 12 | `vmaf_rb_v0.6.3/` | `vmaf_rb_v0.6.3.json` | RESIDUE_BOOTSTRAP_LIBSVMNUSVR | 21 |
| 13 | `vmaf_4k_rb_v0.6.2/` | `vmaf_4k_rb_v0.6.2.json` | RESIDUE_BOOTSTRAP_LIBSVMNUSVR | 20 |
| 14 | `vmaf_float_b_v0.6.3/` | `vmaf_float_b_v0.6.3.json` | BOOTSTRAP_LIBSVMNUSVR | 21 |

### 2.3 Legacy / Other Models (model/other_models/)

These are older or experimental model files shipped for backward compatibility. Only the JSON-format files are loadable by current libvmaf; `.pkl` and `.pkl.model` files are deprecated.

| # | File | Format | Notes |
|---|------|--------|-------|
| 15 | `vmaf_v0.6.0.json` | JSON | Older integer model |
| 16 | `vmaf_v0.6.1mfz.json` | JSON | Integer model variant (motion-free zero) |
| 17 | `nflxtrain_norm_type_none.json` | JSON | Experimental normalization |
| 18 | `nflx_v1.json` | JSON | First-generation model |
| 19 | `model_V8a.model` | pkl.model | Legacy (should fail to load with JSON loader) |
| 20-37 | `*.pkl` / `*.pkl.model` | pkl | Legacy (should fail to load with JSON loader) |

### 2.4 Feature Dependency Map

All shipped models depend on a subset of these feature extractors:

| Feature Name in Model | Feature Extractor | Extractor Symbol |
|-----------------------|-------------------|------------------|
| `VMAF_integer_feature_adm2_score` | `integer_adm` | `vmaf_fex_integer_adm` |
| `VMAF_integer_feature_motion2_score` | `integer_motion` | `vmaf_fex_integer_motion` |
| `VMAF_integer_feature_vif_scale0_score` | `integer_vif` | `vmaf_fex_integer_vif` |
| `VMAF_integer_feature_vif_scale1_score` | `integer_vif` | `vmaf_fex_integer_vif` |
| `VMAF_integer_feature_vif_scale2_score` | `integer_vif` | `vmaf_fex_integer_vif` |
| `VMAF_integer_feature_vif_scale3_score` | `integer_vif` | `vmaf_fex_integer_vif` |
| `VMAF_feature_adm2_score` | `float_adm` | `vmaf_fex_float_adm` |
| `VMAF_feature_motion2_score` | `float_motion` | `vmaf_fex_float_motion` |
| `VMAF_feature_vif_scale0_score` | `float_vif` | `vmaf_fex_float_vif` |
| `VMAF_feature_vif_scale1_score` | `float_vif` | `vmaf_fex_float_vif` |
| `VMAF_feature_vif_scale2_score` | `float_vif` | `vmaf_fex_float_vif` |
| `VMAF_feature_vif_scale3_score` | `float_vif` | `vmaf_fex_float_vif` |

---

## 3. Test Categories

### 3.1 Model Loading Tests

Verify that every shipped model file can be loaded and destroyed without error.

#### 3.1.1 Built-in Model Loading (`vmaf_model_load`)

For each built-in version string listed in Section 2.1, call `vmaf_model_load()` and assert:

1. Return value is 0 (success).
2. The returned `VmafModel *` is non-NULL.
3. `model->n_features` matches the expected feature count (6 for all current models).
4. `model->score_clip.enabled` is true.
5. `model->score_clip.min` is 0.0 and `model->score_clip.max` is 100.0.
6. `vmaf_model_destroy()` completes without error (no crash, no leak).

```c
static const char *builtin_versions[] = {
    "vmaf_v0.6.1",
    "vmaf_v0.6.1neg",
    "vmaf_4k_v0.6.1",
    "vmaf_4k_v0.6.1neg",
    "vmaf_b_v0.6.3",
#if VMAF_FLOAT_FEATURES
    "vmaf_float_v0.6.1",
    "vmaf_float_v0.6.1neg",
    "vmaf_float_4k_v0.6.1",
    "vmaf_float_b_v0.6.3",
#endif
};

static char *test_load_all_builtin_models(void) {
    for (unsigned i = 0; i < sizeof(builtin_versions)/sizeof(builtin_versions[0]); i++) {
        VmafModel *model = NULL;
        VmafModelConfig cfg = { 0 };
        int err = vmaf_model_load(&model, &cfg, builtin_versions[i]);
        mu_assert("vmaf_model_load failed", !err);
        mu_assert("model is NULL", model != NULL);
        mu_assert("unexpected feature count", model->n_features == 6);
        mu_assert("score_clip not enabled", model->score_clip.enabled);
        mu_assert("score_clip.min != 0", model->score_clip.min == 0.0);
        mu_assert("score_clip.max != 100", model->score_clip.max == 100.0);
        vmaf_model_destroy(model);
    }
    return NULL;
}
```

#### 3.1.2 File-based Model Loading (`vmaf_model_load_from_path`)

For each JSON model file listed in Sections 2.1 and 2.3, call `vmaf_model_load_from_path()` and assert the same conditions as 3.1.1.

```c
static const char *json_model_paths[] = {
    JSON_MODEL_PATH "vmaf_v0.6.1.json",
    JSON_MODEL_PATH "vmaf_v0.6.1neg.json",
    JSON_MODEL_PATH "vmaf_4k_v0.6.1.json",
    JSON_MODEL_PATH "vmaf_4k_v0.6.1neg.json",
    JSON_MODEL_PATH "vmaf_float_v0.6.1.json",
    JSON_MODEL_PATH "vmaf_float_v0.6.1neg.json",
    JSON_MODEL_PATH "vmaf_float_4k_v0.6.1.json",
    JSON_MODEL_PATH "vmaf_float_b_v0.6.3.json",
    JSON_MODEL_PATH "other_models/vmaf_v0.6.0.json",
    JSON_MODEL_PATH "other_models/vmaf_v0.6.1mfz.json",
    JSON_MODEL_PATH "other_models/nflxtrain_norm_type_none.json",
    JSON_MODEL_PATH "other_models/nflx_v1.json",
};
```

#### 3.1.3 Model Collection Loading (`vmaf_model_collection_load_from_path`)

For each model collection directory listed in Section 2.2, load the collection and assert:

1. Return value is 0 (success).
2. The base `VmafModel *` is non-NULL.
3. The `VmafModelCollection *` is non-NULL.
4. `model_collection->cnt` equals the expected sub-model count.
5. Each sub-model has the same `n_features` as the base model.
6. `vmaf_model_collection_destroy()` completes without error.

```c
static char *test_load_model_collection_rb_v0_6_2(void) {
    VmafModel *model = NULL;
    VmafModelCollection *mc = NULL;
    VmafModelConfig cfg = { 0 };
    const char *path = JSON_MODEL_PATH "vmaf_rb_v0.6.2/vmaf_rb_v0.6.2.json";
    int err = vmaf_model_collection_load_from_path(&model, &mc, &cfg, path);
    mu_assert("collection load failed", !err);
    mu_assert("base model is NULL", model != NULL);
    mu_assert("collection is NULL", mc != NULL);
    mu_assert("expected 20 sub-models", mc->cnt == 20);
    vmaf_model_collection_destroy(mc);
    return NULL;
}
```

#### 3.1.4 Built-in vs. File Equivalence

For each model that has both a built-in version and a JSON file, load both and verify structural equivalence:

1. `n_features` matches.
2. Feature names match for all features.
3. Slopes and intercepts match for all features.
4. Score clip parameters match.
5. Score transform parameters match.
6. Normalization type matches.

This follows the existing `model_compare()` pattern in `test_model.c`.

### 3.2 Feature Dependency Validation

For each loaded model, verify that every feature it requires maps to an available feature extractor.

#### 3.2.1 Feature Extractor Resolution

After loading a model, iterate over `model->feature[i].name` for `i` in `[0, model->n_features)` and call `vmaf_get_feature_extractor_by_feature_name()` for each. Assert the returned `VmafFeatureExtractor *` is non-NULL.

```c
static char *test_feature_dependency_resolution(void) {
    for (unsigned v = 0; v < sizeof(builtin_versions)/sizeof(builtin_versions[0]); v++) {
        VmafModel *model = NULL;
        VmafModelConfig cfg = { 0 };
        int err = vmaf_model_load(&model, &cfg, builtin_versions[v]);
        mu_assert("vmaf_model_load failed", !err);

        for (unsigned i = 0; i < model->n_features; i++) {
            VmafFeatureExtractor *fex =
                vmaf_get_feature_extractor_by_feature_name(model->feature[i].name, 0);
            mu_assert("feature extractor not found for model feature", fex != NULL);
        }
        vmaf_model_destroy(model);
    }
    return NULL;
}
```

#### 3.2.2 Feature Registration

After loading a model, call `vmaf_use_features_from_model()` and verify it returns 0. This validates the full registration path including option dictionary propagation.

```c
static char *test_feature_registration(void) {
    VmafConfiguration vmaf_cfg = {
        .log_level = VMAF_LOG_LEVEL_NONE,
        .n_threads = 1,
    };
    VmafContext *vmaf = NULL;
    int err = vmaf_init(&vmaf, vmaf_cfg);
    mu_assert("vmaf_init failed", !err);

    for (unsigned v = 0; v < sizeof(builtin_versions)/sizeof(builtin_versions[0]); v++) {
        VmafModel *model = NULL;
        VmafModelConfig cfg = { 0 };
        err = vmaf_model_load(&model, &cfg, builtin_versions[v]);
        mu_assert("vmaf_model_load failed", !err);

        err = vmaf_use_features_from_model(vmaf, model);
        mu_assert("vmaf_use_features_from_model failed", !err);

        vmaf_model_destroy(model);
    }

    vmaf_close(vmaf);
    return NULL;
}
```

#### 3.2.3 Expected Feature Names

For each model, verify the exact feature names match what is expected. This table defines the mandatory feature set:

| Model | Expected Feature Names |
|-------|----------------------|
| Integer models (vmaf_v0.6.1, vmaf_v0.6.1neg, vmaf_4k_v0.6.1, vmaf_4k_v0.6.1neg, vmaf_b_v0.6.3) | `VMAF_integer_feature_adm2_score`, `VMAF_integer_feature_motion2_score`, `VMAF_integer_feature_vif_scale0_score`, `VMAF_integer_feature_vif_scale1_score`, `VMAF_integer_feature_vif_scale2_score`, `VMAF_integer_feature_vif_scale3_score` |
| Float models (vmaf_float_v0.6.1, vmaf_float_v0.6.1neg, vmaf_float_4k_v0.6.1, vmaf_float_b_v0.6.3) | `VMAF_feature_adm2_score`, `VMAF_feature_motion2_score`, `VMAF_feature_vif_scale0_score`, `VMAF_feature_vif_scale1_score`, `VMAF_feature_vif_scale2_score`, `VMAF_feature_vif_scale3_score` |

### 3.3 Round-Trip Scoring Tests

For each primary model, execute a complete scoring pipeline and verify the result is in the valid range [0, 100].

#### 3.3.1 Pipeline Steps

1. Create a `VmafContext` with `vmaf_init()`.
2. Load the model with `vmaf_model_load()`.
3. Register features with `vmaf_use_features_from_model()`.
4. Allocate a reference and distorted `VmafPicture` (8-bit, YUV420P, 64x64).
5. Fill reference with a constant mid-gray value (128).
6. Fill distorted with a different constant value (96) to produce a non-trivial score.
7. Call `vmaf_read_pictures()` with the ref/dist pair at index 0.
8. Flush with `vmaf_read_pictures(vmaf, NULL, NULL, 0)`.
9. Call `vmaf_score_at_index()` at index 0.
10. Assert `score >= 0.0 && score <= 100.0`.
11. Clean up with `vmaf_model_destroy()` and `vmaf_close()`.

```c
static char *test_roundtrip_score(const char *version) {
    int err;
    VmafConfiguration vmaf_cfg = {
        .log_level = VMAF_LOG_LEVEL_NONE,
        .n_threads = 1,
    };
    VmafContext *vmaf = NULL;
    err = vmaf_init(&vmaf, vmaf_cfg);
    mu_assert("vmaf_init failed", !err);

    VmafModel *model = NULL;
    VmafModelConfig model_cfg = { 0 };
    err = vmaf_model_load(&model, &model_cfg, version);
    mu_assert("vmaf_model_load failed", !err);

    err = vmaf_use_features_from_model(vmaf, model);
    mu_assert("vmaf_use_features_from_model failed", !err);

    VmafPicture ref, dist;
    err = vmaf_picture_alloc(&ref, VMAF_PIX_FMT_YUV420P, 8, 64, 64);
    mu_assert("ref alloc failed", !err);
    err = vmaf_picture_alloc(&dist, VMAF_PIX_FMT_YUV420P, 8, 64, 64);
    mu_assert("dist alloc failed", !err);

    /* Fill luma plane: ref=128, dist=96 */
    memset(ref.data[0], 128, ref.stride[0] * 64);
    memset(ref.data[1], 128, ref.stride[1] * 32);
    memset(ref.data[2], 128, ref.stride[2] * 32);
    memset(dist.data[0], 96, dist.stride[0] * 64);
    memset(dist.data[1], 96, dist.stride[1] * 32);
    memset(dist.data[2], 96, dist.stride[2] * 32);

    err = vmaf_read_pictures(vmaf, &ref, &dist, 0);
    mu_assert("vmaf_read_pictures failed", !err);
    err = vmaf_read_pictures(vmaf, NULL, NULL, 0);
    mu_assert("vmaf_read_pictures flush failed", !err);

    double score;
    err = vmaf_score_at_index(vmaf, model, &score, 0);
    mu_assert("vmaf_score_at_index failed", !err);
    mu_assert("score out of range [0, 100]", score >= 0.0 && score <= 100.0);

    vmaf_model_destroy(model);
    vmaf_close(vmaf);
    return NULL;
}
```

#### 3.3.2 Models to Test

Round-trip scoring must be tested for all 9 primary built-in models (Section 2.1). Each model gets its own test function that calls the common `test_roundtrip_score()` helper.

#### 3.3.3 Identity Score Test

When the reference and distorted pictures are identical (both filled with the same constant value), the VMAF score should be near 100.0. Assert `score >= 95.0` for all models. This validates that the model's score transform and clip logic produce a near-perfect score for identical inputs.

```c
static char *test_identity_score(const char *version) {
    /* Same pipeline as 3.3.1, but dist = ref (both filled with 128) */
    /* ... */
    mu_assert("identity score should be >= 95.0", score >= 95.0);
    /* ... */
}
```

#### 3.3.4 Model Collection Round-Trip

For the bootstrap model collections (Section 2.2), perform the same round-trip pipeline using `vmaf_model_collection_load()` / `vmaf_use_features_from_model_collection()` / `vmaf_score_at_index_model_collection()`. Assert:

1. The bagging score is in [0, 100].
2. The stddev is non-negative.
3. The 95% CI bounds satisfy `ci.p95.lo <= bagging_score <= ci.p95.hi`.

### 3.4 Version Compatibility Tests

Verify that older model files continue to work with the current library version.

#### 3.4.1 Legacy JSON Models

Load each JSON model from `model/other_models/` via `vmaf_model_load_from_path()`:

| File | Expected Result |
|------|----------------|
| `vmaf_v0.6.0.json` | Loads successfully |
| `vmaf_v0.6.1mfz.json` | Loads successfully |
| `nflxtrain_norm_type_none.json` | Loads successfully |
| `nflx_v1.json` | Loads successfully |

For each, verify the model loads without error and `n_features > 0`.

#### 3.4.2 Model Type Backward Compatibility

Verify that all three supported model types load correctly:

| Model Type | Example Version |
|------------|----------------|
| `VMAF_MODEL_TYPE_SVM_NUSVR` | `vmaf_v0.6.1` |
| `VMAF_MODEL_BOOTSTRAP_SVM_NUSVR` | `vmaf_b_v0.6.3` |
| `VMAF_MODEL_RESIDUE_BOOTSTRAP_SVM_NUSVR` | `vmaf_rb_v0.6.2/vmaf_rb_v0.6.2.json` (collection) |

Assert the `model->type` field matches the expected enum value after loading.

#### 3.4.3 Normalization Type Compatibility

Verify both supported normalization types:

| Normalization Type | Example |
|-------------------|---------|
| `VMAF_MODEL_NORMALIZATION_TYPE_LINEAR_RESCALE` | `vmaf_v0.6.1.json` |
| `VMAF_MODEL_NORMALIZATION_TYPE_NONE` | `nflxtrain_norm_type_none.json` |

### 3.5 Error Handling Tests

Verify that invalid inputs produce appropriate error codes without crashing.

#### 3.5.1 Non-existent Model Path

```c
static char *test_load_nonexistent_path(void) {
    VmafModel *model = NULL;
    VmafModelConfig cfg = { 0 };
    int err = vmaf_model_load_from_path(&model, &cfg, "/nonexistent/path/model.json");
    mu_assert("loading nonexistent path should fail", err != 0);
    mu_assert("model should be NULL on failure", model == NULL);
    return NULL;
}
```

#### 3.5.2 Non-existent Built-in Version

```c
static char *test_load_nonexistent_builtin(void) {
    VmafModel *model = NULL;
    VmafModelConfig cfg = { 0 };
    int err = vmaf_model_load(&model, &cfg, "vmaf_nonexistent_v99.99");
    mu_assert("loading nonexistent version should fail", err != 0);
    return NULL;
}
```

#### 3.5.3 Corrupt Model File

Create a temporary file with invalid JSON content and attempt to load it. Assert that `vmaf_model_load_from_path()` returns an error code and does not crash.

```c
static char *test_load_corrupt_json(void) {
    const char *corrupt_path = "/tmp/vmaf_corrupt_model.json";
    FILE *f = fopen(corrupt_path, "w");
    mu_assert("could not create temp file", f != NULL);
    fprintf(f, "{ this is not valid json !!!");
    fclose(f);

    VmafModel *model = NULL;
    VmafModelConfig cfg = { 0 };
    int err = vmaf_model_load_from_path(&model, &cfg, corrupt_path);
    mu_assert("loading corrupt JSON should fail", err != 0);

    remove(corrupt_path);
    return NULL;
}
```

#### 3.5.4 Truncated Model File

Create a temporary file with valid JSON that is truncated mid-stream (e.g., cut off after the first 100 bytes of a valid model). Assert failure.

#### 3.5.5 Empty File

```c
static char *test_load_empty_file(void) {
    const char *empty_path = "/tmp/vmaf_empty_model.json";
    FILE *f = fopen(empty_path, "w");
    fclose(f);

    VmafModel *model = NULL;
    VmafModelConfig cfg = { 0 };
    int err = vmaf_model_load_from_path(&model, &cfg, empty_path);
    mu_assert("loading empty file should fail", err != 0);

    remove(empty_path);
    return NULL;
}
```

#### 3.5.6 PKL Format Rejection

Attempt to load a `.pkl` file via `vmaf_model_load_from_path()`. Assert that it fails (pkl support has been removed).

```c
static char *test_load_pkl_rejected(void) {
    VmafModel *model = NULL;
    VmafModelConfig cfg = { 0 };
    const char *pkl_path = JSON_MODEL_PATH "other_models/vmaf_v0.6.0.pkl";
    int err = vmaf_model_load_from_path(&model, &cfg, pkl_path);
    mu_assert("loading pkl file should fail", err != 0);
    return NULL;
}
```

#### 3.5.7 NULL Argument Handling

Verify that API functions handle NULL arguments without crashing:

```c
static char *test_null_arguments(void) {
    /* vmaf_model_destroy(NULL) should be a no-op */
    vmaf_model_destroy(NULL);

    /* vmaf_model_collection_destroy(NULL) should be a no-op */
    vmaf_model_collection_destroy(NULL);

    /* vmaf_model_feature_overload with NULL model */
    int err = vmaf_model_feature_overload(NULL, "adm", NULL);
    mu_assert("overload with NULL model should fail", err != 0);

    return NULL;
}
```

### 3.6 Model Configuration Tests

#### 3.6.1 Flag Behavior

Verify that model configuration flags are applied correctly:

| Flag | Expected Behavior |
|------|------------------|
| `VMAF_MODEL_FLAGS_DEFAULT` | Clip enabled, transform disabled |
| `VMAF_MODEL_FLAG_DISABLE_CLIP` | Clip disabled |
| `VMAF_MODEL_FLAG_ENABLE_TRANSFORM` | Score transform enabled |
| `VMAF_MODEL_FLAG_DISABLE_TRANSFORM` | Score transform disabled (redundant with default) |

```c
static char *test_model_flags(void) {
    int err;

    /* Default flags: clip enabled, transform disabled */
    VmafModel *m1 = NULL;
    VmafModelConfig cfg1 = { .flags = VMAF_MODEL_FLAGS_DEFAULT };
    err = vmaf_model_load(&m1, &cfg1, "vmaf_v0.6.1");
    mu_assert("load failed", !err);
    mu_assert("clip should be enabled", m1->score_clip.enabled);
    mu_assert("transform should be disabled", !m1->score_transform.enabled);
    vmaf_model_destroy(m1);

    /* DISABLE_CLIP */
    VmafModel *m2 = NULL;
    VmafModelConfig cfg2 = { .flags = VMAF_MODEL_FLAG_DISABLE_CLIP };
    err = vmaf_model_load(&m2, &cfg2, "vmaf_v0.6.1");
    mu_assert("load failed", !err);
    mu_assert("clip should be disabled", !m2->score_clip.enabled);
    vmaf_model_destroy(m2);

    /* ENABLE_TRANSFORM */
    VmafModel *m3 = NULL;
    VmafModelConfig cfg3 = { .flags = VMAF_MODEL_FLAG_ENABLE_TRANSFORM };
    err = vmaf_model_load(&m3, &cfg3, "vmaf_v0.6.1");
    mu_assert("load failed", !err);
    mu_assert("transform should be enabled", m3->score_transform.enabled);
    vmaf_model_destroy(m3);

    return NULL;
}
```

#### 3.6.2 Custom Model Name

Verify that `VmafModelConfig.name` is correctly applied:

```c
static char *test_custom_model_name(void) {
    VmafModel *model = NULL;
    VmafModelConfig cfg = { .name = "my_custom_vmaf" };
    int err = vmaf_model_load(&model, &cfg, "vmaf_v0.6.1");
    mu_assert("load failed", !err);
    mu_assert("name should match", !strcmp(model->name, "my_custom_vmaf"));
    vmaf_model_destroy(model);
    return NULL;
}
```

#### 3.6.3 Feature Overload

Verify that `vmaf_model_feature_overload()` correctly modifies a model's feature options:

1. Load `vmaf_v0.6.1`.
2. Call `vmaf_model_feature_overload(model, "adm", dict)` with `adm_enhn_gain_limit=1.0`.
3. Verify `model->feature[0].opts_dict` now contains the key-value pair.
4. Verify the overloaded model's structure matches `vmaf_v0.6.1neg` (which has the same overload baked in).

### 3.7 Neg Model Option Validation

For "neg" model variants (`vmaf_v0.6.1neg`, `vmaf_4k_v0.6.1neg`), verify that the `feature_opts_dicts` are correctly parsed:

1. Feature 0 (adm2) should have `opts_dict` with key `adm_enhn_gain_limit` = `"1"`.
2. Feature 1 (motion2) should have NULL `opts_dict`.
3. Features 2-5 (vif_scale0-3) should each have `opts_dict` with key `vif_enhn_gain_limit` = `"1"`.

---

## 4. Test Architecture

### 4.1 Design Principle

Tests use the public libvmaf API (`libvmaf/include/libvmaf/*.h`) as much as possible. Internal model structure access (via `#include "model.c"` or `#include "model.h"`) is used only for structural validation tests (comparing feature names, slopes, intercepts) that cannot be performed through the public API.

### 4.2 Test File Organization

```
libvmaf/test/
    test_model_validation.c    # All model validation tests
```

A single test file following the existing Minunit pattern (`test.h`).

### 4.3 Test Data

- **Synthetic pictures:** Allocated programmatically using `vmaf_picture_alloc()` with constant fill values. No external video files required for model validation tests.
- **Model files:** Referenced via `JSON_MODEL_PATH` macro, which resolves to the repository's `model/` directory at compile time.
- **Corrupt/invalid files:** Created in `/tmp/` during test execution and cleaned up after.

### 4.4 Conditional Compilation

Float-feature model tests must be guarded:

```c
#if VMAF_FLOAT_FEATURES
    mu_run_test(test_load_float_v0_6_1);
    mu_run_test(test_roundtrip_float_v0_6_1);
    /* ... */
#endif
```

---

## 5. Integration with Build System

### 5.1 Meson Configuration

Add to `libvmaf/test/meson.build`:

```meson
test_model_validation = executable('test_model_validation',
    ['test.c', 'test_model_validation.c',
     '../src/dict.c', '../src/pdjson.c',
     '../src/read_json_model.c', '../src/log.c',
     json_model_c_sources],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src')],
    link_with : get_option('default_library') == 'both' ? libvmaf.get_static_lib() : libvmaf,
    c_args : [vmaf_cflags_common,
              '-DJSON_MODEL_PATH="' + join_paths(meson.project_source_root(), '../model/') + '"'],
    dependencies : [thread_lib, cuda_dependency],
    objects : libsvm_static_lib.extract_all_objects(recursive: true),
)

test('test_model_validation', test_model_validation)
```

### 5.2 CI Integration

The model validation tests run as part of `ninja test` on all CI platforms. No additional CI workflow changes are needed. The tests have no external dependencies beyond the model files already in the repository.

---

## 6. Test Execution Matrix

### 6.1 Loading Tests

| Test Category | Models Tested | Assertions per Model | Total |
|--------------|---------------|---------------------|-------|
| Built-in loading | 9 (5 integer + 4 float) | 6 | 54 |
| File-based loading | 12 JSON files | 6 | 72 |
| Collection loading | 5 collections | 5 | 25 |
| Built-in vs. file equivalence | 9 pairs | ~15 structural checks | 135 |
| **Subtotal** | | | **286** |

### 6.2 Feature Dependency Tests

| Test Category | Models Tested | Assertions per Model | Total |
|--------------|---------------|---------------------|-------|
| Feature extractor resolution | 9 built-in | 6 features each | 54 |
| Feature registration | 9 built-in | 1 | 9 |
| Feature name validation | 9 built-in | 6 names each | 54 |
| **Subtotal** | | | **117** |

### 6.3 Round-Trip Scoring Tests

| Test Category | Models Tested | Assertions per Model | Total |
|--------------|---------------|---------------------|-------|
| Standard round-trip | 9 built-in | 1 range check | 9 |
| Identity score | 9 built-in | 1 near-100 check | 9 |
| Collection round-trip | 3 collections | 3 (bagging, stddev, CI) | 9 |
| **Subtotal** | | | **27** |

### 6.4 Error Handling Tests

| Test Category | Assertions | Total |
|--------------|------------|-------|
| Non-existent path | 2 | 2 |
| Non-existent built-in | 1 | 1 |
| Corrupt JSON | 1 | 1 |
| Truncated file | 1 | 1 |
| Empty file | 1 | 1 |
| PKL rejection | 1 | 1 |
| NULL arguments | 2 | 2 |
| **Subtotal** | | **9** |

### 6.5 Configuration Tests

| Test Category | Assertions | Total |
|--------------|------------|-------|
| Flag behavior | 5 | 5 |
| Custom name | 1 | 1 |
| Feature overload | 3 | 3 |
| Neg model opts | 6 | 6 |
| **Subtotal** | | **15** |

**Total assertion points: ~454**

---

## 7. Completion Requirements

The model validation test suite is **complete** when ALL of the following requirements are met:

### R1. Full Model Coverage

Every model listed in Section 2.1 (all 9 primary built-in models) has a loading test, a feature dependency test, and a round-trip scoring test. No shipped model is left untested.

**Verification:** For each entry in the table in Section 2.1, there exists a test function that loads the model, checks its features, and scores a synthetic picture pair.

### R2. Model Collection Coverage

Every model collection listed in Section 2.2 (all 5 collections) has a loading test that verifies the expected sub-model count.

**Verification:** Count the `mu_run_test()` calls for collection loading tests. There must be at least 5.

### R3. Feature Dependency Completeness

For every loaded model, every feature name in `model->feature[i].name` is resolved to a non-NULL `VmafFeatureExtractor*` via `vmaf_get_feature_extractor_by_feature_name()`.

**Verification:** Code review confirms the resolution loop covers all features and all models.

### R4. Valid Score Range

Every round-trip scoring test asserts that the resulting VMAF score is in the range [0.0, 100.0]. Identity tests (ref == dist) assert the score is >= 95.0.

**Verification:** Code review confirms the range assertions exist for all round-trip tests.

### R5. Error Handling Coverage

The test suite includes at least 7 error-path tests: non-existent path, non-existent built-in, corrupt JSON, truncated JSON, empty file, PKL rejection, and NULL arguments. Each asserts a non-zero return code and no crash.

**Verification:** Count the error-path test functions. There must be at least 7.

### R6. Legacy Model Compatibility

At least 4 models from `model/other_models/` (JSON format) are tested for successful loading. This verifies backward compatibility with older model file formats.

**Verification:** The legacy model loading test iterates over at least 4 paths from `other_models/`.

### R7. Configuration Flag Tests

All 4 flag values (`VMAF_MODEL_FLAGS_DEFAULT`, `VMAF_MODEL_FLAG_DISABLE_CLIP`, `VMAF_MODEL_FLAG_ENABLE_TRANSFORM`, `VMAF_MODEL_FLAG_DISABLE_TRANSFORM`) are tested for correct behavior.

**Verification:** Code review confirms each flag is set and its effect is asserted.

### R8. Neg Option Validation

For both `vmaf_v0.6.1neg` and `vmaf_4k_v0.6.1neg`, the `feature_opts_dicts` are verified to contain the correct enhancement gain limit keys and values.

**Verification:** Test functions exist that check `opts_dict` keys/values for features 0 and 2-5 of both neg models.

### R9. Build Integration

The tests compile and run as part of `ninja test` on all CI platforms (Ubuntu x86_64, Ubuntu ARM64, macOS x86_64, Windows x86_64) without manual intervention.

**Verification:** CI pipeline passes with the new tests on all platforms.

### R10. No External Dependencies

The tests do not require any external video files, network access, or resources beyond the model files already in the repository. All test pictures are synthesized programmatically.

**Verification:** Code review confirms no file I/O to external paths (only `/tmp/` for corrupt file tests, with cleanup).

### R11. Float Feature Guard

All tests for float-feature models (vmaf_float_v0.6.1, vmaf_float_v0.6.1neg, vmaf_float_4k_v0.6.1, vmaf_float_b_v0.6.3) are guarded by `#if VMAF_FLOAT_FEATURES`. The test suite compiles and passes both with and without float features enabled.

**Verification:** Build the test suite with and without `-Denable_float=true`. Both configurations compile and pass.

### R12. Deterministic Reproducibility

All tests are fully deterministic. Synthetic picture data uses fixed constant values. No PRNG, no wall-clock time, no thread scheduling dependency. Running the test suite twice produces identical results.

**Verification:** Run the test suite twice and confirm identical pass/fail results.

---

## 8. Acceptance Criteria Summary

| Req | Description | Pass Condition |
|-----|-------------|----------------|
| R1 | Full model coverage | 9/9 primary models have load + feature + scoring tests |
| R2 | Collection coverage | 5/5 model collections have load tests with sub-model count checks |
| R3 | Feature dependency | All model features resolve to non-NULL feature extractors |
| R4 | Valid score range | All round-trip scores in [0, 100]; identity scores >= 95 |
| R5 | Error handling | >= 7 error-path tests, all assert non-zero return and no crash |
| R6 | Legacy compatibility | >= 4 legacy JSON models from other_models/ load successfully |
| R7 | Configuration flags | All 4 flag values tested with correct behavior |
| R8 | Neg option validation | adm_enhn_gain_limit and vif_enhn_gain_limit verified for neg models |
| R9 | Build integration | Tests pass in CI on all 4 platforms via `ninja test` |
| R10 | No external deps | Only repo model files and /tmp/ used; no network or external video |
| R11 | Float feature guard | Compiles and passes with and without float features |
| R12 | Deterministic | Repeated runs produce identical results |

All 12 requirements must be met for the model validation test suite to be considered complete.
