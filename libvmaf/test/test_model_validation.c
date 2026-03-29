/**
 *
 *  Copyright 2016-2020 Netflix, Inc.
 *
 *     Licensed under the BSD+Patent License (the "License");
 *     you may not use this file except in compliance with the License.
 *     You may obtain a copy of the License at
 *
 *         https://opensource.org/licenses/BSDplusPatent
 *
 *     Unless required by applicable law or agreed to in writing, software
 *     distributed under the License is distributed on an "AS IS" BASIS,
 *     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *     See the License for the specific language governing permissions and
 *     limitations under the License.
 *
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "config.h"
#include "test.h"
#include "model.c"
#include "read_json_model.h"
#include "feature/feature_extractor.h"

#include <libvmaf/libvmaf.h>
#include <libvmaf/picture.h>

/* ===================================================================
 * Section 1: Model version strings and paths
 *
 * "single" models can be loaded with vmaf_model_load().
 * "collection" / bootstrap models must use vmaf_model_collection_load().
 * =================================================================== */

/* Single (non-collection) built-in integer models */
static const char *builtin_single_versions[] = {
    "vmaf_v0.6.1",
    "vmaf_v0.6.1neg",
    "vmaf_4k_v0.6.1",
    "vmaf_4k_v0.6.1neg",
};
#define BUILTIN_SINGLE_COUNT \
    (sizeof(builtin_single_versions) / sizeof(builtin_single_versions[0]))

/* Built-in collection (bootstrap) integer models */
static const char *builtin_collection_versions[] = {
    "vmaf_b_v0.6.3",
};
#define BUILTIN_COLLECTION_COUNT \
    (sizeof(builtin_collection_versions) / sizeof(builtin_collection_versions[0]))

#if VMAF_FLOAT_FEATURES
/* Single (non-collection) built-in float models */
static const char *builtin_float_single_versions[] = {
    "vmaf_float_v0.6.1",
    "vmaf_float_v0.6.1neg",
    "vmaf_float_4k_v0.6.1",
};
#define BUILTIN_FLOAT_SINGLE_COUNT \
    (sizeof(builtin_float_single_versions) / sizeof(builtin_float_single_versions[0]))

/* Built-in collection (bootstrap) float models */
static const char *builtin_float_collection_versions[] = {
    "vmaf_float_b_v0.6.3",
};
#define BUILTIN_FLOAT_COLLECTION_COUNT \
    (sizeof(builtin_float_collection_versions) / sizeof(builtin_float_collection_versions[0]))
#endif

/* JSON file paths for single models (file-based loading) */
static const char *json_single_model_paths[] = {
    JSON_MODEL_PATH "vmaf_v0.6.1.json",
    JSON_MODEL_PATH "vmaf_v0.6.1neg.json",
    JSON_MODEL_PATH "vmaf_4k_v0.6.1.json",
    JSON_MODEL_PATH "vmaf_4k_v0.6.1neg.json",
    JSON_MODEL_PATH "vmaf_float_v0.6.1.json",
    JSON_MODEL_PATH "vmaf_float_v0.6.1neg.json",
    JSON_MODEL_PATH "vmaf_float_4k_v0.6.1.json",
};
#define JSON_SINGLE_MODEL_COUNT \
    (sizeof(json_single_model_paths) / sizeof(json_single_model_paths[0]))

/* Legacy / other models */
static const char *legacy_json_model_paths[] = {
    JSON_MODEL_PATH "other_models/vmaf_v0.6.0.json",
    JSON_MODEL_PATH "other_models/vmaf_v0.6.1mfz.json",
    JSON_MODEL_PATH "other_models/nflxtrain_norm_type_none.json",
    JSON_MODEL_PATH "other_models/nflx_v1.json",
};
#define LEGACY_MODEL_COUNT \
    (sizeof(legacy_json_model_paths) / sizeof(legacy_json_model_paths[0]))

/* Expected feature names for integer models */
static const char *integer_feature_names[] = {
    "VMAF_integer_feature_adm2_score",
    "VMAF_integer_feature_motion2_score",
    "VMAF_integer_feature_vif_scale0_score",
    "VMAF_integer_feature_vif_scale1_score",
    "VMAF_integer_feature_vif_scale2_score",
    "VMAF_integer_feature_vif_scale3_score",
};

#if VMAF_FLOAT_FEATURES
/* Expected feature names for float models */
static const char *float_feature_names[] = {
    "VMAF_feature_adm2_score",
    "VMAF_feature_motion2_score",
    "VMAF_feature_vif_scale0_score",
    "VMAF_feature_vif_scale1_score",
    "VMAF_feature_vif_scale2_score",
    "VMAF_feature_vif_scale3_score",
};
#endif

/* ===================================================================
 * Internal helper: model_compare (same as in test_model.c)
 * =================================================================== */

static int model_compare(VmafModel *model_a, VmafModel *model_b)
{
    int err = 0;

    err += model_a->slope != model_b->slope;
    err += model_a->intercept != model_b->intercept;

    err += model_a->n_features != model_b->n_features;
    for (unsigned i = 0; i < model_a->n_features; i++) {
        err += strcmp(model_a->feature[i].name,
                      model_b->feature[i].name) != 0;
        err += model_a->feature[i].slope != model_b->feature[i].slope;
        err += model_a->feature[i].intercept != model_b->feature[i].intercept;
        err += !model_a->feature[i].opts_dict != !model_b->feature[i].opts_dict;
    }

    err += model_a->score_clip.enabled != model_b->score_clip.enabled;
    err += model_a->score_clip.min != model_b->score_clip.min;
    err += model_a->score_clip.max != model_b->score_clip.max;

    err += model_a->norm_type != model_b->norm_type;

    err += model_a->score_transform.enabled != model_b->score_transform.enabled;
    err += model_a->score_transform.p0.enabled !=
           model_b->score_transform.p0.enabled;
    err += model_a->score_transform.p0.value !=
           model_b->score_transform.p0.value;
    err += model_a->score_transform.p1.enabled !=
           model_b->score_transform.p1.enabled;
    err += model_a->score_transform.p1.value !=
           model_b->score_transform.p1.value;
    err += model_a->score_transform.p2.enabled !=
           model_b->score_transform.p2.enabled;
    err += model_a->score_transform.p2.value !=
           model_b->score_transform.p2.value;
    err += model_a->score_transform.knots.enabled !=
           model_b->score_transform.knots.enabled;
    for (unsigned i = 0; i < model_a->score_transform.knots.n_knots; i++) {
        err += model_a->score_transform.knots.list[i].x !=
               model_b->score_transform.knots.list[i].x;
        err += model_a->score_transform.knots.list[i].y !=
               model_b->score_transform.knots.list[i].y;
    }
    err += model_a->score_transform.out_lte_in !=
           model_b->score_transform.out_lte_in;
    err += model_a->score_transform.out_gte_in !=
           model_b->score_transform.out_gte_in;

    return err;
}

/* ===================================================================
 * 3.1.1: Built-in Model Loading Tests (single models)
 * =================================================================== */

static char *test_load_all_builtin_single_models(void)
{
    for (unsigned i = 0; i < BUILTIN_SINGLE_COUNT; i++) {
        VmafModel *model = NULL;
        VmafModelConfig cfg = { 0 };
        int err = vmaf_model_load(&model, &cfg, builtin_single_versions[i]);
        mu_assert("vmaf_model_load failed for built-in model", !err);
        mu_assert("model is NULL after load", model != NULL);
        mu_assert("unexpected feature count (expected 6)",
                  model->n_features == 6);
        mu_assert("score_clip not enabled", model->score_clip.enabled);
        mu_assert("score_clip.min != 0", model->score_clip.min == 0.0);
        mu_assert("score_clip.max != 100", model->score_clip.max == 100.0);
        vmaf_model_destroy(model);
    }
    return NULL;
}

/* Built-in collection models: load via vmaf_model_collection_load */
static char *test_load_all_builtin_collection_models(void)
{
    for (unsigned i = 0; i < BUILTIN_COLLECTION_COUNT; i++) {
        VmafModel *model = NULL;
        VmafModelCollection *mc = NULL;
        VmafModelConfig cfg = { 0 };
        int err = vmaf_model_collection_load(&model, &mc, &cfg,
                                             builtin_collection_versions[i]);
        mu_assert("vmaf_model_collection_load failed for built-in collection",
                  !err);
        mu_assert("base model is NULL after collection load", model != NULL);
        mu_assert("collection is NULL after collection load", mc != NULL);
        mu_assert("unexpected feature count (expected 6)",
                  model->n_features == 6);
        mu_assert("score_clip not enabled", model->score_clip.enabled);
        mu_assert("score_clip.min != 0", model->score_clip.min == 0.0);
        mu_assert("score_clip.max != 100", model->score_clip.max == 100.0);
        vmaf_model_collection_destroy(mc);
    }
    return NULL;
}

#if VMAF_FLOAT_FEATURES
static char *test_load_all_builtin_float_single_models(void)
{
    for (unsigned i = 0; i < BUILTIN_FLOAT_SINGLE_COUNT; i++) {
        VmafModel *model = NULL;
        VmafModelConfig cfg = { 0 };
        int err = vmaf_model_load(&model, &cfg,
                                  builtin_float_single_versions[i]);
        mu_assert("vmaf_model_load failed for float built-in model", !err);
        mu_assert("float model is NULL after load", model != NULL);
        mu_assert("unexpected feature count (expected 6)",
                  model->n_features == 6);
        mu_assert("score_clip not enabled", model->score_clip.enabled);
        mu_assert("score_clip.min != 0", model->score_clip.min == 0.0);
        mu_assert("score_clip.max != 100", model->score_clip.max == 100.0);
        vmaf_model_destroy(model);
    }
    return NULL;
}

static char *test_load_all_builtin_float_collection_models(void)
{
    for (unsigned i = 0; i < BUILTIN_FLOAT_COLLECTION_COUNT; i++) {
        VmafModel *model = NULL;
        VmafModelCollection *mc = NULL;
        VmafModelConfig cfg = { 0 };
        int err = vmaf_model_collection_load(
            &model, &mc, &cfg, builtin_float_collection_versions[i]);
        mu_assert("vmaf_model_collection_load failed for float collection",
                  !err);
        mu_assert("base model is NULL", model != NULL);
        mu_assert("collection is NULL", mc != NULL);
        mu_assert("unexpected feature count (expected 6)",
                  model->n_features == 6);
        mu_assert("score_clip not enabled", model->score_clip.enabled);
        mu_assert("score_clip.min != 0", model->score_clip.min == 0.0);
        mu_assert("score_clip.max != 100", model->score_clip.max == 100.0);
        vmaf_model_collection_destroy(mc);
    }
    return NULL;
}
#endif

/* ===================================================================
 * 3.1.2: File-based Model Loading Tests
 * =================================================================== */

static char *test_load_all_json_single_models(void)
{
    for (unsigned i = 0; i < JSON_SINGLE_MODEL_COUNT; i++) {
        VmafModel *model = NULL;
        VmafModelConfig cfg = { 0 };
        int err = vmaf_model_load_from_path(&model, &cfg,
                                            json_single_model_paths[i]);
        mu_assert("vmaf_model_load_from_path failed for JSON model", !err);
        mu_assert("model is NULL after file load", model != NULL);
        mu_assert("unexpected feature count (expected 6)",
                  model->n_features == 6);
        mu_assert("score_clip not enabled", model->score_clip.enabled);
        mu_assert("score_clip.min != 0", model->score_clip.min == 0.0);
        mu_assert("score_clip.max != 100", model->score_clip.max == 100.0);
        vmaf_model_destroy(model);
    }
    return NULL;
}

/* File-based loading for collection models */
static char *test_load_json_collection_models(void)
{
    static const char *collection_json_paths[] = {
        JSON_MODEL_PATH "vmaf_b_v0.6.3.json",
        JSON_MODEL_PATH "vmaf_float_b_v0.6.3.json",
    };
    unsigned count = sizeof(collection_json_paths) /
                     sizeof(collection_json_paths[0]);

    for (unsigned i = 0; i < count; i++) {
        VmafModel *model = NULL;
        VmafModelCollection *mc = NULL;
        VmafModelConfig cfg = { 0 };
        int err = vmaf_model_collection_load_from_path(
            &model, &mc, &cfg, collection_json_paths[i]);
        mu_assert("collection load from path failed", !err);
        mu_assert("base model is NULL", model != NULL);
        mu_assert("collection is NULL", mc != NULL);
        mu_assert("unexpected feature count (expected 6)",
                  model->n_features == 6);
        vmaf_model_collection_destroy(mc);
    }
    return NULL;
}

/* ===================================================================
 * 3.1.3: Model Collection Loading Tests
 * =================================================================== */

static char *test_load_model_collection_b_v0_6_3(void)
{
    VmafModel *model = NULL;
    VmafModelCollection *mc = NULL;
    VmafModelConfig cfg = { 0 };
    const char *path = JSON_MODEL_PATH "vmaf_b_v0.6.3.json";
    int err = vmaf_model_collection_load_from_path(&model, &mc, &cfg, path);
    mu_assert("collection load failed for vmaf_b_v0.6.3", !err);
    mu_assert("base model is NULL", model != NULL);
    mu_assert("collection is NULL", mc != NULL);
    /* 21 entries in JSON: key "0" is base model, keys "1"-"20" in collection */
    mu_assert("expected 20 sub-models for vmaf_b_v0.6.3", mc->cnt == 20);
    for (unsigned i = 0; i < mc->cnt; i++) {
        mu_assert("sub-model n_features mismatch",
                  mc->model[i]->n_features == model->n_features);
    }
    vmaf_model_collection_destroy(mc);
    return NULL;
}

static char *test_load_model_collection_rb_v0_6_2(void)
{
    VmafModel *model = NULL;
    VmafModelCollection *mc = NULL;
    VmafModelConfig cfg = { 0 };
    const char *path =
        JSON_MODEL_PATH "vmaf_rb_v0.6.2/vmaf_rb_v0.6.2.json";
    int err = vmaf_model_collection_load_from_path(&model, &mc, &cfg, path);
    mu_assert("collection load failed for vmaf_rb_v0.6.2", !err);
    mu_assert("base model is NULL", model != NULL);
    mu_assert("collection is NULL", mc != NULL);
    /* 20 entries: key "0" is base, keys "1"-"19" in collection */
    mu_assert("expected 19 sub-models for vmaf_rb_v0.6.2", mc->cnt == 19);
    for (unsigned i = 0; i < mc->cnt; i++) {
        mu_assert("sub-model n_features mismatch",
                  mc->model[i]->n_features == model->n_features);
    }
    vmaf_model_collection_destroy(mc);
    return NULL;
}

static char *test_load_model_collection_rb_v0_6_3(void)
{
    VmafModel *model = NULL;
    VmafModelCollection *mc = NULL;
    VmafModelConfig cfg = { 0 };
    const char *path =
        JSON_MODEL_PATH "vmaf_rb_v0.6.3/vmaf_rb_v0.6.3.json";
    int err = vmaf_model_collection_load_from_path(&model, &mc, &cfg, path);
    mu_assert("collection load failed for vmaf_rb_v0.6.3", !err);
    mu_assert("base model is NULL", model != NULL);
    mu_assert("collection is NULL", mc != NULL);
    /* 21 entries: key "0" is base, keys "1"-"20" in collection */
    mu_assert("expected 20 sub-models for vmaf_rb_v0.6.3", mc->cnt == 20);
    for (unsigned i = 0; i < mc->cnt; i++) {
        mu_assert("sub-model n_features mismatch",
                  mc->model[i]->n_features == model->n_features);
    }
    vmaf_model_collection_destroy(mc);
    return NULL;
}

static char *test_load_model_collection_4k_rb_v0_6_2(void)
{
    VmafModel *model = NULL;
    VmafModelCollection *mc = NULL;
    VmafModelConfig cfg = { 0 };
    const char *path =
        JSON_MODEL_PATH "vmaf_4k_rb_v0.6.2/vmaf_4k_rb_v0.6.2.json";
    int err = vmaf_model_collection_load_from_path(&model, &mc, &cfg, path);
    mu_assert("collection load failed for vmaf_4k_rb_v0.6.2", !err);
    mu_assert("base model is NULL", model != NULL);
    mu_assert("collection is NULL", mc != NULL);
    /* 20 entries: key "0" is base, keys "1"-"19" in collection */
    mu_assert("expected 19 sub-models for vmaf_4k_rb_v0.6.2", mc->cnt == 19);
    for (unsigned i = 0; i < mc->cnt; i++) {
        mu_assert("sub-model n_features mismatch",
                  mc->model[i]->n_features == model->n_features);
    }
    vmaf_model_collection_destroy(mc);
    return NULL;
}

#if VMAF_FLOAT_FEATURES
static char *test_load_model_collection_float_b_v0_6_3(void)
{
    VmafModel *model = NULL;
    VmafModelCollection *mc = NULL;
    VmafModelConfig cfg = { 0 };
    const char *path =
        JSON_MODEL_PATH "vmaf_float_b_v0.6.3/vmaf_float_b_v0.6.3.json";
    int err = vmaf_model_collection_load_from_path(&model, &mc, &cfg, path);
    mu_assert("collection load failed for vmaf_float_b_v0.6.3", !err);
    mu_assert("base model is NULL", model != NULL);
    mu_assert("collection is NULL", mc != NULL);
    /* 21 entries: key "0" is base, keys "1"-"20" in collection */
    mu_assert("expected 20 sub-models for vmaf_float_b_v0.6.3",
              mc->cnt == 20);
    for (unsigned i = 0; i < mc->cnt; i++) {
        mu_assert("sub-model n_features mismatch",
                  mc->model[i]->n_features == model->n_features);
    }
    vmaf_model_collection_destroy(mc);
    return NULL;
}
#endif

/* ===================================================================
 * 3.1.4: Built-in vs File Equivalence Tests
 * =================================================================== */

static char *test_builtin_vs_file_equivalence(void)
{
    static const struct {
        const char *version;
        const char *json_path;
    } pairs[] = {
        { "vmaf_v0.6.1",      JSON_MODEL_PATH "vmaf_v0.6.1.json" },
        { "vmaf_v0.6.1neg",   JSON_MODEL_PATH "vmaf_v0.6.1neg.json" },
        { "vmaf_4k_v0.6.1",   JSON_MODEL_PATH "vmaf_4k_v0.6.1.json" },
        { "vmaf_4k_v0.6.1neg", JSON_MODEL_PATH "vmaf_4k_v0.6.1neg.json" },
    };

    for (unsigned i = 0; i < sizeof(pairs) / sizeof(pairs[0]); i++) {
        VmafModel *model_builtin = NULL;
        VmafModelConfig cfg_builtin = { 0 };
        int err = vmaf_model_load(&model_builtin, &cfg_builtin,
                                  pairs[i].version);
        mu_assert("vmaf_model_load failed in equivalence test", !err);

        VmafModel *model_file = NULL;
        VmafModelConfig cfg_file = { 0 };
        err = vmaf_model_load_from_path(&model_file, &cfg_file,
                                        pairs[i].json_path);
        mu_assert("vmaf_model_load_from_path failed in equivalence test",
                  !err);

        err = model_compare(model_builtin, model_file);
        mu_assert("built-in and file-loaded models do not match", !err);

        vmaf_model_destroy(model_builtin);
        vmaf_model_destroy(model_file);
    }
    return NULL;
}

#if VMAF_FLOAT_FEATURES
static char *test_builtin_vs_file_equivalence_float(void)
{
    static const struct {
        const char *version;
        const char *json_path;
    } pairs[] = {
        { "vmaf_float_v0.6.1",
          JSON_MODEL_PATH "vmaf_float_v0.6.1.json" },
        { "vmaf_float_v0.6.1neg",
          JSON_MODEL_PATH "vmaf_float_v0.6.1neg.json" },
        { "vmaf_float_4k_v0.6.1",
          JSON_MODEL_PATH "vmaf_float_4k_v0.6.1.json" },
    };

    for (unsigned i = 0; i < sizeof(pairs) / sizeof(pairs[0]); i++) {
        VmafModel *model_builtin = NULL;
        VmafModelConfig cfg_builtin = { 0 };
        int err = vmaf_model_load(&model_builtin, &cfg_builtin,
                                  pairs[i].version);
        mu_assert("vmaf_model_load failed in float equivalence test", !err);

        VmafModel *model_file = NULL;
        VmafModelConfig cfg_file = { 0 };
        err = vmaf_model_load_from_path(&model_file, &cfg_file,
                                        pairs[i].json_path);
        mu_assert("vmaf_model_load_from_path failed in float equiv test",
                  !err);

        err = model_compare(model_builtin, model_file);
        mu_assert("built-in and file-loaded float models do not match", !err);

        vmaf_model_destroy(model_builtin);
        vmaf_model_destroy(model_file);
    }
    return NULL;
}
#endif

/* ===================================================================
 * 3.2.1: Feature Dependency Resolution Tests
 * =================================================================== */

static char *test_feature_dependency_resolution(void)
{
    for (unsigned v = 0; v < BUILTIN_SINGLE_COUNT; v++) {
        VmafModel *model = NULL;
        VmafModelConfig cfg = { 0 };
        int err = vmaf_model_load(&model, &cfg, builtin_single_versions[v]);
        mu_assert("vmaf_model_load failed in feature dependency test", !err);

        for (unsigned i = 0; i < model->n_features; i++) {
            VmafFeatureExtractor *fex =
                vmaf_get_feature_extractor_by_feature_name(
                    model->feature[i].name, 0);
            mu_assert("feature extractor not found for model feature",
                      fex != NULL);
        }
        vmaf_model_destroy(model);
    }

    /* Also check the collection models via their base model */
    for (unsigned v = 0; v < BUILTIN_COLLECTION_COUNT; v++) {
        VmafModel *model = NULL;
        VmafModelCollection *mc = NULL;
        VmafModelConfig cfg = { 0 };
        int err = vmaf_model_collection_load(&model, &mc, &cfg,
                                             builtin_collection_versions[v]);
        mu_assert("collection load failed in feature dep test", !err);

        for (unsigned i = 0; i < model->n_features; i++) {
            VmafFeatureExtractor *fex =
                vmaf_get_feature_extractor_by_feature_name(
                    model->feature[i].name, 0);
            mu_assert("feature extractor not found for collection feature",
                      fex != NULL);
        }
        vmaf_model_collection_destroy(mc);
    }

    return NULL;
}

#if VMAF_FLOAT_FEATURES
static char *test_feature_dependency_resolution_float(void)
{
    for (unsigned v = 0; v < BUILTIN_FLOAT_SINGLE_COUNT; v++) {
        VmafModel *model = NULL;
        VmafModelConfig cfg = { 0 };
        int err = vmaf_model_load(&model, &cfg,
                                  builtin_float_single_versions[v]);
        mu_assert("vmaf_model_load failed in float feature dependency test",
                  !err);

        for (unsigned i = 0; i < model->n_features; i++) {
            VmafFeatureExtractor *fex =
                vmaf_get_feature_extractor_by_feature_name(
                    model->feature[i].name, 0);
            mu_assert("feature extractor not found for float model feature",
                      fex != NULL);
        }
        vmaf_model_destroy(model);
    }

    /* Float collection models */
    for (unsigned v = 0; v < BUILTIN_FLOAT_COLLECTION_COUNT; v++) {
        VmafModel *model = NULL;
        VmafModelCollection *mc = NULL;
        VmafModelConfig cfg = { 0 };
        int err = vmaf_model_collection_load(
            &model, &mc, &cfg, builtin_float_collection_versions[v]);
        mu_assert("float collection load failed in feature dep test", !err);

        for (unsigned i = 0; i < model->n_features; i++) {
            VmafFeatureExtractor *fex =
                vmaf_get_feature_extractor_by_feature_name(
                    model->feature[i].name, 0);
            mu_assert("feature extractor not found for float collection",
                      fex != NULL);
        }
        vmaf_model_collection_destroy(mc);
    }

    return NULL;
}
#endif

/* ===================================================================
 * 3.2.2: Feature Registration Tests
 * =================================================================== */

static char *test_feature_registration(void)
{
    VmafConfiguration vmaf_cfg = {
        .log_level = VMAF_LOG_LEVEL_NONE,
        .n_threads = 1,
    };
    VmafContext *vmaf = NULL;
    int err = vmaf_init(&vmaf, vmaf_cfg);
    mu_assert("vmaf_init failed", !err);

    for (unsigned v = 0; v < BUILTIN_SINGLE_COUNT; v++) {
        VmafModel *model = NULL;
        VmafModelConfig cfg = { 0 };
        err = vmaf_model_load(&model, &cfg, builtin_single_versions[v]);
        mu_assert("vmaf_model_load failed in feature registration test", !err);

        err = vmaf_use_features_from_model(vmaf, model);
        mu_assert("vmaf_use_features_from_model failed", !err);

        vmaf_model_destroy(model);
    }

    vmaf_close(vmaf);
    return NULL;
}

#if VMAF_FLOAT_FEATURES
static char *test_feature_registration_float(void)
{
    VmafConfiguration vmaf_cfg = {
        .log_level = VMAF_LOG_LEVEL_NONE,
        .n_threads = 1,
    };
    VmafContext *vmaf = NULL;
    int err = vmaf_init(&vmaf, vmaf_cfg);
    mu_assert("vmaf_init failed", !err);

    for (unsigned v = 0; v < BUILTIN_FLOAT_SINGLE_COUNT; v++) {
        VmafModel *model = NULL;
        VmafModelConfig cfg = { 0 };
        err = vmaf_model_load(&model, &cfg,
                              builtin_float_single_versions[v]);
        mu_assert("vmaf_model_load failed in float feature registration test",
                  !err);

        err = vmaf_use_features_from_model(vmaf, model);
        mu_assert("vmaf_use_features_from_model failed for float model", !err);

        vmaf_model_destroy(model);
    }

    vmaf_close(vmaf);
    return NULL;
}
#endif

/* ===================================================================
 * 3.2.3: Expected Feature Names Tests
 * =================================================================== */

static char *test_expected_feature_names_integer(void)
{
    for (unsigned v = 0; v < BUILTIN_SINGLE_COUNT; v++) {
        VmafModel *model = NULL;
        VmafModelConfig cfg = { 0 };
        int err = vmaf_model_load(&model, &cfg, builtin_single_versions[v]);
        mu_assert("vmaf_model_load failed", !err);
        mu_assert("expected 6 features", model->n_features == 6);

        for (unsigned i = 0; i < 6; i++) {
            mu_assert("integer feature name mismatch",
                      !strcmp(model->feature[i].name,
                              integer_feature_names[i]));
        }
        vmaf_model_destroy(model);
    }
    return NULL;
}

#if VMAF_FLOAT_FEATURES
static char *test_expected_feature_names_float(void)
{
    for (unsigned v = 0; v < BUILTIN_FLOAT_SINGLE_COUNT; v++) {
        VmafModel *model = NULL;
        VmafModelConfig cfg = { 0 };
        int err = vmaf_model_load(&model, &cfg,
                                  builtin_float_single_versions[v]);
        mu_assert("vmaf_model_load failed", !err);
        mu_assert("expected 6 features", model->n_features == 6);

        for (unsigned i = 0; i < 6; i++) {
            mu_assert("float feature name mismatch",
                      !strcmp(model->feature[i].name,
                              float_feature_names[i]));
        }
        vmaf_model_destroy(model);
    }
    return NULL;
}
#endif

/* ===================================================================
 * 3.3: Round-Trip Scoring Tests
 * =================================================================== */

static char *roundtrip_score_helper(const char *version)
{
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
    mu_assert("vmaf_model_load failed in round-trip test", !err);

    err = vmaf_use_features_from_model(vmaf, model);
    mu_assert("vmaf_use_features_from_model failed in round-trip test", !err);

    VmafPicture ref, dist;
    err = vmaf_picture_alloc(&ref, VMAF_PIX_FMT_YUV420P, 8, 64, 64);
    mu_assert("ref alloc failed", !err);
    err = vmaf_picture_alloc(&dist, VMAF_PIX_FMT_YUV420P, 8, 64, 64);
    mu_assert("dist alloc failed", !err);

    /* Fill planes: ref=128, dist=96 */
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
    mu_assert("score out of range [0, 100]",
              score >= 0.0 && score <= 100.0);

    vmaf_model_destroy(model);
    vmaf_close(vmaf);
    return NULL;
}

static char *test_roundtrip_vmaf_v0_6_1(void)
{
    return roundtrip_score_helper("vmaf_v0.6.1");
}

static char *test_roundtrip_vmaf_v0_6_1neg(void)
{
    return roundtrip_score_helper("vmaf_v0.6.1neg");
}

static char *test_roundtrip_vmaf_4k_v0_6_1(void)
{
    return roundtrip_score_helper("vmaf_4k_v0.6.1");
}

static char *test_roundtrip_vmaf_4k_v0_6_1neg(void)
{
    return roundtrip_score_helper("vmaf_4k_v0.6.1neg");
}

#if VMAF_FLOAT_FEATURES
static char *test_roundtrip_float_v0_6_1(void)
{
    return roundtrip_score_helper("vmaf_float_v0.6.1");
}

static char *test_roundtrip_float_v0_6_1neg(void)
{
    return roundtrip_score_helper("vmaf_float_v0.6.1neg");
}

static char *test_roundtrip_float_4k_v0_6_1(void)
{
    return roundtrip_score_helper("vmaf_float_4k_v0.6.1");
}
#endif

/* Round-trip for collection models */
static char *roundtrip_collection_helper(const char *version,
                                          unsigned expected_sub_models)
{
    (void)expected_sub_models;
    int err;
    VmafConfiguration vmaf_cfg = {
        .log_level = VMAF_LOG_LEVEL_NONE,
        .n_threads = 1,
    };
    VmafContext *vmaf = NULL;
    err = vmaf_init(&vmaf, vmaf_cfg);
    mu_assert("vmaf_init failed", !err);

    VmafModel *model = NULL;
    VmafModelCollection *mc = NULL;
    VmafModelConfig model_cfg = { 0 };
    err = vmaf_model_collection_load(&model, &mc, &model_cfg, version);
    mu_assert("vmaf_model_collection_load failed in round-trip test", !err);

    err = vmaf_use_features_from_model_collection(vmaf, mc);
    mu_assert("vmaf_use_features_from_model_collection failed", !err);

    VmafPicture ref, dist;
    err = vmaf_picture_alloc(&ref, VMAF_PIX_FMT_YUV420P, 8, 64, 64);
    mu_assert("ref alloc failed", !err);
    err = vmaf_picture_alloc(&dist, VMAF_PIX_FMT_YUV420P, 8, 64, 64);
    mu_assert("dist alloc failed", !err);

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

    VmafModelCollectionScore mc_score;
    err = vmaf_score_at_index_model_collection(vmaf, mc, &mc_score, 0);
    mu_assert("vmaf_score_at_index_model_collection failed", !err);
    mu_assert("bagging score out of range [0, 100]",
              mc_score.bootstrap.bagging_score >= 0.0 &&
              mc_score.bootstrap.bagging_score <= 100.0);
    mu_assert("stddev should be non-negative",
              mc_score.bootstrap.stddev >= 0.0);
    mu_assert("CI lo <= bagging score",
              mc_score.bootstrap.ci.p95.lo <=
              mc_score.bootstrap.bagging_score);
    mu_assert("bagging score <= CI hi",
              mc_score.bootstrap.bagging_score <=
              mc_score.bootstrap.ci.p95.hi);

    vmaf_model_collection_destroy(mc);
    vmaf_close(vmaf);
    return NULL;
}

static char *test_roundtrip_vmaf_b_v0_6_3(void)
{
    return roundtrip_collection_helper("vmaf_b_v0.6.3", 21);
}

#if VMAF_FLOAT_FEATURES
static char *test_roundtrip_float_b_v0_6_3(void)
{
    return roundtrip_collection_helper("vmaf_float_b_v0.6.3", 21);
}
#endif

/* ===================================================================
 * 3.3.3: Identity Score Tests (ref == dist)
 * =================================================================== */

static char *identity_score_helper(const char *version)
{
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
    mu_assert("vmaf_model_load failed in identity test", !err);

    err = vmaf_use_features_from_model(vmaf, model);
    mu_assert("vmaf_use_features_from_model failed in identity test", !err);

    VmafPicture ref, dist;
    err = vmaf_picture_alloc(&ref, VMAF_PIX_FMT_YUV420P, 8, 64, 64);
    mu_assert("ref alloc failed", !err);
    err = vmaf_picture_alloc(&dist, VMAF_PIX_FMT_YUV420P, 8, 64, 64);
    mu_assert("dist alloc failed", !err);

    /* Fill both with same value: identical pictures */
    memset(ref.data[0], 128, ref.stride[0] * 64);
    memset(ref.data[1], 128, ref.stride[1] * 32);
    memset(ref.data[2], 128, ref.stride[2] * 32);
    memset(dist.data[0], 128, dist.stride[0] * 64);
    memset(dist.data[1], 128, dist.stride[1] * 32);
    memset(dist.data[2], 128, dist.stride[2] * 32);

    err = vmaf_read_pictures(vmaf, &ref, &dist, 0);
    mu_assert("vmaf_read_pictures failed", !err);
    err = vmaf_read_pictures(vmaf, NULL, NULL, 0);
    mu_assert("vmaf_read_pictures flush failed", !err);

    double score;
    err = vmaf_score_at_index(vmaf, model, &score, 0);
    mu_assert("vmaf_score_at_index failed in identity test", !err);
    mu_assert("identity score should be >= 95.0", score >= 95.0);

    vmaf_model_destroy(model);
    vmaf_close(vmaf);
    return NULL;
}

static char *test_identity_vmaf_v0_6_1(void)
{
    return identity_score_helper("vmaf_v0.6.1");
}

static char *test_identity_vmaf_v0_6_1neg(void)
{
    return identity_score_helper("vmaf_v0.6.1neg");
}

static char *test_identity_vmaf_4k_v0_6_1(void)
{
    return identity_score_helper("vmaf_4k_v0.6.1");
}

static char *test_identity_vmaf_4k_v0_6_1neg(void)
{
    return identity_score_helper("vmaf_4k_v0.6.1neg");
}

/* Identity test for bootstrap collection models */
static char *identity_collection_helper(const char *version)
{
    int err;
    VmafConfiguration vmaf_cfg = {
        .log_level = VMAF_LOG_LEVEL_NONE,
        .n_threads = 1,
    };
    VmafContext *vmaf = NULL;
    err = vmaf_init(&vmaf, vmaf_cfg);
    mu_assert("vmaf_init failed", !err);

    VmafModel *model = NULL;
    VmafModelCollection *mc = NULL;
    VmafModelConfig model_cfg = { 0 };
    err = vmaf_model_collection_load(&model, &mc, &model_cfg, version);
    mu_assert("vmaf_model_collection_load failed in identity test", !err);

    err = vmaf_use_features_from_model_collection(vmaf, mc);
    mu_assert("vmaf_use_features_from_model_collection failed", !err);

    VmafPicture ref, dist;
    err = vmaf_picture_alloc(&ref, VMAF_PIX_FMT_YUV420P, 8, 64, 64);
    mu_assert("ref alloc failed", !err);
    err = vmaf_picture_alloc(&dist, VMAF_PIX_FMT_YUV420P, 8, 64, 64);
    mu_assert("dist alloc failed", !err);

    memset(ref.data[0], 128, ref.stride[0] * 64);
    memset(ref.data[1], 128, ref.stride[1] * 32);
    memset(ref.data[2], 128, ref.stride[2] * 32);
    memset(dist.data[0], 128, dist.stride[0] * 64);
    memset(dist.data[1], 128, dist.stride[1] * 32);
    memset(dist.data[2], 128, dist.stride[2] * 32);

    err = vmaf_read_pictures(vmaf, &ref, &dist, 0);
    mu_assert("vmaf_read_pictures failed", !err);
    err = vmaf_read_pictures(vmaf, NULL, NULL, 0);
    mu_assert("vmaf_read_pictures flush failed", !err);

    VmafModelCollectionScore mc_score;
    err = vmaf_score_at_index_model_collection(vmaf, mc, &mc_score, 0);
    mu_assert("vmaf_score_at_index_model_collection failed", !err);
    mu_assert("identity bagging score should be >= 95.0",
              mc_score.bootstrap.bagging_score >= 95.0);

    vmaf_model_collection_destroy(mc);
    vmaf_close(vmaf);
    return NULL;
}

static char *test_identity_vmaf_b_v0_6_3(void)
{
    return identity_collection_helper("vmaf_b_v0.6.3");
}

#if VMAF_FLOAT_FEATURES
static char *test_identity_float_v0_6_1(void)
{
    return identity_score_helper("vmaf_float_v0.6.1");
}

static char *test_identity_float_v0_6_1neg(void)
{
    return identity_score_helper("vmaf_float_v0.6.1neg");
}

static char *test_identity_float_4k_v0_6_1(void)
{
    return identity_score_helper("vmaf_float_4k_v0.6.1");
}

static char *test_identity_float_b_v0_6_3(void)
{
    return identity_collection_helper("vmaf_float_b_v0.6.3");
}
#endif

/* ===================================================================
 * 3.3.4: Model Collection Round-Trip (file-based)
 * =================================================================== */

static char *test_collection_roundtrip_rb_v0_6_2(void)
{
    int err;
    VmafConfiguration vmaf_cfg = {
        .log_level = VMAF_LOG_LEVEL_NONE,
        .n_threads = 1,
    };
    VmafContext *vmaf = NULL;
    err = vmaf_init(&vmaf, vmaf_cfg);
    mu_assert("vmaf_init failed", !err);

    VmafModel *model = NULL;
    VmafModelCollection *mc = NULL;
    VmafModelConfig model_cfg = { 0 };
    const char *path =
        JSON_MODEL_PATH "vmaf_rb_v0.6.2/vmaf_rb_v0.6.2.json";
    err = vmaf_model_collection_load_from_path(&model, &mc, &model_cfg, path);
    mu_assert("vmaf_model_collection_load_from_path failed", !err);

    err = vmaf_use_features_from_model_collection(vmaf, mc);
    mu_assert("vmaf_use_features_from_model_collection failed", !err);

    VmafPicture ref, dist;
    err = vmaf_picture_alloc(&ref, VMAF_PIX_FMT_YUV420P, 8, 64, 64);
    mu_assert("ref alloc failed", !err);
    err = vmaf_picture_alloc(&dist, VMAF_PIX_FMT_YUV420P, 8, 64, 64);
    mu_assert("dist alloc failed", !err);

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

    VmafModelCollectionScore mc_score;
    err = vmaf_score_at_index_model_collection(vmaf, mc, &mc_score, 0);
    mu_assert("vmaf_score_at_index_model_collection failed", !err);
    mu_assert("bagging score out of range [0, 100]",
              mc_score.bootstrap.bagging_score >= 0.0 &&
              mc_score.bootstrap.bagging_score <= 100.0);
    mu_assert("stddev should be non-negative",
              mc_score.bootstrap.stddev >= 0.0);
    mu_assert("CI lo <= bagging score",
              mc_score.bootstrap.ci.p95.lo <=
              mc_score.bootstrap.bagging_score);
    mu_assert("bagging score <= CI hi",
              mc_score.bootstrap.bagging_score <=
              mc_score.bootstrap.ci.p95.hi);

    vmaf_model_collection_destroy(mc);
    vmaf_close(vmaf);
    return NULL;
}

/* ===================================================================
 * 3.4: Version Compatibility Tests
 * =================================================================== */

static char *test_load_legacy_json_models(void)
{
    for (unsigned i = 0; i < LEGACY_MODEL_COUNT; i++) {
        VmafModel *model = NULL;
        VmafModelConfig cfg = { 0 };
        int err = vmaf_model_load_from_path(&model, &cfg,
                                            legacy_json_model_paths[i]);
        mu_assert("legacy model load failed", !err);
        mu_assert("legacy model is NULL", model != NULL);
        mu_assert("legacy model n_features should be > 0",
                  model->n_features > 0);
        vmaf_model_destroy(model);
    }
    return NULL;
}

static char *test_model_type_backward_compat(void)
{
    int err;

    /* SVM_NUSVR */
    VmafModel *m1 = NULL;
    VmafModelConfig cfg1 = { 0 };
    err = vmaf_model_load(&m1, &cfg1, "vmaf_v0.6.1");
    mu_assert("load vmaf_v0.6.1 failed", !err);
    mu_assert("vmaf_v0.6.1 should be SVM_NUSVR",
              m1->type == VMAF_MODEL_TYPE_SVM_NUSVR);
    vmaf_model_destroy(m1);

    /* BOOTSTRAP_SVM_NUSVR (via collection load) */
    VmafModel *m2 = NULL;
    VmafModelCollection *mc2 = NULL;
    VmafModelConfig cfg2 = { 0 };
    err = vmaf_model_collection_load(&m2, &mc2, &cfg2, "vmaf_b_v0.6.3");
    mu_assert("load vmaf_b_v0.6.3 collection failed", !err);
    mu_assert("vmaf_b_v0.6.3 should be BOOTSTRAP_SVM_NUSVR",
              mc2->type == VMAF_MODEL_BOOTSTRAP_SVM_NUSVR);
    vmaf_model_collection_destroy(mc2);

    /* RESIDUE_BOOTSTRAP_SVM_NUSVR (collection from path) */
    VmafModel *m3 = NULL;
    VmafModelCollection *mc3 = NULL;
    VmafModelConfig cfg3 = { 0 };
    const char *path = JSON_MODEL_PATH "vmaf_rb_v0.6.2/vmaf_rb_v0.6.2.json";
    err = vmaf_model_collection_load_from_path(&m3, &mc3, &cfg3, path);
    mu_assert("load vmaf_rb_v0.6.2 collection failed", !err);
    mu_assert("vmaf_rb_v0.6.2 collection should be RESIDUE_BOOTSTRAP",
              mc3->type == VMAF_MODEL_RESIDUE_BOOTSTRAP_SVM_NUSVR);
    vmaf_model_collection_destroy(mc3);

    return NULL;
}

static char *test_normalization_type_compat(void)
{
    int err;

    /* LINEAR_RESCALE */
    VmafModel *m1 = NULL;
    VmafModelConfig cfg1 = { 0 };
    err = vmaf_model_load_from_path(&m1, &cfg1,
                                    JSON_MODEL_PATH "vmaf_v0.6.1.json");
    mu_assert("load vmaf_v0.6.1.json failed", !err);
    mu_assert("vmaf_v0.6.1.json should have LINEAR_RESCALE normalization",
              m1->norm_type == VMAF_MODEL_NORMALIZATION_TYPE_LINEAR_RESCALE);
    vmaf_model_destroy(m1);

    /* NONE */
    VmafModel *m2 = NULL;
    VmafModelConfig cfg2 = { 0 };
    err = vmaf_model_load_from_path(
        &m2, &cfg2,
        JSON_MODEL_PATH "other_models/nflxtrain_norm_type_none.json");
    mu_assert("load nflxtrain_norm_type_none.json failed", !err);
    mu_assert("nflxtrain_norm_type_none should have NONE normalization",
              m2->norm_type == VMAF_MODEL_NORMALIZATION_TYPE_NONE);
    vmaf_model_destroy(m2);

    return NULL;
}

/* ===================================================================
 * 3.5: Error Handling Tests
 * =================================================================== */

static char *test_load_nonexistent_path(void)
{
    VmafModel *model = NULL;
    VmafModelConfig cfg = { 0 };
    int err = vmaf_model_load_from_path(&model, &cfg,
                                        "/nonexistent/path/model.json");
    mu_assert("loading nonexistent path should fail", err != 0);
    mu_assert("model should be NULL on failure", model == NULL);
    return NULL;
}

static char *test_load_nonexistent_builtin(void)
{
    VmafModel *model = NULL;
    VmafModelConfig cfg = { 0 };
    int err = vmaf_model_load(&model, &cfg, "vmaf_nonexistent_v99.99");
    mu_assert("loading nonexistent version should fail", err != 0);
    return NULL;
}

static char *test_load_corrupt_json(void)
{
#ifdef _WIN32
    const char *corrupt_path = "vmaf_corrupt_model.json";
#else
    const char *corrupt_path = "/tmp/vmaf_corrupt_model.json";
#endif
    FILE *f = fopen(corrupt_path, "w");
    mu_assert("could not create temp file for corrupt JSON", f != NULL);
    fprintf(f, "{ this is not valid json !!!");
    fclose(f);

    VmafModel *model = NULL;
    VmafModelConfig cfg = { 0 };
    int err = vmaf_model_load_from_path(&model, &cfg, corrupt_path);
    mu_assert("loading corrupt JSON should fail", err != 0);

    remove(corrupt_path);
    return NULL;
}

static char *test_load_truncated_json(void)
{
    const char *src_path = JSON_MODEL_PATH "vmaf_v0.6.1.json";
    FILE *src = fopen(src_path, "rb");
    mu_assert("could not open source model for truncation test", src != NULL);

    char buf[100];
    size_t nread = fread(buf, 1, sizeof(buf), src);
    fclose(src);
    mu_assert("could not read from source model", nread == sizeof(buf));

#ifdef _WIN32
    const char *trunc_path = "vmaf_truncated_model.json";
#else
    const char *trunc_path = "/tmp/vmaf_truncated_model.json";
#endif
    FILE *dst = fopen(trunc_path, "wb");
    mu_assert("could not create truncated file", dst != NULL);
    fwrite(buf, 1, nread, dst);
    fclose(dst);

    VmafModel *model = NULL;
    VmafModelConfig cfg = { 0 };
    int err = vmaf_model_load_from_path(&model, &cfg, trunc_path);
    mu_assert("loading truncated JSON should fail", err != 0);

    remove(trunc_path);
    return NULL;
}

static char *test_load_empty_file(void)
{
#ifdef _WIN32
    const char *empty_path = "vmaf_empty_model.json";
#else
    const char *empty_path = "/tmp/vmaf_empty_model.json";
#endif
    FILE *f = fopen(empty_path, "w");
    mu_assert("could not create empty file", f != NULL);
    fclose(f);

    VmafModel *model = NULL;
    VmafModelConfig cfg = { 0 };
    int err = vmaf_model_load_from_path(&model, &cfg, empty_path);
    mu_assert("loading empty file should fail", err != 0);

    remove(empty_path);
    return NULL;
}

static char *test_load_pkl_rejected(void)
{
    VmafModel *model = NULL;
    VmafModelConfig cfg = { 0 };
    const char *pkl_path =
        JSON_MODEL_PATH "other_models/vmaf_v0.6.0.pkl";
    int err = vmaf_model_load_from_path(&model, &cfg, pkl_path);
    mu_assert("loading pkl file should fail", err != 0);
    return NULL;
}

static char *test_null_arguments(void)
{
    /* vmaf_model_destroy(NULL) should be a no-op */
    vmaf_model_destroy(NULL);

    /* vmaf_model_collection_destroy(NULL) should be a no-op */
    vmaf_model_collection_destroy(NULL);

    /* vmaf_model_feature_overload with NULL model */
    int err = vmaf_model_feature_overload(NULL, "adm", NULL);
    mu_assert("overload with NULL model should fail", err != 0);

    return NULL;
}

/* ===================================================================
 * 3.6: Model Configuration Tests
 * =================================================================== */

static char *test_model_flags(void)
{
    int err;

    /* Default flags: clip enabled, transform disabled */
    VmafModel *m1 = NULL;
    VmafModelConfig cfg1 = { .flags = VMAF_MODEL_FLAGS_DEFAULT };
    err = vmaf_model_load(&m1, &cfg1, "vmaf_v0.6.1");
    mu_assert("load failed", !err);
    mu_assert("clip should be enabled with DEFAULT flags",
              m1->score_clip.enabled);
    mu_assert("transform should be disabled with DEFAULT flags",
              !m1->score_transform.enabled);
    vmaf_model_destroy(m1);

    /* DISABLE_CLIP */
    VmafModel *m2 = NULL;
    VmafModelConfig cfg2 = { .flags = VMAF_MODEL_FLAG_DISABLE_CLIP };
    err = vmaf_model_load(&m2, &cfg2, "vmaf_v0.6.1");
    mu_assert("load failed", !err);
    mu_assert("clip should be disabled with DISABLE_CLIP flag",
              !m2->score_clip.enabled);
    vmaf_model_destroy(m2);

    /* ENABLE_TRANSFORM */
    VmafModel *m3 = NULL;
    VmafModelConfig cfg3 = { .flags = VMAF_MODEL_FLAG_ENABLE_TRANSFORM };
    err = vmaf_model_load(&m3, &cfg3, "vmaf_v0.6.1");
    mu_assert("load failed", !err);
    mu_assert("transform should be enabled with ENABLE_TRANSFORM flag",
              m3->score_transform.enabled);
    vmaf_model_destroy(m3);

    /* DISABLE_TRANSFORM */
    VmafModel *m4 = NULL;
    VmafModelConfig cfg4 = { .flags = VMAF_MODEL_FLAG_DISABLE_TRANSFORM };
    err = vmaf_model_load(&m4, &cfg4, "vmaf_v0.6.1");
    mu_assert("load failed", !err);
    mu_assert("transform should be disabled with DISABLE_TRANSFORM flag",
              !m4->score_transform.enabled);
    vmaf_model_destroy(m4);

    return NULL;
}

static char *test_custom_model_name(void)
{
    VmafModel *model = NULL;
    VmafModelConfig cfg = { .name = "my_custom_vmaf" };
    int err = vmaf_model_load(&model, &cfg, "vmaf_v0.6.1");
    mu_assert("load failed", !err);
    mu_assert("name should match custom name",
              !strcmp(model->name, "my_custom_vmaf"));
    vmaf_model_destroy(model);
    return NULL;
}

static char *test_feature_overload(void)
{
    int err;

    VmafModel *model = NULL;
    VmafModelConfig cfg = { 0 };
    err = vmaf_model_load(&model, &cfg, "vmaf_v0.6.1");
    mu_assert("load failed", !err);

    /* Verify feature 0 starts with NULL opts_dict */
    mu_assert("feature 0 should be adm2",
              !strcmp(model->feature[0].name,
                      "VMAF_integer_feature_adm2_score"));
    mu_assert("feature 0 opts_dict should be NULL before overload",
              !model->feature[0].opts_dict);

    /* Overload */
    VmafFeatureDictionary *dict = NULL;
    err = vmaf_feature_dictionary_set(&dict, "adm_enhn_gain_limit", "1.0");
    mu_assert("vmaf_feature_dictionary_set failed", !err);

    err = vmaf_model_feature_overload(model, "adm", dict);
    mu_assert("vmaf_model_feature_overload failed", !err);

    /* Verify overload took effect */
    mu_assert("feature 0 opts_dict should be non-NULL after overload",
              model->feature[0].opts_dict != NULL);

    const VmafDictionaryEntry *e =
        vmaf_dictionary_get(&model->feature[0].opts_dict,
                            "adm_enhn_gain_limit", 0);
    mu_assert("opts_dict should have adm_enhn_gain_limit key", e != NULL);
    /* vmaf_feature_dictionary_set normalizes "1.0" to "1" via %g format */
    mu_assert("adm_enhn_gain_limit value should be '1'",
              !strcmp(e->val, "1"));

    vmaf_model_destroy(model);
    return NULL;
}

/* ===================================================================
 * 3.7: Neg Model Option Validation
 * =================================================================== */

static char *test_neg_model_opts_vmaf_v0_6_1neg(void)
{
    VmafModel *model = NULL;
    VmafModelConfig cfg = { 0 };
    int err = vmaf_model_load(&model, &cfg, "vmaf_v0.6.1neg");
    mu_assert("load vmaf_v0.6.1neg failed", !err);

    /* Feature 0 (adm2): adm_enhn_gain_limit = "1" */
    mu_assert("neg model feature 0 opts_dict should be non-NULL",
              model->feature[0].opts_dict != NULL);
    const VmafDictionaryEntry *e0 =
        vmaf_dictionary_get(&model->feature[0].opts_dict,
                            "adm_enhn_gain_limit", 0);
    mu_assert("feature 0 should have adm_enhn_gain_limit", e0 != NULL);
    mu_assert("adm_enhn_gain_limit should be '1'",
              !strcmp(e0->val, "1"));

    /* Feature 1 (motion2): NULL opts_dict */
    mu_assert("neg model feature 1 opts_dict should be NULL",
              !model->feature[1].opts_dict);

    /* Features 2-5 (vif_scale0-3): vif_enhn_gain_limit = "1" */
    for (unsigned i = 2; i <= 5; i++) {
        mu_assert("neg model vif feature opts_dict should be non-NULL",
                  model->feature[i].opts_dict != NULL);
        const VmafDictionaryEntry *e =
            vmaf_dictionary_get(&model->feature[i].opts_dict,
                                "vif_enhn_gain_limit", 0);
        mu_assert("vif feature should have vif_enhn_gain_limit", e != NULL);
        mu_assert("vif_enhn_gain_limit should be '1'",
                  !strcmp(e->val, "1"));
    }

    vmaf_model_destroy(model);
    return NULL;
}

static char *test_neg_model_opts_vmaf_4k_v0_6_1neg(void)
{
    VmafModel *model = NULL;
    VmafModelConfig cfg = { 0 };
    int err = vmaf_model_load(&model, &cfg, "vmaf_4k_v0.6.1neg");
    mu_assert("load vmaf_4k_v0.6.1neg failed", !err);

    /* Feature 0 (adm2): adm_enhn_gain_limit = "1" */
    mu_assert("4k neg model feature 0 opts_dict should be non-NULL",
              model->feature[0].opts_dict != NULL);
    const VmafDictionaryEntry *e0 =
        vmaf_dictionary_get(&model->feature[0].opts_dict,
                            "adm_enhn_gain_limit", 0);
    mu_assert("feature 0 should have adm_enhn_gain_limit", e0 != NULL);
    mu_assert("adm_enhn_gain_limit should be '1'",
              !strcmp(e0->val, "1"));

    /* Feature 1 (motion2): NULL opts_dict */
    mu_assert("4k neg model feature 1 opts_dict should be NULL",
              !model->feature[1].opts_dict);

    /* Features 2-5 (vif_scale0-3): vif_enhn_gain_limit = "1" */
    for (unsigned i = 2; i <= 5; i++) {
        mu_assert("4k neg model vif feature opts_dict should be non-NULL",
                  model->feature[i].opts_dict != NULL);
        const VmafDictionaryEntry *e =
            vmaf_dictionary_get(&model->feature[i].opts_dict,
                                "vif_enhn_gain_limit", 0);
        mu_assert("vif feature should have vif_enhn_gain_limit", e != NULL);
        mu_assert("vif_enhn_gain_limit should be '1'",
                  !strcmp(e->val, "1"));
    }

    vmaf_model_destroy(model);
    return NULL;
}

/* ===================================================================
 * run_tests: Test runner
 * =================================================================== */

char *run_tests(void)
{
    /* 3.1.1: Built-in model loading (single + collection) */
    mu_run_test(test_load_all_builtin_single_models);
    mu_run_test(test_load_all_builtin_collection_models);
#if VMAF_FLOAT_FEATURES
    mu_run_test(test_load_all_builtin_float_single_models);
    mu_run_test(test_load_all_builtin_float_collection_models);
#endif

    /* 3.1.2: File-based model loading */
    mu_run_test(test_load_all_json_single_models);
    mu_run_test(test_load_json_collection_models);

    /* 3.1.3: Model collection loading (5 collections) */
    mu_run_test(test_load_model_collection_b_v0_6_3);
    mu_run_test(test_load_model_collection_rb_v0_6_2);
    mu_run_test(test_load_model_collection_rb_v0_6_3);
    mu_run_test(test_load_model_collection_4k_rb_v0_6_2);
#if VMAF_FLOAT_FEATURES
    mu_run_test(test_load_model_collection_float_b_v0_6_3);
#endif

    /* 3.1.4: Built-in vs file equivalence */
    mu_run_test(test_builtin_vs_file_equivalence);
#if VMAF_FLOAT_FEATURES
    mu_run_test(test_builtin_vs_file_equivalence_float);
#endif

    /* 3.2.1: Feature dependency resolution */
    mu_run_test(test_feature_dependency_resolution);
#if VMAF_FLOAT_FEATURES
    mu_run_test(test_feature_dependency_resolution_float);
#endif

    /* 3.2.2: Feature registration */
    mu_run_test(test_feature_registration);
#if VMAF_FLOAT_FEATURES
    mu_run_test(test_feature_registration_float);
#endif

    /* 3.2.3: Expected feature names */
    mu_run_test(test_expected_feature_names_integer);
#if VMAF_FLOAT_FEATURES
    mu_run_test(test_expected_feature_names_float);
#endif

    /* 3.3.1/3.3.2: Round-trip scoring (all 9 primary models) */
    mu_run_test(test_roundtrip_vmaf_v0_6_1);
    mu_run_test(test_roundtrip_vmaf_v0_6_1neg);
    mu_run_test(test_roundtrip_vmaf_4k_v0_6_1);
    mu_run_test(test_roundtrip_vmaf_4k_v0_6_1neg);
    mu_run_test(test_roundtrip_vmaf_b_v0_6_3);
#if VMAF_FLOAT_FEATURES
    mu_run_test(test_roundtrip_float_v0_6_1);
    mu_run_test(test_roundtrip_float_v0_6_1neg);
    mu_run_test(test_roundtrip_float_4k_v0_6_1);
    mu_run_test(test_roundtrip_float_b_v0_6_3);
#endif

    /* 3.3.3: Identity score (all 9 primary models) */
    mu_run_test(test_identity_vmaf_v0_6_1);
    mu_run_test(test_identity_vmaf_v0_6_1neg);
    mu_run_test(test_identity_vmaf_4k_v0_6_1);
    mu_run_test(test_identity_vmaf_4k_v0_6_1neg);
    mu_run_test(test_identity_vmaf_b_v0_6_3);
#if VMAF_FLOAT_FEATURES
    mu_run_test(test_identity_float_v0_6_1);
    mu_run_test(test_identity_float_v0_6_1neg);
    mu_run_test(test_identity_float_4k_v0_6_1);
    mu_run_test(test_identity_float_b_v0_6_3);
#endif

    /* 3.3.4: Model collection round-trip (uses float model vmaf_rb_v0.6.2) */
#if VMAF_FLOAT_FEATURES
    mu_run_test(test_collection_roundtrip_rb_v0_6_2);
#endif

    /* 3.4: Version compatibility */
    mu_run_test(test_load_legacy_json_models);
    mu_run_test(test_model_type_backward_compat);
    mu_run_test(test_normalization_type_compat);

    /* 3.5: Error handling (7+ tests) */
    mu_run_test(test_load_nonexistent_path);
    mu_run_test(test_load_nonexistent_builtin);
    mu_run_test(test_load_corrupt_json);
    mu_run_test(test_load_truncated_json);
    mu_run_test(test_load_empty_file);
    mu_run_test(test_load_pkl_rejected);
    mu_run_test(test_null_arguments);

    /* 3.6: Model configuration */
    mu_run_test(test_model_flags);
    mu_run_test(test_custom_model_name);
    mu_run_test(test_feature_overload);

    /* 3.7: Neg model option validation */
    mu_run_test(test_neg_model_opts_vmaf_v0_6_1neg);
    mu_run_test(test_neg_model_opts_vmaf_4k_v0_6_1neg);

    return NULL;
}
