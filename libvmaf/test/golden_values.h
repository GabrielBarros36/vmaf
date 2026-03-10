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

/**
 * Cross-Architecture Golden Value Definitions
 *
 * These golden values are canonical reference scores computed with
 * -Denable_asm=false (C reference code path, no SIMD). All SIMD-enabled
 * builds must reproduce these scores within the defined tolerances.
 *
 * To regenerate: build with -Denable_asm=false -Denable_float=true,
 * run the test, and update the values below.
 *
 * Test videos:  python/test/resource/yuv/
 * Models:       built-in (vmaf_model_load)
 */

#ifndef GOLDEN_VALUES_H_
#define GOLDEN_VALUES_H_

#include "libvmaf/picture.h"

/* ---- Tolerance constants (from spec Section 6.1) ---- */
#define TOL_INTEGER_FEAT  1e-6   /* VIF, ADM2, Motion2 integer paths */
#define TOL_FLOAT_FEAT    1e-4   /* float_ssim, float_ms_ssim */
#define TOL_VMAF_SCORE    1e-4   /* VMAF composite score */
#define TOL_PSNR          1e-4   /* PSNR (integer and float) */
#define TOL_CAMBI         1e-4   /* CAMBI */

/* ---- Score entry ---- */
typedef struct {
    const char *name;       /* Feature name as stored in feature collector */
    double golden_mean;     /* Expected mean score across all frames */
    double tolerance;       /* Absolute tolerance for comparison */
} GoldenScore;

/* ---- Test case definition ---- */
typedef struct {
    const char *id;               /* Unique test case identifier */
    const char *ref_file;         /* Reference YUV filename (relative to test_video_dir) */
    const char *dis_file;         /* Distorted YUV filename */
    unsigned width;
    unsigned height;
    unsigned bit_depth;
    enum VmafPixelFormat pix_fmt;
    const char *model_version;    /* Model version string for vmaf_model_load, or NULL */
    int is_model_collection;      /* 1 if model_version is a model collection */
    unsigned n_frames;            /* Expected number of frames */
    const GoldenScore *scores;
    unsigned n_scores;
    const char **extra_features;  /* Additional feature extractors to register */
    unsigned n_extra_features;
} GoldenTestCase;

/* ---- Extra features for primary 8-bit tests ---- */
static const char *extra_features_full[] = {
    "psnr", "float_ssim", "float_ms_ssim"
};

static const char *extra_features_cambi[] = {
    "cambi"
};

/* ==================================================================
 * Test Case 1: vmaf_8bit_576x324
 *   VMAF v0.6.1 on 8-bit 576x324, src01 ref vs. dis (48 frames)
 * ================================================================== */
static const GoldenScore golden_scores_vmaf_8bit[] = {
    /* VMAF composite score */
    { "golden_test",                                76.6689048244369,   TOL_VMAF_SCORE },
    /* Integer VIF per-scale */
    { "VMAF_integer_feature_vif_scale0_score",       0.363662071526051, TOL_INTEGER_FEAT },
    { "VMAF_integer_feature_vif_scale1_score",       0.767495281994343, TOL_INTEGER_FEAT },
    { "VMAF_integer_feature_vif_scale2_score",       0.863107771923145, TOL_INTEGER_FEAT },
    { "VMAF_integer_feature_vif_scale3_score",       0.915720089028279, TOL_INTEGER_FEAT },
    /* Integer ADM2 */
    { "VMAF_integer_feature_adm2_score",             0.9345057762924,   TOL_INTEGER_FEAT },
    /* Integer Motion2 */
    { "VMAF_integer_feature_motion2_score",          3.89534507195155,  TOL_INTEGER_FEAT },
    /* PSNR (Y, Cb, Cr) */
    { "psnr_y",                                     30.755064021049,    TOL_PSNR },
    { "psnr_cb",                                    38.4494410571618,   TOL_PSNR },
    { "psnr_cr",                                    40.991910248629,    TOL_PSNR },
    /* Float SSIM */
    { "float_ssim",                                  0.863226603716612, TOL_FLOAT_FEAT },
    /* Float MS-SSIM */
    { "float_ms_ssim",                               0.963240616895571, TOL_FLOAT_FEAT },
};

/* ==================================================================
 * Test Case 2: vmaf_8bit_identity
 *   VMAF v0.6.1 on 8-bit 576x324, ref vs. ref (48 frames)
 * ================================================================== */
static const GoldenScore golden_scores_vmaf_identity[] = {
    { "golden_test",                                99.9464266316011,   TOL_VMAF_SCORE },
    { "VMAF_integer_feature_adm2_score",             1.00000225228643,  TOL_INTEGER_FEAT },
    { "VMAF_integer_feature_vif_scale0_score",       0.999999955296516, TOL_INTEGER_FEAT },
    { "VMAF_integer_feature_vif_scale1_score",       0.999999555448691, TOL_INTEGER_FEAT },
    { "VMAF_integer_feature_vif_scale2_score",       0.999999310821295, TOL_INTEGER_FEAT },
    { "VMAF_integer_feature_vif_scale3_score",       0.999999190370242, TOL_INTEGER_FEAT },
    { "psnr_y",                                     60.0,               TOL_PSNR },
    { "psnr_cb",                                    60.0,               TOL_PSNR },
    { "psnr_cr",                                    60.0,               TOL_PSNR },
    { "float_ssim",                                  1.0,               TOL_FLOAT_FEAT },
    { "float_ms_ssim",                               1.0,               TOL_FLOAT_FEAT },
};

/* ==================================================================
 * Test Case 3: vmaf_12bit_576x324
 *   VMAF v0.6.1 on 12-bit 576x324, src01 ref vs. dis (3 frames)
 * ================================================================== */
static const GoldenScore golden_scores_vmaf_12bit[] = {
    { "golden_test",                                82.5652300478838,   TOL_VMAF_SCORE },
    { "VMAF_integer_feature_vif_scale0_score",       0.433089315891266, TOL_INTEGER_FEAT },
    { "VMAF_integer_feature_vif_scale1_score",       0.830613315105438, TOL_INTEGER_FEAT },
    { "VMAF_integer_feature_vif_scale2_score",       0.907212158044179, TOL_INTEGER_FEAT },
    { "VMAF_integer_feature_vif_scale3_score",       0.945895850658417, TOL_INTEGER_FEAT },
    { "VMAF_integer_feature_adm2_score",             0.951770445547814, TOL_INTEGER_FEAT },
    { "VMAF_integer_feature_motion2_score",          2.81045309702555,  TOL_INTEGER_FEAT },
};

/* ==================================================================
 * Test Case 4: vmaf_neg_8bit
 *   VMAF NEG (v0.6.1neg) on 8-bit 576x324 (48 frames)
 * ================================================================== */
static const GoldenScore golden_scores_vmaf_neg[] = {
    { "golden_test",                                75.0747291238163,   TOL_VMAF_SCORE },
};

/* ==================================================================
 * Test Case 5: vmaf_4k_8bit
 *   VMAF 4K on 8-bit 576x324 (48 frames)
 * ================================================================== */
static const GoldenScore golden_scores_vmaf_4k[] = {
    { "golden_test",                                84.9506472652734,   TOL_VMAF_SCORE },
};

/* ==================================================================
 * Test Case 6: vmaf_float_8bit
 *   VMAF float on 8-bit 576x324 (48 frames)
 * ================================================================== */
static const GoldenScore golden_scores_vmaf_float[] = {
    { "golden_test",                                76.6842971132081,   TOL_VMAF_SCORE },
};

/* ==================================================================
 * Test Case 7: vmaf_b_v063_8bit
 *   VMAF v0.6.3 (model collection) on 8-bit 576x324 (48 frames)
 *   Uses vmaf_model_collection_load
 * ================================================================== */
static const GoldenScore golden_scores_vmaf_b_v063[] = {
    { "golden_test_bagging",                        74.9363363308294,   TOL_VMAF_SCORE },
};

/* ==================================================================
 * Test Case 8: cambi_8bit
 *   CAMBI (no-ref) on 8-bit 576x324, src01 distorted (48 frames)
 * ================================================================== */
static const GoldenScore golden_scores_cambi[] = {
    { "cambi",                                       0.259684181909746, TOL_CAMBI },
};

/* ---- Master test case table ---- */
#define ARRAY_SIZE(a) (sizeof(a) / sizeof((a)[0]))

static const GoldenTestCase golden_test_cases[] = {
    {
        .id = "vmaf_8bit_576x324",
        .ref_file = "src01_hrc00_576x324.yuv",
        .dis_file = "src01_hrc01_576x324.yuv",
        .width = 576, .height = 324,
        .bit_depth = 8,
        .pix_fmt = VMAF_PIX_FMT_YUV420P,
        .model_version = "vmaf_v0.6.1",
        .is_model_collection = 0,
        .n_frames = 48,
        .scores = golden_scores_vmaf_8bit,
        .n_scores = ARRAY_SIZE(golden_scores_vmaf_8bit),
        .extra_features = extra_features_full,
        .n_extra_features = ARRAY_SIZE(extra_features_full),
    },
    {
        .id = "vmaf_8bit_identity",
        .ref_file = "src01_hrc00_576x324.yuv",
        .dis_file = "src01_hrc00_576x324.yuv",
        .width = 576, .height = 324,
        .bit_depth = 8,
        .pix_fmt = VMAF_PIX_FMT_YUV420P,
        .model_version = "vmaf_v0.6.1",
        .is_model_collection = 0,
        .n_frames = 48,
        .scores = golden_scores_vmaf_identity,
        .n_scores = ARRAY_SIZE(golden_scores_vmaf_identity),
        .extra_features = extra_features_full,
        .n_extra_features = ARRAY_SIZE(extra_features_full),
    },
    {
        .id = "vmaf_12bit_576x324",
        .ref_file = "src01_hrc00_576x324.yuv420p12le.yuv",
        .dis_file = "src01_hrc01_576x324.yuv420p12le.yuv",
        .width = 576, .height = 324,
        .bit_depth = 12,
        .pix_fmt = VMAF_PIX_FMT_YUV420P,
        .model_version = "vmaf_v0.6.1",
        .is_model_collection = 0,
        .n_frames = 3,
        .scores = golden_scores_vmaf_12bit,
        .n_scores = ARRAY_SIZE(golden_scores_vmaf_12bit),
        .extra_features = NULL,
        .n_extra_features = 0,
    },
    {
        .id = "vmaf_neg_8bit_576x324",
        .ref_file = "src01_hrc00_576x324.yuv",
        .dis_file = "src01_hrc01_576x324.yuv",
        .width = 576, .height = 324,
        .bit_depth = 8,
        .pix_fmt = VMAF_PIX_FMT_YUV420P,
        .model_version = "vmaf_v0.6.1neg",
        .is_model_collection = 0,
        .n_frames = 48,
        .scores = golden_scores_vmaf_neg,
        .n_scores = ARRAY_SIZE(golden_scores_vmaf_neg),
        .extra_features = NULL,
        .n_extra_features = 0,
    },
    {
        .id = "vmaf_4k_8bit_576x324",
        .ref_file = "src01_hrc00_576x324.yuv",
        .dis_file = "src01_hrc01_576x324.yuv",
        .width = 576, .height = 324,
        .bit_depth = 8,
        .pix_fmt = VMAF_PIX_FMT_YUV420P,
        .model_version = "vmaf_4k_v0.6.1",
        .is_model_collection = 0,
        .n_frames = 48,
        .scores = golden_scores_vmaf_4k,
        .n_scores = ARRAY_SIZE(golden_scores_vmaf_4k),
        .extra_features = NULL,
        .n_extra_features = 0,
    },
    {
        .id = "vmaf_float_8bit_576x324",
        .ref_file = "src01_hrc00_576x324.yuv",
        .dis_file = "src01_hrc01_576x324.yuv",
        .width = 576, .height = 324,
        .bit_depth = 8,
        .pix_fmt = VMAF_PIX_FMT_YUV420P,
        .model_version = "vmaf_float_v0.6.1",
        .is_model_collection = 0,
        .n_frames = 48,
        .scores = golden_scores_vmaf_float,
        .n_scores = ARRAY_SIZE(golden_scores_vmaf_float),
        .extra_features = NULL,
        .n_extra_features = 0,
    },
    {
        .id = "vmaf_b_v063_8bit_576x324",
        .ref_file = "src01_hrc00_576x324.yuv",
        .dis_file = "src01_hrc01_576x324.yuv",
        .width = 576, .height = 324,
        .bit_depth = 8,
        .pix_fmt = VMAF_PIX_FMT_YUV420P,
        .model_version = "vmaf_b_v0.6.3",
        .is_model_collection = 1,
        .n_frames = 48,
        .scores = golden_scores_vmaf_b_v063,
        .n_scores = ARRAY_SIZE(golden_scores_vmaf_b_v063),
        .extra_features = NULL,
        .n_extra_features = 0,
    },
    {
        .id = "cambi_8bit_576x324",
        .ref_file = "src01_hrc01_576x324.yuv",
        .dis_file = "src01_hrc01_576x324.yuv",
        .width = 576, .height = 324,
        .bit_depth = 8,
        .pix_fmt = VMAF_PIX_FMT_YUV420P,
        .model_version = NULL,
        .is_model_collection = 0,
        .n_frames = 48,
        .scores = golden_scores_cambi,
        .n_scores = ARRAY_SIZE(golden_scores_cambi),
        .extra_features = extra_features_cambi,
        .n_extra_features = ARRAY_SIZE(extra_features_cambi),
    },
};

#define GOLDEN_TEST_CASE_CNT ARRAY_SIZE(golden_test_cases)

#endif /* GOLDEN_VALUES_H_ */
