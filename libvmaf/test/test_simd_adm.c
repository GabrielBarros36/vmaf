/**
 * SIMD Oracle Tests — ADM Module
 *
 * Tests adm_dwt2_8 C reference vs AVX2/NEON implementations.
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "test.h"
#include "test_simd_common.h"

#include "config.h"
#include "feature/integer_adm.h"

#if ARCH_X86
#include "feature/x86/adm_avx2.h"
#elif ARCH_AARCH64
#include "feature/arm64/adm_neon.h"
#endif

/* ========== C Reference Implementation ========== */

static void ref_adm_dwt2_8(const uint8_t *src, const adm_dwt_band_t *dst,
                            AdmBuffer *buf, int w, int h, int src_stride,
                            int dst_stride)
{
    const int16_t *filter_lo = dwt2_db2_coeffs_lo;
    const int16_t *filter_hi = dwt2_db2_coeffs_hi;

    const int16_t shift_VP = 8;
    const int16_t shift_HP = 16;
    const int32_t add_shift_VP = 128;
    const int32_t add_shift_HP = 32768;

    int **ind_y = buf->ind_y;
    int **ind_x = buf->ind_x;

    int16_t *tmplo = (int16_t *)buf->tmp_ref;
    int16_t *tmphi = tmplo + w;
    int32_t accum;

    for (int i = 0; i < (h + 1) / 2; ++i) {
        /* Vertical pass. */
        for (int j = 0; j < w; ++j) {
            uint16_t u_s0 = src[ind_y[0][i] * src_stride + j];
            uint16_t u_s1 = src[ind_y[1][i] * src_stride + j];
            uint16_t u_s2 = src[ind_y[2][i] * src_stride + j];
            uint16_t u_s3 = src[ind_y[3][i] * src_stride + j];

            accum = 0;
            accum += (int32_t)filter_lo[0] * (int32_t)u_s0;
            accum += (int32_t)filter_lo[1] * (int32_t)u_s1;
            accum += (int32_t)filter_lo[2] * (int32_t)u_s2;
            accum += (int32_t)filter_lo[3] * (int32_t)u_s3;
            accum -= (int32_t)dwt2_db2_coeffs_lo_sum * add_shift_VP;
            tmplo[j] = (accum + add_shift_VP) >> shift_VP;

            accum = 0;
            accum += (int32_t)filter_hi[0] * (int32_t)u_s0;
            accum += (int32_t)filter_hi[1] * (int32_t)u_s1;
            accum += (int32_t)filter_hi[2] * (int32_t)u_s2;
            accum += (int32_t)filter_hi[3] * (int32_t)u_s3;
            accum -= (int32_t)dwt2_db2_coeffs_hi_sum * add_shift_VP;
            tmphi[j] = (accum + add_shift_VP) >> shift_VP;
        }

        /* Horizontal pass (lo and hi). */
        for (int j = 0; j < (w + 1) / 2; ++j) {
            int j0 = ind_x[0][j];
            int j1 = ind_x[1][j];
            int j2 = ind_x[2][j];
            int j3 = ind_x[3][j];

            int16_t s0 = tmplo[j0];
            int16_t s1 = tmplo[j1];
            int16_t s2 = tmplo[j2];
            int16_t s3 = tmplo[j3];

            accum = 0;
            accum += (int32_t)filter_lo[0] * s0;
            accum += (int32_t)filter_lo[1] * s1;
            accum += (int32_t)filter_lo[2] * s2;
            accum += (int32_t)filter_lo[3] * s3;
            dst->band_a[i * dst_stride + j] = (accum + add_shift_HP) >> shift_HP;

            accum = 0;
            accum += (int32_t)filter_hi[0] * s0;
            accum += (int32_t)filter_hi[1] * s1;
            accum += (int32_t)filter_hi[2] * s2;
            accum += (int32_t)filter_hi[3] * s3;
            dst->band_v[i * dst_stride + j] = (accum + add_shift_HP) >> shift_HP;

            s0 = tmphi[j0];
            s1 = tmphi[j1];
            s2 = tmphi[j2];
            s3 = tmphi[j3];

            accum = 0;
            accum += (int32_t)filter_lo[0] * s0;
            accum += (int32_t)filter_lo[1] * s1;
            accum += (int32_t)filter_lo[2] * s2;
            accum += (int32_t)filter_lo[3] * s3;
            dst->band_h[i * dst_stride + j] = (accum + add_shift_HP) >> shift_HP;

            accum = 0;
            accum += (int32_t)filter_hi[0] * s0;
            accum += (int32_t)filter_hi[1] * s1;
            accum += (int32_t)filter_hi[2] * s2;
            accum += (int32_t)filter_hi[3] * s3;
            dst->band_d[i * dst_stride + j] = (accum + add_shift_HP) >> shift_HP;
        }
    }
}

/* ========== Index Generation (mirror boundary) ========== */

static void generate_indices(int **ind, int full_size) {
    const int half = (full_size + 1) / 2;
    /* i = 0 */
    ind[0][0] = 1;
    ind[1][0] = 0;
    ind[2][0] = 1;
    ind[3][0] = 2;

    for (int i = 1; i < half - 2; ++i) {
        int idx1 = 2 * i;
        ind[0][i] = idx1 - 1;
        ind[1][i] = idx1;
        ind[2][i] = idx1 + 1;
        ind[3][i] = idx1 + 2;
    }
    for (int i = (half >= 2 ? half - 2 : 0); i < half; ++i) {
        int idx1 = 2 * i;
        int idx0 = idx1 - 1;
        int idx2 = idx1 + 1;
        int idx3 = idx1 + 2;
        if (idx0 >= full_size) idx0 = 2 * full_size - idx0 - 1;
        if (idx1 >= full_size) idx1 = 2 * full_size - idx1 - 1;
        if (idx2 >= full_size) idx2 = 2 * full_size - idx2 - 1;
        if (idx3 >= full_size) idx3 = 2 * full_size - idx3 - 1;
        if (i == 0) { /* Handles when half <= 2 and this loop overlaps i=0 */
            idx0 = 1; idx1 = 0; idx2 = 1; idx3 = 2;
        }
        ind[0][i] = idx0;
        ind[1][i] = idx1;
        ind[2][i] = idx2;
        ind[3][i] = idx3;
    }
}

/* ========== Buffer Setup ========== */

typedef struct {
    AdmBuffer buf;
    adm_dwt_band_t dst;
    void *ind_y_mem;
    void *ind_x_mem;
    void *band_mem;
    void *tmp_mem;
} AdmTestContext;

static int adm_test_ctx_init(AdmTestContext *ctx, int w, int h) {
    memset(ctx, 0, sizeof(*ctx));

    int out_w = (w + 1) / 2;
    int out_h = (h + 1) / 2;
    size_t ind_y_size = ALIGN_CEIL(out_h * sizeof(int));
    size_t ind_x_size = ALIGN_CEIL(out_w * sizeof(int));

    ctx->ind_y_mem = aligned_malloc(ind_y_size * 4, MAX_ALIGN);
    ctx->ind_x_mem = aligned_malloc(ind_x_size * 4, MAX_ALIGN);
    if (!ctx->ind_y_mem || !ctx->ind_x_mem) return -1;

    char *p = (char *)ctx->ind_y_mem;
    for (int i = 0; i < 4; i++) {
        ctx->buf.ind_y[i] = (int *)p;
        p += ind_y_size;
    }
    p = (char *)ctx->ind_x_mem;
    for (int i = 0; i < 4; i++) {
        ctx->buf.ind_x[i] = (int *)p;
        p += ind_x_size;
    }
    ctx->buf.ind_size_x = ind_x_size;
    ctx->buf.ind_size_y = ind_y_size;

    generate_indices(ctx->buf.ind_y, h);
    generate_indices(ctx->buf.ind_x, w);

    /* tmp_ref: needs at least 2*w int16_t for tmplo and tmphi */
    size_t tmp_size = ALIGN_CEIL(w * sizeof(int16_t) * 2 + MAX_ALIGN);
    ctx->tmp_mem = aligned_malloc(tmp_size, MAX_ALIGN);
    if (!ctx->tmp_mem) return -1;
    ctx->buf.tmp_ref = ctx->tmp_mem;

    return 0;
}

static int adm_dst_init(adm_dwt_band_t *dst, int w, int h, void **mem) {
    int out_w = (w + 1) / 2;
    int out_h = (h + 1) / 2;
    size_t stride = ALIGN_CEIL(out_w * sizeof(int16_t));
    size_t band_size = stride * out_h;
    size_t total = band_size * 4;
    *mem = aligned_malloc(total, MAX_ALIGN);
    if (!*mem) return -1;
    memset(*mem, FILL_SENTINEL, total);

    char *p = (char *)*mem;
    dst->band_a = (int16_t *)p; p += band_size;
    dst->band_v = (int16_t *)p; p += band_size;
    dst->band_h = (int16_t *)p; p += band_size;
    dst->band_d = (int16_t *)p;
    return 0;
}

static void adm_test_ctx_free(AdmTestContext *ctx) {
    if (ctx->ind_y_mem) aligned_free(ctx->ind_y_mem);
    if (ctx->ind_x_mem) aligned_free(ctx->ind_x_mem);
    if (ctx->tmp_mem)   aligned_free(ctx->tmp_mem);
}

#if ARCH_X86 || ARCH_AARCH64
/* ADM dimensions: production requires w > 32 && h > 32, SIMD requires w % 8 == 0 */
static const TestDim adm_simd_dims[] = {
    {64, 64}, {120, 68}, {576, 324}, {1920, 1080}
};
#define NUM_ADM_SIMD_DIMS (sizeof(adm_simd_dims) / sizeof(adm_simd_dims[0]))
#endif

/* ========== AVX2 Oracle Test ========== */

#if ARCH_X86
static char *test_adm_dwt2_8_avx2(void) {
    for (unsigned d = 0; d < NUM_ADM_SIMD_DIMS; d++) {
        unsigned w = adm_simd_dims[d].w;
        unsigned h = adm_simd_dims[d].h;

        /* ADM SIMD requires w % 8 == 0 */
        if (w % 8 != 0) continue;

        int out_w = (w + 1) / 2;
        int out_h = (h + 1) / 2;
        ptrdiff_t src_stride = (ptrdiff_t)ALIGN_CEIL(w);
        ptrdiff_t dst_stride = ALIGN_CEIL(out_w * sizeof(int16_t)) / sizeof(int16_t);

        for (int cat = 0; cat < INPUT_CATEGORY_COUNT; cat++) {
            /* Setup buffers */
            AdmTestContext ctx;
            mu_assert("ctx init failed", adm_test_ctx_init(&ctx, w, h) == 0);

            /* Allocate source */
            size_t src_size = src_stride * h;
            void *src_mem = aligned_malloc(src_size, MAX_ALIGN);
            mu_assert("src alloc failed", src_mem != NULL);
            generate_input_8((uint8_t *)src_mem, w, h, src_stride, (InputCategory)cat);

            /* Allocate output bands for ref and simd */
            adm_dwt_band_t dst_ref, dst_simd;
            void *band_ref_mem, *band_simd_mem;
            mu_assert("band ref alloc failed",
                adm_dst_init(&dst_ref, w, h, &band_ref_mem) == 0);
            mu_assert("band simd alloc failed",
                adm_dst_init(&dst_simd, w, h, &band_simd_mem) == 0);

            /* Run C reference */
            ref_adm_dwt2_8((const uint8_t *)src_mem, &dst_ref, &ctx.buf,
                           w, h, (int)src_stride, (int)dst_stride);

            /* Reset tmp_ref for SIMD call */
            memset(ctx.buf.tmp_ref, 0,
                ALIGN_CEIL(w * sizeof(int16_t) * 2 + MAX_ALIGN));

            /* Run AVX2 */
            adm_dwt2_8_avx2((const uint8_t *)src_mem, &dst_simd, &ctx.buf,
                            w, h, (int)src_stride, (int)dst_stride);

            /* Compare all 4 bands */
            size_t band_elems = (size_t)out_w * out_h;
            size_t band_bytes;

            /* band_a */
            band_bytes = 0;
            for (int i = 0; i < out_h; i++)
                for (int j = 0; j < out_w; j++) {
                    int16_t rv = dst_ref.band_a[i * dst_stride + j];
                    int16_t sv = dst_simd.band_a[i * dst_stride + j];
                    if (rv != sv) {
                        snprintf(diag_msg, sizeof(diag_msg),
                            "adm_dwt2_8_avx2 band_a MISMATCH [%s %ux%u]: "
                            "pos (%d,%d), ref=%d, simd=%d",
                            input_category_names[cat], w, h, i, j, rv, sv);
                        mu_assert(diag_msg, 0);
                    }
                }

            /* band_v */
            for (int i = 0; i < out_h; i++)
                for (int j = 0; j < out_w; j++) {
                    int16_t rv = dst_ref.band_v[i * dst_stride + j];
                    int16_t sv = dst_simd.band_v[i * dst_stride + j];
                    if (rv != sv) {
                        snprintf(diag_msg, sizeof(diag_msg),
                            "adm_dwt2_8_avx2 band_v MISMATCH [%s %ux%u]: "
                            "pos (%d,%d), ref=%d, simd=%d",
                            input_category_names[cat], w, h, i, j, rv, sv);
                        mu_assert(diag_msg, 0);
                    }
                }

            /* band_h */
            for (int i = 0; i < out_h; i++)
                for (int j = 0; j < out_w; j++) {
                    int16_t rv = dst_ref.band_h[i * dst_stride + j];
                    int16_t sv = dst_simd.band_h[i * dst_stride + j];
                    if (rv != sv) {
                        snprintf(diag_msg, sizeof(diag_msg),
                            "adm_dwt2_8_avx2 band_h MISMATCH [%s %ux%u]: "
                            "pos (%d,%d), ref=%d, simd=%d",
                            input_category_names[cat], w, h, i, j, rv, sv);
                        mu_assert(diag_msg, 0);
                    }
                }

            /* band_d */
            for (int i = 0; i < out_h; i++)
                for (int j = 0; j < out_w; j++) {
                    int16_t rv = dst_ref.band_d[i * dst_stride + j];
                    int16_t sv = dst_simd.band_d[i * dst_stride + j];
                    if (rv != sv) {
                        snprintf(diag_msg, sizeof(diag_msg),
                            "adm_dwt2_8_avx2 band_d MISMATCH [%s %ux%u]: "
                            "pos (%d,%d), ref=%d, simd=%d",
                            input_category_names[cat], w, h, i, j, rv, sv);
                        mu_assert(diag_msg, 0);
                    }
                }

            aligned_free(src_mem);
            aligned_free(band_ref_mem);
            aligned_free(band_simd_mem);
            adm_test_ctx_free(&ctx);
        }
    }
    return NULL;
}
#endif

/* ========== NEON Oracle Test ========== */

#if ARCH_AARCH64
static char *test_adm_dwt2_8_neon(void) {
    for (unsigned d = 0; d < NUM_ADM_SIMD_DIMS; d++) {
        unsigned w = adm_simd_dims[d].w;
        unsigned h = adm_simd_dims[d].h;

        if (w % 8 != 0) continue;

        int out_w = (w + 1) / 2;
        int out_h = (h + 1) / 2;
        ptrdiff_t src_stride = (ptrdiff_t)ALIGN_CEIL(w);
        ptrdiff_t dst_stride = ALIGN_CEIL(out_w * sizeof(int16_t)) / sizeof(int16_t);

        for (int cat = 0; cat < INPUT_CATEGORY_COUNT; cat++) {
            AdmTestContext ctx;
            mu_assert("ctx init failed", adm_test_ctx_init(&ctx, w, h) == 0);

            size_t src_size = src_stride * h;
            void *src_mem = aligned_malloc(src_size, MAX_ALIGN);
            mu_assert("src alloc failed", src_mem != NULL);
            generate_input_8((uint8_t *)src_mem, w, h, src_stride, (InputCategory)cat);

            adm_dwt_band_t dst_ref, dst_simd;
            void *band_ref_mem, *band_simd_mem;
            mu_assert("band ref alloc failed",
                adm_dst_init(&dst_ref, w, h, &band_ref_mem) == 0);
            mu_assert("band simd alloc failed",
                adm_dst_init(&dst_simd, w, h, &band_simd_mem) == 0);

            ref_adm_dwt2_8((const uint8_t *)src_mem, &dst_ref, &ctx.buf,
                           w, h, (int)src_stride, (int)dst_stride);

            memset(ctx.buf.tmp_ref, 0,
                ALIGN_CEIL(w * sizeof(int16_t) * 2 + MAX_ALIGN));

            adm_dwt2_8_neon((const uint8_t *)src_mem, &dst_simd, &ctx.buf,
                            w, h, (int)src_stride, (int)dst_stride);

            for (int i = 0; i < out_h; i++)
                for (int j = 0; j < out_w; j++) {
                    if (dst_ref.band_a[i * dst_stride + j] != dst_simd.band_a[i * dst_stride + j] ||
                        dst_ref.band_v[i * dst_stride + j] != dst_simd.band_v[i * dst_stride + j] ||
                        dst_ref.band_h[i * dst_stride + j] != dst_simd.band_h[i * dst_stride + j] ||
                        dst_ref.band_d[i * dst_stride + j] != dst_simd.band_d[i * dst_stride + j]) {
                        snprintf(diag_msg, sizeof(diag_msg),
                            "adm_dwt2_8_neon MISMATCH [%s %ux%u]: pos (%d,%d)",
                            input_category_names[cat], w, h, i, j);
                        mu_assert(diag_msg, 0);
                    }
                }

            aligned_free(src_mem);
            aligned_free(band_ref_mem);
            aligned_free(band_simd_mem);
            adm_test_ctx_free(&ctx);
        }
    }
    return NULL;
}
#endif

/* ========== C Reference: s1_combined ========== */

static void ref_adm_dwt2_s1_combined(const int16_t *i2_ref_scale,
                                      const int16_t *i2_dis_scale,
                                      AdmBuffer *buf, int w, int h,
                                      int ref_stride, int dis_stride,
                                      int dst_stride)
{
    const i4_adm_dwt_band_t *i4_ref_dwt2 = &buf->i4_ref_dwt2;
    const i4_adm_dwt_band_t *i4_dis_dwt2 = &buf->i4_dis_dwt2;
    int **ind_y = buf->ind_y;
    int **ind_x = buf->ind_x;
    const int16_t *filter_lo = dwt2_db2_coeffs_lo;
    const int16_t *filter_hi = dwt2_db2_coeffs_hi;
    const int32_t add_HP = 16384;
    const int shift_HP = 15;

    int32_t *tmplo_ref = buf->tmp_ref;
    int32_t *tmphi_ref = tmplo_ref + w;
    int32_t *tmplo_dis = tmphi_ref + w;
    int32_t *tmphi_dis = tmplo_dis + w;

    for (int i = 0; i < (h + 1) / 2; ++i) {
        for (int j = 0; j < w; ++j) {
            int32_t s0 = (int32_t)i2_ref_scale[ind_y[0][i] * ref_stride + j];
            int32_t s1 = (int32_t)i2_ref_scale[ind_y[1][i] * ref_stride + j];
            int32_t s2 = (int32_t)i2_ref_scale[ind_y[2][i] * ref_stride + j];
            int32_t s3 = (int32_t)i2_ref_scale[ind_y[3][i] * ref_stride + j];
            int64_t acc = (int64_t)filter_lo[0]*s0 + (int64_t)filter_lo[1]*s1 +
                          (int64_t)filter_lo[2]*s2 + (int64_t)filter_lo[3]*s3;
            tmplo_ref[j] = (int32_t)acc;
            acc = (int64_t)filter_hi[0]*s0 + (int64_t)filter_hi[1]*s1 +
                  (int64_t)filter_hi[2]*s2 + (int64_t)filter_hi[3]*s3;
            tmphi_ref[j] = (int32_t)acc;

            s0 = (int32_t)i2_dis_scale[ind_y[0][i] * dis_stride + j];
            s1 = (int32_t)i2_dis_scale[ind_y[1][i] * dis_stride + j];
            s2 = (int32_t)i2_dis_scale[ind_y[2][i] * dis_stride + j];
            s3 = (int32_t)i2_dis_scale[ind_y[3][i] * dis_stride + j];
            acc = (int64_t)filter_lo[0]*s0 + (int64_t)filter_lo[1]*s1 +
                  (int64_t)filter_lo[2]*s2 + (int64_t)filter_lo[3]*s3;
            tmplo_dis[j] = (int32_t)acc;
            acc = (int64_t)filter_hi[0]*s0 + (int64_t)filter_hi[1]*s1 +
                  (int64_t)filter_hi[2]*s2 + (int64_t)filter_hi[3]*s3;
            tmphi_dis[j] = (int32_t)acc;
        }
        for (int j = 0; j < (w + 1) / 2; ++j) {
            int j0 = ind_x[0][j], j1 = ind_x[1][j], j2 = ind_x[2][j], j3 = ind_x[3][j];
            int32_t s0, s1, s2, s3;
            int64_t acc;

#define S1_HORZ(tmp, band, stream) do { \
    s0 = tmp[j0]; s1 = tmp[j1]; s2 = tmp[j2]; s3 = tmp[j3]; \
    acc = (int64_t)filter_lo[0]*s0 + (int64_t)filter_lo[1]*s1 + \
          (int64_t)filter_lo[2]*s2 + (int64_t)filter_lo[3]*s3; \
    stream->band_a[i * dst_stride + j] = (int32_t)((acc + add_HP) >> shift_HP); \
    acc = (int64_t)filter_hi[0]*s0 + (int64_t)filter_hi[1]*s1 + \
          (int64_t)filter_hi[2]*s2 + (int64_t)filter_hi[3]*s3; \
    stream->band_v[i * dst_stride + j] = (int32_t)((acc + add_HP) >> shift_HP); \
} while(0)

#define S1_HORZ_HI(tmp, band, stream) do { \
    s0 = tmp[j0]; s1 = tmp[j1]; s2 = tmp[j2]; s3 = tmp[j3]; \
    acc = (int64_t)filter_lo[0]*s0 + (int64_t)filter_lo[1]*s1 + \
          (int64_t)filter_lo[2]*s2 + (int64_t)filter_lo[3]*s3; \
    stream->band_h[i * dst_stride + j] = (int32_t)((acc + add_HP) >> shift_HP); \
    acc = (int64_t)filter_hi[0]*s0 + (int64_t)filter_hi[1]*s1 + \
          (int64_t)filter_hi[2]*s2 + (int64_t)filter_hi[3]*s3; \
    stream->band_d[i * dst_stride + j] = (int32_t)((acc + add_HP) >> shift_HP); \
} while(0)

            S1_HORZ(tmplo_ref, lo, i4_ref_dwt2);
            S1_HORZ_HI(tmphi_ref, hi, i4_ref_dwt2);
            S1_HORZ(tmplo_dis, lo, i4_dis_dwt2);
            S1_HORZ_HI(tmphi_dis, hi, i4_dis_dwt2);
#undef S1_HORZ
#undef S1_HORZ_HI
        }
    }
}

/* ========== C Reference: s123_combined ========== */

static void ref_adm_dwt2_s123_combined(const int32_t *i4_ref_scale,
                                         const int32_t *i4_curr_dis,
                                         AdmBuffer *buf, int w, int h,
                                         int ref_stride, int dis_stride,
                                         int dst_stride, int scale)
{
    const i4_adm_dwt_band_t *i4_ref_dwt2 = &buf->i4_ref_dwt2;
    const i4_adm_dwt_band_t *i4_dis_dwt2 = &buf->i4_dis_dwt2;
    int **ind_y = buf->ind_y;
    int **ind_x = buf->ind_x;
    const int16_t *filter_lo = dwt2_db2_coeffs_lo;
    const int16_t *filter_hi = dwt2_db2_coeffs_hi;
    const int32_t add_VP_arr[3] = { 0, 32768, 32768 };
    const int32_t add_HP_arr[3] = { 16384, 32768, 16384 };
    const int shift_VP_arr[3] = { 0, 16, 16 };
    const int shift_HP_arr[3] = { 15, 16, 15 };
    const int32_t add_VP = add_VP_arr[scale - 1];
    const int shift_VP = shift_VP_arr[scale - 1];
    const int32_t add_HP = add_HP_arr[scale - 1];
    const int shift_HP = shift_HP_arr[scale - 1];

    int32_t *tmplo_ref = buf->tmp_ref;
    int32_t *tmphi_ref = tmplo_ref + w;
    int32_t *tmplo_dis = tmphi_ref + w;
    int32_t *tmphi_dis = tmplo_dis + w;

    for (int i = 0; i < (h + 1) / 2; ++i) {
        for (int j = 0; j < w; ++j) {
            int32_t s0, s1, s2, s3;
            int64_t acc;
            s0 = i4_ref_scale[ind_y[0][i] * ref_stride + j];
            s1 = i4_ref_scale[ind_y[1][i] * ref_stride + j];
            s2 = i4_ref_scale[ind_y[2][i] * ref_stride + j];
            s3 = i4_ref_scale[ind_y[3][i] * ref_stride + j];
            acc = (int64_t)filter_lo[0]*s0 + (int64_t)filter_lo[1]*s1 +
                  (int64_t)filter_lo[2]*s2 + (int64_t)filter_lo[3]*s3;
            tmplo_ref[j] = (int32_t)((acc + add_VP) >> shift_VP);
            acc = (int64_t)filter_hi[0]*s0 + (int64_t)filter_hi[1]*s1 +
                  (int64_t)filter_hi[2]*s2 + (int64_t)filter_hi[3]*s3;
            tmphi_ref[j] = (int32_t)((acc + add_VP) >> shift_VP);

            s0 = i4_curr_dis[ind_y[0][i] * dis_stride + j];
            s1 = i4_curr_dis[ind_y[1][i] * dis_stride + j];
            s2 = i4_curr_dis[ind_y[2][i] * dis_stride + j];
            s3 = i4_curr_dis[ind_y[3][i] * dis_stride + j];
            acc = (int64_t)filter_lo[0]*s0 + (int64_t)filter_lo[1]*s1 +
                  (int64_t)filter_lo[2]*s2 + (int64_t)filter_lo[3]*s3;
            tmplo_dis[j] = (int32_t)((acc + add_VP) >> shift_VP);
            acc = (int64_t)filter_hi[0]*s0 + (int64_t)filter_hi[1]*s1 +
                  (int64_t)filter_hi[2]*s2 + (int64_t)filter_hi[3]*s3;
            tmphi_dis[j] = (int32_t)((acc + add_VP) >> shift_VP);
        }
        for (int j = 0; j < (w + 1) / 2; ++j) {
            int j0 = ind_x[0][j], j1 = ind_x[1][j], j2 = ind_x[2][j], j3 = ind_x[3][j];
            int32_t s0, s1, s2, s3;
            int64_t acc;

#define S123_HORZ(tmp, stream) do { \
    s0 = tmp[j0]; s1 = tmp[j1]; s2 = tmp[j2]; s3 = tmp[j3]; \
    acc = (int64_t)filter_lo[0]*s0 + (int64_t)filter_lo[1]*s1 + \
          (int64_t)filter_lo[2]*s2 + (int64_t)filter_lo[3]*s3; \
    stream->band_a[i * dst_stride + j] = (int32_t)((acc + add_HP) >> shift_HP); \
    acc = (int64_t)filter_hi[0]*s0 + (int64_t)filter_hi[1]*s1 + \
          (int64_t)filter_hi[2]*s2 + (int64_t)filter_hi[3]*s3; \
    stream->band_v[i * dst_stride + j] = (int32_t)((acc + add_HP) >> shift_HP); \
} while(0)

#define S123_HORZ_HI(tmp, stream) do { \
    s0 = tmp[j0]; s1 = tmp[j1]; s2 = tmp[j2]; s3 = tmp[j3]; \
    acc = (int64_t)filter_lo[0]*s0 + (int64_t)filter_lo[1]*s1 + \
          (int64_t)filter_lo[2]*s2 + (int64_t)filter_lo[3]*s3; \
    stream->band_h[i * dst_stride + j] = (int32_t)((acc + add_HP) >> shift_HP); \
    acc = (int64_t)filter_hi[0]*s0 + (int64_t)filter_hi[1]*s1 + \
          (int64_t)filter_hi[2]*s2 + (int64_t)filter_hi[3]*s3; \
    stream->band_d[i * dst_stride + j] = (int32_t)((acc + add_HP) >> shift_HP); \
} while(0)

            S123_HORZ(tmplo_ref, i4_ref_dwt2);
            S123_HORZ_HI(tmphi_ref, i4_ref_dwt2);
            S123_HORZ(tmplo_dis, i4_dis_dwt2);
            S123_HORZ_HI(tmphi_dis, i4_dis_dwt2);
#undef S123_HORZ
#undef S123_HORZ_HI
        }
    }
}

/* ========== i4 band helpers ========== */

typedef struct {
    AdmBuffer buf;
    void *ind_y_mem;
    void *ind_x_mem;
    void *tmp_mem;
    void *i4_ref_band_mem;
    void *i4_dis_band_mem;
} AdmI4TestContext;

static int adm_i4_test_ctx_init(AdmI4TestContext *ctx, int w, int h,
                                  int need_tmp_i32) {
    (void)need_tmp_i32;
    memset(ctx, 0, sizeof(*ctx));
    int out_w = (w + 1) / 2;
    int out_h = (h + 1) / 2;
    size_t ind_y_size = ALIGN_CEIL(out_h * sizeof(int));
    size_t ind_x_size = ALIGN_CEIL(out_w * sizeof(int));

    ctx->ind_y_mem = aligned_malloc(ind_y_size * 4, MAX_ALIGN);
    ctx->ind_x_mem = aligned_malloc(ind_x_size * 4, MAX_ALIGN);
    if (!ctx->ind_y_mem || !ctx->ind_x_mem) return -1;

    char *p = (char *)ctx->ind_y_mem;
    for (int i = 0; i < 4; i++) { ctx->buf.ind_y[i] = (int *)p; p += ind_y_size; }
    p = (char *)ctx->ind_x_mem;
    for (int i = 0; i < 4; i++) { ctx->buf.ind_x[i] = (int *)p; p += ind_x_size; }
    ctx->buf.ind_size_x = ind_x_size;
    ctx->buf.ind_size_y = ind_y_size;
    generate_indices(ctx->buf.ind_y, h);
    generate_indices(ctx->buf.ind_x, w);

    /* tmp_ref: 4*w int32_t (tmplo_ref, tmphi_ref, tmplo_dis, tmphi_dis) */
    size_t tmp_size = ALIGN_CEIL(w * sizeof(int32_t) * 4 + MAX_ALIGN);
    ctx->tmp_mem = aligned_malloc(tmp_size, MAX_ALIGN);
    if (!ctx->tmp_mem) return -1;
    ctx->buf.tmp_ref = ctx->tmp_mem;

    return 0;
}

static int adm_i4_bands_init(i4_adm_dwt_band_t *bands, int w, int h,
                              void **mem) {
    int out_w = (w + 1) / 2;
    int out_h = (h + 1) / 2;
    size_t stride = ALIGN_CEIL(out_w * sizeof(int32_t));
    size_t band_size = stride * out_h;
    size_t total = band_size * 4;
    *mem = aligned_malloc(total, MAX_ALIGN);
    if (!*mem) return -1;
    memset(*mem, FILL_SENTINEL, total);
    char *p = (char *)*mem;
    bands->band_a = (int32_t *)p; p += band_size;
    bands->band_v = (int32_t *)p; p += band_size;
    bands->band_h = (int32_t *)p; p += band_size;
    bands->band_d = (int32_t *)p;
    return 0;
}

static void adm_i4_test_ctx_free(AdmI4TestContext *ctx) {
    if (ctx->ind_y_mem) aligned_free(ctx->ind_y_mem);
    if (ctx->ind_x_mem) aligned_free(ctx->ind_x_mem);
    if (ctx->tmp_mem)   aligned_free(ctx->tmp_mem);
}

/* ========== Int16 input generator ========== */

static inline void generate_input_i16(int16_t *buf, unsigned w, unsigned h,
                                        ptrdiff_t stride_elems, InputCategory cat)
{
    for (unsigned i = 0; i < h; i++)
        memset(buf + i * stride_elems, 0, stride_elems * sizeof(int16_t));
    switch (cat) {
    case INPUT_ZERO: break;
    case INPUT_MAX:
        for (unsigned i = 0; i < h; i++)
            for (unsigned j = 0; j < w; j++)
                buf[i * stride_elems + j] = 32767;
        break;
    case INPUT_CONSTANT:
        for (unsigned i = 0; i < h; i++)
            for (unsigned j = 0; j < w; j++)
                buf[i * stride_elems + j] = 1000;
        break;
    case INPUT_GRADIENT_H:
        for (unsigned i = 0; i < h; i++)
            for (unsigned j = 0; j < w; j++)
                buf[i * stride_elems + j] = (int16_t)((int32_t)j * 200 / (w > 1 ? w - 1 : 1) - 100);
        break;
    case INPUT_GRADIENT_V:
        for (unsigned i = 0; i < h; i++)
            for (unsigned j = 0; j < w; j++)
                buf[i * stride_elems + j] = (int16_t)((int32_t)i * 200 / (h > 1 ? h - 1 : 1) - 100);
        break;
    case INPUT_CHECKERBOARD:
        for (unsigned i = 0; i < h; i++)
            for (unsigned j = 0; j < w; j++)
                buf[i * stride_elems + j] = ((i + j) & 1) ? 10000 : -10000;
        break;
    case INPUT_RANDOM_42: {
        uint32_t state = 42;
        for (unsigned i = 0; i < h; i++)
            for (unsigned j = 0; j < w; j++)
                buf[i * stride_elems + j] = (int16_t)(prng_next(&state) & 0xFFFF);
        break;
    }
    case INPUT_RANDOM_123: {
        uint32_t state = 123;
        for (unsigned i = 0; i < h; i++)
            for (unsigned j = 0; j < w; j++)
                buf[i * stride_elems + j] = (int16_t)(prng_next(&state) & 0xFFFF);
        break;
    }
    case INPUT_SINGLE_HOT:
        buf[(h / 2) * stride_elems + (w / 2)] = 32767;
        break;
    case INPUT_BOUNDARY_STRIPE:
        for (unsigned j = 0; j < w; j++) {
            buf[0 * stride_elems + j] = 20000;
            buf[(h - 1) * stride_elems + j] = 20000;
        }
        for (unsigned i = 0; i < h; i++) {
            buf[i * stride_elems + 0] = 20000;
            buf[i * stride_elems + (w - 1)] = 20000;
        }
        break;
    default: break;
    }
}

/* ========== Int32 input generator ========== */

static inline void generate_input_i32(int32_t *buf, unsigned w, unsigned h,
                                        ptrdiff_t stride_elems, InputCategory cat)
{
    for (unsigned i = 0; i < h; i++)
        memset(buf + i * stride_elems, 0, stride_elems * sizeof(int32_t));
    switch (cat) {
    case INPUT_ZERO: break;
    case INPUT_MAX:
        for (unsigned i = 0; i < h; i++)
            for (unsigned j = 0; j < w; j++)
                buf[i * stride_elems + j] = 500000;
        break;
    case INPUT_CONSTANT:
        for (unsigned i = 0; i < h; i++)
            for (unsigned j = 0; j < w; j++)
                buf[i * stride_elems + j] = 100000;
        break;
    case INPUT_GRADIENT_H:
        for (unsigned i = 0; i < h; i++)
            for (unsigned j = 0; j < w; j++)
                buf[i * stride_elems + j] = (int32_t)((int64_t)j * 400000 / (w > 1 ? w - 1 : 1) - 200000);
        break;
    case INPUT_GRADIENT_V:
        for (unsigned i = 0; i < h; i++)
            for (unsigned j = 0; j < w; j++)
                buf[i * stride_elems + j] = (int32_t)((int64_t)i * 400000 / (h > 1 ? h - 1 : 1) - 200000);
        break;
    case INPUT_CHECKERBOARD:
        for (unsigned i = 0; i < h; i++)
            for (unsigned j = 0; j < w; j++)
                buf[i * stride_elems + j] = ((i + j) & 1) ? 300000 : -300000;
        break;
    case INPUT_RANDOM_42: {
        uint32_t state = 42;
        for (unsigned i = 0; i < h; i++)
            for (unsigned j = 0; j < w; j++)
                buf[i * stride_elems + j] = (int32_t)(prng_next(&state) % 1000000) - 500000;
        break;
    }
    case INPUT_RANDOM_123: {
        uint32_t state = 123;
        for (unsigned i = 0; i < h; i++)
            for (unsigned j = 0; j < w; j++)
                buf[i * stride_elems + j] = (int32_t)(prng_next(&state) % 1000000) - 500000;
        break;
    }
    case INPUT_SINGLE_HOT:
        buf[(h / 2) * stride_elems + (w / 2)] = 500000;
        break;
    case INPUT_BOUNDARY_STRIPE:
        for (unsigned j = 0; j < w; j++) {
            buf[0 * stride_elems + j] = 200000;
            buf[(h - 1) * stride_elems + j] = 200000;
        }
        for (unsigned i = 0; i < h; i++) {
            buf[i * stride_elems + 0] = 200000;
            buf[i * stride_elems + (w - 1)] = 200000;
        }
        break;
    default: break;
    }
}

/* ========== AVX2 Oracle: s1_combined and s123_combined ========== */

#if ARCH_X86
static char *test_adm_dwt2_s1_combined_avx2(void) {
    for (unsigned d = 0; d < NUM_ADM_SIMD_DIMS; d++) {
        unsigned w = adm_simd_dims[d].w;
        unsigned h = adm_simd_dims[d].h;

        int out_w = (w + 1) / 2;
        int out_h = (h + 1) / 2;
        ptrdiff_t src_stride = ALIGN_CEIL(w * sizeof(int16_t)) / sizeof(int16_t);
        ptrdiff_t dst_stride = ALIGN_CEIL(out_w * sizeof(int32_t)) / sizeof(int32_t);

        for (int cat = 0; cat < INPUT_CATEGORY_COUNT; cat++) {
            AdmI4TestContext ctx_ref, ctx_simd;
            mu_assert("ctx_ref init", adm_i4_test_ctx_init(&ctx_ref, w, h, 1) == 0);
            mu_assert("ctx_simd init", adm_i4_test_ctx_init(&ctx_simd, w, h, 1) == 0);

            /* Allocate i4 output bands */
            void *ref_ref_mem, *ref_dis_mem, *simd_ref_mem, *simd_dis_mem;
            mu_assert("ref ref bands", adm_i4_bands_init(&ctx_ref.buf.i4_ref_dwt2, w, h, &ref_ref_mem) == 0);
            mu_assert("ref dis bands", adm_i4_bands_init(&ctx_ref.buf.i4_dis_dwt2, w, h, &ref_dis_mem) == 0);
            mu_assert("simd ref bands", adm_i4_bands_init(&ctx_simd.buf.i4_ref_dwt2, w, h, &simd_ref_mem) == 0);
            mu_assert("simd dis bands", adm_i4_bands_init(&ctx_simd.buf.i4_dis_dwt2, w, h, &simd_dis_mem) == 0);

            /* Allocate and fill source (int16 for ref and dis) */
            size_t src_size = src_stride * h * sizeof(int16_t);
            void *ref_src = aligned_malloc(src_size, MAX_ALIGN);
            void *dis_src = aligned_malloc(src_size, MAX_ALIGN);
            mu_assert("ref_src alloc", ref_src != NULL);
            mu_assert("dis_src alloc", dis_src != NULL);
            generate_input_i16((int16_t *)ref_src, w, h, src_stride, (InputCategory)cat);
            /* Use a different seed for dis */
            generate_input_i16((int16_t *)dis_src, w, h, src_stride,
                               (InputCategory)((cat + 1) % INPUT_CATEGORY_COUNT));

            /* Run C reference */
            ref_adm_dwt2_s1_combined((const int16_t *)ref_src, (const int16_t *)dis_src,
                                      &ctx_ref.buf, w, h,
                                      (int)src_stride, (int)src_stride, (int)dst_stride);

            /* Run AVX2 */
            adm_dwt2_s1_combined_avx2((const int16_t *)ref_src, (const int16_t *)dis_src,
                                       &ctx_simd.buf, w, h,
                                       (int)src_stride, (int)src_stride, (int)dst_stride);

            /* Compare ref stream bands */
            for (int i = 0; i < out_h; i++)
                for (int j = 0; j < out_w; j++) {
                    int idx = i * (int)dst_stride + j;
                    if (ctx_ref.buf.i4_ref_dwt2.band_a[idx] != ctx_simd.buf.i4_ref_dwt2.band_a[idx]) {
                        snprintf(diag_msg, sizeof(diag_msg),
                            "s1_combined_avx2 ref band_a MISMATCH [%s %ux%u]: pos (%d,%d), ref=%d, simd=%d",
                            input_category_names[cat], w, h, i, j,
                            ctx_ref.buf.i4_ref_dwt2.band_a[idx], ctx_simd.buf.i4_ref_dwt2.band_a[idx]);
                        mu_assert(diag_msg, 0);
                    }
                    if (ctx_ref.buf.i4_ref_dwt2.band_v[idx] != ctx_simd.buf.i4_ref_dwt2.band_v[idx]) {
                        snprintf(diag_msg, sizeof(diag_msg),
                            "s1_combined_avx2 ref band_v MISMATCH [%s %ux%u]: pos (%d,%d)",
                            input_category_names[cat], w, h, i, j);
                        mu_assert(diag_msg, 0);
                    }
                    if (ctx_ref.buf.i4_ref_dwt2.band_h[idx] != ctx_simd.buf.i4_ref_dwt2.band_h[idx]) {
                        snprintf(diag_msg, sizeof(diag_msg),
                            "s1_combined_avx2 ref band_h MISMATCH [%s %ux%u]: pos (%d,%d)",
                            input_category_names[cat], w, h, i, j);
                        mu_assert(diag_msg, 0);
                    }
                    if (ctx_ref.buf.i4_ref_dwt2.band_d[idx] != ctx_simd.buf.i4_ref_dwt2.band_d[idx]) {
                        snprintf(diag_msg, sizeof(diag_msg),
                            "s1_combined_avx2 ref band_d MISMATCH [%s %ux%u]: pos (%d,%d)",
                            input_category_names[cat], w, h, i, j);
                        mu_assert(diag_msg, 0);
                    }
                    /* Dis stream */
                    if (ctx_ref.buf.i4_dis_dwt2.band_a[idx] != ctx_simd.buf.i4_dis_dwt2.band_a[idx]) {
                        snprintf(diag_msg, sizeof(diag_msg),
                            "s1_combined_avx2 dis band_a MISMATCH [%s %ux%u]: pos (%d,%d)",
                            input_category_names[cat], w, h, i, j);
                        mu_assert(diag_msg, 0);
                    }
                    if (ctx_ref.buf.i4_dis_dwt2.band_v[idx] != ctx_simd.buf.i4_dis_dwt2.band_v[idx]) {
                        snprintf(diag_msg, sizeof(diag_msg),
                            "s1_combined_avx2 dis band_v MISMATCH [%s %ux%u]: pos (%d,%d)",
                            input_category_names[cat], w, h, i, j);
                        mu_assert(diag_msg, 0);
                    }
                    if (ctx_ref.buf.i4_dis_dwt2.band_h[idx] != ctx_simd.buf.i4_dis_dwt2.band_h[idx]) {
                        snprintf(diag_msg, sizeof(diag_msg),
                            "s1_combined_avx2 dis band_h MISMATCH [%s %ux%u]: pos (%d,%d)",
                            input_category_names[cat], w, h, i, j);
                        mu_assert(diag_msg, 0);
                    }
                    if (ctx_ref.buf.i4_dis_dwt2.band_d[idx] != ctx_simd.buf.i4_dis_dwt2.band_d[idx]) {
                        snprintf(diag_msg, sizeof(diag_msg),
                            "s1_combined_avx2 dis band_d MISMATCH [%s %ux%u]: pos (%d,%d)",
                            input_category_names[cat], w, h, i, j);
                        mu_assert(diag_msg, 0);
                    }
                }

            aligned_free(ref_src);
            aligned_free(dis_src);
            aligned_free(ref_ref_mem);
            aligned_free(ref_dis_mem);
            aligned_free(simd_ref_mem);
            aligned_free(simd_dis_mem);
            adm_i4_test_ctx_free(&ctx_ref);
            adm_i4_test_ctx_free(&ctx_simd);
        }
    }
    return NULL;
}

static char *test_adm_dwt2_s123_combined_avx2(void) {
    int test_scales[] = { 2, 3 };
    for (int si = 0; si < 2; si++) {
        int scale = test_scales[si];
        for (unsigned d = 0; d < NUM_ADM_SIMD_DIMS; d++) {
            unsigned w = adm_simd_dims[d].w;
            unsigned h = adm_simd_dims[d].h;

            int out_w = (w + 1) / 2;
            int out_h = (h + 1) / 2;
            ptrdiff_t src_stride = ALIGN_CEIL(w * sizeof(int32_t)) / sizeof(int32_t);
            ptrdiff_t dst_stride = ALIGN_CEIL(out_w * sizeof(int32_t)) / sizeof(int32_t);

            for (int cat = 0; cat < INPUT_CATEGORY_COUNT; cat++) {
                AdmI4TestContext ctx_ref, ctx_simd;
                mu_assert("ctx_ref init", adm_i4_test_ctx_init(&ctx_ref, w, h, 1) == 0);
                mu_assert("ctx_simd init", adm_i4_test_ctx_init(&ctx_simd, w, h, 1) == 0);

                void *ref_ref_mem, *ref_dis_mem, *simd_ref_mem, *simd_dis_mem;
                mu_assert("ref ref bands", adm_i4_bands_init(&ctx_ref.buf.i4_ref_dwt2, w, h, &ref_ref_mem) == 0);
                mu_assert("ref dis bands", adm_i4_bands_init(&ctx_ref.buf.i4_dis_dwt2, w, h, &ref_dis_mem) == 0);
                mu_assert("simd ref bands", adm_i4_bands_init(&ctx_simd.buf.i4_ref_dwt2, w, h, &simd_ref_mem) == 0);
                mu_assert("simd dis bands", adm_i4_bands_init(&ctx_simd.buf.i4_dis_dwt2, w, h, &simd_dis_mem) == 0);

                size_t src_size = src_stride * h * sizeof(int32_t);
                void *ref_src = aligned_malloc(src_size, MAX_ALIGN);
                void *dis_src = aligned_malloc(src_size, MAX_ALIGN);
                mu_assert("ref_src alloc", ref_src != NULL);
                mu_assert("dis_src alloc", dis_src != NULL);
                generate_input_i32((int32_t *)ref_src, w, h, src_stride, (InputCategory)cat);
                generate_input_i32((int32_t *)dis_src, w, h, src_stride,
                                   (InputCategory)((cat + 1) % INPUT_CATEGORY_COUNT));

                /* C reference */
                ref_adm_dwt2_s123_combined((const int32_t *)ref_src, (const int32_t *)dis_src,
                                            &ctx_ref.buf, w, h,
                                            (int)src_stride, (int)src_stride,
                                            (int)dst_stride, scale);

                /* AVX2 */
                adm_dwt2_s123_combined_avx2((const int32_t *)ref_src, (const int32_t *)dis_src,
                                             &ctx_simd.buf, w, h,
                                             (int)src_stride, (int)src_stride,
                                             (int)dst_stride, scale);

                for (int i = 0; i < out_h; i++)
                    for (int j = 0; j < out_w; j++) {
                        int idx = i * (int)dst_stride + j;
#define CHECK_BAND(stream, band_name) do { \
    if (ctx_ref.buf.stream.band_name[idx] != ctx_simd.buf.stream.band_name[idx]) { \
        snprintf(diag_msg, sizeof(diag_msg), \
            "s123_combined_avx2 " #stream " " #band_name " MISMATCH [%s %ux%u scale=%d]: " \
            "pos (%d,%d), ref=%d, simd=%d", \
            input_category_names[cat], w, h, scale, i, j, \
            ctx_ref.buf.stream.band_name[idx], ctx_simd.buf.stream.band_name[idx]); \
        mu_assert(diag_msg, 0); \
    } \
} while(0)
                        CHECK_BAND(i4_ref_dwt2, band_a);
                        CHECK_BAND(i4_ref_dwt2, band_v);
                        CHECK_BAND(i4_ref_dwt2, band_h);
                        CHECK_BAND(i4_ref_dwt2, band_d);
                        CHECK_BAND(i4_dis_dwt2, band_a);
                        CHECK_BAND(i4_dis_dwt2, band_v);
                        CHECK_BAND(i4_dis_dwt2, band_h);
                        CHECK_BAND(i4_dis_dwt2, band_d);
#undef CHECK_BAND
                    }

                aligned_free(ref_src);
                aligned_free(dis_src);
                aligned_free(ref_ref_mem);
                aligned_free(ref_dis_mem);
                aligned_free(simd_ref_mem);
                aligned_free(simd_dis_mem);
                adm_i4_test_ctx_free(&ctx_ref);
                adm_i4_test_ctx_free(&ctx_simd);
            }
        }
    }
    return NULL;
}
#endif /* ARCH_X86 */

/* ========== ADM Fallback Width Tests ========== */

static char *test_adm_dwt2_8_fallback_widths(void) {
    for (unsigned d = 0; d < NUM_ADM_FALLBACK_DIMS; d++) {
        unsigned w = adm_fallback_dims[d].w;
        unsigned h = adm_fallback_dims[d].h;

        int out_w = (w + 1) / 2;
        int out_h = (h + 1) / 2;
        ptrdiff_t src_stride = (ptrdiff_t)ALIGN_CEIL(w);
        ptrdiff_t dst_stride = ALIGN_CEIL(out_w * sizeof(int16_t)) / sizeof(int16_t);

        for (int cat = 0; cat < INPUT_CATEGORY_COUNT; cat++) {
            AdmTestContext ctx;
            mu_assert("ctx init failed", adm_test_ctx_init(&ctx, w, h) == 0);

            size_t src_size = src_stride * h;
            void *src_mem = aligned_malloc(src_size, MAX_ALIGN);
            mu_assert("src alloc failed", src_mem != NULL);
            generate_input_8((uint8_t *)src_mem, w, h, src_stride, (InputCategory)cat);

            adm_dwt_band_t dst_ref;
            void *band_ref_mem;
            mu_assert("band ref alloc failed",
                adm_dst_init(&dst_ref, w, h, &band_ref_mem) == 0);

            /* Only run C reference — no SIMD for non-aligned widths */
            ref_adm_dwt2_8((const uint8_t *)src_mem, &dst_ref, &ctx.buf,
                           w, h, (int)src_stride, (int)dst_stride);

            /* Verify w % 8 != 0 (ensures dispatch wouldn't select SIMD) */
            mu_assert("fallback dim should not be multiple of 8", w % 8 != 0);

            /* Basic sanity: check that random input produces non-zero output.
             * Constant and single-hot inputs may legitimately produce zero bands
             * due to DWT normalization centering at midpoint. */
            if (cat == INPUT_RANDOM_42 || cat == INPUT_RANDOM_123) {
                int found_nonzero = 0;
                for (int i = 0; i < out_h && !found_nonzero; i++)
                    for (int j = 0; j < out_w && !found_nonzero; j++)
                        if (dst_ref.band_a[i * dst_stride + j] != 0)
                            found_nonzero = 1;
                if (!found_nonzero) {
                    snprintf(diag_msg, sizeof(diag_msg),
                        "C reference produced all zeros for random input [%s %ux%u]",
                        input_category_names[cat], w, h);
                    mu_assert(diag_msg, 0);
                }
            }

            aligned_free(src_mem);
            aligned_free(band_ref_mem);
            adm_test_ctx_free(&ctx);
        }
    }
    return NULL;
}

/* ========== Test Runner ========== */

char *run_tests(void) {
#if ARCH_X86
    mu_run_test(test_adm_dwt2_8_avx2);
    mu_run_test(test_adm_dwt2_s1_combined_avx2);
    mu_run_test(test_adm_dwt2_s123_combined_avx2);
#endif
#if ARCH_AARCH64
    mu_run_test(test_adm_dwt2_8_neon);
#endif
    mu_run_test(test_adm_dwt2_8_fallback_widths);
    return NULL;
}
