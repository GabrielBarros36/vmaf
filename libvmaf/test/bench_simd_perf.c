/**
 * SIMD Performance Regression Benchmarks
 *
 * Benchmarks all 9 SIMD dispatch points (ADM, VIF, Motion, CAMBI)
 * against their C reference implementations. Measures wall-clock time
 * using clock_gettime(CLOCK_MONOTONIC) and reports median-based
 * statistics for regression detection.
 *
 * Usage:
 *   bench_simd_perf [--output FILE] [--reps N] [--help]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>

#include "bench_simd_perf.h"
#include "test_simd_common.h"
#include "config.h"

/* Feature module headers */
#include "feature/integer_adm.h"
#include "feature/integer_vif.h"
#include "feature/integer_motion.h"
#include "feature/common/alignment.h"

#if ARCH_X86
#include "feature/x86/adm_avx2.h"
#include "feature/x86/vif_avx2.h"
#include "feature/x86/motion_avx2.h"
#include "feature/x86/cambi_avx2.h"
#if HAVE_AVX512
#include "feature/x86/vif_avx512.h"
#include "feature/x86/motion_avx512.h"
#endif
#endif

#if ARCH_AARCH64
#include "feature/arm64/adm_neon.h"
#include "feature/arm64/vif_neon.h"
#endif

/* ========== Benchmark Input Dimensions ========== */
#define BENCH_W 1920
#define BENCH_H 1080

/* ========== Maximum benchmark results ========== */
#define MAX_RESULTS 64

/* ========================================================================== */
/* ADM dwt2_8 — C Reference Implementation (from test_simd_adm.c)            */
/* ========================================================================== */

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

/* ========== ADM index generation (mirror boundary) ========== */

static void generate_indices(int **ind, int full_size) {
    const int half = (full_size + 1) / 2;
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
        if (i == 0) {
            idx0 = 1; idx1 = 0; idx2 = 1; idx3 = 2;
        }
        ind[0][i] = idx0;
        ind[1][i] = idx1;
        ind[2][i] = idx2;
        ind[3][i] = idx3;
    }
}

/* ========== VIF C Reference Implementations ========== */

static void ref_pad_top_and_bottom(VifBuffer buf, unsigned h, int fwidth)
{
    const unsigned fwidth_half = fwidth / 2;
    unsigned char *ref = buf.ref;
    unsigned char *dis = buf.dis;
    for (unsigned i = 1; i <= fwidth_half; ++i) {
        size_t offset = buf.stride * i;
        memcpy(ref - offset, ref + offset, buf.stride);
        memcpy(dis - offset, dis + offset, buf.stride);
        memcpy(ref + buf.stride * (h - 1) + buf.stride * i,
               ref + buf.stride * (h - 1) - buf.stride * i,
               buf.stride);
        memcpy(dis + buf.stride * (h - 1) + buf.stride * i,
               dis + buf.stride * (h - 1) - buf.stride * i,
               buf.stride);
    }
}

static void ref_decimate_and_pad(VifBuffer buf, unsigned w, unsigned h, int scale)
{
    uint16_t *ref = buf.ref;
    uint16_t *dis = buf.dis;
    const ptrdiff_t stride = buf.stride / sizeof(uint16_t);
    const ptrdiff_t mu_stride = buf.stride_16 / sizeof(uint16_t);

    for (unsigned i = 0; i < h / 2; ++i) {
        for (unsigned j = 0; j < w / 2; ++j) {
            ref[i * stride + j] = buf.mu1[(i * 2) * mu_stride + (j * 2)];
            dis[i * stride + j] = buf.mu2[(i * 2) * mu_stride + (j * 2)];
        }
    }
    ref_pad_top_and_bottom(buf, h / 2, vif_filter1d_width[scale]);
}

static void ref_subsample_rd_8(VifBuffer buf, unsigned w, unsigned h)
{
    const unsigned fwidth = vif_filter1d_width[1];
    const uint16_t *vif_filt_s1 = vif_filter1d_table[1];

    for (unsigned i = 0; i < h; ++i) {
        for (unsigned j = 0; j < w; ++j) {
            uint32_t accum_ref = 0;
            uint32_t accum_dis = 0;
            for (unsigned fi = 0; fi < fwidth; ++fi) {
                int ii = i - fwidth / 2;
                int ii_check = ii + fi;
                const uint16_t fcoeff = vif_filt_s1[fi];
                const uint8_t *ref_data = (uint8_t *)buf.ref;
                const uint8_t *dis_data = (uint8_t *)buf.dis;
                accum_ref += fcoeff * (uint32_t)ref_data[ii_check * buf.stride + j];
                accum_dis += fcoeff * (uint32_t)dis_data[ii_check * buf.stride + j];
            }
            buf.tmp.ref_convol[j] = (accum_ref + 128) >> 8;
            buf.tmp.dis_convol[j] = (accum_dis + 128) >> 8;
        }

        PADDING_SQ_DATA_2(buf, w, fwidth / 2);

        for (unsigned j = 0; j < w; ++j) {
            uint32_t accum_ref = 0;
            uint32_t accum_dis = 0;
            for (unsigned fj = 0; fj < fwidth; ++fj) {
                int jj = j - fwidth / 2;
                int jj_check = jj + fj;
                const uint16_t fcoeff = vif_filt_s1[fj];
                accum_ref += fcoeff * buf.tmp.ref_convol[jj_check];
                accum_dis += fcoeff * buf.tmp.dis_convol[jj_check];
            }
            const ptrdiff_t stride = buf.stride_16 / sizeof(uint16_t);
            buf.mu1[i * stride + j] = (uint16_t)((accum_ref + 32768) >> 16);
            buf.mu2[i * stride + j] = (uint16_t)((accum_dis + 32768) >> 16);
        }
    }
    ref_decimate_and_pad(buf, w, h, 0);
}

static void ref_subsample_rd_16(VifBuffer buf, unsigned w, unsigned h,
    int scale, int bpc)
{
    const unsigned fwidth = vif_filter1d_width[scale + 1];
    const uint16_t *vif_filt = vif_filter1d_table[scale + 1];
    int32_t add_shift_round_VP, shift_VP;

    if (scale == 0) {
        add_shift_round_VP = 1 << (bpc - 1);
        shift_VP = bpc;
    } else {
        add_shift_round_VP = 32768;
        shift_VP = 16;
    }

    for (unsigned i = 0; i < h; ++i) {
        for (unsigned j = 0; j < w; ++j) {
            uint32_t accum_ref = 0;
            uint32_t accum_dis = 0;
            for (unsigned fi = 0; fi < fwidth; ++fi) {
                int ii = i - fwidth / 2;
                int ii_check = ii + fi;
                const uint16_t fcoeff = vif_filt[fi];
                const ptrdiff_t stride = buf.stride / sizeof(uint16_t);
                uint16_t *ref_data = buf.ref;
                uint16_t *dis_data = buf.dis;
                accum_ref += fcoeff * ((uint32_t)ref_data[ii_check * stride + j]);
                accum_dis += fcoeff * ((uint32_t)dis_data[ii_check * stride + j]);
            }
            buf.tmp.ref_convol[j] = (uint16_t)((accum_ref + add_shift_round_VP) >> shift_VP);
            buf.tmp.dis_convol[j] = (uint16_t)((accum_dis + add_shift_round_VP) >> shift_VP);
        }

        PADDING_SQ_DATA_2(buf, w, fwidth / 2);

        for (unsigned j = 0; j < w; ++j) {
            uint32_t accum_ref = 0;
            uint32_t accum_dis = 0;
            for (unsigned fj = 0; fj < fwidth; ++fj) {
                int jj = j - fwidth / 2;
                int jj_check = jj + fj;
                const uint16_t fcoeff = vif_filt[fj];
                accum_ref += fcoeff * ((uint32_t)buf.tmp.ref_convol[jj_check]);
                accum_dis += fcoeff * ((uint32_t)buf.tmp.dis_convol[jj_check]);
            }
            const ptrdiff_t stride = buf.stride_16 / sizeof(uint16_t);
            buf.mu1[i * stride + j] = (uint16_t)((accum_ref + 32768) >> 16);
            buf.mu2[i * stride + j] = (uint16_t)((accum_dis + 32768) >> 16);
        }
    }
    ref_decimate_and_pad(buf, w, h, scale);
}

/* ========== Motion C Reference ========== */

static void ref_x_convolution_16(const uint16_t *src, uint16_t *dst,
    unsigned width, unsigned height, ptrdiff_t src_stride, ptrdiff_t dst_stride)
{
    const unsigned radius = filter_width / 2;
    const unsigned left_edge = vmaf_ceiln(radius, 1);
    const unsigned right_edge = vmaf_floorn(width - (filter_width - radius), 1);
    const unsigned shift_add_round = 32768;

    uint16_t *src_p = (uint16_t *)src + (left_edge - radius);
    for (unsigned i = 0; i < height; ++i) {
        for (unsigned j = 0; j < left_edge; j++) {
            dst[i * dst_stride + j] =
                (edge_16(true, src, width, height, src_stride, i, j) +
                 shift_add_round) >> 16;
        }

        uint16_t *src_p1 = src_p;
        for (unsigned j = left_edge; j < right_edge; j++) {
            uint32_t accum = 0;
            uint16_t *src_p2 = src_p1;
            for (int k = 0; k < filter_width; ++k) {
                accum += filter[k] * (*src_p2);
                src_p2++;
            }
            src_p1++;
            dst[i * dst_stride + j] = (accum + shift_add_round) >> 16;
        }

        for (unsigned j = right_edge; j < width; j++) {
            dst[i * dst_stride + j] =
                (edge_16(true, src, width, height, src_stride, i, j) +
                 shift_add_round) >> 16;
        }

        src_p += src_stride;
    }
}

/* ========== CAMBI C Reference ========== */

static void ref_increment_range(uint16_t *arr, int left, int right) {
    for (int i = left; i < right; i++) {
        arr[i]++;
    }
}

static void ref_decrement_range(uint16_t *arr, int left, int right) {
    for (int i = left; i < right; i++) {
        arr[i]--;
    }
}

static void ref_get_derivative_data_for_row(const uint16_t *image_data,
    uint16_t *derivative_buffer, int width, int height, int row, int stride)
{
    for (int col = 0; col < width; col++) {
        int horizontal_derivative = (col == width - 1 ||
            image_data[row * stride + col] == image_data[row * stride + col + 1]);
        int vertical_derivative = (row == height - 1 ||
            image_data[row * stride + col] == image_data[(row + 1) * stride + col]);
        derivative_buffer[col] = horizontal_derivative && vertical_derivative;
    }
}

/* ========== Log table generator (for VIF statistic functions) ========== */

static void bench_log_generate(uint16_t *log2_table)
{
    for (unsigned i = 32767; i < 65536; ++i) {
        log2_table[i] = (uint16_t)round(log2f((float)i) * 2048);
    }
}

/* ========================================================================== */
/* ADM Benchmark Context                                                      */
/* ========================================================================== */

typedef struct {
    uint8_t *src;
    adm_dwt_band_t dst;
    AdmBuffer buf;
    int w, h, src_stride, dst_stride;
    void *ind_y_mem;
    void *ind_x_mem;
    void *tmp_mem;
    void *band_mem;
    void *src_mem;
} AdmBenchCtx;

static int adm_bench_ctx_init(AdmBenchCtx *ctx, int w, int h) {
    memset(ctx, 0, sizeof(*ctx));
    ctx->w = w;
    ctx->h = h;

    int out_w = (w + 1) / 2;
    int out_h = (h + 1) / 2;
    ctx->src_stride = (int)ALIGN_CEIL(w);
    ctx->dst_stride = (int)(ALIGN_CEIL(out_w * sizeof(int16_t)) / sizeof(int16_t));

    /* Allocate source */
    size_t src_size = (size_t)ctx->src_stride * h;
    ctx->src_mem = aligned_malloc(src_size, MAX_ALIGN);
    if (!ctx->src_mem) return -1;
    generate_input_8((uint8_t *)ctx->src_mem, w, h, ctx->src_stride, INPUT_RANDOM_42);
    ctx->src = (uint8_t *)ctx->src_mem;

    /* Index arrays */
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

    /* tmp_ref buffer */
    size_t tmp_size = ALIGN_CEIL(w * sizeof(int16_t) * 2 + MAX_ALIGN);
    ctx->tmp_mem = aligned_malloc(tmp_size, MAX_ALIGN);
    if (!ctx->tmp_mem) return -1;
    ctx->buf.tmp_ref = ctx->tmp_mem;

    /* Output bands */
    size_t stride_bytes = ALIGN_CEIL(out_w * sizeof(int16_t));
    size_t band_size = stride_bytes * out_h;
    size_t total = band_size * 4;
    ctx->band_mem = aligned_malloc(total, MAX_ALIGN);
    if (!ctx->band_mem) return -1;
    memset(ctx->band_mem, 0, total);

    p = (char *)ctx->band_mem;
    ctx->dst.band_a = (int16_t *)p; p += band_size;
    ctx->dst.band_v = (int16_t *)p; p += band_size;
    ctx->dst.band_h = (int16_t *)p; p += band_size;
    ctx->dst.band_d = (int16_t *)p;

    return 0;
}

static void adm_bench_ctx_free(AdmBenchCtx *ctx) {
    if (ctx->src_mem)   aligned_free(ctx->src_mem);
    if (ctx->ind_y_mem) aligned_free(ctx->ind_y_mem);
    if (ctx->ind_x_mem) aligned_free(ctx->ind_x_mem);
    if (ctx->tmp_mem)   aligned_free(ctx->tmp_mem);
    if (ctx->band_mem)  aligned_free(ctx->band_mem);
}

/* ADM benchmark wrappers */
static void bench_adm_dwt2_8_c(void *ctx) {
    AdmBenchCtx *c = (AdmBenchCtx *)ctx;
    ref_adm_dwt2_8(c->src, &c->dst, &c->buf, c->w, c->h,
                    c->src_stride, c->dst_stride);
}

#if ARCH_X86
static void bench_adm_dwt2_8_avx2(void *ctx) {
    AdmBenchCtx *c = (AdmBenchCtx *)ctx;
    adm_dwt2_8_avx2(c->src, &c->dst, &c->buf, c->w, c->h,
                     c->src_stride, c->dst_stride);
}
#endif

#if ARCH_AARCH64
static void bench_adm_dwt2_8_neon(void *ctx) {
    AdmBenchCtx *c = (AdmBenchCtx *)ctx;
    adm_dwt2_8_neon(c->src, &c->dst, &c->buf, c->w, c->h,
                     c->src_stride, c->dst_stride);
}
#endif

/* ========================================================================== */
/* VIF Benchmark Context                                                      */
/* ========================================================================== */

typedef struct {
    void *alloc;
    size_t alloc_sz;
    VifBuffer buf;
} VifBenchBuf;

static int vif_bench_buf_init(VifBenchBuf *vb, unsigned w, unsigned h, int hbd) {
    memset(vb, 0, sizeof(*vb));

    vb->buf.stride = ALIGN_CEIL(w << hbd);
    vb->buf.stride_16 = ALIGN_CEIL(w * sizeof(uint16_t));
    vb->buf.stride_32 = ALIGN_CEIL(w * sizeof(uint32_t));
    vb->buf.stride_tmp = ALIGN_CEIL((MAX_ALIGN + w + MAX_ALIGN) * sizeof(uint32_t));

    const size_t frame_size = vb->buf.stride * h;
    const size_t pad_size = vb->buf.stride * 8;
    const size_t data_sz =
        2 * (pad_size + frame_size + pad_size) +
        2 * (h * vb->buf.stride_16) +
        5 * vb->buf.stride_32 +
        7 * vb->buf.stride_tmp;

    vb->alloc_sz = data_sz;
    vb->alloc = aligned_malloc(data_sz, MAX_ALIGN);
    if (!vb->alloc) return -1;
    memset(vb->alloc, 0, data_sz);

    void *data = vb->alloc;
    vb->buf.data = data; data += pad_size;
    vb->buf.ref = data;  data += frame_size + pad_size + pad_size;
    vb->buf.dis = data;  data += frame_size + pad_size;
    vb->buf.mu1 = data;  data += h * vb->buf.stride_16;
    vb->buf.mu2 = data;  data += h * vb->buf.stride_16;
    vb->buf.mu1_32 = data; data += vb->buf.stride_32;
    vb->buf.mu2_32 = data; data += vb->buf.stride_32;
    vb->buf.ref_sq = data; data += vb->buf.stride_32;
    vb->buf.dis_sq = data; data += vb->buf.stride_32;
    vb->buf.ref_dis = data; data += vb->buf.stride_32;

    vb->buf.tmp.mu1 = (uint32_t *)data + MAX_ALIGN;
    data += vb->buf.stride_tmp;
    vb->buf.tmp.mu2 = (uint32_t *)data + MAX_ALIGN;
    data += vb->buf.stride_tmp;
    vb->buf.tmp.ref = (uint32_t *)data + MAX_ALIGN;
    data += vb->buf.stride_tmp;
    vb->buf.tmp.dis = (uint32_t *)data + MAX_ALIGN;
    data += vb->buf.stride_tmp;
    vb->buf.tmp.ref_dis = (uint32_t *)data + MAX_ALIGN;
    data += vb->buf.stride_tmp;
    vb->buf.tmp.ref_convol = (uint32_t *)data + MAX_ALIGN;
    data += vb->buf.stride_tmp;
    vb->buf.tmp.dis_convol = (uint32_t *)data + MAX_ALIGN;

    return 0;
}

static void vif_bench_buf_free(VifBenchBuf *vb) {
    if (vb->alloc) { aligned_free(vb->alloc); vb->alloc = NULL; }
}

/* ========== VIF subsample_rd_8 ========== */

typedef struct {
    VifBenchBuf tb;
    void *orig_data;     /* saved copy of initial data for reset */
    unsigned w, h;
} VifSubsample8Ctx;

static int vif_sub8_ctx_init(VifSubsample8Ctx *ctx, unsigned w, unsigned h) {
    memset(ctx, 0, sizeof(*ctx));
    ctx->w = w;
    ctx->h = h;
    if (vif_bench_buf_init(&ctx->tb, w, h, 0) != 0) return -1;

    /* Fill with random data */
    uint8_t *ref = (uint8_t *)ctx->tb.buf.ref;
    uint8_t *dis = (uint8_t *)ctx->tb.buf.dis;
    generate_input_8(ref, w, h, ctx->tb.buf.stride, INPUT_RANDOM_42);
    {
        uint32_t state = 4242;
        for (unsigned i = 0; i < h; i++)
            for (unsigned j = 0; j < w; j++)
                dis[i * ctx->tb.buf.stride + j] = (uint8_t)(prng_next(&state) & 0xFF);
    }
    ref_pad_top_and_bottom(ctx->tb.buf, h, vif_filter1d_width[0]);

    /* Save a copy for reset between iterations */
    ctx->orig_data = aligned_malloc(ctx->tb.alloc_sz, MAX_ALIGN);
    if (!ctx->orig_data) return -1;
    memcpy(ctx->orig_data, ctx->tb.alloc, ctx->tb.alloc_sz);

    return 0;
}

static void vif_sub8_ctx_free(VifSubsample8Ctx *ctx) {
    vif_bench_buf_free(&ctx->tb);
    if (ctx->orig_data) aligned_free(ctx->orig_data);
}

static void vif_sub8_ctx_reset(VifSubsample8Ctx *ctx) {
    memcpy(ctx->tb.alloc, ctx->orig_data, ctx->tb.alloc_sz);
}

static void bench_vif_subsample_rd_8_c(void *v) {
    VifSubsample8Ctx *ctx = (VifSubsample8Ctx *)v;
    vif_sub8_ctx_reset(ctx);
    ref_subsample_rd_8(ctx->tb.buf, ctx->w, ctx->h);
}

#if ARCH_X86
static void bench_vif_subsample_rd_8_avx2(void *v) {
    VifSubsample8Ctx *ctx = (VifSubsample8Ctx *)v;
    vif_sub8_ctx_reset(ctx);
    vif_subsample_rd_8_avx2(ctx->tb.buf, ctx->w, ctx->h);
}
#if HAVE_AVX512
static void bench_vif_subsample_rd_8_avx512(void *v) {
    VifSubsample8Ctx *ctx = (VifSubsample8Ctx *)v;
    vif_sub8_ctx_reset(ctx);
    vif_subsample_rd_8_avx512(ctx->tb.buf, ctx->w, ctx->h);
}
#endif
#endif

#if ARCH_AARCH64
static void bench_vif_subsample_rd_8_neon(void *v) {
    VifSubsample8Ctx *ctx = (VifSubsample8Ctx *)v;
    vif_sub8_ctx_reset(ctx);
    vif_subsample_rd_8_neon(ctx->tb.buf, ctx->w, ctx->h);
}
#endif

/* ========== VIF subsample_rd_16 ========== */

typedef struct {
    VifBenchBuf tb;
    void *orig_data;
    unsigned w, h;
    int scale, bpc;
} VifSubsample16Ctx;

static int vif_sub16_ctx_init(VifSubsample16Ctx *ctx, unsigned w, unsigned h,
                               int scale, int bpc) {
    memset(ctx, 0, sizeof(*ctx));
    ctx->w = w;
    ctx->h = h;
    ctx->scale = scale;
    ctx->bpc = bpc;
    uint16_t max_val = (1 << bpc) - 1;

    if (vif_bench_buf_init(&ctx->tb, w, h, 1) != 0) return -1;

    uint16_t *ref = (uint16_t *)ctx->tb.buf.ref;
    uint16_t *dis = (uint16_t *)ctx->tb.buf.dis;
    ptrdiff_t stride = ctx->tb.buf.stride / sizeof(uint16_t);
    generate_input_16(ref, w, h, stride, INPUT_RANDOM_42, max_val);
    {
        uint32_t state = 4242;
        for (unsigned i = 0; i < h; i++)
            for (unsigned j = 0; j < w; j++)
                dis[i * stride + j] = (uint16_t)(prng_next(&state) % (max_val + 1));
    }

    int fwidth = vif_filter1d_width[scale + 1];
    ref_pad_top_and_bottom(ctx->tb.buf, h, fwidth);

    ctx->orig_data = aligned_malloc(ctx->tb.alloc_sz, MAX_ALIGN);
    if (!ctx->orig_data) return -1;
    memcpy(ctx->orig_data, ctx->tb.alloc, ctx->tb.alloc_sz);

    return 0;
}

static void vif_sub16_ctx_free(VifSubsample16Ctx *ctx) {
    vif_bench_buf_free(&ctx->tb);
    if (ctx->orig_data) aligned_free(ctx->orig_data);
}

static void vif_sub16_ctx_reset(VifSubsample16Ctx *ctx) {
    memcpy(ctx->tb.alloc, ctx->orig_data, ctx->tb.alloc_sz);
}

static void bench_vif_subsample_rd_16_c(void *v) {
    VifSubsample16Ctx *ctx = (VifSubsample16Ctx *)v;
    vif_sub16_ctx_reset(ctx);
    ref_subsample_rd_16(ctx->tb.buf, ctx->w, ctx->h, ctx->scale, ctx->bpc);
}

#if ARCH_X86
static void bench_vif_subsample_rd_16_avx2(void *v) {
    VifSubsample16Ctx *ctx = (VifSubsample16Ctx *)v;
    vif_sub16_ctx_reset(ctx);
    vif_subsample_rd_16_avx2(ctx->tb.buf, ctx->w, ctx->h, ctx->scale, ctx->bpc);
}
#if HAVE_AVX512
static void bench_vif_subsample_rd_16_avx512(void *v) {
    VifSubsample16Ctx *ctx = (VifSubsample16Ctx *)v;
    vif_sub16_ctx_reset(ctx);
    vif_subsample_rd_16_avx512(ctx->tb.buf, ctx->w, ctx->h, ctx->scale, ctx->bpc);
}
#endif
#endif

#if ARCH_AARCH64
static void bench_vif_subsample_rd_16_neon(void *v) {
    VifSubsample16Ctx *ctx = (VifSubsample16Ctx *)v;
    vif_sub16_ctx_reset(ctx);
    vif_subsample_rd_16_neon(ctx->tb.buf, ctx->w, ctx->h, ctx->scale, ctx->bpc);
}
#endif

/* ========== VIF vif_statistic_8 ========== */

typedef struct {
    VifBenchBuf tb;
    VifPublicState *state;
    void *orig_data;
    unsigned w, h;
    float num, den;
} VifStat8Ctx;

static int vif_stat8_ctx_init(VifStat8Ctx *ctx, unsigned w, unsigned h) {
    memset(ctx, 0, sizeof(*ctx));
    ctx->w = w;
    ctx->h = h;

    ctx->state = aligned_malloc(sizeof(VifPublicState), MAX_ALIGN);
    if (!ctx->state) return -1;
    memset(ctx->state, 0, sizeof(*ctx->state));
    ctx->state->vif_enhn_gain_limit = 100.0;
    bench_log_generate(ctx->state->log2_table);

    if (vif_bench_buf_init(&ctx->tb, w, h, 0) != 0) return -1;

    uint8_t *ref = (uint8_t *)ctx->tb.buf.ref;
    uint8_t *dis = (uint8_t *)ctx->tb.buf.dis;
    generate_input_8(ref, w, h, ctx->tb.buf.stride, INPUT_RANDOM_42);
    {
        uint32_t state = 4242;
        for (unsigned i = 0; i < h; i++)
            for (unsigned j = 0; j < w; j++)
                dis[i * ctx->tb.buf.stride + j] = (uint8_t)(prng_next(&state) & 0xFF);
    }
    ref_pad_top_and_bottom(ctx->tb.buf, h, vif_filter1d_width[0]);

    ctx->state->buf = ctx->tb.buf;

    ctx->orig_data = aligned_malloc(ctx->tb.alloc_sz, MAX_ALIGN);
    if (!ctx->orig_data) return -1;
    memcpy(ctx->orig_data, ctx->tb.alloc, ctx->tb.alloc_sz);

    return 0;
}

static void vif_stat8_ctx_free(VifStat8Ctx *ctx) {
    vif_bench_buf_free(&ctx->tb);
    if (ctx->state) aligned_free(ctx->state);
    if (ctx->orig_data) aligned_free(ctx->orig_data);
}

static void vif_stat8_ctx_reset(VifStat8Ctx *ctx) {
    memcpy(ctx->tb.alloc, ctx->orig_data, ctx->tb.alloc_sz);
    ctx->state->buf = ctx->tb.buf;
    ctx->num = 0;
    ctx->den = 0;
}

static void bench_vif_statistic_8_c(void *v) {
    VifStat8Ctx *ctx = (VifStat8Ctx *)v;
    vif_stat8_ctx_reset(ctx);
    vif_statistic_8(ctx->state, &ctx->num, &ctx->den, ctx->w, ctx->h);
}

#if ARCH_X86
static void bench_vif_statistic_8_avx2(void *v) {
    VifStat8Ctx *ctx = (VifStat8Ctx *)v;
    vif_stat8_ctx_reset(ctx);
    vif_statistic_8_avx2(ctx->state, &ctx->num, &ctx->den, ctx->w, ctx->h);
}
#if HAVE_AVX512
static void bench_vif_statistic_8_avx512(void *v) {
    VifStat8Ctx *ctx = (VifStat8Ctx *)v;
    vif_stat8_ctx_reset(ctx);
    vif_statistic_8_avx512(ctx->state, &ctx->num, &ctx->den, ctx->w, ctx->h);
}
#endif
#endif

#if ARCH_AARCH64
static void bench_vif_statistic_8_neon(void *v) {
    VifStat8Ctx *ctx = (VifStat8Ctx *)v;
    vif_stat8_ctx_reset(ctx);
    vif_statistic_8_neon(ctx->state, &ctx->num, &ctx->den, ctx->w, ctx->h);
}
#endif

/* ========== VIF vif_statistic_16 ========== */

typedef struct {
    VifBenchBuf tb;
    VifPublicState *state;
    void *orig_data;
    unsigned w, h;
    int bpc, scale;
    float num, den;
} VifStat16Ctx;

static int vif_stat16_ctx_init(VifStat16Ctx *ctx, unsigned w, unsigned h,
                                int bpc, int scale) {
    memset(ctx, 0, sizeof(*ctx));
    ctx->w = w;
    ctx->h = h;
    ctx->bpc = bpc;
    ctx->scale = scale;
    uint16_t max_val = (uint16_t)((1 << bpc) - 1);

    ctx->state = aligned_malloc(sizeof(VifPublicState), MAX_ALIGN);
    if (!ctx->state) return -1;
    memset(ctx->state, 0, sizeof(*ctx->state));
    ctx->state->vif_enhn_gain_limit = 100.0;
    bench_log_generate(ctx->state->log2_table);

    if (vif_bench_buf_init(&ctx->tb, w, h, 1) != 0) return -1;

    uint16_t *ref = (uint16_t *)ctx->tb.buf.ref;
    uint16_t *dis = (uint16_t *)ctx->tb.buf.dis;
    ptrdiff_t stride = ctx->tb.buf.stride / sizeof(uint16_t);
    generate_input_16(ref, w, h, stride, INPUT_RANDOM_42, max_val);
    {
        uint32_t state = 4242;
        for (unsigned i = 0; i < h; i++)
            for (unsigned j = 0; j < w; j++)
                dis[i * stride + j] = (uint16_t)(prng_next(&state) % (max_val + 1));
    }
    ref_pad_top_and_bottom(ctx->tb.buf, h, vif_filter1d_width[0]);

    ctx->state->buf = ctx->tb.buf;

    ctx->orig_data = aligned_malloc(ctx->tb.alloc_sz, MAX_ALIGN);
    if (!ctx->orig_data) return -1;
    memcpy(ctx->orig_data, ctx->tb.alloc, ctx->tb.alloc_sz);

    return 0;
}

static void vif_stat16_ctx_free(VifStat16Ctx *ctx) {
    vif_bench_buf_free(&ctx->tb);
    if (ctx->state) aligned_free(ctx->state);
    if (ctx->orig_data) aligned_free(ctx->orig_data);
}

static void vif_stat16_ctx_reset(VifStat16Ctx *ctx) {
    memcpy(ctx->tb.alloc, ctx->orig_data, ctx->tb.alloc_sz);
    ctx->state->buf = ctx->tb.buf;
    ctx->num = 0;
    ctx->den = 0;
}

static void bench_vif_statistic_16_c(void *v) {
    VifStat16Ctx *ctx = (VifStat16Ctx *)v;
    vif_stat16_ctx_reset(ctx);
    vif_statistic_16(ctx->state, &ctx->num, &ctx->den, ctx->w, ctx->h,
                     ctx->bpc, ctx->scale);
}

#if ARCH_X86
static void bench_vif_statistic_16_avx2(void *v) {
    VifStat16Ctx *ctx = (VifStat16Ctx *)v;
    vif_stat16_ctx_reset(ctx);
    vif_statistic_16_avx2(ctx->state, &ctx->num, &ctx->den, ctx->w, ctx->h,
                          ctx->bpc, ctx->scale);
}
#if HAVE_AVX512
static void bench_vif_statistic_16_avx512(void *v) {
    VifStat16Ctx *ctx = (VifStat16Ctx *)v;
    vif_stat16_ctx_reset(ctx);
    vif_statistic_16_avx512(ctx->state, &ctx->num, &ctx->den, ctx->w, ctx->h,
                            ctx->bpc, ctx->scale);
}
#endif
#endif

#if ARCH_AARCH64
static void bench_vif_statistic_16_neon(void *v) {
    VifStat16Ctx *ctx = (VifStat16Ctx *)v;
    vif_stat16_ctx_reset(ctx);
    vif_statistic_16_neon(ctx->state, &ctx->num, &ctx->den, ctx->w, ctx->h,
                          ctx->bpc, ctx->scale);
}
#endif

/* ========================================================================== */
/* Motion Benchmark Context                                                   */
/* ========================================================================== */

typedef struct {
    uint16_t *src;
    uint16_t *dst;
    unsigned w, h;
    ptrdiff_t stride;
    void *src_mem;
    void *dst_mem;
} MotionBenchCtx;

static int motion_bench_ctx_init(MotionBenchCtx *ctx, unsigned w, unsigned h) {
    memset(ctx, 0, sizeof(*ctx));
    ctx->w = w;
    ctx->h = h;
    ctx->stride = (ptrdiff_t)(ALIGN_CEIL(w * sizeof(uint16_t)) / sizeof(uint16_t));

    size_t buf_size = ctx->stride * h * sizeof(uint16_t);
    ctx->src_mem = aligned_malloc(buf_size, MAX_ALIGN);
    ctx->dst_mem = aligned_malloc(buf_size, MAX_ALIGN);
    if (!ctx->src_mem || !ctx->dst_mem) return -1;

    ctx->src = (uint16_t *)ctx->src_mem;
    ctx->dst = (uint16_t *)ctx->dst_mem;

    generate_input_16(ctx->src, w, h, ctx->stride, INPUT_RANDOM_42, 65535);
    memset(ctx->dst, 0, buf_size);

    return 0;
}

static void motion_bench_ctx_free(MotionBenchCtx *ctx) {
    if (ctx->src_mem) aligned_free(ctx->src_mem);
    if (ctx->dst_mem) aligned_free(ctx->dst_mem);
}

static void bench_motion_x_conv_16_c(void *v) {
    MotionBenchCtx *ctx = (MotionBenchCtx *)v;
    ref_x_convolution_16(ctx->src, ctx->dst, ctx->w, ctx->h,
                         ctx->stride, ctx->stride);
}

#if ARCH_X86
static void bench_motion_x_conv_16_avx2(void *v) {
    MotionBenchCtx *ctx = (MotionBenchCtx *)v;
    x_convolution_16_avx2(ctx->src, ctx->dst, ctx->w, ctx->h,
                          ctx->stride, ctx->stride);
}
#if HAVE_AVX512
static void bench_motion_x_conv_16_avx512(void *v) {
    MotionBenchCtx *ctx = (MotionBenchCtx *)v;
    x_convolution_16_avx512(ctx->src, ctx->dst, ctx->w, ctx->h,
                            ctx->stride, ctx->stride);
}
#endif
#endif

/* ========================================================================== */
/* CAMBI Benchmark Contexts                                                   */
/* ========================================================================== */

/* increment_range / decrement_range context */
typedef struct {
    uint16_t *arr;
    uint16_t *orig_arr;
    int len;
    void *arr_mem;
    void *orig_mem;
} CambiRangeCtx;

static int cambi_range_ctx_init(CambiRangeCtx *ctx, int w, int h) {
    memset(ctx, 0, sizeof(*ctx));
    ctx->len = w * h;
    size_t buf_size = ctx->len * sizeof(uint16_t);

    ctx->arr_mem = aligned_malloc(buf_size, MAX_ALIGN);
    ctx->orig_mem = aligned_malloc(buf_size, MAX_ALIGN);
    if (!ctx->arr_mem || !ctx->orig_mem) return -1;

    ctx->arr = (uint16_t *)ctx->arr_mem;
    ctx->orig_arr = (uint16_t *)ctx->orig_mem;

    generate_input_16(ctx->orig_arr, w, h, w, INPUT_RANDOM_42, 1023);
    /* Ensure minimum value of 1 for decrement */
    for (int i = 0; i < ctx->len; i++) {
        if (ctx->orig_arr[i] == 0) ctx->orig_arr[i] = 1;
    }
    memcpy(ctx->arr, ctx->orig_arr, buf_size);

    return 0;
}

static void cambi_range_ctx_free(CambiRangeCtx *ctx) {
    if (ctx->arr_mem) aligned_free(ctx->arr_mem);
    if (ctx->orig_mem) aligned_free(ctx->orig_mem);
}

static void cambi_range_ctx_reset(CambiRangeCtx *ctx) {
    memcpy(ctx->arr, ctx->orig_arr, ctx->len * sizeof(uint16_t));
}

static void bench_cambi_increment_c(void *v) {
    CambiRangeCtx *ctx = (CambiRangeCtx *)v;
    cambi_range_ctx_reset(ctx);
    ref_increment_range(ctx->arr, 0, ctx->len);
}

static void bench_cambi_decrement_c(void *v) {
    CambiRangeCtx *ctx = (CambiRangeCtx *)v;
    cambi_range_ctx_reset(ctx);
    ref_decrement_range(ctx->arr, 0, ctx->len);
}

#if ARCH_X86
static void bench_cambi_increment_avx2(void *v) {
    CambiRangeCtx *ctx = (CambiRangeCtx *)v;
    cambi_range_ctx_reset(ctx);
    cambi_increment_range_avx2(ctx->arr, 0, ctx->len);
}

static void bench_cambi_decrement_avx2(void *v) {
    CambiRangeCtx *ctx = (CambiRangeCtx *)v;
    cambi_range_ctx_reset(ctx);
    cambi_decrement_range_avx2(ctx->arr, 0, ctx->len);
}
#endif

/* get_derivative_data_for_row context */
typedef struct {
    uint16_t *image;
    uint16_t *deriv;
    int w, h;
    ptrdiff_t stride;
    void *img_mem;
    void *deriv_mem;
} CambiDerivCtx;

static int cambi_deriv_ctx_init(CambiDerivCtx *ctx, int w, int h) {
    memset(ctx, 0, sizeof(*ctx));
    ctx->w = w;
    ctx->h = h;
    ctx->stride = (ptrdiff_t)(ALIGN_CEIL(w * sizeof(uint16_t)) / sizeof(uint16_t));

    size_t img_size = ctx->stride * h * sizeof(uint16_t);
    size_t deriv_size = w * sizeof(uint16_t);

    ctx->img_mem = aligned_malloc(img_size, MAX_ALIGN);
    ctx->deriv_mem = aligned_malloc(deriv_size, MAX_ALIGN);
    if (!ctx->img_mem || !ctx->deriv_mem) return -1;

    ctx->image = (uint16_t *)ctx->img_mem;
    ctx->deriv = (uint16_t *)ctx->deriv_mem;

    generate_input_16(ctx->image, w, h, ctx->stride, INPUT_RANDOM_42, 1023);
    memset(ctx->deriv, 0, deriv_size);

    return 0;
}

static void cambi_deriv_ctx_free(CambiDerivCtx *ctx) {
    if (ctx->img_mem) aligned_free(ctx->img_mem);
    if (ctx->deriv_mem) aligned_free(ctx->deriv_mem);
}

static void bench_cambi_derivative_c(void *v) {
    CambiDerivCtx *ctx = (CambiDerivCtx *)v;
    /* Benchmark over all rows to get meaningful timing */
    for (int row = 0; row < ctx->h; row++) {
        ref_get_derivative_data_for_row(ctx->image, ctx->deriv,
            ctx->w, ctx->h, row, (int)ctx->stride);
    }
}

#if ARCH_X86
static void bench_cambi_derivative_avx2(void *v) {
    CambiDerivCtx *ctx = (CambiDerivCtx *)v;
    for (int row = 0; row < ctx->h; row++) {
        get_derivative_data_for_row_avx2(ctx->image, ctx->deriv,
            ctx->w, ctx->h, row, (int)ctx->stride);
    }
}
#endif

/* ========================================================================== */
/* Main — Run All Benchmarks                                                  */
/* ========================================================================== */

int main(int argc, char **argv) {
    const char *output_file = NULL;
    int num_reps = 7;

    /* Parse arguments */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--output") == 0 && i + 1 < argc)
            output_file = argv[++i];
        else if (strcmp(argv[i], "--reps") == 0 && i + 1 < argc)
            num_reps = atoi(argv[++i]);
        else if (strcmp(argv[i], "--help") == 0) {
            fprintf(stderr,
                "Usage: bench_simd_perf [--output FILE] [--reps N]\n"
                "  --output FILE   Write JSON results to FILE (default: stdout)\n"
                "  --reps N        Number of repetitions per benchmark (default: 7)\n");
            return 0;
        }
    }

    BenchResult results[MAX_RESULTS];
    int n_results = 0;

    fprintf(stderr, "SIMD Performance Benchmark — %dx%d, seed=42\n", BENCH_W, BENCH_H);
    fprintf(stderr, "Repetitions per benchmark: %d\n\n", num_reps);

    /* ====================================================================== */
    /* 1. ADM dwt2_8                                                          */
    /* ====================================================================== */
    fprintf(stderr, "Benchmarking: adm_dwt2_8...\n");
    {
        AdmBenchCtx adm_ctx;
        if (adm_bench_ctx_init(&adm_ctx, BENCH_W, BENCH_H) != 0) {
            fprintf(stderr, "ERROR: Failed to initialize ADM context\n");
            return 1;
        }

        bench_run(&results[n_results++], bench_adm_dwt2_8_c, &adm_ctx,
                  "adm_dwt2_8", "c", num_reps);
#if ARCH_X86
        bench_run(&results[n_results++], bench_adm_dwt2_8_avx2, &adm_ctx,
                  "adm_dwt2_8", "avx2", num_reps);
#endif
#if ARCH_AARCH64
        bench_run(&results[n_results++], bench_adm_dwt2_8_neon, &adm_ctx,
                  "adm_dwt2_8", "neon", num_reps);
#endif
        adm_bench_ctx_free(&adm_ctx);
    }

    /* ====================================================================== */
    /* 2. VIF subsample_rd_8                                                  */
    /* ====================================================================== */
    fprintf(stderr, "Benchmarking: vif_subsample_rd_8...\n");
    {
        VifSubsample8Ctx sub8_ctx;
        if (vif_sub8_ctx_init(&sub8_ctx, BENCH_W, BENCH_H) != 0) {
            fprintf(stderr, "ERROR: Failed to initialize VIF subsample_rd_8 context\n");
            return 1;
        }

        bench_run(&results[n_results++], bench_vif_subsample_rd_8_c, &sub8_ctx,
                  "vif_subsample_rd_8", "c", num_reps);
#if ARCH_X86
        bench_run(&results[n_results++], bench_vif_subsample_rd_8_avx2, &sub8_ctx,
                  "vif_subsample_rd_8", "avx2", num_reps);
#if HAVE_AVX512
        bench_run(&results[n_results++], bench_vif_subsample_rd_8_avx512, &sub8_ctx,
                  "vif_subsample_rd_8", "avx512", num_reps);
#endif
#endif
#if ARCH_AARCH64
        bench_run(&results[n_results++], bench_vif_subsample_rd_8_neon, &sub8_ctx,
                  "vif_subsample_rd_8", "neon", num_reps);
#endif
        vif_sub8_ctx_free(&sub8_ctx);
    }

    /* ====================================================================== */
    /* 3. VIF subsample_rd_16 (scale=0, bpc=10)                               */
    /* ====================================================================== */
    fprintf(stderr, "Benchmarking: vif_subsample_rd_16...\n");
    {
        VifSubsample16Ctx sub16_ctx;
        if (vif_sub16_ctx_init(&sub16_ctx, BENCH_W, BENCH_H, 0, 10) != 0) {
            fprintf(stderr, "ERROR: Failed to initialize VIF subsample_rd_16 context\n");
            return 1;
        }

        bench_run(&results[n_results++], bench_vif_subsample_rd_16_c, &sub16_ctx,
                  "vif_subsample_rd_16", "c", num_reps);
#if ARCH_X86
        bench_run(&results[n_results++], bench_vif_subsample_rd_16_avx2, &sub16_ctx,
                  "vif_subsample_rd_16", "avx2", num_reps);
#if HAVE_AVX512
        bench_run(&results[n_results++], bench_vif_subsample_rd_16_avx512, &sub16_ctx,
                  "vif_subsample_rd_16", "avx512", num_reps);
#endif
#endif
#if ARCH_AARCH64
        bench_run(&results[n_results++], bench_vif_subsample_rd_16_neon, &sub16_ctx,
                  "vif_subsample_rd_16", "neon", num_reps);
#endif
        vif_sub16_ctx_free(&sub16_ctx);
    }

    /* ====================================================================== */
    /* 4. VIF vif_statistic_8                                                 */
    /* ====================================================================== */
    fprintf(stderr, "Benchmarking: vif_statistic_8...\n");
    {
        VifStat8Ctx stat8_ctx;
        if (vif_stat8_ctx_init(&stat8_ctx, BENCH_W, BENCH_H) != 0) {
            fprintf(stderr, "ERROR: Failed to initialize VIF statistic_8 context\n");
            return 1;
        }

        bench_run(&results[n_results++], bench_vif_statistic_8_c, &stat8_ctx,
                  "vif_statistic_8", "c", num_reps);
#if ARCH_X86
        bench_run(&results[n_results++], bench_vif_statistic_8_avx2, &stat8_ctx,
                  "vif_statistic_8", "avx2", num_reps);
#if HAVE_AVX512
        bench_run(&results[n_results++], bench_vif_statistic_8_avx512, &stat8_ctx,
                  "vif_statistic_8", "avx512", num_reps);
#endif
#endif
#if ARCH_AARCH64
        bench_run(&results[n_results++], bench_vif_statistic_8_neon, &stat8_ctx,
                  "vif_statistic_8", "neon", num_reps);
#endif
        vif_stat8_ctx_free(&stat8_ctx);
    }

    /* ====================================================================== */
    /* 5. VIF vif_statistic_16 (bpc=10, scale=0)                              */
    /* ====================================================================== */
    fprintf(stderr, "Benchmarking: vif_statistic_16...\n");
    {
        VifStat16Ctx stat16_ctx;
        if (vif_stat16_ctx_init(&stat16_ctx, BENCH_W, BENCH_H, 10, 0) != 0) {
            fprintf(stderr, "ERROR: Failed to initialize VIF statistic_16 context\n");
            return 1;
        }

        bench_run(&results[n_results++], bench_vif_statistic_16_c, &stat16_ctx,
                  "vif_statistic_16", "c", num_reps);
#if ARCH_X86
        bench_run(&results[n_results++], bench_vif_statistic_16_avx2, &stat16_ctx,
                  "vif_statistic_16", "avx2", num_reps);
#if HAVE_AVX512
        bench_run(&results[n_results++], bench_vif_statistic_16_avx512, &stat16_ctx,
                  "vif_statistic_16", "avx512", num_reps);
#endif
#endif
#if ARCH_AARCH64
        bench_run(&results[n_results++], bench_vif_statistic_16_neon, &stat16_ctx,
                  "vif_statistic_16", "neon", num_reps);
#endif
        vif_stat16_ctx_free(&stat16_ctx);
    }

    /* ====================================================================== */
    /* 6. Motion x_convolution_16                                             */
    /* ====================================================================== */
    fprintf(stderr, "Benchmarking: motion_x_convolution_16...\n");
    {
        MotionBenchCtx motion_ctx;
        if (motion_bench_ctx_init(&motion_ctx, BENCH_W, BENCH_H) != 0) {
            fprintf(stderr, "ERROR: Failed to initialize Motion context\n");
            return 1;
        }

        bench_run(&results[n_results++], bench_motion_x_conv_16_c, &motion_ctx,
                  "motion_x_convolution_16", "c", num_reps);
#if ARCH_X86
        bench_run(&results[n_results++], bench_motion_x_conv_16_avx2, &motion_ctx,
                  "motion_x_convolution_16", "avx2", num_reps);
#if HAVE_AVX512
        bench_run(&results[n_results++], bench_motion_x_conv_16_avx512, &motion_ctx,
                  "motion_x_convolution_16", "avx512", num_reps);
#endif
#endif
        /* Motion has no NEON variant */
        motion_bench_ctx_free(&motion_ctx);
    }

    /* ====================================================================== */
    /* 7. CAMBI increment_range                                               */
    /* ====================================================================== */
    fprintf(stderr, "Benchmarking: cambi_increment_range...\n");
    {
        CambiRangeCtx inc_ctx;
        if (cambi_range_ctx_init(&inc_ctx, BENCH_W, BENCH_H) != 0) {
            fprintf(stderr, "ERROR: Failed to initialize CAMBI increment context\n");
            return 1;
        }

        bench_run(&results[n_results++], bench_cambi_increment_c, &inc_ctx,
                  "cambi_increment_range", "c", num_reps);
#if ARCH_X86
        bench_run(&results[n_results++], bench_cambi_increment_avx2, &inc_ctx,
                  "cambi_increment_range", "avx2", num_reps);
#endif
        /* CAMBI has no NEON variant */
        cambi_range_ctx_free(&inc_ctx);
    }

    /* ====================================================================== */
    /* 8. CAMBI decrement_range                                               */
    /* ====================================================================== */
    fprintf(stderr, "Benchmarking: cambi_decrement_range...\n");
    {
        CambiRangeCtx dec_ctx;
        if (cambi_range_ctx_init(&dec_ctx, BENCH_W, BENCH_H) != 0) {
            fprintf(stderr, "ERROR: Failed to initialize CAMBI decrement context\n");
            return 1;
        }

        bench_run(&results[n_results++], bench_cambi_decrement_c, &dec_ctx,
                  "cambi_decrement_range", "c", num_reps);
#if ARCH_X86
        bench_run(&results[n_results++], bench_cambi_decrement_avx2, &dec_ctx,
                  "cambi_decrement_range", "avx2", num_reps);
#endif
        cambi_range_ctx_free(&dec_ctx);
    }

    /* ====================================================================== */
    /* 9. CAMBI get_derivative_data_for_row                                   */
    /* ====================================================================== */
    fprintf(stderr, "Benchmarking: cambi_derivative_row...\n");
    {
        CambiDerivCtx deriv_ctx;
        if (cambi_deriv_ctx_init(&deriv_ctx, BENCH_W, BENCH_H) != 0) {
            fprintf(stderr, "ERROR: Failed to initialize CAMBI derivative context\n");
            return 1;
        }

        bench_run(&results[n_results++], bench_cambi_derivative_c, &deriv_ctx,
                  "cambi_derivative_row", "c", num_reps);
#if ARCH_X86
        bench_run(&results[n_results++], bench_cambi_derivative_avx2, &deriv_ctx,
                  "cambi_derivative_row", "avx2", num_reps);
#endif
        cambi_deriv_ctx_free(&deriv_ctx);
    }

    /* ====================================================================== */
    /* Output Results                                                         */
    /* ====================================================================== */

    /* Print human-readable table to stderr */
    bench_print_table(results, n_results, stderr);

    /* Write JSON output */
    FILE *out = output_file ? fopen(output_file, "w") : stdout;
    if (!out) {
        perror("fopen");
        return 1;
    }

    fprintf(out, "{\n  \"benchmarks\": [\n");
    for (int i = 0; i < n_results; i++) {
        if (i > 0) fprintf(out, ",\n");
        bench_result_to_json(&results[i], out);
    }
    fprintf(out, "\n  ]\n}\n");

    if (output_file) fclose(out);

    fprintf(stderr, "Total benchmarks run: %d\n", n_results);

    return 0;
}
