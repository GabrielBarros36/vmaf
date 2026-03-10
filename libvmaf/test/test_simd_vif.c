/**
 * SIMD Oracle Tests — VIF Module
 *
 * Tests subsample_rd_8, subsample_rd_16, vif_statistic_8, vif_statistic_16
 * C reference vs AVX2/AVX-512/NEON implementations.
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <stdbool.h>

#include "test.h"
#include "test_simd_common.h"

#include "config.h"
#include "cpu.h"
#include "feature/integer_vif.h"
#include "feature/common/alignment.h"

#if ARCH_X86
#include "feature/x86/vif_avx2.h"
#if HAVE_AVX512
#include "feature/x86/vif_avx512.h"
#endif
#elif ARCH_AARCH64
#include "feature/arm64/vif_neon.h"
#endif

/* ========== Filter constants (from integer_vif.h, included) ========== */

/* ========== C Reference: pad_top_and_bottom ========== */

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

/* ========== C Reference: decimate_and_pad ========== */

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

/* ========== C Reference: subsample_rd_8 ========== */

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
                const uint8_t *ref = (uint8_t *)buf.ref;
                const uint8_t *dis = (uint8_t *)buf.dis;
                accum_ref += fcoeff * (uint32_t)ref[ii_check * buf.stride + j];
                accum_dis += fcoeff * (uint32_t)dis[ii_check * buf.stride + j];
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

/* ========== C Reference: subsample_rd_16 ========== */

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
                uint16_t *ref = buf.ref;
                uint16_t *dis = buf.dis;
                accum_ref += fcoeff * ((uint32_t)ref[ii_check * stride + j]);
                accum_dis += fcoeff * ((uint32_t)dis[ii_check * stride + j]);
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

/* ========== VIF Buffer Allocation Helper ========== */

typedef struct {
    void *alloc;          /* Single allocation for all buffer data */
    size_t alloc_sz;      /* Total allocation size */
    VifBuffer buf;
} VifTestBuf;

static int vif_test_buf_init(VifTestBuf *vb, unsigned w, unsigned h, int hbd)
{
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

    /* tmp pointers are offset by MAX_ALIGN elements for negative indexing */
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

static void vif_test_buf_free(VifTestBuf *vb) {
    if (vb->alloc) {
        aligned_free(vb->alloc);
        vb->alloc = NULL;
    }
}

/* Fill 8-bit ref/dis data in VifBuffer */
static void vif_fill_8(VifBuffer *buf, unsigned w, unsigned h, InputCategory cat)
{
    uint8_t *ref = (uint8_t *)buf->ref;
    uint8_t *dis = (uint8_t *)buf->dis;
    generate_input_8(ref, w, h, buf->stride, cat);
    /* Use a different seed offset for dis to make ref != dis */
    if (cat == INPUT_RANDOM_42) {
        uint32_t state = 4242;
        for (unsigned i = 0; i < h; i++)
            for (unsigned j = 0; j < w; j++)
                dis[i * buf->stride + j] = (uint8_t)(prng_next(&state) & 0xFF);
    } else if (cat == INPUT_RANDOM_123) {
        uint32_t state = 12312;
        for (unsigned i = 0; i < h; i++)
            for (unsigned j = 0; j < w; j++)
                dis[i * buf->stride + j] = (uint8_t)(prng_next(&state) & 0xFF);
    } else {
        /* For non-random patterns, make dis = ref with slight variation */
        generate_input_8(dis, w, h, buf->stride, cat);
    }
}

/* Fill 16-bit ref/dis data in VifBuffer */
static void vif_fill_16(VifBuffer *buf, unsigned w, unsigned h,
    InputCategory cat, uint16_t max_val)
{
    uint16_t *ref = (uint16_t *)buf->ref;
    uint16_t *dis = (uint16_t *)buf->dis;
    ptrdiff_t stride = buf->stride / sizeof(uint16_t);
    generate_input_16(ref, w, h, stride, cat, max_val);
    if (cat == INPUT_RANDOM_42) {
        uint32_t state = 4242;
        for (unsigned i = 0; i < h; i++)
            for (unsigned j = 0; j < w; j++)
                dis[i * stride + j] = (uint16_t)(prng_next(&state) % (max_val + 1));
    } else if (cat == INPUT_RANDOM_123) {
        uint32_t state = 12312;
        for (unsigned i = 0; i < h; i++)
            for (unsigned j = 0; j < w; j++)
                dis[i * stride + j] = (uint16_t)(prng_next(&state) % (max_val + 1));
    } else {
        generate_input_16(dis, w, h, stride, cat, max_val);
    }
}

/* ========== subsample_rd_8 Tests ========== */

#if ARCH_X86
static char *test_subsample_rd_8_avx2(void) {
    for (unsigned d = 0; d < NUM_STANDARD_DIMS; d++) {
        unsigned w = standard_dims[d].w;
        unsigned h = standard_dims[d].h;

        /* Need minimum height for filter (fwidth=9, half=4) plus padding */
        if (h < 9) continue;

        for (int cat = 0; cat < INPUT_CATEGORY_COUNT; cat++) {
            VifTestBuf tb_ref, tb_simd;
            mu_assert("vif buf ref init failed",
                vif_test_buf_init(&tb_ref, w, h, 0) == 0);
            mu_assert("vif buf simd init failed",
                vif_test_buf_init(&tb_simd, w, h, 0) == 0);

            /* Fill identical data in both */
            vif_fill_8(&tb_ref.buf, w, h, (InputCategory)cat);
            memcpy(tb_simd.alloc, tb_ref.alloc, tb_ref.alloc_sz);

            /* Pad boundaries */
            ref_pad_top_and_bottom(tb_ref.buf, h, vif_filter1d_width[0]);
            ref_pad_top_and_bottom(tb_simd.buf, h, vif_filter1d_width[0]);

            /* Run both */
            ref_subsample_rd_8(tb_ref.buf, w, h);
            vif_subsample_rd_8_avx2(tb_simd.buf, w, h);

            /* Compare mu1 and mu2.
             * AVX2 stores subsampled output at compact positions [h/2 x w/2],
             * while the C reference stores full resolution [h x w].
             * Map: SIMD(i, j) == Ref(i*2, j*2). */
            ptrdiff_t mu_stride = tb_ref.buf.stride_16 / sizeof(uint16_t);
            for (unsigned i = 0; i < h / 2; i++) {
                for (unsigned j = 0; j < w / 2; j++) {
                    uint16_t rv = ((uint16_t *)tb_ref.buf.mu1)[(i * 2) * mu_stride + (j * 2)];
                    uint16_t sv = ((uint16_t *)tb_simd.buf.mu1)[i * mu_stride + j];
                    if (rv != sv) {
                        snprintf(diag_msg, sizeof(diag_msg),
                            "subsample_rd_8_avx2 mu1 MISMATCH [%s %ux%u]: "
                            "pos (%u,%u), ref=%u, simd=%u",
                            input_category_names[cat], w, h, i, j, rv, sv);
                        mu_assert(diag_msg, 0);
                    }
                }
            }
            for (unsigned i = 0; i < h / 2; i++) {
                for (unsigned j = 0; j < w / 2; j++) {
                    uint16_t rv = ((uint16_t *)tb_ref.buf.mu2)[(i * 2) * mu_stride + (j * 2)];
                    uint16_t sv = ((uint16_t *)tb_simd.buf.mu2)[i * mu_stride + j];
                    if (rv != sv) {
                        snprintf(diag_msg, sizeof(diag_msg),
                            "subsample_rd_8_avx2 mu2 MISMATCH [%s %ux%u]: "
                            "pos (%u,%u), ref=%u, simd=%u",
                            input_category_names[cat], w, h, i, j, rv, sv);
                        mu_assert(diag_msg, 0);
                    }
                }
            }

            vif_test_buf_free(&tb_ref);
            vif_test_buf_free(&tb_simd);
        }
    }
    return NULL;
}
#endif

#if ARCH_X86 && HAVE_AVX512
static char *test_subsample_rd_8_avx512(void) {
    for (unsigned d = 0; d < NUM_STANDARD_DIMS; d++) {
        unsigned w = standard_dims[d].w;
        unsigned h = standard_dims[d].h;
        if (h < 9) continue;

        for (int cat = 0; cat < INPUT_CATEGORY_COUNT; cat++) {
            VifTestBuf tb_ref, tb_simd;
            mu_assert("vif buf ref init failed",
                vif_test_buf_init(&tb_ref, w, h, 0) == 0);
            mu_assert("vif buf simd init failed",
                vif_test_buf_init(&tb_simd, w, h, 0) == 0);

            vif_fill_8(&tb_ref.buf, w, h, (InputCategory)cat);
            memcpy(tb_simd.alloc, tb_ref.alloc, tb_ref.alloc_sz);

            ref_pad_top_and_bottom(tb_ref.buf, h, vif_filter1d_width[0]);
            ref_pad_top_and_bottom(tb_simd.buf, h, vif_filter1d_width[0]);

            ref_subsample_rd_8(tb_ref.buf, w, h);
            vif_subsample_rd_8_avx512(tb_simd.buf, w, h);

            ptrdiff_t mu_stride = tb_ref.buf.stride_16 / sizeof(uint16_t);
            for (unsigned i = 0; i < h; i++) {
                for (unsigned j = 0; j < w; j++) {
                    uint16_t rv = ((uint16_t *)tb_ref.buf.mu1)[i * mu_stride + j];
                    uint16_t sv = ((uint16_t *)tb_simd.buf.mu1)[i * mu_stride + j];
                    if (rv != sv) {
                        snprintf(diag_msg, sizeof(diag_msg),
                            "subsample_rd_8_avx512 mu1 MISMATCH [%s %ux%u]: "
                            "pos (%u,%u), ref=%u, simd=%u",
                            input_category_names[cat], w, h, i, j, rv, sv);
                        mu_assert(diag_msg, 0);
                    }
                }
            }

            vif_test_buf_free(&tb_ref);
            vif_test_buf_free(&tb_simd);
        }
    }
    return NULL;
}
#endif

#if ARCH_AARCH64
static char *test_subsample_rd_8_neon(void) {
    for (unsigned d = 0; d < NUM_STANDARD_DIMS; d++) {
        unsigned w = standard_dims[d].w;
        unsigned h = standard_dims[d].h;
        if (h < 9) continue;

        for (int cat = 0; cat < INPUT_CATEGORY_COUNT; cat++) {
            VifTestBuf tb_ref, tb_simd;
            mu_assert("vif buf ref init failed",
                vif_test_buf_init(&tb_ref, w, h, 0) == 0);
            mu_assert("vif buf simd init failed",
                vif_test_buf_init(&tb_simd, w, h, 0) == 0);

            vif_fill_8(&tb_ref.buf, w, h, (InputCategory)cat);
            memcpy(tb_simd.alloc, tb_ref.alloc, tb_ref.alloc_sz);

            ref_pad_top_and_bottom(tb_ref.buf, h, vif_filter1d_width[0]);
            ref_pad_top_and_bottom(tb_simd.buf, h, vif_filter1d_width[0]);

            ref_subsample_rd_8(tb_ref.buf, w, h);
            vif_subsample_rd_8_neon(tb_simd.buf, w, h);

            ptrdiff_t mu_stride = tb_ref.buf.stride_16 / sizeof(uint16_t);
            for (unsigned i = 0; i < h; i++) {
                for (unsigned j = 0; j < w; j++) {
                    uint16_t rv = ((uint16_t *)tb_ref.buf.mu1)[i * mu_stride + j];
                    uint16_t sv = ((uint16_t *)tb_simd.buf.mu1)[i * mu_stride + j];
                    if (rv != sv) {
                        snprintf(diag_msg, sizeof(diag_msg),
                            "subsample_rd_8_neon mu1 MISMATCH [%s %ux%u]: "
                            "pos (%u,%u), ref=%u, simd=%u",
                            input_category_names[cat], w, h, i, j, rv, sv);
                        mu_assert(diag_msg, 0);
                    }
                }
            }

            vif_test_buf_free(&tb_ref);
            vif_test_buf_free(&tb_simd);
        }
    }
    return NULL;
}
#endif

/* ========== subsample_rd_16 Tests ========== */

static const int test_bpc_values[] = {10, 12, 16};
#define NUM_BPC_VALUES 3
static const int test_scale_values[] = {0, 1, 2, 3};
#define NUM_SCALE_VALUES 4

#if ARCH_X86
static char *test_subsample_rd_16_avx2(void) {
    for (unsigned d = 0; d < NUM_STANDARD_DIMS; d++) {
        unsigned w = standard_dims[d].w;
        unsigned h = standard_dims[d].h;
        if (h < 17) continue; /* Need enough height for largest filter */

        for (int bpc_idx = 0; bpc_idx < NUM_BPC_VALUES; bpc_idx++) {
            int bpc = test_bpc_values[bpc_idx];
            uint16_t max_val = (1 << bpc) - 1;

            for (int scale_idx = 0; scale_idx < NUM_SCALE_VALUES; scale_idx++) {
                int scale = test_scale_values[scale_idx];
                if (scale + 1 >= 4) continue; /* filter table only has 4 entries */

                for (int cat = 0; cat < INPUT_CATEGORY_COUNT; cat++) {
                    VifTestBuf tb_ref, tb_simd;
                    mu_assert("vif buf ref init failed",
                        vif_test_buf_init(&tb_ref, w, h, 1) == 0);
                    mu_assert("vif buf simd init failed",
                        vif_test_buf_init(&tb_simd, w, h, 1) == 0);

                    vif_fill_16(&tb_ref.buf, w, h, (InputCategory)cat, max_val);
                    /* Copy the full data region */
                    memcpy(tb_simd.alloc, tb_ref.alloc, tb_ref.alloc_sz);

                    int fwidth = vif_filter1d_width[scale + 1];
                    ref_pad_top_and_bottom(tb_ref.buf, h, fwidth);
                    ref_pad_top_and_bottom(tb_simd.buf, h, fwidth);

                    ref_subsample_rd_16(tb_ref.buf, w, h, scale, bpc);
                    vif_subsample_rd_16_avx2(tb_simd.buf, w, h, scale, bpc);

                    /* AVX2 stores subsampled rows: SIMD row i == Ref row i*2 */
                    ptrdiff_t mu_stride = tb_ref.buf.stride_16 / sizeof(uint16_t);
                    for (unsigned i = 0; i < h / 2; i++) {
                        for (unsigned j = 0; j < w; j++) {
                            uint16_t rv = ((uint16_t *)tb_ref.buf.mu1)[(i * 2) * mu_stride + j];
                            uint16_t sv = ((uint16_t *)tb_simd.buf.mu1)[i * mu_stride + j];
                            if (rv != sv) {
                                snprintf(diag_msg, sizeof(diag_msg),
                                    "subsample_rd_16_avx2 mu1 MISMATCH [%s %ux%u bpc=%d scale=%d]: "
                                    "pos (%u,%u), ref=%u, simd=%u",
                                    input_category_names[cat], w, h, bpc, scale, i, j, rv, sv);
                                mu_assert(diag_msg, 0);
                            }
                        }
                    }

                    vif_test_buf_free(&tb_ref);
                    vif_test_buf_free(&tb_simd);
                }
            }
        }
    }
    return NULL;
}
#endif

#if ARCH_X86 && HAVE_AVX512
static char *test_subsample_rd_16_avx512(void) {
    for (unsigned d = 0; d < NUM_STANDARD_DIMS; d++) {
        unsigned w = standard_dims[d].w;
        unsigned h = standard_dims[d].h;
        if (h < 17) continue;

        for (int bpc_idx = 0; bpc_idx < NUM_BPC_VALUES; bpc_idx++) {
            int bpc = test_bpc_values[bpc_idx];
            uint16_t max_val = (1 << bpc) - 1;

            for (int scale_idx = 0; scale_idx < NUM_SCALE_VALUES; scale_idx++) {
                int scale = test_scale_values[scale_idx];
                if (scale + 1 >= 4) continue;

                for (int cat = 0; cat < INPUT_CATEGORY_COUNT; cat++) {
                    VifTestBuf tb_ref, tb_simd;
                    mu_assert("vif buf ref init",
                        vif_test_buf_init(&tb_ref, w, h, 1) == 0);
                    mu_assert("vif buf simd init",
                        vif_test_buf_init(&tb_simd, w, h, 1) == 0);

                    vif_fill_16(&tb_ref.buf, w, h, (InputCategory)cat, max_val);
                    memcpy(tb_simd.alloc, tb_ref.alloc, tb_ref.alloc_sz);

                    int fwidth = vif_filter1d_width[scale + 1];
                    ref_pad_top_and_bottom(tb_ref.buf, h, fwidth);
                    ref_pad_top_and_bottom(tb_simd.buf, h, fwidth);

                    ref_subsample_rd_16(tb_ref.buf, w, h, scale, bpc);
                    vif_subsample_rd_16_avx512(tb_simd.buf, w, h, scale, bpc);

                    ptrdiff_t mu_stride = tb_ref.buf.stride_16 / sizeof(uint16_t);
                    for (unsigned i = 0; i < h; i++) {
                        for (unsigned j = 0; j < w; j++) {
                            uint16_t rv = ((uint16_t *)tb_ref.buf.mu1)[i * mu_stride + j];
                            uint16_t sv = ((uint16_t *)tb_simd.buf.mu1)[i * mu_stride + j];
                            if (rv != sv) {
                                snprintf(diag_msg, sizeof(diag_msg),
                                    "subsample_rd_16_avx512 mu1 MISMATCH [%s %ux%u bpc=%d scale=%d]: "
                                    "pos (%u,%u), ref=%u, simd=%u",
                                    input_category_names[cat], w, h, bpc, scale, i, j, rv, sv);
                                mu_assert(diag_msg, 0);
                            }
                        }
                    }

                    vif_test_buf_free(&tb_ref);
                    vif_test_buf_free(&tb_simd);
                }
            }
        }
    }
    return NULL;
}
#endif

#if ARCH_AARCH64
static char *test_subsample_rd_16_neon(void) {
    for (unsigned d = 0; d < NUM_STANDARD_DIMS; d++) {
        unsigned w = standard_dims[d].w;
        unsigned h = standard_dims[d].h;
        if (h < 17) continue;

        for (int bpc_idx = 0; bpc_idx < NUM_BPC_VALUES; bpc_idx++) {
            int bpc = test_bpc_values[bpc_idx];
            uint16_t max_val = (1 << bpc) - 1;

            for (int scale_idx = 0; scale_idx < NUM_SCALE_VALUES; scale_idx++) {
                int scale = test_scale_values[scale_idx];
                if (scale + 1 >= 4) continue;

                for (int cat = 0; cat < INPUT_CATEGORY_COUNT; cat++) {
                    VifTestBuf tb_ref, tb_simd;
                    mu_assert("vif buf ref init",
                        vif_test_buf_init(&tb_ref, w, h, 1) == 0);
                    mu_assert("vif buf simd init",
                        vif_test_buf_init(&tb_simd, w, h, 1) == 0);

                    vif_fill_16(&tb_ref.buf, w, h, (InputCategory)cat, max_val);
                    memcpy(tb_simd.alloc, tb_ref.alloc, tb_ref.alloc_sz);

                    int fwidth = vif_filter1d_width[scale + 1];
                    ref_pad_top_and_bottom(tb_ref.buf, h, fwidth);
                    ref_pad_top_and_bottom(tb_simd.buf, h, fwidth);

                    ref_subsample_rd_16(tb_ref.buf, w, h, scale, bpc);
                    vif_subsample_rd_16_neon(tb_simd.buf, w, h, scale, bpc);

                    ptrdiff_t mu_stride = tb_ref.buf.stride_16 / sizeof(uint16_t);
                    for (unsigned i = 0; i < h; i++) {
                        for (unsigned j = 0; j < w; j++) {
                            uint16_t rv = ((uint16_t *)tb_ref.buf.mu1)[i * mu_stride + j];
                            uint16_t sv = ((uint16_t *)tb_simd.buf.mu1)[i * mu_stride + j];
                            if (rv != sv) {
                                snprintf(diag_msg, sizeof(diag_msg),
                                    "subsample_rd_16_neon mu1 MISMATCH [%s %ux%u bpc=%d scale=%d]: "
                                    "pos (%u,%u), ref=%u, simd=%u",
                                    input_category_names[cat], w, h, bpc, scale, i, j, rv, sv);
                                mu_assert(diag_msg, 0);
                            }
                        }
                    }

                    vif_test_buf_free(&tb_ref);
                    vif_test_buf_free(&tb_simd);
                }
            }
        }
    }
    return NULL;
}
#endif

/* ========== vif_statistic_8 Tests ========== */

/* Log table generator (from integer_vif.c) */
static void test_log_generate(uint16_t *log2_table)
{
    for (unsigned i = 32767; i < 65536; ++i) {
        log2_table[i] = (uint16_t)round(log2f((float)i) * 2048);
    }
}

#if ARCH_X86
static char *test_vif_statistic_8_avx2(void) {
    /* Use a subset of dimensions for the expensive statistic tests */
    static const TestDim stat_dims[] = {
        {8, 8}, {16, 16}, {32, 32}, {64, 64}, {120, 68}, {576, 324}
    };
    const unsigned num_stat_dims = sizeof(stat_dims) / sizeof(stat_dims[0]);

    for (unsigned d = 0; d < num_stat_dims; d++) {
        unsigned w = stat_dims[d].w;
        unsigned h = stat_dims[d].h;
        if (h < 17) continue;

        for (int cat = 0; cat < INPUT_CATEGORY_COUNT; cat++) {
            /* Set up two complete VifPublicState structs */
            VifPublicState *state_ref = aligned_malloc(sizeof(VifPublicState), MAX_ALIGN);
            VifPublicState *state_simd = aligned_malloc(sizeof(VifPublicState), MAX_ALIGN);
            mu_assert("state alloc failed", state_ref && state_simd);
            memset(state_ref, 0, sizeof(*state_ref));
            memset(state_simd, 0, sizeof(*state_simd));

            state_ref->vif_enhn_gain_limit = 100.0;
            state_simd->vif_enhn_gain_limit = 100.0;
            test_log_generate(state_ref->log2_table);
            memcpy(state_simd->log2_table, state_ref->log2_table, sizeof(state_ref->log2_table));

            /* Set up buffers */
            VifTestBuf tb_ref, tb_simd;
            mu_assert("vif stat buf ref init",
                vif_test_buf_init(&tb_ref, w, h, 0) == 0);
            mu_assert("vif stat buf simd init",
                vif_test_buf_init(&tb_simd, w, h, 0) == 0);

            vif_fill_8(&tb_ref.buf, w, h, (InputCategory)cat);
            memcpy(tb_simd.alloc, tb_ref.alloc, tb_ref.alloc_sz);

            ref_pad_top_and_bottom(tb_ref.buf, h, vif_filter1d_width[0]);
            ref_pad_top_and_bottom(tb_simd.buf, h, vif_filter1d_width[0]);

            state_ref->buf = tb_ref.buf;
            state_simd->buf = tb_simd.buf;

            float num_ref = 0, den_ref = 0;
            float num_simd = 0, den_simd = 0;

            vif_statistic_8(state_ref, &num_ref, &den_ref, w, h);
            vif_statistic_8_avx2(state_simd, &num_simd, &den_simd, w, h);

            mu_assert(diag_msg,
                floats_match(num_ref, den_ref, num_simd, den_simd,
                    "vif_statistic_8_avx2", input_category_names[cat], w, h));

            vif_test_buf_free(&tb_ref);
            vif_test_buf_free(&tb_simd);
            aligned_free(state_ref);
            aligned_free(state_simd);
        }
    }
    return NULL;
}
#endif

#if ARCH_X86 && HAVE_AVX512
static char *test_vif_statistic_8_avx512(void) {
    static const TestDim stat_dims[] = {
        {8, 8}, {16, 16}, {32, 32}, {64, 64}, {120, 68}, {576, 324}
    };
    const unsigned num_stat_dims = sizeof(stat_dims) / sizeof(stat_dims[0]);

    for (unsigned d = 0; d < num_stat_dims; d++) {
        unsigned w = stat_dims[d].w;
        unsigned h = stat_dims[d].h;
        if (h < 17) continue;

        for (int cat = 0; cat < INPUT_CATEGORY_COUNT; cat++) {
            VifPublicState *state_ref = aligned_malloc(sizeof(VifPublicState), MAX_ALIGN);
            VifPublicState *state_simd = aligned_malloc(sizeof(VifPublicState), MAX_ALIGN);
            mu_assert("state alloc failed", state_ref && state_simd);
            memset(state_ref, 0, sizeof(*state_ref));
            memset(state_simd, 0, sizeof(*state_simd));

            state_ref->vif_enhn_gain_limit = 100.0;
            state_simd->vif_enhn_gain_limit = 100.0;
            test_log_generate(state_ref->log2_table);
            memcpy(state_simd->log2_table, state_ref->log2_table, sizeof(state_ref->log2_table));

            VifTestBuf tb_ref, tb_simd;
            mu_assert("vif stat buf ref init",
                vif_test_buf_init(&tb_ref, w, h, 0) == 0);
            mu_assert("vif stat buf simd init",
                vif_test_buf_init(&tb_simd, w, h, 0) == 0);

            vif_fill_8(&tb_ref.buf, w, h, (InputCategory)cat);
            memcpy(tb_simd.alloc, tb_ref.alloc, tb_ref.alloc_sz);

            ref_pad_top_and_bottom(tb_ref.buf, h, vif_filter1d_width[0]);
            ref_pad_top_and_bottom(tb_simd.buf, h, vif_filter1d_width[0]);

            state_ref->buf = tb_ref.buf;
            state_simd->buf = tb_simd.buf;

            float num_ref = 0, den_ref = 0;
            float num_simd = 0, den_simd = 0;

            vif_statistic_8(state_ref, &num_ref, &den_ref, w, h);
            vif_statistic_8_avx512(state_simd, &num_simd, &den_simd, w, h);

            mu_assert(diag_msg,
                floats_match(num_ref, den_ref, num_simd, den_simd,
                    "vif_statistic_8_avx512", input_category_names[cat], w, h));

            vif_test_buf_free(&tb_ref);
            vif_test_buf_free(&tb_simd);
            aligned_free(state_ref);
            aligned_free(state_simd);
        }
    }
    return NULL;
}
#endif

#if ARCH_AARCH64
static char *test_vif_statistic_8_neon(void) {
    static const TestDim stat_dims[] = {
        {8, 8}, {16, 16}, {32, 32}, {64, 64}, {120, 68}, {576, 324}
    };
    const unsigned num_stat_dims = sizeof(stat_dims) / sizeof(stat_dims[0]);

    for (unsigned d = 0; d < num_stat_dims; d++) {
        unsigned w = stat_dims[d].w;
        unsigned h = stat_dims[d].h;
        if (h < 17) continue;

        for (int cat = 0; cat < INPUT_CATEGORY_COUNT; cat++) {
            VifPublicState *state_ref = aligned_malloc(sizeof(VifPublicState), MAX_ALIGN);
            VifPublicState *state_simd = aligned_malloc(sizeof(VifPublicState), MAX_ALIGN);
            mu_assert("state alloc failed", state_ref && state_simd);
            memset(state_ref, 0, sizeof(*state_ref));
            memset(state_simd, 0, sizeof(*state_simd));

            state_ref->vif_enhn_gain_limit = 100.0;
            state_simd->vif_enhn_gain_limit = 100.0;
            test_log_generate(state_ref->log2_table);
            memcpy(state_simd->log2_table, state_ref->log2_table, sizeof(state_ref->log2_table));

            VifTestBuf tb_ref, tb_simd;
            mu_assert("vif stat buf ref init",
                vif_test_buf_init(&tb_ref, w, h, 0) == 0);
            mu_assert("vif stat buf simd init",
                vif_test_buf_init(&tb_simd, w, h, 0) == 0);

            vif_fill_8(&tb_ref.buf, w, h, (InputCategory)cat);
            memcpy(tb_simd.alloc, tb_ref.alloc, tb_ref.alloc_sz);

            ref_pad_top_and_bottom(tb_ref.buf, h, vif_filter1d_width[0]);
            ref_pad_top_and_bottom(tb_simd.buf, h, vif_filter1d_width[0]);

            state_ref->buf = tb_ref.buf;
            state_simd->buf = tb_simd.buf;

            float num_ref = 0, den_ref = 0;
            float num_simd = 0, den_simd = 0;

            vif_statistic_8(state_ref, &num_ref, &den_ref, w, h);
            vif_statistic_8_neon(state_simd, &num_simd, &den_simd, w, h);

            mu_assert(diag_msg,
                floats_match(num_ref, den_ref, num_simd, den_simd,
                    "vif_statistic_8_neon", input_category_names[cat], w, h));

            vif_test_buf_free(&tb_ref);
            vif_test_buf_free(&tb_simd);
            aligned_free(state_ref);
            aligned_free(state_simd);
        }
    }
    return NULL;
}
#endif

/* ========== vif_statistic_16 Tests ========== */

#if ARCH_X86
static char *test_vif_statistic_16_avx2(void) {
    static const TestDim stat_dims[] = {
        {16, 16}, {32, 32}, {64, 64}, {120, 68}, {576, 324}
    };
    const unsigned num_stat_dims = sizeof(stat_dims) / sizeof(stat_dims[0]);

    for (unsigned d = 0; d < num_stat_dims; d++) {
        unsigned w = stat_dims[d].w;
        unsigned h = stat_dims[d].h;
        if (h < 17) continue;

        for (int bpc_idx = 0; bpc_idx < NUM_BPC_VALUES; bpc_idx++) {
            int bpc = test_bpc_values[bpc_idx];
            uint16_t max_val = (uint16_t)((1 << bpc) - 1);

            for (int scale = 0; scale < 4; scale++) {
                for (int cat = 0; cat < INPUT_CATEGORY_COUNT; cat++) {
                    VifPublicState *state_ref = aligned_malloc(sizeof(VifPublicState), MAX_ALIGN);
                    VifPublicState *state_simd = aligned_malloc(sizeof(VifPublicState), MAX_ALIGN);
                    mu_assert("state alloc failed", state_ref && state_simd);
                    memset(state_ref, 0, sizeof(*state_ref));
                    memset(state_simd, 0, sizeof(*state_simd));

                    state_ref->vif_enhn_gain_limit = 100.0;
                    state_simd->vif_enhn_gain_limit = 100.0;
                    test_log_generate(state_ref->log2_table);
                    memcpy(state_simd->log2_table, state_ref->log2_table,
                        sizeof(state_ref->log2_table));

                    VifTestBuf tb_ref, tb_simd;
                    mu_assert("vif stat16 buf ref init",
                        vif_test_buf_init(&tb_ref, w, h, 1) == 0);
                    mu_assert("vif stat16 buf simd init",
                        vif_test_buf_init(&tb_simd, w, h, 1) == 0);

                    vif_fill_16(&tb_ref.buf, w, h, (InputCategory)cat, max_val);
                    memcpy(tb_simd.alloc, tb_ref.alloc, tb_ref.alloc_sz);

                    ref_pad_top_and_bottom(tb_ref.buf, h, vif_filter1d_width[0]);
                    ref_pad_top_and_bottom(tb_simd.buf, h, vif_filter1d_width[0]);

                    state_ref->buf = tb_ref.buf;
                    state_simd->buf = tb_simd.buf;

                    float num_ref = 0, den_ref = 0;
                    float num_simd = 0, den_simd = 0;

                    vif_statistic_16(state_ref, &num_ref, &den_ref, w, h, bpc, scale);
                    vif_statistic_16_avx2(state_simd, &num_simd, &den_simd, w, h, bpc, scale);

                    mu_assert(diag_msg,
                        floats_match(num_ref, den_ref, num_simd, den_simd,
                            "vif_statistic_16_avx2", input_category_names[cat], w, h));

                    vif_test_buf_free(&tb_ref);
                    vif_test_buf_free(&tb_simd);
                    aligned_free(state_ref);
                    aligned_free(state_simd);
                }
            }
        }
    }
    return NULL;
}
#endif

#if ARCH_X86 && HAVE_AVX512
static char *test_vif_statistic_16_avx512(void) {
    static const TestDim stat_dims[] = {
        {16, 16}, {32, 32}, {64, 64}, {120, 68}, {576, 324}
    };
    const unsigned num_stat_dims = sizeof(stat_dims) / sizeof(stat_dims[0]);

    for (unsigned d = 0; d < num_stat_dims; d++) {
        unsigned w = stat_dims[d].w;
        unsigned h = stat_dims[d].h;
        if (h < 17) continue;

        for (int bpc_idx = 0; bpc_idx < NUM_BPC_VALUES; bpc_idx++) {
            int bpc = test_bpc_values[bpc_idx];
            uint16_t max_val = (uint16_t)((1 << bpc) - 1);

            for (int scale = 0; scale < 4; scale++) {
                for (int cat = 0; cat < INPUT_CATEGORY_COUNT; cat++) {
                    VifPublicState *state_ref = aligned_malloc(sizeof(VifPublicState), MAX_ALIGN);
                    VifPublicState *state_simd = aligned_malloc(sizeof(VifPublicState), MAX_ALIGN);
                    mu_assert("state alloc failed", state_ref && state_simd);
                    memset(state_ref, 0, sizeof(*state_ref));
                    memset(state_simd, 0, sizeof(*state_simd));

                    state_ref->vif_enhn_gain_limit = 100.0;
                    state_simd->vif_enhn_gain_limit = 100.0;
                    test_log_generate(state_ref->log2_table);
                    memcpy(state_simd->log2_table, state_ref->log2_table,
                        sizeof(state_ref->log2_table));

                    VifTestBuf tb_ref, tb_simd;
                    mu_assert("vif stat16 buf ref init",
                        vif_test_buf_init(&tb_ref, w, h, 1) == 0);
                    mu_assert("vif stat16 buf simd init",
                        vif_test_buf_init(&tb_simd, w, h, 1) == 0);

                    vif_fill_16(&tb_ref.buf, w, h, (InputCategory)cat, max_val);
                    memcpy(tb_simd.alloc, tb_ref.alloc, tb_ref.alloc_sz);

                    ref_pad_top_and_bottom(tb_ref.buf, h, vif_filter1d_width[0]);
                    ref_pad_top_and_bottom(tb_simd.buf, h, vif_filter1d_width[0]);

                    state_ref->buf = tb_ref.buf;
                    state_simd->buf = tb_simd.buf;

                    float num_ref = 0, den_ref = 0;
                    float num_simd = 0, den_simd = 0;

                    vif_statistic_16(state_ref, &num_ref, &den_ref, w, h, bpc, scale);
                    vif_statistic_16_avx512(state_simd, &num_simd, &den_simd, w, h, bpc, scale);

                    mu_assert(diag_msg,
                        floats_match(num_ref, den_ref, num_simd, den_simd,
                            "vif_statistic_16_avx512", input_category_names[cat], w, h));

                    vif_test_buf_free(&tb_ref);
                    vif_test_buf_free(&tb_simd);
                    aligned_free(state_ref);
                    aligned_free(state_simd);
                }
            }
        }
    }
    return NULL;
}
#endif

#if ARCH_AARCH64
static char *test_vif_statistic_16_neon(void) {
    static const TestDim stat_dims[] = {
        {16, 16}, {32, 32}, {64, 64}, {120, 68}, {576, 324}
    };
    const unsigned num_stat_dims = sizeof(stat_dims) / sizeof(stat_dims[0]);

    for (unsigned d = 0; d < num_stat_dims; d++) {
        unsigned w = stat_dims[d].w;
        unsigned h = stat_dims[d].h;
        if (h < 17) continue;

        for (int bpc_idx = 0; bpc_idx < NUM_BPC_VALUES; bpc_idx++) {
            int bpc = test_bpc_values[bpc_idx];
            uint16_t max_val = (uint16_t)((1 << bpc) - 1);

            for (int scale = 0; scale < 4; scale++) {
                for (int cat = 0; cat < INPUT_CATEGORY_COUNT; cat++) {
                    VifPublicState *state_ref = aligned_malloc(sizeof(VifPublicState), MAX_ALIGN);
                    VifPublicState *state_simd = aligned_malloc(sizeof(VifPublicState), MAX_ALIGN);
                    mu_assert("state alloc failed", state_ref && state_simd);
                    memset(state_ref, 0, sizeof(*state_ref));
                    memset(state_simd, 0, sizeof(*state_simd));

                    state_ref->vif_enhn_gain_limit = 100.0;
                    state_simd->vif_enhn_gain_limit = 100.0;
                    test_log_generate(state_ref->log2_table);
                    memcpy(state_simd->log2_table, state_ref->log2_table,
                        sizeof(state_ref->log2_table));

                    VifTestBuf tb_ref, tb_simd;
                    mu_assert("vif stat16 buf ref init",
                        vif_test_buf_init(&tb_ref, w, h, 1) == 0);
                    mu_assert("vif stat16 buf simd init",
                        vif_test_buf_init(&tb_simd, w, h, 1) == 0);

                    vif_fill_16(&tb_ref.buf, w, h, (InputCategory)cat, max_val);
                    memcpy(tb_simd.alloc, tb_ref.alloc, tb_ref.alloc_sz);

                    ref_pad_top_and_bottom(tb_ref.buf, h, vif_filter1d_width[0]);
                    ref_pad_top_and_bottom(tb_simd.buf, h, vif_filter1d_width[0]);

                    state_ref->buf = tb_ref.buf;
                    state_simd->buf = tb_simd.buf;

                    float num_ref = 0, den_ref = 0;
                    float num_simd = 0, den_simd = 0;

                    vif_statistic_16(state_ref, &num_ref, &den_ref, w, h, bpc, scale);
                    vif_statistic_16_neon(state_simd, &num_simd, &den_simd, w, h, bpc, scale);

                    mu_assert(diag_msg,
                        floats_match(num_ref, den_ref, num_simd, den_simd,
                            "vif_statistic_16_neon", input_category_names[cat], w, h));

                    vif_test_buf_free(&tb_ref);
                    vif_test_buf_free(&tb_simd);
                    aligned_free(state_ref);
                    aligned_free(state_simd);
                }
            }
        }
    }
    return NULL;
}
#endif

/* ========== Test Runner ========== */

char *run_tests(void) {
    vmaf_init_cpu();
#if ARCH_X86
    mu_run_test(test_subsample_rd_8_avx2);
    mu_run_test(test_subsample_rd_16_avx2);
    mu_run_test(test_vif_statistic_8_avx2);
    mu_run_test(test_vif_statistic_16_avx2);
#if HAVE_AVX512
    if (vmaf_get_cpu_flags() & VMAF_X86_CPU_FLAG_AVX512) {
        mu_run_test(test_subsample_rd_8_avx512);
        mu_run_test(test_subsample_rd_16_avx512);
        mu_run_test(test_vif_statistic_8_avx512);
        mu_run_test(test_vif_statistic_16_avx512);
    } else {
        fprintf(stderr, "AVX-512 VIF tests: skipped (no AVX-512)\n");
    }
#endif
#endif
#if ARCH_AARCH64
    mu_run_test(test_subsample_rd_8_neon);
    mu_run_test(test_subsample_rd_16_neon);
    mu_run_test(test_vif_statistic_8_neon);
    mu_run_test(test_vif_statistic_16_neon);
#endif
    return NULL;
}
