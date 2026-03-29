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

#include "stdio.h"
#include <immintrin.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>
#include "feature/integer_vif.h"
#include "feature/common/macros.h"

#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#define MAX(x, y) (((x) > (y)) ? (x) : (y))

/* Maximum filter width across all scales (scale 0 = 17 taps) */
#define VIF_FILT_MAX 17


#if defined __GNUC__
#define ALIGNED(x) __attribute__ ((aligned (x)))
#elif defined (_MSC_VER)  && (!defined UNDER_CE)
#define ALIGNED(x) __declspec (align(x))
#else
#define ALIGNED(x)
#endif


static FORCE_INLINE void
pad_top_and_bottom(VifBuffer buf, unsigned h, int fwidth)
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

static FORCE_INLINE void
copy_and_pad(VifBuffer buf, unsigned w, unsigned h, int scale)
{
    uint16_t *ref = buf.ref;
    uint16_t *dis = buf.dis;
    const ptrdiff_t stride = buf.stride / sizeof(uint16_t);
    const ptrdiff_t mu_stride = buf.stride_16 / sizeof(uint16_t);

    for (unsigned i = 0; i < h / 2; ++i) {
        for (unsigned j = 0; j < w / 2; ++j) {
            ref[i * stride + j] = buf.mu1[i * mu_stride + j];
            dis[i * stride + j] = buf.mu2[i * mu_stride + j];
        }
    }
    pad_top_and_bottom(buf, h / 2, vif_filter1d_width[scale]);
}

// multiply r0 * f and store in 32-bit accumulators (shuffled 0 1 2 3 8 9 10 11 / 4 5 6 7 12 13 14 15)
#define multiply2(acc_left, acc_right, r0, f) \
{ \
__m256i zero = _mm256_setzero_si256(); \
acc_left = _mm256_madd_epi16(_mm256_unpacklo_epi16(r0, zero), f); \
acc_right = _mm256_madd_epi16(_mm256_unpackhi_epi16(r0, zero), f); \
}

// multiply r0 * f and r1 * f and store in 32-bit accumulators (shuffled 0 1 2 3 8 9 10 11 / 4 5 6 7 12 13 14 15)
#define multiply2_and_accumulate(acc_left, acc_right, r0, r1, f) \
  acc_left = _mm256_add_epi32(acc_left, _mm256_madd_epi16(_mm256_unpacklo_epi16(r0, r1), f)); \
  acc_right = _mm256_add_epi32(acc_right, _mm256_madd_epi16(_mm256_unpackhi_epi16(r0, r1), f));


// compute r0 * r1 * f and set 32-bit accumulators (shuffled 0 1 2 3 8 9 10 11 / 4 5 6 7 12 13 14 15)
#define multiply3(accum_ref_left, accum_ref_right, r0, r1, f) \
{ \
    __m256i mul = _mm256_mullo_epi16(r0, r1); \
    __m256i lo = _mm256_mullo_epi16(mul, f); \
    __m256i hi = _mm256_mulhi_epu16(mul, f); \
    accum_ref_left = _mm256_unpacklo_epi16(lo, hi); \
    accum_ref_right = _mm256_unpackhi_epi16(lo, hi); \
}

// compute r0 * r1 * f and add to 32-bit accumulators (shuffled 0 1 2 3 8 9 10 11 / 4 5 6 7 12 13 14 15)
#define multiply3_and_accumulate(accum_ref_left, accum_ref_right, r0, r1, f) \
{ \
    __m256i mul = _mm256_mullo_epi16(r0, r1); \
    __m256i lo = _mm256_mullo_epi16(mul, f); \
    __m256i hi = _mm256_mulhi_epu16(mul, f); \
    __m256i left = _mm256_unpacklo_epi16(lo, hi); \
    __m256i right = _mm256_unpackhi_epi16(lo, hi); \
    accum_ref_left = _mm256_add_epi32(accum_ref_left, left); \
    accum_ref_right = _mm256_add_epi32(accum_ref_right, right); \
}


#define shuffle_and_save(addr, x, y) \
{ \
   __m256i left = _mm256_permute2x128_si256(x, y, 0x20); \
   __m256i right = _mm256_permute2x128_si256(x, y, 0x31); \
   _mm256_storeu_si256((__m256i*)(addr), left); \
   _mm256_storeu_si256(((__m256i*)(addr)) + 1, right); \
}

/* Horizontal sum of 8x int32 lanes -> single int32 */
static FORCE_INLINE int32_t hsum_epi32(__m256i v) {
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    lo = _mm_add_epi32(lo, hi);
    lo = _mm_add_epi32(lo, _mm_shuffle_epi32(lo, 0x4E)); /* 01 00 11 10 */
    lo = _mm_add_epi32(lo, _mm_shuffle_epi32(lo, 0xB1)); /* 10 11 00 01 */
    return _mm_cvtsi128_si32(lo);
}

/*
 * Vectorized log2_32 sum: compute sum of log2_32(vals[i]) for masked elements.
 * Uses float exponent extraction for CLZ equivalent + gather for table lookup.
 *
 * log2_table: uint16_t[65537] log2 lookup table
 * vals: 8 x int32 values (must be > 0 for all masked lanes)
 * mask_v: 8 x int32 mask (-1 for active, 0 for inactive)
 * Returns sum of log2_32(vals[i]) for active elements.
 */
static FORCE_INLINE int64_t vec_log2_32_masked_sum(
    const uint16_t *log2_table, __m256i vals, __m256i mask_v)
{
    /* floor(log2(x)) via float exponent: cvt to float, extract exponent bits.
     * For x in [2^n, 2^(n+1)), float exponent field = n + 127.
     * We need k = 16 - __builtin_clz(x) = floor(log2(x)) - 15.
     * So k = (exponent - 127) - 15 = exponent - 142. */
    __m256 f = _mm256_cvtepi32_ps(vals);
    __m256i fbits = _mm256_castps_si256(f);
    __m256i exp_biased = _mm256_srli_epi32(fbits, 23);
    __m256i k = _mm256_sub_epi32(exp_biased, _mm256_set1_epi32(142));

    /* table_idx = vals >> k, normalizes to [2^15, 2^16) range */
    __m256i table_idx = _mm256_srlv_epi32(vals, k);

    /* Gather from uint16_t table with scale=2 (byte offset = idx * 2).
     * Reads 32 bits starting at &log2_table[idx]; mask off upper 16 bits. */
    __m256i table_vals = _mm256_i32gather_epi32(
        (const int *)log2_table, table_idx, 2);
    table_vals = _mm256_and_si256(table_vals, _mm256_set1_epi32(0xFFFF));

    /* result[i] = table_vals[i] + 2048 * k[i], masked to active lanes */
    __m256i result = _mm256_add_epi32(table_vals, _mm256_slli_epi32(k, 11));
    result = _mm256_and_si256(result, mask_v);

    return (int64_t)hsum_epi32(result);
}

/*
 * Vectorized threshold comparison and non-log accumulation for 8 elements.
 * Processes one __m256i chunk of xx, yy, xy:
 *   - Non-log path (sigma1_sq < sigma_nsq): accumulates sigma2_sq and count
 *   - Log path (sigma1_sq >= sigma_nsq): batch FP division/multiplication
 *     via AVX2 double-precision, then scalar log table lookups
 *
 * xx_v, yy_v: 8x int32 packed sigma values
 * xx, yy, xy: stack arrays (already stored) for extraction
 * base: starting index into xx/yy/xy arrays (0 or 8)
 */
#define VIF_ACCUM_BLOCK(xx_v, yy_v, xx, yy, xy, base, \
                        sigma_nsq, log2_table, vif_enhn_gain_limit, \
                        accum_num_log, accum_den_log, \
                        accum_num_non_log, accum_den_non_log) \
do { \
    __m256i mask_ = _mm256_cmpgt_epi32(xx_v, _mm256_set1_epi32((sigma_nsq) - 1)); \
    __m256i non_log_yy_ = _mm256_andnot_si256(mask_, yy_v); \
    accum_num_non_log += hsum_epi32(non_log_yy_); \
    int log_mask_ps_ = _mm256_movemask_ps(_mm256_castsi256_ps(mask_)); \
    accum_den_non_log += 8 - __builtin_popcount(log_mask_ps_); \
    if (log_mask_ps_) { \
        /* Batch FP division and multiplication for all 8 elements. */ \
        /* Elements not on the log path compute harmlessly; results unused. */ \
        __m256d xx_d0_ = _mm256_cvtepi32_pd(_mm_loadu_si128((__m128i*)&(xx)[(base)])); \
        __m256d xx_d1_ = _mm256_cvtepi32_pd(_mm_loadu_si128((__m128i*)&(xx)[(base)+4])); \
        __m256d yy_d0_ = _mm256_cvtepi32_pd(_mm_loadu_si128((__m128i*)&(yy)[(base)])); \
        __m256d yy_d1_ = _mm256_cvtepi32_pd(_mm_loadu_si128((__m128i*)&(yy)[(base)+4])); \
        __m256d xy_d0_ = _mm256_cvtepi32_pd(_mm_loadu_si128((__m128i*)&(xy)[(base)])); \
        __m256d xy_d1_ = _mm256_cvtepi32_pd(_mm_loadu_si128((__m128i*)&(xy)[(base)+4])); \
        __m256d eps_v_ = _mm256_set1_pd(65536.0 * 1.0e-10); \
        __m256d limit_v_ = _mm256_set1_pd(vif_enhn_gain_limit); \
        /* g = sigma12 / (sigma1_sq + eps) — 2 vector divides for 8 elements */ \
        __m256d g0_ = _mm256_div_pd(xy_d0_, _mm256_add_pd(xx_d0_, eps_v_)); \
        __m256d g1_ = _mm256_div_pd(xy_d1_, _mm256_add_pd(xx_d1_, eps_v_)); \
        /* sv_sq = sigma2_sq - g * sigma12 (unclamped g), using FMA: fnmadd = -(a*b)+c */ \
        __m256d sv0_ = _mm256_fnmadd_pd(g0_, xy_d0_, yy_d0_); \
        __m256d sv1_ = _mm256_fnmadd_pd(g1_, xy_d1_, yy_d1_); \
        /* g = min(g, limit) — clamp after sv_sq computation */ \
        g0_ = _mm256_min_pd(g0_, limit_v_); \
        g1_ = _mm256_min_pd(g1_, limit_v_); \
        /* g*g*sigma1_sq (clamped g) */ \
        __m256d gg_xx0_ = _mm256_mul_pd(_mm256_mul_pd(g0_, g0_), xx_d0_); \
        __m256d gg_xx1_ = _mm256_mul_pd(_mm256_mul_pd(g1_, g1_), xx_d1_); \
        ALIGNED(32) double sv_arr_[8], gg_xx_arr_[8]; \
        _mm256_store_pd(&sv_arr_[0], sv0_); \
        _mm256_store_pd(&sv_arr_[4], sv1_); \
        _mm256_store_pd(&gg_xx_arr_[0], gg_xx0_); \
        _mm256_store_pd(&gg_xx_arr_[4], gg_xx1_); \
        /* Vectorized den_log: log2_32(sigma_nsq + sigma1_sq) for all masked lanes */ \
        { \
            __m256i den_vals_ = _mm256_add_epi32(xx_v, _mm256_set1_epi32(sigma_nsq)); \
            int64_t den_sum_ = vec_log2_32_masked_sum(log2_table, den_vals_, mask_); \
            int den_count_ = __builtin_popcount(log_mask_ps_); \
            accum_den_log += den_sum_ - (int64_t)den_count_ * 2048 * 17; \
        } \
        /* Scalar loop: num_log needs log2_64 + conditional (not vectorizable) */ \
        unsigned int log_bits_ = (unsigned int)log_mask_ps_; \
        while (log_bits_) { \
            int b_ = __builtin_ctz(log_bits_); \
            log_bits_ &= log_bits_ - 1; \
            int32_t sigma2_sq_ = (int32_t)(yy)[(base) + b_]; \
            int32_t sigma12_ = (int32_t)(xy)[(base) + b_]; \
            if (sigma12_ > 0 && sigma2_sq_ > 0) { \
                double sv_d_ = sv_arr_[b_]; \
                uint32_t sv_sq_ = sv_d_ > 0.0 ? (uint32_t)sv_d_ : 0; \
                uint32_t numer1_ = (sv_sq_ + sigma_nsq); \
                int64_t numer1_tmp_ = (int64_t)(gg_xx_arr_[b_]) + numer1_; \
                accum_num_log += log2_64(log2_table, numer1_tmp_) - log2_64(log2_table, numer1_); \
            } \
        } \
    } \
} while (0)


void vif_statistic_8_avx2(struct VifPublicState *s, float *num, float *den, unsigned w, unsigned h) {
    assert(vif_filter1d_width[0] == 17);
    static const unsigned fwidth = 17;
    const uint16_t *vif_filt_s0 = vif_filter1d_table[0];
    VifBuffer buf = s->buf;

    //float equivalent of 2. (2 * 65536)
    static const int32_t sigma_nsq = 65536 << 1;
    double vif_enhn_gain_limit = s->vif_enhn_gain_limit;

    int64_t accum_num_log = 0;
    int64_t accum_den_log = 0;
    int64_t accum_num_non_log = 0;
    int64_t accum_den_non_log = 0;
    uint16_t *log2_table = s->log2_table;

    // variables used for 16 sample block vif computation
    ALIGNED(32) uint32_t xx[16];
    ALIGNED(32) uint32_t yy[16];
    ALIGNED(32) uint32_t xy[16];

    // Pre-broadcast filter coefficients (hoisted from inner loops)
    __m256i vf_epi16[VIF_FILT_MAX];
    __m256i vf_epi32[VIF_FILT_MAX];
    __m256i vf_epi64[VIF_FILT_MAX];
    for (unsigned fi = 0; fi < fwidth; ++fi) {
        vf_epi16[fi] = _mm256_set1_epi16(vif_filt_s0[fi]);
        vf_epi32[fi] = _mm256_set1_epi32(vif_filt_s0[fi]);
        vf_epi64[fi] = _mm256_set1_epi64x(vif_filt_s0[fi]);
    }

    // loop on row, each iteration produces one line of output
    for (unsigned i = 0; i < h; ++i) {
        // Precompute row pointers — hoisted from inner loops to eliminate
        // scalar imul instructions for row offset calculation.
        const uint8_t *ref_center = (const uint8_t *)buf.ref + buf.stride * i;
        const uint8_t *dis_center = (const uint8_t *)buf.dis + buf.stride * i;
        const uint8_t *ref_above[VIF_FILT_MAX / 2];
        const uint8_t *ref_below[VIF_FILT_MAX / 2];
        const uint8_t *dis_above[VIF_FILT_MAX / 2];
        const uint8_t *dis_below[VIF_FILT_MAX / 2];
        for (unsigned tap = 0; tap < fwidth / 2; tap++) {
            int ii_above = i - fwidth / 2 + tap;
            int ii_below = i + fwidth / 2 - tap;
            ref_above[tap] = (const uint8_t *)buf.ref + buf.stride * ii_above;
            ref_below[tap] = (const uint8_t *)buf.ref + buf.stride * ii_below;
            dis_above[tap] = (const uint8_t *)buf.dis + buf.stride * ii_above;
            dis_below[tap] = (const uint8_t *)buf.dis + buf.stride * ii_below;
        }

        // Filter vertically
        // First consider all blocks of 16 elements until it's not possible anymore
        unsigned n = w >> 4;
        for (unsigned jj = 0; jj < n << 4; jj += 16) {
            __m256i accum_ref_left, accum_ref_right;
            __m256i accum_dis_left, accum_dis_right;
            __m256i accum_ref_dis_left, accum_ref_dis_right;
            __m256i accum_mu2_left, accum_mu2_right;
            __m256i accum_mu1_left, accum_mu1_right;

            __m256i f0 = vf_epi16[fwidth / 2];
            __m256i r0 = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(ref_center + jj)));
            __m256i d0 = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(dis_center + jj)));

            // filtered r,d
            multiply2(accum_mu1_left, accum_mu1_right, r0, f0);
            multiply2(accum_mu2_left, accum_mu2_right, d0, f0);

            // filtered(r * r, d * d, r * d)
            multiply3(accum_ref_left, accum_ref_right, r0, r0, f0);
            multiply3(accum_dis_left, accum_dis_right, d0, d0, f0);
            multiply3(accum_ref_dis_left, accum_ref_dis_right, d0, r0, f0);

            for (unsigned int tap = 0; tap < fwidth / 2; tap++) {
                __m256i f16 = vf_epi16[tap];
                __m256i f32 = vf_epi32[tap];
                __m256i r0 = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(ref_above[tap] + jj)));
                __m256i r1 = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(ref_below[tap] + jj)));
                __m256i d0 = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(dis_above[tap] + jj)));
                __m256i d1 = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(dis_below[tap] + jj)));

                // Interleave symmetric tap pairs for shared MADD operations
                __m256i r_lo = _mm256_unpacklo_epi16(r0, r1);
                __m256i r_hi = _mm256_unpackhi_epi16(r0, r1);
                __m256i d_lo = _mm256_unpacklo_epi16(d0, d1);
                __m256i d_hi = _mm256_unpackhi_epi16(d0, d1);

                // mu filter: (r0+r1)*f, (d0+d1)*f via madd_epi16
                accum_mu1_left = _mm256_add_epi32(accum_mu1_left, _mm256_madd_epi16(r_lo, f16));
                accum_mu1_right = _mm256_add_epi32(accum_mu1_right, _mm256_madd_epi16(r_hi, f16));
                accum_mu2_left = _mm256_add_epi32(accum_mu2_left, _mm256_madd_epi16(d_lo, f16));
                accum_mu2_right = _mm256_add_epi32(accum_mu2_right, _mm256_madd_epi16(d_hi, f16));

                // Squared/cross filter: madd_epi16 pre-sums symmetric pairs
                // (r0²+r1²), (d0²+d1²), (r0*d0+r1*d1) in 32-bit, then
                // mullo_epi32 scales by filter coeff. Replaces 6 separate
                // multiply3_and_accumulate calls (18 mul16 ops) with 6 madd
                // + 6 mullo_epi32 (12 ops total).
                accum_ref_left = _mm256_add_epi32(accum_ref_left,
                    _mm256_mullo_epi32(_mm256_madd_epi16(r_lo, r_lo), f32));
                accum_ref_right = _mm256_add_epi32(accum_ref_right,
                    _mm256_mullo_epi32(_mm256_madd_epi16(r_hi, r_hi), f32));
                accum_dis_left = _mm256_add_epi32(accum_dis_left,
                    _mm256_mullo_epi32(_mm256_madd_epi16(d_lo, d_lo), f32));
                accum_dis_right = _mm256_add_epi32(accum_dis_right,
                    _mm256_mullo_epi32(_mm256_madd_epi16(d_hi, d_hi), f32));
                accum_ref_dis_left = _mm256_add_epi32(accum_ref_dis_left,
                    _mm256_mullo_epi32(_mm256_madd_epi16(r_lo, d_lo), f32));
                accum_ref_dis_right = _mm256_add_epi32(accum_ref_dis_right,
                    _mm256_mullo_epi32(_mm256_madd_epi16(r_hi, d_hi), f32));
            }

            __m256i x = _mm256_set1_epi32(128);

            accum_mu1_left = _mm256_add_epi32(accum_mu1_left, x);
            accum_mu1_right = _mm256_add_epi32(accum_mu1_right, x);
            accum_mu2_left = _mm256_add_epi32(accum_mu2_left, x);
            accum_mu2_right = _mm256_add_epi32(accum_mu2_right, x);

            accum_mu1_left = _mm256_srli_epi32(accum_mu1_left, 0x08);
            accum_mu1_right = _mm256_srli_epi32(accum_mu1_right, 0x08);
            accum_mu2_left = _mm256_srli_epi32(accum_mu2_left, 0x08);
            accum_mu2_right = _mm256_srli_epi32(accum_mu2_right, 0x08);

            shuffle_and_save(buf.tmp.mu1 + jj, accum_mu1_left, accum_mu1_right);
            shuffle_and_save(buf.tmp.mu2 + jj, accum_mu2_left, accum_mu2_right);
            shuffle_and_save(buf.tmp.ref + jj, accum_ref_left, accum_ref_right);
            shuffle_and_save(buf.tmp.dis + jj, accum_dis_left, accum_dis_right);
            shuffle_and_save(buf.tmp.ref_dis + jj, accum_ref_dis_left, accum_ref_dis_right);
        }

        // Then consider the remaining elements individually
        for (unsigned j = n << 4; j < w; ++j) {
            uint32_t accum_mu1 = 0;
            uint32_t accum_mu2 = 0;
            uint64_t accum_ref = 0;
            uint64_t accum_dis = 0;
            uint64_t accum_ref_dis = 0;

            for (unsigned fi = 0; fi < fwidth; ++fi) {
                int ii = i - fwidth / 2;
                int ii_check = ii + fi;
                const uint16_t fcoeff = vif_filt_s0[fi];
                const uint8_t *ref = (uint8_t*)buf.ref;
                const uint8_t *dis = (uint8_t*)buf.dis;
                uint16_t imgcoeff_ref = ref[ii_check * buf.stride + j];
                uint16_t imgcoeff_dis = dis[ii_check * buf.stride + j];
                uint32_t img_coeff_ref = fcoeff * (uint32_t)imgcoeff_ref;
                uint32_t img_coeff_dis = fcoeff * (uint32_t)imgcoeff_dis;
                accum_mu1 += img_coeff_ref;
                accum_mu2 += img_coeff_dis;
                accum_ref += img_coeff_ref * (uint64_t)imgcoeff_ref;
                accum_dis += img_coeff_dis * (uint64_t)imgcoeff_dis;
                accum_ref_dis += img_coeff_ref * (uint64_t)imgcoeff_dis;
            }

            buf.tmp.mu1[j] = (accum_mu1 + 128) >> 8;
            buf.tmp.mu2[j] = (accum_mu2 + 128) >> 8;
            buf.tmp.ref[j] = accum_ref;
            buf.tmp.dis[j] = accum_dis;
            buf.tmp.ref_dis[j] = accum_ref_dis;
        }

        PADDING_SQ_DATA(buf, w, fwidth / 2);

        //HORIZONTAL
        for (unsigned j = 0; j < n << 4; j += 16) {
            __m256i mu1_lo;
            __m256i mu1_hi;
            __m256i mu1sq_lo; // shuffled
            __m256i mu1sq_hi; // shuffled
            __m256i mu2sq_lo; // shuffled
            __m256i mu2sq_hi; // shuffled
            __m256i mu1mu2_lo; // shuffled
            __m256i mu1mu2_hi; // shuffled

            // compute mu1 and mu2 filtered (fused), then mu1sq, mu2sq, mu1mu2
            {
                __m256i fq = vf_epi32[fwidth / 2];
                mu1_lo = _mm256_mullo_epi32(_mm256_loadu_si256((__m256i*)(buf.tmp.mu1 + j + 0)), fq);
                mu1_hi = _mm256_mullo_epi32(_mm256_loadu_si256((__m256i*)(buf.tmp.mu1 + j + 8)), fq);
                __m256i mu2_lo = _mm256_mullo_epi32(_mm256_loadu_si256((__m256i*)(buf.tmp.mu2 + j + 0)), fq);
                __m256i mu2_hi = _mm256_mullo_epi32(_mm256_loadu_si256((__m256i*)(buf.tmp.mu2 + j + 8)), fq);
                for (unsigned fj = 0; fj < fwidth / 2; ++fj) {
                    __m256i fq = vf_epi32[fj];
                    // Symmetric-tap pre-addition: add left+right in 32-bit before multiply.
                    // mu values up to 65280 (coeff_sum=65536, pixel=255, >>8). Sum ≤130560 fits 32-bit.
                    // Product 130560*max_coeff(7784) = 1.02B fits 32-bit.
                    __m256i mu1_sum_lo = _mm256_add_epi32(
                        _mm256_loadu_si256((__m256i*)(buf.tmp.mu1 + j - fwidth / 2 + fj + 0)),
                        _mm256_loadu_si256((__m256i*)(buf.tmp.mu1 + j + fwidth / 2 - fj + 0)));
                    __m256i mu1_sum_hi = _mm256_add_epi32(
                        _mm256_loadu_si256((__m256i*)(buf.tmp.mu1 + j - fwidth / 2 + fj + 8)),
                        _mm256_loadu_si256((__m256i*)(buf.tmp.mu1 + j + fwidth / 2 - fj + 8)));
                    mu1_lo = _mm256_add_epi32(mu1_lo, _mm256_mullo_epi32(mu1_sum_lo, fq));
                    mu1_hi = _mm256_add_epi32(mu1_hi, _mm256_mullo_epi32(mu1_sum_hi, fq));
                    __m256i mu2_sum_lo = _mm256_add_epi32(
                        _mm256_loadu_si256((__m256i*)(buf.tmp.mu2 + j - fwidth / 2 + fj + 0)),
                        _mm256_loadu_si256((__m256i*)(buf.tmp.mu2 + j + fwidth / 2 - fj + 0)));
                    __m256i mu2_sum_hi = _mm256_add_epi32(
                        _mm256_loadu_si256((__m256i*)(buf.tmp.mu2 + j - fwidth / 2 + fj + 8)),
                        _mm256_loadu_si256((__m256i*)(buf.tmp.mu2 + j + fwidth / 2 - fj + 8)));
                    mu2_lo = _mm256_add_epi32(mu2_lo, _mm256_mullo_epi32(mu2_sum_lo, fq));
                    mu2_hi = _mm256_add_epi32(mu2_hi, _mm256_mullo_epi32(mu2_sum_hi, fq));
                }

                // mu1sq from mu1
                __m256i acc0_lo = _mm256_unpacklo_epi32(mu1_lo, _mm256_setzero_si256());
                __m256i acc0_hi = _mm256_unpackhi_epi32(mu1_lo, _mm256_setzero_si256());
                __m256i acc1_lo = _mm256_unpacklo_epi32(mu1_hi, _mm256_setzero_si256());
                __m256i acc1_hi = _mm256_unpackhi_epi32(mu1_hi, _mm256_setzero_si256());

                // mu2 unpacked (need before squaring overwrites acc0/acc1)
                __m256i mu2_0_lo = _mm256_unpacklo_epi32(mu2_lo, _mm256_setzero_si256());
                __m256i mu2_0_hi = _mm256_unpackhi_epi32(mu2_lo, _mm256_setzero_si256());
                __m256i mu2_1_lo = _mm256_unpacklo_epi32(mu2_hi, _mm256_setzero_si256());
                __m256i mu2_1_hi = _mm256_unpackhi_epi32(mu2_hi, _mm256_setzero_si256());

                // mu1*mu2
                __m256i mu1mu2_0_lo = _mm256_mul_epu32(acc0_lo, mu2_0_lo);
                __m256i mu1mu2_0_hi = _mm256_mul_epu32(acc0_hi, mu2_0_hi);
                mu1mu2_0_lo = _mm256_srli_epi64(_mm256_add_epi64(mu1mu2_0_lo, _mm256_set1_epi64x(0x80000000)), 32);
                mu1mu2_0_hi = _mm256_srli_epi64(_mm256_add_epi64(mu1mu2_0_hi, _mm256_set1_epi64x(0x80000000)), 32);
                __m256i mu1mu2_1_lo = _mm256_mul_epu32(acc1_lo, mu2_1_lo);
                __m256i mu1mu2_1_hi = _mm256_mul_epu32(acc1_hi, mu2_1_hi);
                mu1mu2_1_lo = _mm256_srli_epi64(_mm256_add_epi64(mu1mu2_1_lo, _mm256_set1_epi64x(0x80000000)), 32);
                mu1mu2_1_hi = _mm256_srli_epi64(_mm256_add_epi64(mu1mu2_1_hi, _mm256_set1_epi64x(0x80000000)), 32);

                // mu1*mu1
                acc0_lo = _mm256_mul_epu32(acc0_lo, acc0_lo);
                acc0_hi = _mm256_mul_epu32(acc0_hi, acc0_hi);
                acc0_lo = _mm256_srli_epi64(_mm256_add_epi64(acc0_lo, _mm256_set1_epi64x(0x80000000)), 32);
                acc0_hi = _mm256_srli_epi64(_mm256_add_epi64(acc0_hi, _mm256_set1_epi64x(0x80000000)), 32);
                acc1_lo = _mm256_mul_epu32(acc1_lo, acc1_lo);
                acc1_hi = _mm256_mul_epu32(acc1_hi, acc1_hi);
                acc1_lo = _mm256_srli_epi64(_mm256_add_epi64(acc1_lo, _mm256_set1_epi64x(0x80000000)), 32);
                acc1_hi = _mm256_srli_epi64(_mm256_add_epi64(acc1_hi, _mm256_set1_epi64x(0x80000000)), 32);

                mu1sq_lo = _mm256_castps_si256(_mm256_shuffle_ps(
                    _mm256_castsi256_ps(acc0_lo), _mm256_castsi256_ps(acc0_hi), 0x88));
                mu1sq_hi = _mm256_castps_si256(_mm256_shuffle_ps(
                    _mm256_castsi256_ps(acc1_lo), _mm256_castsi256_ps(acc1_hi), 0x88));

                // mu2*mu2
                mu2_0_lo = _mm256_mul_epu32(mu2_0_lo, mu2_0_lo);
                mu2_0_hi = _mm256_mul_epu32(mu2_0_hi, mu2_0_hi);
                mu2_0_lo = _mm256_srli_epi64(_mm256_add_epi64(mu2_0_lo, _mm256_set1_epi64x(0x80000000)), 32);
                mu2_0_hi = _mm256_srli_epi64(_mm256_add_epi64(mu2_0_hi, _mm256_set1_epi64x(0x80000000)), 32);
                mu2_1_lo = _mm256_mul_epu32(mu2_1_lo, mu2_1_lo);
                mu2_1_hi = _mm256_mul_epu32(mu2_1_hi, mu2_1_hi);
                mu2_1_lo = _mm256_srli_epi64(_mm256_add_epi64(mu2_1_lo, _mm256_set1_epi64x(0x80000000)), 32);
                mu2_1_hi = _mm256_srli_epi64(_mm256_add_epi64(mu2_1_hi, _mm256_set1_epi64x(0x80000000)), 32);

                mu2sq_lo = _mm256_castps_si256(_mm256_shuffle_ps(
                    _mm256_castsi256_ps(mu2_0_lo), _mm256_castsi256_ps(mu2_0_hi), 0x88));
                mu2sq_hi = _mm256_castps_si256(_mm256_shuffle_ps(
                    _mm256_castsi256_ps(mu2_1_lo), _mm256_castsi256_ps(mu2_1_hi), 0x88));

                mu1mu2_lo = _mm256_castps_si256(_mm256_shuffle_ps(
                    _mm256_castsi256_ps(mu1mu2_0_lo), _mm256_castsi256_ps(mu1mu2_0_hi), 0x88));
                mu1mu2_hi = _mm256_castps_si256(_mm256_shuffle_ps(
                    _mm256_castsi256_ps(mu1mu2_1_lo), _mm256_castsi256_ps(mu1mu2_1_hi), 0x88));
            }

            // compute xx (refsq filtered - mu1*mu1), yy (dissq filtered - mu2*mu2),
            // xy (ref*dis filtered - mu1*mu2) -- fused three-pass horizontal filter
            {
                __m256i rounder = _mm256_set1_epi64x(0x8000);
                __m256i fq = vf_epi64[fwidth / 2];

                // ref (xx)
                __m256i rm0 = _mm256_loadu_si256((__m256i*)(buf.tmp.ref + j + 0));
                __m256i rm1 = _mm256_loadu_si256((__m256i*)(buf.tmp.ref + j + 8));
                __m256i racc0 = _mm256_add_epi64(rounder, _mm256_mul_epu32(_mm256_unpacklo_epi32(rm0, _mm256_setzero_si256()), fq));
                __m256i racc1 = _mm256_add_epi64(rounder, _mm256_mul_epu32(_mm256_unpackhi_epi32(rm0, _mm256_setzero_si256()), fq));
                __m256i racc2 = _mm256_add_epi64(rounder, _mm256_mul_epu32(_mm256_unpacklo_epi32(rm1, _mm256_setzero_si256()), fq));
                __m256i racc3 = _mm256_add_epi64(rounder, _mm256_mul_epu32(_mm256_unpackhi_epi32(rm1, _mm256_setzero_si256()), fq));

                // dis (yy)
                __m256i dm0 = _mm256_loadu_si256((__m256i*)(buf.tmp.dis + j + 0));
                __m256i dm1 = _mm256_loadu_si256((__m256i*)(buf.tmp.dis + j + 8));
                __m256i dacc0 = _mm256_add_epi64(rounder, _mm256_mul_epu32(_mm256_unpacklo_epi32(dm0, _mm256_setzero_si256()), fq));
                __m256i dacc1 = _mm256_add_epi64(rounder, _mm256_mul_epu32(_mm256_unpackhi_epi32(dm0, _mm256_setzero_si256()), fq));
                __m256i dacc2 = _mm256_add_epi64(rounder, _mm256_mul_epu32(_mm256_unpacklo_epi32(dm1, _mm256_setzero_si256()), fq));
                __m256i dacc3 = _mm256_add_epi64(rounder, _mm256_mul_epu32(_mm256_unpackhi_epi32(dm1, _mm256_setzero_si256()), fq));

                // ref_dis (xy)
                __m256i xm0 = _mm256_loadu_si256((__m256i*)(buf.tmp.ref_dis + j + 0));
                __m256i xm1 = _mm256_loadu_si256((__m256i*)(buf.tmp.ref_dis + j + 8));
                __m256i xacc0 = _mm256_add_epi64(rounder, _mm256_mul_epu32(_mm256_unpacklo_epi32(xm0, _mm256_setzero_si256()), fq));
                __m256i xacc1 = _mm256_add_epi64(rounder, _mm256_mul_epu32(_mm256_unpackhi_epi32(xm0, _mm256_setzero_si256()), fq));
                __m256i xacc2 = _mm256_add_epi64(rounder, _mm256_mul_epu32(_mm256_unpacklo_epi32(xm1, _mm256_setzero_si256()), fq));
                __m256i xacc3 = _mm256_add_epi64(rounder, _mm256_mul_epu32(_mm256_unpackhi_epi32(xm1, _mm256_setzero_si256()), fq));

                for (unsigned fj = 0; fj < fwidth / 2; ++fj) {
                    __m256i fq = vf_epi64[fj];

                    __m256i m0 = _mm256_loadu_si256((__m256i*)(buf.tmp.ref + j - fwidth / 2 + fj + 0));
                    __m256i m1 = _mm256_loadu_si256((__m256i*)(buf.tmp.ref + j - fwidth / 2 + fj + 8));
                    __m256i m2 = _mm256_loadu_si256((__m256i*)(buf.tmp.ref + j + fwidth / 2 - fj + 0));
                    __m256i m3 = _mm256_loadu_si256((__m256i*)(buf.tmp.ref + j + fwidth / 2 - fj + 8));

                    racc0 = _mm256_add_epi64(racc0, _mm256_mul_epu32(_mm256_unpacklo_epi32(m0, _mm256_setzero_si256()), fq));
                    racc1 = _mm256_add_epi64(racc1, _mm256_mul_epu32(_mm256_unpackhi_epi32(m0, _mm256_setzero_si256()), fq));
                    racc2 = _mm256_add_epi64(racc2, _mm256_mul_epu32(_mm256_unpacklo_epi32(m1, _mm256_setzero_si256()), fq));
                    racc3 = _mm256_add_epi64(racc3, _mm256_mul_epu32(_mm256_unpackhi_epi32(m1, _mm256_setzero_si256()), fq));

                    racc0 = _mm256_add_epi64(racc0, _mm256_mul_epu32(_mm256_unpacklo_epi32(m2, _mm256_setzero_si256()), fq));
                    racc1 = _mm256_add_epi64(racc1, _mm256_mul_epu32(_mm256_unpackhi_epi32(m2, _mm256_setzero_si256()), fq));
                    racc2 = _mm256_add_epi64(racc2, _mm256_mul_epu32(_mm256_unpacklo_epi32(m3, _mm256_setzero_si256()), fq));
                    racc3 = _mm256_add_epi64(racc3, _mm256_mul_epu32(_mm256_unpackhi_epi32(m3, _mm256_setzero_si256()), fq));

                    m0 = _mm256_loadu_si256((__m256i*)(buf.tmp.dis + j - fwidth / 2 + fj + 0));
                    m1 = _mm256_loadu_si256((__m256i*)(buf.tmp.dis + j - fwidth / 2 + fj + 8));
                    m2 = _mm256_loadu_si256((__m256i*)(buf.tmp.dis + j + fwidth / 2 - fj + 0));
                    m3 = _mm256_loadu_si256((__m256i*)(buf.tmp.dis + j + fwidth / 2 - fj + 8));

                    dacc0 = _mm256_add_epi64(dacc0, _mm256_mul_epu32(_mm256_unpacklo_epi32(m0, _mm256_setzero_si256()), fq));
                    dacc1 = _mm256_add_epi64(dacc1, _mm256_mul_epu32(_mm256_unpackhi_epi32(m0, _mm256_setzero_si256()), fq));
                    dacc2 = _mm256_add_epi64(dacc2, _mm256_mul_epu32(_mm256_unpacklo_epi32(m1, _mm256_setzero_si256()), fq));
                    dacc3 = _mm256_add_epi64(dacc3, _mm256_mul_epu32(_mm256_unpackhi_epi32(m1, _mm256_setzero_si256()), fq));

                    dacc0 = _mm256_add_epi64(dacc0, _mm256_mul_epu32(_mm256_unpacklo_epi32(m2, _mm256_setzero_si256()), fq));
                    dacc1 = _mm256_add_epi64(dacc1, _mm256_mul_epu32(_mm256_unpackhi_epi32(m2, _mm256_setzero_si256()), fq));
                    dacc2 = _mm256_add_epi64(dacc2, _mm256_mul_epu32(_mm256_unpacklo_epi32(m3, _mm256_setzero_si256()), fq));
                    dacc3 = _mm256_add_epi64(dacc3, _mm256_mul_epu32(_mm256_unpackhi_epi32(m3, _mm256_setzero_si256()), fq));

                    m0 = _mm256_loadu_si256((__m256i*)(buf.tmp.ref_dis + j - fwidth / 2 + fj + 0));
                    m1 = _mm256_loadu_si256((__m256i*)(buf.tmp.ref_dis + j - fwidth / 2 + fj + 8));
                    m2 = _mm256_loadu_si256((__m256i*)(buf.tmp.ref_dis + j + fwidth / 2 - fj + 0));
                    m3 = _mm256_loadu_si256((__m256i*)(buf.tmp.ref_dis + j + fwidth / 2 - fj + 8));

                    xacc0 = _mm256_add_epi64(xacc0, _mm256_mul_epu32(_mm256_unpacklo_epi32(m0, _mm256_setzero_si256()), fq));
                    xacc1 = _mm256_add_epi64(xacc1, _mm256_mul_epu32(_mm256_unpackhi_epi32(m0, _mm256_setzero_si256()), fq));
                    xacc2 = _mm256_add_epi64(xacc2, _mm256_mul_epu32(_mm256_unpacklo_epi32(m1, _mm256_setzero_si256()), fq));
                    xacc3 = _mm256_add_epi64(xacc3, _mm256_mul_epu32(_mm256_unpackhi_epi32(m1, _mm256_setzero_si256()), fq));

                    xacc0 = _mm256_add_epi64(xacc0, _mm256_mul_epu32(_mm256_unpacklo_epi32(m2, _mm256_setzero_si256()), fq));
                    xacc1 = _mm256_add_epi64(xacc1, _mm256_mul_epu32(_mm256_unpackhi_epi32(m2, _mm256_setzero_si256()), fq));
                    xacc2 = _mm256_add_epi64(xacc2, _mm256_mul_epu32(_mm256_unpacklo_epi32(m3, _mm256_setzero_si256()), fq));
                    xacc3 = _mm256_add_epi64(xacc3, _mm256_mul_epu32(_mm256_unpackhi_epi32(m3, _mm256_setzero_si256()), fq));
                }

                // xx: shift, pack, subtract mu1sq
                racc0 = _mm256_srli_epi64(racc0, 16);
                racc1 = _mm256_srli_epi64(racc1, 16);
                racc2 = _mm256_srli_epi64(racc2, 16);
                racc3 = _mm256_srli_epi64(racc3, 16);
                racc0 = _mm256_castps_si256(_mm256_shuffle_ps(
                    _mm256_castsi256_ps(racc0), _mm256_castsi256_ps(racc1), 0x88));
                racc1 = _mm256_castps_si256(_mm256_shuffle_ps(
                    _mm256_castsi256_ps(racc2), _mm256_castsi256_ps(racc3), 0x88));
                racc0 = _mm256_sub_epi32(racc0, mu1sq_lo);
                racc1 = _mm256_sub_epi32(racc1, mu1sq_hi);
                _mm256_storeu_si256((__m256i*)& xx[0], racc0);
                _mm256_storeu_si256((__m256i*)& xx[8], racc1);

                // yy: shift, pack, subtract mu2sq
                dacc0 = _mm256_srli_epi64(dacc0, 16);
                dacc1 = _mm256_srli_epi64(dacc1, 16);
                dacc2 = _mm256_srli_epi64(dacc2, 16);
                dacc3 = _mm256_srli_epi64(dacc3, 16);
                dacc0 = _mm256_castps_si256(_mm256_shuffle_ps(
                    _mm256_castsi256_ps(dacc0), _mm256_castsi256_ps(dacc1), 0x88));
                dacc1 = _mm256_castps_si256(_mm256_shuffle_ps(
                    _mm256_castsi256_ps(dacc2), _mm256_castsi256_ps(dacc3), 0x88));
                dacc0 = _mm256_sub_epi32(dacc0, mu2sq_lo);
                dacc1 = _mm256_sub_epi32(dacc1, mu2sq_hi);
                _mm256_storeu_si256((__m256i*) & yy[0], _mm256_max_epi32(dacc0, _mm256_setzero_si256()));
                _mm256_storeu_si256((__m256i*) & yy[8], _mm256_max_epi32(dacc1, _mm256_setzero_si256()));

                // xy: shift, pack, subtract mu1mu2
                xacc0 = _mm256_srli_epi64(xacc0, 16);
                xacc1 = _mm256_srli_epi64(xacc1, 16);
                xacc2 = _mm256_srli_epi64(xacc2, 16);
                xacc3 = _mm256_srli_epi64(xacc3, 16);
                xacc0 = _mm256_castps_si256(_mm256_shuffle_ps(
                    _mm256_castsi256_ps(xacc0), _mm256_castsi256_ps(xacc1), 0x88));
                xacc1 = _mm256_castps_si256(_mm256_shuffle_ps(
                    _mm256_castsi256_ps(xacc2), _mm256_castsi256_ps(xacc3), 0x88));
                xacc0 = _mm256_sub_epi32(xacc0, mu1mu2_lo);
                xacc1 = _mm256_sub_epi32(xacc1, mu1mu2_hi);
                _mm256_storeu_si256((__m256i*) & xy[0], xacc0);
                _mm256_storeu_si256((__m256i*) & xy[8], xacc1);
            }

            /* Vectorized threshold test + non-log accumulation, bitmask log-path iteration */
            {
                __m256i xx_lo = _mm256_loadu_si256((__m256i*)&xx[0]);
                __m256i yy_lo = _mm256_loadu_si256((__m256i*)&yy[0]);
                VIF_ACCUM_BLOCK(xx_lo, yy_lo, xx, yy, xy, 0,
                                sigma_nsq, log2_table, vif_enhn_gain_limit,
                                accum_num_log, accum_den_log,
                                accum_num_non_log, accum_den_non_log);

                __m256i xx_hi = _mm256_loadu_si256((__m256i*)&xx[8]);
                __m256i yy_hi = _mm256_loadu_si256((__m256i*)&yy[8]);
                VIF_ACCUM_BLOCK(xx_hi, yy_hi, xx, yy, xy, 8,
                                sigma_nsq, log2_table, vif_enhn_gain_limit,
                                accum_num_log, accum_den_log,
                                accum_num_non_log, accum_den_non_log);
            }
        }
        if ((n << 4) != w) {
            VifResiduals residuals = vif_compute_line_residuals(s, n << 4, w, 0);
            accum_num_log += residuals.accum_num_log;
            accum_den_log += residuals.accum_den_log;
            accum_num_non_log += residuals.accum_num_non_log;
            accum_den_non_log += residuals.accum_den_non_log;
        }
    }

    //log has to be divided by 2048 as log_value = log2(i*2048)  i=16384 to 65535
    //num[0] = accum_num_log / 2048.0 + (accum_den_non_log - (accum_num_non_log / 65536.0) / (255.0*255.0));
    //den[0] = accum_den_log / 2048.0 + accum_den_non_log;

    //changed calculation to increase performance
    num[0] = accum_num_log / 2048.0 + (accum_den_non_log - ((accum_num_non_log) / 16384.0) / (65025.0));
    den[0] = accum_den_log / 2048.0 + accum_den_non_log;

}

void vif_statistic_16_avx2(struct VifPublicState *s, float *num, float *den, unsigned w, unsigned h, int bpc, int scale) {
    const unsigned fwidth = vif_filter1d_width[scale];
    const uint16_t *vif_filt = vif_filter1d_table[scale];
    VifBuffer buf = s->buf;
    const ptrdiff_t stride = buf.stride / sizeof(uint16_t);
    int fwidth_half = fwidth >> 1;

    int32_t add_shift_round_VP, shift_VP;
    int32_t add_shift_round_VP_sq, shift_VP_sq;

    //float equivalent of 2. (2 * 65536)
    static const int32_t sigma_nsq = 65536 << 1;

    int64_t accum_num_log = 0;
    int64_t accum_den_log = 0;
    int64_t accum_num_non_log = 0;
    int64_t accum_den_non_log = 0;
    const uint16_t *log2_table = s->log2_table;
    double vif_enhn_gain_limit = s->vif_enhn_gain_limit;

    // variables used for 16 sample block vif computation
    ALIGNED(32) uint32_t xx[16];
    ALIGNED(32) uint32_t yy[16];
    ALIGNED(32) uint32_t xy[16];

    if (scale == 0) {
        shift_VP = bpc;
        add_shift_round_VP = 1 << (bpc - 1);
        shift_VP_sq = (bpc - 8) * 2;
        add_shift_round_VP_sq = (bpc == 8) ? 0 : 1 << (shift_VP_sq - 1);
    } else {
        shift_VP = 16;
        add_shift_round_VP = 32768;
        shift_VP_sq = 16;
        add_shift_round_VP_sq = 32768;
    }

    for (unsigned i = 0; i < h; ++i) {
        // VERTICAL
        int ii = i - fwidth_half;
        unsigned n = w >> 4;
        for (unsigned j = 0; j < n << 4; j = j + 16) {
            __m256i mask2 = _mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0);
            int ii_check = ii;

            uint16_t *ref = buf.ref;
            uint16_t *dis = buf.dis;
            __m256i accumr_lo, accumr_hi, accumd_lo, accumd_hi, rmul1, rmul2,
                dmul1, dmul2, accumref1, accumref2, accumref3, accumref4,
                accumrefdis1, accumrefdis2, accumrefdis3, accumrefdis4,
                accumdis1, accumdis2, accumdis3, accumdis4;
            accumr_lo = accumr_hi = accumd_lo = accumd_hi = rmul1 = rmul2 =
                dmul1 = dmul2 = accumref1 = accumref2 = accumref3 = accumref4 =
                    accumrefdis1 = accumrefdis2 = accumrefdis3 = accumrefdis4 =
                        accumdis1 = accumdis2 = accumdis3 = accumdis4 =
                            _mm256_setzero_si256();
            __m256i addnum = _mm256_set1_epi32(add_shift_round_VP);
            for (unsigned fi = 0; fi < fwidth; ++fi, ii_check = ii + fi) {
                __m256i f1 = _mm256_set1_epi16(vif_filt[fi]);
                __m256i ref1 = _mm256_loadu_si256(
                    (__m256i *)(ref + (ii_check * stride) + j));
                __m256i dis1 = _mm256_loadu_si256(
                    (__m256i *)(dis + (ii_check * stride) + j));
                __m256i result2 = _mm256_mulhi_epu16(ref1, f1);
                __m256i result2lo = _mm256_mullo_epi16(ref1, f1);
                rmul1 = _mm256_unpacklo_epi16(result2lo, result2);
                rmul2 = _mm256_unpackhi_epi16(result2lo, result2);
                accumr_lo = _mm256_add_epi32(accumr_lo, rmul1);
                accumr_hi = _mm256_add_epi32(accumr_hi, rmul2);
                __m256i d0 = _mm256_mulhi_epu16(dis1, f1);
                __m256i d0lo = _mm256_mullo_epi16(dis1, f1);
                dmul1 = _mm256_unpacklo_epi16(d0lo, d0);
                dmul2 = _mm256_unpackhi_epi16(d0lo, d0);
                accumd_lo = _mm256_add_epi32(accumd_lo, dmul1);
                accumd_hi = _mm256_add_epi32(accumd_hi, dmul2);

                __m256i sg0 =
                    _mm256_cvtepu32_epi64(_mm256_castsi256_si128(rmul1));
                __m256i sg1 =
                    _mm256_cvtepu32_epi64(_mm256_extracti128_si256(rmul1, 1));
                __m256i sg2 =
                    _mm256_cvtepu32_epi64(_mm256_castsi256_si128(rmul2));
                __m256i sg3 =
                    _mm256_cvtepu32_epi64(_mm256_extracti128_si256(rmul2, 1));
                __m128i l0 = _mm256_castsi256_si128(ref1);
                __m128i l1 = _mm256_extracti128_si256(ref1, 1);
                accumref1 = _mm256_add_epi64(
                    accumref1,
                    _mm256_mul_epu32(sg0, _mm256_cvtepu16_epi64(l0)));
                accumref2 = _mm256_add_epi64(
                    accumref2,
                    _mm256_mul_epu32(
                        sg2, _mm256_cvtepu16_epi64(_mm_bsrli_si128(l0, 8))));
                accumref3 = _mm256_add_epi64(
                    accumref3,
                    _mm256_mul_epu32(sg1, _mm256_cvtepu16_epi64(l1)));
                accumref4 = _mm256_add_epi64(
                    accumref4,
                    _mm256_mul_epu32(
                        sg3, _mm256_cvtepu16_epi64(_mm_bsrli_si128(l1, 8))));
                l0 = _mm256_castsi256_si128(dis1);
                l1 = _mm256_extracti128_si256(dis1, 1);
                accumrefdis1 = _mm256_add_epi64(
                    accumrefdis1,
                    _mm256_mul_epu32(sg0, _mm256_cvtepu16_epi64(l0)));
                accumrefdis2 = _mm256_add_epi64(
                    accumrefdis2,
                    _mm256_mul_epu32(
                        sg2, _mm256_cvtepu16_epi64(_mm_bsrli_si128(l0, 8))));
                accumrefdis3 = _mm256_add_epi64(
                    accumrefdis3,
                    _mm256_mul_epu32(sg1, _mm256_cvtepu16_epi64(l1)));
                accumrefdis4 = _mm256_add_epi64(
                    accumrefdis4,
                    _mm256_mul_epu32(
                        sg3, _mm256_cvtepu16_epi64(_mm_bsrli_si128(l1, 8))));
                __m256i sd0 =
                    _mm256_cvtepu32_epi64(_mm256_castsi256_si128(dmul1));
                __m256i sd1 =
                    _mm256_cvtepu32_epi64(_mm256_extracti128_si256(dmul1, 1));
                __m256i sd2 =
                    _mm256_cvtepu32_epi64(_mm256_castsi256_si128(dmul2));
                __m256i sd3 =
                    _mm256_cvtepu32_epi64(_mm256_extracti128_si256(dmul2, 1));
                accumdis1 = _mm256_add_epi64(
                    accumdis1,
                    _mm256_mul_epu32(sd0, _mm256_cvtepu16_epi64(l0)));
                accumdis2 = _mm256_add_epi64(
                    accumdis2,
                    _mm256_mul_epu32(
                        sd2, _mm256_cvtepu16_epi64(_mm_bsrli_si128(l0, 8))));
                accumdis3 = _mm256_add_epi64(
                    accumdis3,
                    _mm256_mul_epu32(sd1, _mm256_cvtepu16_epi64(l1)));
                accumdis4 = _mm256_add_epi64(
                    accumdis4,
                    _mm256_mul_epu32(
                        sd3, _mm256_cvtepu16_epi64(_mm_bsrli_si128(l1, 8))));
            }
            accumr_lo = _mm256_add_epi32(accumr_lo, addnum);
            accumr_hi = _mm256_add_epi32(accumr_hi, addnum);
            accumr_lo = _mm256_srli_epi32(accumr_lo, shift_VP);
            accumr_hi = _mm256_srli_epi32(accumr_hi, shift_VP);
            __m256i accumu2_lo =
                _mm256_permute2x128_si256(accumr_lo, accumr_hi, 0x20);
            __m256i accumu2_hi =
                _mm256_permute2x128_si256(accumr_lo, accumr_hi, 0x31);
            _mm256_storeu_si256((__m256i *)(buf.tmp.mu1 + j), accumu2_lo);
            _mm256_storeu_si256((__m256i *)(buf.tmp.mu1 + j + 8), accumu2_hi);

            accumd_lo = _mm256_add_epi32(accumd_lo, addnum);
            accumd_hi = _mm256_add_epi32(accumd_hi, addnum);
            accumd_lo = _mm256_srli_epi32(accumd_lo, shift_VP);
            accumd_hi = _mm256_srli_epi32(accumd_hi, shift_VP);
            __m256i accumu3_lo =
                _mm256_permute2x128_si256(accumd_lo, accumd_hi, 0x20);
            __m256i accumu3_hi =
                _mm256_permute2x128_si256(accumd_lo, accumd_hi, 0x31);
            _mm256_storeu_si256((__m256i *)(buf.tmp.mu2 + j), accumu3_lo);
            _mm256_storeu_si256((__m256i *)(buf.tmp.mu2 + j + 8), accumu3_hi);
            addnum = _mm256_set1_epi64x(add_shift_round_VP_sq);
            accumref1 = _mm256_add_epi64(accumref1, addnum);
            accumref2 = _mm256_add_epi64(accumref2, addnum);
            accumref3 = _mm256_add_epi64(accumref3, addnum);
            accumref4 = _mm256_add_epi64(accumref4, addnum);
            accumref1 = _mm256_srli_epi64(accumref1, shift_VP_sq);
            accumref2 = _mm256_srli_epi64(accumref2, shift_VP_sq);
            accumref3 = _mm256_srli_epi64(accumref3, shift_VP_sq);
            accumref4 = _mm256_srli_epi64(accumref4, shift_VP_sq);

            accumref2 = _mm256_slli_si256(accumref2, 4);
            accumref1 = _mm256_blend_epi32(accumref1, accumref2, 0xAA);
            accumref1 = _mm256_permutevar8x32_epi32(accumref1, mask2);

            _mm256_storeu_si256((__m256i *)(buf.tmp.ref + j), accumref1);
            accumref4 = _mm256_slli_si256(accumref4, 4);
            accumref3 = _mm256_blend_epi32(accumref3, accumref4, 0xAA);
            accumref3 = _mm256_permutevar8x32_epi32(accumref3, mask2);

            _mm256_storeu_si256((__m256i *)(buf.tmp.ref + j + 8), accumref3);

            accumrefdis1 = _mm256_add_epi64(accumrefdis1, addnum);
            accumrefdis2 = _mm256_add_epi64(accumrefdis2, addnum);
            accumrefdis3 = _mm256_add_epi64(accumrefdis3, addnum);
            accumrefdis4 = _mm256_add_epi64(accumrefdis4, addnum);
            accumrefdis1 = _mm256_srli_epi64(accumrefdis1, shift_VP_sq);
            accumrefdis2 = _mm256_srli_epi64(accumrefdis2, shift_VP_sq);
            accumrefdis3 = _mm256_srli_epi64(accumrefdis3, shift_VP_sq);
            accumrefdis4 = _mm256_srli_epi64(accumrefdis4, shift_VP_sq);

            accumrefdis2 = _mm256_slli_si256(accumrefdis2, 4);
            accumrefdis1 = _mm256_blend_epi32(accumrefdis1, accumrefdis2, 0xAA);
            accumrefdis1 = _mm256_permutevar8x32_epi32(accumrefdis1, mask2);

            _mm256_storeu_si256((__m256i *)(buf.tmp.ref_dis + j), accumrefdis1);
            accumrefdis4 = _mm256_slli_si256(accumrefdis4, 4);
            accumrefdis3 = _mm256_blend_epi32(accumrefdis3, accumrefdis4, 0xAA);
            accumrefdis3 = _mm256_permutevar8x32_epi32(accumrefdis3, mask2);

            _mm256_storeu_si256((__m256i *)(buf.tmp.ref_dis + j + 8),
                                accumrefdis3);

            accumdis1 = _mm256_add_epi64(accumdis1, addnum);
            accumdis2 = _mm256_add_epi64(accumdis2, addnum);
            accumdis3 = _mm256_add_epi64(accumdis3, addnum);
            accumdis4 = _mm256_add_epi64(accumdis4, addnum);
            accumdis1 = _mm256_srli_epi64(accumdis1, shift_VP_sq);
            accumdis2 = _mm256_srli_epi64(accumdis2, shift_VP_sq);
            accumdis3 = _mm256_srli_epi64(accumdis3, shift_VP_sq);
            accumdis4 = _mm256_srli_epi64(accumdis4, shift_VP_sq);

            accumdis2 = _mm256_slli_si256(accumdis2, 4);
            accumdis1 = _mm256_blend_epi32(accumdis1, accumdis2, 0xAA);
            accumdis1 = _mm256_permutevar8x32_epi32(accumdis1, mask2);

            _mm256_storeu_si256((__m256i *)(buf.tmp.dis + j), accumdis1);
            accumdis4 = _mm256_slli_si256(accumdis4, 4);
            accumdis3 = _mm256_blend_epi32(accumdis3, accumdis4, 0xAA);
            accumdis3 = _mm256_permutevar8x32_epi32(accumdis3, mask2);

            _mm256_storeu_si256((__m256i *)(buf.tmp.dis + j + 8), accumdis3);
        }

        for (unsigned j = n << 4; j < w; ++j) {
            uint32_t accum_mu1 = 0;
            uint32_t accum_mu2 = 0;
            uint64_t accum_ref = 0;
            uint64_t accum_dis = 0;
            uint64_t accum_ref_dis = 0;
            uint16_t *ref = buf.ref;
            uint16_t *dis = buf.dis;

            int ii_check = ii;
            for (unsigned fi = 0; fi < fwidth; ++fi, ii_check = ii + fi) {
                const uint16_t fcoeff = vif_filt[fi];
                uint16_t imgcoeff_ref = ref[ii_check * stride + j];
                uint16_t imgcoeff_dis = dis[ii_check * stride + j];
                uint32_t img_coeff_ref = fcoeff * (uint32_t)imgcoeff_ref;
                uint32_t img_coeff_dis = fcoeff * (uint32_t)imgcoeff_dis;
                accum_mu1 += img_coeff_ref;
                accum_mu2 += img_coeff_dis;
                accum_ref += img_coeff_ref * (uint64_t)imgcoeff_ref;
                accum_dis += img_coeff_dis * (uint64_t)imgcoeff_dis;
                accum_ref_dis += img_coeff_ref * (uint64_t)imgcoeff_dis;
            }
            buf.tmp.mu1[j] =
                (uint16_t)((accum_mu1 + add_shift_round_VP) >> shift_VP);
            buf.tmp.mu2[j] =
                (uint16_t)((accum_mu2 + add_shift_round_VP) >> shift_VP);
            buf.tmp.ref[j] =
                (uint32_t)((accum_ref + add_shift_round_VP_sq) >> shift_VP_sq);
            buf.tmp.ref_dis[j] = (uint32_t)(
                (accum_ref_dis + add_shift_round_VP_sq) >> shift_VP_sq);
            buf.tmp.dis[j] =
                (uint32_t)((accum_dis + add_shift_round_VP_sq) >> shift_VP_sq);
        }

        PADDING_SQ_DATA(buf, w, fwidth_half);

        //HORIZONTAL
        for (unsigned jj = 0; jj < n << 4; jj += 16) {
            __m256i mu1_lo;
            __m256i mu1_hi;
            __m256i mu1sq_lo;
            __m256i mu1sq_hi;
            __m256i mu2sq_lo;
            __m256i mu2sq_hi;
            __m256i mu1mu2_lo;
            __m256i mu1mu2_hi;
            {
                __m256i fq = _mm256_set1_epi32(vif_filt[fwidth / 2]);
                __m256i acc0 = _mm256_mullo_epi32(_mm256_loadu_si256((__m256i*)(buf.tmp.mu1 + jj + 0)), fq);
                __m256i acc1 = _mm256_mullo_epi32(_mm256_loadu_si256((__m256i*)(buf.tmp.mu1 + jj + 8)), fq);
                for (unsigned fj = 0; fj < fwidth / 2; ++fj) {
                    __m256i fq = _mm256_set1_epi32(vif_filt[fj]);
                    acc0 = _mm256_add_epi32(acc0, _mm256_mullo_epi32(_mm256_loadu_si256((__m256i*)(buf.tmp.mu1 + jj - fwidth / 2 + fj + 0)), fq));
                    acc1 = _mm256_add_epi32(acc1, _mm256_mullo_epi32(_mm256_loadu_si256((__m256i*)(buf.tmp.mu1 + jj - fwidth / 2 + fj + 8)), fq));
                    acc0 = _mm256_add_epi32(acc0, _mm256_mullo_epi32(_mm256_loadu_si256((__m256i*)(buf.tmp.mu1 + jj + fwidth / 2 - fj + 0)), fq));
                    acc1 = _mm256_add_epi32(acc1, _mm256_mullo_epi32(_mm256_loadu_si256((__m256i*)(buf.tmp.mu1 + jj + fwidth / 2 - fj + 8)), fq));
                }

                mu1_lo = acc0;
                mu1_hi = acc1;

                __m256i acc0_lo = _mm256_unpacklo_epi32(acc0, _mm256_setzero_si256());
                __m256i acc0_hi = _mm256_unpackhi_epi32(acc0, _mm256_setzero_si256());
                acc0_lo = _mm256_mul_epu32(acc0_lo, acc0_lo);
                acc0_hi = _mm256_mul_epu32(acc0_hi, acc0_hi);
                acc0_lo = _mm256_srli_epi64(_mm256_add_epi64(acc0_lo, _mm256_set1_epi64x(0x80000000)), 32);
                acc0_hi = _mm256_srli_epi64(_mm256_add_epi64(acc0_hi, _mm256_set1_epi64x(0x80000000)), 32);

                __m256i acc1_lo = _mm256_unpacklo_epi32(acc1, _mm256_setzero_si256());
                __m256i acc1_hi = _mm256_unpackhi_epi32(acc1, _mm256_setzero_si256());
                acc1_lo = _mm256_mul_epu32(acc1_lo, acc1_lo);
                acc1_hi = _mm256_mul_epu32(acc1_hi, acc1_hi);
                acc1_lo = _mm256_srli_epi64(_mm256_add_epi64(acc1_lo, _mm256_set1_epi64x(0x80000000)), 32);
                acc1_hi = _mm256_srli_epi64(_mm256_add_epi64(acc1_hi, _mm256_set1_epi64x(0x80000000)), 32);

                __m256i acc0_sq = _mm256_blend_epi32(acc0_lo, _mm256_slli_si256(acc0_hi, 4), 0xAA);
                acc0_sq = _mm256_shuffle_epi32(acc0_sq, 0xD8);
                __m256i acc1_sq = _mm256_blend_epi32(acc1_lo, _mm256_slli_si256(acc1_hi, 4), 0xAA);
                acc1_sq = _mm256_shuffle_epi32(acc1_sq, 0xD8);
                mu1sq_lo = acc0_sq;
                mu1sq_hi = acc1_sq;
            }

            {
                __m256i fq = _mm256_set1_epi32(vif_filt[fwidth / 2]);
                __m256i acc0 = _mm256_mullo_epi32(_mm256_loadu_si256((__m256i*)(buf.tmp.mu2 + jj + 0)), fq);
                __m256i acc1 = _mm256_mullo_epi32(_mm256_loadu_si256((__m256i*)(buf.tmp.mu2 + jj + 8)), fq);
                for (unsigned fj = 0; fj < fwidth / 2; ++fj) {
                    __m256i fq = _mm256_set1_epi32(vif_filt[fj]);
                    acc0 = _mm256_add_epi32(acc0, _mm256_mullo_epi32(_mm256_loadu_si256((__m256i*)(buf.tmp.mu2 + jj - fwidth / 2 + fj + 0)), fq));
                    acc1 = _mm256_add_epi32(acc1, _mm256_mullo_epi32(_mm256_loadu_si256((__m256i*)(buf.tmp.mu2 + jj - fwidth / 2 + fj + 8)), fq));
                    acc0 = _mm256_add_epi32(acc0, _mm256_mullo_epi32(_mm256_loadu_si256((__m256i*)(buf.tmp.mu2 + jj + fwidth / 2 - fj + 0)), fq));
                    acc1 = _mm256_add_epi32(acc1, _mm256_mullo_epi32(_mm256_loadu_si256((__m256i*)(buf.tmp.mu2 + jj + fwidth / 2 - fj + 8)), fq));
                }

                __m256i acc0_lo = _mm256_unpacklo_epi32(acc0, _mm256_setzero_si256());
                __m256i acc0_hi = _mm256_unpackhi_epi32(acc0, _mm256_setzero_si256());

                __m256i mu1lo_lo = _mm256_unpacklo_epi32(mu1_lo, _mm256_setzero_si256());
                __m256i mu1lo_hi = _mm256_unpackhi_epi32(mu1_lo, _mm256_setzero_si256());
                __m256i mu1hi_lo = _mm256_unpacklo_epi32(mu1_hi, _mm256_setzero_si256());
                __m256i mu1hi_hi = _mm256_unpackhi_epi32(mu1_hi, _mm256_setzero_si256());


                mu1lo_lo = _mm256_mul_epu32(mu1lo_lo, acc0_lo);
                mu1lo_hi = _mm256_mul_epu32(mu1lo_hi, acc0_hi);
                mu1lo_lo = _mm256_srli_epi64(_mm256_add_epi64(mu1lo_lo, _mm256_set1_epi64x(0x80000000)), 32);
                mu1lo_hi = _mm256_srli_epi64(_mm256_add_epi64(mu1lo_hi, _mm256_set1_epi64x(0x80000000)), 32);

                acc0_lo = _mm256_mul_epu32(acc0_lo, acc0_lo);
                acc0_hi = _mm256_mul_epu32(acc0_hi, acc0_hi);
                acc0_lo = _mm256_srli_epi64(_mm256_add_epi64(acc0_lo, _mm256_set1_epi64x(0x80000000)), 32);
                acc0_hi = _mm256_srli_epi64(_mm256_add_epi64(acc0_hi, _mm256_set1_epi64x(0x80000000)), 32);


                __m256i acc1_lo = _mm256_unpacklo_epi32(acc1, _mm256_setzero_si256());
                __m256i acc1_hi = _mm256_unpackhi_epi32(acc1, _mm256_setzero_si256());

                mu1hi_lo = _mm256_mul_epu32(mu1hi_lo, acc1_lo);
                mu1hi_hi = _mm256_mul_epu32(mu1hi_hi, acc1_hi);
                mu1hi_lo = _mm256_srli_epi64(_mm256_add_epi64(mu1hi_lo, _mm256_set1_epi64x(0x80000000)), 32);
                mu1hi_hi = _mm256_srli_epi64(_mm256_add_epi64(mu1hi_hi, _mm256_set1_epi64x(0x80000000)), 32);

                acc1_lo = _mm256_mul_epu32(acc1_lo, acc1_lo);
                acc1_hi = _mm256_mul_epu32(acc1_hi, acc1_hi);
                acc1_lo = _mm256_srli_epi64(_mm256_add_epi64(acc1_lo, _mm256_set1_epi64x(0x80000000)), 32);
                acc1_hi = _mm256_srli_epi64(_mm256_add_epi64(acc1_hi, _mm256_set1_epi64x(0x80000000)), 32);


                __m256i acc0_sq = _mm256_blend_epi32(acc0_lo, _mm256_slli_si256(acc0_hi, 4), 0xAA);
                acc0_sq = _mm256_shuffle_epi32(acc0_sq, 0xD8);
                __m256i acc1_sq = _mm256_blend_epi32(acc1_lo, _mm256_slli_si256(acc1_hi, 4), 0xAA);
                acc1_sq = _mm256_shuffle_epi32(acc1_sq, 0xD8);
                mu2sq_lo = acc0_sq;
                mu2sq_hi = acc1_sq;

                mu1mu2_lo = _mm256_blend_epi32(mu1lo_lo, _mm256_slli_si256(mu1lo_hi, 4), 0xAA);
                mu1mu2_lo = _mm256_shuffle_epi32(mu1mu2_lo, 0xD8);
                mu1mu2_hi = _mm256_blend_epi32(mu1hi_lo, _mm256_slli_si256(mu1hi_hi, 4), 0xAA);
                mu1mu2_hi = _mm256_shuffle_epi32(mu1mu2_hi, 0xD8);


            }

            // filter horizontally ref
            {
                __m256i rounder = _mm256_set1_epi64x(0x8000);
                __m256i fq = _mm256_set1_epi64x(vif_filt[fwidth / 2]);
                __m256i acc0 = _mm256_add_epi64(rounder, _mm256_mul_epu32(_mm256_cvtepu32_epi64(_mm_loadu_si128((__m128i*)(buf.tmp.ref + jj + 0))), fq));
                __m256i acc1 = _mm256_add_epi64(rounder, _mm256_mul_epu32(_mm256_cvtepu32_epi64(_mm_loadu_si128((__m128i*)(buf.tmp.ref + jj + 4))), fq));
                __m256i acc2 = _mm256_add_epi64(rounder, _mm256_mul_epu32(_mm256_cvtepu32_epi64(_mm_loadu_si128((__m128i*)(buf.tmp.ref + jj + 8))), fq));
                __m256i acc3 = _mm256_add_epi64(rounder, _mm256_mul_epu32(_mm256_cvtepu32_epi64(_mm_loadu_si128((__m128i*)(buf.tmp.ref + jj + 12))), fq));
                for (unsigned fj = 0; fj < fwidth / 2; ++fj) {
                    __m256i fq = _mm256_set1_epi64x(vif_filt[fj]);
                    acc0 = _mm256_add_epi64(acc0, _mm256_mul_epu32(_mm256_cvtepu32_epi64(_mm_loadu_si128((__m128i*)(buf.tmp.ref + jj - fwidth / 2 + fj + 0))), fq));
                    acc1 = _mm256_add_epi64(acc1, _mm256_mul_epu32(_mm256_cvtepu32_epi64(_mm_loadu_si128((__m128i*)(buf.tmp.ref + jj - fwidth / 2 + fj + 4))), fq));
                    acc2 = _mm256_add_epi64(acc2, _mm256_mul_epu32(_mm256_cvtepu32_epi64(_mm_loadu_si128((__m128i*)(buf.tmp.ref + jj - fwidth / 2 + fj + 8))), fq));
                    acc3 = _mm256_add_epi64(acc3, _mm256_mul_epu32(_mm256_cvtepu32_epi64(_mm_loadu_si128((__m128i*)(buf.tmp.ref + jj - fwidth / 2 + fj + 12))), fq));
                    acc0 = _mm256_add_epi64(acc0, _mm256_mul_epu32(_mm256_cvtepu32_epi64(_mm_loadu_si128((__m128i*)(buf.tmp.ref + jj + fwidth / 2 - fj + 0))), fq));
                    acc1 = _mm256_add_epi64(acc1, _mm256_mul_epu32(_mm256_cvtepu32_epi64(_mm_loadu_si128((__m128i*)(buf.tmp.ref + jj + fwidth / 2 - fj + 4))), fq));
                    acc2 = _mm256_add_epi64(acc2, _mm256_mul_epu32(_mm256_cvtepu32_epi64(_mm_loadu_si128((__m128i*)(buf.tmp.ref + jj + fwidth / 2 - fj + 8))), fq));
                    acc3 = _mm256_add_epi64(acc3, _mm256_mul_epu32(_mm256_cvtepu32_epi64(_mm_loadu_si128((__m128i*)(buf.tmp.ref + jj + fwidth / 2 - fj + 12))), fq));
                }
                acc0 = _mm256_srli_epi64(acc0, 16);
                acc1 = _mm256_srli_epi64(acc1, 16);
                acc2 = _mm256_srli_epi64(acc2, 16);
                acc3 = _mm256_srli_epi64(acc3, 16);

                __m256i mask1 = _mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0);

                // pack acc0,acc1,acc2,acc3 to acc0, acc1
                acc1 = _mm256_slli_si256(acc1, 4);
                acc1 = _mm256_blend_epi32(acc0, acc1, 0xAA);
                acc0 = _mm256_permutevar8x32_epi32(acc1, mask1);
                acc3 = _mm256_slli_si256(acc3, 4);
                acc3 = _mm256_blend_epi32(acc2, acc3, 0xAA);

                acc1 = _mm256_permutevar8x32_epi32(acc3, mask1);

                acc0 = _mm256_sub_epi32(acc0, mu1sq_lo);
                acc1 = _mm256_sub_epi32(acc1, mu1sq_hi);
                _mm256_storeu_si256((__m256i*) & xx[0], acc0);
                _mm256_storeu_si256((__m256i*) & xx[8], acc1);
            }

            // filter horizontally dis
            {
                __m256i rounder = _mm256_set1_epi64x(0x8000);
                __m256i fq = _mm256_set1_epi64x(vif_filt[fwidth / 2]);
                __m256i acc0 = _mm256_add_epi64(rounder, _mm256_mul_epu32(_mm256_cvtepu32_epi64(_mm_loadu_si128((__m128i*)(buf.tmp.dis + jj + 0))), fq));
                __m256i acc1 = _mm256_add_epi64(rounder, _mm256_mul_epu32(_mm256_cvtepu32_epi64(_mm_loadu_si128((__m128i*)(buf.tmp.dis + jj + 4))), fq));
                __m256i acc2 = _mm256_add_epi64(rounder, _mm256_mul_epu32(_mm256_cvtepu32_epi64(_mm_loadu_si128((__m128i*)(buf.tmp.dis + jj + 8))), fq));
                __m256i acc3 = _mm256_add_epi64(rounder, _mm256_mul_epu32(_mm256_cvtepu32_epi64(_mm_loadu_si128((__m128i*)(buf.tmp.dis + jj + 12))), fq));
                for (unsigned fj = 0; fj < fwidth / 2; ++fj) {
                    __m256i fq = _mm256_set1_epi64x(vif_filt[fj]);
                    acc0 = _mm256_add_epi64(acc0, _mm256_mul_epu32(_mm256_cvtepu32_epi64(_mm_loadu_si128((__m128i*)(buf.tmp.dis + jj - fwidth / 2 + fj + 0))), fq));
                    acc1 = _mm256_add_epi64(acc1, _mm256_mul_epu32(_mm256_cvtepu32_epi64(_mm_loadu_si128((__m128i*)(buf.tmp.dis + jj - fwidth / 2 + fj + 4))), fq));
                    acc2 = _mm256_add_epi64(acc2, _mm256_mul_epu32(_mm256_cvtepu32_epi64(_mm_loadu_si128((__m128i*)(buf.tmp.dis + jj - fwidth / 2 + fj + 8))), fq));
                    acc3 = _mm256_add_epi64(acc3, _mm256_mul_epu32(_mm256_cvtepu32_epi64(_mm_loadu_si128((__m128i*)(buf.tmp.dis + jj - fwidth / 2 + fj + 12))), fq));
                    acc0 = _mm256_add_epi64(acc0, _mm256_mul_epu32(_mm256_cvtepu32_epi64(_mm_loadu_si128((__m128i*)(buf.tmp.dis + jj + fwidth / 2 - fj + 0))), fq));
                    acc1 = _mm256_add_epi64(acc1, _mm256_mul_epu32(_mm256_cvtepu32_epi64(_mm_loadu_si128((__m128i*)(buf.tmp.dis + jj + fwidth / 2 - fj + 4))), fq));
                    acc2 = _mm256_add_epi64(acc2, _mm256_mul_epu32(_mm256_cvtepu32_epi64(_mm_loadu_si128((__m128i*)(buf.tmp.dis + jj + fwidth / 2 - fj + 8))), fq));
                    acc3 = _mm256_add_epi64(acc3, _mm256_mul_epu32(_mm256_cvtepu32_epi64(_mm_loadu_si128((__m128i*)(buf.tmp.dis + jj + fwidth / 2 - fj + 12))), fq));
                }
                acc0 = _mm256_srli_epi64(acc0, 16);
                acc1 = _mm256_srli_epi64(acc1, 16);
                acc2 = _mm256_srli_epi64(acc2, 16);
                acc3 = _mm256_srli_epi64(acc3, 16);

                __m256i mask1 = _mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0);

                // pack acc0,acc1,acc2,acc3 to acc0, acc1
                acc1 = _mm256_slli_si256(acc1, 4);
                acc1 = _mm256_blend_epi32(acc0, acc1, 0xAA);
                acc0 = _mm256_permutevar8x32_epi32(acc1, mask1);
                acc3 = _mm256_slli_si256(acc3, 4);
                acc3 = _mm256_blend_epi32(acc2, acc3, 0xAA);
                acc1 = _mm256_permutevar8x32_epi32(acc3, mask1);

                acc0 = _mm256_sub_epi32(acc0, mu2sq_lo);
                acc1 = _mm256_sub_epi32(acc1, mu2sq_hi);
                _mm256_storeu_si256((__m256i*) & yy[0], _mm256_max_epi32(acc0, _mm256_setzero_si256()));
                _mm256_storeu_si256((__m256i*) & yy[8], _mm256_max_epi32(acc1, _mm256_setzero_si256()));
            }

            // filter horizontally ref_dis, producing 16 samples
            {
                __m256i rounder = _mm256_set1_epi64x(0x8000);
                __m256i fq = _mm256_set1_epi64x(vif_filt[fwidth / 2]);
                __m256i acc0 = _mm256_add_epi64(rounder, _mm256_mul_epu32(_mm256_cvtepu32_epi64(_mm_loadu_si128((__m128i*)(buf.tmp.ref_dis + jj + 0))), fq));
                __m256i acc1 = _mm256_add_epi64(rounder, _mm256_mul_epu32(_mm256_cvtepu32_epi64(_mm_loadu_si128((__m128i*)(buf.tmp.ref_dis + jj + 4))), fq));
                __m256i acc2 = _mm256_add_epi64(rounder, _mm256_mul_epu32(_mm256_cvtepu32_epi64(_mm_loadu_si128((__m128i*)(buf.tmp.ref_dis + jj + 8))), fq));
                __m256i acc3 = _mm256_add_epi64(rounder, _mm256_mul_epu32(_mm256_cvtepu32_epi64(_mm_loadu_si128((__m128i*)(buf.tmp.ref_dis + jj + 12))), fq));

                for (unsigned fj = 0; fj < fwidth / 2; ++fj) {
                    __m256i fq = _mm256_set1_epi64x(vif_filt[fj]);
                    acc0 = _mm256_add_epi64(acc0, _mm256_mul_epu32(_mm256_cvtepu32_epi64(_mm_loadu_si128((__m128i*)(buf.tmp.ref_dis + jj - fwidth / 2 + fj + 0))), fq));
                    acc1 = _mm256_add_epi64(acc1, _mm256_mul_epu32(_mm256_cvtepu32_epi64(_mm_loadu_si128((__m128i*)(buf.tmp.ref_dis + jj - fwidth / 2 + fj + 4))), fq));
                    acc2 = _mm256_add_epi64(acc2, _mm256_mul_epu32(_mm256_cvtepu32_epi64(_mm_loadu_si128((__m128i*)(buf.tmp.ref_dis + jj - fwidth / 2 + fj + 8))), fq));
                    acc3 = _mm256_add_epi64(acc3, _mm256_mul_epu32(_mm256_cvtepu32_epi64(_mm_loadu_si128((__m128i*)(buf.tmp.ref_dis + jj - fwidth / 2 + fj + 12))), fq));
                    acc0 = _mm256_add_epi64(acc0, _mm256_mul_epu32(_mm256_cvtepu32_epi64(_mm_loadu_si128((__m128i*)(buf.tmp.ref_dis + jj + fwidth / 2 - fj + 0))), fq));
                    acc1 = _mm256_add_epi64(acc1, _mm256_mul_epu32(_mm256_cvtepu32_epi64(_mm_loadu_si128((__m128i*)(buf.tmp.ref_dis + jj + fwidth / 2 - fj + 4))), fq));
                    acc2 = _mm256_add_epi64(acc2, _mm256_mul_epu32(_mm256_cvtepu32_epi64(_mm_loadu_si128((__m128i*)(buf.tmp.ref_dis + jj + fwidth / 2 - fj + 8))), fq));
                    acc3 = _mm256_add_epi64(acc3, _mm256_mul_epu32(_mm256_cvtepu32_epi64(_mm_loadu_si128((__m128i*)(buf.tmp.ref_dis + jj + fwidth / 2 - fj + 12))), fq));
                }

                acc0 = _mm256_srli_epi64(acc0, 16);
                acc1 = _mm256_srli_epi64(acc1, 16);
                acc2 = _mm256_srli_epi64(acc2, 16);
                acc3 = _mm256_srli_epi64(acc3, 16);

                __m256i mask1 = _mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0);

                // pack acc0,acc1,acc2,acc3 to acc0, acc1
                acc1 = _mm256_slli_si256(acc1, 4);
                acc1 = _mm256_blend_epi32(acc0, acc1, 0xAA);
                acc0 = _mm256_permutevar8x32_epi32(acc1, mask1);
                acc3 = _mm256_slli_si256(acc3, 4);
                acc3 = _mm256_blend_epi32(acc2, acc3, 0xAA);
                acc1 = _mm256_permutevar8x32_epi32(acc3, mask1);

                acc0 = _mm256_sub_epi32(acc0, mu1mu2_lo);
                acc1 = _mm256_sub_epi32(acc1, mu1mu2_hi);
                _mm256_storeu_si256((__m256i*) & xy[0], acc0);
                _mm256_storeu_si256((__m256i*) & xy[8], acc1);
            }

            for (unsigned int j = 0; j < 16; j++) {
                int32_t sigma1_sq = xx[j];
                int32_t sigma2_sq = yy[j];
                int32_t sigma12 = xy[j];

                if (sigma1_sq >= sigma_nsq) {
                    /**
                    * log values are taken from the look-up table generated by
                    * log_generate() function which is called in integer_combo_threadfunc
                    * den_val in float is log2(1 + sigma1_sq/2)
                    * here it is converted to equivalent of log2(2+sigma1_sq) - log2(2) i.e log2(2*65536+sigma1_sq) - 17
                    * multiplied by 2048 as log_value = log2(i)*2048 i=16384 to 65535 generated using log_value
                    * x because best 16 bits are taken
                    */
                    accum_den_log += log2_32(log2_table, sigma_nsq + sigma1_sq) - 2048 * 17;

                    if (sigma12 > 0 && sigma2_sq > 0) {
                        /**
                        * In floating-point numerator = log2((1.0f + (g * g * sigma1_sq)/(sv_sq + sigma_nsq))
                        *
                        * In Fixed-point the above is converted to
                        * numerator = log2((sv_sq + sigma_nsq)+(g * g * sigma1_sq))- log2(sv_sq + sigma_nsq)
                        */

                        const double eps = 65536 * 1.0e-10;
                        double g = sigma12 / (sigma1_sq + eps); // this epsilon can go away
                        int32_t sv_sq = sigma2_sq - g * sigma12;

                        sv_sq = (uint32_t)(MAX(sv_sq, 0));

                        g = MIN(g, vif_enhn_gain_limit);

                        uint32_t numer1 = (sv_sq + sigma_nsq);
                        int64_t numer1_tmp = (int64_t)((g * g * sigma1_sq)) + numer1; //numerator
                        accum_num_log += log2_64(log2_table, numer1_tmp) - log2_64(log2_table, numer1);
                    }
                }
                else {
                    accum_num_non_log += sigma2_sq;
                    accum_den_non_log++;
                }
            }
        }


        if ((n << 4) != w) {
            VifResiduals residuals =
                vif_compute_line_residuals(s, n << 4, w, scale);
            accum_num_log += residuals.accum_num_log;
            accum_den_log += residuals.accum_den_log;
            accum_num_non_log += residuals.accum_num_non_log;
            accum_den_non_log += residuals.accum_den_non_log;
        }
    }

    num[0] = accum_num_log / 2048.0 + (accum_den_non_log - ((accum_num_non_log) / 16384.0) / (65025.0));
    den[0] = accum_den_log / 2048.0 + accum_den_non_log;
}


void vif_subsample_rd_8_avx2(VifBuffer buf, unsigned w, unsigned h) {
    const unsigned fwidth = vif_filter1d_width[1];
    const uint16_t *vif_filt_s1 = vif_filter1d_table[1];
    const uint8_t *ref = (uint8_t *)buf.ref;
    const uint8_t *dis = (uint8_t *)buf.dis;
    const ptrdiff_t stride = buf.stride_16 / sizeof(uint16_t);
    __m256i addnum = _mm256_set1_epi32(32768);
    __m256i mask1 = _mm256_set_epi32(6, 4, 2, 0, 6, 4, 2, 0);
    __m256i x = _mm256_set1_epi32(128);
    int fwidth_half = fwidth >> 1;

    __m256i fcoeff0 = _mm256_set1_epi16(vif_filt_s1[0]);
    __m256i fcoeff1 = _mm256_set1_epi16(vif_filt_s1[1]);
    __m256i fcoeff2 = _mm256_set1_epi16(vif_filt_s1[2]);
    __m256i fcoeff3 = _mm256_set1_epi16(vif_filt_s1[3]);
    __m256i fcoeff4 = _mm256_set1_epi16(vif_filt_s1[4]);

    for (unsigned i = 0; i < h / 2; i ++) {
        // VERTICAL
        unsigned n = w >> 4;
        for (unsigned j = 0; j < n << 4; j = j + 16) {
            int ii = i * 2 - fwidth_half;
            int ii_check = ii;
            __m256i accum_mu1_lo, accum_mu1_hi;
            __m256i accum_mu2_lo, accum_mu2_hi;
            __m256i g0, g1, g2, g3, g4, g5, g6, g7, g8;
            __m256i s0, s1, s2, s3, s4, s5, s6, s7, s8;

            g0 = _mm256_cvtepu8_epi16(_mm_loadu_si128(
                (__m128i *)(ref + (buf.stride * ii_check) + j)));
            g1 = _mm256_cvtepu8_epi16(_mm_loadu_si128(
                (__m128i *)(ref + buf.stride * (ii_check + 1) + j)));
            g2 = _mm256_cvtepu8_epi16(_mm_loadu_si128(
                (__m128i *)(ref + buf.stride * (ii_check + 2) + j)));
            g3 = _mm256_cvtepu8_epi16(_mm_loadu_si128(
                (__m128i *)(ref + buf.stride * (ii_check + 3) + j)));
            g4 = _mm256_cvtepu8_epi16(_mm_loadu_si128(
                (__m128i *)(ref + buf.stride * (ii_check + 4) + j)));
            g5 = _mm256_cvtepu8_epi16(_mm_loadu_si128(
                (__m128i *)(ref + buf.stride * (ii_check + 5) + j)));
            g6 = _mm256_cvtepu8_epi16(_mm_loadu_si128(
                (__m128i *)(ref + buf.stride * (ii_check + 6) + j)));
            g7 = _mm256_cvtepu8_epi16(_mm_loadu_si128(
                (__m128i *)(ref + buf.stride * (ii_check + 7) + j)));
            g8 = _mm256_cvtepu8_epi16(_mm_loadu_si128(
                (__m128i *)(ref + buf.stride * (ii_check + 8) + j)));

            s0 = _mm256_cvtepu8_epi16(_mm_loadu_si128(
                (__m128i *)(dis + (buf.stride * ii_check) + j)));
            s1 = _mm256_cvtepu8_epi16(_mm_loadu_si128(
                (__m128i *)(dis + buf.stride * (ii_check + 1) + j)));
            s2 = _mm256_cvtepu8_epi16(_mm_loadu_si128(
                (__m128i *)(dis + buf.stride * (ii_check + 2) + j)));
            s3 = _mm256_cvtepu8_epi16(_mm_loadu_si128(
                (__m128i *)(dis + buf.stride * (ii_check + 3) + j)));
            s4 = _mm256_cvtepu8_epi16(_mm_loadu_si128(
                (__m128i *)(dis + buf.stride * (ii_check + 4) + j)));
            s5 = _mm256_cvtepu8_epi16(_mm_loadu_si128(
                (__m128i *)(dis + buf.stride * (ii_check + 5) + j)));
            s6 = _mm256_cvtepu8_epi16(_mm_loadu_si128(
                (__m128i *)(dis + buf.stride * (ii_check + 6) + j)));
            s7 = _mm256_cvtepu8_epi16(_mm_loadu_si128(
                (__m128i *)(dis + buf.stride * (ii_check + 7) + j)));
            s8 = _mm256_cvtepu8_epi16(_mm_loadu_si128(
                (__m128i *)(dis + buf.stride * (ii_check + 8) + j)));

            multiply2(accum_mu2_lo, accum_mu2_hi, s4, fcoeff4);
            multiply2_and_accumulate(accum_mu2_lo, accum_mu2_hi, s0, s8, fcoeff0);
            multiply2_and_accumulate(accum_mu2_lo, accum_mu2_hi, s1, s7, fcoeff1);
            multiply2_and_accumulate(accum_mu2_lo, accum_mu2_hi, s2, s6, fcoeff2);
            multiply2_and_accumulate(accum_mu2_lo, accum_mu2_hi, s3, s5, fcoeff3);

            multiply2(accum_mu1_lo, accum_mu1_hi, g4, fcoeff4);
            multiply2_and_accumulate(accum_mu1_lo, accum_mu1_hi, g0, g8, fcoeff0);
            multiply2_and_accumulate(accum_mu1_lo, accum_mu1_hi, g1, g7, fcoeff1);
            multiply2_and_accumulate(accum_mu1_lo, accum_mu1_hi, g2, g6, fcoeff2);
            multiply2_and_accumulate(accum_mu1_lo, accum_mu1_hi, g3, g5, fcoeff3);

            __m256i accumu1_lo = _mm256_add_epi32(
                x, _mm256_permute2x128_si256(accum_mu1_lo, accum_mu1_hi, 0x20));
            __m256i accumu1_hi = _mm256_add_epi32(
                x, _mm256_permute2x128_si256(accum_mu1_lo, accum_mu1_hi, 0x31));
            __m256i accumu2_lo = _mm256_add_epi32(
                x, _mm256_permute2x128_si256(accum_mu2_lo, accum_mu2_hi, 0x20));
            __m256i accumu2_hi = _mm256_add_epi32(
                x, _mm256_permute2x128_si256(accum_mu2_lo, accum_mu2_hi, 0x31));
            accumu1_lo = _mm256_srli_epi32(accumu1_lo, 0x08);
            accumu1_hi = _mm256_srli_epi32(accumu1_hi, 0x08);
            accumu2_lo = _mm256_srli_epi32(accumu2_lo, 0x08);
            accumu2_hi = _mm256_srli_epi32(accumu2_hi, 0x08);
            _mm256_storeu_si256((__m256i *)(buf.tmp.ref_convol + j),
                                accumu1_lo);
            _mm256_storeu_si256((__m256i *)(buf.tmp.ref_convol + j + 8),
                                accumu1_hi);
            _mm256_storeu_si256((__m256i *)(buf.tmp.dis_convol + j),
                                accumu2_lo);
            _mm256_storeu_si256((__m256i *)(buf.tmp.dis_convol + j + 8),
                                accumu2_hi);
        }
        for (unsigned j = n << 4; j < w; ++j) {
            uint32_t accum_ref = 0;
            uint32_t accum_dis = 0;
            for (unsigned fi = 0; fi < fwidth; ++fi) {
                int ii = i * 2 - fwidth_half;
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

        PADDING_SQ_DATA_2(buf, w, fwidth_half);

        // HORIZONTAL
        n = w >> 3;
        for (unsigned j = 0; j < n << 3; j = j + 8) {
            int jj = j - fwidth_half;
            int jj_check = jj;
            __m256i accumrlo, accumdlo, accumrhi, accumdhi;
            accumrlo = accumdlo = accumrhi = accumdhi = _mm256_setzero_si256();
            __m256i refconvol0 = _mm256_loadu_si256((__m256i *)(buf.tmp.ref_convol + jj_check));
            __m256i refconvol4 = _mm256_loadu_si256((__m256i*)(buf.tmp.ref_convol + jj_check + 4));
            __m256i refconvol8 = _mm256_loadu_si256((__m256i*)(buf.tmp.ref_convol + jj_check + 8));
            __m256i refconvol1 = _mm256_alignr_epi8(refconvol4, refconvol0, 4);
            __m256i refconvol2 = _mm256_alignr_epi8(refconvol4, refconvol0, 8);
            __m256i refconvol3 = _mm256_alignr_epi8(refconvol4, refconvol0, 12);
            __m256i refconvol5 = _mm256_alignr_epi8(refconvol8, refconvol4, 4);
            __m256i refconvol6 = _mm256_alignr_epi8(refconvol8, refconvol4, 8);
            __m256i refconvol7 = _mm256_alignr_epi8(refconvol8, refconvol4, 12);

            __m256i result2 = _mm256_mulhi_epu16(refconvol0, fcoeff0);
            __m256i result2lo = _mm256_mullo_epi16(refconvol0, fcoeff0);
            accumrlo = _mm256_add_epi32(
                accumrlo, _mm256_unpacklo_epi16(result2lo, result2));
            accumrhi = _mm256_add_epi32(
                accumrhi, _mm256_unpackhi_epi16(result2lo, result2));
            __m256i result3 = _mm256_mulhi_epu16(refconvol1, fcoeff1);
            __m256i result3lo = _mm256_mullo_epi16(refconvol1, fcoeff1);
            accumrlo = _mm256_add_epi32(
                accumrlo, _mm256_unpacklo_epi16(result3lo, result3));
            accumrhi = _mm256_add_epi32(
                accumrhi, _mm256_unpackhi_epi16(result3lo, result3));
            __m256i result4 = _mm256_mulhi_epu16(refconvol2, fcoeff2);
            __m256i result4lo = _mm256_mullo_epi16(refconvol2, fcoeff2);
            accumrlo = _mm256_add_epi32(
                accumrlo, _mm256_unpacklo_epi16(result4lo, result4));
            accumrhi = _mm256_add_epi32(
                accumrhi, _mm256_unpackhi_epi16(result4lo, result4));
            __m256i result5 = _mm256_mulhi_epu16(refconvol3, fcoeff3);
            __m256i result5lo = _mm256_mullo_epi16(refconvol3, fcoeff3);
            accumrlo = _mm256_add_epi32(
                accumrlo, _mm256_unpacklo_epi16(result5lo, result5));
            accumrhi = _mm256_add_epi32(
                accumrhi, _mm256_unpackhi_epi16(result5lo, result5));
            __m256i result6 = _mm256_mulhi_epu16(refconvol4, fcoeff4);
            __m256i result6lo = _mm256_mullo_epi16(refconvol4, fcoeff4);
            accumrlo = _mm256_add_epi32(
                accumrlo, _mm256_unpacklo_epi16(result6lo, result6));
            accumrhi = _mm256_add_epi32(
                accumrhi, _mm256_unpackhi_epi16(result6lo, result6));
            __m256i result7 = _mm256_mulhi_epu16(refconvol5, fcoeff3);
            __m256i result7lo = _mm256_mullo_epi16(refconvol5, fcoeff3);
            accumrlo = _mm256_add_epi32(
                accumrlo, _mm256_unpacklo_epi16(result7lo, result7));
            accumrhi = _mm256_add_epi32(
                accumrhi, _mm256_unpackhi_epi16(result7lo, result7));
            __m256i result8 = _mm256_mulhi_epu16(refconvol6, fcoeff2);
            __m256i result8lo = _mm256_mullo_epi16(refconvol6, fcoeff2);
            accumrlo = _mm256_add_epi32(
                accumrlo, _mm256_unpacklo_epi16(result8lo, result8));
            accumrhi = _mm256_add_epi32(
                accumrhi, _mm256_unpackhi_epi16(result8lo, result8));
            __m256i result9 = _mm256_mulhi_epu16(refconvol7, fcoeff1);
            __m256i result9lo = _mm256_mullo_epi16(refconvol7, fcoeff1);
            accumrlo = _mm256_add_epi32(
                accumrlo, _mm256_unpacklo_epi16(result9lo, result9));
            accumrhi = _mm256_add_epi32(
                accumrhi, _mm256_unpackhi_epi16(result9lo, result9));
            __m256i result10 = _mm256_mulhi_epu16(refconvol8, fcoeff0);
            __m256i result10lo = _mm256_mullo_epi16(refconvol8, fcoeff0);
            accumrlo = _mm256_add_epi32(
                accumrlo, _mm256_unpacklo_epi16(result10lo, result10));
            accumrhi = _mm256_add_epi32(
                accumrhi, _mm256_unpackhi_epi16(result10lo, result10));

            __m256i disconvol0 =_mm256_loadu_si256((__m256i *)(buf.tmp.dis_convol + jj_check));
            __m256i disconvol4 = _mm256_loadu_si256((__m256i*)(buf.tmp.dis_convol + jj_check + 4));
            __m256i disconvol8 = _mm256_loadu_si256((__m256i*)(buf.tmp.dis_convol + jj_check + 8));
            __m256i disconvol1 = _mm256_alignr_epi8(disconvol4, disconvol0, 4);
            __m256i disconvol2 = _mm256_alignr_epi8(disconvol4, disconvol0, 8);
            __m256i disconvol3 = _mm256_alignr_epi8(disconvol4, disconvol0, 12);
            __m256i disconvol5 = _mm256_alignr_epi8(disconvol8, disconvol4, 4);
            __m256i disconvol6 = _mm256_alignr_epi8(disconvol8, disconvol4, 8);
            __m256i disconvol7 = _mm256_alignr_epi8(disconvol8, disconvol4, 12);
            result2 = _mm256_mulhi_epu16(disconvol0, fcoeff0);
            result2lo = _mm256_mullo_epi16(disconvol0, fcoeff0);
            accumdlo = _mm256_add_epi32(
                accumdlo, _mm256_unpacklo_epi16(result2lo, result2));
            accumdhi = _mm256_add_epi32(
                accumdhi, _mm256_unpackhi_epi16(result2lo, result2));
            result3 = _mm256_mulhi_epu16(disconvol1, fcoeff1);
            result3lo = _mm256_mullo_epi16(disconvol1, fcoeff1);
            accumdlo = _mm256_add_epi32(
                accumdlo, _mm256_unpacklo_epi16(result3lo, result3));
            accumdhi = _mm256_add_epi32(
                accumdhi, _mm256_unpackhi_epi16(result3lo, result3));
            result4 = _mm256_mulhi_epu16(disconvol2, fcoeff2);
            result4lo = _mm256_mullo_epi16(disconvol2, fcoeff2);
            accumdlo = _mm256_add_epi32(
                accumdlo, _mm256_unpacklo_epi16(result4lo, result4));
            accumdhi = _mm256_add_epi32(
                accumdhi, _mm256_unpackhi_epi16(result4lo, result4));
            result5 = _mm256_mulhi_epu16(disconvol3, fcoeff3);
            result5lo = _mm256_mullo_epi16(disconvol3, fcoeff3);
            accumdlo = _mm256_add_epi32(
                accumdlo, _mm256_unpacklo_epi16(result5lo, result5));
            accumdhi = _mm256_add_epi32(
                accumdhi, _mm256_unpackhi_epi16(result5lo, result5));
            result6 = _mm256_mulhi_epu16(disconvol4, fcoeff4);
            result6lo = _mm256_mullo_epi16(disconvol4, fcoeff4);
            accumdlo = _mm256_add_epi32(
                accumdlo, _mm256_unpacklo_epi16(result6lo, result6));
            accumdhi = _mm256_add_epi32(
                accumdhi, _mm256_unpackhi_epi16(result6lo, result6));
            result7 = _mm256_mulhi_epu16(disconvol5, fcoeff3);
            result7lo = _mm256_mullo_epi16(disconvol5, fcoeff3);
            accumdlo = _mm256_add_epi32(
                accumdlo, _mm256_unpacklo_epi16(result7lo, result7));
            accumdhi = _mm256_add_epi32(
                accumdhi, _mm256_unpackhi_epi16(result7lo, result7));
            result8 = _mm256_mulhi_epu16(disconvol6, fcoeff2);
            result8lo = _mm256_mullo_epi16(disconvol6, fcoeff2);
            accumdlo = _mm256_add_epi32(
                accumdlo, _mm256_unpacklo_epi16(result8lo, result8));
            accumdhi = _mm256_add_epi32(
                accumdhi, _mm256_unpackhi_epi16(result8lo, result8));
            result9 = _mm256_mulhi_epu16(disconvol7, fcoeff1);
            result9lo = _mm256_mullo_epi16(disconvol7, fcoeff1);
            accumdlo = _mm256_add_epi32(
                accumdlo, _mm256_unpacklo_epi16(result9lo, result9));
            accumdhi = _mm256_add_epi32(
                accumdhi, _mm256_unpackhi_epi16(result9lo, result9));
            result10 = _mm256_mulhi_epu16(disconvol8, fcoeff0);
            result10lo = _mm256_mullo_epi16(disconvol8, fcoeff0);
            accumdlo = _mm256_add_epi32(
                accumdlo, _mm256_unpacklo_epi16(result10lo, result10));
            accumdhi = _mm256_add_epi32(
                accumdhi, _mm256_unpackhi_epi16(result10lo, result10));

            accumdlo = _mm256_add_epi32(accumdlo, addnum);
            accumdhi = _mm256_add_epi32(accumdhi, addnum);
            accumrlo = _mm256_add_epi32(accumrlo, addnum);
            accumrhi = _mm256_add_epi32(accumrhi, addnum);
            accumdlo = _mm256_srli_epi32(accumdlo, 0x10);
            accumdhi = _mm256_srli_epi32(accumdhi, 0x10);
            accumrlo = _mm256_srli_epi32(accumrlo, 0x10);
            accumrhi = _mm256_srli_epi32(accumrhi, 0x10);

            __m256i result = _mm256_packus_epi32(accumdlo, accumdhi);
            __m256i resultd = _mm256_packus_epi32(accumrlo, accumrhi);
            resultd = _mm256_permutevar8x32_epi32(resultd, mask1);
            result = _mm256_permutevar8x32_epi32(result, mask1);
            resultd = _mm256_packus_epi32(resultd, resultd);
            result = _mm256_packus_epi32(result, result);
            _mm_storel_epi64((__m128i *)(buf.mu1 + i  * stride + (j >> 1)), _mm256_castsi256_si128(resultd));
            _mm_storel_epi64((__m128i *)(buf.mu2 + i  * stride + (j >> 1)), _mm256_castsi256_si128(result));
        }
        for (unsigned j = n << 3; j < w; j += 2) {
            uint32_t accum_ref = 0;
            uint32_t accum_dis = 0;
            int jj = j - fwidth_half;
            int jj_check = jj;
            for (unsigned fj = 0; fj < fwidth; ++fj, jj_check = jj + fj) {
                const uint16_t fcoeff = vif_filt_s1[fj];
                accum_ref += fcoeff * buf.tmp.ref_convol[jj_check];
                accum_dis += fcoeff * buf.tmp.dis_convol[jj_check];
            }
            buf.mu1[i * stride + (j >> 1)] = (uint16_t)((accum_ref + 32768) >> 16);
            buf.mu2[i * stride + (j >> 1)] = (uint16_t)((accum_dis + 32768) >> 16);
        }
    }
    copy_and_pad(buf, w, h, 0);
}

void vif_subsample_rd_16_avx2(VifBuffer buf, unsigned w, unsigned h, int scale,
                             int bpc) {
    const unsigned fwidth = vif_filter1d_width[scale + 1];
    const uint16_t *vif_filt = vif_filter1d_table[scale + 1];
    int32_t add_shift_round_VP, shift_VP;
    int fwidth_half = fwidth >> 1;
    const ptrdiff_t stride = buf.stride / sizeof(uint16_t);
    const ptrdiff_t stride16 = buf.stride_16 / sizeof(uint16_t);
    uint16_t *ref = buf.ref;
    uint16_t *dis = buf.dis;
    __m256i mask1 = _mm256_set_epi32(6, 4, 2, 0, 6, 4, 2, 0);

    if (scale == 0) {
        add_shift_round_VP = 1 << (bpc - 1);
        shift_VP = bpc;
    } else {
        add_shift_round_VP = 32768;
        shift_VP = 16;
    }

    /* Pre-broadcast all filter coefficients (max fwidth = 9 for subsample) */
    __m256i vif_coeffs[9];
    for (unsigned fi = 0; fi < fwidth; ++fi) {
        vif_coeffs[fi] = _mm256_set1_epi16(vif_filt[fi]);
    }
    __m256i addnum_vp = _mm256_set1_epi32(add_shift_round_VP);
    __m256i addnum_hp = _mm256_set1_epi32(32768);

    for (unsigned i = 0; i < h / 2; i++) {
        // VERTICAL

        unsigned n = w >> 4;
        int ii = i * 2 - fwidth_half;
        for (unsigned j = 0; j < n << 4; j = j + 16) {
            int ii_check = ii;
            __m256i accumr_lo, accumr_hi, accumd_lo, accumd_hi, rmul1, rmul2,
                dmul1, dmul2;
            accumr_lo = accumr_hi = accumd_lo = accumd_hi = rmul1 = rmul2 =
                dmul1 = dmul2 = _mm256_setzero_si256();
            for (unsigned fi = 0; fi < fwidth; ++fi, ii_check = ii + fi) {
                __m256i f1 = vif_coeffs[fi];
                __m256i ref1 = _mm256_loadu_si256(
                    (__m256i *)(ref + (ii_check * stride) + j));
                __m256i dis1 = _mm256_loadu_si256(
                    (__m256i *)(dis + (ii_check * stride) + j));
                __m256i result2 = _mm256_mulhi_epu16(ref1, f1);
                __m256i result2lo = _mm256_mullo_epi16(ref1, f1);
                rmul1 = _mm256_unpacklo_epi16(result2lo, result2);
                rmul2 = _mm256_unpackhi_epi16(result2lo, result2);
                accumr_lo = _mm256_add_epi32(accumr_lo, rmul1);
                accumr_hi = _mm256_add_epi32(accumr_hi, rmul2);

                __m256i d0 = _mm256_mulhi_epu16(dis1, f1);
                __m256i d0lo = _mm256_mullo_epi16(dis1, f1);
                dmul1 = _mm256_unpacklo_epi16(d0lo, d0);
                dmul2 = _mm256_unpackhi_epi16(d0lo, d0);
                accumd_lo = _mm256_add_epi32(accumd_lo, dmul1);
                accumd_hi = _mm256_add_epi32(accumd_hi, dmul2);
            }
            accumr_lo = _mm256_add_epi32(accumr_lo, addnum_vp);
            accumr_hi = _mm256_add_epi32(accumr_hi, addnum_vp);
            accumr_lo = _mm256_srli_epi32(accumr_lo, shift_VP);
            accumr_hi = _mm256_srli_epi32(accumr_hi, shift_VP);

            __m256i accumu2_lo =
                _mm256_permute2x128_si256(accumr_lo, accumr_hi, 0x20);
            __m256i accumu2_hi =
                _mm256_permute2x128_si256(accumr_lo, accumr_hi, 0x31);
            _mm256_storeu_si256((__m256i *)(buf.tmp.ref_convol + j),
                                accumu2_lo);
            _mm256_storeu_si256((__m256i *)(buf.tmp.ref_convol + j + 8),
                                accumu2_hi);

            accumd_lo = _mm256_add_epi32(accumd_lo, addnum_vp);
            accumd_hi = _mm256_add_epi32(accumd_hi, addnum_vp);
            accumd_lo = _mm256_srli_epi32(accumd_lo, shift_VP);
            accumd_hi = _mm256_srli_epi32(accumd_hi, shift_VP);
            accumu2_lo = _mm256_permute2x128_si256(accumd_lo, accumd_hi, 0x20);
            accumu2_hi = _mm256_permute2x128_si256(accumd_lo, accumd_hi, 0x31);
            _mm256_storeu_si256((__m256i *)(buf.tmp.dis_convol + j),
                                accumu2_lo);
            _mm256_storeu_si256((__m256i *)(buf.tmp.dis_convol + j + 8),
                                accumu2_hi);
        }

        // VERTICAL
        for (unsigned j = n << 4; j < w; ++j) {
            uint32_t accum_ref = 0;
            uint32_t accum_dis = 0;
            int ii_check = ii;
            for (unsigned fi = 0; fi < fwidth; ++fi, ii_check = ii + fi) {
                const uint16_t fcoeff = vif_filt[fi];
                accum_ref += fcoeff * ((uint32_t)ref[ii_check * stride + j]);
                accum_dis += fcoeff * ((uint32_t)dis[ii_check * stride + j]);
            }
            buf.tmp.ref_convol[j] =
                (uint16_t)((accum_ref + add_shift_round_VP) >> shift_VP);
            buf.tmp.dis_convol[j] =
                (uint16_t)((accum_dis + add_shift_round_VP) >> shift_VP);
        }

        PADDING_SQ_DATA_2(buf, w, fwidth_half);

        // HORIZONTAL
        n = w >> 3;
        for (unsigned j = 0; j < n << 3; j = j + 8) {
            int jj = j - fwidth_half;
            int jj_check = jj;
            __m256i accumrlo, accumdlo, accumrhi, accumdhi;
            accumrlo = accumdlo = accumrhi = accumdhi = _mm256_setzero_si256();
            for (unsigned fj = 0; fj < fwidth; ++fj, jj_check = jj + fj) {
                __m256i refconvol = _mm256_loadu_si256(
                    (__m256i *)(buf.tmp.ref_convol + jj_check));
                __m256i fcoeff = vif_coeffs[fj];
                __m256i result2 = _mm256_mulhi_epu16(refconvol, fcoeff);
                __m256i result2lo = _mm256_mullo_epi16(refconvol, fcoeff);
                accumrlo = _mm256_add_epi32(
                    accumrlo, _mm256_unpacklo_epi16(result2lo, result2));
                accumrhi = _mm256_add_epi32(
                    accumrhi, _mm256_unpackhi_epi16(result2lo, result2));
                __m256i disconvol = _mm256_loadu_si256(
                    (__m256i *)(buf.tmp.dis_convol + jj_check));
                result2 = _mm256_mulhi_epu16(disconvol, fcoeff);
                result2lo = _mm256_mullo_epi16(disconvol, fcoeff);
                accumdlo = _mm256_add_epi32(
                    accumdlo, _mm256_unpacklo_epi16(result2lo, result2));
                accumdhi = _mm256_add_epi32(
                    accumdhi, _mm256_unpackhi_epi16(result2lo, result2));
            }

            accumdlo = _mm256_add_epi32(accumdlo, addnum_hp);
            accumdhi = _mm256_add_epi32(accumdhi, addnum_hp);
            accumrlo = _mm256_add_epi32(accumrlo, addnum_hp);
            accumrhi = _mm256_add_epi32(accumrhi, addnum_hp);
            accumdlo = _mm256_srli_epi32(accumdlo, 0x10);
            accumdhi = _mm256_srli_epi32(accumdhi, 0x10);
            accumrlo = _mm256_srli_epi32(accumrlo, 0x10);
            accumrhi = _mm256_srli_epi32(accumrhi, 0x10);

            __m256i result = _mm256_packus_epi32(accumdlo, accumdhi);
            __m256i resultd = _mm256_packus_epi32(accumrlo, accumrhi);
            __m256i resulttmp = _mm256_srli_si256(resultd, 2);
            resultd = _mm256_blend_epi16(resultd, resulttmp, 0xAA);
            resultd = _mm256_permutevar8x32_epi32(resultd, mask1);
            _mm_storeu_si128((__m128i *)(buf.mu1 + i * stride16 + j),
                             _mm256_castsi256_si128(resultd));

            resulttmp = _mm256_srli_si256(result, 2);
            result = _mm256_blend_epi16(result, resulttmp, 0xAA);
            result = _mm256_permutevar8x32_epi32(result, mask1);
            _mm_storeu_si128((__m128i *)(buf.mu2 + i * stride16 + j),
                             _mm256_castsi256_si128(result));
        }

        for (unsigned j = n << 3; j < w; ++j) {
            uint32_t accum_ref = 0;
            uint32_t accum_dis = 0;
            int jj = j - fwidth_half;
            int jj_check = jj;
            for (unsigned fj = 0; fj < fwidth; ++fj, jj_check = jj + fj) {
                const uint16_t fcoeff = vif_filt[fj];
                accum_ref += fcoeff * ((uint32_t)buf.tmp.ref_convol[jj_check]);
                accum_dis += fcoeff * ((uint32_t)buf.tmp.dis_convol[jj_check]);
            }
            buf.mu1[i * stride16 + j] = (uint16_t)((accum_ref + 32768) >> 16);
            buf.mu2[i * stride16 + j] = (uint16_t)((accum_dis + 32768) >> 16);
        }
    }

    ref = buf.ref;
    dis = buf.dis;

    for (unsigned i = 0; i < h / 2; ++i) {
        for (unsigned j = 0; j < w / 2; ++j) {
            ref[i * stride + j] = buf.mu1[i * stride16 + (j * 2)];
            dis[i * stride + j] = buf.mu2[i * stride16 + (j * 2)];
        }
    }
    pad_top_and_bottom(buf, h / 2, vif_filter1d_width[scale]);
}
