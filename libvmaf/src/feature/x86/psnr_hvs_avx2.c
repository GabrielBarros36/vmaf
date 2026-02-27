/*
Copyright 2001-2012 Xiph.Org and contributors.
AVX2 implementation Copyright 2024 Netflix, Inc.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

- Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

- Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <immintrin.h>
#include <stdint.h>

/*
 * AVX2 vectorized 8x8 DCT for PSNR-HVS.
 *
 * Processes all 8 elements simultaneously using AVX2 int32 operations.
 * Each __m256i register holds one "row" of 8 int32 values, where each lane
 * corresponds to a different column.  The butterfly is applied vertically
 * (across rows), then after an 8x8 transpose, applied again horizontally.
 *
 * The butterfly operations exactly replicate od_bin_fdct8 from psnr_hvs.c.
 */

/*
 * OD_DCT_RSHIFT equivalent for AVX2:
 * ((int32_t)(((uint32_t)(a) >> (32 - b)) + a)) >> b
 */
#define AVX2_OD_DCT_RSHIFT(a, b) \
    _mm256_srai_epi32( \
        _mm256_add_epi32( \
            _mm256_srli_epi32(a, 32 - (b)), \
            a \
        ), \
        (b) \
    )

/*
 * Multiply-add-shift: (a * coeff + offset) >> shift
 * Uses _mm256_mullo_epi32 for 32-bit multiply.
 */
#define AVX2_MUL_ADD_RSHIFT(a, coeff, offset, shift) \
    _mm256_srai_epi32( \
        _mm256_add_epi32( \
            _mm256_mullo_epi32(a, _mm256_set1_epi32(coeff)), \
            _mm256_set1_epi32(offset) \
        ), \
        (shift) \
    )

/*
 * 8x8 transpose using AVX2 unpack and permute instructions.
 * Input: r0..r7 are 8 __m256i registers each holding 8 int32 values.
 * Output: r0..r7 are transposed.
 */
static inline void transpose_8x8_epi32(
    __m256i *r0, __m256i *r1, __m256i *r2, __m256i *r3,
    __m256i *r4, __m256i *r5, __m256i *r6, __m256i *r7)
{
    /* Stage 1: interleave 32-bit pairs */
    __m256i a0 = _mm256_unpacklo_epi32(*r0, *r1); /* r0[0],r1[0],r0[1],r1[1],r0[4],r1[4],r0[5],r1[5] */
    __m256i a1 = _mm256_unpackhi_epi32(*r0, *r1); /* r0[2],r1[2],r0[3],r1[3],r0[6],r1[6],r0[7],r1[7] */
    __m256i a2 = _mm256_unpacklo_epi32(*r2, *r3);
    __m256i a3 = _mm256_unpackhi_epi32(*r2, *r3);
    __m256i a4 = _mm256_unpacklo_epi32(*r4, *r5);
    __m256i a5 = _mm256_unpackhi_epi32(*r4, *r5);
    __m256i a6 = _mm256_unpacklo_epi32(*r6, *r7);
    __m256i a7 = _mm256_unpackhi_epi32(*r6, *r7);

    /* Stage 2: interleave 64-bit pairs */
    __m256i b0 = _mm256_unpacklo_epi64(a0, a2); /* r0[0],r1[0],r2[0],r3[0],r0[4],r1[4],r2[4],r3[4] */
    __m256i b1 = _mm256_unpackhi_epi64(a0, a2);
    __m256i b2 = _mm256_unpacklo_epi64(a1, a3);
    __m256i b3 = _mm256_unpackhi_epi64(a1, a3);
    __m256i b4 = _mm256_unpacklo_epi64(a4, a6);
    __m256i b5 = _mm256_unpackhi_epi64(a4, a6);
    __m256i b6 = _mm256_unpacklo_epi64(a5, a7);
    __m256i b7 = _mm256_unpackhi_epi64(a5, a7);

    /* Stage 3: permute 128-bit lanes */
    *r0 = _mm256_permute2x128_si256(b0, b4, 0x20);
    *r1 = _mm256_permute2x128_si256(b1, b5, 0x20);
    *r2 = _mm256_permute2x128_si256(b2, b6, 0x20);
    *r3 = _mm256_permute2x128_si256(b3, b7, 0x20);
    *r4 = _mm256_permute2x128_si256(b0, b4, 0x31);
    *r5 = _mm256_permute2x128_si256(b1, b5, 0x31);
    *r6 = _mm256_permute2x128_si256(b2, b6, 0x31);
    *r7 = _mm256_permute2x128_si256(b3, b7, 0x31);
}

/*
 * Apply the od_bin_fdct8 butterfly to 8 values packed across 8 AVX2 registers.
 * Each register holds the same "position" across 8 transforms.
 *
 * Input: t0..t7 correspond to input positions 0,7,2,5,4,1,6,3
 *        (after the initial permutation in od_bin_fdct8)
 * Actually — the permutation is handled by the caller loading into the
 * correct registers.
 *
 * This function applies all butterfly operations from od_bin_fdct8 in-place.
 */
static inline void od_bin_fdct8_vec(
    __m256i *t0, __m256i *t1, __m256i *t2, __m256i *t3,
    __m256i *t4, __m256i *t5, __m256i *t6, __m256i *t7)
{
    __m256i t1h, t4h, t6h;

    /* +1/-1 butterflies */
    *t1 = _mm256_sub_epi32(*t0, *t1);
    t1h = AVX2_OD_DCT_RSHIFT(*t1, 1);
    *t0 = _mm256_sub_epi32(*t0, t1h);
    *t4 = _mm256_add_epi32(*t4, *t5);
    t4h = AVX2_OD_DCT_RSHIFT(*t4, 1);
    *t5 = _mm256_sub_epi32(*t5, t4h);
    *t3 = _mm256_sub_epi32(*t2, *t3);
    *t2 = _mm256_sub_epi32(*t2, AVX2_OD_DCT_RSHIFT(*t3, 1));
    *t6 = _mm256_add_epi32(*t6, *t7);
    t6h = AVX2_OD_DCT_RSHIFT(*t6, 1);
    *t7 = _mm256_sub_epi32(t6h, *t7);

    /* Embedded 4-point type-II DCT */
    *t0 = _mm256_add_epi32(*t0, t6h);
    *t6 = _mm256_sub_epi32(*t0, *t6);
    *t2 = _mm256_sub_epi32(t4h, *t2);
    *t4 = _mm256_sub_epi32(*t2, *t4);

    /* Embedded 2-point type-II DCT */
    /* t0 -= (t4 * 13573 + 16384) >> 15 */
    *t0 = _mm256_sub_epi32(*t0, AVX2_MUL_ADD_RSHIFT(*t4, 13573, 16384, 15));
    /* t4 += (t0 * 11585 + 8192) >> 14 */
    *t4 = _mm256_add_epi32(*t4, AVX2_MUL_ADD_RSHIFT(*t0, 11585, 8192, 14));
    /* t0 -= (t4 * 13573 + 16384) >> 15 */
    *t0 = _mm256_sub_epi32(*t0, AVX2_MUL_ADD_RSHIFT(*t4, 13573, 16384, 15));

    /* Embedded 2-point type-IV DST */
    /* t6 -= (t2 * 21895 + 16384) >> 15 */
    *t6 = _mm256_sub_epi32(*t6, AVX2_MUL_ADD_RSHIFT(*t2, 21895, 16384, 15));
    /* t2 += (t6 * 15137 + 8192) >> 14 */
    *t2 = _mm256_add_epi32(*t2, AVX2_MUL_ADD_RSHIFT(*t6, 15137, 8192, 14));
    /* t6 -= (t2 * 21895 + 16384) >> 15 */
    *t6 = _mm256_sub_epi32(*t6, AVX2_MUL_ADD_RSHIFT(*t2, 21895, 16384, 15));

    /* Embedded 4-point type-IV DST */
    /* t3 += (t5 * 19195 + 16384) >> 15 */
    *t3 = _mm256_add_epi32(*t3, AVX2_MUL_ADD_RSHIFT(*t5, 19195, 16384, 15));
    /* t5 += (t3 * 11585 + 8192) >> 14 */
    *t5 = _mm256_add_epi32(*t5, AVX2_MUL_ADD_RSHIFT(*t3, 11585, 8192, 14));
    /* t3 -= (t5 * 7489 + 4096) >> 13 */
    *t3 = _mm256_sub_epi32(*t3, AVX2_MUL_ADD_RSHIFT(*t5, 7489, 4096, 13));

    *t7 = _mm256_sub_epi32(AVX2_OD_DCT_RSHIFT(*t5, 1), *t7);
    *t5 = _mm256_sub_epi32(*t5, *t7);
    *t3 = _mm256_sub_epi32(t1h, *t3);
    *t1 = _mm256_sub_epi32(*t1, *t3);

    /* t7 += (t1 * 3227 + 16384) >> 15 */
    *t7 = _mm256_add_epi32(*t7, AVX2_MUL_ADD_RSHIFT(*t1, 3227, 16384, 15));
    /* t1 -= (t7 * 6393 + 16384) >> 15 */
    *t1 = _mm256_sub_epi32(*t1, AVX2_MUL_ADD_RSHIFT(*t7, 6393, 16384, 15));
    /* t7 += (t1 * 3227 + 16384) >> 15 */
    *t7 = _mm256_add_epi32(*t7, AVX2_MUL_ADD_RSHIFT(*t1, 3227, 16384, 15));

    /* t5 += (t3 * 2485 + 4096) >> 13 */
    *t5 = _mm256_add_epi32(*t5, AVX2_MUL_ADD_RSHIFT(*t3, 2485, 4096, 13));
    /* t3 -= (t5 * 18205 + 16384) >> 15 */
    *t3 = _mm256_sub_epi32(*t3, AVX2_MUL_ADD_RSHIFT(*t5, 18205, 16384, 15));
    /* t5 += (t3 * 2485 + 4096) >> 13 */
    *t5 = _mm256_add_epi32(*t5, AVX2_MUL_ADD_RSHIFT(*t3, 2485, 4096, 13));
}

void od_bin_fdct8x8_avx2(int32_t *y, int ystride, const int32_t *x, int xstride)
{
    __m256i r0, r1, r2, r3, r4, r5, r6, r7;

    /* Load 8 rows of input */
    r0 = _mm256_loadu_si256((const __m256i *)(x + 0 * xstride));
    r1 = _mm256_loadu_si256((const __m256i *)(x + 1 * xstride));
    r2 = _mm256_loadu_si256((const __m256i *)(x + 2 * xstride));
    r3 = _mm256_loadu_si256((const __m256i *)(x + 3 * xstride));
    r4 = _mm256_loadu_si256((const __m256i *)(x + 4 * xstride));
    r5 = _mm256_loadu_si256((const __m256i *)(x + 5 * xstride));
    r6 = _mm256_loadu_si256((const __m256i *)(x + 6 * xstride));
    r7 = _mm256_loadu_si256((const __m256i *)(x + 7 * xstride));

    /* Transpose to get columns in registers */
    transpose_8x8_epi32(&r0, &r1, &r2, &r3, &r4, &r5, &r6, &r7);

    /*
     * Apply initial permutation and DCT butterfly (column transform).
     * od_bin_fdct8 reads input as: x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]
     * and assigns them as: t0=x[0], t4=x[1], t2=x[2], t6=x[3], t7=x[4], t3=x[5], t5=x[6], t1=x[7]
     *
     * After transpose, r0 is column 0 (all x[0] values), r1 is column 1 (all x[1] values), etc.
     * So: t0=r0, t4=r1, t2=r2, t6=r3, t7=r4, t3=r5, t5=r6, t1=r7
     */
    {
        __m256i t0 = r0, t4 = r1, t2 = r2, t6 = r3;
        __m256i t7 = r4, t3 = r5, t5 = r6, t1 = r7;

        od_bin_fdct8_vec(&t0, &t1, &t2, &t3, &t4, &t5, &t6, &t7);

        /* Output: y[0]=t0, y[1]=t1, y[2]=t2, y[3]=t3, y[4]=t4, y[5]=t5, y[6]=t6, y[7]=t7 */
        r0 = t0; r1 = t1; r2 = t2; r3 = t3;
        r4 = t4; r5 = t5; r6 = t6; r7 = t7;
    }

    /* Transpose for row transform */
    transpose_8x8_epi32(&r0, &r1, &r2, &r3, &r4, &r5, &r6, &r7);

    /* Apply permutation and DCT butterfly (row transform) */
    {
        __m256i t0 = r0, t4 = r1, t2 = r2, t6 = r3;
        __m256i t7 = r4, t3 = r5, t5 = r6, t1 = r7;

        od_bin_fdct8_vec(&t0, &t1, &t2, &t3, &t4, &t5, &t6, &t7);

        r0 = t0; r1 = t1; r2 = t2; r3 = t3;
        r4 = t4; r5 = t5; r6 = t6; r7 = t7;
    }

    /* Store 8 rows of output */
    _mm256_storeu_si256((__m256i *)(y + 0 * ystride), r0);
    _mm256_storeu_si256((__m256i *)(y + 1 * ystride), r1);
    _mm256_storeu_si256((__m256i *)(y + 2 * ystride), r2);
    _mm256_storeu_si256((__m256i *)(y + 3 * ystride), r3);
    _mm256_storeu_si256((__m256i *)(y + 4 * ystride), r4);
    _mm256_storeu_si256((__m256i *)(y + 5 * ystride), r5);
    _mm256_storeu_si256((__m256i *)(y + 6 * ystride), r6);
    _mm256_storeu_si256((__m256i *)(y + 7 * ystride), r7);
}
