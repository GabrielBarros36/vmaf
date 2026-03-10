/**
 * Format and Codec Coverage Test Common Utilities
 *
 * Shared Y4M generation, fill patterns, and helper functions for format
 * coverage tests. All test data is generated programmatically at runtime.
 */

#ifndef TEST_FORMAT_COMMON_H_
#define TEST_FORMAT_COMMON_H_

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ========== Fill Patterns ========== */

enum fill_pattern {
    FILL_ZERO,
    FILL_MAX,
    FILL_MID,
    FILL_GRADIENT_H,
    FILL_GRADIENT_V,
    FILL_RANDOM_42,
    FILL_CHECKERBOARD,
    FILL_PATTERN_COUNT,
};

/* ========== Y4M Helpers ========== */

static inline void write_sample_16le(uint8_t *dst, uint16_t val) {
    dst[0] = val & 0xFF;
    dst[1] = (val >> 8) & 0xFF;
}

static inline uint16_t read_sample_16le(const uint8_t *src) {
    return (uint16_t)src[0] | ((uint16_t)src[1] << 8);
}

/**
 * Compute chroma plane dimensions for a given chroma string.
 */
static void chroma_dimensions(const char *chroma, unsigned w, unsigned h,
                              unsigned *c_w, unsigned *c_h)
{
    if (strncmp(chroma, "444", 3) == 0) {
        *c_w = w; *c_h = h;
    } else if (strncmp(chroma, "422", 3) == 0) {
        *c_w = (w + 1) / 2; *c_h = h;
    } else if (strcmp(chroma, "mono") == 0) {
        *c_w = 0; *c_h = 0;
    } else {
        /* 420 variants */
        *c_w = (w + 1) / 2; *c_h = (h + 1) / 2;
    }
}

/**
 * Compute the raw frame data size (excluding "FRAME\n").
 */
static size_t compute_frame_data_size(unsigned w, unsigned h,
                                      const char *chroma, unsigned bpc)
{
    unsigned xstride = (bpc > 8) ? 2 : 1;
    size_t y_sz = (size_t)w * h * xstride;
    unsigned c_w, c_h;
    chroma_dimensions(chroma, w, h, &c_w, &c_h);
    return y_sz + 2 * (size_t)c_w * c_h * xstride;
}

/**
 * Simple deterministic PRNG (xorshift32) seeded per-call.
 */
static uint32_t xorshift32(uint32_t *state) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

/**
 * Get a sample value according to the fill pattern.
 * x, y are pixel coordinates within the plane.
 * w is the plane width.
 * max_val is (1 << bpc) - 1.
 */
static uint16_t get_fill_value(enum fill_pattern pattern,
                               unsigned x, unsigned y,
                               unsigned w __attribute__((unused)),
                               uint16_t max_val, uint32_t *rng)
{
    switch (pattern) {
    case FILL_ZERO:
        return 0;
    case FILL_MAX:
        return max_val;
    case FILL_MID:
        return max_val / 2;
    case FILL_GRADIENT_H:
        return (uint16_t)(x % (max_val + 1));
    case FILL_GRADIENT_V:
        return (uint16_t)(y % (max_val + 1));
    case FILL_RANDOM_42:
        return (uint16_t)(xorshift32(rng) % (max_val + 1));
    case FILL_CHECKERBOARD:
        return ((x + y) & 1) ? max_val : 0;
    default:
        return 0;
    }
}

/**
 * Fill a plane buffer with test pattern data.
 * Buffer is laid out in row-major order, tightly packed (Y4M layout).
 */
static void fill_plane(uint8_t *buf, unsigned w, unsigned h,
                       unsigned bpc, enum fill_pattern pattern,
                       uint32_t *rng)
{
    uint16_t max_val = (uint16_t)((1u << bpc) - 1);
    unsigned xstride = (bpc > 8) ? 2 : 1;

    for (unsigned y = 0; y < h; y++) {
        for (unsigned x = 0; x < w; x++) {
            uint16_t val = get_fill_value(pattern, x, y, w, max_val, rng);
            if (bpc > 8) {
                write_sample_16le(buf + (y * w + x) * 2, val);
            } else {
                buf[y * w + x] = (uint8_t)val;
            }
        }
    }
    (void)xstride;
}

/**
 * Create a temporary file containing only the raw Y4M header.
 * The header string is written verbatim. Caller controls exact byte content.
 * Returns a FILE* rewound to the beginning, or NULL on failure.
 * Caller must fclose() the returned file.
 */
static FILE *create_y4m_header_only(const char *header, size_t len)
{
    FILE *f = tmpfile();
    if (!f) return NULL;
    if (fwrite(header, 1, len, f) != len) {
        fclose(f);
        return NULL;
    }
    fseek(f, 0, SEEK_SET);
    return f;
}

/**
 * Create a temporary file containing a Y4M stream with the given parameters.
 * Returns a FILE* rewound to the beginning, or NULL on failure.
 * Caller must fclose() the returned file.
 */
static FILE *create_y4m_file(unsigned w, unsigned h, const char *chroma,
                             unsigned bpc, unsigned n_frames,
                             enum fill_pattern pattern)
{
    FILE *f = tmpfile();
    if (!f) return NULL;

    /* Write stream header */
    char hdr[256];
    int hdr_len = snprintf(hdr, sizeof(hdr),
                           "YUV4MPEG2 W%u H%u F30:1 Ip C%s\n",
                           w, h, chroma);
    if (hdr_len < 0 || (size_t)hdr_len >= sizeof(hdr)) {
        fclose(f);
        return NULL;
    }
    if (fwrite(hdr, 1, (size_t)hdr_len, f) != (size_t)hdr_len) {
        fclose(f);
        return NULL;
    }

    /* Compute plane sizes */
    unsigned c_w, c_h;
    chroma_dimensions(chroma, w, h, &c_w, &c_h);

    size_t frame_data_sz = compute_frame_data_size(w, h, chroma, bpc);
    uint8_t *frame_buf = (uint8_t *)malloc(frame_data_sz);
    if (!frame_buf) {
        fclose(f);
        return NULL;
    }

    for (unsigned fr = 0; fr < n_frames; fr++) {
        /* Write FRAME header */
        if (fwrite("FRAME\n", 1, 6, f) != 6) {
            free(frame_buf);
            fclose(f);
            return NULL;
        }

        /* Fill planes */
        uint32_t rng = 42 + fr;
        uint8_t *ptr = frame_buf;

        /* Y plane */
        fill_plane(ptr, w, h, bpc, pattern, &rng);
        ptr += (size_t)w * h * ((bpc > 8) ? 2 : 1);

        /* Cb and Cr planes */
        if (c_w > 0 && c_h > 0) {
            fill_plane(ptr, c_w, c_h, bpc, pattern, &rng);
            ptr += (size_t)c_w * c_h * ((bpc > 8) ? 2 : 1);
            fill_plane(ptr, c_w, c_h, bpc, pattern, &rng);
        }

        if (fwrite(frame_buf, 1, frame_data_sz, f) != frame_data_sz) {
            free(frame_buf);
            fclose(f);
            return NULL;
        }
    }

    free(frame_buf);
    fseek(f, 0, SEEK_SET);
    return f;
}

#endif /* TEST_FORMAT_COMMON_H_ */
