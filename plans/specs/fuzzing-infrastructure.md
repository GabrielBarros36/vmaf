# Fuzzing Infrastructure -- Specification

## 1. Objective

Discover crashes, hangs, memory corruption, and undefined behavior in libvmaf's input-processing and scoring paths through two complementary techniques: **structure-aware coverage-guided fuzzing** (libFuzzer and AFL++) of C parsing code, and **property-based testing** (Python Hypothesis) of high-level scoring invariants.

This specification covers fuzz target design, corpus seeding, build integration, CI execution, crash triage, and acceptance criteria.

---

## 2. Scope -- Attack Surfaces

There are **3 structure-aware fuzz targets** and **6 property-based invariants**.

### 2.1 Fuzz Targets (C / libFuzzer + AFL++)

| # | Target | Entry Point | Input Type | Bug Class |
|---|--------|-------------|------------|-----------|
| F1 | Y4M header parser | `y4m_input_open` via `fmemopen` | Raw bytes (Y4M header + frame data) | Buffer overflow, integer overflow in dimension calculation, OOM on crafted dimensions |
| F2 | Model JSON loader | `vmaf_read_json_model_from_buffer` | Raw bytes (JSON) | Heap corruption in pdjson, out-of-bounds in SVM model parsing, null dereference on malformed fields |
| F3 | Picture read pipeline | `vmaf_read_pictures` with random pixel data | Structured: valid `VmafPicture` pair with fuzzed pixel content | Numeric overflow in feature extractors, division by zero, NaN propagation |

### 2.2 Property-Based Invariants (Python Hypothesis)

| # | Property | Metric | Invariant |
|---|----------|--------|-----------|
| P1 | Identity reference | VMAF | `VMAF(ref, ref) == 100.0` for any valid input |
| P2 | Identity reference | PSNR | `PSNR(ref, ref) == inf` for any valid input |
| P3 | Bounded range | SSIM | `0 <= SSIM(ref, dis) <= 1` for any valid inputs |
| P4 | Determinism | VMAF | `VMAF(ref, dis)` is identical across repeated runs |
| P5 | Thread invariance | VMAF | Score is invariant to `n_threads` (1 vs 4 vs 8) |
| P6 | Int/float agreement | VMAF | Integer and float feature extractors agree within tolerance (relative error < 0.02) |

---

## 3. Fuzz Target F1 -- Y4M Parser

### 3.1 Attack Surface Analysis

The Y4M parser in `libvmaf/tools/y4m_input.c` reads from a `FILE *` stream with a fixed 256-byte header buffer (`Y4M_HEADER_BUFSIZE`). It uses `sscanf` to parse width, height, framerate, and chroma type from tag fields. The `y4m_input_open_impl` function then computes buffer sizes as products of parsed dimensions, allocates buffers with `malloc`, and reads frame data. Key risks:

- Integer overflow in `pic_w * pic_h` or chroma size calculations
- Unchecked allocation sizes derived from attacker-controlled dimensions
- `memcpy` of chroma type into a fixed 16-byte buffer (bounds-checked but worth validating)
- Frame header parsing loop without length bound

### 3.2 Fuzz Harness

```c
/* fuzz_y4m_parser.c */
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "vidinput.h"

/* Limit maximum input size to prevent OOM on huge dimension values.
   The fuzzer should explore header parsing, not exhaust memory. */
#define MAX_INPUT_SIZE (1 << 20)  /* 1 MiB */

/* Limit allocations to prevent OOM kills that mask real bugs. */
#define MAX_ALLOC_SIZE (64 << 20)  /* 64 MiB */

extern const video_input_vtbl Y4M_INPUT_VTBL;

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    if (size < 10 || size > MAX_INPUT_SIZE)
        return 0;

    /* Create a FILE* from the fuzzer-provided buffer. */
    FILE *f = fmemopen((void *)data, size, "rb");
    if (!f)
        return 0;

    /* Attempt to open as Y4M. This exercises y4m_parse_tags and
       y4m_input_open_impl, including all chroma format branches,
       dimension computation, and buffer allocation. */
    void *ctx = Y4M_INPUT_VTBL.open(f);
    if (ctx) {
        video_input_info info;
        Y4M_INPUT_VTBL.get_info(ctx, &info);

        /* Attempt to read one frame if the header parsed successfully.
           This exercises y4m_input_fetch_frame and convert functions. */
        video_input_ycbcr ycbcr;
        char tag[5];
        Y4M_INPUT_VTBL.fetch_frame(ctx, f, ycbcr, tag);

        Y4M_INPUT_VTBL.close(ctx);
        free(ctx);
    }

    fclose(f);
    return 0;
}
```

### 3.3 Seed Corpus

Generate minimal valid Y4M files covering each chroma format:

```bash
#!/bin/bash
# generate_y4m_seeds.sh
# Creates minimal 2x2 Y4M files for each supported chroma type.

SEED_DIR="corpus/y4m"
mkdir -p "$SEED_DIR"

for CHROMA in 420 420jpeg 420mpeg2 420p10 420p12 420paldv 422 422p10 422p12 \
              444 444p10 444p12 444alpha 411 mono; do
    HEADER="YUV4MPEG2 W2 H2 F30:1 Ip C${CHROMA}"
    FRAME="FRAME\n"
    # Compute frame data size based on chroma type and write zero-filled data.
    case "$CHROMA" in
        420|420jpeg|420mpeg2) BYTES=6 ;;    # 4 + 1 + 1
        420p10|420p12)        BYTES=12 ;;   # (4 + 1 + 1) * 2
        420paldv)             BYTES=6 ;;
        422|422p10|422p12)    BYTES=$((2*2 + 2*2)) ;; # simplified
        444|444alpha)         BYTES=12 ;;
        444p10|444p12)        BYTES=24 ;;
        411)                  BYTES=6 ;;
        mono)                 BYTES=4 ;;
        *)                    BYTES=12 ;;
    esac
    {
        printf '%s\n' "$HEADER"
        printf 'FRAME\n'
        dd if=/dev/zero bs=1 count=$BYTES 2>/dev/null
    } > "$SEED_DIR/seed_${CHROMA}.y4m"
done
```

### 3.4 Dictionary

```
# y4m.dict -- tokens for structure-aware mutation
"YUV4MPEG2"
"FRAME"
" W"
" H"
" F"
" Ip"
" It"
" Ib"
" C420"
" C420jpeg"
" C420mpeg2"
" C420p10"
" C420p12"
" C420paldv"
" C422"
" C422p10"
" C422p12"
" C444"
" C444p10"
" C444p12"
" C444alpha"
" C411"
" Cmono"
" A1:1"
```

---

## 4. Fuzz Target F2 -- Model JSON Loader

### 4.1 Attack Surface Analysis

`vmaf_read_json_model_from_buffer` in `libvmaf/src/read_json_model.c` parses untrusted JSON using the pdjson streaming parser. The parsing code allocates a `VmafModel` struct with fixed-size arrays (`MAX_FEATURE_COUNT = 64`, `MAX_KNOT_COUNT = 10`) and populates fields from JSON values. Key risks:

- `strdup` of unbounded JSON strings (feature names, model strings)
- `svm_parse_model_from_buffer` parses a text-format SVM model embedded as a JSON string value
- Index-based array writes guarded by `MAX_FEATURE_COUNT` but worth validating under mutation
- pdjson itself may have parsing bugs on malformed JSON

### 4.2 Fuzz Harness

```c
/* fuzz_json_model.c */
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include "libvmaf/model.h"

/* Forward declaration of the buffer-based loader. */
int vmaf_read_json_model_from_buffer(VmafModel **model, VmafModelConfig *cfg,
                                     const char *data, const int data_len);

#define MAX_INPUT_SIZE (1 << 20)  /* 1 MiB */

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    if (size == 0 || size > MAX_INPUT_SIZE)
        return 0;

    VmafModel *model = NULL;
    VmafModelConfig cfg = {
        .name = "fuzz",
        .flags = VMAF_MODEL_FLAGS_DEFAULT,
    };

    int err = vmaf_read_json_model_from_buffer(&model, &cfg,
                                                (const char *)data,
                                                (int)size);
    if (!err && model) {
        vmaf_model_destroy(model);
    }

    return 0;
}
```

### 4.3 Seed Corpus

Use existing model JSON files from the repository as seeds:

```
model/vmaf_v0.6.1.json
model/vmaf_v0.6.1neg.json
model/vmaf_4k_v0.6.1.json
model/vmaf_b_v0.6.3.json
model/vmaf_float_v0.6.1.json
```

Additionally, create a minimal valid model JSON seed:

```json
{
    "model_dict": {
        "model_type": "LIBSVMNUSVR",
        "norm_type": "linear_rescale",
        "score_clip": [0.0, 100.0],
        "slopes": [1.0, 1.0],
        "intercepts": [0.0, 0.0],
        "feature_names": ["VMAF_integer_feature_vif_scale0_score"],
        "model": "svm_type nu_svr\nkernel_type rbf\ngamma 0.04\nnr_class 2\ntotal_sv 1\nrho 0\nSV\n1 1:1\n"
    }
}
```

### 4.4 Dictionary

```
# json_model.dict -- tokens for model JSON mutation
"model_dict"
"model_type"
"LIBSVMNUSVR"
"BOOTSTRAP_LIBSVMNUSVR"
"RESIDUEBOOTSTRAP_LIBSVMNUSVR"
"norm_type"
"linear_rescale"
"none"
"score_clip"
"slopes"
"intercepts"
"feature_names"
"feature_opts_dicts"
"model"
"score_transform"
"enabled"
"p0"
"p1"
"p2"
"knots"
"out_lte_in"
"out_gte_in"
"svm_type"
"nu_svr"
"kernel_type"
"rbf"
"gamma"
"nr_class"
"total_sv"
"rho"
"SV"
```

---

## 5. Fuzz Target F3 -- Picture Read Pipeline

### 5.1 Attack Surface Analysis

`vmaf_read_pictures` processes `VmafPicture` pairs through the registered feature extractors. Even with valid picture dimensions, adversarial pixel data can trigger numeric issues in the feature extraction kernels (ADM, VIF, Motion, CAMBI): integer overflow in accumulation, division by zero in normalization, NaN propagation through the SVM model.

### 5.2 Fuzz Harness

```c
/* fuzz_read_pictures.c */
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include "libvmaf/libvmaf.h"
#include "libvmaf/picture.h"
#include "libvmaf/model.h"

/*
 * Structured fuzzing: the first 4 bytes select width/height from a
 * constrained set, and the remaining bytes fill pixel data.
 * This avoids spending fuzzer cycles on invalid picture configurations.
 */

static const unsigned WIDTHS[]  = { 64, 128, 192, 576 };
static const unsigned HEIGHTS[] = { 64, 128, 108, 324 };
#define N_DIMS 4

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    if (size < 8)
        return 0;

    /* Use first 2 bytes to select dimensions and bit depth. */
    unsigned dim_idx = data[0] % N_DIMS;
    unsigned w = WIDTHS[dim_idx];
    unsigned h = HEIGHTS[dim_idx];
    unsigned bpc = (data[1] & 1) ? 10 : 8;

    data += 2;
    size -= 2;

    /* Compute minimum pixel data needed for YUV420P. */
    unsigned bytes_per_sample = (bpc > 8) ? 2 : 1;
    size_t luma_sz = (size_t)w * h * bytes_per_sample;
    size_t chroma_sz = (size_t)(w / 2) * (h / 2) * bytes_per_sample;
    size_t frame_sz = luma_sz + 2 * chroma_sz;

    if (size < frame_sz)
        return 0;

    /* Initialize VMAF context. */
    VmafContext *vmaf = NULL;
    VmafConfiguration cfg = {
        .log_level = VMAF_LOG_LEVEL_NONE,
        .n_threads = 1,
        .n_subsample = 0,
        .cpumask = 0,
        .gpumask = 0,
    };
    if (vmaf_init(&vmaf, cfg) < 0)
        return 0;

    /* Load built-in model. */
    VmafModel *model = NULL;
    VmafModelConfig model_cfg = {
        .name = "fuzz",
        .flags = VMAF_MODEL_FLAGS_DEFAULT,
    };
    if (vmaf_model_load(&model, &model_cfg, "vmaf_v0.6.1") < 0) {
        vmaf_close(vmaf);
        return 0;
    }
    if (vmaf_use_features_from_model(vmaf, model) < 0) {
        vmaf_model_destroy(model);
        vmaf_close(vmaf);
        return 0;
    }

    /* Allocate reference and distorted pictures. */
    VmafPicture ref, dist;
    if (vmaf_picture_alloc(&ref, VMAF_PIX_FMT_YUV420P, bpc, w, h) < 0) {
        vmaf_model_destroy(model);
        vmaf_close(vmaf);
        return 0;
    }
    if (vmaf_picture_alloc(&dist, VMAF_PIX_FMT_YUV420P, bpc, w, h) < 0) {
        vmaf_picture_unref(&ref);
        vmaf_model_destroy(model);
        vmaf_close(vmaf);
        return 0;
    }

    /* Fill reference with constant mid-gray. */
    unsigned mid = (bpc > 8) ? 512 : 128;
    for (int p = 0; p < 3; p++) {
        size_t plane_sz = (size_t)ref.w[p] * ref.h[p] * bytes_per_sample;
        memset(ref.data[p], mid & 0xFF, plane_sz);
    }

    /* Fill distorted with fuzz data. */
    const uint8_t *src = data;
    for (int p = 0; p < 3; p++) {
        size_t row_bytes = (size_t)dist.w[p] * bytes_per_sample;
        for (unsigned y = 0; y < dist.h[p]; y++) {
            uint8_t *row = (uint8_t *)dist.data[p] + y * dist.stride[p];
            size_t to_copy = row_bytes;
            if (src + to_copy > data + size)
                to_copy = (size_t)((data + size) - src);
            if (to_copy > 0)
                memcpy(row, src, to_copy);
            src += row_bytes;
        }
    }

    /* Process the frame pair. */
    vmaf_read_pictures(vmaf, &ref, &dist, 0);

    /* Flush. */
    vmaf_read_pictures(vmaf, NULL, NULL, 0);

    /* Read score (exercises predict path). */
    double score;
    vmaf_score_at_index(vmaf, model, &score, 0);

    /* Cleanup. */
    vmaf_model_destroy(model);
    vmaf_close(vmaf);

    return 0;
}
```

### 5.3 Seed Corpus

No file-based corpus needed. The harness uses structured input. Create a few synthetic seeds:

```python
#!/usr/bin/env python3
"""generate_picture_seeds.py -- create seed inputs for fuzz_read_pictures."""
import struct, os

os.makedirs("corpus/pictures", exist_ok=True)

for dim_idx in range(4):
    widths  = [64, 128, 192, 576]
    heights = [64, 128, 108, 324]
    w, h = widths[dim_idx], heights[dim_idx]
    for bpc_flag in [0, 1]:
        bpc = 10 if (bpc_flag & 1) else 8
        bps = 2 if bpc > 8 else 1
        luma = w * h * bps
        chroma = (w // 2) * (h // 2) * bps
        frame = luma + 2 * chroma
        header = bytes([dim_idx, bpc_flag])
        # Fill with mid-gray
        mid = 128 if bpc == 8 else 0  # 0x0200 little-endian = 512
        pixel_data = bytes([mid]) * frame
        seed = header + pixel_data
        name = f"corpus/pictures/seed_{w}x{h}_{bpc}bit.bin"
        with open(name, "wb") as f:
            f.write(seed)
```

---

## 6. AFL++ Integration

### 6.1 Harness Adaptation

The same fuzz harness source files work for both libFuzzer and AFL++. For AFL++, compile with `afl-clang-fast` and use the persistent mode shim:

```c
/* afl_persistent_wrapper.c -- only compiled for AFL++ builds */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

__AFL_FUZZ_INIT();

extern int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size);

int main(void) {
    __AFL_INIT();
    unsigned char *buf = __AFL_FUZZ_TESTCASE_BUF;

    while (__AFL_LOOP(10000)) {
        int len = __AFL_FUZZ_TESTCASE_LEN;
        LLVMFuzzerTestOneInput(buf, len);
    }
    return 0;
}
```

### 6.2 Build Commands

```bash
# AFL++ build
export CC=afl-clang-fast
export CXX=afl-clang-fast++

cd libvmaf
meson setup build_afl --default-library=static \
    -Denable_tests=false -Denable_docs=false
ninja -C build_afl

# Compile a specific target (e.g., F2 JSON model fuzzer)
$CC -g -O1 -fsanitize=address,undefined \
    -I include -I src \
    fuzz/fuzz_json_model.c fuzz/afl_persistent_wrapper.c \
    -o build_afl/fuzz_json_model_afl \
    -Lbuild_afl -lvmaf -lm -lpthread
```

### 6.3 Sanitizer Configurations

All fuzz targets must be compiled with at least two sanitizer configurations:

| Configuration | Compiler Flags | Purpose |
|---------------|---------------|---------|
| ASan + UBSan | `-fsanitize=address,undefined -fno-sanitize-recover=all` | Heap/stack overflow, use-after-free, signed integer overflow, null dereference |
| MSan | `-fsanitize=memory` | Uninitialized memory reads |

---

## 7. Build System Integration

### 7.1 Directory Layout

```
libvmaf/fuzz/
    fuzz_y4m_parser.c         # F1 harness
    fuzz_json_model.c         # F2 harness
    fuzz_read_pictures.c      # F3 harness
    afl_persistent_wrapper.c  # AFL++ persistent mode adapter
    corpus/
        y4m/                  # Y4M seed files
        json_model/           # Model JSON seed files
        pictures/             # Structured picture seeds
    dict/
        y4m.dict              # Y4M dictionary
        json_model.dict       # JSON model dictionary
    meson.build               # Fuzz target build definitions
```

### 7.2 Meson Configuration

Add `libvmaf/fuzz/meson.build`:

```meson
if not get_option('enable_fuzz')
    subdir_done()
endif

fuzz_inc = include_directories('../src/', '../tools/')

# Determine the fuzzing engine.
# When building with -Db_sanitize=address and clang, libFuzzer is built-in.
# For standalone builds, link against an external fuzzer engine.
fuzzer_engine = []
if cc.get_id() == 'clang'
    fuzzer_engine_flags = ['-fsanitize=fuzzer']
else
    # Fall back to linking libFuzzer statically
    fuzzer_engine = [cc.find_library('Fuzzer', required: false)]
endif

fuzz_common_flags = ['-DFUZZ_BUILD']

# F1: Y4M parser fuzz target
fuzz_y4m = executable('fuzz_y4m_parser',
    ['fuzz_y4m_parser.c', '../tools/y4m_input.c'],
    include_directories: [libvmaf_inc, fuzz_inc,
                          include_directories('../tools/')],
    c_args: fuzz_common_flags + fuzzer_engine_flags,
    link_args: fuzzer_engine_flags,
    link_with: get_option('default_library') == 'both'
               ? libvmaf.get_static_lib() : libvmaf,
    dependencies: [math_lib, thread_lib],
)

# F2: JSON model fuzz target
fuzz_json_model = executable('fuzz_json_model',
    ['fuzz_json_model.c', '../src/read_json_model.c',
     '../src/pdjson.c', '../src/dict.c', '../src/log.c'],
    include_directories: [libvmaf_inc, fuzz_inc,
                          include_directories('../src/')],
    c_args: fuzz_common_flags + fuzzer_engine_flags,
    link_args: fuzzer_engine_flags,
    link_with: get_option('default_library') == 'both'
               ? libvmaf.get_static_lib() : libvmaf,
    dependencies: [math_lib, thread_lib],
    objects: libsvm_static_lib.extract_all_objects(recursive: true),
)

# F3: Picture read pipeline fuzz target
fuzz_read_pictures = executable('fuzz_read_pictures',
    ['fuzz_read_pictures.c'],
    include_directories: [libvmaf_inc, fuzz_inc],
    c_args: fuzz_common_flags + fuzzer_engine_flags,
    link_args: fuzzer_engine_flags,
    link_with: get_option('default_library') == 'both'
               ? libvmaf.get_static_lib() : libvmaf,
    dependencies: [math_lib, thread_lib],
)
```

Add the option to `libvmaf/meson_options.txt`:

```meson
option('enable_fuzz',
    type: 'boolean',
    value: false,
    description: 'Build fuzz targets (requires clang with libFuzzer)')
```

Add the subdirectory inclusion to `libvmaf/test/meson.build` (or top-level `libvmaf/meson.build`):

```meson
subdir('fuzz')
```

### 7.3 Build and Run Locally

```bash
# Configure with fuzzing enabled (requires clang)
CC=clang CXX=clang++ meson setup build_fuzz libvmaf \
    -Denable_fuzz=true \
    -Denable_tests=false \
    -Db_sanitize=address,undefined \
    -Db_lundef=false \
    --default-library=static

ninja -C build_fuzz

# Run the Y4M parser fuzzer for 10 minutes
./build_fuzz/fuzz/fuzz_y4m_parser \
    -dict=libvmaf/fuzz/dict/y4m.dict \
    -max_len=65536 \
    -timeout=10 \
    -rss_limit_mb=2048 \
    libvmaf/fuzz/corpus/y4m/

# Run the JSON model fuzzer
./build_fuzz/fuzz/fuzz_json_model \
    -dict=libvmaf/fuzz/dict/json_model.dict \
    -max_len=1048576 \
    -timeout=30 \
    -rss_limit_mb=2048 \
    libvmaf/fuzz/corpus/json_model/

# Run the picture pipeline fuzzer
./build_fuzz/fuzz/fuzz_read_pictures \
    -max_len=2097152 \
    -timeout=60 \
    -rss_limit_mb=4096 \
    libvmaf/fuzz/corpus/pictures/
```

---

## 8. CI Integration

### 8.1 Standalone CI Job (GitHub Actions)

```yaml
# .github/workflows/fuzz.yml
name: Fuzzing

on:
  push:
    branches: [master]
  pull_request:
    branches: [master]
  schedule:
    # Run extended fuzzing nightly at 03:00 UTC
    - cron: '0 3 * * *'

jobs:
  fuzz:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        target:
          - name: y4m_parser
            binary: fuzz_y4m_parser
            dict: libvmaf/fuzz/dict/y4m.dict
            corpus: libvmaf/fuzz/corpus/y4m
            max_len: 65536
            timeout: 10
          - name: json_model
            binary: fuzz_json_model
            dict: libvmaf/fuzz/dict/json_model.dict
            corpus: libvmaf/fuzz/corpus/json_model
            max_len: 1048576
            timeout: 30
          - name: read_pictures
            binary: fuzz_read_pictures
            dict: ""
            corpus: libvmaf/fuzz/corpus/pictures
            max_len: 2097152
            timeout: 60

    steps:
      - uses: actions/checkout@v6

      - name: Install dependencies
        run: |
          sudo apt-get update
          sudo apt-get install -y meson ninja-build nasm clang

      - name: Build fuzz targets
        run: |
          CC=clang CXX=clang++ meson setup build_fuzz libvmaf \
              -Denable_fuzz=true \
              -Denable_tests=false \
              -Db_sanitize=address,undefined \
              -Db_lundef=false \
              --default-library=static
          ninja -C build_fuzz

      - name: Generate seed corpus
        run: |
          cd libvmaf/fuzz
          bash generate_y4m_seeds.sh
          python3 generate_picture_seeds.py
          mkdir -p corpus/json_model
          cp ../../model/vmaf_v0.6.1.json corpus/json_model/
          cp ../../model/vmaf_v0.6.1neg.json corpus/json_model/
          cp ../../model/vmaf_b_v0.6.3.json corpus/json_model/

      - name: Run fuzzer
        run: |
          DURATION=${{ github.event_name == 'schedule' && '3600' || '120' }}
          DICT_FLAG=""
          if [ -n "${{ matrix.target.dict }}" ]; then
              DICT_FLAG="-dict=${{ matrix.target.dict }}"
          fi
          ./build_fuzz/fuzz/${{ matrix.target.binary }} \
              $DICT_FLAG \
              -max_len=${{ matrix.target.max_len }} \
              -timeout=${{ matrix.target.timeout }} \
              -rss_limit_mb=4096 \
              -max_total_time=$DURATION \
              -print_final_stats=1 \
              ${{ matrix.target.corpus }}

      - name: Upload crash artifacts
        if: failure()
        uses: actions/upload-artifact@v5
        with:
          name: fuzz-crash-${{ matrix.target.name }}
          path: |
            crash-*
            leak-*
            timeout-*
```

### 8.2 OSS-Fuzz Integration (Future)

For long-term continuous fuzzing, integrate with [OSS-Fuzz](https://github.com/google/oss-fuzz). This requires:

1. A `project.yaml` in the OSS-Fuzz repository declaring the VMAF project.
2. A `Dockerfile` that installs build dependencies.
3. A `build.sh` script that compiles the fuzz targets with the OSS-Fuzz-provided compiler flags.

OSS-Fuzz `build.sh`:

```bash
#!/bin/bash -eu
# OSS-Fuzz build script for VMAF

cd libvmaf

# Build libvmaf statically
meson setup build --default-library=static \
    -Denable_tests=false \
    -Denable_docs=false \
    -Denable_fuzz=false
ninja -C build

# Build each fuzz target, linking against $LIB_FUZZING_ENGINE
for target in fuzz_y4m_parser fuzz_json_model fuzz_read_pictures; do
    $CC $CFLAGS -I include -I src -I tools \
        fuzz/${target}.c \
        -c -o build/${target}.o

    $CXX $CXXFLAGS $LIB_FUZZING_ENGINE \
        build/${target}.o \
        -o $OUT/${target} \
        -Lbuild -lvmaf -lm -lpthread
done

# Copy seed corpora and dictionaries
for target in y4m json_model pictures; do
    zip -j $OUT/fuzz_${target}_seed_corpus.zip fuzz/corpus/${target}/*
done
cp fuzz/dict/*.dict $OUT/
```

---

## 9. Coverage-Guided Fuzzing Configuration

### 9.1 libFuzzer Options

| Option | F1 (Y4M) | F2 (JSON) | F3 (Pictures) | Rationale |
|--------|----------|-----------|----------------|-----------|
| `-max_len` | 65536 | 1048576 | 2097152 | Y4M headers are small; JSON models can be large; picture data needs full frames |
| `-timeout` | 10 | 30 | 60 | Y4M parsing is fast; JSON parsing moderate; full VMAF pipeline is expensive |
| `-rss_limit_mb` | 2048 | 2048 | 4096 | Prevent OOM on crafted dimensions |
| `-dict` | y4m.dict | json_model.dict | (none) | Structure-aware mutation |
| `-max_total_time` | 120 (CI) / 3600 (nightly) | 120 / 3600 | 120 / 3600 | Short on PR, extended nightly |
| `-jobs` | 4 | 4 | 2 | Parallel fuzzing instances; F3 is memory-heavy |
| `-workers` | 4 | 4 | 2 | Match jobs |
| `-print_final_stats` | 1 | 1 | 1 | Log coverage at end |

### 9.2 Coverage Tracking

Build with source-level coverage to measure which code paths the fuzzer has explored:

```bash
# Build with coverage instrumentation
CC=clang CXX=clang++ meson setup build_cov libvmaf \
    -Denable_fuzz=true \
    -Denable_tests=false \
    -Db_sanitize=address,undefined \
    -Db_lundef=false \
    --default-library=static
CFLAGS="-fprofile-instr-generate -fcoverage-mapping" \
    ninja -C build_cov

# After fuzzing, generate coverage report
llvm-profdata merge -sparse default.profraw -o fuzz.profdata
llvm-cov show ./build_cov/fuzz/fuzz_json_model \
    -instr-profile=fuzz.profdata \
    -format=html -output-dir=coverage_report/
```

### 9.3 Coverage Targets

| Fuzz Target | Minimum Line Coverage | Key Files |
|-------------|----------------------|-----------|
| F1 (Y4M) | 80% of `y4m_input.c` | `libvmaf/tools/y4m_input.c` |
| F2 (JSON) | 90% of `read_json_model.c`, 70% of `pdjson.c` | `libvmaf/src/read_json_model.c`, `libvmaf/src/pdjson.c` |
| F3 (Pictures) | 50% of feature extractors exercised | `libvmaf/src/feature/integer_adm.c`, `integer_vif.c`, `integer_motion.c` |

---

## 10. Python Property-Based Tests (Hypothesis)

### 10.1 Test File

Create `python/test/test_property_fuzz.py`:

```python
"""Property-based tests for libvmaf scoring invariants using Hypothesis."""

import math
import subprocess
import tempfile
import os
import struct

import pytest

try:
    from hypothesis import given, settings, HealthCheck, assume
    from hypothesis import strategies as st
    HAS_HYPOTHESIS = True
except ImportError:
    HAS_HYPOTHESIS = False

from vmaf.config import VmafConfig

pytestmark = pytest.mark.skipif(
    not HAS_HYPOTHESIS,
    reason="hypothesis not installed"
)

# ---- Helpers ----

def _vmafexec_path():
    """Locate the vmaf executable."""
    # Try common build locations
    for candidate in [
        os.path.join(VmafConfig.root_path(), "libvmaf", "build", "tools", "vmaf"),
        os.path.join(VmafConfig.root_path(), "libvmaf", "build_fuzz", "tools", "vmaf"),
    ]:
        if os.path.isfile(candidate):
            return candidate
    pytest.skip("vmaf binary not found")


def _write_y4m(path, width, height, bpc, pix_data):
    """Write a single-frame Y4M file with the given pixel data."""
    chroma = "420" if bpc == 8 else "420p10"
    header = f"YUV4MPEG2 W{width} H{height} F30:1 Ip C{chroma}\n"
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(b"FRAME\n")
        f.write(pix_data)


def _make_constant_y4m(path, width, height, bpc, value):
    """Create a Y4M file with constant pixel value."""
    bps = 2 if bpc > 8 else 1
    luma_sz = width * height * bps
    chroma_w, chroma_h = width // 2, height // 2
    chroma_sz = chroma_w * chroma_h * bps
    if bpc == 8:
        data = bytes([value]) * luma_sz + bytes([128]) * (2 * chroma_sz)
    else:
        val_bytes = struct.pack("<H", value)
        chroma_bytes = struct.pack("<H", 512)
        data = val_bytes * (width * height) + chroma_bytes * (2 * chroma_w * chroma_h)
    _write_y4m(path, width, height, bpc, data)


def _run_vmaf(ref_y4m, dis_y4m, model="vmaf_v0.6.1", feature=None,
              n_threads=1):
    """Run vmaf CLI and return a dict of feature scores."""
    vmaf_bin = _vmafexec_path()
    cmd = [
        vmaf_bin,
        "-r", ref_y4m,
        "-d", dis_y4m,
        "--threads", str(n_threads),
        "--json", "--output", "/dev/stdout",
    ]
    if model:
        cmd += ["-m", f"version={model}"]
    if feature:
        cmd += ["--feature", feature]

    result = subprocess.run(cmd, capture_output=True, timeout=120)
    if result.returncode != 0:
        pytest.fail(f"vmaf failed: {result.stderr.decode()}")

    import json
    output = json.loads(result.stdout)
    scores = {}
    if "frames" in output and len(output["frames"]) > 0:
        for metric in output["frames"][0].get("metrics", {}):
            scores[metric] = output["frames"][0]["metrics"][metric]
    return scores


# ---- Strategies ----

# Constrained dimensions: even, reasonable size for CI speed
valid_dimensions = st.tuples(
    st.sampled_from([32, 64, 128, 192]),   # width (must be even for 420)
    st.sampled_from([32, 64, 108, 128]),   # height (must be even for 420)
)

valid_bpc = st.sampled_from([8, 10])

pixel_value_8bit = st.integers(min_value=0, max_value=255)
pixel_value_10bit = st.integers(min_value=0, max_value=1023)


# ---- P1: VMAF(ref, ref) == 100.0 ----

@given(dims=valid_dimensions, bpc=valid_bpc, val=pixel_value_8bit)
@settings(max_examples=20, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
def test_vmaf_identity_score(dims, bpc, val):
    """VMAF of a frame compared with itself must be 100.0."""
    w, h = dims
    if bpc == 10:
        val = min(val * 4, 1023)
    with tempfile.TemporaryDirectory() as tmpdir:
        ref = os.path.join(tmpdir, "ref.y4m")
        _make_constant_y4m(ref, w, h, bpc, val)
        scores = _run_vmaf(ref, ref, model="vmaf_v0.6.1")
        vmaf_score = scores.get("vmaf", None)
        assert vmaf_score is not None, "VMAF score not found in output"
        assert abs(vmaf_score - 100.0) < 0.01, \
            f"VMAF(ref,ref) = {vmaf_score}, expected 100.0"


# ---- P2: PSNR(ref, ref) == inf ----

@given(dims=valid_dimensions, bpc=valid_bpc, val=pixel_value_8bit)
@settings(max_examples=20, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
def test_psnr_identity_infinity(dims, bpc, val):
    """PSNR of identical frames must be infinity (reported as 60+ dB)."""
    w, h = dims
    if bpc == 10:
        val = min(val * 4, 1023)
    with tempfile.TemporaryDirectory() as tmpdir:
        ref = os.path.join(tmpdir, "ref.y4m")
        _make_constant_y4m(ref, w, h, bpc, val)
        scores = _run_vmaf(ref, ref, model=None, feature="float_psnr")
        psnr = scores.get("psnr_y", scores.get("float_psnr", None))
        assert psnr is not None, "PSNR score not found"
        # libvmaf caps identical-frame PSNR at 60.0 dB
        assert psnr >= 60.0 or math.isinf(psnr), \
            f"PSNR(ref,ref) = {psnr}, expected >= 60.0"


# ---- P3: 0 <= SSIM(ref, dis) <= 1 ----

@given(dims=valid_dimensions, val_ref=pixel_value_8bit,
       val_dis=pixel_value_8bit)
@settings(max_examples=20, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
def test_ssim_bounded(dims, val_ref, val_dis):
    """SSIM must be in [0, 1] for any valid input pair."""
    w, h = dims
    with tempfile.TemporaryDirectory() as tmpdir:
        ref = os.path.join(tmpdir, "ref.y4m")
        dis = os.path.join(tmpdir, "dis.y4m")
        _make_constant_y4m(ref, w, h, 8, val_ref)
        _make_constant_y4m(dis, w, h, 8, val_dis)
        scores = _run_vmaf(ref, dis, model=None, feature="float_ssim")
        ssim = scores.get("float_ssim", None)
        assert ssim is not None, "SSIM score not found"
        assert 0.0 <= ssim <= 1.0, \
            f"SSIM = {ssim}, expected in [0, 1]"


# ---- P4: Determinism ----

@given(dims=valid_dimensions, val_ref=pixel_value_8bit,
       val_dis=pixel_value_8bit)
@settings(max_examples=10, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
def test_vmaf_deterministic(dims, val_ref, val_dis):
    """Repeated VMAF runs on identical input must produce identical scores."""
    w, h = dims
    with tempfile.TemporaryDirectory() as tmpdir:
        ref = os.path.join(tmpdir, "ref.y4m")
        dis = os.path.join(tmpdir, "dis.y4m")
        _make_constant_y4m(ref, w, h, 8, val_ref)
        _make_constant_y4m(dis, w, h, 8, val_dis)
        score1 = _run_vmaf(ref, dis).get("vmaf")
        score2 = _run_vmaf(ref, dis).get("vmaf")
        assert score1 == score2, \
            f"Non-deterministic: run1={score1}, run2={score2}"


# ---- P5: Thread invariance ----

@given(dims=valid_dimensions, val_ref=pixel_value_8bit,
       val_dis=pixel_value_8bit)
@settings(max_examples=10, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
def test_vmaf_thread_invariant(dims, val_ref, val_dis):
    """VMAF score must be identical regardless of thread count."""
    w, h = dims
    with tempfile.TemporaryDirectory() as tmpdir:
        ref = os.path.join(tmpdir, "ref.y4m")
        dis = os.path.join(tmpdir, "dis.y4m")
        _make_constant_y4m(ref, w, h, 8, val_ref)
        _make_constant_y4m(dis, w, h, 8, val_dis)
        score_1t = _run_vmaf(ref, dis, n_threads=1).get("vmaf")
        score_4t = _run_vmaf(ref, dis, n_threads=4).get("vmaf")
        assert score_1t == score_4t, \
            f"Thread-dependent: 1-thread={score_1t}, 4-thread={score_4t}"


# ---- P6: Integer/float agreement ----

@given(dims=valid_dimensions, val_ref=pixel_value_8bit,
       val_dis=pixel_value_8bit)
@settings(max_examples=10, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
def test_int_float_agreement(dims, val_ref, val_dis):
    """Integer and float feature extractors must agree within tolerance."""
    w, h = dims
    with tempfile.TemporaryDirectory() as tmpdir:
        ref = os.path.join(tmpdir, "ref.y4m")
        dis = os.path.join(tmpdir, "dis.y4m")
        _make_constant_y4m(ref, w, h, 8, val_ref)
        _make_constant_y4m(dis, w, h, 8, val_dis)

        int_scores = _run_vmaf(ref, dis, model="vmaf_v0.6.1")
        float_scores = _run_vmaf(ref, dis, model="vmaf_float_v0.6.1")

        int_vmaf = int_scores.get("vmaf")
        float_vmaf = float_scores.get("vmaf")
        assume(int_vmaf is not None and float_vmaf is not None)

        # Allow up to 2% relative difference between integer and float paths.
        denom = max(abs(int_vmaf), abs(float_vmaf), 1.0)
        rel_err = abs(int_vmaf - float_vmaf) / denom
        assert rel_err < 0.02, \
            f"Int/float divergence: int={int_vmaf}, float={float_vmaf}, " \
            f"rel_err={rel_err:.6f}"
```

### 10.2 Running Property Tests

```bash
# Install hypothesis
pip install hypothesis

# Run property tests
cd /path/to/vmaf
python -m pytest python/test/test_property_fuzz.py -v --tb=short

# Run with more examples (for nightly CI)
HYPOTHESIS_MAX_EXAMPLES=100 python -m pytest python/test/test_property_fuzz.py -v
```

---

## 11. Crash Triage and Deduplication

### 11.1 Crash Output Organization

libFuzzer writes crashing inputs to the current directory with filenames like `crash-<sha1>`, `leak-<sha1>`, `timeout-<sha1>`. Organize these by target:

```bash
mkdir -p crashes/{y4m,json_model,pictures}

# After a fuzzing run, move crashes
mv crash-* leak-* timeout-* crashes/<target_name>/
```

### 11.2 Deduplication Strategy

| Step | Method | Tool |
|------|--------|------|
| 1. Stack hash | Group crashes by unique stack trace hash | ASan provides this automatically |
| 2. Minimize | Reduce crashing input to minimal reproducer | `./fuzz_target -minimize_crash=1 -max_total_time=60 crash-file` |
| 3. Bisect | If regression, find the introducing commit | `git bisect` with the minimized reproducer |
| 4. Classify | Categorize by bug class | Manual review of ASan/UBSan report |

### 11.3 Crash Reproduction

Every crash artifact must be reproducible outside the fuzzer:

```bash
# Reproduce a crash with full ASan output
./build_fuzz/fuzz/fuzz_json_model crashes/json_model/crash-abc123

# Minimize the crash input
./build_fuzz/fuzz/fuzz_json_model \
    -minimize_crash=1 \
    -exact_artifact_path=crashes/json_model/crash-abc123-min \
    -max_total_time=60 \
    crashes/json_model/crash-abc123
```

### 11.4 Bug Severity Classification

| Severity | ASan/UBSan Signal | Example |
|----------|-------------------|---------|
| Critical | heap-buffer-overflow (write), stack-buffer-overflow (write), use-after-free | Overwriting heap metadata in model parser |
| High | heap-buffer-overflow (read), out-of-bounds-index, null-dereference | Reading past chroma buffer in Y4M parser |
| Medium | integer-overflow (signed), shift-exponent | Dimension multiplication overflow in Y4M |
| Low | uninitialized-value (MSan) | Reading uninitialized padding bytes |

---

## 12. Completion Requirements

The fuzzing infrastructure is **complete** when ALL of the following requirements are met:

### R1. All Three Fuzz Targets Implemented

Fuzz harness source files exist for F1 (Y4M parser), F2 (JSON model loader), and F3 (picture read pipeline). Each harness compiles and runs without errors under libFuzzer.

**Verification:** `ninja -C build_fuzz` succeeds and all three binaries exist. Running each with an empty corpus for 10 seconds produces no build or runtime errors.

### R2. Seed Corpora Populated

Each fuzz target has a non-empty seed corpus:
- F1: at least one valid Y4M file per supported chroma format (minimum 10 seeds)
- F2: at least 3 model JSON files from the repository plus 1 minimal synthetic seed
- F3: at least 4 synthetic structured seeds covering different dimension/bpc combinations

**Verification:** `ls corpus/<target>/` shows the expected file count for each target.

### R3. Dictionaries Provided

F1 and F2 have dictionary files containing tokens relevant to their input formats.

**Verification:** Dictionary files exist and contain at least 10 tokens each.

### R4. ASan and UBSan Active

All fuzz targets are compiled with `-fsanitize=address,undefined` and `-fno-sanitize-recover=all`. The sanitizers are active during fuzzing (verified by testing a known-bad input).

**Verification:** Intentionally trigger a buffer overflow in a test harness and confirm ASan reports it.

### R5. No Crashes in 1-Hour Run

Each fuzz target runs for at least 1 hour on a single core with its seed corpus and dictionary without finding any crashes, leaks, or timeouts.

**Verification:** libFuzzer `print_final_stats` shows 0 crashes after a 3600-second run for each target.

### R6. Coverage Thresholds Met

After a 1-hour fuzzing run, source-level coverage meets the targets in Section 9.3:
- F1: >= 80% line coverage of `y4m_input.c`
- F2: >= 90% line coverage of `read_json_model.c`
- F3: >= 50% of feature extractor source lines exercised

**Verification:** `llvm-cov report` output shows coverage percentages meeting or exceeding thresholds.

### R7. All Six Property Tests Pass

The Python Hypothesis property tests (P1-P6) pass with their default `max_examples` setting.

**Verification:** `python -m pytest python/test/test_property_fuzz.py -v` exits with 0 failures.

### R8. CI Integration Active

The GitHub Actions fuzzing workflow runs on every push to master and every pull request. Nightly scheduled runs execute for at least 1 hour per target.

**Verification:** The workflow file exists, is syntactically valid, and has successfully run on at least one CI execution.

### R9. Meson Build Option Works

The `enable_fuzz` meson option defaults to `false` and does not affect production builds. When set to `true` with clang, all fuzz targets build successfully.

**Verification:** `meson setup build_prod libvmaf` (without `-Denable_fuzz`) produces no fuzz-related build artifacts. `meson setup build_fuzz libvmaf -Denable_fuzz=true` with `CC=clang` produces all three fuzz binaries.

### R10. Crash Triage Process Documented and Functional

The crash minimization and reproduction commands work on a synthetic crash. The deduplication strategy is documented and the crash classification table is referenced in CI failure notifications.

**Verification:** A synthetic test crash can be minimized with `minimize_crash=1` and reproduced outside the fuzzer.

### R11. AFL++ Compatibility

The fuzz harness source files compile and run under AFL++ persistent mode using `afl-clang-fast` with the provided wrapper.

**Verification:** At least one target (F2) builds and runs with AFL++ for 60 seconds without errors.

### R12. Zero Production Code Changes

The fuzzing infrastructure does not modify any existing source files in `libvmaf/src/` or `libvmaf/tools/`. All fuzz targets, corpora, and dictionaries reside under `libvmaf/fuzz/`. The only changes to existing files are the new meson option and the `subdir('fuzz')` inclusion.

**Verification:** `git diff --name-only` on production source files shows no changes beyond `meson_options.txt` and the top-level `meson.build`.

---

## 13. Acceptance Criteria Summary

| Req | Description | Pass Condition |
|-----|-------------|----------------|
| R1 | All fuzz targets implemented | 3/3 harnesses compile and run under libFuzzer |
| R2 | Seed corpora populated | >= 10 Y4M seeds, >= 4 JSON seeds, >= 4 picture seeds |
| R3 | Dictionaries provided | y4m.dict and json_model.dict exist with >= 10 tokens each |
| R4 | Sanitizers active | ASan + UBSan detect injected buffer overflow |
| R5 | No crashes in 1-hour run | 0 crashes/leaks/timeouts per target after 3600s |
| R6 | Coverage thresholds met | F1 >= 80%, F2 >= 90%, F3 >= 50% line coverage |
| R7 | Property tests pass | 6/6 Hypothesis tests pass |
| R8 | CI integration active | GitHub Actions workflow runs on push and nightly |
| R9 | Meson option works | `enable_fuzz=false` is no-op; `=true` builds all targets |
| R10 | Crash triage functional | Minimize and reproduce commands work on synthetic crash |
| R11 | AFL++ compatible | At least 1 target runs under AFL++ persistent mode |
| R12 | Zero production changes | No modifications to existing `src/` or `tools/` files |

All 12 requirements must be met for the fuzzing infrastructure to be considered complete.
