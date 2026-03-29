# Fuzzing Infrastructure -- Implementation Notes

## Files Created

### Fuzz Target Harnesses

| File | Description |
|------|-------------|
| `libvmaf/fuzz/fuzz_y4m_parser.c` | F1: libFuzzer harness for the Y4M parser. Feeds arbitrary bytes to `Y4M_INPUT_VTBL.open()` via `fmemopen`, then exercises `get_info` and `fetch_frame`. |
| `libvmaf/fuzz/fuzz_json_model.c` | F2: libFuzzer harness for the JSON model loader. Feeds arbitrary bytes to `vmaf_read_json_model_from_buffer()`. |
| `libvmaf/fuzz/fuzz_read_pictures.c` | F3: libFuzzer harness for the picture scoring pipeline. Uses structured fuzzing: first 2 bytes select dimensions and bit depth; remaining bytes fill distorted pixel data. |
| `libvmaf/fuzz/afl_persistent_wrapper.c` | AFL++ persistent mode adapter wrapping the `LLVMFuzzerTestOneInput` interface. |

### Build System

| File | Description |
|------|-------------|
| `libvmaf/fuzz/meson.build` | Meson build file for all three fuzz targets. Guarded by `enable_fuzz` option. |
| `libvmaf/meson_options.txt` | Added `enable_fuzz` boolean option (default: `false`). |
| `libvmaf/meson.build` | Added `subdir('fuzz')` at the end. |

### Seed Corpora

| Directory | Contents |
|-----------|----------|
| `libvmaf/fuzz/corpus/y4m/` | 15 minimal 2x2 Y4M files, one per supported chroma format (420, 420jpeg, 420mpeg2, 420p10, 420p12, 420paldv, 422, 422p10, 422p12, 444, 444p10, 444p12, 444alpha, 411, mono). |
| `libvmaf/fuzz/corpus/json_model/` | 4 model JSON seeds: 3 from the repository (`vmaf_v0.6.1.json`, `vmaf_v0.6.1neg.json`, `vmaf_b_v0.6.3.json`) plus 1 minimal synthetic seed (`seed_minimal.json`). |
| `libvmaf/fuzz/corpus/pictures/` | 8 structured binary seeds covering 4 dimension sets (64x64, 128x128, 192x108, 576x324) at both 8-bit and 10-bit depths. |

### Seed Generation Scripts

| File | Description |
|------|-------------|
| `libvmaf/fuzz/generate_y4m_seeds.sh` | Bash script to regenerate Y4M seed corpus. |
| `libvmaf/fuzz/generate_picture_seeds.py` | Python script to regenerate picture seed corpus. |

### Dictionaries

| File | Tokens |
|------|--------|
| `libvmaf/fuzz/dict/y4m.dict` | 24 tokens: Y4M header keywords and chroma type identifiers. |
| `libvmaf/fuzz/dict/json_model.dict` | 30 tokens: JSON model field names and SVM model keywords. |

### CI Integration

| File | Description |
|------|-------------|
| `.github/workflows/fuzz.yml` | GitHub Actions workflow running all three fuzz targets. Runs for 120s on push/PR and 3600s on nightly schedule. Uploads crash artifacts on failure. |

### Property-Based Tests

| File | Description |
|------|-------------|
| `python/test/test_property_fuzz.py` | Six Hypothesis property-based tests (P1-P6) covering: identity VMAF score, identity PSNR, SSIM bounds, determinism, thread invariance, and integer/float agreement. |

## How to Build Fuzz Targets

### Prerequisites

- Clang compiler with libFuzzer support
- Meson >= 0.56.1 and Ninja
- NASM assembler

### libFuzzer Build

```bash
CC=clang CXX=clang++ meson setup build_fuzz libvmaf \
    -Denable_fuzz=true \
    -Denable_tests=false \
    -Db_sanitize=address,undefined \
    -Db_lundef=false \
    --default-library=static

ninja -C build_fuzz
```

The three fuzz binaries will be at:
- `build_fuzz/fuzz/fuzz_y4m_parser`
- `build_fuzz/fuzz/fuzz_json_model`
- `build_fuzz/fuzz/fuzz_read_pictures`

### AFL++ Build

```bash
export CC=afl-clang-fast
export CXX=afl-clang-fast++

cd libvmaf
meson setup build_afl --default-library=static \
    -Denable_tests=false -Denable_docs=false
ninja -C build_afl

# Compile a target with AFL++ persistent mode wrapper
$CC -g -O1 -fsanitize=address,undefined \
    -I include -I tools \
    fuzz/fuzz_y4m_parser.c fuzz/afl_persistent_wrapper.c \
    tools/y4m_input.c \
    -o build_afl/fuzz_y4m_parser_afl \
    -Lbuild_afl -lvmaf -lm -lpthread
```

## How to Run Fuzz Targets

### Y4M Parser Fuzzer

```bash
./build_fuzz/fuzz/fuzz_y4m_parser \
    -dict=libvmaf/fuzz/dict/y4m.dict \
    -max_len=65536 \
    -timeout=10 \
    -rss_limit_mb=2048 \
    libvmaf/fuzz/corpus/y4m/
```

### JSON Model Fuzzer

```bash
./build_fuzz/fuzz/fuzz_json_model \
    -dict=libvmaf/fuzz/dict/json_model.dict \
    -max_len=1048576 \
    -timeout=30 \
    -rss_limit_mb=2048 \
    libvmaf/fuzz/corpus/json_model/
```

### Picture Pipeline Fuzzer

```bash
./build_fuzz/fuzz/fuzz_read_pictures \
    -max_len=2097152 \
    -timeout=60 \
    -rss_limit_mb=4096 \
    libvmaf/fuzz/corpus/pictures/
```

## How to Use Property-Based Tests

### Install Dependencies

```bash
pip install hypothesis
```

### Run Tests

```bash
cd /path/to/vmaf
python -m pytest python/test/test_property_fuzz.py -v --tb=short
```

### Run with More Examples (Nightly CI)

```bash
HYPOTHESIS_MAX_EXAMPLES=100 python -m pytest python/test/test_property_fuzz.py -v
```

## Crash Triage

### Reproducing a Crash

```bash
# Reproduce with full ASan output
./build_fuzz/fuzz/fuzz_json_model crash-<hash>

# Minimize a crashing input
./build_fuzz/fuzz/fuzz_json_model \
    -minimize_crash=1 \
    -exact_artifact_path=crash-minimized \
    -max_total_time=60 \
    crash-<hash>
```

### Severity Classification

| Severity | ASan/UBSan Signal |
|----------|-------------------|
| Critical | heap-buffer-overflow (write), use-after-free |
| High | heap-buffer-overflow (read), null-dereference |
| Medium | integer-overflow (signed), shift-exponent |
| Low | uninitialized-value (MSan) |

## Production Code Changes

The only changes to existing files are:
- `libvmaf/meson_options.txt`: Added `enable_fuzz` option
- `libvmaf/meson.build`: Added `subdir('fuzz')` line

No production source code in `libvmaf/src/` or `libvmaf/tools/` was modified.
