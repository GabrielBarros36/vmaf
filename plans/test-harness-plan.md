# VMAF Test Harness — Comprehensive Plan

## 1. Current State of Testing

### 1.1 Test Infrastructure Overview

| Dimension | What Exists |
|-----------|-------------|
| **C unit tests** | 22 test files, ~83 test functions, using a minimal custom framework (Minunit-style) |
| **Python tests** | 48 test files, ~407 test methods, using pytest + unittest |
| **CI/CD** | GitHub Actions: 4 workflows (libvmaf, windows, ffmpeg, docker) |
| **Platforms in CI** | Ubuntu x86_64, Ubuntu ARM64, macOS x86_64, Windows x86_64 |
| **Compilers in CI** | GCC (default), GCC-9, Clang |
| **Test data** | ~18 YUV video files (576x324 primary), synthetic patterns (flat, checkerboard), on-demand downloads from Netflix/vmaf_resource |
| **Models tested** | 17+ JSON model files including float, integer, neg, 4K, and ensemble variants |

### 1.2 Algorithm-by-Algorithm Test Coverage

| Algorithm | C Unit Tests | Python Integration | Golden Values | SIMD Tested | Edge Cases | Overall |
|-----------|:---:|:---:|:---:|:---:|:---:|:---:|
| **CAMBI** | 18 tests | Yes | Yes | Indirect only | Good | Excellent |
| **CIEDE2000** | 4 tests | Yes | Yes | N/A | Fair | Good |
| **ADM** | None | Extensive | Yes | Indirect only | Fair | Good (integration only) |
| **VIF** | None | Extensive | Yes | Indirect only | Fair | Good (integration only) |
| **PSNR** | 1 test | Yes | Yes | N/A | Poor | Fair |
| **SSIM/MS-SSIM** | None | Yes | Yes | N/A | Poor | Fair |
| **Motion** | None | Yes | Yes | Indirect only | Poor | Fair |
| **ANSNR** | None | Basic | Basic | N/A | Poor | Poor |

### 1.3 SIMD Coverage Matrix

Every SIMD implementation is tested only indirectly. CI runs on the right hardware so dispatch selects AVX2/NEON, but no test explicitly validates SIMD output against the C reference.

| Function | C Ref | AVX2 | AVX-512 | NEON | Direct comparison test? |
|----------|:---:|:---:|:---:|:---:|:---:|
| ADM dwt2_8 | Yes | Yes | — | Yes | **No** |
| ADM dwt2_16 | Yes | — | — | — | N/A |
| VIF subsample_rd_8 | Yes | Yes | Yes | Yes | **No** |
| VIF subsample_rd_16 | Yes | Yes | Yes | Yes | **No** |
| VIF statistic_8 | Yes | Yes | Yes | Yes | **No** |
| VIF statistic_16 | Yes | Yes | Yes | Yes | **No** |
| Motion x_conv_16 | Yes | Yes | Yes | — | **No** |
| CAMBI ops | Yes | Yes | — | — | **No** |
| Convolution (float) | Yes | AVX | — | — | **No** |

### 1.4 CI/CD Gaps

| What's Present | What's Missing |
|---|---|
| Release builds on 3 OSes | No debug builds |
| GCC + Clang | No sanitizers (ASan, UBSan, MSan, TSan) |
| x86_64 + ARM64 runners | No 32-bit (disabled due to score mismatch — itself a bug signal) |
| Basic `ninja test` | No fuzzing |
| Python tox on Ubuntu | No code coverage enforcement |
| FFmpeg integration (continue-on-error) | No performance regression tracking |
| Docker build | No CUDA/GPU testing |

---

## 2. Gap Analysis — What Could Go Wrong Today Undetected

### 2.1 Silent SIMD Divergence (Critical)

The biggest risk: a SIMD implementation (AVX2, AVX-512, NEON) produces slightly different results from the C reference, and no test catches it because CI only runs on one architecture at a time, golden values were captured from whatever path ran on the CI machine, and there is no test that forces C-reference execution alongside SIMD execution for comparison.

Evidence this is real: the 32-bit Windows build is disabled with the comment "Disabled 32-bit job due to vmaf score mismatch" — demonstrating that architecture-dependent numerical discrepancies already exist.

### 2.2 Untested Width/Alignment Edge Cases (High)

SIMD dispatch has width constraints (must be multiple of 8). If width is not a multiple of 8, the C fallback runs. But no test verifies the fallback path actually activates correctly, no test exercises width=7, width=9, width=15 to confirm the boundary, and alignment is set to 32 bytes but no test verifies behavior on misaligned data.

### 2.3 Bit-Depth Boundary Conditions (High)

ADM 16-bit has no SIMD at all — performance cliff on HDR content, and limited testing. Integer overflow potential in 32-bit accumulators on high bit-depth content is not stress-tested. Pack-with-saturation (`_mm256_packus_epi32`) could silently clamp values if intermediate results exceed uint16 range.

### 2.4 Degenerate Input Handling (Medium)

No tests for: all-black or all-white frames (division-by-zero in VIF/ADM denominators), identical ref and distorted (perfect score edge), single-pixel or 1x1 images, extremely large resolutions (4K/8K), zero-frame or single-frame sequences for temporal metrics (Motion).

### 2.5 Memory Safety (Medium)

No sanitizer builds means buffer overreads in SIMD tail handling go undetected, use-after-free in multi-threaded frame sync is not caught, and integer overflow UB in C code is not flagged.

---

## 3. Recommendations

### 3.1 SIMD Correctness Oracle Tests (Priority: Critical)

See [plans/specs/SIMD-correctness-oracle.md](specs/SIMD-correctness-oracle.md) for full specification.

Create a test harness that, for each SIMD function:
1. Allocates a test image (synthetic + random data)
2. Runs the C reference implementation directly (bypassing dispatch)
3. Runs the SIMD implementation directly
4. Asserts bit-exact equality (for integer paths) or ULP tolerance (for float paths)

Covers every function in the dispatch table, multiple image sizes (including SIMD boundary widths), multiple bit depths, and edge pixel mirror-boundary handling.

### 3.2 Sanitizer CI Jobs (Priority: Critical)

Add to GitHub Actions matrix:

| Sanitizer | Catches | Build Flags |
|-----------|---------|-------------|
| **ASan** | Buffer overflows, use-after-free, stack overflow | `-fsanitize=address` |
| **UBSan** | Integer overflow, shift UB, null deref, alignment | `-fsanitize=undefined` |
| **MSan** | Uninitialized memory reads | `-fsanitize=memory` (Clang only) |
| **TSan** | Data races in thread pool / frame sync | `-fsanitize=thread` |

Run full test suite under each. This is the single highest-value addition alongside SIMD oracle tests — it catches entire classes of bugs automatically.

### 3.3 Degenerate Input Test Suite (Priority: High)

Create a synthetic test data generator that produces:

| Test Pattern | Purpose |
|---|---|
| All-zero (black) frame | Division-by-zero in VIF/ADM denominators |
| All-max (white) frame | Saturation handling, overflow |
| Identical ref=dis | Perfect score boundary (VMAF=100, PSNR=inf) |
| Single-pixel difference | Minimum detectable distortion |
| 1x1, 2x2, 3x3 images | Minimum dimension handling |
| Width=7, 9, 15, 17 | SIMD dispatch boundary (multiple-of-8 threshold) |
| Width=4096, 7680 | 4K/8K stress test |
| Gradient ramp (0-max) | Full dynamic range |
| Random noise | Statistical stability |
| Checkerboard at Nyquist | Worst-case for downsampling/filtering |
| Single frame | Temporal metric edge case (Motion) |

### 3.4 Cross-Architecture Golden Value Tests (Priority: High)

1. Compute VMAF/VIF/ADM/PSNR/SSIM/Motion scores on the test videos using C reference only (no SIMD)
2. Store these as canonical golden values in the repo
3. On every CI platform, run the same computation (with SIMD enabled) and assert results match the canonical values within a defined tolerance (0 for integer paths, <1 ULP for float)
4. If a platform diverges, the test fails explicitly — no silent drift

### 3.5 Fuzzing Infrastructure (Priority: High)

Two complementary approaches:

**Structure-aware fuzzing** with libFuzzer/AFL++:
- Fuzz the Y4M/YUV input parser
- Fuzz model JSON loading
- Fuzz the `vmaf_read_pictures` API with random pixel data
- Target: crashes, hangs, memory corruption

**Property-based testing** (Python hypothesis or C-level):
- `VMAF(ref, ref) == 100.0` for any valid input
- `PSNR(ref, ref) == infinity` for any valid input
- `0 <= SSIM(ref, dis) <= 1` for any valid inputs
- `VMAF(ref, dis)` is deterministic (same input, same output)
- Score is invariant to thread count
- Integer and float implementations agree within tolerance

### 3.6 Numerical Precision Differential Tests (Priority: Medium)

For every metric that has both float and integer implementations (ADM, VIF, PSNR, Motion, SSIM):
1. Run both on the same input
2. Assert they agree within a documented tolerance
3. Track the maximum observed delta across the test corpus
4. Alert if the delta exceeds historical bounds

### 3.7 Build Configuration Matrix Expansion (Priority: Medium)

| Configuration | Value |
|---|---|
| Debug build | Catches assert failures, enables debug-only checks |
| LTO build | Catches ODR violations, cross-TU UB |
| `-Denable_asm=false` | Forces C reference path — validates fallback works |
| `-Denable_avx512=false` | Tests AVX2-only path on AVX-512 capable machines |
| `-Denable_float=true/false` | Both float and integer codepaths |
| 32-bit build | Fix the known score mismatch instead of disabling |
| Older compilers | GCC-7, GCC-8 to catch standards-compliance issues |
| `-O0` build | Catches optimized-only bugs (UB that works at -O2 but fails at -O0) |

### 3.8 Thread Safety and Concurrency Tests (Priority: Medium)

- Run VMAF scoring with 1, 2, 4, 8, 16 threads and verify identical results
- Stress-test frame synchronization with out-of-order frame submission
- Test `VmafContext` teardown while scoring is in-flight
- Test concurrent `VmafContext` instances sharing models

### 3.9 Performance Regression Tests (Priority: Low-Medium)

- Benchmark each SIMD path against the C reference on a fixed input
- Store timing baselines per platform
- Alert on >10% regression (catches accidental fallback to C path)
- Use `google/benchmark` or similar framework

### 3.10 Model Validation Tests (Priority: Low-Medium)

- For each shipped model file, verify it loads successfully
- Verify the feature set it requires matches available feature extractors
- Run a round-trip: score, check score is in valid range [0, 100]
- Test model version compatibility (older models on newer library)

### 3.11 Format and Codec Coverage (Priority: Low)

- Add Y4M format tests (header parsing edge cases)
- Test YUV444p color space (currently untested)
- Test limited vs. full range YUV handling
- Test non-standard strides / padded rows

---

## 4. Summary Scorecard

| Testing Dimension | Current State | Target State | Gap |
|---|---|---|---|
| Algorithm correctness | Golden values in Python | + C unit tests per algorithm | Medium |
| SIMD vs. C reference | Not tested | Bit-exact comparison per function | **Critical** |
| Architecture coverage | x86_64 + ARM64 in CI | + 32-bit fix, forced-fallback testing | High |
| Memory safety | None | ASan + UBSan + MSan + TSan in CI | **Critical** |
| Degenerate inputs | Minimal | Comprehensive synthetic suite | High |
| Bit-depth coverage | 8/10/12-bit in Python | + 16-bit, per-algorithm C tests | Medium |
| Concurrency | 1 framesync test | Thread count sweep, stress tests | Medium |
| Fuzzing | None | libFuzzer targets for parsers + API | High |
| Performance regression | None | Per-platform benchmarks in CI | Low-Medium |
| Float vs. Integer parity | Not validated | Tolerance-bounded comparison | Medium |
| Build config diversity | Release only | + Debug, sanitizers, LTO, ASM-off | High |
