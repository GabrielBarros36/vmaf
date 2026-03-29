# SIMD Correctness Oracle Tests — Specification

## 1. Objective

Validate that every SIMD-optimized function in libvmaf produces **bit-exact** output (integer paths) or **bounded-tolerance** output (float paths) relative to the C reference implementation, across all supported input dimensions, bit depths, and data patterns.

This test suite must catch any divergence introduced by architecture-specific code before it reaches production.

---

## 2. Scope — Functions Under Test

There are **9 dispatch points** in 4 modules. Each must be tested.

### 2.1 ADM Module

| # | C Reference | AVX2 | AVX-512 | NEON | Struct Field | Width Constraint |
|---|-------------|------|---------|------|--------------|------------------|
| 1 | `adm_dwt2_8` | `adm_dwt2_8_avx2` | — | `adm_dwt2_8_neon` | `dwt2_8` | `w % 8 == 0` |

**Signature:**
```c
void adm_dwt2_8(const uint8_t *src, const adm_dwt_band_t *dst,
                AdmBuffer *buf, int w, int h, int src_stride, int dst_stride);
```

**Output to compare:** The four `int16_t` band arrays in `adm_dwt_band_t` (`band_a`, `band_v`, `band_h`, `band_d`), each of size `(w/2) * (h/2)`.

### 2.2 VIF Module

| # | C Reference | AVX2 | AVX-512 | NEON | Struct Field | Width Constraint |
|---|-------------|------|---------|------|--------------|------------------|
| 2 | `subsample_rd_8` | `vif_subsample_rd_8_avx2` | `vif_subsample_rd_8_avx512` | `vif_subsample_rd_8_neon` | `subsample_rd_8` | None |
| 3 | `subsample_rd_16` | `vif_subsample_rd_16_avx2` | `vif_subsample_rd_16_avx512` | `vif_subsample_rd_16_neon` | `subsample_rd_16` | None |
| 4 | `vif_statistic_8` | `vif_statistic_8_avx2` | `vif_statistic_8_avx512` | `vif_statistic_8_neon` | `vif_statistic_8` | None |
| 5 | `vif_statistic_16` | `vif_statistic_16_avx2` | `vif_statistic_16_avx512` | `vif_statistic_16_neon` | `vif_statistic_16` | None |

**Signatures:**
```c
void subsample_rd_8(VifBuffer buf, unsigned w, unsigned h);
void subsample_rd_16(VifBuffer buf, unsigned w, unsigned h, int scale, int bpc);
void vif_statistic_8(VifPublicState *s, float *num, float *den, unsigned w, unsigned h);
void vif_statistic_16(VifPublicState *s, float *num, float *den, unsigned w, unsigned h, int bpc, int scale);
```

**Output to compare:**
- `subsample_rd_*`: The downsampled buffer contents in `VifBuffer` (mu1, mu2, ref_sq, dis_sq, ref_dis fields).
- `vif_statistic_*`: The `float *num` and `float *den` output values.

### 2.3 Motion Module

| # | C Reference | AVX2 | AVX-512 | NEON | Struct Field | Width Constraint |
|---|-------------|------|---------|------|--------------|------------------|
| 6 | `x_convolution_16` | `x_convolution_16_avx2` | `x_convolution_16_avx512` | — | `x_convolution` | None |

**Signature:**
```c
void x_convolution_16(const uint16_t *src, uint16_t *dst, unsigned width,
                      unsigned height, ptrdiff_t src_stride, ptrdiff_t dst_stride);
```

**Output to compare:** The entire `dst` buffer of size `width * height` (uint16_t values).

### 2.4 CAMBI Module

| # | C Reference | AVX2 | AVX-512 | NEON | Struct Field | Width Constraint |
|---|-------------|------|---------|------|--------------|------------------|
| 7 | `increment_range` | `cambi_increment_range_avx2` | — | — | `inc_range_callback` | None |
| 8 | `decrement_range` | `cambi_decrement_range_avx2` | — | — | `dec_range_callback` | None |
| 9 | `get_derivative_data_for_row` | `get_derivative_data_for_row_avx2` | — | — | `derivative_callback` | None |

**Signatures:**
```c
void increment_range(uint16_t *arr, int left, int right);
void decrement_range(uint16_t *arr, int left, int right);
void get_derivative_data_for_row(const uint16_t *image_data, uint16_t *derivative_buffer,
                                 int width, int height, int row, int stride);
```

**Output to compare:**
- `increment_range` / `decrement_range`: The `uint16_t *arr` contents over `[left, right]`.
- `get_derivative_data_for_row`: The entire `derivative_buffer` for the given row.

---

## 3. Test Architecture

### 3.1 Design Principle: Direct Function Pointer Invocation

Tests must **not** rely on runtime CPU dispatch. Instead, each test must:

1. Call the C reference function directly by its symbol name.
2. Call the SIMD variant function directly by its symbol name.
3. Compare outputs.

This requires the SIMD functions to be **externally visible** (non-static, declared in headers). Currently, the AVX2/AVX-512/NEON functions are already declared in their respective headers (`x86/adm_avx2.h`, `x86/vif_avx2.h`, etc.) and are externally linked. The C reference functions for ADM, VIF subsample, and Motion x_convolution are `static` in their respective `.c` files and will need to be exposed for testing.

### 3.2 Exposing C Reference Functions

**Approach:** Add a compile-time flag `-DVMAF_SIMD_ORACLE_TEST` that makes the C reference functions non-static and exposes them through a test header.

Create a new header `libvmaf/src/feature/simd_oracle_export.h`:
```c
#ifdef VMAF_SIMD_ORACLE_TEST
#define SIMD_ORACLE_STATIC
#else
#define SIMD_ORACLE_STATIC static
#endif
```

Replace `static` with `SIMD_ORACLE_STATIC` on the 6 reference functions:
- `adm_dwt2_8` in `integer_adm.c`
- `subsample_rd_8` in `integer_vif.c`
- `subsample_rd_16` in `integer_vif.c`
- `vif_statistic_8` in `integer_vif.c` (already non-static)
- `vif_statistic_16` in `integer_vif.c` (already non-static)
- `x_convolution_16` in `integer_motion.c`
- `increment_range` in `cambi.c`
- `decrement_range` in `cambi.c`
- `get_derivative_data_for_row` in `cambi.c`

**Alternative approach (zero production impact):** Extract the C reference function bodies into standalone `.c` files that are compiled only for the test target. This avoids modifying production code entirely.

### 3.3 Test File Organization

```
libvmaf/test/
  test_simd_adm.c          # ADM oracle tests
  test_simd_vif.c           # VIF oracle tests
  test_simd_motion.c        # Motion oracle tests
  test_simd_cambi.c         # CAMBI oracle tests
  test_simd_common.h        # Shared test data generators and comparison utilities
```

Each test file is a standalone executable following the existing Minunit pattern (`test.h`).

### 3.4 Conditional Compilation

Tests must compile on all architectures. Each test file wraps SIMD-specific tests in architecture guards:

```c
#if ARCH_X86
static char *test_adm_dwt2_8_avx2_vs_c(void) { ... }
#endif
#if ARCH_AARCH64
static char *test_adm_dwt2_8_neon_vs_c(void) { ... }
#endif

static char *run_tests(void) {
#if ARCH_X86
    mu_run_test(test_adm_dwt2_8_avx2_vs_c);
#endif
#if ARCH_AARCH64
    mu_run_test(test_adm_dwt2_8_neon_vs_c);
#endif
    return NULL;
}
```

---

## 4. Test Input Generation

### 4.1 Input Data Categories

Every dispatch function must be tested with **all** of the following input categories:

| Category | Description | Purpose |
|----------|-------------|---------|
| **Zero** | All pixels = 0 | Division-by-zero, zero-accumulator behavior |
| **Max** | All pixels = max value (255 for 8-bit, 65535 for 16-bit) | Overflow / saturation in multiply-accumulate |
| **Constant** | All pixels = 128 (8-bit) or 32768 (16-bit) | Mid-range constant — verifies filter DC response |
| **Gradient-H** | Horizontal ramp 0..max across width | Exercises all pixel values; detects byte-lane issues |
| **Gradient-V** | Vertical ramp 0..max across height | Exercises stride handling |
| **Checkerboard** | Alternating 0 and max | Worst-case high-frequency; stresses filter taps |
| **Random (seeded)** | Deterministic PRNG (seed=42) filling the buffer | General correctness; reproducible failures |
| **Random (seeded, alternate)** | Deterministic PRNG (seed=123) filling the buffer | Second random pattern to avoid seed-specific coincidences |
| **Single-hot** | One pixel = max, rest = 0 | Impulse response; catches off-by-one in indexing |
| **Boundary-stripe** | First and last row/column = max, rest = 0 | Edge/mirror handling in convolution filters |

### 4.2 Input Dimensions

Each function must be tested at every dimension in this set:

| Width | Height | Rationale |
|-------|--------|-----------|
| 8 | 8 | Minimum for SIMD (ADM constraint) |
| 16 | 16 | Two SIMD register widths |
| 24 | 24 | Three SIMD register widths (non-power-of-2) |
| 32 | 32 | Full AVX-512 width |
| 64 | 64 | Multiple cache lines |
| 120 | 68 | Non-power-of-2, not multiple of 16 but multiple of 8 |
| 576 | 324 | Matches primary test video resolution |
| 1920 | 1080 | Full HD |

For ADM (width must be multiple of 8), additionally test:
| Width | Height | Rationale |
|-------|--------|-----------|
| 7 | 8 | Below SIMD threshold — must fall back to C reference |
| 9 | 8 | Just above SIMD threshold — falls back to C reference |
| 15 | 8 | Odd, non-multiple of 8 — C reference path |

These fallback-path dimensions verify that dispatch correctly routes to the C reference when the width constraint is not met, and that the C reference produces correct output for those sizes.

### 4.3 Bit Depths (where applicable)

- `subsample_rd_16` and `vif_statistic_16`: test with `bpc` = 10, 12, 16
- `subsample_rd_16`: test with `scale` = 0, 1, 2, 3
- `vif_statistic_16`: test with `scale` = 0, 1, 2, 3

### 4.4 Memory Allocation Requirements

All input and output buffers must be:
- Allocated with **32-byte alignment** (matching `VMAF_ALIGNMENT` from `alignment.h`)
- Sized with stride padded to the next 32-byte boundary beyond the row width
- Initialized to a known fill pattern (0xDE) before each test to detect under-writes

---

## 5. Comparison Criteria

### 5.1 Integer Paths (ADM, VIF subsample_rd, Motion, CAMBI)

**Requirement: Bit-exact match.**

All integer SIMD implementations use identical arithmetic operations (add, multiply, shift) as the C reference. The SIMD paths use the same shift amounts and rounding constants. Therefore, outputs must be identical byte-for-byte.

```c
mu_assert("SIMD output diverges from C reference",
          memcmp(output_c, output_simd, output_size) == 0);
```

If a bit-exact match fails, the test must report:
- The first differing offset (byte position)
- The C reference value at that position
- The SIMD value at that position
- The input dimensions and data category that triggered the failure

### 5.2 Float Paths (VIF vif_statistic num/den)

**Requirement: Tolerance-bounded match.**

The `vif_statistic_*` functions produce `float *num` and `float *den` output values. Due to non-associative floating-point addition (SIMD processes multiple elements simultaneously with different accumulation order), results may differ in the least significant bits.

**Tolerance:** Maximum relative error of **1e-6** (1 part per million), or absolute error of **1e-9** when the reference value is near zero.

```c
static int float_eq(float a, float b) {
    if (fabsf(a) < 1e-9f && fabsf(b) < 1e-9f) return 1;  // both near zero
    return fabsf(a - b) / fmaxf(fabsf(a), fabsf(b)) < 1e-6f;
}
```

If a tolerance check fails, the test must report:
- The C reference value
- The SIMD value
- The relative error
- The input dimensions and data category

### 5.3 Buffer Overrun Detection

After each SIMD function call, verify that memory **outside** the expected output region was not modified. Achieve this by:

1. Allocating output buffers with 64-byte guard regions before and after the data region.
2. Filling guard regions with a sentinel value (0xAA).
3. After the SIMD call, asserting guard regions are unchanged.

```c
mu_assert("SIMD wrote before output buffer", guard_before_intact(output_buf));
mu_assert("SIMD wrote past output buffer", guard_after_intact(output_buf));
```

---

## 6. Test Execution Matrix

Each test function runs the full cross-product of:

```
for each input_category in [zero, max, constant, gradient_h, gradient_v,
                             checkerboard, random_42, random_123,
                             single_hot, boundary_stripe]:
    for each (width, height) in dimension_set:
        for each bit_depth in applicable_bpc_set:
            1. Generate input data
            2. Allocate two independent output buffers (with guard regions)
            3. Call C reference -> output_c
            4. Call SIMD variant -> output_simd
            5. Assert comparison criteria (Section 5)
            6. Assert guard region integrity
```

### 6.1 Total Test Points

| Module | Functions | Input Categories | Dimensions | Bit Depths | Total Comparisons |
|--------|-----------|-----------------|------------|------------|-------------------|
| ADM | 1 | 10 | 11 | 1 | 110 |
| VIF subsample | 2 | 10 | 8 | 1 (8-bit) + 3x4 (16-bit) | 80 + 960 = 1,040 |
| VIF statistic | 2 | 10 | 8 | 1 (8-bit) + 3x4 (16-bit) | 80 + 960 = 1,040 |
| Motion | 1 | 10 | 8 | 1 | 80 |
| CAMBI | 3 | 10 | 8 | 1 | 240 |

**Total per architecture: ~2,510 comparison points.**

Multiply by the number of SIMD variants available on the platform (e.g., on x86-64 with AVX-512: both AVX2 and AVX-512 variants are tested).

---

## 7. Integration with Build System

### 7.1 Meson Configuration

Add to `libvmaf/test/meson.build`:

```meson
test_simd_common_sources = files('test_simd_common.h')

if host_machine.cpu_family().startswith('x86')
    simd_test_deps = [libvmaf_cpu_avx2_static_lib]
    if get_option('enable_avx512')
        simd_test_deps += [libvmaf_cpu_avx512_static_lib]
    endif
elif host_machine.cpu_family() == 'aarch64'
    simd_test_deps = [libvmaf_cpu_neon_static_lib]
endif

simd_oracle_tests = {
    'test_simd_adm':    files('test_simd_adm.c'),
    'test_simd_vif':    files('test_simd_vif.c'),
    'test_simd_motion': files('test_simd_motion.c'),
    'test_simd_cambi':  files('test_simd_cambi.c'),
}

foreach name, src : simd_oracle_tests
    exe = executable(name, src, test.c_src,
        c_args: ['-DVMAF_SIMD_ORACLE_TEST'],
        include_directories: [libvmaf_inc, test_inc],
        link_with: [libvmaf_static_lib] + simd_test_deps,
    )
    test(name, exe)
endforeach
```

### 7.2 CI Configuration

The SIMD oracle tests run automatically as part of `ninja test` on every CI platform:
- **Ubuntu x86_64:** Tests AVX2 variants (and AVX-512 if the runner supports it)
- **Ubuntu ARM64:** Tests NEON variants
- **macOS x86_64:** Tests AVX2 variants
- **Windows x86_64:** Tests AVX2 variants

No additional CI workflow changes needed — the tests are built and run by the existing `ninja test` step.

### 7.3 Forced-Fallback CI Job

Add a new CI matrix entry that builds with `-Denable_asm=false`. This forces all dispatch to use C reference functions, validating that:
1. The C reference path still compiles and works
2. The golden values in the Python integration tests match the C reference output (not SIMD)

---

## 8. Completion Requirements

The SIMD correctness oracle test harness is **complete** when ALL of the following requirements are met:

### R1. Full Dispatch Coverage
Every function pointer dispatch point listed in Section 2 (all 9 functions) has a corresponding oracle test. No dispatch point is left untested.

**Verification:** For each entry in the table in Section 2, there exists a test function that calls both the C reference and at least one SIMD variant and compares their outputs.

### R2. All SIMD Variants Tested
For each dispatch point, every SIMD variant that exists for the build target's architecture is tested:
- On x86-64: all AVX2 variants (9 functions) and all AVX-512 variants (4 functions: VIF subsample_rd_8, VIF subsample_rd_16, VIF statistic_8, VIF statistic_16, Motion x_convolution_16 — 5 functions total).
- On ARM64: all NEON variants (5 functions: ADM dwt2_8, VIF subsample_rd_8, VIF subsample_rd_16, VIF statistic_8, VIF statistic_16).

**Verification:** Count the number of `mu_run_test()` calls inside architecture guards. On x86-64 builds, there must be at least 14 test functions (9 AVX2 + 5 AVX-512). On ARM64, at least 5 test functions.

### R3. Input Coverage
Every test function exercises all 10 input data categories from Section 4.1 and all 8 standard dimensions from Section 4.2.

**Verification:** Each test function contains a loop (or equivalent) over the full input category set and dimension set. The total comparison count per architecture matches or exceeds the numbers in Section 6.1.

### R4. Bit-Exact Integer Comparison
All integer-path tests (ADM dwt2, VIF subsample_rd, Motion x_convolution, CAMBI increment/decrement/derivative) use `memcmp` for byte-exact comparison of output buffers.

**Verification:** Code review confirms `memcmp` (or equivalent byte-exact comparison) is used for all integer outputs.

### R5. Bounded Float Comparison
All float-path tests (VIF vif_statistic num/den) use a relative-error comparison with tolerance no greater than 1e-6 and an absolute near-zero threshold no greater than 1e-9.

**Verification:** Code review confirms the comparison function and tolerance constants.

### R6. Buffer Overrun Detection
Every test function allocates output buffers with guard regions and verifies guard integrity after each SIMD call.

**Verification:** Code review confirms guard allocation, sentinel fill, and post-call guard check for every test.

### R7. Diagnostic Failure Messages
When a comparison fails, the test output includes: (a) the function name, (b) the input category, (c) the dimensions, (d) the bit depth, (e) the first differing position, and (f) the reference and SIMD values at that position.

**Verification:** Intentionally introduce a 1-bit flip in a SIMD output and confirm the failure message contains all required fields.

### R8. Build Integration
The tests compile and run as part of `ninja test` on all CI platforms (Ubuntu x86_64, Ubuntu ARM64, macOS x86_64, Windows x86_64) without manual intervention.

**Verification:** CI pipeline passes with the new tests on all 4 platforms.

### R9. Fallback Path Validation
A CI job builds with `-Denable_asm=false` and runs the full test suite. All existing tests (including Python golden value tests) must pass with the C reference path, confirming that the golden values are architecture-independent.

**Verification:** CI job with `-Denable_asm=false` passes.

### R10. ADM Width-Constraint Boundary Tests
The ADM test includes dimensions where `w % 8 != 0` (specifically w=7, w=9, w=15) and verifies that:
1. The dispatch logic does NOT assign the SIMD variant for these widths.
2. The C reference function produces correct output for these widths.

**Verification:** Test functions for non-multiple-of-8 widths exist and pass.

### R11. Zero Production Code Regression
The oracle test infrastructure does not change the behavior of any production code path. If the `SIMD_ORACLE_STATIC` approach is used, it must be verified that production builds (without `-DVMAF_SIMD_ORACLE_TEST`) produce identical binaries.

**Verification:** Diff the object files of a production build before and after the changes. They must be identical.

### R12. Deterministic Reproducibility
All tests are fully deterministic. PRNG seeds are hardcoded. Running the same test twice produces identical results. No dependency on wall-clock time, system load, or thread scheduling.

**Verification:** Run the test suite twice and confirm identical pass/fail results and identical output on failure.

---

## 9. Acceptance Criteria Summary

| Req | Description | Pass Condition |
|-----|-------------|----------------|
| R1 | Full dispatch coverage | 9/9 dispatch points have oracle tests |
| R2 | All SIMD variants | 14 AVX2+AVX512 tests on x86-64; 5 NEON tests on ARM64 |
| R3 | Input coverage | 10 categories x 8+ dimensions per function |
| R4 | Bit-exact integer | `memcmp` used for all integer outputs |
| R5 | Bounded float | Relative tolerance <= 1e-6 for float outputs |
| R6 | Buffer overrun detection | Guard regions checked after every call |
| R7 | Diagnostic messages | Failing tests report function, input, dimensions, values |
| R8 | Build integration | Tests pass in CI on all 4 platforms |
| R9 | Fallback validation | CI job with `-Denable_asm=false` passes all tests |
| R10 | Width boundary | Non-multiple-of-8 widths tested for ADM |
| R11 | Zero production impact | Production binaries unchanged |
| R12 | Deterministic | Repeated runs produce identical results |

All 12 requirements must be met for the SIMD correctness oracle test harness to be considered complete.
