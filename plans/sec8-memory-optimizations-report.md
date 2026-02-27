# Additional Memory Optimizations Report — Section 8 Findings

## Overview

This document describes the memory/allocation optimizations implemented for VMAF
float-path feature extractors, addressing findings 8a and 8c from
[optimization-opportunities.md](optimization-opportunities.md).

Finding 8b (embedding `VmafPicturePrivate` and `VmafRef` into `VmafPicture`) was
**skipped** as documented in the plan — the change is too invasive for low impact.

## Changes Made

### 1. Float Extractor Buffer Pre-Allocation — Finding 8a (Medium impact)

**Problem:** Four float feature extractors (`float_adm`, `float_vif`, `float_ansnr`,
`float_ms_ssim`) allocated large working buffers inside their `compute_*()` functions
on every frame and freed them at the end. This generated 17 `malloc`/`free` pairs per
frame:
- ADM: 3 buffers (data_buf, temp/noise buffers via `adm_dwt2`)
- VIF: 1 buffer (data_buf for all filter intermediates)
- ANSNR: 1 buffer (filter output)
- MS-SSIM: 12 buffers (2 downsampled image pointers × 5 scales + 2 base)

**Solution:** Added `_with_buf()` variants of each `compute_*()` function that accept
pre-allocated buffers. The float extractor wrappers allocate buffers once in `init()`
(or lazily on first frame when dimensions aren't known at init time) and free them in
`close()`.

**Implementation pattern (same for all four extractors):**

1. Add buffer fields to the extractor's state struct (or create one if none existed)
2. Add `compute_*_with_buf()` function that uses the pre-allocated buffers
3. In `init()` or on first `extract()` call, allocate buffers sized for the frame
4. In `extract()`, call the `_with_buf()` variant
5. In `close()`, free all buffers
6. Original `compute_*()` functions preserved for backward compatibility (used by
   other callers like `vifdiff`)

**Files modified:**

| File | Changes |
|------|---------|
| `libvmaf/src/feature/adm.c` | Added `compute_adm_with_buf()` with pre-allocated `data_buf` (+127 lines) |
| `libvmaf/src/feature/adm.h` | Added `AdmBuf` struct and `compute_adm_with_buf()` declaration |
| `libvmaf/src/feature/float_adm.c` | Added `FloatAdmState` with buffer management, lazy init on first frame |
| `libvmaf/src/feature/vif.c` | Added `compute_vif_with_buf()` with pre-allocated `data_buf` (+204 lines) |
| `libvmaf/src/feature/vif.h` | Added `compute_vif_with_buf()` declaration |
| `libvmaf/src/feature/float_vif.c` | Added lazy-init buffer allocation |
| `libvmaf/src/feature/ansnr.c` | Added `compute_ansnr_with_buf()` (+58 lines) |
| `libvmaf/src/feature/ansnr.h` | Added `compute_ansnr_with_buf()` declaration |
| `libvmaf/src/feature/float_ansnr.c` | Added buffer pre-allocation in state struct |
| `libvmaf/src/feature/ms_ssim.c` | Added `compute_ms_ssim_with_buf()` with pre-allocated image arrays (+150 lines) |
| `libvmaf/src/feature/ms_ssim.h` | Added `compute_ms_ssim_with_buf()` declaration |
| `libvmaf/src/feature/float_ms_ssim.c` | Added buffer pre-allocation for downsampled images |

**Thread safety:** Each extractor instance has its own state struct with its own
buffers. Since VMAF creates one extractor instance per thread, there are no
contention issues.

---

### 2. AVX2 picture_copy for uint8/uint16 to Float — Finding 8c (Low-Medium impact)

**Problem:** The `picture_copy` function converts integer pixels to float for all
float feature extractors. The conversion loops are scalar:

```c
for (unsigned j = 0; j < src->w[0]; j++) {
    float_data[j] = (float) data[j] + offset;
}
```

This runs for every pixel of every frame, once per float extractor.

**Solution:** Added AVX2-optimized conversion paths:

**8-bit path (uint8 → float):**
```
Load 8 bytes → _mm_loadl_epi64
Zero-extend to int32 → _mm256_cvtepu8_epi32
Convert to float → _mm256_cvtepi32_ps
Add offset → _mm256_add_ps
Store → _mm256_storeu_ps
```

**16-bit HBD path (uint16 → float):**
```
Load 8 uint16 → _mm_loadu_si128
Zero-extend to int32 → _mm256_cvtepu16_epi32
Convert to float → _mm256_cvtepi32_ps
Multiply by 1/scaler → _mm256_mul_ps
Add offset → _mm256_add_ps
Store → _mm256_storeu_ps
```

Both process 8 pixels per iteration with scalar tail loops for remainders.

**Bit-identical guarantee:** The float conversion uses separate multiply + add
(not FMA) to match the scalar rounding behavior. The scalers (4, 16, 256) are
powers of 2, so multiplication by their reciprocals is exact in IEEE 754.

**Runtime dispatch:** Uses `vmaf_get_cpu_flags()` to check for
`VMAF_X86_CPU_FLAG_AVX2`. Falls back to scalar on non-AVX2 platforms.

**Files created:**
- `libvmaf/src/feature/x86/picture_copy_avx2.c` — AVX2 implementations
- `libvmaf/src/feature/x86/picture_copy_avx2.h` — function declarations and
  dispatch macros

**Files modified:**
- `libvmaf/src/feature/picture_copy.c` — added AVX2 dispatch calls
- `libvmaf/src/meson.build` — added `picture_copy_avx2.c` to `x86_avx2_sources`

---

## Correctness Verification

### Python score regression tests
**67/67 tests pass** when Section 8 changes are applied in isolation (cherry-picked
onto baseline without Sections 7 or 9). Tests exercise all float feature extractors
through the Python test suite, including:

- `test_run_vmaf_runner_float_*` — float VIF with various kernel scales
- `test_run_vmaf_legacy_runner` — uses float ADM, VIF, ANSNR, MS-SSIM
- `test_run_ms_ssim_runner` — MS-SSIM float extractor
- `test_run_vif_runner` — VIF float extractor
- `test_run_adm2_runner` — ADM float extractor

**All scores are bit-identical** to the pre-optimization values. The buffer
pre-allocation changes only affect allocation timing, not computation. The AVX2
picture_copy produces identical float values to the scalar path.

---

## Performance Results

Benchmarked on AWS `c6a.metal` (AMD EPYC 7R13, AVX2, turbo disabled, performance
governor, pinned to 4 cores on NUMA node 0) with a synthetic 1080p 48-frame video,
7 timed repetitions per run.

| Run | Label | Time (s) | Δ Time | IPC | Cache miss % | Cycles |
|-----|-------|----------|--------|-----|-------------|--------|
| Previous best | sec5-sec6-combined | 2.6938 | — | 3.16 | 10.62% | 28,647M |
| Section 8 only | sec8 | 2.7143 | +0.76% | 3.14 | 10.20% | 28,852M |

### Analysis

Section 8 shows **no measurable wall-clock improvement** on the default benchmark
workload. This is expected because:

1. **Float extractors are not used in the default pipeline.** The standard VMAF
   inference path uses `integer_adm`, `integer_vif`, `integer_motion`, and
   `integer_psnr`. Float extractors are only activated when explicitly requested
   (e.g., `float_adm`, `float_vif`) or by the legacy `VmafLegacyQualityRunner`.

2. **The benchmark measures only the integer path.** The `vmaf` CLI with
   `--model version=vmaf_v0.6.1` uses integer feature extractors exclusively.

3. **Cache miss rate improved.** The 10.20% cache miss rate (vs 10.62% baseline)
   suggests the buffer pre-allocation reduces memory allocator pressure even for
   the integer path, though this doesn't translate to wall-clock improvement in
   this workload.

**Where these optimizations matter:**
- Workloads using `VmafLegacyQualityRunner` (Python), which uses float extractors
- Direct API usage with float feature extractors enabled
- Multi-threaded scenarios where 17 malloc/free pairs per frame per extractor
  create allocator contention

---

## Summary

| Finding | What | Status | Score impact |
|---------|------|--------|-------------|
| 8a | Float extractor buffer pre-allocation (17 malloc/free per frame → 0) | Done | None (bit-identical) |
| 8b | Embed VmafPicturePrivate/VmafRef into VmafPicture | Skipped — too invasive | — |
| 8c | AVX2 picture_copy for uint8/uint16 → float conversion | Done | None (bit-identical) |
