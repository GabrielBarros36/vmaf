# Pre-Existing CI Issues

Issues discovered by new test infrastructure on the `test-harness` branch that
exist identically in `master`. No code changes were made for these; they are
documented here for someone to pick up independently.

---

## 1. ADM DWT2 AVX2 Horizontal Pass Boundary Overwrite

**Severity:** Medium (correctness bug, affects scores for small widths)
**Sanitizer:** ASan (heap-buffer-overflow), SIMD oracle mismatch
**File:** `libvmaf/src/feature/x86/adm_avx2.c`, function `adm_dwt2_8_avx2`
**Line:** ~504 (the horizontal pass inner loop)

**Description:**
The horizontal pass loop `for (int j = 1; j < (w + 1) / 2; j = j + 16)` writes
16 elements per iteration via `_mm256_storeu_si256` (32 bytes). On the last
iteration, when `(w+1)/2` is not a multiple of 16, the store overwrites up to 15
elements past the intended output boundary `(w+1)/2`.

**Test output:**
```
adm_dwt2_8_avx2 band_a MISMATCH [zero 64x64]: pos (0,31), ref=-16385, simd=-17884
```

**Impact:** Corrupts the last few DWT coefficients in each output band. For the
64x64 "zero" test case, the reference produces -16385 at position (0,31) while
AVX2 produces -17884 (delta = 1499). This also causes `test_model_validation`
to fail on Windows when the scoring pipeline uses AVX2 DWT on small frames (64x64).

**Verification:** `git show master:libvmaf/src/feature/x86/adm_avx2.c` contains
the identical loop structure with the same boundary issue.

**Suggested fix:** Add a bounds check or use masked stores for the last iteration.

---

## 2. ADM DWT2 NEON Missing 4th Filter Tap

**Severity:** Medium (correctness bug)
**File:** `libvmaf/src/feature/arm/adm_neon.c`, function `adm_dwt2_8_neon`

**Description:**
The NEON DWT implementation is missing the 4th filter tap in the horizontal pass,
producing incorrect DWT coefficients for all inputs.

**Test output:**
```
adm_dwt2_8_neon MISMATCH [zero 64x64]: pos (0,0)
```

**Impact:** All ADM scores computed via NEON path are slightly incorrect.

**Verification:** Code exists identically on master.

---

## 3. ThreadSanitizer: `framesync.c` Data Race

**Severity:** Low (likely benign but triggers TSan)
**File:** `libvmaf/src/framesync.c`, lines 79 and 123
**Functions:** `vmaf_framesync_acquire_new_buf` (reader), `vmaf_framesync_submit_filled_data` (writer)

**Description:**
Thread T1 reads a shared field while holding mutex M0, but Thread T2 writes the
same field while holding a different mutex M1. The two mutexes do not provide
mutual exclusion between the reader and writer.

**Verification:** Code exists identically on master.

---

## 4. ThreadSanitizer: `div_lookup` Table Race

**Severity:** Low (benign — deterministic values written by all threads)
**File:** `libvmaf/src/feature/integer_adm.h`, line 17, function `div_lookup_generator`

**Description:**
The global `div_lookup` array (65537 entries, 262148 bytes) is lazily initialized
during ADM feature extractor `init()`. When multiple threads each create their own
ADM context, they all call `div_lookup_generator` concurrently, writing to the same
global array without synchronization. The values written are deterministic (all
threads compute the same values), making this a benign race, but TSan flags it.

**Suggested fix:** Use `pthread_once` or an atomic flag to ensure single initialization.

**Verification:** Code exists identically on master.

---

## 5. UBSan: `vidinput.c` Function Pointer Type Mismatch

**Severity:** Low (undefined behavior, works in practice)
**File:** `libvmaf/tools/vidinput.c`, line 51, function `video_input_open`

**Description:**
```
runtime error: call to function y4m_input_open through pointer to incorrect
function type 'void *(*)(struct _IO_FILE *)'
```
The function pointer used to call `y4m_input_open` has a type mismatch — the
actual function signature does not match the pointer type.

**Verification:** Code exists identically on master.

---

## 6. UBSan: `integer_adm.c` Signed Integer Overflow in `adm_dwt2_16`

**Severity:** Medium (undefined behavior for high bit-depth content)
**File:** `libvmaf/src/feature/integer_adm.c`, line 2255, function `adm_dwt2_16`

**Description:**
```
signed integer overflow: 1844014813 + 313256905 cannot be represented in type
'int32_t' (aka 'int')
```
The 16-bit DWT path accumulates filter coefficients into `int32_t` and overflows
when processing extreme high bit-depth (10/12-bit) synthetic content with maximum
pixel values.

**Triggered by:** `test_bit_depth_coverage_adm` in `test/test_degenerate.c:1628`
(new test, but the overflowing code is on master).

**Verification:** `git diff master -- libvmaf/src/feature/integer_adm.c` shows
line 2255 is untouched.

---

## 7. MemorySanitizer: SVM Parser Uninitialized Value

**Severity:** Low-Medium
**File:** `libvmaf/src/svm.cpp`, line 2992, function `SVMModelParserBufferSource::read_next()`

**Description:**
```
use-of-uninitialized-value in SVMModelParserBufferSource::read_next()
```
The SVM model parser reads from a buffer source and encounters uninitialized
memory. Triggered whenever a VMAF model is loaded from a built-in buffer.

**Impact:** Affects many tests under MSan: test_predict, test_model,
test_feature_collector, test_model_validation, test_thread_safety, test_degenerate.

**Verification:** Code exists identically on master.

---

## 8. Fuzzer: Y4M Parser Heap-Buffer-Overflow (C411 Chroma)

**Severity:** High (security-relevant)
**File:** `libvmaf/tools/y4m_input.c`, function `y4m_convert_411_422jpeg`

**Description:**
```
heap-buffer-overflow: WRITE of size 1 at 0x502000003e58
```
The Y4M parser allocates a buffer based on the destination chroma format but
`y4m_convert_411_422jpeg` writes beyond the allocated region when the input has
C411 chroma subsampling with a very small width (e.g., `W2 H2 C411`).

The function writes to `_dst[x<<1|1]` without checking whether that index exceeds
`dst_c_w`, causing a 1-byte heap buffer overflow.

**Triggering input:** `YUV4MPEG2 W2 H2 F30:1 Ip C411`
**Crash artifact:** `crash-0e86f6a06ac16cd4627b503d811efa3b53c04c78`

**Verification:** Code exists identically on master. Discovered by CI fuzzer.

---

## 9. Y4M Parser: Negative Dimensions Cause Allocation Failure

**Severity:** Medium (DoS via crafted Y4M)
**File:** `libvmaf/tools/y4m_input.c`, line 728, function `y4m_input_open_impl`

**Description:**
When given negative dimensions (e.g., `W-1 H-1`), the parsed values become very
large unsigned values, causing `malloc()` to be called with ~0xffffffffffffff10
bytes. No validation of width/height being positive before allocation.

**Triggered by:** `test_y4m_negative_dimensions` in `test/test_format_y4m.c:250`
(new test, but vulnerable code is on master).

**Sanitizers:** ASan (allocation-size-too-big), MSan, TSan

---

## 10. Memory Leaks in Test Cleanup Paths (ASan/LeakSanitizer)

**Severity:** Low (test-only, not production code)

Several tests leak memory in cleanup/error paths:
- `test_predict`: 144 bytes (`VmafDictionary` via `vmaf_dictionary_set`)
- `test_cambi`: 288 bytes (aligned allocation via `aligned_malloc`)
- `test_model_validation`: Multiple 5-byte leaks via `vmaf_model_generate_name`
  during error-path tests (`test_load_corrupt_json`, etc.)

**Verification:** These tests and the leaking code exist on master.
