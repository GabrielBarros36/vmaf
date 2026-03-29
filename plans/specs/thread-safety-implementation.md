# Thread Safety and Concurrency Tests -- Implementation Notes

## Files Created

| File | Purpose |
|------|---------|
| `libvmaf/test/test_thread_safety.c` | All 4 thread safety and concurrency tests |

## Files Modified

| File | Change |
|------|--------|
| `libvmaf/test/meson.build` | Added `test_thread_safety` (standard) and `test_thread_safety_stress` (extended) executables and test registrations |

## Tests Implemented

### 1. `test_thread_count_determinism`
Runs VMAF scoring with `n_threads` = 1, 2, 4, 8, 16 on the same synthetic 64x64 YUV420P input (10 frames). Verifies that all per-frame scores and the mean pooled score are bit-identical (`memcmp` on `double`) to the single-threaded baseline. Uses the `vmaf_v0.6.1` model, which exercises ADM, VIF, and motion feature extractors concurrently.

On failure, prints the thread count, frame index, baseline score, actual score, and absolute difference.

### 2. `test_framesync_out_of_order_stress`
Stress-tests `VmafFrameSyncContext` by submitting 50 frames (200 in extended mode) from 4 concurrent threads via `VmafThreadPool`. Each worker fills a 1024-byte buffer with a deterministic pattern, then retrieves and verifies the previous frame's buffer byte-by-byte. Repeats for 5 iterations (50 in extended mode).

Uses the internal `framesync.h` and `thread_pool.h` APIs, following the precedent of the existing `test_framesync.c`.

### 3. `test_inflight_teardown`
Creates a `VmafContext` with 8 threads, submits 50 frames (200 in extended mode) rapidly without flushing, then immediately calls `vmaf_close()`. Verifies `vmaf_close()` returns 0 without hanging or crashing. Repeats for 10 iterations (100 in extended mode).

The 120-second meson test timeout acts as a deadlock detector.

### 4. `test_concurrent_contexts_shared_model`
Loads a single `VmafModel`, then spawns 4 pthreads, each creating its own `VmafContext` with `n_threads=4`. All contexts share the same model pointer. Each thread submits 10 identical frames, flushes, retrieves scores, and closes its context. After all threads join, verifies all 4 contexts produced bit-identical per-frame and pooled scores.

## How to Build and Run

### Standard Run
```bash
cd libvmaf
meson setup build -Denable_tests=true
ninja -C build
meson test -C build test_thread_safety --verbose
```

### Extended Stress Run
```bash
meson test -C build test_thread_safety_stress --verbose
# Or run only the stress suite:
meson test -C build --suite stress
```

### ThreadSanitizer Run
```bash
CC=clang meson setup builddir_tsan -Denable_tests=true -Db_sanitize=thread
ninja -C builddir_tsan
meson test -C builddir_tsan test_thread_safety
```

## Spec Requirements Coverage

| Req | Description | Status |
|-----|-------------|--------|
| R1 | Thread count sweep (5 counts, bit-identical scores) | Covered |
| R2 | Feature extractor coverage (ADM + VIF + motion via vmaf_v0.6.1) | Covered |
| R3 | Out-of-order frames (50+ frames, 4+ threads, byte-level verification) | Covered |
| R4 | In-flight teardown (10+ iterations, vmaf_close returns 0) | Covered |
| R5 | Concurrent contexts (4 contexts sharing 1 model, identical scores) | Covered |
| R6 | TSan clean (no races in libvmaf code) | Build configuration provided |
| R7 | Deterministic (fixed arithmetic, no PRNG) | Covered |
| R8 | Extended stress (10x iterations within 600s timeout) | Covered via stress executable |
| R9 | Build integration (compiles in ninja test) | Covered |
| R10 | Diagnostic failure messages (thread count, frame index, scores) | Covered |
| R11 | Zero production code changes | Covered |
