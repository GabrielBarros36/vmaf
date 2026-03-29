# Sanitizer CI Jobs --- Specification

## 1. Objective

Integrate four compiler sanitizers --- AddressSanitizer (ASan), UndefinedBehaviorSanitizer (UBSan), MemorySanitizer (MSan), and ThreadSanitizer (TSan) --- into the GitHub Actions CI pipeline for libvmaf. Each sanitizer build must compile the full library and run the complete test suite (`ninja test`) so that memory safety violations, undefined behavior, uninitialized reads, and data races are caught automatically on every push and pull request.

This specification defines the exact compiler flags, meson configuration, suppression files, workflow YAML, and platform constraints required for each sanitizer.

---

## 2. Sanitizer Overview

### 2.1 Summary Table

| Sanitizer | Abbreviation | Catches | Compiler Support | Runtime Overhead |
|-----------|-------------|---------|------------------|------------------|
| AddressSanitizer | ASan | Buffer overflows (heap/stack/global), use-after-free, use-after-return, stack overflow, memory leaks | GCC >= 4.8, Clang >= 3.1 | ~2x slowdown, ~3x memory |
| UndefinedBehaviorSanitizer | UBSan | Signed integer overflow, shift past bit-width, null pointer dereference, misaligned access, unreachable code, implicit conversions | GCC >= 4.9, Clang >= 3.3 | ~1.2x slowdown |
| MemorySanitizer | MSan | Reads of uninitialized memory (heap and stack) | Clang only (no GCC support) | ~3x slowdown, ~2x memory |
| ThreadSanitizer | TSan | Data races, lock-order inversions, deadlocks, use of destroyed mutexes | GCC >= 4.8, Clang >= 3.2 | ~5-15x slowdown, ~5-10x memory |

### 2.2 Incompatibilities

The following sanitizer combinations are **mutually exclusive** and must never be combined in a single build:

| Combination | Reason |
|-------------|--------|
| ASan + TSan | Both instrument memory accesses; runtime conflict and false positives |
| ASan + MSan | Both shadow the address space; incompatible shadow layouts |
| MSan + TSan | Incompatible shadow memory implementations |

Each sanitizer must be a **separate CI job** with its own build directory and configuration.

---

## 3. Compiler and Linker Flags

### 3.1 AddressSanitizer (ASan)

```
CFLAGS:   -fsanitize=address -fno-omit-frame-pointer -g -O1
CXXFLAGS: -fsanitize=address -fno-omit-frame-pointer -g -O1
LDFLAGS:  -fsanitize=address
```

**Flag rationale:**
- `-fsanitize=address`: Enables ASan instrumentation.
- `-fno-omit-frame-pointer`: Preserves frame pointers for readable stack traces.
- `-g`: Full debug info for source-level error reports.
- `-O1`: Minimal optimization to preserve variable visibility while keeping reasonable performance. (`-O0` works but is significantly slower; `-O2` may optimize away useful debugging context.)

### 3.2 UndefinedBehaviorSanitizer (UBSan)

```
CFLAGS:   -fsanitize=undefined -fno-sanitize-recover=all -fno-omit-frame-pointer -g -O1
CXXFLAGS: -fsanitize=undefined -fno-sanitize-recover=all -fno-omit-frame-pointer -g -O1
LDFLAGS:  -fsanitize=undefined
```

**Flag rationale:**
- `-fsanitize=undefined`: Enables all UBSan checks (signed-integer-overflow, shift, null, alignment, bool, enum, vla-bound, float-cast-overflow, nonnull-attribute, returns-nonnull-attribute, unreachable, vptr).
- `-fno-sanitize-recover=all`: Makes all UB violations fatal (non-zero exit code) instead of printing a warning and continuing. This is critical for CI --- a warning-only UBSan finding would not fail the build.

### 3.3 MemorySanitizer (MSan)

```
CFLAGS:   -fsanitize=memory -fsanitize-memory-track-origins=2 -fno-omit-frame-pointer -g -O1
CXXFLAGS: -fsanitize=memory -fsanitize-memory-track-origins=2 -fno-omit-frame-pointer -g -O1
LDFLAGS:  -fsanitize=memory
```

**Flag rationale:**
- `-fsanitize=memory`: Enables MSan instrumentation.
- `-fsanitize-memory-track-origins=2`: Tracks where uninitialized memory was allocated (level 2 tracks through stores, giving the most useful reports at moderate additional cost).

**Platform constraint:** MSan is **Clang-only**. GCC does not support MSan. The CI job must use `clang`/`clang++`.

**Important:** MSan requires that **all** linked libraries are instrumented. System libraries (libc, libm, libpthread) are not instrumented by default. The build must use static linking where possible and may need an MSan-instrumented libc++ for C++ code. In practice, for libvmaf (which is predominantly C with minimal C++ in `svm.cpp`), linking against the default system libraries works for detecting issues within libvmaf's own code. False positives from uninstrumented system library calls are handled via suppressions (Section 5).

### 3.4 ThreadSanitizer (TSan)

```
CFLAGS:   -fsanitize=thread -fno-omit-frame-pointer -g -O1
CXXFLAGS: -fsanitize=thread -fno-omit-frame-pointer -g -O1
LDFLAGS:  -fsanitize=thread
```

**Flag rationale:**
- `-fsanitize=thread`: Enables TSan instrumentation.

**Relevant libvmaf code paths:** TSan is particularly valuable for libvmaf because the library uses a thread pool (`src/thread_pool.c`) and frame synchronization (`src/framesync.c`) with pthread mutexes and condition variables. The `feature_collector.c` module also uses atomic operations and locking for concurrent score aggregation.

---

## 4. Meson Build Configuration

### 4.1 Passing Sanitizer Flags to Meson

Meson accepts compiler and linker flags via environment variables or command-line arguments. For sanitizer builds, use meson's `setup` command with explicit C/C++ arguments and link arguments.

**ASan:**
```bash
CC=clang CXX=clang++ \
meson setup libvmaf libvmaf/build_asan \
    --buildtype=debugoptimized \
    -Denable_float=true \
    -Db_sanitize=address \
    -Db_lundef=false
```

**UBSan:**
```bash
CC=clang CXX=clang++ \
meson setup libvmaf libvmaf/build_ubsan \
    --buildtype=debugoptimized \
    -Denable_float=true \
    -Db_sanitize=undefined \
    -Db_lundef=false \
    -Dc_args="-fno-sanitize-recover=all" \
    -Dcpp_args="-fno-sanitize-recover=all"
```

**MSan:**
```bash
CC=clang CXX=clang++ \
meson setup libvmaf libvmaf/build_msan \
    --buildtype=debugoptimized \
    -Denable_float=true \
    -Db_sanitize=memory \
    -Db_lundef=false \
    -Dc_args="-fsanitize-memory-track-origins=2" \
    -Dcpp_args="-fsanitize-memory-track-origins=2"
```

**TSan:**
```bash
CC=clang CXX=clang++ \
meson setup libvmaf libvmaf/build_tsan \
    --buildtype=debugoptimized \
    -Denable_float=true \
    -Db_sanitize=thread \
    -Db_lundef=false
```

### 4.2 Key Meson Options

| Option | Value | Rationale |
|--------|-------|-----------|
| `--buildtype=debugoptimized` | `-O2 -g` | Balances debug info with optimization; sanitizers add their own `-fsanitize` flags on top |
| `-Db_sanitize=<name>` | `address`, `undefined`, `memory`, `thread` | Meson's built-in sanitizer support; adds `-fsanitize=<name>` to both compile and link |
| `-Db_lundef=false` | Disables `-Wl,--no-undefined` | Required because sanitizer runtimes provide symbols at load time that are not present in any linked library |
| `-Denable_float=true` | Builds float feature extractors | Ensures maximum code coverage under sanitizers |

### 4.3 NASM / Assembly Interaction

Sanitizer instrumentation only applies to C/C++ code compiled by the compiler. NASM assembly files are **not instrumented** by any sanitizer. This is expected and correct --- the NASM code does not go through the compiler's instrumentation pass.

For MSan specifically, calls from instrumented C code into uninstrumented NASM assembly may produce false positives if the assembly writes to memory that MSan has not seen initialized. If this occurs, the affected test must be suppressed (see Section 5) or the MSan job should disable assembly:

```bash
meson setup libvmaf libvmaf/build_msan \
    --buildtype=debugoptimized \
    -Denable_float=true \
    -Denable_asm=false \
    -Db_sanitize=memory \
    -Db_lundef=false \
    -Dc_args="-fsanitize-memory-track-origins=2" \
    -Dcpp_args="-fsanitize-memory-track-origins=2"
```

Disabling assembly for MSan is the recommended approach because MSan cannot track initialization through handwritten assembly.

---

## 5. Suppression Files

### 5.1 Suppression File Format

Each sanitizer uses a different suppression mechanism and file format.

**ASan suppressions** (`libvmaf/test/sanitizers/asan.supp`):
```
# ASan suppression file for libvmaf
# Format: interceptor_name:function_or_library_pattern
# Reference: https://github.com/google/sanitizers/wiki/AddressSanitizerLeakSanitizer#suppressions

# Suppress leak detection in third-party libsvm allocations
leak:svm_predict_values
leak:svm_load_model
```

**LSan suppressions** (ASan includes LeakSanitizer; same file or separate `lsan.supp`):
```
# LeakSanitizer suppressions
leak:svm_
```

**UBSan suppressions** (`libvmaf/test/sanitizers/ubsan.supp`):
```
# UBSan suppression file for libvmaf
# Format: check_type:function_or_file_pattern

# Suppress known intentional unsigned wrap in SVM scoring
unsigned-integer-overflow:svm.cpp
```

**TSan suppressions** (`libvmaf/test/sanitizers/tsan.supp`):
```
# TSan suppression file for libvmaf
# Format: race_type:function_or_library_pattern

# Example: suppress benign races in third-party code
race:svm_predict_values
```

**MSan suppressions** (`libvmaf/test/sanitizers/msan.ignorelist`):

MSan uses a compile-time ignorelist (not a runtime suppression file):
```
# MSan ignorelist for libvmaf
# Format: src:file_pattern or fun:function_pattern
# Passed via: -fsanitize-blacklist=msan.ignorelist

# Ignore uninstrumented NASM assembly output buffers
src:*/x86/*.asm
fun:cpuid
```

### 5.2 Passing Suppressions to the Runtime

Suppressions are passed via environment variables in the CI workflow:

```yaml
env:
  ASAN_OPTIONS: "suppressions=libvmaf/test/sanitizers/asan.supp:detect_leaks=1:halt_on_error=1"
  UBSAN_OPTIONS: "suppressions=libvmaf/test/sanitizers/ubsan.supp:halt_on_error=1:print_stacktrace=1"
  TSAN_OPTIONS: "suppressions=libvmaf/test/sanitizers/tsan.supp:halt_on_error=1:second_deadlock_stack=1"
  LSAN_OPTIONS: "suppressions=libvmaf/test/sanitizers/lsan.supp"
```

For MSan, the ignorelist is a compile-time flag:
```bash
-Dc_args="-fsanitize-memory-track-origins=2 -fsanitize-blacklist=$PWD/libvmaf/test/sanitizers/msan.ignorelist"
```

### 5.3 Suppression File Location

```
libvmaf/test/sanitizers/
    asan.supp           # ASan runtime suppressions
    lsan.supp           # LeakSanitizer runtime suppressions
    ubsan.supp          # UBSan runtime suppressions
    tsan.supp           # TSan runtime suppressions
    msan.ignorelist     # MSan compile-time ignorelist
```

These files must be checked into the repository. Start with empty or minimal suppression files. Add suppressions only after confirming that a finding is a false positive (not a real bug), and document the justification in a comment above each entry.

---

## 6. Handling Sanitizer-Incompatible Tests

### 6.1 Tests That Must Be Skipped

Some tests may be incompatible with specific sanitizers:

| Test | Sanitizer | Reason | Action |
|------|-----------|--------|--------|
| SIMD oracle tests (NASM paths) | MSan | Uninstrumented assembly produces false uninitialized-read reports | Disable assembly (`-Denable_asm=false`) for the MSan build |
| `test_thread_pool` (stress mode) | TSan | TSan's ~10x overhead may cause timeout on stress tests | Increase test timeout via `meson test --timeout-multiplier 10` |
| Any test using `fork()` | ASan | ASan does not fully support `fork()` without `exec()` | Suppress or skip via `ASAN_OPTIONS=detect_leaks=0` if leak detection in forked child conflicts |

### 6.2 Timeout Handling

Sanitizers significantly increase runtime. All sanitizer CI jobs must increase the test timeout:

```bash
meson test -C libvmaf/build_<sanitizer> --timeout-multiplier 10
```

This multiplies the default test timeout (30 seconds) by 10, giving 300 seconds per test. TSan may need a higher multiplier for thread-heavy tests.

### 6.3 Environment Variable Controls

Each sanitizer provides `*_OPTIONS` environment variables for fine-grained runtime control:

```bash
# ASan: enable leak detection, abort on first error, unmap shadow on exit
ASAN_OPTIONS="detect_leaks=1:halt_on_error=1:unmap_shadow_on_exit=1"

# UBSan: abort on first UB, print stack traces
UBSAN_OPTIONS="halt_on_error=1:print_stacktrace=1"

# MSan: abort on first uninitialized read, print origin tracking
MSAN_OPTIONS="halt_on_error=1:print_stats=1"

# TSan: abort on first race, print second-thread stack for deadlocks
TSAN_OPTIONS="halt_on_error=1:second_deadlock_stack=1:history_size=4"
```

---

## 7. GitHub Actions Workflow

### 7.1 Workflow File

Create `.github/workflows/sanitizers.yml`:

```yaml
name: Sanitizers

on:
  push:
  pull_request:

env:
  DEBIAN_FRONTEND: noninteractive

jobs:
  asan:
    name: AddressSanitizer
    runs-on: ubuntu-latest
    env:
      CC: clang
      CXX: clang++
      ASAN_OPTIONS: >-
        detect_leaks=1:halt_on_error=1:unmap_shadow_on_exit=1:suppressions=libvmaf/test/sanitizers/asan.supp
      LSAN_OPTIONS: >-
        suppressions=libvmaf/test/sanitizers/lsan.supp
    steps:
      - name: Setup python
        uses: actions/setup-python@v6
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install meson
          sudo apt-get update
          sudo -E apt-get -yq install ninja-build clang nasm

      - uses: actions/checkout@v6

      - name: Configure
        run: |
          meson setup libvmaf libvmaf/build_asan \
            --buildtype=debugoptimized \
            -Denable_float=true \
            -Db_sanitize=address \
            -Db_lundef=false

      - name: Build
        run: ninja -vC libvmaf/build_asan

      - name: Test
        run: |
          meson test -C libvmaf/build_asan \
            --timeout-multiplier 10 \
            --print-errorlogs

  ubsan:
    name: UndefinedBehaviorSanitizer
    runs-on: ubuntu-latest
    env:
      CC: clang
      CXX: clang++
      UBSAN_OPTIONS: >-
        halt_on_error=1:print_stacktrace=1:suppressions=libvmaf/test/sanitizers/ubsan.supp
    steps:
      - name: Setup python
        uses: actions/setup-python@v6
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install meson
          sudo apt-get update
          sudo -E apt-get -yq install ninja-build clang nasm

      - uses: actions/checkout@v6

      - name: Configure
        run: |
          meson setup libvmaf libvmaf/build_ubsan \
            --buildtype=debugoptimized \
            -Denable_float=true \
            -Db_sanitize=undefined \
            -Db_lundef=false \
            -Dc_args="-fno-sanitize-recover=all" \
            -Dcpp_args="-fno-sanitize-recover=all"

      - name: Build
        run: ninja -vC libvmaf/build_ubsan

      - name: Test
        run: |
          meson test -C libvmaf/build_ubsan \
            --timeout-multiplier 10 \
            --print-errorlogs

  msan:
    name: MemorySanitizer
    runs-on: ubuntu-latest
    env:
      CC: clang
      CXX: clang++
      MSAN_OPTIONS: >-
        halt_on_error=1:print_stats=1
    steps:
      - name: Setup python
        uses: actions/setup-python@v6
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install meson
          sudo apt-get update
          sudo -E apt-get -yq install ninja-build clang

      - uses: actions/checkout@v6

      - name: Configure
        run: |
          meson setup libvmaf libvmaf/build_msan \
            --buildtype=debugoptimized \
            -Denable_float=true \
            -Denable_asm=false \
            -Db_sanitize=memory \
            -Db_lundef=false \
            -Dc_args="-fsanitize-memory-track-origins=2" \
            -Dcpp_args="-fsanitize-memory-track-origins=2"

      - name: Build
        run: ninja -vC libvmaf/build_msan

      - name: Test
        run: |
          meson test -C libvmaf/build_msan \
            --timeout-multiplier 10 \
            --print-errorlogs

  tsan:
    name: ThreadSanitizer
    runs-on: ubuntu-latest
    env:
      CC: clang
      CXX: clang++
      TSAN_OPTIONS: >-
        halt_on_error=1:second_deadlock_stack=1:history_size=4:suppressions=libvmaf/test/sanitizers/tsan.supp
    steps:
      - name: Setup python
        uses: actions/setup-python@v6
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install meson
          sudo apt-get update
          sudo -E apt-get -yq install ninja-build clang nasm

      - uses: actions/checkout@v6

      - name: Configure
        run: |
          meson setup libvmaf libvmaf/build_tsan \
            --buildtype=debugoptimized \
            -Denable_float=true \
            -Db_sanitize=thread \
            -Db_lundef=false

      - name: Build
        run: ninja -vC libvmaf/build_tsan

      - name: Test
        run: |
          meson test -C libvmaf/build_tsan \
            --timeout-multiplier 15 \
            --print-errorlogs
```

### 7.2 Design Decisions

| Decision | Rationale |
|----------|-----------|
| Clang for all four jobs | Clang supports all four sanitizers (MSan is Clang-only); using a single compiler across all jobs simplifies maintenance and ensures consistent diagnostics |
| `ubuntu-latest` runner | All sanitizers are Linux-native; macOS support exists but is less mature for MSan and TSan |
| Separate jobs (not matrix) | Each sanitizer needs distinct meson flags and environment variables; a matrix would require complex conditional logic |
| `--print-errorlogs` | Dumps test stderr/stdout into the CI log on failure, making sanitizer reports immediately visible without downloading artifacts |
| NASM not installed for MSan | MSan build uses `-Denable_asm=false`; no assembler needed |
| TSan timeout multiplier 15 | TSan has the highest overhead (5-15x); thread-heavy tests need generous timeouts |

### 7.3 Integration with Existing CI

The new `sanitizers.yml` workflow runs **independently** of the existing workflows:

| Existing Workflow | Purpose | Interaction |
|-------------------|---------|-------------|
| `libvmaf.yml` | Release builds, GCC/Clang matrix, tox tests | No change. Sanitizer jobs are additive. |
| `simd-oracle.yml` | SIMD correctness oracle tests | No change. Sanitizer jobs test the full suite including SIMD oracle tests (except MSan which disables ASM). |
| `windows.yml` | Windows MinGW build | No change. Sanitizers are Linux-only in this spec. |
| `docker.yml` | Docker image build | No change. |
| `ffmpeg.yml` | FFmpeg integration test | No change. |

The sanitizer workflow triggers on the same events (`push`, `pull_request`) as existing workflows.

---

## 8. Platform Constraints

### 8.1 Constraint Matrix

| Constraint | Affected Sanitizer | Detail |
|------------|--------------------|--------|
| Clang-only | MSan | GCC does not implement MemorySanitizer. The MSan job must use `CC=clang CXX=clang++`. |
| ASan+TSan mutual exclusion | ASan, TSan | Cannot combine in one build. Separate CI jobs. |
| ASan+MSan mutual exclusion | ASan, MSan | Cannot combine in one build. Separate CI jobs. |
| MSan+TSan mutual exclusion | MSan, TSan | Cannot combine in one build. Separate CI jobs. |
| Uninstrumented assembly | MSan | MSan cannot track initialization through NASM assembly. MSan build must use `-Denable_asm=false`. |
| Stack size | ASan | ASan uses shadow memory that increases stack usage. If stack overflow occurs in tests, set `ASAN_OPTIONS=detect_stack_use_after_return=0` or increase `ulimit -s`. |
| Address space | MSan, TSan | Both MSan and TSan require large virtual address space mappings. GitHub Actions runners (7GB RAM, 14GB swap) are sufficient. |
| Kernel version | TSan | TSan requires Linux kernel >= 3.2 for its shadow memory layout. `ubuntu-latest` runners satisfy this. |

### 8.2 Why Not Windows or macOS

| Platform | ASan | UBSan | MSan | TSan |
|----------|------|-------|------|------|
| Linux (Clang) | Supported | Supported | Supported | Supported |
| macOS (Clang) | Supported | Supported | Not supported | Partial (no `detect_deadlocks`) |
| Windows (MSVC) | Partial (`/fsanitize=address`) | Not supported | Not supported | Not supported |
| Windows (MinGW) | Partial | Partial | Not supported | Not supported |

Linux with Clang provides the most complete and reliable sanitizer coverage. macOS and Windows jobs may be added in the future but are out of scope for this initial implementation.

---

## 9. Interpreting Sanitizer Output

### 9.1 ASan Error Report Format

```
==12345==ERROR: AddressSanitizer: heap-buffer-overflow on address 0x6020000001fc
READ of size 4 at 0x6020000001fc thread T0
    #0 0x55a1c3 in adm_dwt2_8 /path/to/integer_adm.c:142
    #1 0x55b2d4 in extract_adm /path/to/integer_adm.c:580
```

**Action:** The stack trace points directly to the offending source line. Fix the out-of-bounds access.

### 9.2 UBSan Error Report Format

```
/path/to/integer_vif.c:203:15: runtime error: signed integer overflow:
2147483647 + 1 cannot be represented in type 'int'
    #0 0x55a1c3 in vif_statistic_8 /path/to/integer_vif.c:203
```

**Action:** Fix the arithmetic to avoid overflow (use wider type or restructure computation).

### 9.3 MSan Error Report Format

```
==12345==WARNING: MemorySanitizer: use-of-uninitialized-value
    #0 0x55a1c3 in subsample_rd_8 /path/to/integer_vif.c:95
  Uninitialized value was created by a heap allocation
    #0 0x55c4d2 in vmaf_alloc /path/to/mem.c:42
```

**Action:** Ensure all allocated buffers are initialized before use.

### 9.4 TSan Error Report Format

```
WARNING: ThreadSanitizer: data race (pid=12345)
  Write of size 8 at 0x7f8c00001234 by thread T2:
    #0 feature_collector_append /path/to/feature_collector.c:156
  Previous read of size 8 at 0x7f8c00001234 by thread T1:
    #0 feature_collector_get /path/to/feature_collector.c:201
```

**Action:** Add proper synchronization (mutex, atomic) around the shared data access.

---

## 10. Completion Requirements

The sanitizer CI integration is **complete** when ALL of the following requirements are met:

### R1. Four Separate CI Jobs

The workflow file `.github/workflows/sanitizers.yml` defines exactly four jobs: `asan`, `ubsan`, `msan`, `tsan`. Each job builds and tests independently.

**Verification:** The workflow file contains four top-level job definitions under `jobs:`.

### R2. Correct Compiler Flags

Each sanitizer job passes the correct `-fsanitize=<type>` flag through meson's `-Db_sanitize=` option. UBSan additionally passes `-fno-sanitize-recover=all`. MSan additionally passes `-fsanitize-memory-track-origins=2`.

**Verification:** Code review of the `meson setup` command in each job confirms the flags match Section 4.1.

### R3. Fatal Error Mode

All sanitizer jobs are configured so that sanitizer findings cause a non-zero exit code, failing the CI job. ASan, MSan, and TSan use `halt_on_error=1` in their `*_OPTIONS` environment variables. UBSan uses `-fno-sanitize-recover=all` at compile time and `halt_on_error=1` at runtime.

**Verification:** Intentionally introduce a bug (e.g., heap-buffer-overflow for ASan, signed overflow for UBSan) and confirm the CI job fails with a sanitizer error report.

### R4. Full Test Suite Execution

Each sanitizer job runs the complete test suite via `meson test`. No tests are silently excluded (except MSan which disables assembly via `-Denable_asm=false`, which is explicitly documented).

**Verification:** The `meson test` output shows the same number of tests as the non-sanitized build (accounting for assembly-dependent tests in the MSan case).

### R5. Suppression Infrastructure

Suppression files exist at `libvmaf/test/sanitizers/` for all four sanitizers (`asan.supp`, `lsan.supp`, `ubsan.supp`, `tsan.supp`, `msan.ignorelist`). Each file is referenced by the corresponding `*_OPTIONS` environment variable or compile-time flag in the workflow.

**Verification:** Suppression files exist in the repository and are correctly referenced in the workflow YAML.

### R6. MSan Clang-Only Enforcement

The MSan job uses `CC=clang CXX=clang++` and disables assembly (`-Denable_asm=false`). The job does not attempt to use GCC.

**Verification:** The MSan job's `env` block and `meson setup` command specify clang and disable ASM.

### R7. No Sanitizer Combination Conflicts

No CI job combines incompatible sanitizers (ASan+TSan, ASan+MSan, MSan+TSan). Each job enables exactly one sanitizer.

**Verification:** Each `meson setup` command contains exactly one `-Db_sanitize=` value.

### R8. Adequate Timeouts

Each sanitizer job uses `--timeout-multiplier` with a value of at least 10 (ASan, UBSan, MSan) or 15 (TSan) to account for sanitizer runtime overhead.

**Verification:** The `meson test` command in each job includes `--timeout-multiplier` with the specified minimum value.

### R9. Error Logs in CI Output

Each sanitizer job uses `--print-errorlogs` so that sanitizer diagnostic output is visible directly in the GitHub Actions log without downloading artifacts.

**Verification:** The `meson test` command in each job includes `--print-errorlogs`.

### R10. Independent of Existing Workflows

The `sanitizers.yml` workflow does not modify any existing workflow file (`libvmaf.yml`, `simd-oracle.yml`, `windows.yml`, `docker.yml`, `ffmpeg.yml`). It is purely additive.

**Verification:** `git diff` of existing workflow files shows no changes.

### R11. Clang Compiler for All Jobs

All four sanitizer jobs use Clang as the compiler for consistency and because MSan requires it. This avoids divergent behavior between GCC-sanitized and Clang-sanitized builds.

**Verification:** All four jobs specify `CC: clang` and `CXX: clang++` in their `env` block.

### R12. Documented Suppressions

Every entry in a suppression file includes a comment explaining why the suppression is necessary (false positive, third-party code, known benign behavior). No uncommented suppressions are allowed.

**Verification:** Code review of all suppression files confirms every entry has an explanatory comment.

---

## 11. Acceptance Criteria Summary

| Req | Description | Pass Condition |
|-----|-------------|----------------|
| R1 | Four separate CI jobs | `sanitizers.yml` defines `asan`, `ubsan`, `msan`, `tsan` jobs |
| R2 | Correct compiler flags | Each job's `meson setup` uses the correct `-Db_sanitize=` value and additional flags per Section 4.1 |
| R3 | Fatal error mode | Intentional bug triggers non-zero exit and visible sanitizer report |
| R4 | Full test suite | `meson test` runs all tests; test count matches non-sanitized build |
| R5 | Suppression infrastructure | Suppression files exist at `libvmaf/test/sanitizers/` and are referenced in YAML |
| R6 | MSan Clang-only | MSan job uses clang and `-Denable_asm=false` |
| R7 | No sanitizer conflicts | Each job enables exactly one sanitizer |
| R8 | Adequate timeouts | `--timeout-multiplier` >= 10 (ASan/UBSan/MSan) or >= 15 (TSan) |
| R9 | Error logs visible | `--print-errorlogs` present in all `meson test` commands |
| R10 | No existing workflow changes | Existing workflow files are unmodified |
| R11 | Clang for all jobs | All four jobs use `CC=clang CXX=clang++` |
| R12 | Documented suppressions | Every suppression entry has an explanatory comment |

All 12 requirements must be met for the sanitizer CI integration to be considered complete.

---

## 12. Implementation Notes

### 12.1 Files Created

| File | Purpose |
|------|---------|
| `.github/workflows/sanitizers.yml` | GitHub Actions workflow with four sanitizer jobs (ASan, UBSan, MSan, TSan) |
| `libvmaf/test/sanitizers/asan.supp` | ASan runtime suppressions (suppresses libsvm leak reports) |
| `libvmaf/test/sanitizers/lsan.supp` | LeakSanitizer runtime suppressions (suppresses libsvm leak reports) |
| `libvmaf/test/sanitizers/ubsan.supp` | UBSan runtime suppressions (suppresses unsigned overflow in svm.cpp) |
| `libvmaf/test/sanitizers/tsan.supp` | TSan runtime suppressions (suppresses benign races in libsvm) |
| `libvmaf/test/sanitizers/msan.ignorelist` | MSan compile-time ignorelist (ignores NASM assembly and cpuid) |

### 12.2 Files Modified

None. The implementation is purely additive. No existing workflow files or production source code were modified.

### 12.3 Deviations from Spec

- **MSan ignorelist not passed as compile-time flag:** Section 5.2 of this spec describes passing the MSan ignorelist via `-fsanitize-blacklist=`. However, the authoritative workflow definition in Section 7.1 does not include this flag, relying instead on `-Denable_asm=false` to avoid MSan false positives from uninstrumented assembly (the recommended approach per Section 4.3). The implementation follows the Section 7.1 workflow exactly. The `msan.ignorelist` file is created for future use if finer-grained suppression becomes necessary.

### 12.4 How to Run Locally

Each sanitizer build can be run locally using the same commands as the CI workflow. Requires `clang`, `clang++`, `meson`, `ninja`, and `nasm` (except MSan which does not need `nasm`).

**ASan:**
```bash
export CC=clang CXX=clang++
export ASAN_OPTIONS="detect_leaks=1:halt_on_error=1:unmap_shadow_on_exit=1:suppressions=libvmaf/test/sanitizers/asan.supp"
export LSAN_OPTIONS="suppressions=libvmaf/test/sanitizers/lsan.supp"
meson setup libvmaf libvmaf/build_asan --buildtype=debugoptimized -Denable_float=true -Db_sanitize=address -Db_lundef=false
ninja -vC libvmaf/build_asan
meson test -C libvmaf/build_asan --timeout-multiplier 10 --print-errorlogs
```

**UBSan:**
```bash
export CC=clang CXX=clang++
export UBSAN_OPTIONS="halt_on_error=1:print_stacktrace=1:suppressions=libvmaf/test/sanitizers/ubsan.supp"
meson setup libvmaf libvmaf/build_ubsan --buildtype=debugoptimized -Denable_float=true -Db_sanitize=undefined -Db_lundef=false -Dc_args="-fno-sanitize-recover=all" -Dcpp_args="-fno-sanitize-recover=all"
ninja -vC libvmaf/build_ubsan
meson test -C libvmaf/build_ubsan --timeout-multiplier 10 --print-errorlogs
```

**MSan:**
```bash
export CC=clang CXX=clang++
export MSAN_OPTIONS="halt_on_error=1:print_stats=1"
meson setup libvmaf libvmaf/build_msan --buildtype=debugoptimized -Denable_float=true -Denable_asm=false -Db_sanitize=memory -Db_lundef=false -Dc_args="-fsanitize-memory-track-origins=2" -Dcpp_args="-fsanitize-memory-track-origins=2"
ninja -vC libvmaf/build_msan
meson test -C libvmaf/build_msan --timeout-multiplier 10 --print-errorlogs
```

**TSan:**
```bash
export CC=clang CXX=clang++
export TSAN_OPTIONS="halt_on_error=1:second_deadlock_stack=1:history_size=4:suppressions=libvmaf/test/sanitizers/tsan.supp"
meson setup libvmaf libvmaf/build_tsan --buildtype=debugoptimized -Denable_float=true -Db_sanitize=thread -Db_lundef=false
ninja -vC libvmaf/build_tsan
meson test -C libvmaf/build_tsan --timeout-multiplier 15 --print-errorlogs
```

All commands should be run from the repository root directory. Suppression file paths are relative to the working directory.
