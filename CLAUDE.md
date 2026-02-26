# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project documents

| Document | What it contains |
|---|---|
| `CLAUDE.md` | This file. Build commands, test commands, architecture overview, links to all other docs. |
| `plans/optimization-opportunities.md` | 22 catalogued inference-path optimisation opportunities (SIMD gaps, per-frame allocations, algorithmic redundancy, cache patterns, parallelism, Python overhead), each with file/line references, impact rating, and suggested fix. Read this before starting any performance work. Sections 1–6 have been addressed — see reports below. |
| `plans/simd-optimizations-report.md` | Report on the Section 1 SIMD work: what was implemented (1a SSIM alloc fix, 1b PSNR AVX2, 1d motion SAD AVX2), what was deferred (1c ADM decouple/csf/cm, 1e ADM horizontal tail), correctness verification (67/67 regression tests pass, bit-identical scores), and benchmark results. |
| `plans/allocation-optimizations-report.md` | Report on the Section 2 allocation work: predict.c feature name/svm_node caching (2b), cambi heatmap buffer (2c), thread pool pre-allocated job pool (2d), feature collector hash map (2e). All 67/67 regression tests pass, bit-identical scores, 0.35% incremental speedup. |
| `plans/algorithmic-optimizations-report.md` | Report on the Section 3 algorithmic redundancy work: dwt_quant_step precomputation (3a/3f), cos_1deg_sq constant (3b), div_lookup guard (3c), pow/log→bit-ops (3d), VIF loop-invariant hoist (3e). All 67/67 regression tests pass, bit-identical scores, 6.65% cache miss reduction. |
| `plans/cache-optimizations-report.md` | Report on the Section 4 cache/memory access pattern work: software prefetch hints in 6 ADM functions (4c alternative), VIF AVX2 loop-invariant hoisting (4a partial), CAMBI histogram layout confirmed optimal (4b skipped). All 67/67 regression tests pass, bit-identical scores. |
| `plans/parallelism-optimizations-report.md` | Report on Section 5 parallelism work: VIF ref/dis parallelism found infeasible (already fused in inner loops), ADM scale pipelining found infeasible (strict serial deps). VIF frame copy optimized instead (bulk memcpy, pad pointer hoisting). 67/67 tests pass, bit-identical scores. |
| `plans/python-orchestration-optimizations-report.md` | Report on Section 6 Python orchestration work: removed shell=True from subprocess calls (6b), batched SVM prediction in VmafLegacyQualityRunner (6c), subprocess consolidation deferred (6a — main path already consolidates). 67/67 tests pass, bit-identical scores. |
| `plans/bench-instance.md` | Everything needed to connect to and manage the AWS bare-metal benchmarking instance. Includes the `bench/run-bench-x86.sh` usage, manual `perf` commands, stop/start/terminate commands, and AWS account notes. |
| `bench/run-bench-x86.sh` | Executable script. Runs a full benchmark on the AWS instance end-to-end: resets machine state, optionally syncs and rebuilds, warms page cache, runs 7 timed repetitions, collects hardware counters, generates flamegraph. Auto-appends one row to `bench-results/history-x86.csv`. Per-run artifacts go in `bench-results/<timestamp>_<label>/` (gitignored). |
| `bench/compare-x86.sh` | Prints `bench-results/history-x86.csv` as a colour-coded table with Δ% columns (time, IPC, cache-miss rate). Use after runs to review regressions or improvements across the history. |
| `bench-results/history-x86.csv` | Git-tracked time-series of every benchmark run: timing, hardware counters, git commit, machine state. Commit this after meaningful runs to preserve the record. |

## Git workflow

Commit early and often. Every logical unit of work — a new script, a doc update, a bug fix, a refactor — should be its own commit. Keep each commit atomic: it should compile, pass tests, and make sense in isolation. Never bundle unrelated changes. Use the imperative mood in commit subject lines (`bench: add compare script`, not `added compare script`).

## Repository Structure

VMAF consists of two main components:

- **`libvmaf/`** — The core C library (`libvmaf v3.0.0`). Contains all feature extractors, models, the `vmaf` CLI tool, and C-level unit tests. Built with Meson/Ninja.
- **`python/`** — Python wrapper library. Provides `QualityRunner`, `FeatureExtractor`, training/validation pipelines, and test infrastructure. Tested with tox/pytest.
- **`model/`** — Pre-built VMAF model files (`.json`). Default model is `vmaf_v0.6.1.json`. Models are compiled into the binary as built-in models.
- **`resource/`** — Documentation, test video fixtures (downloaded on demand from `vmaf_resource` GitHub).

## Building libvmaf

```bash
# Set up build directory (release mode with float feature extractors enabled)
meson setup libvmaf/build libvmaf --buildtype release -Denable_float=true

# Compile
ninja -vC libvmaf/build

# Install (places vmaf binary and headers under a prefix)
ninja -vC libvmaf/build install
```

Using the top-level `Makefile` (manages a `.venv` with meson/ninja):
```bash
make          # builds release
make debug    # builds debug
make install  # builds + installs
```

Key meson options: `-Denable_float=true` (float feature extractors), `-Denable_avx512=true`, `-Denable_cuda=true`.

**Note:** The Makefile expects `python3.10` which may not be available. If it fails, set up the venv manually and build with explicit compiler paths:
```bash
# One-time setup (requires: apt-get install -y nasm python3.13-dev)
python3.13 -m venv .venv
.venv/bin/pip install meson ninja
PATH=".venv/bin:/usr/bin:/bin:$PATH" meson setup libvmaf/build libvmaf --buildtype release -Denable_float=true
CC=/usr/bin/gcc CXX=/usr/bin/g++ PATH=".venv/bin:/usr/bin:/bin:$PATH" ninja -C libvmaf/build

# Install Python test dependencies
cd python && /root/Development/vmaf/.venv/bin/pip install -r requirements.txt -r test/requirements.txt -e . && cd ..
```

## Running Tests

### C unit tests (libvmaf)
```bash
ninja -vC libvmaf/build test
```
Run a single test binary directly:
```bash
./libvmaf/build/test/test_feature_extractor
./libvmaf/build/test/test_cambi
./libvmaf/build/test/test_psnr
```

### Python tests (tox)
From the repo root:
```bash
tox -c python
```
Run a single test file or test case:
```bash
cd python
python -m pytest test/quality_runner_test.py -vv
python -m pytest test/quality_runner_test.py::QualityRunnerTest::test_run_vmaf_legacy_runner -vv
```
The default pytest marker is `main`. Tests not marked `main` are skipped unless `TEST_MARKER` env var is changed.

## Benchmarking

A dedicated bare-metal instance (AWS `c6a.metal`, AMD EPYC 7R13, AVX2) exists for reliable performance profiling.

```bash
bench/run-bench-x86.sh --label baseline          # benchmark current build
bench/run-bench-x86.sh --label my-change --sync  # push src changes, rebuild, benchmark
```

Produces timing stats (7 runs, mean ± stddev), hardware counters (IPC, cache-miss rate), a per-function hotspot report, and a flamegraph SVG. Each run automatically appends one row to `bench-results/history-x86.csv` (git-tracked). Per-run artifact directories land in `bench-results/<timestamp>_<label>/` (gitignored).

To review results across runs:
```bash
bench/compare-x86.sh                          # last 20 runs, all labels, colour-coded Δ%
bench/compare-x86.sh --label baseline --last 5  # filter to one label
```

See **[`plans/bench-instance.md`](plans/bench-instance.md)** for full details, instance management, and manual perf commands.

## Score Regression Tests

The primary regression tests for VMAF scores are in `python/test/quality_runner_test.py`. These tests run the `VmafQualityRunner` (and friends) against known video pairs and assert per-feature and final VMAF scores to 4 decimal places.

**Run regression tests (exact working command):**
```bash
cd /root/Development/vmaf/python && \
  /root/Development/vmaf/.venv/bin/python -m pytest test/quality_runner_test.py -vv -p no:warnings -m main
```

Run a single regression test:
```bash
cd /root/Development/vmaf/python && \
  /root/Development/vmaf/.venv/bin/python -m pytest test/quality_runner_test.py::QualityRunnerTest::test_run_vmaf_runner -vv -p no:warnings -m main
```

Key test video pair used throughout: `src01_hrc00_576x324.yuv` (reference) vs `src01_hrc01_576x324.yuv` (distorted), 576×324, YUV 4:2:0 8-bit. These are auto-downloaded from `vmaf_resource` GitHub if missing.

The `vmaf` CLI tool can also be used directly to verify scores:
```bash
./libvmaf/build/tools/vmaf \
    --reference src01_hrc00_576x324.yuv \
    --distorted src01_hrc01_576x324.yuv \
    --width 576 --height 324 --pixel_format 420 --bitdepth 8 \
    --model version=vmaf_v0.6.1 \
    --output output.json --json
```
Expected pooled mean VMAF score: ~76.67.

## Architecture

### libvmaf C library

- **`libvmaf/src/libvmaf.c`** — Public API entry point (`vmaf_init`, `vmaf_read_pictures`, `vmaf_score_at_index`, etc.)
- **`libvmaf/src/feature/`** — Feature extractors: `integer_adm.c`, `integer_vif.c`, `integer_motion.c`, `integer_psnr.c`, `integer_ssim.c` are the fixed-point SIMD-optimized implementations used by default. `float_*.c` are floating-point variants. `cambi.c` is the banding detector.
- **`libvmaf/src/feature/x86/`** — AVX2/AVX-512 SIMD intrinsics for x86. `libvmaf/src/feature/arm64/` for ARM.
- **`libvmaf/src/predict.c`** — SVM prediction (libsvm) that fuses feature scores into final VMAF score.
- **`libvmaf/src/model.c`** / **`read_json_model.c`** — Model loading from `.json` files or built-in compiled-in models.
- **`libvmaf/tools/vmaf.c`** — CLI tool implementation.

Feature extractors implement the `VmafFeatureExtractor` interface (see `libvmaf/src/feature/feature_extractor.h`): `.init()`, `.extract()`, `.flush()`, `.close()` callbacks. Set `VMAF_FEATURE_EXTRACTOR_TEMPORAL` flag for motion-dependent extractors that require serial execution.

### Python library

- **`python/vmaf/core/feature_extractor.py`** — Base `FeatureExtractor` class. Concrete implementations call `libvmaf` CLI or vmafexec binary.
- **`python/vmaf/core/quality_runner.py`** — `QualityRunner` subclasses: `VmafQualityRunner` (current), `VmafLegacyQualityRunner`, `PsnrQualityRunner`, etc. These orchestrate feature extraction + SVM prediction.
- **`python/vmaf/core/vmafexec_feature_extractor.py`** — Wrappers that invoke the compiled `vmaf` binary directly.
- **`python/vmaf/config.py`** — `VmafConfig` provides canonical paths (model dir, test resources, workspace). `VmafExternalConfig` reads `externals.py` for optional tool paths (ffmpeg, vmaf binary, etc.).
- **`python/test/testutil.py`** — Shared test fixtures; `set_default_576_324_videos_for_testing()` and variants provide the standard test asset pairs.

Test resources (YUV files) are downloaded reactively from `https://github.com/Netflix/vmaf_resource` on first use via `VmafConfig.test_resource_path()`.

### Score computation flow

1. Reference + distorted YUV frames → `vmaf_read_pictures()`
2. Per-frame feature extraction (VIF scales, ADM scales, motion, PSNR, etc.) → `VmafFeatureCollector`
3. Feature scores fed to pre-trained SVM model → per-frame VMAF score
4. Pooling (mean/harmonic-mean/min/max) → final score via `vmaf_score_pooled()`
