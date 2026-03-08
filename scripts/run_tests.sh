#!/usr/bin/env bash
# =============================================================================
# VMAF Master Test Runner
# =============================================================================
#
# Unified test runner that organizes all VMAF tests into six tiers, from
# quick local smoke tests through CI-only infrastructure tests.
#
# Usage:
#   ./run_tests.sh [TIER] [OPTIONS]
#
# Tiers:
#   quick       (default) Tier 1 -- fast unit tests, <30s
#   standard              Tiers 1+2 -- core tests + SIMD/thread/degenerate, ~1-2min
#   extended              Tiers 1+2+3 -- all above + golden values + stress, ~5-15min
#   benchmark             Tier 4 only -- performance benchmarks (no pass/fail)
#   all                   Tiers 1+2+3+4
#   ci-status             Show CI workflow inventory and purpose
#
# Options:
#   --build-dir DIR       Use DIR as the meson build directory
#   --no-color            Disable colored output
#   --verbose             Show full test output (pass --verbose to meson test)
#   --help                Show this help message
#
# Examples:
#   ./run_tests.sh                          # Quick tier from auto-detected build
#   ./run_tests.sh standard                 # Standard tier
#   ./run_tests.sh --build-dir build_debug  # Quick tier from specific build dir
#   ./run_tests.sh extended --verbose       # Extended tier with full output
#
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# The repository root is one level above the scripts/ directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LIBVMAF_DIR="$REPO_ROOT/libvmaf"

# Default build directory search order
DEFAULT_BUILD_DIRS=(
    "$LIBVMAF_DIR/build"
    "$LIBVMAF_DIR/build_verify"
    "$LIBVMAF_DIR/builddir"
)

# ---------------------------------------------------------------------------
# Color helpers (disabled with --no-color or when stdout is not a terminal)
# ---------------------------------------------------------------------------

USE_COLOR=true
if [[ ! -t 1 ]]; then
    USE_COLOR=false
fi

_color() {
    if $USE_COLOR; then printf "%b" "$1"; fi
}

RED=$'\033[1;31m'
GREEN=$'\033[1;32m'
YELLOW=$'\033[1;33m'
BLUE=$'\033[1;34m'
CYAN=$'\033[1;36m'
BOLD=$'\033[1m'
RESET=$'\033[0m'

color_red()    { if $USE_COLOR; then echo "${RED}$*${RESET}";    else echo "$*"; fi; }
color_green()  { if $USE_COLOR; then echo "${GREEN}$*${RESET}";  else echo "$*"; fi; }
color_yellow() { if $USE_COLOR; then echo "${YELLOW}$*${RESET}"; else echo "$*"; fi; }
color_blue()   { if $USE_COLOR; then echo "${BLUE}$*${RESET}";   else echo "$*"; fi; }
color_cyan()   { if $USE_COLOR; then echo "${CYAN}$*${RESET}";   else echo "$*"; fi; }
color_bold()   { if $USE_COLOR; then echo "${BOLD}$*${RESET}";   else echo "$*"; fi; }

# ---------------------------------------------------------------------------
# Test definitions -- each tier is a bash array of test names.
# These names correspond to meson test() targets or executable names
# registered in libvmaf/test/meson.build.
# ---------------------------------------------------------------------------

# Tier 1: Quick Local Tests (<30s)
# Core unit tests that exercise individual components with no heavy I/O.
TIER1_TESTS=(
    test_picture
    test_feature_collector
    test_thread_pool
    test_model
    test_predict
    test_dict
    test_feature_extractor
    test_cpu
    test_ref
    test_feature
    test_ciede
    test_cambi
    test_luminance_tools
    test_cli_parse
    test_psnr
    test_propagate_metadata
    test_format_picture
    test_model_validation
)

# Tier 2: Standard Local Tests (~1-2 min on top of Tier 1)
# Adds SIMD correctness, thread safety, degenerate inputs, framesync,
# precision differential, and Y4M format coverage.
TIER2_TESTS=(
    test_degenerate
    test_thread_safety
    test_format_y4m
    test_simd_adm
    test_simd_vif
    test_simd_motion
    test_simd_cambi
    test_precision_differential
    test_framesync
)

# Tier 3: Extended Local Tests (~5-15 min on top of Tiers 1+2)
# Golden value cross-architecture validation and thread-safety stress tests.
TIER3_TESTS=(
    test_golden_values
    test_thread_safety_stress
)

# Tier 4: Benchmarks (performance data, not pass/fail)
# Registered via meson benchmark() -- not part of `meson test`.
TIER4_BENCHMARKS=(
    bench_simd_perf
)

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------

TIER="quick"
BUILD_DIR=""
VERBOSE=""
SHOW_HELP=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        quick|standard|extended|benchmark|all|ci-status)
            TIER="$1"
            shift
            ;;
        --build-dir)
            if [[ -z "${2:-}" ]]; then
                echo "Error: --build-dir requires a directory argument" >&2
                exit 1
            fi
            BUILD_DIR="$2"
            shift 2
            ;;
        --build-dir=*)
            BUILD_DIR="${1#--build-dir=}"
            shift
            ;;
        --no-color)
            USE_COLOR=false
            shift
            ;;
        --verbose)
            VERBOSE="--verbose"
            shift
            ;;
        --help|-h)
            SHOW_HELP=true
            shift
            ;;
        *)
            echo "Unknown argument: $1" >&2
            echo "Run with --help for usage." >&2
            exit 1
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------

if $SHOW_HELP; then
    cat <<'HELPEOF'
VMAF Master Test Runner
=======================

Usage: run_tests.sh [TIER] [OPTIONS]

Tiers (pick one):
  quick       (default) Tier 1 unit tests, <30 seconds
  standard    Tiers 1+2: adds SIMD, thread safety, degenerate inputs (~1-2 min)
  extended    Tiers 1+2+3: adds golden values, stress tests (~5-15 min)
  benchmark   Tier 4 only: SIMD performance benchmarks (informational)
  all         Tiers 1 through 4
  ci-status   Display CI workflow inventory (no tests run)

Options:
  --build-dir DIR   Meson build directory (auto-detected if omitted)
  --no-color        Disable colored output
  --verbose         Pass --verbose to meson test for full output
  --help, -h        Show this help

The script auto-detects build directories in this order:
  libvmaf/build  ->  libvmaf/build_verify  ->  libvmaf/builddir

If none exist, it offers to create one via `meson setup`.

Tier overview:
  Tier 1 (quick)     Core unit tests -- picture, model, dict, features, etc.
  Tier 2 (standard)  SIMD oracle, thread safety, degenerate, precision, framesync
  Tier 3 (extended)  Cross-arch golden values, thread-safety stress suite
  Tier 4 (benchmark) bench_simd_perf -- performance data, not pass/fail
  Tier 5 (CI-only)   Sanitizers, build-config matrix, fuzzing (GitHub Actions)
  Tier 6 (manual)    Property-based fuzz tests, libFuzzer targets

HELPEOF
    exit 0
fi

# ---------------------------------------------------------------------------
# ci-status: show CI workflow inventory and exit
# ---------------------------------------------------------------------------

if [[ "$TIER" == "ci-status" ]]; then
    echo ""
    color_bold "=== VMAF CI Workflow Inventory ==="
    echo ""

    # Helper to show workflow status
    show_workflow() {
        local file="$1" tier="$2" desc="$3"
        local path="$REPO_ROOT/.github/workflows/$file"
        if [[ -f "$path" ]]; then
            color_green "  [EXISTS]  $file"
        else
            color_red   "  [MISSING] $file"
        fi
        echo "            Tier $tier -- $desc"
        echo ""
    }

    color_cyan "Tier 5: CI-Only Tests (require GitHub Actions infrastructure)"
    echo ""
    show_workflow "sanitizers.yml"   "5" "ASan, UBSan, MSan, TSan builds with clang"
    show_workflow "build-matrix.yml" "5" "Debug, LTO, ASM-disabled, AVX-512-disabled, float-toggle, -O0"
    show_workflow "fuzz.yml"         "5" "libFuzzer targets: Y4M parser, JSON model, picture reader"

    color_cyan "CI Performance & SIMD (also runnable locally as Tiers 2/4)"
    echo ""
    show_workflow "simd-oracle.yml"    "2/5" "SIMD oracle tests on x86_64, ARM64, and C-reference"
    show_workflow "perf-regression.yml" "4/5" "Performance benchmarks on x86_64 and ARM64 runners"

    color_cyan "Existing CI Workflows"
    echo ""
    show_workflow "libvmaf.yml"  "N/A" "Main libvmaf build matrix (gcc, clang, macOS, ARM)"
    show_workflow "docker.yml"   "N/A" "Docker image build"
    show_workflow "ffmpeg.yml"   "N/A" "FFmpeg integration test"
    show_workflow "windows.yml"  "N/A" "Windows build"

    color_cyan "Tier 6: Manual/Specialized (require specific tooling)"
    echo ""
    py_fuzz="$REPO_ROOT/python/test/test_property_fuzz.py"
    if [[ -f "$py_fuzz" ]]; then
        color_green "  [EXISTS]  python/test/test_property_fuzz.py"
    else
        color_yellow "  [ABSENT]  python/test/test_property_fuzz.py"
    fi
    echo "            Property-based tests -- requires: pip install hypothesis"
    echo ""
    fuzz_dir="$REPO_ROOT/libvmaf/fuzz"
    if [[ -d "$fuzz_dir" ]]; then
        color_green "  [EXISTS]  libvmaf/fuzz/"
    else
        color_yellow "  [ABSENT]  libvmaf/fuzz/"
    fi
    echo "            Fuzz targets -- requires: clang with -fsanitize=fuzzer"
    echo ""

    exit 0
fi

# ---------------------------------------------------------------------------
# Locate or create build directory
# ---------------------------------------------------------------------------

find_build_dir() {
    # If user specified --build-dir, resolve it (could be relative to LIBVMAF_DIR
    # or an absolute path).
    if [[ -n "$BUILD_DIR" ]]; then
        # Try as-is first, then relative to libvmaf/
        if [[ -d "$BUILD_DIR" ]]; then
            echo "$BUILD_DIR"
        elif [[ -d "$LIBVMAF_DIR/$BUILD_DIR" ]]; then
            echo "$LIBVMAF_DIR/$BUILD_DIR"
        else
            echo ""
        fi
        return
    fi

    # Auto-detect from the default list
    for dir in "${DEFAULT_BUILD_DIRS[@]}"; do
        if [[ -d "$dir" && -f "$dir/build.ninja" ]]; then
            echo "$dir"
            return
        fi
    done
    echo ""
}

BUILD_DIR_RESOLVED="$(find_build_dir)"

if [[ -z "$BUILD_DIR_RESOLVED" ]]; then
    echo ""
    color_yellow "No meson build directory found."
    echo ""
    echo "Searched:"
    for dir in "${DEFAULT_BUILD_DIRS[@]}"; do
        echo "  - $dir"
    done
    if [[ -n "$BUILD_DIR" ]]; then
        echo "  - $BUILD_DIR (user-specified)"
    fi
    echo ""

    read -rp "Create a new build at libvmaf/build? [y/N] " answer
    if [[ "$answer" =~ ^[Yy] ]]; then
        echo ""
        color_blue "Running: meson setup $LIBVMAF_DIR $LIBVMAF_DIR/build --buildtype debugoptimized -Denable_float=true"
        meson setup "$LIBVMAF_DIR" "$LIBVMAF_DIR/build" \
            --buildtype debugoptimized \
            -Denable_float=true
        echo ""
        color_blue "Building..."
        ninja -C "$LIBVMAF_DIR/build"
        BUILD_DIR_RESOLVED="$LIBVMAF_DIR/build"
    else
        echo "Aborted. Pass --build-dir to specify a build directory." >&2
        exit 1
    fi
fi

echo ""
color_bold "=== VMAF Master Test Runner ==="
echo ""
echo "  Build directory: $BUILD_DIR_RESOLVED"
echo "  Tier:            $TIER"

# ---------------------------------------------------------------------------
# Detect build configuration (enable_float, enable_asm)
# ---------------------------------------------------------------------------

detect_build_option() {
    local option_name="$1"
    local intro_file="$BUILD_DIR_RESOLVED/meson-info/intro-buildoptions.json"
    if [[ -f "$intro_file" ]] && command -v python3 &>/dev/null; then
        python3 -c "
import json, sys
with open('$intro_file') as f:
    opts = {o['name']: o['value'] for o in json.load(f)}
print(opts.get('$option_name', 'unknown'))
" 2>/dev/null || echo "unknown"
    else
        echo "unknown"
    fi
}

ENABLE_FLOAT="$(detect_build_option enable_float)"
ENABLE_ASM="$(detect_build_option enable_asm)"

echo "  enable_float:    $ENABLE_FLOAT"
echo "  enable_asm:      $ENABLE_ASM"
echo ""

# ---------------------------------------------------------------------------
# Test runner helpers
# ---------------------------------------------------------------------------

TOTAL_PASS=0
TOTAL_FAIL=0
TOTAL_SKIP=0
FAILED_TESTS=()
SKIPPED_TESTS=()
START_TIME=$(date +%s)

# Run a single test executable directly (fallback when meson test is not ideal)
run_test_direct() {
    local test_name="$1"
    local test_exe="$BUILD_DIR_RESOLVED/test/$test_name"

    if [[ ! -x "$test_exe" ]]; then
        color_yellow "  SKIP  $test_name (executable not found)"
        TOTAL_SKIP=$((TOTAL_SKIP + 1))
        SKIPPED_TESTS+=("$test_name")
        return 0
    fi

    local test_start
    test_start=$(date +%s%N 2>/dev/null || date +%s)

    local output
    local rc=0
    output=$("$test_exe" 2>&1) || rc=$?

    local test_end
    test_end=$(date +%s%N 2>/dev/null || date +%s)

    # Calculate duration in milliseconds if nanosecond precision is available
    local duration_ms=""
    if [[ ${#test_start} -gt 10 && ${#test_end} -gt 10 ]]; then
        duration_ms=$(( (test_end - test_start) / 1000000 ))
        duration_ms="${duration_ms}ms"
    fi

    if [[ $rc -eq 0 ]]; then
        color_green "  PASS  $test_name ${duration_ms:+($duration_ms)}"
        TOTAL_PASS=$((TOTAL_PASS + 1))
    else
        color_red "  FAIL  $test_name (exit code $rc) ${duration_ms:+($duration_ms)}"
        TOTAL_FAIL=$((TOTAL_FAIL + 1))
        FAILED_TESTS+=("$test_name")
        if [[ -n "$VERBOSE" ]]; then
            echo "$output"
        fi
    fi
}

# Run a list of tests using meson test with --no-suite to avoid suite filtering
# issues. Falls back to direct execution if meson is unavailable.
run_test_list() {
    local tier_label="$1"
    shift
    local tests=("$@")

    if [[ ${#tests[@]} -eq 0 ]]; then
        return 0
    fi

    echo ""
    color_bold "--- $tier_label ---"
    echo ""

    for test_name in "${tests[@]}"; do
        # Skip precision_differential if enable_float is false
        if [[ "$test_name" == "test_precision_differential" && "$ENABLE_FLOAT" == "False" ]]; then
            color_yellow "  SKIP  $test_name (requires enable_float=true)"
            TOTAL_SKIP=$((TOTAL_SKIP + 1))
            SKIPPED_TESTS+=("$test_name (enable_float=false)")
            continue
        fi

        run_test_direct "$test_name"
    done
}

# Run benchmarks via meson benchmark or direct execution
run_benchmarks() {
    echo ""
    color_bold "--- Tier 4: Benchmarks ---"
    echo ""
    color_cyan "  Note: Benchmarks report performance data; they are not pass/fail."
    echo ""

    for bench_name in "${TIER4_BENCHMARKS[@]}"; do
        local bench_exe="$BUILD_DIR_RESOLVED/test/$bench_name"

        if [[ ! -x "$bench_exe" ]]; then
            color_yellow "  SKIP  $bench_name (executable not found)"
            TOTAL_SKIP=$((TOTAL_SKIP + 1))
            SKIPPED_TESTS+=("$bench_name")
            continue
        fi

        color_blue "  Running $bench_name --reps 5 ..."
        echo ""

        # Run benchmark and display output directly
        "$bench_exe" --reps 5 2>&1 | sed 's/^/    /'
        local rc=${PIPESTATUS[0]}
        echo ""

        if [[ $rc -eq 0 ]]; then
            color_green "  DONE  $bench_name"
        else
            color_yellow "  WARN  $bench_name exited with code $rc"
        fi
    done
}

# ---------------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------------

print_summary() {
    local end_time
    end_time=$(date +%s)
    local elapsed=$((end_time - START_TIME))
    local minutes=$((elapsed / 60))
    local seconds=$((elapsed % 60))

    echo ""
    color_bold "========================================"
    color_bold "  Test Summary"
    color_bold "========================================"
    echo ""
    echo "  Tier:     $TIER"
    echo "  Duration: ${minutes}m ${seconds}s"
    echo ""

    if [[ $TOTAL_PASS -gt 0 ]]; then
        color_green "  Passed:  $TOTAL_PASS"
    fi
    if [[ $TOTAL_SKIP -gt 0 ]]; then
        color_yellow "  Skipped: $TOTAL_SKIP"
    fi
    if [[ $TOTAL_FAIL -gt 0 ]]; then
        color_red "  Failed:  $TOTAL_FAIL"
    fi

    if [[ ${#SKIPPED_TESTS[@]} -gt 0 ]]; then
        echo ""
        color_yellow "  Skipped tests:"
        for t in "${SKIPPED_TESTS[@]}"; do
            color_yellow "    - $t"
        done
    fi

    if [[ ${#FAILED_TESTS[@]} -gt 0 ]]; then
        echo ""
        color_red "  Failed tests:"
        for t in "${FAILED_TESTS[@]}"; do
            color_red "    - $t"
        done
    fi

    echo ""

    if [[ $TOTAL_FAIL -gt 0 ]]; then
        color_red "  RESULT: FAIL"
        echo ""
        return 1
    elif [[ $TOTAL_PASS -eq 0 && "$TIER" != "benchmark" ]]; then
        color_yellow "  RESULT: NO TESTS RAN"
        echo ""
        return 1
    else
        color_green "  RESULT: PASS"
        echo ""
        return 0
    fi
}

# ---------------------------------------------------------------------------
# Main dispatch
# ---------------------------------------------------------------------------

case "$TIER" in
    quick)
        run_test_list "Tier 1: Quick Local Tests" "${TIER1_TESTS[@]}"
        ;;
    standard)
        run_test_list "Tier 1: Quick Local Tests"    "${TIER1_TESTS[@]}"
        run_test_list "Tier 2: Standard Local Tests"  "${TIER2_TESTS[@]}"
        ;;
    extended)
        run_test_list "Tier 1: Quick Local Tests"     "${TIER1_TESTS[@]}"
        run_test_list "Tier 2: Standard Local Tests"  "${TIER2_TESTS[@]}"
        run_test_list "Tier 3: Extended Local Tests"  "${TIER3_TESTS[@]}"
        ;;
    benchmark)
        run_benchmarks
        ;;
    all)
        run_test_list "Tier 1: Quick Local Tests"     "${TIER1_TESTS[@]}"
        run_test_list "Tier 2: Standard Local Tests"  "${TIER2_TESTS[@]}"
        run_test_list "Tier 3: Extended Local Tests"  "${TIER3_TESTS[@]}"
        run_benchmarks
        ;;
esac

# Print summary and exit with appropriate code
print_summary
