#!/usr/bin/env python3
"""Compare benchmark results against stored baselines.

Usage:
    check_perf_regression.py <baseline.json> <current.json>

Exit codes:
    0 - All benchmarks within tolerance (PASS)
    1 - One or more confirmed regressions detected (FAIL)
    2 - Usage error
"""

import json
import sys

REGRESSION_THRESHOLD = 1.10   # 10%
WARNING_THRESHOLD = 1.25      # 25%
FALLBACK_THRESHOLD = 3.00     # 3x (likely C fallback)
MIN_SPEEDUP = 1.5             # SIMD must be at least 1.5x faster than C
CV_QUALITY_GATE = 5.0         # CV% above this = noisy measurement


def load_json(path):
    with open(path) as f:
        return json.load(f)


def find_bench(benchmarks, name, variant):
    for b in benchmarks:
        if b["name"] == name and b["variant"] == variant:
            return b
    return None


def check_regression(baseline_path, current_path):
    baseline = load_json(baseline_path)
    current = load_json(current_path)

    exit_code = 0
    warnings = []
    failures = []

    current_benchmarks = current["benchmarks"]
    baseline_benchmarks = baseline["benchmarks"]

    for cb in current_benchmarks:
        name, variant = cb["name"], cb["variant"]
        bb = find_bench(baseline_benchmarks, name, variant)
        if bb is None:
            warnings.append(
                f"{name} [{variant}]: no baseline found (new benchmark?)")
            continue

        ratio = cb["median_ns"] / bb["median_ns"]
        cv = cb.get("cv_pct", 0)

        if cv > CV_QUALITY_GATE:
            warnings.append(
                f"{name} [{variant}]: CV={cv:.1f}% exceeds {CV_QUALITY_GATE}% "
                f"quality gate (noisy measurement, skipping pass/fail)")
            continue

        if ratio > FALLBACK_THRESHOLD:
            failures.append(
                f"{name} [{variant}]: {ratio:.2f}x slower than baseline "
                f"(CRITICAL: likely fallback to C path)")
            exit_code = 1
        elif ratio > WARNING_THRESHOLD:
            # Check statistical significance
            if cb["min_ns"] > bb["median_ns"]:
                failures.append(
                    f"{name} [{variant}]: {ratio:.2f}x slower than baseline "
                    f"(confirmed: min > baseline median)")
                exit_code = 1
            else:
                warnings.append(
                    f"{name} [{variant}]: {ratio:.2f}x slower than baseline "
                    f"(not confirmed: min <= baseline median)")
        elif ratio > REGRESSION_THRESHOLD:
            if cb["min_ns"] > bb["median_ns"]:
                failures.append(
                    f"{name} [{variant}]: {ratio:.2f}x slower than baseline")
                exit_code = 1
            else:
                warnings.append(
                    f"{name} [{variant}]: {ratio:.2f}x slower but within noise "
                    f"(min <= baseline median)")

    # Check SIMD-vs-C speedup floor
    for cb in current_benchmarks:
        if cb["variant"] == "c":
            continue
        c_bench = find_bench(current_benchmarks, cb["name"], "c")
        if c_bench is None:
            continue
        speedup = c_bench["median_ns"] / cb["median_ns"]
        if speedup < 1.1:
            failures.append(
                f"{cb['name']} [{cb['variant']}]: speedup vs C = {speedup:.2f}x "
                f"(FAIL: no measurable speedup)")
            exit_code = 1
        elif speedup < MIN_SPEEDUP:
            warnings.append(
                f"{cb['name']} [{cb['variant']}]: speedup vs C = {speedup:.2f}x "
                f"(below expected {MIN_SPEEDUP}x)")

    # Report
    for w in warnings:
        print(f"WARNING: {w}")
    for f in failures:
        print(f"FAIL: {f}")
    if not warnings and not failures:
        print("PASS: All benchmarks within tolerance")

    return exit_code


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <baseline.json> <current.json>")
        sys.exit(2)
    sys.exit(check_regression(sys.argv[1], sys.argv[2]))
