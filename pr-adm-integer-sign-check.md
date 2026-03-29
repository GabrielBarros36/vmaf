# integer_adm: replace float sign check with integer comparison in adm_angle_flag

## Summary

In `adm_decouple` and `adm_decouple_s123`, the first condition of the
`angle_flag` computation was:

```c
(((float)ot_dp / 4096.0) >= 0.0f)
```

This promotes an `int64` to `float` and divides by a positive constant solely to
test the sign. Since dividing by a positive constant preserves sign, this is
algebraically equivalent to:

```c
(ot_dp >= 0)
```

The replacement eliminates an `int64`→`float` conversion and a float division on
the hot path. The second condition (magnitude comparison) is left in float to
preserve the original rounding behavior.

## Why this matters

`adm_decouple` and `adm_decouple_s123` together account for ~10.5% of total VMAF
CPU time. The sign check is the first condition in a short-circuit `&&`, so when
`ot_dp < 0` (the common case), the entire `angle_flag` evaluation terminates at
the first branch. Replacing the float promotion with an integer comparison makes
this fast path significantly cheaper.

## Benchmarks

Measured on AWS `c6a.metal` (AMD EPYC 7R13, 2.65 GHz locked, turbo disabled,
performance governor, pinned to 4 cores on NUMA node 0). Workload: 1080p 48-frame
YUV video scored with default VMAF model, 7 repetitions per configuration (CV <0.15%).

| Configuration | Time (s) | Δ vs baseline | Cycles | IPC |
|---|---|---|---|---|
| Baseline (master) | 2.762 | — | 28.76B | 3.18 |
| **This change** | **2.707** | **−2.0%** | **28.16B** | **3.24** |

IPC improves from 3.18 → 3.24, consistent with removing a float pipeline stall on
a frequently-taken branch.

This benchmarking infrastructure lives on the `bench-history-tracking` branch, which
contains a harness for reproducible perf measurements, historical results in
`bench-results/history-x86.csv`, and detailed optimization reports in `plans/`. Several
other branches (`top-optimizations`, `sec7-9-only`) contain additional SIMD and
infrastructure optimizations being evaluated separately.

## Correctness

- **67/67 Python regression tests pass** with bit-identical scores across all
  feature extractors and models
- The transformation is provably equivalent: for any integer `x` and positive
  constant `c`, `sign(x / c) == sign(x)`

## Test plan

- [ ] CI passes (no score regressions)
- [ ] `meson test` in libvmaf builds and passes on x86-64
