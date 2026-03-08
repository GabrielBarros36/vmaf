# Optimization #4: Link-Time Optimization (LTO)

## Summary

Added an `enable_lto` meson build option that enables Link-Time Optimization
for both C and C++ compilation and linking via the `-flto` compiler/linker flag.

## What Changed

- **`libvmaf/meson_options.txt`**: Added `enable_lto` boolean option (default: `false`).
- **`libvmaf/meson.build`**: When `enable_lto` is true, passes `-flto` to both
  compiler and linker for C and C++ via `add_project_arguments` and
  `add_project_link_arguments`.

## Usage

```bash
cd libvmaf
meson setup build --buildtype release -Denable_lto=true
ninja -C build
```

To enable LTO on an existing build directory:

```bash
meson configure build -Denable_lto=true
ninja -C build
```

## How LTO Works

Link-Time Optimization allows the compiler to perform whole-program optimization
at link time, when it has visibility into all translation units simultaneously.
This enables:

- **Cross-TU inlining**: Functions defined in one `.c` file can be inlined into
  callers in another `.c` file.
- **Dead code elimination**: Unreachable functions and variables across TUs are
  removed.
- **Interprocedural constant propagation**: Constants can propagate across
  translation unit boundaries.

Typical performance improvement is 5-15% depending on workload characteristics.

## Compatibility

- Works with GCC and Clang on Linux, macOS, and other Unix-like systems.
- MSVC uses a different mechanism (`/GL` + `/LTCG`); this implementation uses
  `-flto` which is GCC/Clang-specific. MSVC users should use meson's built-in
  `b_lto=true` option instead.
- NASM-assembled SIMD object files are not affected by LTO (they pass through
  the linker unchanged).

## Test Results

All 17 meson-registered tests pass with LTO enabled (tested on GCC 13.3.0,
aarch64 Linux).
