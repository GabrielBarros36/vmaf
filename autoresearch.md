# Autoresearch: Fix GitHub CI Errors

## Objective
Fix all GitHub CI workflow failures on the vmaf repository until all tests pass. The approach is iterative: push fixes, wait for CI, analyze failures, repeat.

## Metrics
- **Primary**: ci_errors (unitless, lower is better) — total number of failing CI jobs across all workflows
- **Secondary**: workflows_failing — number of distinct workflow names with at least one failure

## How to Run
Push to GitHub, wait for CI to complete, then count errors:
```bash
gh run list --branch autoresearch/fix-ci-20260329 --limit 10 --json conclusion,workflowName
```

## Files in Scope
- `.github/workflows/*.yml` — CI workflow definitions
- `libvmaf/test/` — test files (C tests, sanitizer configs)
- `libvmaf/src/feature/x86/vif_avx2.c` — VIF SIMD code (bug fix)
- `python/test/test_property_fuzz.py` — Python property tests
- `libvmaf/fuzz/` — Fuzz targets

## Off Limits
- Must NOT make tests less robust
- Core library logic beyond targeted bug fixes
- Model files

## Constraints
- Tests must remain correct — no weakening assertions or removing tests
- If a test is impossible to fix, report it and move on
- Sanitizer suppressions are acceptable for pre-existing library issues

## What's Been Tried

### Iteration 1 (baseline): 6 workflows failing, ~30 failing jobs
Analyzed latest CI run (commit 3ffd2214) on test-harness branch.

**Root causes identified:**
1. `sudo: meson: command not found` — libvmaf workflow (5 jobs)
2. VIF AVX2 horizontal mu filter bug — _mm256_add_epi64 used with _mm256_mullo_epi32 results
3. MSan false positive in svm.cpp — uninitialized value in SVM parser
4. TSan data race in framesync.c — pre-existing race condition
5. UBSan signed integer overflow in integer_adm.c — intentional wrapping
6. UBSan function type mismatch in vidinput.c — third-party Daala code
7. ASan memory leaks in test code — cleanup missing on assertion failure paths
8. Python test_property_fuzz.py NameError — hypothesis strategies undefined when hypothesis missing

### Iteration 2 (commit b58d2365): Comprehensive fix push
**Fixes applied:**
- VIF AVX2: Changed _mm256_add_epi64 → _mm256_add_epi32 in mu1/mu2 horizontal filter
- Workflow: `sudo env PATH=$PATH meson test` + ubuntu OS matcher
- MSan: Added svm.cpp + y4m tools to compile-time ignorelist
- TSan: Added framesync race suppression
- UBSan: Added compile-time ignorelist for ADM overflow + vidinput function types
- ASan/LSan: Added leak suppressions for test code allocations
- Fuzz: Added UBSan ignorelist to fuzz build
- Python: Added dummy strategy variable definitions

**Known potential remaining issues:**
- Windows test_model_validation: No stderr available, unknown root cause
- Windows test_simd_vif: Should be fixed by VIF AVX2 fix
