"""Property-based tests for libvmaf scoring invariants using Hypothesis."""

import math
import subprocess
import tempfile
import os
import struct

import pytest

try:
    from hypothesis import given, settings, HealthCheck, assume
    from hypothesis import strategies as st
    HAS_HYPOTHESIS = True
except ImportError:
    HAS_HYPOTHESIS = False
    # Provide no-op fallbacks so @given/@settings decorators don't crash at import
    def given(**_kw):
        return lambda f: f
    def settings(**_kw):
        def decorator(f):
            return f
        return decorator
    def assume(_):
        pass
    class HealthCheck:
        too_slow = None
    class st:
        @staticmethod
        def tuples(*_a):
            return None
        @staticmethod
        def sampled_from(_a):
            return None
        @staticmethod
        def integers(**_kw):
            return None

from vmaf.config import VmafConfig

pytestmark = pytest.mark.skipif(
    not HAS_HYPOTHESIS,
    reason="hypothesis not installed"
)

# ---- Helpers ----

def _vmafexec_path():
    """Locate the vmaf executable."""
    # Try common build locations
    for candidate in [
        os.path.join(VmafConfig.root_path(), "libvmaf", "build", "tools", "vmaf"),
        os.path.join(VmafConfig.root_path(), "libvmaf", "build_fuzz", "tools", "vmaf"),
    ]:
        if os.path.isfile(candidate):
            return candidate
    pytest.skip("vmaf binary not found")


def _write_y4m(path, width, height, bpc, pix_data):
    """Write a single-frame Y4M file with the given pixel data."""
    chroma = "420" if bpc == 8 else "420p10"
    header = f"YUV4MPEG2 W{width} H{height} F30:1 Ip C{chroma}\n"
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(b"FRAME\n")
        f.write(pix_data)


def _make_constant_y4m(path, width, height, bpc, value):
    """Create a Y4M file with constant pixel value."""
    bps = 2 if bpc > 8 else 1
    luma_sz = width * height * bps
    chroma_w, chroma_h = width // 2, height // 2
    chroma_sz = chroma_w * chroma_h * bps
    if bpc == 8:
        data = bytes([value]) * luma_sz + bytes([128]) * (2 * chroma_sz)
    else:
        val_bytes = struct.pack("<H", value)
        chroma_bytes = struct.pack("<H", 512)
        data = val_bytes * (width * height) + chroma_bytes * (2 * chroma_w * chroma_h)
    _write_y4m(path, width, height, bpc, data)


def _run_vmaf(ref_y4m, dis_y4m, model="vmaf_v0.6.1", feature=None,
              n_threads=1):
    """Run vmaf CLI and return a dict of feature scores."""
    vmaf_bin = _vmafexec_path()
    cmd = [
        vmaf_bin,
        "-r", ref_y4m,
        "-d", dis_y4m,
        "--threads", str(n_threads),
        "--json", "--output", "/dev/stdout",
    ]
    if model:
        cmd += ["-m", f"version={model}"]
    if feature:
        cmd += ["--feature", feature]

    result = subprocess.run(cmd, capture_output=True, timeout=120)
    if result.returncode != 0:
        pytest.fail(f"vmaf failed: {result.stderr.decode()}")

    import json
    output = json.loads(result.stdout)
    scores = {}
    if "frames" in output and len(output["frames"]) > 0:
        for metric in output["frames"][0].get("metrics", {}):
            scores[metric] = output["frames"][0]["metrics"][metric]
    return scores


# ---- Strategies ----

if HAS_HYPOTHESIS:
    # Constrained dimensions: even, reasonable size for CI speed
    valid_dimensions = st.tuples(
        st.sampled_from([32, 64, 128, 192]),   # width (must be even for 420)
        st.sampled_from([32, 64, 108, 128]),   # height (must be even for 420)
    )

    valid_bpc = st.sampled_from([8, 10])

    pixel_value_8bit = st.integers(min_value=0, max_value=255)
    pixel_value_10bit = st.integers(min_value=0, max_value=1023)
else:
    # Dummy values so that @given decorators don't raise NameError at import time.
    # The pytestmark skipif above will prevent these tests from actually running.
    valid_dimensions = None
    valid_bpc = None
    pixel_value_8bit = None
    pixel_value_10bit = None


# ---- P1: VMAF(ref, ref) == 100.0 ----

@given(dims=valid_dimensions, bpc=valid_bpc, val=pixel_value_8bit)
@settings(max_examples=20, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
def test_vmaf_identity_score(dims, bpc, val):
    """VMAF of a frame compared with itself must be 100.0."""
    w, h = dims
    if bpc == 10:
        val = min(val * 4, 1023)
    with tempfile.TemporaryDirectory() as tmpdir:
        ref = os.path.join(tmpdir, "ref.y4m")
        _make_constant_y4m(ref, w, h, bpc, val)
        scores = _run_vmaf(ref, ref, model="vmaf_v0.6.1")
        vmaf_score = scores.get("vmaf", None)
        assert vmaf_score is not None, "VMAF score not found in output"
        assert abs(vmaf_score - 100.0) < 0.01, \
            f"VMAF(ref,ref) = {vmaf_score}, expected 100.0"


# ---- P2: PSNR(ref, ref) == inf ----

@given(dims=valid_dimensions, bpc=valid_bpc, val=pixel_value_8bit)
@settings(max_examples=20, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
def test_psnr_identity_infinity(dims, bpc, val):
    """PSNR of identical frames must be infinity (reported as 60+ dB)."""
    w, h = dims
    if bpc == 10:
        val = min(val * 4, 1023)
    with tempfile.TemporaryDirectory() as tmpdir:
        ref = os.path.join(tmpdir, "ref.y4m")
        _make_constant_y4m(ref, w, h, bpc, val)
        scores = _run_vmaf(ref, ref, model=None, feature="float_psnr")
        psnr = scores.get("psnr_y", scores.get("float_psnr", None))
        assert psnr is not None, "PSNR score not found"
        # libvmaf caps identical-frame PSNR at 60.0 dB
        assert psnr >= 60.0 or math.isinf(psnr), \
            f"PSNR(ref,ref) = {psnr}, expected >= 60.0"


# ---- P3: 0 <= SSIM(ref, dis) <= 1 ----

@given(dims=valid_dimensions, val_ref=pixel_value_8bit,
       val_dis=pixel_value_8bit)
@settings(max_examples=20, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
def test_ssim_bounded(dims, val_ref, val_dis):
    """SSIM must be in [0, 1] for any valid input pair."""
    w, h = dims
    with tempfile.TemporaryDirectory() as tmpdir:
        ref = os.path.join(tmpdir, "ref.y4m")
        dis = os.path.join(tmpdir, "dis.y4m")
        _make_constant_y4m(ref, w, h, 8, val_ref)
        _make_constant_y4m(dis, w, h, 8, val_dis)
        scores = _run_vmaf(ref, dis, model=None, feature="float_ssim")
        ssim = scores.get("float_ssim", None)
        assert ssim is not None, "SSIM score not found"
        assert 0.0 <= ssim <= 1.0, \
            f"SSIM = {ssim}, expected in [0, 1]"


# ---- P4: Determinism ----

@given(dims=valid_dimensions, val_ref=pixel_value_8bit,
       val_dis=pixel_value_8bit)
@settings(max_examples=10, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
def test_vmaf_deterministic(dims, val_ref, val_dis):
    """Repeated VMAF runs on identical input must produce identical scores."""
    w, h = dims
    with tempfile.TemporaryDirectory() as tmpdir:
        ref = os.path.join(tmpdir, "ref.y4m")
        dis = os.path.join(tmpdir, "dis.y4m")
        _make_constant_y4m(ref, w, h, 8, val_ref)
        _make_constant_y4m(dis, w, h, 8, val_dis)
        score1 = _run_vmaf(ref, dis).get("vmaf")
        score2 = _run_vmaf(ref, dis).get("vmaf")
        assert score1 == score2, \
            f"Non-deterministic: run1={score1}, run2={score2}"


# ---- P5: Thread invariance ----

@given(dims=valid_dimensions, val_ref=pixel_value_8bit,
       val_dis=pixel_value_8bit)
@settings(max_examples=10, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
def test_vmaf_thread_invariant(dims, val_ref, val_dis):
    """VMAF score must be identical regardless of thread count."""
    w, h = dims
    with tempfile.TemporaryDirectory() as tmpdir:
        ref = os.path.join(tmpdir, "ref.y4m")
        dis = os.path.join(tmpdir, "dis.y4m")
        _make_constant_y4m(ref, w, h, 8, val_ref)
        _make_constant_y4m(dis, w, h, 8, val_dis)
        score_1t = _run_vmaf(ref, dis, n_threads=1).get("vmaf")
        score_4t = _run_vmaf(ref, dis, n_threads=4).get("vmaf")
        assert score_1t == score_4t, \
            f"Thread-dependent: 1-thread={score_1t}, 4-thread={score_4t}"


# ---- P6: Integer/float agreement ----

@given(dims=valid_dimensions, val_ref=pixel_value_8bit,
       val_dis=pixel_value_8bit)
@settings(max_examples=10, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
def test_int_float_agreement(dims, val_ref, val_dis):
    """Integer and float feature extractors must agree within tolerance."""
    w, h = dims
    with tempfile.TemporaryDirectory() as tmpdir:
        ref = os.path.join(tmpdir, "ref.y4m")
        dis = os.path.join(tmpdir, "dis.y4m")
        _make_constant_y4m(ref, w, h, 8, val_ref)
        _make_constant_y4m(dis, w, h, 8, val_dis)

        int_scores = _run_vmaf(ref, dis, model="vmaf_v0.6.1")
        float_scores = _run_vmaf(ref, dis, model="vmaf_float_v0.6.1")

        int_vmaf = int_scores.get("vmaf")
        float_vmaf = float_scores.get("vmaf")
        assume(int_vmaf is not None and float_vmaf is not None)

        # Allow up to 2% relative difference between integer and float paths.
        denom = max(abs(int_vmaf), abs(float_vmaf), 1.0)
        rel_err = abs(int_vmaf - float_vmaf) / denom
        assert rel_err < 0.02, \
            f"Int/float divergence: int={int_vmaf}, float={float_vmaf}, " \
            f"rel_err={rel_err:.6f}"
