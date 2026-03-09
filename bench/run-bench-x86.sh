#!/usr/bin/env bash
# =============================================================================
# VMAF Benchmark Runner
# Drives the AWS bare-metal instance (c6a.metal) end-to-end:
#   1. Ensures instance is running
#   2. Applies clean, reproducible machine state
#   3. Optionally syncs local libvmaf/src changes and rebuilds
#   4. Generates synthetic 1080p test video (cached on instance EBS)
#   5. Warm-up run (page cache)
#   6. Timed benchmark: perf stat --repeat (timing + hardware counters)
#   7. Flamegraph via perf record + FlameGraph tools
#   8. Copies results locally and prints summary
#
# Usage:
#   bench/run-bench.sh [--sync] [--label LABEL] [--runs N]
#
#   --sync          rsync libvmaf/src/ to instance and rebuild before benchmarking
#   --label LABEL   tag for this run, e.g. "baseline" or "ssim-avx2" (default: git HEAD)
#   --runs N        number of timed repetitions for perf stat (default: 7)
#
# Results land in bench-results/<timestamp>_<label>/ with:
#   env.txt          machine state snapshot
#   perf-stat.txt    timing + hardware counters (N runs, mean ± stddev)
#   perf-report.txt  per-function CPU hotspots
#   flamegraph.svg   interactive flamegraph (open in browser)
#   summary.txt      extracted key numbers — easiest for an agent to read
# =============================================================================
set -euo pipefail

# ---- Configuration -----------------------------------------------------------
INSTANCE_ID="i-0022bb66e6b04211e"
KEY="$HOME/.ssh/vmaf-bench.pem"
SSH_OPTS="-i $KEY -o StrictHostKeyChecking=no -o BatchMode=yes -o ServerAliveInterval=30"
REMOTE_USER="ubuntu"
REMOTE_ROOT="/opt/vmaf"

# Benchmark parameters
RUNS=7
BENCH_CORES="2,3,4,5"       # Physical cores on NUMA node 0, avoiding 0/1 (IRQ sink)
NUMA_NODE=0
THREADS=8                   # 2× physical cores for SMT utilization
PERF_FREQ=99                # Hz for perf record sampling

# Paths on the instance
REF_YUV="/opt/vmaf/testdata/bench_ref_1920x1080.yuv"
DIS_YUV="/opt/vmaf/testdata/bench_dis_1920x1080.yuv"
RELEASE_BIN="$REMOTE_ROOT/libvmaf/build/tools/vmaf"
PROF_BIN="$REMOTE_ROOT/libvmaf/build-prof/tools/vmaf"

# ---- Parse arguments ---------------------------------------------------------
SYNC=0
LABEL=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --sync)          SYNC=1 ;;
        --label)         LABEL="$2"; shift ;;
        --runs)          RUNS="$2"; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
    shift
done

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
[[ -z "$LABEL" ]] && LABEL=$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo "unknown")
TIMESTAMP=$(date -u +%Y%m%d_%H%M%S)
RESULT_DIR="$REPO_ROOT/bench-results/${TIMESTAMP}_${LABEL}"
mkdir -p "$RESULT_DIR"

log()  { echo "[bench] $*"; }
die()  { echo "[bench] ERROR: $*" >&2; exit 1; }
remote() { ssh $SSH_OPTS ${REMOTE_USER}@${IP} "$@"; }

# =============================================================================
# STEP 1 — Ensure instance is running, get IP
# =============================================================================
log "Step 1/7 — Checking instance $INSTANCE_ID..."

STATE=$(aws ec2 describe-instances --instance-ids "$INSTANCE_ID" \
    --query 'Reservations[0].Instances[0].State.Name' --output text)

if [[ "$STATE" == "stopped" ]]; then
    log "  Instance is stopped — starting..."
    aws ec2 start-instances --instance-ids "$INSTANCE_ID" > /dev/null
    aws ec2 wait instance-running --instance-ids "$INSTANCE_ID"
elif [[ "$STATE" != "running" ]]; then
    die "Instance is in unexpected state: $STATE"
fi

IP=$(aws ec2 describe-instances --instance-ids "$INSTANCE_ID" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
log "  Running at $IP"

log "  Waiting for SSH..."
until ssh $SSH_OPTS ${REMOTE_USER}@${IP} true 2>/dev/null; do sleep 3; done

# =============================================================================
# STEP 2 — Apply clean, reproducible machine state
# =============================================================================
log "Step 2/7 — Applying clean machine state..."

remote sudo sh << 'CLEAN'
set -e

# CPU frequency governor: performance on all cores (may not exist on VMs)
for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    [ -f "$g" ] && echo performance > "$g" || true
done

# AMD turbo boost: off (prevents frequency jitter between runs; missing on VMs)
[ -f /sys/devices/system/cpu/cpufreq/boost ] && echo 0 > /sys/devices/system/cpu/cpufreq/boost || true

# IRQ affinity: steer all IRQs away from benchmark cores 2-5
# (keep them on cores 0-1 which we reserve for the OS)
for irq_aff in /proc/irq/*/smp_affinity_list; do
    echo "0-1" > "$irq_aff" 2>/dev/null || true
done

# ASLR: off (eliminates memory layout variability between runs)
echo 0 > /proc/sys/kernel/randomize_va_space

# Transparent Huge Pages: always (consistent, reduces TLB pressure)
echo always > /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null || true

# perf access
echo -1 > /proc/sys/kernel/perf_event_paranoid
echo  0 > /proc/sys/kernel/kptr_restrict

# Drop page cache, dentries, inodes — clean slate for every run
sync
echo 3 > /proc/sys/vm/drop_caches

# Wait for load to settle (any lingering processes to drain)
sleep 2
CLEAN

# Capture and verify machine state
remote bash << 'ENVSNAP' > "$RESULT_DIR/env.txt"
echo "timestamp:    $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "hostname:     $(hostname)"
echo "kernel:       $(uname -r)"
echo "cpu:          $(awk -F': ' '/model name/{print $2; exit}' /proc/cpuinfo)"
echo "logical_cpus: $(nproc)"
echo "numa_nodes:   $(ls -d /sys/devices/system/node/node* 2>/dev/null | wc -l)"
echo "governor:     $(cat /sys/devices/system/cpu/cpu2/cpufreq/scaling_governor 2>/dev/null || echo 'N/A (VM)')"
echo "cpu_freq_mhz: $(awk -F': ' '/cpu MHz/{print $2; exit}' /proc/cpuinfo)"
echo "boost:        $(cat /sys/devices/system/cpu/cpufreq/boost 2>/dev/null || echo 'N/A (VM)')"
echo "aslr:         $(cat /proc/sys/kernel/randomize_va_space)"
echo "thp:          $(cat /sys/kernel/mm/transparent_hugepage/enabled | grep -o '\[.*\]' | tr -d '[]')"
echo "perf_paranoid:$(cat /proc/sys/kernel/perf_event_paranoid)"
echo "avx_flags:    $(grep -o 'avx[^ ]*' /proc/cpuinfo | sort -u | tr '\n' ' ')"
echo "loadavg_1m:   $(awk '{print $1}' /proc/loadavg)"
echo "vmaf_commit:  $(sudo git -C /opt/vmaf rev-parse HEAD)"
ENVSNAP

# Extract load from the already-downloaded env.txt (avoids SSH quoting complexity)
LOAD=$(awk '/loadavg_1m/{print $2}' "$RESULT_DIR/env.txt")
LOAD_INT=${LOAD%.*}
if [[ ${LOAD_INT:-0} -ge 4 ]]; then
    die "Load average too high ($LOAD) — won't produce reliable results. Wait and retry."
fi

log "  Machine state OK (load=$LOAD):"
sed 's/^/    /' "$RESULT_DIR/env.txt"

# =============================================================================
# STEP 3 — Sync changes and rebuild (optional)
# =============================================================================
if [[ $SYNC -eq 1 ]]; then
    log "Step 3/7 — Syncing libvmaf/ sources to instance..."
    rsync -a --delete -e "ssh $SSH_OPTS" \
        "$REPO_ROOT/libvmaf/src/" \
        "${REMOTE_USER}@${IP}:${REMOTE_ROOT}/libvmaf/src/"
    rsync -a --delete --rsync-path="sudo rsync" -e "ssh $SSH_OPTS" \
        "$REPO_ROOT/libvmaf/test/" \
        "${REMOTE_USER}@${IP}:${REMOTE_ROOT}/libvmaf/test/"
    rsync -a --rsync-path="sudo rsync" -e "ssh $SSH_OPTS" \
        "$REPO_ROOT/libvmaf/meson.build" \
        "${REMOTE_USER}@${IP}:${REMOTE_ROOT}/libvmaf/meson.build"
    rsync -a --rsync-path="sudo rsync" -e "ssh $SSH_OPTS" \
        "$REPO_ROOT/libvmaf/meson_options.txt" \
        "${REMOTE_USER}@${IP}:${REMOTE_ROOT}/libvmaf/meson_options.txt"

    log "  Rebuilding release binary..."
    remote sudo bash << 'REBUILD'
set -e
cd /opt/vmaf
export PATH="/opt/vmaf/.venv/bin:/usr/bin:/bin:$PATH"
.venv/bin/meson configure libvmaf/build -Denable_lto=false -Dc_args="-march=znver3"
CC=/usr/bin/gcc CXX=/usr/bin/g++ ninja -C libvmaf/build

# Build or rebuild profiling binary (frame pointers for accurate flamegraph)
if [ ! -d libvmaf/build-prof ]; then
    .venv/bin/meson setup libvmaf/build-prof libvmaf \
        --buildtype release -Denable_float=true \
        -Dc_args="-fno-omit-frame-pointer -O3 -march=znver3"
fi
CC=/usr/bin/gcc CXX=/usr/bin/g++ ninja -C libvmaf/build-prof tools/vmaf
echo "  Rebuild done."
REBUILD
else
    log "Step 3/7 — Skipping sync (pass --sync to push local changes)"

    # Ensure profiling binary exists even without --sync (build once)
    HAS_PROF=$(remote test -f "$PROF_BIN" && echo yes || echo no)
    if [[ "$HAS_PROF" == "no" ]]; then
        log "  Profiling binary not found — building it now (one-time)..."
        remote sudo bash << 'BUILDPROF'
set -e
cd /opt/vmaf
export PATH="/opt/vmaf/.venv/bin:/usr/bin:/bin:$PATH"
if [ ! -d libvmaf/build-prof ]; then
    .venv/bin/meson setup libvmaf/build-prof libvmaf \
        --buildtype release -Denable_float=true \
        -Dc_args="-fno-omit-frame-pointer -O3 -march=znver3"
fi
# Build only the vmaf binary, not tests (avoids vcs_version.h race in parallel build)
CC=/usr/bin/gcc CXX=/usr/bin/g++ ninja -C libvmaf/build-prof tools/vmaf
BUILDPROF
    fi
fi

# Install FlameGraph tools if not present (one-time, quiet)
remote bash << 'FLAMEINST'
[ -d /opt/FlameGraph ] || sudo git clone --depth=1 https://github.com/brendangregg/FlameGraph /opt/FlameGraph
FLAMEINST

# =============================================================================
# STEP 4 — Generate synthetic test video (cached on EBS, survives stop/start)
# =============================================================================
log "Step 4/7 — Preparing test video..."

remote sudo bash << 'GENVIDEO'
set -e
mkdir -p /opt/vmaf/testdata

if [ ! -f /opt/vmaf/testdata/bench_ref_1920x1080.yuv ]; then
    echo "  Generating synthetic 1080p test video (120 frames, ~356 MB each)..."
    python3 << 'PYEOF'
import os

W, H, N = 1920, 1080, 120
frame_size = W * H * 3 // 2  # 4:2:0: Y plane + U/2 + V/2

# Generate two independent streams of random bytes (no numpy needed).
# Different content for ref vs distorted ensures VMAF computes non-trivial scores.
with open('/opt/vmaf/testdata/bench_ref_1920x1080.yuv', 'wb') as f:
    for _ in range(N):
        f.write(os.urandom(frame_size))

with open('/opt/vmaf/testdata/bench_dis_1920x1080.yuv', 'wb') as f:
    for _ in range(N):
        f.write(os.urandom(frame_size))

print("  Generated: %d frames, %.0f MB each" % (N, frame_size * N / 1e6))
PYEOF
else
    echo "  Test video already present — reusing."
fi
GENVIDEO

# =============================================================================
# STEP 5 — Warm-up run (fills page cache; timing not recorded)
# =============================================================================
log "Step 5/7 — Warm-up run (page cache fill)..."

remote sudo taskset -c "$BENCH_CORES" \
    numactl --cpunodebind=$NUMA_NODE --membind=$NUMA_NODE \
    "$RELEASE_BIN" \
        --reference  "$REF_YUV" \
        --distorted  "$DIS_YUV" \
        --width 1920 --height 1080 --pixel_format 420 --bitdepth 8 \
        --model version=vmaf_v0.6.1 \
        --threads "$THREADS" \
        --output /dev/null 2>&1 | grep -E "VMAF score|error|ERR" || true

log "  Warm-up done."

# =============================================================================
# STEP 6 — Timed benchmark: perf stat --repeat
# Runs the binary $RUNS times; reports mean ± stddev for every counter
# and for wall-clock time. Page cache is warm from step 5.
# =============================================================================
log "Step 6/7 — Running timed benchmark ($RUNS repetitions)..."

# Detect if hardware PMU counters are available (VMs often lack them)
HW_PMU=$(remote sudo perf stat -e cycles -- true 2>&1)
if echo "$HW_PMU" | grep -q '<not supported>'; then
    log "  Hardware PMU not available (VM) — using software events only"
    PERF_EVENTS="task-clock,context-switches,page-faults"
else
    PERF_EVENTS="cycles,instructions,cache-references,cache-misses,branch-misses"
fi

remote sudo perf stat \
    --repeat "$RUNS" \
    --event "$PERF_EVENTS" \
    taskset -c "$BENCH_CORES" \
    numactl --cpunodebind=$NUMA_NODE --membind=$NUMA_NODE \
    "$RELEASE_BIN" \
        --reference  "$REF_YUV" \
        --distorted  "$DIS_YUV" \
        --width 1920 --height 1080 --pixel_format 420 --bitdepth 8 \
        --model version=vmaf_v0.6.1 \
        --threads "$THREADS" \
        --output /dev/null \
    2>&1 | tee "$RESULT_DIR/perf-stat.txt" >/dev/null

log "  perf stat done."

# =============================================================================
# STEP 7 — Flamegraph via perf record
# Uses the profiling binary (frame pointers) for accurate call stacks.
# =============================================================================
log "Step 7/7 — Generating flamegraph..."

remote sudo bash << FLAMECMD
set -e
# Use cpu-clock (software) if hardware cycles not available (VM environment)
PERF_EVENT="cycles"
perf stat -e cycles -- true 2>&1 | grep -q '<not supported>' && PERF_EVENT="cpu-clock"
perf record -e \$PERF_EVENT -F $PERF_FREQ --call-graph=fp -o /tmp/vmaf-perf.data -- \\
    taskset -c "$BENCH_CORES" \\
    numactl --cpunodebind=$NUMA_NODE --membind=$NUMA_NODE \\
    "$PROF_BIN" \\
        --reference  "$REF_YUV" \\
        --distorted  "$DIS_YUV" \\
        --width 1920 --height 1080 --pixel_format 420 --bitdepth 8 \\
        --model version=vmaf_v0.6.1 \\
        --threads "$THREADS" \\
        --output /dev/null \\
    2>&1

# Collapse stacks and render flamegraph
perf script -i /tmp/vmaf-perf.data \\
    | /opt/FlameGraph/stackcollapse-perf.pl \\
    | /opt/FlameGraph/flamegraph.pl --title "VMAF — $LABEL" \\
    > /tmp/vmaf-flamegraph.svg

# Per-function hotspot report (top functions by sample count)
perf report --stdio -i /tmp/vmaf-perf.data -n --percent-limit=0.3 \\
    > /tmp/vmaf-perf-report.txt 2>&1
FLAMECMD

# Copy result files back
scp $SSH_OPTS "${REMOTE_USER}@${IP}:/tmp/vmaf-flamegraph.svg"  "$RESULT_DIR/flamegraph.svg"
scp $SSH_OPTS "${REMOTE_USER}@${IP}:/tmp/vmaf-perf-report.txt" "$RESULT_DIR/perf-report.txt"

log "  Flamegraph done."

# =============================================================================
# Build summary.txt — clean, agent-readable extract of key numbers
# =============================================================================
{
    echo "VMAF Benchmark Summary"
    echo "======================"
    echo ""
    echo "Label:   $LABEL"
    echo "Date:    $TIMESTAMP"
    echo ""

    echo "--- Machine State ---"
    cat "$RESULT_DIR/env.txt"
    echo ""

    echo "--- Timing and Hardware Counters ($RUNS runs, mean ± stddev) ---"
    grep -E "seconds time elapsed|cycles|instructions|cache-references|cache-misses|branch-miss|insns per cycle|task-clock|context-switches|page-faults" \
        "$RESULT_DIR/perf-stat.txt" || true
    echo ""

    echo "--- Top CPU Hotspots ---"
    # Skip comment header lines, print the first call tree block (up to first blank line after data)
    grep -v "^#" "$RESULT_DIR/perf-report.txt" 2>/dev/null | head -50 || true
    echo ""

    echo "--- Result Files ---"
    echo "  $RESULT_DIR/"
    echo "  ├── env.txt          machine state"
    echo "  ├── perf-stat.txt    raw perf stat output (timing + hw counters)"
    echo "  ├── perf-report.txt  per-function hotspots"
    echo "  ├── flamegraph.svg   open in browser for interactive call tree"
    echo "  └── summary.txt      this file"
} > "$RESULT_DIR/summary.txt"

# =============================================================================
# Print to terminal
# =============================================================================
echo ""
echo "========================================================"
printf "  BENCHMARK RESULTS: %s\n" "$LABEL"
echo "========================================================"
cat "$RESULT_DIR/summary.txt"
echo ""
echo "Full results: $RESULT_DIR"

# Append this run to the persistent time-series record
"$REPO_ROOT/bench/record-result-x86.sh" "$RESULT_DIR"
