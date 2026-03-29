#!/usr/bin/env bash
# =============================================================================
# VMAF PGO Benchmark Runner
# 3-phase PGO: instrumented build → training → optimized build → benchmark
# =============================================================================
set -euo pipefail

INSTANCE_ID="i-0022bb66e6b04211e"
KEY="$HOME/.ssh/vmaf-bench.pem"
SSH_OPTS="-i $KEY -o StrictHostKeyChecking=no -o BatchMode=yes -o ServerAliveInterval=30"
REMOTE_USER="ubuntu"
REMOTE_ROOT="/opt/vmaf"

RUNS=5
THREADS=8
LABEL="pgo-O3"
BENCH_CORES="2,3,4,5"
NUMA_NODE=0
PERF_FREQ=99

REF_YUV="/opt/vmaf/testdata/bench_ref_1920x1080.yuv"
DIS_YUV="/opt/vmaf/testdata/bench_dis_1920x1080.yuv"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --runs)    RUNS="$2"; shift ;;
        --threads) THREADS="$2"; shift ;;
        --label)   LABEL="$2"; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
    shift
done

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TIMESTAMP=$(date -u +%Y%m%d_%H%M%S)
RESULT_DIR="$REPO_ROOT/bench-results/${TIMESTAMP}_${LABEL}"
mkdir -p "$RESULT_DIR"

log()  { echo "[pgo-bench] $*"; }
die()  { echo "[pgo-bench] ERROR: $*" >&2; exit 1; }
remote() { ssh $SSH_OPTS ${REMOTE_USER}@${IP} "$@"; }

# ---- Get instance IP ----
IP=$(aws ec2 describe-instances --instance-ids "$INSTANCE_ID" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
log "Instance at $IP"
until ssh $SSH_OPTS ${REMOTE_USER}@${IP} true 2>/dev/null; do sleep 3; done

# ---- Sync sources ----
log "Phase 0: Syncing sources..."
rsync -a --delete -e "ssh $SSH_OPTS" \
    "$REPO_ROOT/libvmaf/src/" \
    "${REMOTE_USER}@${IP}:${REMOTE_ROOT}/libvmaf/src/"
rsync -a --rsync-path="sudo rsync" -e "ssh $SSH_OPTS" \
    "$REPO_ROOT/libvmaf/meson.build" \
    "${REMOTE_USER}@${IP}:${REMOTE_ROOT}/libvmaf/meson.build"
rsync -a --rsync-path="sudo rsync" -e "ssh $SSH_OPTS" \
    "$REPO_ROOT/libvmaf/meson_options.txt" \
    "${REMOTE_USER}@${IP}:${REMOTE_ROOT}/libvmaf/meson_options.txt"

# ---- Phase 1+2: Build instrumented, train, build optimized ----
log "Phase 1: Instrumented build + training..."
remote sudo bash << 'PGOALL'
set -e
cd /opt/vmaf
export PATH="/opt/vmaf/.venv/bin:/usr/bin:/bin:$PATH"
PGO_DIR="/tmp/vmaf-pgo"

echo "=== Phase 1a: Instrumented build ==="
rm -rf "$PGO_DIR" libvmaf/build-pgo
mkdir -p "$PGO_DIR"

# Use env vars so meson applies profile flags to both compile AND link
CC=/usr/bin/gcc CXX=/usr/bin/g++ \
    CFLAGS="-march=znver3 -fprofile-generate=$PGO_DIR" \
    LDFLAGS="-fprofile-generate=$PGO_DIR" \
    meson setup libvmaf/build-pgo libvmaf \
    --buildtype release \
    --optimization 3 \
    -Ddefault_library=static \
    -Denable_avx512=false

ninja -C libvmaf/build-pgo tools/vmaf

echo "=== Phase 1b: Training run ==="
taskset -c 2,3,4,5 numactl --cpunodebind=0 --membind=0 \
    libvmaf/build-pgo/tools/vmaf \
    --reference  /opt/vmaf/testdata/bench_ref_1920x1080.yuv \
    --distorted  /opt/vmaf/testdata/bench_dis_1920x1080.yuv \
    --width 1920 --height 1080 --pixel_format 420 --bitdepth 8 \
    --model version=vmaf_v0.6.1 \
    --threads 8 \
    --output /dev/null 2>&1

GCDA_COUNT=$(find "$PGO_DIR" -name '*.gcda' | wc -l)
echo "  Profile files: $GCDA_COUNT .gcda"
[ "$GCDA_COUNT" -gt 0 ] || { echo "ERROR: no profile data"; exit 1; }

echo "=== Phase 2: PGO-optimized build ==="
# Reconfigure to use profile data (meson configure preserves env-var flags)
meson configure libvmaf/build-pgo \
    -Dc_args="-march=znver3 -O3 -fprofile-use=$PGO_DIR -fprofile-correction"
# Clear LDFLAGS for the optimized build (no gcov needed)
ninja -C libvmaf/build-pgo tools/vmaf

echo "=== PGO build complete ==="
PGOALL

log "PGO build complete. Running benchmark..."

# ---- Apply clean machine state ----
remote sudo sh << 'CLEAN'
set -e
for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    [ -f "$g" ] && echo performance > "$g" || true
done
[ -f /sys/devices/system/cpu/cpufreq/boost ] && echo 0 > /sys/devices/system/cpu/cpufreq/boost || true
for irq_aff in /proc/irq/*/smp_affinity_list; do
    echo "0-1" > "$irq_aff" 2>/dev/null || true
done
echo 0 > /proc/sys/kernel/randomize_va_space
echo always > /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null || true
echo -1 > /proc/sys/kernel/perf_event_paranoid
echo  0 > /proc/sys/kernel/kptr_restrict
sync
echo 3 > /proc/sys/vm/drop_caches
sleep 2
CLEAN

# Capture env
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

PGO_BIN="$REMOTE_ROOT/libvmaf/build-pgo/tools/vmaf"

# ---- Warm-up ----
log "Warm-up run..."
remote sudo taskset -c "$BENCH_CORES" \
    numactl --cpunodebind=$NUMA_NODE --membind=$NUMA_NODE \
    "$PGO_BIN" \
        --reference  "$REF_YUV" \
        --distorted  "$DIS_YUV" \
        --width 1920 --height 1080 --pixel_format 420 --bitdepth 8 \
        --model version=vmaf_v0.6.1 \
        --threads "$THREADS" \
        --output /dev/null 2>&1 | grep -E "VMAF score|error|ERR" || true

# ---- Timed benchmark ----
log "Running benchmark ($RUNS repetitions, $THREADS threads)..."
PERF_EVENTS="task-clock,context-switches,page-faults"
remote sudo perf stat \
    --repeat "$RUNS" \
    --event "$PERF_EVENTS" \
    taskset -c "$BENCH_CORES" \
    numactl --cpunodebind=$NUMA_NODE --membind=$NUMA_NODE \
    "$PGO_BIN" \
        --reference  "$REF_YUV" \
        --distorted  "$DIS_YUV" \
        --width 1920 --height 1080 --pixel_format 420 --bitdepth 8 \
        --model version=vmaf_v0.6.1 \
        --threads "$THREADS" \
        --output /dev/null \
    2>&1 | tee "$RESULT_DIR/perf-stat.txt" >/dev/null

# ---- Flamegraph (using existing profiling binary) ----
PROF_BIN="$REMOTE_ROOT/libvmaf/build-prof/tools/vmaf"
log "Generating flamegraph..."
remote sudo bash << FLAMECMD
set -e
PERF_EVENT="cycles"
perf stat -e cycles -- true 2>&1 | grep -q '<not supported>' && PERF_EVENT="cpu-clock"
perf record -e \$PERF_EVENT -F $PERF_FREQ --call-graph=fp -o /tmp/vmaf-perf.data -- \
    taskset -c "$BENCH_CORES" \
    numactl --cpunodebind=$NUMA_NODE --membind=$NUMA_NODE \
    "$PGO_BIN" \
        --reference  "$REF_YUV" \
        --distorted  "$DIS_YUV" \
        --width 1920 --height 1080 --pixel_format 420 --bitdepth 8 \
        --model version=vmaf_v0.6.1 \
        --threads "$THREADS" \
        --output /dev/null \
    2>&1
perf script -i /tmp/vmaf-perf.data \
    | /opt/FlameGraph/stackcollapse-perf.pl \
    | /opt/FlameGraph/flamegraph.pl --title "VMAF — $LABEL (PGO)" \
    > /tmp/vmaf-flamegraph.svg
perf report --stdio -i /tmp/vmaf-perf.data -n --percent-limit=0.3 \
    > /tmp/vmaf-perf-report.txt 2>&1
FLAMECMD

scp $SSH_OPTS "${REMOTE_USER}@${IP}:/tmp/vmaf-flamegraph.svg"  "$RESULT_DIR/flamegraph.svg"
scp $SSH_OPTS "${REMOTE_USER}@${IP}:/tmp/vmaf-perf-report.txt" "$RESULT_DIR/perf-report.txt"

# ---- Summary ----
{
    echo "VMAF PGO Benchmark Summary"
    echo "=========================="
    echo ""
    echo "Label:   $LABEL"
    echo "Date:    $TIMESTAMP"
    echo "Build:   PGO (-O3 -march=znver3 -fprofile-use)"
    echo "Threads: $THREADS"
    echo ""
    echo "--- Machine State ---"
    cat "$RESULT_DIR/env.txt"
    echo ""
    echo "--- Timing ($RUNS runs, mean ± stddev) ---"
    grep -E "seconds time elapsed|task-clock|context-switches|page-faults" \
        "$RESULT_DIR/perf-stat.txt" || true
    echo ""
    echo "--- Top CPU Hotspots ---"
    grep -v "^#" "$RESULT_DIR/perf-report.txt" 2>/dev/null | head -50 || true
} > "$RESULT_DIR/summary.txt"

echo ""
echo "========================================================"
printf "  PGO BENCHMARK RESULTS: %s\n" "$LABEL"
echo "========================================================"
cat "$RESULT_DIR/summary.txt"
echo ""
echo "Full results: $RESULT_DIR"

# Record
"$REPO_ROOT/bench/record-result-x86.sh" "$RESULT_DIR"

# ---- Restore normal build ----
log "Restoring normal build..."
remote sudo bash << 'RESTORE'
set -e
cd /opt/vmaf
export PATH="/opt/vmaf/.venv/bin:/usr/bin:/bin:$PATH"
meson configure libvmaf/build -Dc_args="-march=znver3"
CC=/usr/bin/gcc CXX=/usr/bin/g++ ninja -C libvmaf/build
RESTORE
log "Done."
