#!/usr/bin/env bash
# =============================================================================
# Remote Build & Test on EC2 x86_64 Instance
# Syncs local changes, rebuilds, and runs the test suite on a c6a instance.
#
# Usage:
#   ./scripts/remote_test.sh [quick|standard|extended]
#   ./scripts/remote_test.sh build-only
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
IP=52.90.229.209
KEY="$HOME/.ssh/vmaf-bench.pem"
SSH="ssh -i $KEY -o StrictHostKeyChecking=no -o BatchMode=yes -o ServerAliveInterval=30"
REMOTE_USER="ubuntu"
REMOTE_DIR="/home/ubuntu/vmaf"
TIER="${1:-standard}"

log() { echo "[remote] $*"; }

# Step 1: Sync source files
log "Syncing sources to $IP..."
rsync -a --delete -e "$SSH" \
    "$REPO_ROOT/libvmaf/src/" \
    "${REMOTE_USER}@${IP}:${REMOTE_DIR}/libvmaf/src/"

rsync -a --delete -e "$SSH" \
    "$REPO_ROOT/libvmaf/test/" \
    "${REMOTE_USER}@${IP}:${REMOTE_DIR}/libvmaf/test/"

rsync -a -e "$SSH" \
    "$REPO_ROOT/libvmaf/include/" \
    "${REMOTE_USER}@${IP}:${REMOTE_DIR}/libvmaf/include/"

rsync -a -e "$SSH" \
    "$REPO_ROOT/scripts/run_tests.sh" \
    "${REMOTE_USER}@${IP}:${REMOTE_DIR}/scripts/run_tests.sh"

log "Sync complete."

# Step 2: Rebuild
log "Rebuilding on remote..."
$SSH ${REMOTE_USER}@${IP} << 'BUILD'
set -e
cd /home/ubuntu/vmaf/libvmaf

# Reconfigure if build dir doesn't exist
if [ ! -d build ]; then
    meson setup build --buildtype release -Denable_float=true -Ddefault_library=shared
fi

ninja -C build 2>&1
echo "BUILD_OK"
BUILD

if [[ "$TIER" == "build-only" ]]; then
    log "Build-only mode - done."
    exit 0
fi

# Step 3: Run tests
log "Running test tier: $TIER"
$SSH ${REMOTE_USER}@${IP} "cd /home/ubuntu/vmaf && LD_LIBRARY_PATH=/home/ubuntu/vmaf/libvmaf/build/src ./scripts/run_tests.sh $TIER --build-dir /home/ubuntu/vmaf/libvmaf/build"
