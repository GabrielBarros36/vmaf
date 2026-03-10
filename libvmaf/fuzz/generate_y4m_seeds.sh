#!/bin/bash
# generate_y4m_seeds.sh
# Creates minimal 2x2 Y4M files for each supported chroma type.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SEED_DIR="$SCRIPT_DIR/corpus/y4m"
mkdir -p "$SEED_DIR"

for CHROMA in 420 420jpeg 420mpeg2 420p10 420p12 420paldv 422 422p10 422p12 \
              444 444p10 444p12 444alpha 411 mono; do
    HEADER="YUV4MPEG2 W2 H2 F30:1 Ip C${CHROMA}"
    # Compute frame data size based on chroma type and write zero-filled data.
    case "$CHROMA" in
        420|420jpeg|420mpeg2) BYTES=6 ;;    # 4 + 1 + 1
        420p10|420p12)        BYTES=12 ;;   # (4 + 1 + 1) * 2
        420paldv)             BYTES=6 ;;
        422)                  BYTES=8 ;;    # 4 + 2 + 2
        422p10|422p12)        BYTES=16 ;;   # (4 + 2 + 2) * 2
        444)                  BYTES=12 ;;   # 4 + 4 + 4
        444alpha)             BYTES=12 ;;
        444p10|444p12)        BYTES=24 ;;   # (4 + 4 + 4) * 2
        411)                  BYTES=6 ;;    # 4 + 1 + 1
        mono)                 BYTES=4 ;;    # 4
        *)                    BYTES=12 ;;
    esac
    {
        printf '%s\n' "$HEADER"
        printf 'FRAME\n'
        dd if=/dev/zero bs=1 count=$BYTES 2>/dev/null
    } > "$SEED_DIR/seed_${CHROMA}.y4m"
done
