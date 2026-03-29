#!/usr/bin/env bash
# =============================================================================
# bench/record-result.sh <result-dir>
#
# Parses a completed bench run directory and appends one row to
# bench-results/history-x86.csv. Creates the file with a header if it doesn't
# exist yet.
#
# Called automatically at the end of bench/run-bench.sh, but can also be run
# manually to backfill existing result directories:
#
#   for d in bench-results/*/; do
#       [ -f "$d/perf-stat.txt" ] && bench/record-result.sh "$d"
#   done
# =============================================================================
set -euo pipefail

RESULT_DIR="${1:?Usage: record-result.sh <result-dir>}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HISTORY="$REPO_ROOT/bench-results/history-x86.csv"

PERF="$RESULT_DIR/perf-stat.txt"
ENV_FILE="$RESULT_DIR/env.txt"

[[ -f "$PERF" ]]     || { echo "[record] No perf-stat.txt in $RESULT_DIR — skipping"; exit 0; }
[[ -f "$ENV_FILE" ]] || { echo "[record] No env.txt in $RESULT_DIR — skipping"; exit 0; }

# --- Extract fields ---
TIMESTAMP=$(basename "$RESULT_DIR" | cut -d_ -f1-2)
LABEL=$(basename "$RESULT_DIR" | cut -d_ -f3-)
GIT_COMMIT=$(awk '/vmaf_commit/{print $2}' "$ENV_FILE")
CPU_FREQ=$(awk '/cpu_freq_mhz/{print $2}' "$ENV_FILE")

# Time: "2.71741 +- 0.00321 seconds time elapsed  ( +-  0.12% )"
TIME_S=$(awk '/seconds time elapsed/{print $1}' "$PERF")
TIME_SD=$(awk '/seconds time elapsed/{print $3}' "$PERF")
TIME_CV=$(awk '/seconds time elapsed/{
    match($0, /\( \+- *([0-9.]+)%/, a); print a[1]}' "$PERF")

# Hardware counters (strip commas from formatted numbers)
CYCLES=$(awk '/[0-9,] +cycles/{gsub(",","",$1); print $1; exit}' "$PERF")
INSTRUCTIONS=$(awk '/[0-9,] +instructions/{gsub(",","",$1); print $1; exit}' "$PERF")
IPC=$(awk '/insn per cycle/{
    for(i=1;i<=NF;i++) if($i~/^[0-9]+\.[0-9]+$/ && $(i+1)=="insns") {print $i; exit}
    }' "$PERF")
# Fallback: IPC appears as "# 3.17 insn per cycle"
[[ -z "$IPC" ]] && IPC=$(awk '/insn per cycle/{
    match($0, /#\s+([0-9.]+)\s+insn/, a); print a[1]}' "$PERF")
CACHE_MISS_PCT=$(awk '/cache-misses/{
    match($0, /([0-9.]+)% of all/, a); print a[1]; exit}' "$PERF")
BRANCH_MISSES=$(awk '/[0-9,] +branch-misses/{gsub(",","",$1); print $1; exit}' "$PERF")

# --- Write header if file is new ---
if [[ ! -f "$HISTORY" ]]; then
    printf 'timestamp,label,git_commit,cpu_freq_mhz,time_s,time_stddev_s,time_cv_pct,cycles,instructions,ipc,cache_miss_pct,branch_misses\n' \
        > "$HISTORY"
fi

# --- Append row ---
printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$TIMESTAMP" "$LABEL" "$GIT_COMMIT" "$CPU_FREQ" \
    "$TIME_S" "$TIME_SD" "$TIME_CV" \
    "$CYCLES" "$INSTRUCTIONS" "$IPC" \
    "$CACHE_MISS_PCT" "$BRANCH_MISSES" \
    >> "$HISTORY"

echo "[bench] Recorded to history-x86.csv  ($TIMESTAMP  $LABEL  ${TIME_S}s)"
