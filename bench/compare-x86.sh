#!/usr/bin/env bash
# =============================================================================
# bench/compare.sh [--label LABEL] [--last N]
#
# Prints benchmark history from bench-results/history-x86.csv as a table with
# percent-delta columns vs. the immediately preceding row (or prior row of the
# same label when --label is given).
#
# Options:
#   --label LABEL   filter to rows with this label (default: all labels)
#   --last N        show only the last N rows (default: 20)
#
# Columns:
#   timestamp   date/time of the run
#   label       --label passed to run-bench.sh
#   commit      short git commit hash
#   time_s      mean wall-clock time (seconds)
#   Δtime       % change vs previous row (negative = faster = good)
#   ipc         instructions per cycle (higher = better)
#   Δipc        % change
#   miss%       cache-miss rate (lower = better)
#   Δmiss       % change
#   cycles      total CPU cycles
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HISTORY="$REPO_ROOT/bench-results/history-x86.csv"
LABEL=""
LAST=20

while [[ $# -gt 0 ]]; do
    case "$1" in
        --label) LABEL="$2"; shift ;;
        --last)  LAST="$2";  shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
    shift
done

[[ -f "$HISTORY" ]] || { echo "No history yet — run bench/run-bench-x86.sh first."; exit 1; }

python3 - "$HISTORY" "$LABEL" "$LAST" << 'PYEOF'
import sys, csv

history_file, label_filter, last_n = sys.argv[1], sys.argv[2], int(sys.argv[3])

with open(history_file) as f:
    rows = list(csv.DictReader(f))

if label_filter:
    rows = [r for r in rows if r["label"] == label_filter]

rows = rows[-last_n:]

def pct_delta(new_row, old_row, key, invert=False):
    """Return coloured delta string. invert=True means lower is better."""
    try:
        nv, ov = new_row.get(key, ""), old_row.get(key, "")
        if nv == "" or ov == "":
            return "    —  "
        n, o = float(nv), float(ov)
        if o == 0:
            return "    —  "
        pct = (n - o) / o * 100
        sign = "+" if pct >= 0 else ""
        s = f"{sign}{pct:.2f}%"
        # Green if improvement, red if regression
        # For time/cycles/cache-miss: lower is better (invert=True → positive delta is bad)
        # For ipc: higher is better (invert=False → positive delta is good)
        good = pct < 0 if invert else pct > 0
        if abs(pct) < 0.05:
            colour = ""          # noise floor — no colour
        elif good:
            colour = "\033[32m"  # green
        else:
            colour = "\033[31m"  # red
        reset = "\033[0m" if colour else ""
        return f"{colour}{s:>8}{reset}"
    except (ValueError, KeyError):
        return "    —  "

# Header
print(f"\n{'timestamp':<17} {'label':<18} {'commit':<9} "
      f"{'time_s':>8} {'Δtime':>9} "
      f"{'ipc':>5} {'Δipc':>9} "
      f"{'miss%':>6} {'Δmiss':>9} "
      f"{'cycles':>15}")
print("-" * 102)

for i, r in enumerate(rows):
    prev = rows[i - 1] if i > 0 else None

    dt = pct_delta(r, prev, "time_s",        invert=True)  if prev else "       —"
    di = pct_delta(r, prev, "ipc",           invert=False) if prev else "       —"
    dm = pct_delta(r, prev, "cache_miss_pct",invert=True)  if prev else "       —"

    def fmt_float(v, fmt):
        try:
            return format(float(v), fmt)
        except (ValueError, TypeError):
            return "?"

    try:
        cycles_fmt = f"{int(r['cycles']):>15,}"
    except (ValueError, KeyError):
        cycles_fmt = f"{'?':>15}"

    miss_str = fmt_float(r.get('cache_miss_pct', ''), '.2f')
    miss_str = f"{miss_str}%" if miss_str != "?" else "   ?"

    print(f"{r['timestamp']:<17} {r['label']:<18} {r['git_commit'][:8]:<9} "
          f"{fmt_float(r['time_s'], '8.4f')} {dt} "
          f"{fmt_float(r['ipc'], '5.2f')} {di} "
          f"{miss_str:>6} {dm} "
          f"{cycles_fmt}")

print()
print("Δ columns: green = improvement, red = regression")
print("  time/miss%: lower is better   ipc: higher is better")
PYEOF
