# Benchmarking Instance

## Running a benchmark

From the repo root on the **development machine** (not the instance):

```bash
# Benchmark current HEAD (uses whatever is built on the instance)
bench/run-bench-x86.sh --label baseline

# Push local libvmaf/src changes, rebuild, then benchmark
bench/run-bench-x86.sh --label my-optimisation --sync

# Change number of timing repetitions (default 7)
bench/run-bench-x86.sh --label baseline --runs 10
```

The script handles everything: starts the instance if stopped, resets machine state, generates test video, runs 7 timed repetitions, produces a flamegraph. Each run lands in `bench-results/<timestamp>_<label>/` (gitignored) **and** automatically appends one row to `bench-results/history-x86.csv` (git-tracked).

**Prerequisites:** `~/.ssh/vmaf-bench.pem` and AWS CLI configured (both already set up on this machine).

## Reviewing benchmark history

```bash
bench/compare-x86.sh                             # last 20 runs, all labels
bench/compare-x86.sh --label baseline            # filter to one label
bench/compare-x86.sh --label baseline --last 5   # last 5 rows of that label
```

Output is a table with colour-coded Δ% columns vs the previous row:

```
timestamp         label              commit     time_s    Δtime   ipc    Δipc  miss%    Δmiss          cycles
20260226_164701   baseline           332dde62   2.7204      —    3.16      —      ?       —  28,881,962,623
20260226_164850   baseline           332dde62   2.7174  -0.11%   3.17  +0.32%  10.49%    —  28,832,459,771
```

- **Green** = improvement (time/miss% down, IPC up)
- **Red** = regression

`bench-results/history-x86.csv` is committed to git. After a meaningful run or series of runs, commit it to preserve the record:

```bash
git add bench-results/history-x86.csv
git commit -m "bench: add <label> results"
```

Per-run artifact directories (`bench-results/<timestamp>_<label>/`) are gitignored — they contain large files (flamegraph SVG, perf data). The `summary.txt` inside each is still human-readable if you want the full detail of a specific run.

## Typical optimization workflow

```bash
# 1. Record baseline
bench/run-bench-x86.sh --label baseline

# 2. Make source changes, push to instance, rebuild, record
bench/run-bench-x86.sh --label my-opt --sync

# 3. Compare
bench/compare-x86.sh --last 5

# 4. Commit the history when satisfied
git add bench-results/history-x86.csv && git commit -m "bench: my-opt results"
```

## Connection

```bash
ssh -i ~/.ssh/vmaf-bench.pem ubuntu@100.53.91.42
```

Private key is at `~/.ssh/vmaf-bench.pem` on the development machine (this machine).
The host key is already accepted; no confirmation prompt will appear.

## Instance details

| Field | Value |
|---|---|
| Instance ID | `i-025bf8165235858b4` |
| Type | `c6a.metal` (AMD EPYC 7R13, 192 logical cores) |
| Region / AZ | `us-east-1b` |
| Public IP | `100.53.91.42` (elastic IP not set — may change if stopped/started) |
| AMI | Ubuntu 22.04 LTS (`ami-04680790a315cd58d`) |
| Cost | ~$3.89/hr (on-demand); **terminate when not in use** |
| EBS | 100 GB gp3, deleted on termination |

## CPU / benchmarking configuration

Applied at boot and re-applied manually after first launch:

| Setting | Value | How to verify |
|---|---|---|
| CPU governor | `performance` | `cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor` |
| AMD turbo boost | Disabled (`0`) | `cat /sys/devices/system/cpu/cpufreq/boost` |
| `perf_event_paranoid` | `-1` (full PMU access) | `cat /proc/sys/kernel/perf_event_paranoid` |
| `kptr_restrict` | `0` (kernel symbols visible) | `cat /proc/sys/kernel/kptr_restrict` |
| ISA | AVX2, no AVX-512 | `grep -o "avx[^ ]*" /proc/cpuinfo \| sort -u` |

**These settings are not persisted across a stop/start.** Re-apply with:

```bash
sudo su -c "
  echo 0   > /sys/devices/system/cpu/cpufreq/boost
  echo -1  > /proc/sys/kernel/perf_event_paranoid
  echo 0   > /proc/sys/kernel/kptr_restrict
"
```

## What is installed

- `gcc`, `g++`, `make`, `cmake`, `nasm`, `git`
- `perf` (v6.8.12, via `linux-tools-generic`)
- `numactl`, `cpufrequtils`, `sysstat`
- `python3`, `python3-venv`, `python3-dev`
- VMAF cloned and built at `/opt/vmaf/` (see below)

## VMAF build on the instance

```
/opt/vmaf/                        ← git clone of Netflix/vmaf (master)
/opt/vmaf/libvmaf/build/tools/vmaf   ← release binary (enable_float=true)
/opt/vmaf/libvmaf/build/test/     ← C unit test binaries
```

Built with:
```bash
meson setup libvmaf/build libvmaf --buildtype release -Denable_float=true
ninja -C libvmaf/build
```

To rebuild after source changes:
```bash
cd /opt/vmaf
git pull
ninja -C libvmaf/build
```

## Running a perf profile

Download test videos first (run once):

```bash
cd /opt/vmaf
python3 -m venv .venv
.venv/bin/pip install -q requests
python3 -c "
import urllib.request, os
base = 'https://media.xiph.org/video/derf/y4m/'
os.makedirs('testdata', exist_ok=True)
# Use any representative 1080p YUV source; or copy from the dev machine
"
```

Or copy from the development machine:
```bash
# On the development machine:
scp -i ~/.ssh/vmaf-bench.pem \
  /root/Development/vmaf/resource/dataset/NFLX_dataset_public/yuv/src01_hrc00_576x324.yuv \
  /root/Development/vmaf/resource/dataset/NFLX_dataset_public/yuv/src01_hrc01_576x324.yuv \
  ubuntu@100.53.91.42:/opt/vmaf/testdata/
```

Profile the full inference pipeline:
```bash
sudo perf record -F 99 -g -o perf.data -- \
  /opt/vmaf/libvmaf/build/tools/vmaf \
    --reference /opt/vmaf/testdata/src01_hrc00_576x324.yuv \
    --distorted /opt/vmaf/testdata/src01_hrc01_576x324.yuv \
    --width 576 --height 324 --pixel_format 420 --bitdepth 8 \
    --model version=vmaf_v0.6.1 \
    --output /dev/null

perf report --stdio -i perf.data | head -60
```

Profile with hardware counters (cache misses, IPC):
```bash
perf stat -e cycles,instructions,cache-misses,LLC-load-misses,LLC-store-misses \
  /opt/vmaf/libvmaf/build/tools/vmaf \
    --reference /opt/vmaf/testdata/src01_hrc00_576x324.yuv \
    --distorted /opt/vmaf/testdata/src01_hrc01_576x324.yuv \
    --width 576 --height 324 --pixel_format 420 --bitdepth 8 \
    --model version=vmaf_v0.6.1 \
    --output /dev/null
```

Pin to isolated cores to reduce scheduler noise (use cores 0–3, avoid hyperthreads on 96–99):
```bash
sudo numactl --cpunodebind=0 --membind=0 \
  taskset -c 0-3 perf stat -e cycles,instructions,cache-misses ...
```

## Managing the instance

```bash
# Check status
aws ec2 describe-instances --instance-ids i-025bf8165235858b4 \
  --query 'Reservations[0].Instances[0].{State:State.Name,IP:PublicIpAddress}' \
  --output table

# Stop (preserves disk, stops billing for compute — ~$0.01/hr for EBS remains)
aws ec2 stop-instances --instance-ids i-025bf8165235858b4

# Start again (IP will change — fetch new IP after starting)
aws ec2 start-instances --instance-ids i-025bf8165235858b4
aws ec2 wait instance-running --instance-ids i-025bf8165235858b4
aws ec2 describe-instances --instance-ids i-025bf8165235858b4 \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text

# Terminate permanently (destroys disk, stops all billing)
aws ec2 terminate-instances --instance-ids i-025bf8165235858b4
```

## AWS account notes

- Credentials configured in `~/.aws/credentials` on the development machine
- Account ID: `677276090156` (root account)
- vCPU quota for standard instances raised to 192 (request `6c115708b1c646d49ea92549b3f7826fkgtvMP2d`, status: `CASE_CLOSED`)
- Security group: `sg-0f9469de730fb56e0` (`vmaf-bench-sg`, SSH/22 open)
- Key pair name in AWS: `vmaf-bench`
