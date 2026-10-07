#!/usr/bin/env bash
# Usage (on a Jetson): ./power_jetson.sh <command...>
# Logs tegrastats while <command> runs and prints the median board input power.
# NOTE: written against the documented tegrastats formats; not yet run on hardware.
#   Jetson Nano (JetPack 4):  "POM_5V_IN 1797/1797"      (mW, current/average)
#   JetPack 5+ modules:       "VDD_IN 4000mW/4000mW"
set -euo pipefail

LOG=$(mktemp)
sudo nvpmodel -m 0 || true
sudo jetson_clocks || true

tegrastats --interval 200 --logfile "$LOG" &
TS_PID=$!
trap 'kill $TS_PID 2>/dev/null || true; rm -f "$LOG"' EXIT

"$@"

kill $TS_PID 2>/dev/null || true
sleep 0.5

python3 - "$LOG" <<'PY'
import re, statistics, sys
mw = []
for line in open(sys.argv[1]):
    m = re.search(r'(?:POM_5V_IN|VDD_IN)\s+(\d+)(?:mW)?/', line)
    if m:
        mw.append(int(m.group(1)))
if mw:
    print(f"samples={len(mw)} median_input_power_W={statistics.median(mw) / 1000:.3f}")
else:
    print("No power samples found in tegrastats log")
PY
