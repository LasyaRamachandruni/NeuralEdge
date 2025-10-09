#!/usr/bin/env bash
set -euo pipefail

# Usage: ./power_jetson.sh <command...>
# Runs tegrastats in background, computes median power and joules/inf if bench.py prints latencies.

TMP=$(mktemp)
sudo nvpmodel -m 0 || true
sudo jetson_clocks || true

tegrastats --interval 200 --logfile "$TMP" &
TS_PID=$!

"$@" --power_log "$TMP"

kill $TS_PID || true

python3 - <<'PY'
import re, numpy as np, sys, statistics
path = sys.argv[1]
watts = []
for line in open(path):
    m = re.search(r'POM_5V_IN\s*([0-9.]+)W', line)
    if m:
        watts.append(float(m.group(1)))
if watts:
    print('Median_W', np.median(watts))
else:
    print('No power samples found')
PY
"$TMP"

rm -f "$TMP"

