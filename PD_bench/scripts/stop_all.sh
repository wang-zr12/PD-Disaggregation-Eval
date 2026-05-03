#!/usr/bin/env bash
# Kill anything started by start_colocated.sh / start_pd.sh.
set -u
PID_DIR=/tmp/pd_bench
for f in colocated pd_prefill pd_decode pd_proxy; do
    pf="$PID_DIR/${f}.pid"
    if [ -f "$pf" ]; then
        pid=$(cat "$pf" || true)
        if [ -n "${pid:-}" ] && kill -0 "$pid" 2>/dev/null; then
            echo "killing $f PID=$pid"
            kill "$pid" 2>/dev/null || true
            sleep 1
            kill -9 "$pid" 2>/dev/null || true
        fi
        rm -f "$pf"
    fi
done

# Belt and braces: anything still holding our ports
for port in 8000 8100 8200; do
    pids=$(lsof -ti:$port 2>/dev/null || true)
    if [ -n "$pids" ]; then
        echo "killing residual on :$port  ($pids)"
        kill -9 $pids 2>/dev/null || true
    fi
done

echo "All PD_bench processes stopped."
