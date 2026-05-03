#!/usr/bin/env bash
# Launch the single-GPU colocated baseline on GPU 0.
# Usage:  bash scripts/start_colocated.sh  [MODEL_DIR]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${1:-${PD_BENCH_MODEL_DIR:-/data/models}}"
MODEL="${PD_BENCH_MODEL:-Qwen2.5-Coder-7B}"
LOG="${PD_BENCH_LOGS:-/tmp/pd_bench}/colocated.log"
mkdir -p "$(dirname "$LOG")"

cd "$ROOT"
echo "[colocated] model=${MODEL_DIR}/${MODEL}  log=${LOG}"

CUDA_VISIBLE_DEVICES=0 \
python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_DIR}/${MODEL}" \
    --served-model-name qwen-coder \
    --dtype float16 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.88 \
    --port 8000 --host 0.0.0.0 \
    --disable-log-requests \
    > "$LOG" 2>&1 &

PID=$!
echo "$PID" > /tmp/pd_bench/colocated.pid
echo "[colocated] PID=$PID  port=8000"
echo "Waiting for /health ..."
for i in $(seq 1 120); do
    if curl -fs http://localhost:8000/health >/dev/null 2>&1; then
        echo "Ready after ${i}×5s"
        exit 0
    fi
    sleep 5
done
echo "Timed out. Tail of log:"; tail -100 "$LOG"; exit 1
