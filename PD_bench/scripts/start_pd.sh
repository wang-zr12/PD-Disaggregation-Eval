#!/usr/bin/env bash
# Launch the PD-disaggregated stack on a 2× A100 host:
#   GPU 0 → prefill (kv_producer, :8100)
#   GPU 1 → decode  (kv_consumer, :8200)
#   proxy on :8000
# Usage:  bash scripts/start_pd.sh  [MODEL_DIR]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${1:-${PD_BENCH_MODEL_DIR:-/data/models}}"
MODEL="${PD_BENCH_MODEL:-Qwen2.5-Coder-7B}"
MODEL_PATH="${MODEL_DIR}/${MODEL}"

LOG_DIR="${PD_BENCH_LOGS:-/tmp/pd_bench}"
mkdir -p "$LOG_DIR"
PID_DIR=/tmp/pd_bench
mkdir -p "$PID_DIR"

export VLLM_HOST_IP="${VLLM_HOST_IP:-127.0.0.1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

KV_PRODUCER='{"kv_connector":"PyNcclConnector","kv_role":"kv_producer","kv_rank":0,"kv_parallel_size":2,"kv_buffer_size":5000000000}'
KV_CONSUMER='{"kv_connector":"PyNcclConnector","kv_role":"kv_consumer","kv_rank":1,"kv_parallel_size":2,"kv_buffer_size":5000000000}'

echo "[pd] model=${MODEL_PATH}"
echo "[pd] starting PREFILL  GPU0  :8100  → ${LOG_DIR}/prefill.log"
CUDA_VISIBLE_DEVICES=0 \
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" --served-model-name qwen-coder \
    --dtype float16 --max-model-len 8192 \
    --gpu-memory-utilization 0.85 \
    --port 8100 --host 0.0.0.0 --disable-log-requests \
    --kv-transfer-config "$KV_PRODUCER" \
    > "$LOG_DIR/prefill.log" 2>&1 &
echo $! > "$PID_DIR/pd_prefill.pid"

echo "[pd] starting DECODE   GPU1  :8200  → ${LOG_DIR}/decode.log"
CUDA_VISIBLE_DEVICES=1 \
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" --served-model-name qwen-coder \
    --dtype float16 --max-model-len 8192 \
    --gpu-memory-utilization 0.85 \
    --port 8200 --host 0.0.0.0 --disable-log-requests \
    --kv-transfer-config "$KV_CONSUMER" \
    > "$LOG_DIR/decode.log" 2>&1 &
echo $! > "$PID_DIR/pd_decode.pid"

echo "Waiting for both vLLM servers to become healthy ..."
for i in $(seq 1 180); do
    if curl -fs http://localhost:8100/health >/dev/null 2>&1 \
       && curl -fs http://localhost:8200/health >/dev/null 2>&1; then
        echo "Both backends ready after ${i}×5s"
        break
    fi
    sleep 5
    [ $i -eq 180 ] && { echo "Timeout."; tail -100 "$LOG_DIR/prefill.log" "$LOG_DIR/decode.log"; exit 1; }
done

echo "[pd] starting PROXY    :8000  → ${LOG_DIR}/proxy.log"
cd "$ROOT"
python -m src.disagg_proxy --port 8000 \
    --prefill-url http://localhost:8100 \
    --decode-url  http://localhost:8200 \
    > "$LOG_DIR/proxy.log" 2>&1 &
echo $! > "$PID_DIR/pd_proxy.pid"

for i in $(seq 1 30); do
    if curl -fs http://localhost:8000/health >/dev/null 2>&1; then
        echo "Proxy ready. PD stack is up at http://localhost:8000"
        exit 0
    fi
    sleep 2
done
echo "Proxy timeout."; tail -60 "$LOG_DIR/proxy.log"; exit 1
