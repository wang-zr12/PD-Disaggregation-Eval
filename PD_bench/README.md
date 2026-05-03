# PD_bench — Prefill–Decode Disaggregation Benchmark

Cloud SSH workflow (2× NVIDIA A100 40 GB on one host) for evaluating
**Prefill–Decode disaggregation on vLLM 0.7.3** against a single-GPU
colocated baseline.

This produces three artefacts:

1. **Headline 20 QPS mixed-workload comparison** — drives the `~40% P99 TTFT
   cut` and `SLO (<1s) compliance 55% → 92%` numbers.
2. **30-config sweep** — 2 architectures × 3 workload profiles × 5 QPS
   levels. Profiles' ISL/OSL distributions are drawn from HumanEval
   (short, function-completion-shaped) and SWE-bench Lite (long-context).
3. **Roofline + KV-cache transfer cost analysis** — analytic break-even
   curve giving the QPS × ISL boundary above which PD becomes profitable.

## Layout

```
PD_bench/
├── src/
│   ├── config.py            paths, model spec, GPU/link spec, PD layout
│   ├── workload.py          tokenizer-precise prompt generator
│   ├── workload_dist.py     HumanEval + SWE-bench distributions, profiles
│   ├── benchmark.py         RequestResult, streaming send_request
│   ├── load_gen.py          open-loop Poisson QPS driver
│   ├── metrics.py           summary metrics, slo_compare, pd_uplift
│   ├── storage.py           parquet/csv/json save + *_latest.parquet
│   ├── colocated_server.py  single-GPU vLLM launcher
│   ├── pd_server.py         dual vLLM launcher (kv_producer + kv_consumer)
│   ├── disagg_proxy.py      aiohttp proxy: prime prefill, stream decode
│   ├── experiments.py       run_one / run_headline_20qps / run_30config_sweep
│   ├── analytics.py         Roofline + KV transfer cost models
│   └── plots.py             plot_headline_20qps / plot_sweep_grid /
│                            plot_profitability_frontier
├── scripts/
│   ├── start_colocated.sh   single-GPU baseline (GPU 0, :8000)
│   ├── start_pd.sh          PD stack: prefill (GPU0:8100) + decode (GPU1:8200)
│   │                         + proxy (:8000)
│   ├── stop_all.sh          kill anything on :8000/8100/8200
│   ├── download_model.py    pre-fetch the model to local disk
│   ├── bench_20qps_mixed.py headline 20-QPS mixed-workload run
│   ├── bench_30config_sweep.py  the 30-config sweep
│   ├── analyze.py           roofline / KV transfer report + frontier plot
│   └── plot_results.py      merge per-arch runs and plot
├── configs/pd_2x_a100_40gb.json
├── results/                 outputs: parquet + csv + png/pdf
├── requirements.txt
└── README.md (this file)
```

## Setup (on the SSH host, once)

```bash
# 0. Clone / sync this directory to the host, then:
cd PD_bench
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Persist paths for subsequent shells
export PD_BENCH_MODEL_DIR=/data/models
export PD_BENCH_RESULTS=$PWD/results
export PD_BENCH_LOGS=/tmp/pd_bench

# 2. Pre-download the model
python scripts/download_model.py
```

## Workflow

### A. Headline 20 QPS comparison

```bash
# Colocated baseline
bash scripts/start_colocated.sh
python scripts/bench_20qps_mixed.py --architecture colocated --duration 300
bash scripts/stop_all.sh

# PD disaggregated
bash scripts/start_pd.sh
python scripts/bench_20qps_mixed.py --architecture pd --duration 300
bash scripts/stop_all.sh

# Plot + print the headline numbers
python scripts/plot_results.py headline
```

The plot script prints the P99 TTFT cut and SLO compliance gain, and
emits `headline_20qps_<ts>.{png,pdf}` (TTFT CDF + SLO/P99 bars).

### B. 30-config sweep

```bash
bash scripts/start_colocated.sh
python scripts/bench_30config_sweep.py --architecture colocated
bash scripts/stop_all.sh

bash scripts/start_pd.sh
python scripts/bench_30config_sweep.py --architecture pd
bash scripts/stop_all.sh

python scripts/plot_results.py sweep
```

Default grid: profiles ∈ {inline_completion, code_explanation,
function_generation}, qps ∈ {5, 10, 20, 30, 50}, 180 s per cell + 10 s
cooldown. Override with `--profiles ... --qps-grid ... --duration ...`.
Total runtime ≈ 95 min per architecture.

### C. Roofline + KV-transfer analysis (no GPU needed)

```bash
python scripts/analyze.py --interconnect nvlink3   # 600 GB/s peer
python scripts/analyze.py --interconnect pcie4     # 32 GB/s peer
```

Produces:
- a per-ISL table: arithmetic intensity, prefill compute time, KV size,
  transfer time, transfer/prefill ratio
- `profitability_frontier_<link>_<ts>.{png,pdf}` — QPS × ISL heatmap
  shaded by analytic PD profitability with the break-even curve overlaid
- `profitability_break_even_<link>_<ts>.csv` — numeric break-even points

## Architecture under test

```
                                    Client (open-loop, Poisson QPS)
                                              │
                                              ▼
                                    Proxy (aiohttp, :8000)
                                       │           │
              ┌────────────────────────┘           └─────────────────────┐
              ▼                                                          ▼
   vLLM @ GPU0 :8100                                          vLLM @ GPU1 :8200
   --kv-transfer-config kv_producer                           --kv-transfer-config kv_consumer
   (prefill, KV pushed via NCCL)                              (decode, KV resident, no re-prefill)
```

The proxy sends a `max_tokens=1, stream=False` primer to the prefill
server (which runs the prompt forward and pushes the KV cache to the
consumer over NCCL), then forwards the original streaming request to
the decode server. The decode server skips its own prefill because the
KV is already resident, so its first SSE event arrives after a single
decode step.

## Why these numbers come out the way they do

- **Prefill is compute-bound** at ISL ≳ ridge point (~200 FLOP/byte for
  A100). Long-context requests saturate the SM. In colocated mode they
  block decode steps for tens of milliseconds, inflating TTFT for any
  concurrent request stuck behind them — this is the queueing tax.
- **Decode is memory-bound** (single-token weight + KV reads). It does
  not benefit from compute throughput, only from being scheduled
  promptly. PD removes the head-of-line blocker.
- **KV transfer overhead** is small over NVLink: at ISL 2048 the
  Qwen2.5-7B KV cache is ~112 MB → ~0.2 ms one-way (NVLink) or ~3.5 ms
  (PCIe Gen4 P2P). On NVLink, PD wins almost everywhere with ISL ≥ 512;
  on PCIe-only SKUs the break-even shifts to noticeably higher QPS.
- The 30-config sweep should show **PD = colocated** at low QPS and
  short prompts (where there is no contention to fix) and **PD ≫
  colocated** at moderate-to-high QPS or long ISLs.

## Files written per run

```
results/
  headline_20qps_colocated_<ts>.{csv,parquet,json}
  headline_20qps_colocated_traces_<ts>.parquet
  headline_20qps_colocated_metrics_latest.parquet
  headline_20qps_colocated_traces_latest.parquet
  headline_20qps_pd_*.parquet                  (same shape)
  sweep_colocated_*.parquet                    (30 metrics rows)
  sweep_pd_*.parquet
  headline_20qps_<ts>.{png,pdf}                (combined plot)
  sweep_grid_<ts>.{png,pdf}
  profitability_frontier_nvlink3_<ts>.{png,pdf}
```

## Notes

- vLLM 0.7.3's PD support is experimental. If `PyNcclConnector` is
  unavailable on the running vLLM build, swap to `LMCacheConnectorV1`
  in `src/pd_server.py::_kv_cfg`.
- The interconnect spec defaults to NVLink3. On a PCIe-only SKU
  (e.g. some cloud A100 40GB PCIe variants), pass
  `--interconnect pcie4` to `analyze.py` and expect the empirical PD
  break-even to track the analytic prediction shifted right (higher QPS).
- The proxy is intentionally minimal. For higher proxy throughput, swap
  `aiohttp.web.run_app` for `gunicorn -k aiohttp.GunicornWebWorker` with
  `--workers 4`.
