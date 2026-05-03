# vLLM PD-Disaggregation Benchmark

**Project period:** October 2024 – December 2024 (archived)

End-to-end evaluation of **Prefill–Decode (PD) disaggregation on vLLM** for
code-completion workloads, comparing a strong single-GPU colocated baseline
against a 2-GPU disaggregated deployment.


## Project structure

```
PD/
├── README.md                  ← this file
├── vllm.ipynb                 ← original exploration notebook (untouched, archive)
│
├── vllm_bench/                ← Stage 1: single-GPU characterization (Colab, A100 80GB)
│   ├── src/                   modular harness (config, server, workload, benchmark,
│   │                          metrics, storage, plots, experiments)
│   ├── notebooks/main.ipynb   minimal orchestration notebook
│   ├── results/               metrics + traces + figures
│   ├── requirements.txt
│   └── ...
│   Three experiments:
│     1. concurrency × workload sweep (3 profiles × 12 concurrency levels)
│     2. ISL sweep — TTFT vs input length
│     3. OSL sweep — TPOT/E2E vs output length
│
└── PD_bench/                  ← Stage 2: PD disaggregation (cloud SSH host, 2× A100 40GB)
    ├── src/                   PD launcher, NCCL-based KV-transfer proxy,
    │                          open-loop Poisson QPS driver, Roofline analytics
    ├── scripts/               start_colocated.sh, start_pd.sh, stop_all.sh,
    │                          bench_20qps_mixed.py, bench_30config_sweep.py,
    │                          analyze.py, plot_results.py
    ├── configs/pd_2x_a100_40gb.json
    ├── results/
    ├── requirements.txt
    └── README.md              detailed SSH workflow
    Three deliverables:
      A. headline 20 QPS comparison (PD vs colocated)
      B. 30-config QPS × Workload × Architecture sweep
      C. analytic Roofline + KV-transfer profitability frontier
```

## How to run

### Stage 1 — Single-GPU characterization (Colab)

Open `vllm_bench/notebooks/main.ipynb` in Colab attached to an A100 runtime,
then execute cells top-to-bottom:

1. mount Drive, install pinned deps (auto-restart once)
2. auto-select model from VRAM, prepare paths
3. download model from HF mirror to Drive (~15 GB, one-time)
4. start vLLM OpenAI server
5. concurrency × workload sweep → save → plot
6. ISL sweep → save → plot
7. OSL sweep → save → plot
8. SLO sensitivity analysis
9. stop server

All implementation lives in `vllm_bench/src/`; the notebook is pure
orchestration. Outputs land in `results/` (parquet + csv + json + png/pdf).

### Stage 2 — PD disaggregation (cloud SSH host)

On a single host with 2× NVIDIA A100 40 GB:

```bash
cd PD_bench
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

export PD_BENCH_MODEL_DIR=/data/models
python scripts/download_model.py        # one-time, ~15 GB to local disk

# A. Headline 20 QPS comparison
bash   scripts/start_colocated.sh
python scripts/bench_20qps_mixed.py --architecture colocated --duration 300
bash   scripts/stop_all.sh

bash   scripts/start_pd.sh
python scripts/bench_20qps_mixed.py --architecture pd --duration 300
bash   scripts/stop_all.sh

python scripts/plot_results.py headline

# B. 30-config sweep
bash   scripts/start_colocated.sh
python scripts/bench_30config_sweep.py --architecture colocated
bash   scripts/stop_all.sh

bash   scripts/start_pd.sh
python scripts/bench_30config_sweep.py --architecture pd
bash   scripts/stop_all.sh

python scripts/plot_results.py sweep

# C. Analytic profitability frontier (no GPU required)
python scripts/analyze.py --interconnect nvlink3
python scripts/analyze.py --interconnect pcie4
```

See `PD_bench/README.md` for the full workflow, architecture diagram, and
notes on vLLM 0.7.3 PD configuration.

## Why two stages

Stage 1 establishes a **workload characterization baseline** on a single GPU:
where TTFT/TPOT degrade with concurrency and sequence length, what the SLO
violation regime looks like, and which QPS range is interesting.
Stage 2 takes the same workload definition (same tokenizer, same three profiles —
inline_completion, code_explanation, function_generation) and compares
architectures at production-relevant QPS levels. The single-GPU stage tells
us *where the pain is*; the disaggregation stage tells us *whether splitting
prefill and decode fixes it, and at what cost*.

## Tech stack

- **vLLM 0.7.3** OpenAI-compatible server, with experimental
  `--kv-transfer-config PyNcclConnector` for PD
- **Qwen2.5-Coder-7B-Instruct** FP16, max_model_len = 8192
- **A100 80 GB** (Stage 1) and **2× A100 40 GB** (Stage 2)

## Status

Archived as of 2024-12. All pinned dependency versions in requirements.txt and the notebook's PINS dict have been bumped post-archival to track the latest stable releases of vLLM / Transformers / Tokenizers; the harness, scripts, and configuration JSON keys remain unchanged. 
