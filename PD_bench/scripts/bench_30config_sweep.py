"""Run the 30-config QPS × Workload × Architecture sweep.

This script drives ONE architecture (whichever is currently running on
``--server-url``). Run it twice — once with the colocated stack up, once
with the PD stack up — and the plotting script will merge them.

Run sequence on the SSH host:

    bash scripts/start_colocated.sh
    python scripts/bench_30config_sweep.py --architecture colocated
    bash scripts/stop_all.sh

    bash scripts/start_pd.sh
    python scripts/bench_30config_sweep.py --architecture pd
    bash scripts/stop_all.sh

    python scripts/plot_results.py sweep
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.config import DEFAULT_RESULTS_DIR, qwen25_coder_7b
from src.experiments import (DEFAULT_PROFILES, DEFAULT_QPS_GRID,
                              run_30config_sweep)
from src.storage import save_run
from src.workload import WorkloadGenerator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--architecture", choices=["colocated", "pd"], required=True)
    ap.add_argument("--server-url",   default="http://localhost:8000")
    ap.add_argument("--profiles",     nargs="+", default=DEFAULT_PROFILES)
    ap.add_argument("--qps-grid",     nargs="+", type=float, default=DEFAULT_QPS_GRID)
    ap.add_argument("--duration",     type=float, default=180.0)
    ap.add_argument("--cooldown",     type=float, default=10.0)
    ap.add_argument("--results-dir",  default=DEFAULT_RESULTS_DIR)
    ap.add_argument("--model-dir",    default=None)
    args = ap.parse_args()

    cfg = qwen25_coder_7b(args.model_dir) if args.model_dir else qwen25_coder_7b()
    gen = WorkloadGenerator(cfg.local_path)
    server_urls = {args.architecture: args.server_url}

    print(f"=== 30-config sweep — arch={args.architecture}  "
          f"profiles={args.profiles}  qps={args.qps_grid} ===")

    metrics, traces = asyncio.run(run_30config_sweep(
        gen, server_urls,
        profiles=args.profiles, qps_grid=args.qps_grid,
        duration_s=args.duration, cooldown_s=args.cooldown,
    ))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_run(metrics, traces, args.results_dir,
             tag=f"sweep_{args.architecture}", timestamp=ts)


if __name__ == "__main__":
    main()
