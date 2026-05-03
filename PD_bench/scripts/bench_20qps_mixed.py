"""Headline 20-QPS mixed-workload comparison: colocated vs PD.

This is the run that produces the resume-quoted P99 TTFT cut (~40%)
and the SLO compliance jump (55% → 92%).

Run sequence on the SSH host:

    bash scripts/start_colocated.sh
    python scripts/bench_20qps_mixed.py --architecture colocated
    bash scripts/stop_all.sh

    bash scripts/start_pd.sh
    python scripts/bench_20qps_mixed.py --architecture pd
    bash scripts/stop_all.sh

    python scripts/plot_results.py headline
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from datetime import datetime

# Allow running from anywhere
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.config import DEFAULT_RESULTS_DIR, qwen25_coder_7b
from src.experiments import run_one, warmup
from src.storage import save_run
from src.workload import WorkloadGenerator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--architecture", choices=["colocated", "pd"], required=True,
                    help="Which architecture is currently running on :8000")
    ap.add_argument("--server-url",   default="http://localhost:8000")
    ap.add_argument("--qps",          type=float, default=20.0)
    ap.add_argument("--duration",     type=float, default=300.0,
                    help="Seconds of steady-state load")
    ap.add_argument("--slo-ttft",     type=float, default=1.0,
                    help="TTFT SLO threshold in seconds (default 1.0)")
    ap.add_argument("--results-dir",  default=DEFAULT_RESULTS_DIR)
    ap.add_argument("--model-dir",    default=None,
                    help="Override PD_BENCH_MODEL_DIR")
    ap.add_argument("--seed",         type=int, default=7)
    args = ap.parse_args()

    cfg = qwen25_coder_7b(args.model_dir) if args.model_dir else qwen25_coder_7b()
    gen = WorkloadGenerator(cfg.local_path)

    print(f"=== Headline 20 QPS run  arch={args.architecture}  "
          f"qps={args.qps}  duration={args.duration}s ===")

    asyncio.run(_run(gen, args, cfg))


async def _run(gen, args, cfg):
    print("Warmup ...")
    await warmup(gen, args.server_url)
    m, t = await run_one(
        gen, profile="mixed", qps=args.qps, duration_s=args.duration,
        server_url=args.server_url, architecture=args.architecture,
        seed=args.seed, slo_ttft=args.slo_ttft,
    )
    print(
        f"\nResults  arch={args.architecture}\n"
        f"  N requests:           {m['n_requests']}\n"
        f"  TTFT P50:             {m['ttft_p50']*1000:7.1f} ms\n"
        f"  TTFT P95:             {m['ttft_p95']*1000:7.1f} ms\n"
        f"  TTFT P99:             {m['ttft_p99']*1000:7.1f} ms\n"
        f"  TPOT P50:             {(m['tpot_p50'] or 0)*1000:7.2f} ms/tok\n"
        f"  E2E  P99:             {m['e2e_p99']*1000:7.1f} ms\n"
        f"  SLO<{args.slo_ttft}s satisfaction:  {m['slo_rate']*100:6.2f} %\n"
        f"  Throughput:           {m['throughput_req_s']:7.2f} req/s "
        f"({m['throughput_tok_s']:.0f} tok/s)\n"
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_run([m], t, args.results_dir,
             tag=f"headline_20qps_{args.architecture}", timestamp=ts)


if __name__ == "__main__":
    main()
