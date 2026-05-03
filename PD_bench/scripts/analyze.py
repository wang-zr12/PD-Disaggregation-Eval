"""Offline analytics report:

  • Roofline + KV transfer cost summary table
  • PD profitability frontier (analytic) — saved as PNG/PDF + CSV

No GPU required. Pure first-order modelling that explains the *shape*
of the empirical PD-vs-colocated frontier produced by the sweep.

Run:
    python scripts/analyze.py --interconnect nvlink3
    python scripts/analyze.py --interconnect pcie4
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd

from src.analytics import (QWEN25_CODER_7B, profitability_frontier, report)
from src.config import DEFAULT_RESULTS_DIR, GPUSpec, InterconnectSpec
from src.plots import plot_profitability_frontier


_LINKS = {
    "nvlink3": InterconnectSpec(name="NVLink3", bw_gbps=600.0),
    "pcie4":   InterconnectSpec(name="PCIe Gen4 P2P", bw_gbps=32.0),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interconnect", choices=list(_LINKS), default="nvlink3")
    ap.add_argument("--results-dir",  default=DEFAULT_RESULTS_DIR)
    args = ap.parse_args()

    arch = QWEN25_CODER_7B
    gpu  = GPUSpec()
    link = _LINKS[args.interconnect]

    print(report(arch, gpu, link))

    isl_grid = list(range(128, 8193, 128))
    qps_grid = list(np.linspace(1, 60, 60))
    fr = profitability_frontier(arch, gpu, link, isl_grid, qps_grid)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(args.results_dir, exist_ok=True)
    out_csv = f"{args.results_dir}/profitability_break_even_{args.interconnect}_{ts}.csv"
    pd.DataFrame(fr["break_even_curve"]).to_csv(out_csv, index=False)
    print(f"\nBreak-even curve → {out_csv}")

    plot_profitability_frontier(fr, args.results_dir, f"{args.interconnect}_{ts}")


if __name__ == "__main__":
    main()
