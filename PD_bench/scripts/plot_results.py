"""Merge per-architecture runs and plot them.

Modes:
    python scripts/plot_results.py headline   # combines headline_20qps_{colo,pd}_latest
    python scripts/plot_results.py sweep      # combines sweep_{colo,pd}_latest
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

import pandas as pd

from src.config import DEFAULT_RESULTS_DIR
from src.experiments import DEFAULT_PROFILES
from src.metrics import pd_uplift, slo_compare
from src.plots import plot_headline_20qps, plot_sweep_grid
from src.storage import load_latest


def _merge(results_dir: str, tag_prefix: str):
    """Load both architectures' latest runs and concat."""
    metrics_parts, traces_parts = [], []
    for arch in ["colocated", "pd"]:
        try:
            m, t = load_latest(results_dir, tag=f"{tag_prefix}_{arch}")
        except FileNotFoundError:
            print(f"WARNING: no run for {tag_prefix}_{arch}; skipping")
            continue
        metrics_parts.append(m)
        traces_parts.append(t)
    if not metrics_parts:
        raise RuntimeError(f"No data found for {tag_prefix}")
    return pd.concat(metrics_parts, ignore_index=True), pd.concat(traces_parts, ignore_index=True)


def cmd_headline(args):
    df_m, df_t = _merge(args.results_dir, "headline_20qps")
    print("\nHeadline metrics:")
    cols = ["architecture", "n_requests", "ttft_p50", "ttft_p95",
            "ttft_p99", "slo_rate", "throughput_req_s"]
    pretty = df_m[cols].copy()
    for c in ["ttft_p50", "ttft_p95", "ttft_p99"]:
        pretty[c] = (pretty[c] * 1000).round(1).astype(str) + "ms"
    pretty["slo_rate"] = (pretty["slo_rate"] * 100).round(2).astype(str) + "%"
    pretty["throughput_req_s"] = pretty["throughput_req_s"].round(2)
    print(pretty.to_string(index=False))

    if {"colocated", "pd"}.issubset(df_m["architecture"].unique()):
        colo = df_m[df_m["architecture"] == "colocated"].iloc[0]
        pd_  = df_m[df_m["architecture"] == "pd"].iloc[0]
        drop = (1 - pd_["ttft_p99"] / colo["ttft_p99"]) * 100
        gain = (pd_["slo_rate"] - colo["slo_rate"]) * 100
        print(f"\n  P99 TTFT cut:        {drop:5.1f}%   "
              f"({colo['ttft_p99']*1000:.0f}ms → {pd_['ttft_p99']*1000:.0f}ms)")
        print(f"  SLO compliance gain: {colo['slo_rate']*100:5.1f}% → "
              f"{pd_['slo_rate']*100:5.1f}%   (+{gain:.1f} pp)")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_headline_20qps(df_t, df_m, args.results_dir, ts,
                        slo_ttft_s=df_m["slo_ttft"].iloc[0])


def cmd_sweep(args):
    df_m, df_t = _merge(args.results_dir, "sweep")
    print("\nSweep summary:")
    print(slo_compare(df_t).to_string(index=False))

    print("\nPD uplift per (profile, qps):")
    print(pd_uplift(df_m).round(2).to_string(index=False))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    profiles = [p for p in DEFAULT_PROFILES if p in df_m["profile"].unique()]
    plot_sweep_grid(df_m, profiles, args.results_dir, ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["headline", "sweep"])
    ap.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    args = ap.parse_args()
    {"headline": cmd_headline, "sweep": cmd_sweep}[args.mode](args)


if __name__ == "__main__":
    main()
