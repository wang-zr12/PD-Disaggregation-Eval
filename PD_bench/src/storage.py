"""Persist metrics + traces. Always writes ``*_latest.parquet`` for plotting."""
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import List, Tuple

import pandas as pd


def _ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_run(metrics: List[dict], traces: List[dict], results_dir: str,
             tag: str, timestamp: str = None
             ) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    timestamp = timestamp or _ts()
    os.makedirs(results_dir, exist_ok=True)
    df_metrics = pd.DataFrame(metrics)
    df_traces  = pd.DataFrame(traces)
    paths = {
        "metrics_csv":     f"{results_dir}/{tag}_{timestamp}.csv",
        "metrics_parquet": f"{results_dir}/{tag}_{timestamp}.parquet",
        "metrics_json":    f"{results_dir}/{tag}_{timestamp}.json",
        "traces_parquet":  f"{results_dir}/{tag}_traces_{timestamp}.parquet",
        "metrics_latest":  f"{results_dir}/{tag}_metrics_latest.parquet",
        "traces_latest":   f"{results_dir}/{tag}_traces_latest.parquet",
    }
    df_metrics.to_csv(paths["metrics_csv"], index=False)
    df_metrics.to_parquet(paths["metrics_parquet"], index=False)
    with open(paths["metrics_json"], "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    df_traces.to_parquet(paths["traces_parquet"], index=False)
    df_metrics.to_parquet(paths["metrics_latest"], index=False)
    df_traces.to_parquet(paths["traces_latest"], index=False)

    print(f"Saved {tag}: {len(df_metrics)} metrics rows, {len(df_traces)} trace rows")
    for k, p in paths.items():
        print(f"  {k:18s} {p}")
    return df_metrics, df_traces, paths


def load_latest(results_dir: str, tag: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    return (
        pd.read_parquet(f"{results_dir}/{tag}_metrics_latest.parquet"),
        pd.read_parquet(f"{results_dir}/{tag}_traces_latest.parquet"),
    )
