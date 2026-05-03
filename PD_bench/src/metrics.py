"""Aggregation: per-run summary metrics + SLO bookkeeping."""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from .benchmark import RequestResult


def compute_metrics(results: List[RequestResult], slo_ttft: float = 1.0) -> dict:
    ok = [r for r in results if r.success and r.ttft is not None]
    if not ok:
        return {
            "error": "no successful results",
            "n_requests": len(results),
            "n_success":  0,
        }
    ttfts  = [r.ttft for r in ok]
    tpots  = [r.tpot for r in ok if r.tpot]
    totals = [r.total_time for r in ok]
    return {
        "n_requests":   len(results),
        "n_success":    len(ok),
        "success_rate": len(ok) / len(results),
        "ttft_p50":  float(np.percentile(ttfts, 50)),
        "ttft_p95":  float(np.percentile(ttfts, 95)),
        "ttft_p99":  float(np.percentile(ttfts, 99)),
        "ttft_mean": float(np.mean(ttfts)),
        "tpot_p50":  float(np.percentile(tpots, 50)) if tpots else None,
        "tpot_p95":  float(np.percentile(tpots, 95)) if tpots else None,
        "tpot_p99":  float(np.percentile(tpots, 99)) if tpots else None,
        "e2e_p50":   float(np.percentile(totals, 50)),
        "e2e_p99":   float(np.percentile(totals, 99)),
        "total_time":       max(totals),
        "throughput_req_s": len(ok) / max(totals),
        "throughput_tok_s": sum(r.output_tokens for r in ok if r.output_tokens) / max(totals),
        "slo_threshold": slo_ttft,
        "slo_rate": sum(1 for t in ttfts if t < slo_ttft) / len(ttfts),
    }


def trace_records(results: List[RequestResult], **extra) -> List[dict]:
    rows = []
    for r in results:
        rows.append({
            **extra,
            "ttft":          r.ttft,
            "total_time":    r.total_time,
            "tpot":          r.tpot,
            "target_isl":    r.target_isl,
            "target_osl":    r.target_osl,
            "output_tokens": r.output_tokens,
            "success":       r.success,
            "arrival_t":     r.arrival_t,
            "error":         r.error,
            "profile":       r.profile,
        })
    return rows


# ── Cross-architecture comparison helpers ───────────────────────

def slo_compare(df_traces: pd.DataFrame, slo_ttft_s: float = 1.0) -> pd.DataFrame:
    """For each (architecture, profile, qps), report TTFT percentiles and SLO rate."""
    df = df_traces[df_traces["success"] == True].copy()
    df["meets_slo"] = df["ttft"] < slo_ttft_s
    g = (df.groupby(["architecture", "profile", "qps"])
           .agg(n=("ttft", "count"),
                ttft_p50=("ttft", lambda x: x.quantile(0.50)),
                ttft_p95=("ttft", lambda x: x.quantile(0.95)),
                ttft_p99=("ttft", lambda x: x.quantile(0.99)),
                slo_rate=("meets_slo", "mean"))
           .reset_index())
    return g


def pd_uplift(df_metrics: pd.DataFrame) -> pd.DataFrame:
    """Pairwise comparison: PD vs colocated for each (profile, qps).

    Returns a tidy frame with the headline numbers a deployment review wants:
    ΔP99 TTFT (relative), ΔSLO rate (absolute pp), Δthroughput.
    """
    pivot = (df_metrics.pivot_table(
        index=["profile", "qps"], columns="architecture",
        values=["ttft_p99", "slo_rate", "throughput_req_s"], aggfunc="mean")
        .reset_index())
    out = pd.DataFrame({
        "profile":   pivot["profile"],
        "qps":       pivot["qps"],
        "p99_colo_ms": pivot[("ttft_p99", "colocated")] * 1000,
        "p99_pd_ms":   pivot[("ttft_p99", "pd")]        * 1000,
        "slo_colo_%":  pivot[("slo_rate", "colocated")] * 100,
        "slo_pd_%":    pivot[("slo_rate", "pd")]        * 100,
        "thr_colo":    pivot[("throughput_req_s", "colocated")],
        "thr_pd":      pivot[("throughput_req_s", "pd")],
    })
    out["p99_drop_%"]   = (1 - out["p99_pd_ms"] / out["p99_colo_ms"]) * 100
    out["slo_gain_pp"]  = out["slo_pd_%"] - out["slo_colo_%"]
    return out
