"""Aggregation: per-run summary metrics and SLO sensitivity sweeps."""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from .benchmark import RequestResult


def compute_metrics(results: List[RequestResult], slo_ttft: float = 2.0) -> dict:
    ok = [r for r in results if r.success and r.ttft is not None]
    if not ok:
        return {"error": "no successful results", "n_requests": len(results), "n_success": 0}

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
        "total_time":       max(totals),
        "throughput_req_s": len(ok) / max(totals),
        "throughput_tok_s": sum(r.output_tokens for r in ok if r.output_tokens) / max(totals),
        "slo_threshold": slo_ttft,
        "slo_rate": sum(1 for t in ttfts if t < slo_ttft) / len(ttfts),
    }


def trace_records(results: List[RequestResult], **extra) -> List[dict]:
    """Flatten RequestResult list to dict rows; ``extra`` adds run-level cols."""
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
            "error":         r.error,
            "profile":       r.profile,
        })
    return rows


# ── SLO sensitivity ───────────────────────────────────────

def compute_slo_table(df_traces: pd.DataFrame, slo_map: Dict[str, float]) -> pd.DataFrame:
    df = df_traces[df_traces["success"] == True].copy()
    df["slo_target"] = df["profile"].map(slo_map)
    df["meets_slo"]  = df["ttft"] < df["slo_target"]
    return (
        df.groupby(["profile", "concurrency"])
          .agg(
              n=("ttft", "count"),
              ttft_p50=("ttft", lambda x: x.quantile(0.50)),
              ttft_p95=("ttft", lambda x: x.quantile(0.95)),
              ttft_p99=("ttft", lambda x: x.quantile(0.99)),
              slo_rate=("meets_slo", "mean"),
          )
          .reset_index()
    )


def slo_comparison_matrix(df_traces: pd.DataFrame,
                          slo_candidates: Dict[str, Dict[str, float]],
                          exclude_profiles=("mixed",)) -> pd.DataFrame:
    """For each (profile, concurrency), compute SLO-met rate under each candidate."""
    df = df_traces[df_traces["success"] == True]
    rows = []
    for profile in sorted(df["profile"].unique()):
        if profile in exclude_profiles:
            continue
        for c in sorted(df["concurrency"].unique()):
            sub = df[(df["profile"] == profile) & (df["concurrency"] == c)]
            if len(sub) == 0:
                continue
            row = {
                "profile": profile, "concurrency": c, "n": len(sub),
                "p50_ms": int(sub["ttft"].quantile(0.50) * 1000),
                "p99_ms": int(sub["ttft"].quantile(0.99) * 1000),
            }
            for name, slo_map in slo_candidates.items():
                target = slo_map.get(profile)
                row[name] = round((sub["ttft"] < target).mean() * 100, 1)
            rows.append(row)
    return pd.DataFrame(rows)


def slo_information_summary(cmp_df: pd.DataFrame,
                            slo_candidates: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """Wider span = more discriminative SLO setting."""
    rows = []
    for name in slo_candidates:
        spans = [
            cmp_df[cmp_df["profile"] == p][name].max()
            - cmp_df[cmp_df["profile"] == p][name].min()
            for p in cmp_df["profile"].unique()
        ]
        rows.append({
            "candidate": name,
            "avg_span":  float(np.mean(spans)),
            "min_rate":  float(cmp_df[name].min()),
            "max_rate":  float(cmp_df[name].max()),
        })
    return pd.DataFrame(rows)


DEFAULT_SLO_CANDIDATES = {
    "strict":   {"inline_completion": 0.3, "code_explanation": 1.0, "function_generation": 0.5},
    "medium":   {"inline_completion": 0.5, "code_explanation": 1.5, "function_generation": 0.5},
    "relaxed":  {"inline_completion": 0.7, "code_explanation": 2.0, "function_generation": 1.0},
    "original": {"inline_completion": 1.0, "code_explanation": 2.0, "function_generation": 1.5},
}
