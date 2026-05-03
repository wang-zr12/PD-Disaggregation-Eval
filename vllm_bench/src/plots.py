"""Plotting: baseline (concurrency × workload), ISL sweep, OSL sweep."""
from __future__ import annotations

import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .workload import WORKLOAD_PROFILES

plt.rcParams.update({"font.size": 11, "figure.dpi": 120})

COLORS = {
    "inline_completion":   "#2196F3",
    "code_explanation":    "#FF9800",
    "function_generation": "#4CAF50",
    "mixed":               "#9C27B0",
}


def _save(fig, results_dir: str, stem: str, timestamp: str) -> None:
    os.makedirs(results_dir, exist_ok=True)
    png = f"{results_dir}/{stem}_{timestamp}.png"
    pdf = f"{results_dir}/{stem}_{timestamp}.pdf"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(f"  PNG: {png}")
    print(f"  PDF: {pdf}")


def plot_baseline(df_metrics: pd.DataFrame, profiles: List[str], model_name: str,
                  results_dir: str, timestamp: str):
    """Three-panel: TTFT (P50 bar + P99 errorbar, log-y), SLO satisfaction, throughput."""
    df = df_metrics[df_metrics["profile"].isin(profiles)].copy()
    concurrency_axis = sorted(df["concurrency"].unique())

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle(
        f"Standard Serving Baseline — {model_name}\n"
        "Code Generation Workload Characterization",
        fontsize=13, fontweight="bold",
    )

    # (1) TTFT: bar P50 + errorbar to P99 (log y)
    ax = axes[0]
    x = np.arange(len(concurrency_axis))
    w = 0.8 / len(profiles)
    for i, profile in enumerate(profiles):
        pdata = (df[df["profile"] == profile]
                 .set_index("concurrency").reindex(concurrency_axis))
        p50 = pdata["ttft_p50"].values * 1000
        p99 = pdata["ttft_p99"].values * 1000
        offset = (i - (len(profiles) - 1) / 2) * w
        ax.bar(x + offset, p50, w, color=COLORS.get(profile, None),
               alpha=0.8, label=profile.replace("_", " "))
        yerr_upper = np.maximum(p99 - p50, 0)
        ax.errorbar(x + offset, p50,
                    yerr=[np.zeros_like(p50), yerr_upper],
                    fmt="none", color="black", capsize=3, lw=1.2)
    ax.set_title("TTFT: P50 (bar) + P99 (error bar)")
    ax.set_xlabel("Concurrency")
    ax.set_yscale("log")
    ax.set_ylim(10, 30000)
    ax.set_ylabel("TTFT (ms, log scale)")
    ax.set_xticks(x)
    ax.set_xticklabels(concurrency_axis)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    # (2) SLO satisfaction
    ax = axes[1]
    for profile in profiles:
        pdata = df[df["profile"] == profile].sort_values("concurrency")
        slo_target = WORKLOAD_PROFILES[profile]["TTFT_SLO"]
        ax.plot(pdata["concurrency"], pdata["slo_rate"] * 100,
                marker="o", color=COLORS.get(profile, None), lw=2,
                label=f"{profile.replace('_', ' ')} (SLO={slo_target}s)")
    ax.axhline(90, color="red", ls="--", lw=1.5, label="90% SLO target")
    ax.set_title("SLO Satisfaction Rate vs Concurrency")
    ax.set_xlabel("Concurrency")
    ax.set_ylabel("SLO Satisfaction Rate (%)")
    ax.set_ylim(0, 105)
    ax.set_xscale("log")
    ax.set_xticks(concurrency_axis)
    ax.set_xticklabels(concurrency_axis)
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.3, which="both")

    # (3) Throughput
    ax = axes[2]
    for profile in profiles:
        pdata = df[df["profile"] == profile].sort_values("concurrency")
        ax.plot(pdata["concurrency"], pdata["throughput_req_s"],
                marker="s", color=COLORS.get(profile, None), lw=2,
                label=profile.replace("_", " "))
    ax.set_title("Throughput vs Concurrency")
    ax.set_xlabel("Concurrency")
    ax.set_ylabel("Throughput (req/s)")
    ax.set_xscale("log")
    ax.set_xticks(concurrency_axis)
    ax.set_xticklabels(concurrency_axis)
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3, which="both")

    plt.tight_layout()
    _save(fig, results_dir, "baseline_plot", timestamp)
    return fig


def plot_isl_sweep(isl_df: pd.DataFrame, fixed_osl: int, fixed_conc: int,
                   n_per_point: int, results_dir: str, timestamp: str,
                   slo_ttft_ms: float = 1000):
    fig, ax = plt.subplots(figsize=(9, 4.8))
    p50 = isl_df["ttft_p50"] * 1000
    p99 = isl_df["ttft_p99"] * 1000

    ax.plot(isl_df["isl"], p50, marker="o", color="#2196F3", lw=2, label="P50", ms=7)
    ax.plot(isl_df["isl"], p99, marker="s", color="#F44336", lw=2, ls="--", label="P99", ms=7)
    ax.fill_between(isl_df["isl"], p50, p99, alpha=0.12, color="#2196F3")
    ax.axhline(slo_ttft_ms, color="orange", ls=":", lw=1.8,
               label=f"TTFT threshold = {slo_ttft_ms/1000:.1f}s")

    crossings = np.where(np.diff((p99.values > slo_ttft_ms).astype(int)) == 1)[0]
    if len(crossings):
        idx = crossings[0]
        x1, x2 = isl_df["isl"].iloc[idx], isl_df["isl"].iloc[idx + 1]
        y1, y2 = p99.iloc[idx], p99.iloc[idx + 1]
        cross = x1 + (slo_ttft_ms - y1) * (x2 - x1) / (y2 - y1)
        ax.axvline(cross, color="red", ls=":", alpha=0.5, lw=1.2)
        ax.annotate(f"P99 breaks SLO\nISL ≈ {cross:.0f}",
                    xy=(cross, slo_ttft_ms), xytext=(cross + 400, slo_ttft_ms * 0.6),
                    fontsize=9, color="red",
                    arrowprops=dict(arrowstyle="->", color="red", alpha=0.6))

    ax.set_title(
        f"TTFT vs Input Sequence Length (ISL)\n"
        f"Standard Serving | OSL={fixed_osl} | Concurrency={fixed_conc} | "
        f"N={n_per_point}/point"
    )
    ax.set_xlabel("Input Sequence Length (tokens)")
    ax.set_ylabel("TTFT (ms)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    _save(fig, results_dir, "isl_sweep", timestamp)
    return fig


def plot_osl_sweep(osl_traces_df: pd.DataFrame, fixed_isl: int, fixed_conc: int,
                   n_per_point: int, results_dir: str, timestamp: str,
                   e2e_slo_ms: float = 5000):
    df = osl_traces_df[osl_traces_df["success"] == True]
    osls = sorted(df["osl"].unique())

    def q(col, p):
        return df.groupby("osl")[col].quantile(p).reindex(osls) * 1000

    ttft_p50, ttft_p99 = q("ttft", 0.50),       q("ttft", 0.99)
    tpot_p50, tpot_p99 = q("tpot", 0.50),       q("tpot", 0.99)
    e2e_p50,  e2e_p99  = q("total_time", 0.50), q("total_time", 0.99)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    for ax, (label, p50_v, p99_v) in zip(axes, [
        ("TTFT (ms)", ttft_p50, ttft_p99),
        ("TPOT (ms/token)", tpot_p50, tpot_p99),
        ("E2E Latency (ms)", e2e_p50, e2e_p99),
    ]):
        ax.plot(osls, p50_v, "o-",  color="#2196F3", lw=2, label="P50", ms=7)
        ax.plot(osls, p99_v, "s--", color="#F44336", lw=2, label="P99", ms=7)
        ax.fill_between(osls, p50_v, p99_v, alpha=0.12, color="#2196F3")
        ax.set_xlabel("Output Sequence Length (tokens)")
        ax.set_ylabel(label)
        ax.set_title(f"{label.split(' (')[0]} vs OSL")
        ax.legend(loc="upper left")
        ax.grid(alpha=0.3)
        ax.set_ylim(bottom=0)

    # E2E SLO crossing annotation on third panel
    ax = axes[2]
    ax.axhline(e2e_slo_ms, color="orange", ls=":", lw=1.8,
               label=f"E2E SLO = {e2e_slo_ms/1000:.0f}s")
    crossings = np.where(np.diff((np.array(e2e_p99) > e2e_slo_ms).astype(int)) == 1)[0]
    if len(crossings):
        idx = crossings[0]
        x1, x2 = osls[idx], osls[idx + 1]
        y1, y2 = e2e_p99.iloc[idx], e2e_p99.iloc[idx + 1]
        cross = x1 + (e2e_slo_ms - y1) * (x2 - x1) / (y2 - y1)
        ax.axvline(cross, color="red", ls=":", alpha=0.5, lw=1.2)
        ax.annotate(f"P99 breaks SLO\nOSL ≈ {cross:.0f}",
                    xy=(cross, e2e_slo_ms),
                    xytext=(cross - 600, e2e_slo_ms + 1500),
                    fontsize=9, color="red",
                    arrowprops=dict(arrowstyle="->", color="red", alpha=0.6))
    ax.legend(loc="upper left")

    fig.suptitle(
        f"OSL Sweep — Standard Serving | ISL={fixed_isl} | "
        f"Concurrency={fixed_conc} | N={n_per_point}/point",
        fontsize=12, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    _save(fig, results_dir, "osl_sweep", timestamp)
    return fig
