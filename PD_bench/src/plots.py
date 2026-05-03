"""Plots for the PD benchmark.

  • plot_headline_20qps:   side-by-side TTFT distributions, SLO bars
  • plot_sweep_grid:       3 profiles × (TTFT P99, SLO rate, throughput) vs QPS
  • plot_profitability:    QPS×ISL heatmap of analytic PD break-even
"""
from __future__ import annotations

import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({"font.size": 11, "figure.dpi": 120})

ARCH_COLOR = {"colocated": "#FF6B35", "pd": "#1F77B4"}


def _save(fig, results_dir: str, stem: str, timestamp: str):
    os.makedirs(results_dir, exist_ok=True)
    png = f"{results_dir}/{stem}_{timestamp}.png"
    pdf = f"{results_dir}/{stem}_{timestamp}.pdf"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(f"  PNG: {png}")
    print(f"  PDF: {pdf}")


# ── Headline 20 QPS comparison ──────────────────────────────────

def plot_headline_20qps(df_traces: pd.DataFrame, df_metrics: pd.DataFrame,
                        results_dir: str, timestamp: str,
                        slo_ttft_s: float = 1.0):
    """Two panels: TTFT CDF (colocated vs pd) and SLO/throughput bars."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    ax = axes[0]
    for arch in ["colocated", "pd"]:
        sub = df_traces[(df_traces["architecture"] == arch)
                        & (df_traces["success"] == True)]
        ttft_ms = np.sort(sub["ttft"].values * 1000)
        if len(ttft_ms) == 0:
            continue
        cdf = np.arange(1, len(ttft_ms) + 1) / len(ttft_ms)
        ax.plot(ttft_ms, cdf, lw=2.2, color=ARCH_COLOR[arch],
                label=arch.upper())
    ax.axvline(slo_ttft_s * 1000, color="red", ls="--", lw=1.5,
               label=f"SLO = {slo_ttft_s:.1f}s")
    ax.set_xscale("log")
    ax.set_xlabel("TTFT (ms, log scale)")
    ax.set_ylabel("CDF")
    ax.set_title("TTFT CDF — 20 QPS mixed workload")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3, which="both")

    ax = axes[1]
    df = df_metrics.set_index("architecture")
    archs = ["colocated", "pd"]
    x = np.arange(len(archs))
    width = 0.35
    slo = [df.loc[a, "slo_rate"] * 100 for a in archs]
    p99 = [df.loc[a, "ttft_p99"] * 1000 for a in archs]
    ax2 = ax.twinx()
    bars1 = ax.bar(x - width / 2, slo, width,
                   color=[ARCH_COLOR[a] for a in archs], alpha=0.85,
                   label="SLO satisfaction (%)")
    bars2 = ax2.bar(x + width / 2, p99, width,
                    color=[ARCH_COLOR[a] for a in archs], alpha=0.45, hatch="//",
                    edgecolor="black", label="P99 TTFT (ms)")
    ax.set_xticks(x); ax.set_xticklabels([a.upper() for a in archs])
    ax.set_ylabel("SLO satisfaction (%)")
    ax2.set_ylabel("P99 TTFT (ms)")
    ax.set_ylim(0, 105)
    ax.axhline(90, color="red", ls=":", lw=1.2)
    ax.set_title("PD vs Colocated — 20 QPS mixed workload")
    for b, v in zip(bars1, slo):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}%",
                ha="center", fontsize=10, fontweight="bold")
    for b, v in zip(bars2, p99):
        ax2.text(b.get_x() + b.get_width() / 2, v * 1.02, f"{v:.0f}ms",
                 ha="center", fontsize=9)

    fig.tight_layout()
    _save(fig, results_dir, "headline_20qps", timestamp)
    return fig


# ── 30-config sweep grid ────────────────────────────────────────

def plot_sweep_grid(df_metrics: pd.DataFrame, profiles: List[str],
                    results_dir: str, timestamp: str):
    """3 rows (profiles) × 3 cols (P99 TTFT, SLO%, throughput) vs QPS,
    one line per architecture."""
    fig, axes = plt.subplots(len(profiles), 3, figsize=(15, 4.2 * len(profiles)),
                             sharex=True)
    if len(profiles) == 1:
        axes = axes[None, :]

    for r, profile in enumerate(profiles):
        sub = df_metrics[df_metrics["profile"] == profile]
        for arch in ["colocated", "pd"]:
            d = sub[sub["architecture"] == arch].sort_values("qps")
            color = ARCH_COLOR[arch]
            axes[r, 0].plot(d["qps"], d["ttft_p99"] * 1000, marker="o",
                            color=color, lw=2, label=arch.upper())
            axes[r, 1].plot(d["qps"], d["slo_rate"] * 100, marker="s",
                            color=color, lw=2, label=arch.upper())
            axes[r, 2].plot(d["qps"], d["throughput_req_s"], marker="^",
                            color=color, lw=2, label=arch.upper())

        axes[r, 0].set_ylabel("P99 TTFT (ms)")
        axes[r, 1].set_ylabel("SLO satisfaction (%)")
        axes[r, 2].set_ylabel("Throughput (req/s)")
        axes[r, 0].set_yscale("log")
        axes[r, 1].set_ylim(0, 105)
        axes[r, 1].axhline(90, color="red", ls=":", lw=1.2)
        for c in range(3):
            axes[r, c].grid(alpha=0.3, which="both")
            axes[r, c].set_title(profile.replace("_", " ") if c == 1 else "",
                                 fontsize=11, fontweight="bold")
            if r == len(profiles) - 1:
                axes[r, c].set_xlabel("QPS")
            if r == 0 and c == 0:
                axes[r, c].legend(loc="upper left")

    fig.suptitle("PD vs Colocated — 30-config sweep (3 profiles × 5 QPS × 2 arch)",
                 fontsize=12, fontweight="bold", y=1.001)
    fig.tight_layout()
    _save(fig, results_dir, "sweep_grid", timestamp)
    return fig


# ── Analytic profitability frontier ─────────────────────────────

def plot_profitability_frontier(frontier_dict: Dict,
                                results_dir: str, timestamp: str):
    """Heatmap of PD profitability and the analytic break-even curve."""
    grid = frontier_dict["grid"]
    isls = frontier_dict["isl_grid"]
    qpss = frontier_dict["qps_grid"]
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    im = ax.imshow(grid.astype(int), aspect="auto", origin="lower",
                   extent=[min(qpss), max(qpss), min(isls), max(isls)],
                   cmap="RdYlGn", alpha=0.7, vmin=0, vmax=1)
    cbar = fig.colorbar(im, ax=ax, ticks=[0, 1])
    cbar.ax.set_yticklabels(["colocated wins", "PD wins"])

    pts = frontier_dict["break_even_curve"]
    xs = [p["qps_break_even"] for p in pts]
    ys = [p["isl"]            for p in pts]
    ax.plot(xs, ys, "k-", lw=2, label="analytic break-even")
    ax.set_xlabel("QPS")
    ax.set_ylabel("Input Sequence Length (tokens)")
    ax.set_title("PD Profitability Frontier  (Roofline + KV-transfer model)")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, results_dir, "profitability_frontier", timestamp)
    return fig
