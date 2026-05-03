"""Roofline modelling + KV-cache transmission cost analysis.

These two analytical models give the *boundary conditions* under which PD
disaggregation pays off versus colocated serving. They are evaluated offline
(no GPU required) and produce:

  • a roofline curve for prefill at varying ISL,
  • a per-token KV-cache size for the chosen model,
  • a (QPS, ISL) profitability frontier — where (prefill compute time saved
    by removing decode contention) > (KV transfer time + overhead).

The numbers are first-order. They explain the *shape* of the empirical
PD-vs-colocated frontier, not the absolute milliseconds.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from .config import GPUSpec, InterconnectSpec


# ── Model architecture spec (Qwen2.5-Coder-7B) ──────────────────
@dataclass
class ModelArch:
    name:           str
    num_layers:     int
    hidden_dim:     int
    num_attn_heads: int
    num_kv_heads:   int       # GQA
    head_dim:       int
    ffn_dim:        int
    bytes_per_elem: int = 2   # FP16

    @property
    def kv_per_token_bytes(self) -> int:
        # K and V, per layer, per token: num_kv_heads * head_dim * bytes
        return 2 * self.num_layers * self.num_kv_heads * self.head_dim * self.bytes_per_elem

    @property
    def weight_bytes(self) -> int:
        # Coarse: attention QKV+O + FFN(up/gate/down). Ignores embeddings/LN.
        per_layer = (
            self.hidden_dim * (2 * self.num_kv_heads * self.head_dim
                               + self.num_attn_heads * self.head_dim)  # KV + Q
            + self.hidden_dim * self.num_attn_heads * self.head_dim     # O
            + 3 * self.hidden_dim * self.ffn_dim                        # up,gate,down
        )
        return per_layer * self.num_layers * self.bytes_per_elem


QWEN25_CODER_7B = ModelArch(
    name="Qwen2.5-Coder-7B",
    num_layers=28, hidden_dim=3584,
    num_attn_heads=28, num_kv_heads=4, head_dim=128,
    ffn_dim=18944, bytes_per_elem=2,
)


# ── Prefill / decode FLOP & memory models ───────────────────────

def prefill_flops(arch: ModelArch, seq_len: int, batch: int = 1) -> float:
    """Forward-pass FLOPs for prefill.

    Per layer:
      • Q,K,V,O projections      ≈ 4 * 2 * B * L * d * d
      • Self-attention scores     ≈ 2 * B * L^2 * d
      • FFN (up,gate,down)        ≈ 3 * 2 * B * L * d * ffn_dim
    """
    L, d, ff, n = seq_len, arch.hidden_dim, arch.ffn_dim, arch.num_layers
    qkvo  = 4 * 2 * batch * L * d * d
    attn  = 2 * batch * L * L * d
    ffn   = 3 * 2 * batch * L * d * ff
    return n * (qkvo + attn + ffn)


def prefill_bytes(arch: ModelArch, seq_len: int, batch: int = 1) -> float:
    """Memory traffic for prefill: weights read once + activation pressure."""
    weights = arch.weight_bytes
    activations = arch.num_layers * batch * seq_len * arch.hidden_dim * arch.bytes_per_elem * 4
    return weights + activations


def decode_flops_per_token(arch: ModelArch, ctx_len: int, batch: int = 1) -> float:
    """One decode step (one new token), with ``ctx_len`` cached tokens."""
    L, d, ff, n = ctx_len, arch.hidden_dim, arch.ffn_dim, arch.num_layers
    qkvo = 4 * 2 * batch * 1 * d * d
    attn = 2 * batch * 1 * L * d
    ffn  = 3 * 2 * batch * 1 * d * ff
    return n * (qkvo + attn + ffn)


def decode_bytes_per_token(arch: ModelArch, ctx_len: int, batch: int = 1) -> float:
    """Decode is memory-bound: full weight read + KV cache read each step."""
    weights = arch.weight_bytes
    kv_read = batch * ctx_len * arch.kv_per_token_bytes
    return weights + kv_read


def arithmetic_intensity(flops: float, bytes_: float) -> float:
    return flops / max(bytes_, 1.0)


def roofline_throughput_flops(ai: float, gpu: GPUSpec) -> float:
    """Min(peak compute, bandwidth × AI)."""
    peak_flops    = gpu.fp16_tflops * 1e12
    bw_bound_flops = gpu.hbm_bw_gbps * 1e9 * ai
    return min(peak_flops, bw_bound_flops)


def ridge_point(gpu: GPUSpec) -> float:
    """AI at which the GPU transitions from memory- to compute-bound."""
    return (gpu.fp16_tflops * 1e12) / (gpu.hbm_bw_gbps * 1e9)


def prefill_time_s(arch: ModelArch, seq_len: int, gpu: GPUSpec,
                   batch: int = 1) -> float:
    f = prefill_flops(arch, seq_len, batch)
    b = prefill_bytes(arch, seq_len, batch)
    ai = arithmetic_intensity(f, b)
    achievable_flops = roofline_throughput_flops(ai, gpu)
    return f / achievable_flops


# ── KV transfer cost ────────────────────────────────────────────

def kv_transfer_time_s(arch: ModelArch, seq_len: int,
                       link: InterconnectSpec) -> float:
    bytes_total = seq_len * arch.kv_per_token_bytes
    return bytes_total / (link.bw_gbps * 1e9)


# ── Profitability frontier ──────────────────────────────────────

def colocated_prefill_overhead_s(arch: ModelArch, seq_len: int, qps: float,
                                 gpu: GPUSpec) -> float:
    """Crude queueing-delay model: in colocated mode, every prefill blocks
    the decode pipeline for ``prefill_time``; aggregated across QPS arrivals
    the *expected* extra TTFT for a request that lands behind ``k`` prefills
    is ~ prefill_time × ρ/(1−ρ) (M/M/1 approximation) where ρ = qps × prefill_time."""
    pt = prefill_time_s(arch, seq_len, gpu)
    rho = min(0.99, qps * pt)
    return pt * rho / max(1e-9, 1 - rho)


def pd_overhead_s(arch: ModelArch, seq_len: int,
                  link: InterconnectSpec,
                  proxy_overhead_s: float = 0.005) -> float:
    """In PD mode, one extra step is the KV transfer + a small proxy hop."""
    return kv_transfer_time_s(arch, seq_len, link) + proxy_overhead_s


def pd_profitable(arch: ModelArch, seq_len: int, qps: float,
                  gpu: GPUSpec, link: InterconnectSpec,
                  proxy_overhead_s: float = 0.005) -> bool:
    """True when PD's transfer overhead is smaller than the queueing tax
    that colocated mode pays at this (ISL, QPS)."""
    return (colocated_prefill_overhead_s(arch, seq_len, qps, gpu)
            > pd_overhead_s(arch, seq_len, link, proxy_overhead_s))


def profitability_frontier(arch: ModelArch, gpu: GPUSpec,
                           link: InterconnectSpec,
                           isl_grid: List[int],
                           qps_grid: List[float]) -> Dict:
    """Return a 2D map and per-ISL break-even QPS."""
    grid = np.zeros((len(isl_grid), len(qps_grid)), dtype=bool)
    for i, isl in enumerate(isl_grid):
        for j, q in enumerate(qps_grid):
            grid[i, j] = pd_profitable(arch, isl, q, gpu, link)

    frontier = []
    for i, isl in enumerate(isl_grid):
        pd_o = pd_overhead_s(arch, isl, link)
        # Solve qps × pt × (qps × pt) / (1 − qps × pt) > pd_o approximately.
        # Closed-form: ρ = qps × pt, queue = pt·ρ/(1−ρ) = pd_o
        # → qps* = ρ*/pt, ρ* such that pt·ρ/(1−ρ)=pd_o → ρ = pd_o/(pt+pd_o)
        pt = prefill_time_s(arch, isl, gpu)
        rho_star = pd_o / (pt + pd_o)
        qps_star = rho_star / pt
        frontier.append({"isl": isl, "qps_break_even": qps_star,
                         "prefill_time_ms": pt * 1000,
                         "kv_xfer_ms": kv_transfer_time_s(arch, isl, link) * 1000})
    return {"grid": grid, "isl_grid": isl_grid, "qps_grid": qps_grid,
            "break_even_curve": frontier}


# ── Pretty-print report ────────────────────────────────────────

def report(arch: ModelArch, gpu: GPUSpec, link: InterconnectSpec) -> str:
    lines = []
    lines.append(f"Model: {arch.name}")
    lines.append(f"  weights:               {arch.weight_bytes / 1e9:6.2f} GB")
    lines.append(f"  KV per token:          {arch.kv_per_token_bytes / 1024:6.1f} KB")
    lines.append(f"")
    lines.append(f"GPU:  {gpu.name}")
    lines.append(f"  peak FP16:             {gpu.fp16_tflops:6.1f} TFLOPS")
    lines.append(f"  HBM bandwidth:         {gpu.hbm_bw_gbps:6.0f} GB/s")
    lines.append(f"  ridge point AI:        {ridge_point(gpu):6.0f} FLOP/byte")
    lines.append(f"")
    lines.append(f"Link: {link.name}  {link.bw_gbps:.0f} GB/s")
    lines.append(f"")
    lines.append(f"  ISL    AI(prefill)  prefill_t  KV_size   xfer_t   xfer/prefill")
    lines.append(f"  ----   ----------   --------   -------   ------   ------------")
    for isl in [128, 512, 1024, 2048, 4096, 8192]:
        f = prefill_flops(arch, isl)
        b = prefill_bytes(arch, isl)
        ai = arithmetic_intensity(f, b)
        pt = prefill_time_s(arch, isl, gpu) * 1000
        kv = isl * arch.kv_per_token_bytes / 1e6
        xt = kv_transfer_time_s(arch, isl, link) * 1000
        lines.append(f"  {isl:5d}  {ai:8.0f}    "
                     f"{pt:7.1f}ms  {kv:6.1f}MB  {xt:5.2f}ms     {xt/pt*100:5.2f}%")
    return "\n".join(lines)
