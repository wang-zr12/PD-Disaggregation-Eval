"""Cluster + model configuration for the PD benchmark.

Targets a single SSH-accessible host with 2× NVIDIA A100 40GB.
No Drive paths, no Colab assumptions.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

# ── Paths ────────────────────────────────────────────────────────
PD_BENCH_ROOT = os.environ.get(
    "PD_BENCH_ROOT",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)
DEFAULT_MODEL_DIR   = os.environ.get("PD_BENCH_MODEL_DIR",
                                     "/data/models")
DEFAULT_RESULTS_DIR = os.environ.get("PD_BENCH_RESULTS",
                                     f"{PD_BENCH_ROOT}/results")
DEFAULT_LOG_DIR     = os.environ.get("PD_BENCH_LOGS", "/tmp/pd_bench")

# ── Server defaults ─────────────────────────────────────────────
SERVED_MODEL_NAME = "qwen-coder"

@dataclass
class Ports:
    colocated: int = 8000          # single-GPU baseline
    pd_proxy:  int = 8000          # PD entry point (proxy)
    pd_prefill: int = 8100
    pd_decode:  int = 8200


# ── Model config ────────────────────────────────────────────────
@dataclass
class ModelConfig:
    """A model living on local disk (downloaded ahead of time, not Drive)."""
    model_id: str
    model_name: str
    dtype: str = "float16"
    quantization: Optional[str] = None
    max_model_len: int = 8192
    local_path: str = ""

    def with_local_path(self, model_dir: str) -> "ModelConfig":
        self.local_path = f"{model_dir}/{self.model_name}"
        return self


def qwen25_coder_7b(model_dir: str = DEFAULT_MODEL_DIR) -> ModelConfig:
    """Default model: Qwen2.5-Coder-7B-Instruct, FP16, fits comfortably on
    one A100 40GB even with KV transfer buffer reserved."""
    return ModelConfig(
        model_id="Qwen/Qwen2.5-Coder-7B-Instruct",
        model_name="Qwen2.5-Coder-7B",
        dtype="float16",
        quantization=None,
        max_model_len=8192,
    ).with_local_path(model_dir)


# ── Hardware / link characteristics ─────────────────────────────
@dataclass
class GPUSpec:
    name: str = "A100-40GB"
    fp16_tflops: float = 312.0       # peak FP16 (TC) TFLOPS
    hbm_bw_gbps: float = 1555.0      # HBM2e bandwidth, GB/s
    vram_gb:     float = 40.0


@dataclass
class InterconnectSpec:
    """Bandwidth between the two A100s on the host."""
    name: str = "NVLink3"
    bw_gbps: float = 600.0           # 600 GB/s for SXM4 NVLink3
    # Override to ~32 GB/s if the cloud SKU only exposes PCIe Gen4 peer copy.


# ── PD layout (per role) ────────────────────────────────────────
@dataclass
class PDLayout:
    """Which GPU runs which role; how much HBM each role can use."""
    prefill_gpu_id:   int = 0
    decode_gpu_id:    int = 1
    prefill_gpu_util: float = 0.85
    decode_gpu_util:  float = 0.85
    kv_buffer_size:   int = 5_000_000_000   # 5 GB NCCL buffer
    kv_parallel_size: int = 2               # 1 producer + 1 consumer
