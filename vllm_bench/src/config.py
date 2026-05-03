"""Global config: paths, model selection, server defaults."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class Paths:
    drive_root: str
    model_dir: str
    results_dir: str

    @classmethod
    def colab_default(cls, drive_root: str = "/content/drive/MyDrive/vllm_bench") -> "Paths":
        return cls(
            drive_root=drive_root,
            model_dir=f"{drive_root}/models",
            results_dir=f"{drive_root}/results",
        )

    def ensure(self) -> "Paths":
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        return self


@dataclass
class ModelConfig:
    model_id: str
    model_name: str
    dtype: str
    quantization: Optional[str]
    max_model_len: int
    gpu_mem_util: float
    local_path: str = ""  # populated after Paths is known

    def with_local_path(self, model_dir: str) -> "ModelConfig":
        self.local_path = f"{model_dir}/{self.model_name}"
        return self


def auto_select_model(total_vram_gb: float) -> ModelConfig:
    if total_vram_gb >= 38:
        return ModelConfig(
            model_id="Qwen/Qwen2.5-Coder-7B-Instruct",
            model_name="Qwen2.5-Coder-7B",
            dtype="float16",
            quantization=None,
            max_model_len=8192,
            gpu_mem_util=0.88,
        )
    if total_vram_gb >= 22:
        return ModelConfig(
            model_id="Qwen/Qwen2.5-Coder-7B-Instruct",
            model_name="Qwen2.5-Coder-7B-INT8",
            dtype="float16",
            quantization="bitsandbytes",
            max_model_len=4096,
            gpu_mem_util=0.90,
        )
    return ModelConfig(
        model_id="Qwen/Qwen2.5-Coder-3B-Instruct",
        model_name="Qwen2.5-Coder-3B",
        dtype="float16",
        quantization=None,
        max_model_len=4096,
        gpu_mem_util=0.88,
    )


def detect_vram_gb() -> float:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    return torch.cuda.get_device_properties(0).total_memory / 1024**3


# vLLM server defaults
SERVER_PORT = 8000
SERVER_LOG = "/tmp/vllm_server.log"
SERVED_MODEL_NAME = "qwen-coder"
