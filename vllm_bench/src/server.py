"""vLLM OpenAI server: download model, launch, health-check, cleanup."""
from __future__ import annotations

import os
import signal
import subprocess
import time
from typing import Optional

import requests as req

from .config import ModelConfig, SERVED_MODEL_NAME, SERVER_LOG, SERVER_PORT


def check_model_cached(path: str) -> bool:
    required = ["config.json", "tokenizer.json"]
    if not os.path.exists(path):
        return False
    files = os.listdir(path)
    return all(f in files for f in required) and any(
        f.endswith(".safetensors") for f in files
    )


def download_model(model_cfg: ModelConfig) -> str:
    """Download from HF (mirror first, then official) into model_cfg.local_path."""
    from huggingface_hub import snapshot_download

    if check_model_cached(model_cfg.local_path):
        print(f"Model already cached: {model_cfg.local_path}")
        return model_cfg.local_path

    print(f"Downloading {model_cfg.model_id} -> {model_cfg.local_path}")
    os.makedirs(model_cfg.local_path, exist_ok=True)
    try:
        snapshot_download(
            repo_id=model_cfg.model_id,
            local_dir=model_cfg.local_path,
            local_dir_use_symlinks=False,
            endpoint="https://hf-mirror.com",
            ignore_patterns=["*.pt", "*.bin"],
        )
    except Exception as e:
        print(f"Mirror failed ({e}); falling back to official endpoint")
        snapshot_download(
            repo_id=model_cfg.model_id,
            local_dir=model_cfg.local_path,
            local_dir_use_symlinks=False,
            ignore_patterns=["*.pt", "*.bin"],
        )
    print("Download complete")
    return model_cfg.local_path


def _build_cmd(model_cfg: ModelConfig, port: int) -> list:
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_cfg.local_path,
        "--served-model-name", SERVED_MODEL_NAME,
        "--dtype", model_cfg.dtype,
        "--max-model-len", str(model_cfg.max_model_len),
        "--gpu-memory-utilization", str(model_cfg.gpu_mem_util),
        "--port", str(port),
        "--host", "0.0.0.0",
        "--disable-log-requests",
    ]
    if model_cfg.quantization:
        cmd += ["--quantization", model_cfg.quantization]
    return cmd


def is_alive(port: int = SERVER_PORT) -> bool:
    try:
        return req.get(f"http://localhost:{port}/health", timeout=2).status_code == 200
    except Exception:
        return False


def kill_existing(port: int = SERVER_PORT) -> None:
    try:
        out = subprocess.check_output(
            f"lsof -ti:{port}", shell=True, stderr=subprocess.DEVNULL
        ).decode().strip()
    except subprocess.CalledProcessError:
        return
    if not out:
        return
    for pid in out.split("\n"):
        try:
            os.kill(int(pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
    print(f"Killed stale processes on :{port}: {out}")
    time.sleep(2)


def wait_for_server(proc: subprocess.Popen, port: int = SERVER_PORT,
                    log_path: str = SERVER_LOG, timeout: int = 600) -> None:
    url = f"http://localhost:{port}/health"
    start = time.time()
    while True:
        rc = proc.poll()
        if rc is not None:
            print(f"\nServer exited rc={rc}; tail of {log_path}:")
            os.system(f"tail -80 {log_path}")
            raise RuntimeError(f"vLLM server failed to start (rc={rc})")
        try:
            if req.get(url, timeout=2).status_code == 200:
                return
        except Exception:
            pass
        elapsed = time.time() - start
        if elapsed > timeout:
            print(f"\nTimed out after {timeout}s; tail of {log_path}:")
            os.system(f"tail -80 {log_path}")
            raise RuntimeError("vLLM server start timeout")
        print(f"\rWaiting for server... {elapsed:.0f}s / {timeout}s", end="")
        time.sleep(5)


def start_server(model_cfg: ModelConfig, port: int = SERVER_PORT,
                 log_path: str = SERVER_LOG, wait_timeout: int = 600
                 ) -> Optional[subprocess.Popen]:
    """Start vLLM if not already running. Returns Popen, or None if reusing existing."""
    if is_alive(port):
        print(f"Server already running on :{port}, reusing")
        return None

    print(f"Starting vLLM server on :{port}")
    print(f"  model: {model_cfg.local_path}")
    print(f"  dtype={model_cfg.dtype} max_len={model_cfg.max_model_len} "
          f"gpu_util={model_cfg.gpu_mem_util} quant={model_cfg.quantization or 'none'}")
    print(f"  log: {log_path}")

    log_f = open(log_path, "w")
    proc = subprocess.Popen(
        _build_cmd(model_cfg, port),
        stdout=log_f, stderr=log_f,
        start_new_session=True,
        preexec_fn=os.setpgrp,
    )
    print(f"Started PID={proc.pid}")
    wait_for_server(proc, port=port, log_path=log_path, timeout=wait_timeout)
    print(f"\nServer ready on :{port}")
    return proc


def verify_inference(port: int = SERVER_PORT) -> str:
    resp = req.post(
        f"http://localhost:{port}/v1/chat/completions",
        json={
            "model": SERVED_MODEL_NAME,
            "messages": [{"role": "user", "content": "Write a Python hello world"}],
            "max_tokens": 50,
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def stop_server(proc: Optional[subprocess.Popen]) -> None:
    if proc is None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
    print("vLLM server stopped")
