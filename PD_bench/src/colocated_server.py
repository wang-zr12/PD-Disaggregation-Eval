"""Single-GPU vLLM baseline (the colocated-prefill-and-decode reference).

This is the architecture the PD layout is being compared *against*. It runs
one OpenAI-compatible vLLM server on one A100; the second A100 is left idle
during the baseline run."""
from __future__ import annotations

import os
import signal
import subprocess
import time
from typing import List, Optional

import requests as req

from .config import ModelConfig, SERVED_MODEL_NAME


def _build_cmd(model_cfg: ModelConfig, port: int, gpu_util: float) -> List[str]:
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model",                  model_cfg.local_path,
        "--served-model-name",      SERVED_MODEL_NAME,
        "--dtype",                  model_cfg.dtype,
        "--max-model-len",          str(model_cfg.max_model_len),
        "--gpu-memory-utilization", str(gpu_util),
        "--port",                   str(port),
        "--host",                   "0.0.0.0",
        "--disable-log-requests",
    ]
    if model_cfg.quantization:
        cmd += ["--quantization", model_cfg.quantization]
    return cmd


def is_alive(port: int) -> bool:
    try:
        return req.get(f"http://localhost:{port}/health",
                       timeout=2).status_code == 200
    except Exception:
        return False


def kill_existing(port: int) -> None:
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


def wait_for(proc: subprocess.Popen, port: int, log_path: str,
             timeout: int = 600) -> None:
    url = f"http://localhost:{port}/health"
    start = time.time()
    while True:
        rc = proc.poll()
        if rc is not None:
            print(f"\nServer exited rc={rc}; tail of {log_path}:")
            os.system(f"tail -120 {log_path}")
            raise RuntimeError(f"vLLM server failed to start (rc={rc})")
        try:
            if req.get(url, timeout=2).status_code == 200:
                return
        except Exception:
            pass
        elapsed = time.time() - start
        if elapsed > timeout:
            print(f"\nTimed out after {timeout}s; tail of {log_path}:")
            os.system(f"tail -120 {log_path}")
            raise RuntimeError("vLLM server start timeout")
        print(f"\rWaiting for server... {elapsed:.0f}s / {timeout}s", end="")
        time.sleep(5)


def start(model_cfg: ModelConfig, *, port: int, gpu_id: int = 0,
          gpu_util: float = 0.88, log_path: str = "/tmp/pd_bench/colocated.log",
          wait_timeout: int = 600) -> Optional[subprocess.Popen]:
    if is_alive(port):
        print(f"Server already running on :{port}, reusing")
        return None
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"Starting colocated vLLM on GPU{gpu_id} :{port}  log={log_path}")
    f = open(log_path, "w")
    proc = subprocess.Popen(
        _build_cmd(model_cfg, port, gpu_util),
        stdout=f, stderr=f, env=env,
        start_new_session=True, preexec_fn=os.setpgrp,
    )
    print(f"Started PID={proc.pid}")
    wait_for(proc, port, log_path, wait_timeout)
    print(f"\nServer ready on :{port}")
    return proc


def stop(proc: Optional[subprocess.Popen]) -> None:
    if proc is None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
    print("Colocated server stopped")
