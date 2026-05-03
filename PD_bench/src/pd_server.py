"""Launch a Prefill–Decode disaggregated vLLM deployment on one host.

  GPU 0  →  prefill server   (kv_role=kv_producer)  :8100
  GPU 1  →  decode  server   (kv_role=kv_consumer)  :8200
                                                     │
                                  proxy (sibling)  ──┘   :8000

The proxy primes the prefill server (max_tokens=1) so it ships the KV cache
via NCCL, then forwards the real request to the decode server. Both vLLM
servers share the same model directory and tokenizer.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass
from typing import List

import requests as req

from .config import ModelConfig, PDLayout, Ports, SERVED_MODEL_NAME


def _kv_cfg(role: str, rank: int, layout: PDLayout) -> str:
    return json.dumps({
        "kv_connector":     "PyNcclConnector",
        "kv_role":          role,
        "kv_rank":          rank,
        "kv_parallel_size": layout.kv_parallel_size,
        "kv_buffer_size":   layout.kv_buffer_size,
    })


def _build_cmd(model_cfg: ModelConfig, port: int, kv_cfg: str,
               gpu_util: float) -> List[str]:
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
        "--kv-transfer-config",     kv_cfg,
    ]
    if model_cfg.quantization:
        cmd += ["--quantization", model_cfg.quantization]
    return cmd


@dataclass
class PDDeployment:
    procs:     List[subprocess.Popen]
    log_paths: List[str]
    ports:     Ports


def start(model_cfg: ModelConfig, layout: PDLayout = PDLayout(),
          ports: Ports = Ports(), log_dir: str = "/tmp/pd_bench",
          wait_timeout: int = 900) -> PDDeployment:
    os.makedirs(log_dir, exist_ok=True)
    procs:  List[subprocess.Popen] = []
    logs:   List[str] = []
    base_env = os.environ.copy()
    base_env.setdefault("VLLM_HOST_IP", "127.0.0.1")
    base_env.setdefault("NCCL_DEBUG", "WARN")

    for role, rank, gpu_id, port, util in [
        ("kv_producer", 0, layout.prefill_gpu_id, ports.pd_prefill,
         layout.prefill_gpu_util),
        ("kv_consumer", 1, layout.decode_gpu_id,  ports.pd_decode,
         layout.decode_gpu_util),
    ]:
        env = dict(base_env)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        cmd = _build_cmd(model_cfg, port, _kv_cfg(role, rank, layout), util)
        log_path = f"{log_dir}/{role}_{port}.log"
        logs.append(log_path)
        f = open(log_path, "w")
        print(f"  starting {role:11s} GPU{gpu_id} :{port} → {log_path}")
        p = subprocess.Popen(cmd, stdout=f, stderr=f, env=env,
                             start_new_session=True, preexec_fn=os.setpgrp)
        procs.append(p)

    _wait_both(ports, procs, logs, wait_timeout)
    return PDDeployment(procs=procs, log_paths=logs, ports=ports)


def _wait_both(ports: Ports, procs: List[subprocess.Popen],
               logs: List[str], timeout: int) -> None:
    targets = [("prefill", ports.pd_prefill), ("decode", ports.pd_decode)]
    start_t = time.time()
    while True:
        for p, lp in zip(procs, logs):
            rc = p.poll()
            if rc is not None:
                print(f"\nServer exited rc={rc}; log:\n  {lp}")
                os.system(f"tail -120 {lp}")
                raise RuntimeError(f"PD server failed (rc={rc})")
        ready = []
        for name, port in targets:
            try:
                if req.get(f"http://localhost:{port}/health",
                           timeout=2).status_code == 200:
                    ready.append(name)
            except Exception:
                pass
        if len(ready) == 2:
            print(f"\nBoth PD servers ready ({(time.time()-start_t):.0f}s)")
            return
        elapsed = time.time() - start_t
        if elapsed > timeout:
            for lp in logs:
                print(f"\n--- {lp} ---")
                os.system(f"tail -120 {lp}")
            raise RuntimeError(f"PD servers timeout after {timeout}s")
        print(f"\rwaiting... ready={ready}  {elapsed:.0f}s/{timeout}s", end="")
        time.sleep(5)


def stop(dep: PDDeployment) -> None:
    for p in dep.procs:
        try:
            p.terminate()
        except ProcessLookupError:
            pass
    for p in dep.procs:
        try:
            p.wait(timeout=30)
        except subprocess.TimeoutExpired:
            p.kill()
    print("PD vLLM servers stopped")
