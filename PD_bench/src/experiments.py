"""Experiment runners: 20 QPS headline comparison + 30-config sweep.

The architecture under test is opaque to this module — it sees only a
``server_url``. The launcher scripts produce the URL.
"""
from __future__ import annotations

import asyncio
import random
import time
from datetime import datetime
from typing import Dict, List, Tuple

import aiohttp

from .benchmark import RequestResult, send_request
from .load_gen import run_qps_load
from .metrics import compute_metrics, trace_records
from .workload import WorkloadGenerator
from .workload_dist import PD_PROFILES, sample_mixed, sample_profile


# ── Closed-loop warmup (flushes CUDA-graph capture) ─────────────

async def warmup(gen: WorkloadGenerator, server_url: str,
                 n: int = 20, conc: int = 4) -> None:
    factory = _request_factory(gen, "function_generation", seed_base=0)
    requests = [factory(i) for i in range(n)]
    sem = asyncio.Semaphore(conc)

    async def one(req, session):
        async with sem:
            return await send_request(session, req, server_url)

    async with aiohttp.ClientSession() as session:
        await asyncio.gather(*(one(r, session) for r in requests))


# ── Request factory (closure over generator + profile) ──────────

def _request_factory(gen: WorkloadGenerator, profile: str, seed_base: int):
    def make(i: int) -> dict:
        rng = random.Random(seed_base + i)
        if profile == "mixed":
            prof, isl, osl = sample_mixed(rng)
        else:
            isl, osl = sample_profile(profile, rng)
            prof = profile
        return {
            "prompt":     gen.generate_prompt(isl, seed=seed_base * 17 + i),
            "max_tokens": osl,
            "target_isl": isl,
            "target_osl": osl,
            "profile":    prof,
        }
    return make


# ── One QPS run = one row of the sweep ──────────────────────────

async def run_one(
    gen: WorkloadGenerator,
    profile: str,
    qps: float,
    duration_s: float,
    server_url: str,
    *,
    architecture: str,
    seed: int = 0,
    slo_ttft: float = 1.0,
) -> Tuple[dict, List[dict]]:
    factory = _request_factory(gen, profile, seed_base=seed * 100_000)
    t0 = time.time()
    results: List[RequestResult] = await run_qps_load(
        factory, qps=qps, duration_s=duration_s,
        server_url=server_url, seed=seed,
    )
    elapsed = time.time() - t0
    m = compute_metrics(results, slo_ttft=slo_ttft)
    m.update({
        "architecture": architecture,
        "profile":      profile,
        "qps":          qps,
        "duration_s":   duration_s,
        "wall_time":    elapsed,
        "slo_ttft":     slo_ttft,
        "timestamp":    datetime.now().isoformat(),
    })
    traces = trace_records(
        results, architecture=architecture, qps=qps, profile=profile,
    )
    return m, traces


# ── Headline 20-QPS mixed-workload comparison ───────────────────

async def run_headline_20qps(
    gen: WorkloadGenerator,
    server_urls: Dict[str, str],          # {"colocated": ..., "pd": ...}
    duration_s: float = 300,
    qps: float = 20.0,
    slo_ttft: float = 1.0,
    seed: int = 7,
) -> Tuple[List[dict], List[dict]]:
    metrics_rows, trace_rows = [], []
    for arch, url in server_urls.items():
        print(f"\n[headline] arch={arch:10s}  qps={qps}  duration={duration_s}s "
              f"slo<{slo_ttft}s")
        await warmup(gen, url)
        m, t = await run_one(
            gen, profile="mixed", qps=qps, duration_s=duration_s,
            server_url=url, architecture=arch, seed=seed, slo_ttft=slo_ttft,
        )
        print(f"  TTFT P50={m['ttft_p50']*1000:6.0f}ms  "
              f"P99={m['ttft_p99']*1000:6.0f}ms  "
              f"SLO<{slo_ttft}s={m['slo_rate']*100:5.1f}%  "
              f"thr={m['throughput_req_s']:.1f}req/s  N={m['n_requests']}")
        metrics_rows.append(m); trace_rows.extend(t)
    return metrics_rows, trace_rows


# ── 30-config sweep: arch × profile × qps ───────────────────────

DEFAULT_QPS_GRID = [5, 10, 20, 30, 50]
DEFAULT_PROFILES = ["inline_completion", "code_explanation", "function_generation"]


async def run_30config_sweep(
    gen: WorkloadGenerator,
    server_urls: Dict[str, str],
    *,
    profiles:   List[str]   = None,
    qps_grid:   List[float] = None,
    duration_s: float = 180,
    cooldown_s: float = 10,
) -> Tuple[List[dict], List[dict]]:
    """architectures × profiles × QPS = 30 configs by default (2·3·5)."""
    profiles = profiles or DEFAULT_PROFILES
    qps_grid = qps_grid or DEFAULT_QPS_GRID
    metrics_rows, trace_rows = [], []

    plan = [(arch, p, q) for arch in server_urls for p in profiles for q in qps_grid]
    print(f"Sweep: {len(plan)} configs  "
          f"({len(server_urls)} arch × {len(profiles)} profile × {len(qps_grid)} qps)")
    print(f"Each run: {duration_s}s + {cooldown_s}s cooldown ≈ "
          f"{len(plan) * (duration_s + cooldown_s) / 60:.1f} min total")

    last_arch = None
    for i, (arch, profile, qps) in enumerate(plan, 1):
        url = server_urls[arch]
        if arch != last_arch:
            print(f"\n>>> switching to architecture={arch} <<<")
            await warmup(gen, url)
            last_arch = arch
        slo = PD_PROFILES[profile]["TTFT_SLO"]
        print(f"\n[{i:2d}/{len(plan)}] arch={arch:10s}  "
              f"profile={profile:22s}  qps={qps}  slo<{slo}s")
        m, t = await run_one(
            gen, profile=profile, qps=qps, duration_s=duration_s,
            server_url=url, architecture=arch, seed=i, slo_ttft=slo,
        )
        print(f"   TTFT P50={m['ttft_p50']*1000:6.0f}ms  "
              f"P99={m['ttft_p99']*1000:6.0f}ms  "
              f"SLO={m['slo_rate']*100:5.1f}%  "
              f"thr={m['throughput_req_s']:.1f}req/s  N={m['n_requests']}")
        metrics_rows.append(m); trace_rows.extend(t)
        await asyncio.sleep(cooldown_s)

    return metrics_rows, trace_rows
