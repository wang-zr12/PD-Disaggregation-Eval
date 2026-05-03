"""Top-level experiment runners. Each returns (metrics_list, traces_list)."""
from __future__ import annotations

import asyncio
import time
from datetime import datetime
from typing import List, Tuple

from .benchmark import run_benchmark
from .metrics import compute_metrics, trace_records
from .workload import WORKLOAD_PROFILES, WorkloadGenerator


def _n_requests_for(concurrency: int) -> int:
    """Low concurrency → 50 reqs; ramp to 8× concurrency, capped at 300."""
    return max(50, min(300, concurrency * 8))


async def run_concurrency_sweep(
    gen: WorkloadGenerator,
    profiles: List[str],
    concurrency_levels: List[int],
    server_url: str,
    *,
    warmup_requests: int = 20,
    cooldown_sec: int = 3,
    model_name: str = "",
    quantization: str = "none",
    serving_mode: str = "standard",
    include_mixed: bool = True,
    mixed_concurrency: int = 8,
    mixed_n: int = 200,
) -> Tuple[List[dict], List[dict]]:
    metrics_rows: List[dict] = []
    trace_rows:   List[dict] = []

    print(f"Warmup ({warmup_requests} requests)...")
    warm = gen.sample_workload("inline_completion", warmup_requests, seed=0)
    await run_benchmark(warm, concurrency=4, server_url=server_url)
    print("Warmup done\n" + "=" * 60)

    runs = [(p, c) for p in profiles for c in concurrency_levels]
    t_overall = time.time()
    for i, (profile, c) in enumerate(runs, 1):
        n_req = _n_requests_for(c)
        slo   = WORKLOAD_PROFILES[profile]["TTFT_SLO"]
        print(f"\n[{i}/{len(runs)}] {profile} | C={c} | N={n_req} | SLO={slo}s")

        requests = gen.sample_workload(profile, n_req, seed=i)
        t0 = time.time()
        results = await run_benchmark(requests, c, server_url=server_url)
        elapsed = time.time() - t0

        m = compute_metrics(results, slo_ttft=slo)
        m.update({
            "experiment_id": i,
            "profile":       profile,
            "concurrency":   c,
            "n_requests":    n_req,
            "slo_ttft":      slo,
            "serving_mode":  serving_mode,
            "model":         model_name,
            "quantization":  quantization,
            "wall_time":     elapsed,
            "timestamp":     datetime.now().isoformat(),
        })
        metrics_rows.append(m)
        trace_rows.extend(trace_records(
            results, experiment_id=i, concurrency=c,
        ))
        print(f"  TTFT P50={m['ttft_p50']*1000:6.0f}ms  P99={m['ttft_p99']*1000:6.0f}ms  "
              f"SLO={m['slo_rate']*100:5.1f}%  thr={m['throughput_req_s']:5.1f}req/s "
              f"({elapsed:.0f}s)")
        await asyncio.sleep(cooldown_sec)

    if include_mixed:
        print(f"\n[mixed] mixed workload, C={mixed_concurrency}")
        mixed_reqs = gen.sample_mixed_workload(mixed_n, seed=999)
        t0 = time.time()
        mixed_res = await run_benchmark(mixed_reqs, mixed_concurrency, server_url=server_url)
        elapsed = time.time() - t0
        m = compute_metrics(mixed_res, slo_ttft=1.5)
        m.update({
            "experiment_id": 999,
            "profile":       "mixed",
            "concurrency":   mixed_concurrency,
            "n_requests":    mixed_n,
            "slo_ttft":      1.5,
            "serving_mode":  serving_mode,
            "model":         model_name,
            "quantization":  quantization,
            "wall_time":     elapsed,
            "timestamp":     datetime.now().isoformat(),
        })
        metrics_rows.append(m)
        trace_rows.extend(trace_records(
            mixed_res, experiment_id=999, concurrency=mixed_concurrency,
        ))
        print(f"  TTFT P50={m['ttft_p50']*1000:.0f}ms  P99={m['ttft_p99']*1000:.0f}ms  "
              f"SLO={m['slo_rate']*100:.1f}%  thr={m['throughput_req_s']:.1f}req/s")

    print(f"\n{'=' * 60}\nDone: {len(metrics_rows)} runs, "
          f"{len(trace_rows)} traces, {(time.time()-t_overall)/60:.1f} min")
    return metrics_rows, trace_rows


async def run_isl_sweep(
    gen: WorkloadGenerator,
    isl_values: List[int],
    server_url: str,
    *,
    fixed_osl: int = 1024,
    fixed_concurrency: int = 8,
    n_per_isl: int = 100,
    cooldown_sec: int = 2,
) -> Tuple[List[dict], List[dict]]:
    metrics_rows, trace_rows = [], []

    print("Warmup (20 requests at ISL=1024)...")
    warm = gen.sample_fixed(1024, fixed_osl, 20, profile="warmup", seed=0)
    await run_benchmark(warm, fixed_concurrency, server_url=server_url)
    print("Warmup done\n" + "─" * 55)

    for isl in isl_values:
        reqs = gen.sample_fixed(isl, fixed_osl, n_per_isl,
                                profile="isl_sweep", seed=isl)
        results = await run_benchmark(reqs, fixed_concurrency, server_url=server_url)
        m = compute_metrics(results, slo_ttft=1.0)
        m["isl"] = isl
        m["n_requests"] = n_per_isl
        metrics_rows.append(m)
        trace_rows.extend(trace_records(results, isl=isl))
        print(f"  ISL={isl:5d}: P50={m['ttft_p50']*1000:5.0f}ms  "
              f"P99={m['ttft_p99']*1000:5.0f}ms  SLO@1s={m['slo_rate']*100:5.1f}%")
        await asyncio.sleep(cooldown_sec)
    return metrics_rows, trace_rows


async def run_osl_sweep(
    gen: WorkloadGenerator,
    osl_values: List[int],
    server_url: str,
    *,
    fixed_isl: int = 512,
    fixed_concurrency: int = 8,
    n_per_osl: int = 200,
    cooldown_sec: int = 2,
) -> Tuple[List[dict], List[dict]]:
    metrics_rows, trace_rows = [], []

    print("Warmup (20 requests, ISL=512, OSL=250)...")
    warm = gen.sample_fixed(fixed_isl, 250, 20, profile="warmup", seed=0)
    await run_benchmark(warm, fixed_concurrency, server_url=server_url)
    print("Warmup done\n" + "─" * 55)

    for osl in osl_values:
        reqs = gen.sample_fixed(fixed_isl, osl, n_per_osl,
                                profile="osl_sweep", seed=osl)
        results = await run_benchmark(reqs, fixed_concurrency, server_url=server_url)
        m = compute_metrics(results, slo_ttft=1.0)
        m["osl"] = osl
        m["isl"] = fixed_isl
        m["concurrency"] = fixed_concurrency
        m["n_requests"] = n_per_osl
        metrics_rows.append(m)
        trace_rows.extend(trace_records(
            results, osl=osl, isl=fixed_isl, concurrency=fixed_concurrency,
        ))
        tpot_p50 = (m.get("tpot_p50") or 0) * 1000
        e2e_p50  = m.get("total_time", 0) * 1000
        print(f"  OSL={osl:5d}: TTFT P50={m['ttft_p50']*1000:5.0f}ms  "
              f"TPOT P50={tpot_p50:5.1f}ms/tok  E2E P50={e2e_p50:6.0f}ms  "
              f"thr={m['throughput_tok_s']:.0f}tok/s")
        await asyncio.sleep(cooldown_sec)
    return metrics_rows, trace_rows
