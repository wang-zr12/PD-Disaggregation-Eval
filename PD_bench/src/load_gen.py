"""Open-loop QPS load generator with Poisson arrivals.

Closed-loop drivers (semaphore-bounded) artificially flatten tail latency:
when a request blocks, no new arrival is generated. SLO measurement requires
the *opposite* — keep firing at λ even when the server stalls, so queueing
shows up in TTFT/TPOT distributions.
"""
from __future__ import annotations

import asyncio
import random
import time
from typing import Callable, List

import aiohttp

from .benchmark import RequestResult, send_request
from .config import SERVED_MODEL_NAME


async def run_qps_load(
    request_factory: Callable[[int], dict],
    qps: float,
    duration_s: float,
    server_url: str,
    *,
    model_name: str = SERVED_MODEL_NAME,
    max_inflight: int = 4096,
    seed: int = 0,
) -> List[RequestResult]:
    """Drive ``server_url`` at ``qps`` for ``duration_s`` seconds, then drain.

    ``request_factory(i)`` returns a request dict for arrival index ``i``.
    A request's ``arrival_t`` is stamped at dispatch time (perf_counter).
    """
    rng = random.Random(seed)
    results: List[RequestResult] = []
    inflight: set = set()
    sem = asyncio.Semaphore(max_inflight)

    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(connector=connector) as session:
        async def dispatch(req):
            async with sem:
                results.append(await send_request(session, req, server_url, model_name))

        t_start = time.perf_counter()
        deadline = t_start + duration_s
        i = 0
        while True:
            await asyncio.sleep(rng.expovariate(qps))
            now = time.perf_counter()
            if now >= deadline:
                break
            req = request_factory(i)
            req["arrival_t"] = now
            task = asyncio.create_task(dispatch(req))
            inflight.add(task)
            task.add_done_callback(inflight.discard)
            i += 1

        if inflight:
            print(f"  draining {len(inflight)} in-flight requests...")
            await asyncio.gather(*inflight, return_exceptions=True)

    return results
