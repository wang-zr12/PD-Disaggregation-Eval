"""Async benchmark harness: send streaming requests, measure TTFT/TPOT."""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import List, Optional

import aiohttp

from .config import SERVED_MODEL_NAME, SERVER_PORT


@dataclass
class RequestResult:
    profile:       str
    target_isl:    int
    target_osl:    int
    ttft:          Optional[float]   # seconds
    total_time:    Optional[float]   # seconds
    output_tokens: Optional[int]
    tpot:          Optional[float]   # seconds / token (decode)
    success:       bool
    error:         str = ""


async def send_request(
    session: aiohttp.ClientSession,
    request: dict,
    server_url: str,
    model_name: str = SERVED_MODEL_NAME,
) -> RequestResult:
    payload = {
        "model":      model_name,
        "messages":   [{"role": "user", "content": request["prompt"]}],
        "max_tokens": request["max_tokens"],
        "stream":     True,
    }

    ttft = None
    output_tokens = 0
    t_start = time.perf_counter()
    try:
        async with session.post(
            f"{server_url}/v1/chat/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=120),
        ) as resp:
            async for chunk in resp.content:
                line = chunk.decode().strip()
                if line.startswith("data: ") and line != "data: [DONE]":
                    if ttft is None:
                        ttft = time.perf_counter() - t_start
                    output_tokens += 1

        total_time = time.perf_counter() - t_start
        tpot = (
            (total_time - ttft) / max(output_tokens - 1, 1)
            if output_tokens > 1 else None
        )
        return RequestResult(
            profile=request["profile"],
            target_isl=request["target_isl"],
            target_osl=request["target_osl"],
            ttft=ttft,
            total_time=total_time,
            output_tokens=output_tokens,
            tpot=tpot,
            success=True,
        )
    except Exception as e:
        return RequestResult(
            profile=request["profile"],
            target_isl=request["target_isl"],
            target_osl=request["target_osl"],
            ttft=None, total_time=None, output_tokens=None, tpot=None,
            success=False, error=str(e),
        )


async def run_benchmark(
    requests: list,
    concurrency: int,
    server_url: str = f"http://localhost:{SERVER_PORT}",
    progress_every: int = 10,
) -> List[RequestResult]:
    """Drive ``requests`` against ``server_url`` with a bounded semaphore."""
    sem = asyncio.Semaphore(concurrency)
    results: List[RequestResult] = []

    async def bounded(req, session):
        async with sem:
            return await send_request(session, req, server_url)

    connector = aiohttp.TCPConnector(limit=concurrency + 10)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [bounded(r, session) for r in requests]
        total = len(tasks)
        for i, coro in enumerate(asyncio.as_completed(tasks)):
            results.append(await coro)
            if (i + 1) % progress_every == 0:
                done = sum(1 for r in results if r.success)
                print(f"\r  progress: {i+1}/{total} ({done} ok)", end="")
    print()
    return results
