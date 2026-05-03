"""Streaming HTTP request → RequestResult."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import aiohttp

from .config import SERVED_MODEL_NAME


@dataclass
class RequestResult:
    profile:       str
    target_isl:    int
    target_osl:    int
    ttft:          Optional[float]   # seconds
    total_time:    Optional[float]   # seconds
    output_tokens: Optional[int]
    tpot:          Optional[float]   # seconds / token (decode only)
    success:       bool
    arrival_t:     Optional[float] = None   # epoch seconds, optional
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
    arrival_t = request.get("arrival_t")
    try:
        async with session.post(
            f"{server_url}/v1/chat/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=180),
        ) as resp:
            async for chunk in resp.content:
                line = chunk.decode().strip()
                if line.startswith("data: ") and line != "data: [DONE]":
                    if ttft is None:
                        ttft = time.perf_counter() - t_start
                    output_tokens += 1
        total_time = time.perf_counter() - t_start
        tpot = ((total_time - ttft) / max(output_tokens - 1, 1)
                if output_tokens > 1 else None)
        return RequestResult(
            profile=request["profile"],
            target_isl=request["target_isl"],
            target_osl=request["target_osl"],
            ttft=ttft, total_time=total_time,
            output_tokens=output_tokens, tpot=tpot,
            success=True, arrival_t=arrival_t,
        )
    except Exception as e:
        return RequestResult(
            profile=request["profile"],
            target_isl=request["target_isl"],
            target_osl=request["target_osl"],
            ttft=None, total_time=None, output_tokens=None, tpot=None,
            success=False, arrival_t=arrival_t, error=str(e),
        )
