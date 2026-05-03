"""HTTP proxy that fans one client request to (prefill server, decode server).

Flow (vLLM 0.7.x ``PyNcclConnector``):

  1. Client → proxy.  Proxy clones the request body.
  2. Proxy → prefill server with ``max_tokens=1, stream=False``.
     The prefill server runs the prompt forward, *publishes* the KV cache via
     NCCL to the matching kv_consumer, then returns one token.
  3. Once prefill returns, proxy → decode server with the *original* request
     (full ``max_tokens``, streaming). The decode server skips its own prefill
     because the KV cache is already resident, so its first SSE event arrives
     after only one decode step.
  4. Proxy streams the decode response back to the client, byte-for-byte.

Run:
    python -m src.disagg_proxy --port 8000 \\
        --prefill-url http://localhost:8100 \\
        --decode-url  http://localhost:8200
"""
from __future__ import annotations

import argparse
import json
import logging

import aiohttp
from aiohttp import web

log = logging.getLogger("disagg_proxy")


async def _prime_prefill(session: aiohttp.ClientSession, prefill_url: str,
                         body: dict) -> None:
    primer = dict(body)
    primer["max_tokens"] = 1
    primer["stream"]     = False
    primer.pop("stream_options", None)
    async with session.post(f"{prefill_url}/v1/chat/completions",
                            json=primer,
                            timeout=aiohttp.ClientTimeout(total=120)) as r:
        await r.read()
        if r.status != 200:
            raise RuntimeError(f"prefill server returned {r.status}")


async def _stream_decode(session: aiohttp.ClientSession, decode_url: str,
                         body: dict, response: web.StreamResponse) -> None:
    async with session.post(f"{decode_url}/v1/chat/completions",
                            json=body,
                            timeout=aiohttp.ClientTimeout(total=600)) as r:
        if r.status != 200:
            err = await r.text()
            await response.write(f"data: {json.dumps({'error': err})}\n\n".encode())
            return
        async for chunk in r.content.iter_any():
            if chunk:
                await response.write(chunk)


async def _handle_chat(request: web.Request) -> web.StreamResponse:
    body = await request.json()
    streaming = body.get("stream", False)
    response = web.StreamResponse(
        status=200,
        headers={"Content-Type": "text/event-stream" if streaming
                                  else "application/json"},
    )
    await response.prepare(request)

    session: aiohttp.ClientSession = request.app["session"]
    try:
        await _prime_prefill(session, request.app["prefill_url"], body)
        await _stream_decode(session, request.app["decode_url"], body, response)
    except Exception as e:
        log.exception("proxy error")
        msg = json.dumps({"error": str(e)})
        await response.write(f"data: {msg}\n\n".encode() if streaming
                             else msg.encode())
    await response.write_eof()
    return response


async def _handle_health(request: web.Request) -> web.Response:
    session: aiohttp.ClientSession = request.app["session"]
    for name, url in [("prefill", request.app["prefill_url"]),
                      ("decode",  request.app["decode_url"])]:
        try:
            async with session.get(f"{url}/health",
                                   timeout=aiohttp.ClientTimeout(total=2)) as r:
                if r.status != 200:
                    return web.Response(status=503, text=f"{name}={r.status}")
        except Exception as e:
            return web.Response(status=503, text=f"{name}={e}")
    return web.Response(text="ok")


def build_app(prefill_url: str, decode_url: str) -> web.Application:
    app = web.Application(client_max_size=64 * 1024 * 1024)
    app["prefill_url"] = prefill_url
    app["decode_url"]  = decode_url

    async def _on_startup(app):
        app["session"] = aiohttp.ClientSession()

    async def _on_cleanup(app):
        await app["session"].close()

    app.on_startup.append(_on_startup)
    app.on_cleanup.append(_on_cleanup)
    app.router.add_post("/v1/chat/completions", _handle_chat)
    app.router.add_post("/v1/completions",       _handle_chat)
    app.router.add_get("/health",                _handle_health)
    return app


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port",         type=int, default=8000)
    ap.add_argument("--prefill-url",  default="http://localhost:8100")
    ap.add_argument("--decode-url",   default="http://localhost:8200")
    ap.add_argument("--log-level",    default="INFO")
    args = ap.parse_args()
    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s %(levelname)s %(message)s")
    app = build_app(args.prefill_url, args.decode_url)
    log.info("PD proxy on :%d  prefill=%s  decode=%s",
             args.port, args.prefill_url, args.decode_url)
    web.run_app(app, host="0.0.0.0", port=args.port, access_log=None)


if __name__ == "__main__":
    main()
