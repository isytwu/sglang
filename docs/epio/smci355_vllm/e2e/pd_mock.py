#!/usr/bin/env python3
"""Stand-in vLLM server for testing e2e/pd_proxy.py without GPUs:  pd_mock.py <port> <prefill|decode> [gap_ms]
prefill: answers the 1-token leg with kv_transfer_params (one block per 64 prompt chars); decode: checks it got them
and streams 40 chat chunks gap_ms apart."""
import asyncio
import json
import sys
import time

from aiohttp import web

PORT, ROLE = int(sys.argv[1]), sys.argv[2]
GAP = float(sys.argv[3]) / 1e3 if len(sys.argv) > 3 else 0.05


async def chat(request):
    body = await request.json()
    kv = body.get("kv_transfer_params") or {}
    n = len(json.dumps(body.get("messages") or body.get("prompt") or ""))
    mx = body.get("max_completion_tokens") or body.get("max_tokens")
    if mx is not None and (body.get("min_tokens") or 0) > mx:
        return web.json_response({"error": "min_tokens > max_tokens"}, status=400)
    if ROLE == "prefill":
        assert kv.get("do_remote_decode") and kv.get("transfer_id"), kv
        return web.json_response({"id": "x", "choices": [{"index": 0, "message": {"role": "assistant", "content": "a"}}],
                                  "usage": {"prompt_tokens": n // 4, "completion_tokens": 1},
                                  "kv_transfer_params": {"do_remote_prefill": True, "remote_block_ids": [list(range(n // 64 + 1))],
                                                         "remote_host": "127.0.0.1", "transfer_id": kv["transfer_id"]}})
    assert kv.get("do_remote_prefill") and kv.get("remote_block_ids"), kv
    resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
    await resp.prepare(request)
    for _ in range(40):
        await asyncio.sleep(GAP)
        ev = {"id": "x", "object": "chat.completion.chunk", "created": int(time.time()), "model": body.get("model"),
              "choices": [{"index": 0, "delta": {"content": "tok "}, "finish_reason": None}]}
        await resp.write(f"data: {json.dumps(ev)}\n\n".encode())
    last = {"id": "x", "object": "chat.completion.chunk", "created": int(time.time()), "model": body.get("model"),
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": n // 4, "completion_tokens": 40, "total_tokens": n // 4 + 40}}
    await resp.write(f"data: {json.dumps(last)}\n\ndata: [DONE]\n\n".encode())
    await resp.write_eof()
    return resp


async def models(request):
    return web.json_response({"object": "list", "data": [{"id": "moonshotai/Kimi-K3.1-Pro", "object": "model"}]})


async def health(request):
    return web.Response(text="ok")


app = web.Application(client_max_size=1 << 31)
app.router.add_post("/v1/chat/completions", chat)
app.router.add_post("/v1/completions", chat)
app.router.add_get("/v1/models", models)
app.router.add_get("/health", health)
web.run_app(app, host="127.0.0.1", port=PORT, access_log=None)
