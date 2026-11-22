#!/usr/bin/env python3
"""OpenAI-compatible proxy for a vLLM prefill / decode split with the MoRIIO connector in READ mode.

  pd_proxy.py --prefill http://P1:8100 [--prefill ...] --decode http://D0:8000 [--decode ...] --port 8000 --out DIR

Per request: the prefill leg (the client's body with max_tokens 1, not streamed, kv_transfer_params do_remote_decode
and a transfer_id) runs on one prefill server; the kv_transfer_params it returns (its block ids, host, ports, DP rank)
go with the client's body to one decode server, whose MoRIIO connector RDMA-reads the KV it does not hold and then
streams the answer back through here. The decode servers are the API servers of the decode's DP ranks (external LB:
one per rank). Both legs of a session (AIPerf's X-Correlation-ID, or --session-header) go to the same prefill and
the same decode rank (rendezvous hashing), so a conversation's later turns hit the prefix caches; requests without
a session id go to the decode rank with the fewest open streams.

DIR/requests.jsonl: one line per request in loadgen.py's format (t0, t_first, chunk offsets) plus the prefill and
decode servers, the prefill leg's timing, the prompt tokens and the number of KV blocks the prefill offered.
DIR/live.jsonl every 10 s (open streams, token gaps: what e2e/windows.sh's alive() reads), DIR/series.jsonl every
second (token gaps, decode legs started and the KV blocks they were offered, per decode server).
"""
import argparse
import asyncio
import hashlib
import json
import os
import time
import uuid

from aiohttp import ClientSession, ClientTimeout, TCPConnector, web


def blocks_of(kv):
    ids = (kv or {}).get("remote_block_ids") or []
    if ids and isinstance(ids[0], list):
        return len(ids[0])
    return len(ids)


class Stats:
    def __init__(self, out, decodes):
        os.makedirs(out, exist_ok=True)
        self.reqs = open(os.path.join(out, "requests.jsonl"), "a")
        self.live = open(os.path.join(out, "live.jsonl"), "a")
        self.series = open(os.path.join(out, "series.jsonl"), "a")
        self.open = {d: 0 for d in decodes}
        self.ok = self.err = 0
        self.last_err = ""
        self.gaps10, self.gaps1 = [], {d: [] for d in decodes}
        self.dstart = {d: [0, 0] for d in decodes}
        self.pstart = self.pdone = 0
        self.dumped = self.rr = 0

    @staticmethod
    def pct(g, q):
        return round(1e3 * g[min(len(g) - 1, int(len(g) * q))], 2) if g else None

    async def reporter(self):
        n = 0
        while True:
            await asyncio.sleep(1.0 - time.time() % 1.0)
            t = int(time.time())
            per = {}
            for d, g in self.gaps1.items():
                g.sort()
                per[d] = {"gaps": len(g), "med": self.pct(g, 0.5), "p90": self.pct(g, 0.9), "open": self.open[d],
                          "dstart": self.dstart[d][0], "dblocks": self.dstart[d][1]}
                self.gaps10 += g
                self.gaps1[d] = []
                self.dstart[d] = [0, 0]
            self.series.write(json.dumps({"t": t, "pstart": self.pstart, "pdone": self.pdone, "per": per}) + "\n")
            self.pstart = self.pdone = 0
            n += 1
            if n % 10 == 0:
                g = sorted(self.gaps10)
                self.gaps10 = []
                self.live.write(json.dumps({"t": time.time(), "open": sum(self.open.values()), "ok": self.ok,
                                            "err": self.err, "gaps": len(g), "gap_med_ms": self.pct(g, 0.5),
                                            "gap_p90_ms": self.pct(g, 0.9), "last_err": self.last_err}) + "\n")
                for f in (self.live, self.series, self.reqs):
                    f.flush()


def pick(urls, key):
    return max(urls, key=lambda u: hashlib.sha1(f"{key}|{u}".encode()).digest())


async def forward_get(request):
    app = request.app
    async with app["session"].get(app["args"].decode[0] + request.path_qs) as r:
        return web.Response(body=await r.read(), status=r.status, content_type=r.content_type)


async def handle(request):
    app = request.app
    args, st, sess = app["args"], app["stats"], app["session"]
    body = await request.json()
    if st.dumped < 3:
        st.dumped += 1
        small = {k: v for k, v in body.items() if not isinstance(v, (list, dict)) or k == "stream_options"}
        print(f"request fields: {sorted(body)}; scalars {small}; headers {sorted(request.headers)}", flush=True)
    key = request.headers.get(args.session_header) or request.headers.get("X-Correlation-ID")
    if key:
        p_url, d_url = pick(args.prefill, key), pick(args.decode, key)
    else:
        p_url = args.prefill[st.rr % len(args.prefill)]
        st.rr += 1
        d_url = min(args.decode, key=lambda d: st.open[d])
    tid = uuid.uuid4().hex
    rec = {"url": d_url, "p": p_url, "sess": key, "t0": time.time(), "t_first": None, "chunks": [], "done": False}
    stream = bool(body.get("stream"))
    st.open[d_url] += 1
    resp = None
    try:
        pb = dict(body)
        pb["max_tokens"] = 1
        if "max_completion_tokens" in pb:
            pb["max_completion_tokens"] = 1
        pb["stream"] = False
        for k in ("stream_options", "min_tokens"):
            pb.pop(k, None)
        pb["kv_transfer_params"] = {"do_remote_decode": True, "do_remote_prefill": False, "remote_engine_id": None,
                                    "remote_block_ids": None, "remote_host": None, "remote_port": None,
                                    "transfer_id": tid}
        st.pstart += 1
        async with sess.post(p_url + request.path, json=pb, headers={"X-Request-Id": f"p-{tid}"}) as r:
            if r.status != 200:
                raise RuntimeError(f"prefill HTTP {r.status}: {(await r.text())[:300]}")
            pj = await r.json()
        st.pdone += 1
        rec["t_pf"] = round(time.time() - rec["t0"], 4)
        kv = pj.get("kv_transfer_params")
        if not kv:
            raise RuntimeError(f"prefill returned no kv_transfer_params: {str(pj)[:300]}")
        usage = pj.get("usage") or {}
        rec["prompt_tokens"] = usage.get("prompt_tokens")
        rec["p_cached"] = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
        rec["kv_blocks"] = blocks_of(kv)
        st.dstart[d_url][0] += 1
        st.dstart[d_url][1] += rec["kv_blocks"]
        db = dict(body)
        db["kv_transfer_params"] = kv
        async with sess.post(d_url + request.path, json=db, headers={"X-Request-Id": f"d-{tid}"}) as r:
            if r.status != 200:
                raise RuntimeError(f"decode HTTP {r.status}: {(await r.text())[:300]}")
            if not stream:
                data = await r.read()
                rec["t_first"] = time.time()
                rec["done"] = True
                st.ok += 1
                return web.Response(body=data, status=200, content_type="application/json")
            resp = web.StreamResponse(status=200, headers={"Content-Type": "text/event-stream",
                                                           "Cache-Control": "no-cache"})
            await resp.prepare(request)
            last = None
            async for raw in r.content:
                line = raw.strip()
                if line.startswith(b"data:") and line != b"data: [DONE]":
                    t = time.time()
                    if rec["t_first"] is None:
                        rec["t_first"] = t
                    else:
                        st.gaps1[d_url].append(t - last)
                    rec["chunks"].append(round(t - rec["t0"], 4))
                    last = t
                await resp.write(raw)
            await resp.write_eof()
        rec["done"] = True
        st.ok += 1
        return resp
    except (ConnectionResetError, asyncio.CancelledError):
        rec["err"] = "client went away"
        raise
    except Exception as e:  # noqa: BLE001
        st.err += 1
        st.last_err = rec["err"] = str(e)[:300]
        if resp is not None:
            return resp
        return web.json_response({"error": {"message": rec["err"], "type": "proxy_error"}}, status=502)
    finally:
        st.open[d_url] -= 1
        rec["t_end"] = time.time()
        st.reqs.write(json.dumps(rec) + "\n")


async def on_startup(app):
    # force_close: a fresh connection per leg. uvicorn drops idle keep-alive connections after 5 s, and a leg sent
    # down one it had just dropped died with a broken pipe (pdn_6954: a 225k-token request after 338 s in the
    # prefill queue), which ends an AgentX run during its warmup
    app["session"] = ClientSession(connector=TCPConnector(limit=0, force_close=True, ttl_dns_cache=300),
                                   timeout=ClientTimeout(total=None, sock_connect=30, sock_read=3600))
    app["reporter"] = asyncio.create_task(app["stats"].reporter())


async def on_cleanup(app):
    app["reporter"].cancel()
    await app["session"].close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefill", action="append", required=True)
    ap.add_argument("--decode", action="append", required=True)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--session-header", default="X-Session-ID")
    args = ap.parse_args()
    app = web.Application(client_max_size=1 << 31)
    app["args"], app["stats"] = args, Stats(args.out, args.decode)
    app.router.add_post("/v1/chat/completions", handle)
    app.router.add_post("/v1/completions", handle)
    app.router.add_get("/v1/models", forward_get)
    app.router.add_get("/health", forward_get)
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    print(f"pd_proxy :{args.port} prefill {args.prefill} decode {args.decode} -> {args.out}", flush=True)
    web.run_app(app, host="0.0.0.0", port=args.port, access_log=None, backlog=4096)


if __name__ == "__main__":
    main()
