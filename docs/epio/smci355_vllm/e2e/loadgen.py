#!/usr/bin/env python3
"""Closed-loop streaming load without a tokenizer: /v1/completions with random prompt token ids, fixed output length.

  loadgen.py --urls http://10.0.0.1:8000,http://10.0.0.2:8000 --conc 128 --isl 1024 --osl 8000 --duration 3600 --out DIR

--conc streams per URL. Every streamed chunk's arrival time is kept per request; DIR/requests.jsonl gets one line per
finished request (start, first chunk, chunk times as offsets), DIR/live.jsonl every 10 s the number of open streams
and the chunks of the last 10 s, so a step time (the median gap between consecutive chunks of a stream = one decode
step at stream-interval 1) can be read for any window. Requests still open at the end are written with done=false.
"""
import argparse
import asyncio
import json
import os
import random
import time

import aiohttp


class Stats:
    def __init__(self, out):
        self.out = out
        self.reqs = open(os.path.join(out, "requests.jsonl"), "a")
        self.live = open(os.path.join(out, "live.jsonl"), "a")
        self.open_streams = 0
        self.ok = 0
        self.err = 0
        self.last_err = ""
        self.recent_gaps = []
        self.inflight = {}

    def write_req(self, rec):
        self.reqs.write(json.dumps(rec) + "\n")
        self.reqs.flush()


async def one_stream(session, url, args, stats, rnd, wid):
    ids = [rnd.randrange(1000, 150000) for _ in range(args.isl)]
    # output lengths spread over osl * (1 -+ jitter): streams that started together do not all finish together
    osl = max(16, int(args.osl * (1 + args.osl_jitter * (2 * rnd.random() - 1))))
    body = {"model": args.model, "prompt": ids, "max_tokens": osl, "ignore_eos": True, "stream": True,
            "temperature": 0.0}
    rec = {"url": url, "wid": wid, "t0": time.time(), "t_first": None, "chunks": [], "done": False}
    key = id(rec)
    stats.inflight[key] = rec
    stats.open_streams += 1
    last = None
    try:
        async with session.post(url + "/v1/completions", json=body) as r:
            if r.status != 200:
                raise RuntimeError(f"HTTP {r.status}: {(await r.text())[:200]}")
            async for raw in r.content:
                line = raw.strip()
                if not line.startswith(b"data:"):
                    continue
                if line == b"data: [DONE]":
                    break
                t = time.time()
                if rec["t_first"] is None:
                    rec["t_first"] = t
                else:
                    stats.recent_gaps.append(t - last)
                rec["chunks"].append(round(t - rec["t0"], 4))
                last = t
        rec["done"] = True
        stats.ok += 1
    except Exception as e:  # noqa: BLE001
        stats.err += 1
        stats.last_err = str(e)[:300]
        rec["err"] = stats.last_err
        await asyncio.sleep(1)
    finally:
        stats.open_streams -= 1
        stats.inflight.pop(key, None)
        rec["t_end"] = time.time()
        stats.write_req(rec)


async def worker(session, url, args, stats, wid, until):
    rnd = random.Random(args.seed * 1000 + wid)
    while time.time() < until:
        await one_stream(session, url, args, stats, rnd, wid)


async def reporter(stats, until):
    while time.time() < until:
        await asyncio.sleep(10)
        g = sorted(stats.recent_gaps)
        stats.recent_gaps = []
        rec = {"t": time.time(), "open": stats.open_streams, "ok": stats.ok, "err": stats.err, "gaps": len(g),
               "gap_med_ms": round(1e3 * g[len(g) // 2], 2) if g else None,
               "gap_p90_ms": round(1e3 * g[int(len(g) * 0.9)], 2) if g else None, "last_err": stats.last_err}
        stats.live.write(json.dumps(rec) + "\n")
        stats.live.flush()


async def main_async(args):
    os.makedirs(args.out, exist_ok=True)
    stats = Stats(args.out)
    urls = [u.rstrip("/") for u in args.urls.split(",")]
    until = time.time() + args.duration
    timeout = aiohttp.ClientTimeout(total=None, sock_read=args.read_timeout)
    conn = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(timeout=timeout, connector=conn) as session:
        tasks = []
        wid = 0
        for i in range(args.conc):
            for u in urls:
                tasks.append(asyncio.create_task(worker(session, u, args, stats, wid, until)))
                wid += 1
            if args.ramp_s:
                await asyncio.sleep(args.ramp_s / args.conc)
        rep = asyncio.create_task(reporter(stats, until + 5))
        stop = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in ("SIGINT", "SIGTERM"):
            import signal
            loop.add_signal_handler(getattr(signal, sig), stop.set)
        done_all = asyncio.gather(*tasks)
        stopper = asyncio.create_task(stop.wait())
        await asyncio.wait([done_all, stopper], return_when=asyncio.FIRST_COMPLETED)
        for t in tasks:
            t.cancel()
        rep.cancel()
        for rec in list(stats.inflight.values()):
            rec["t_end"] = time.time()
            stats.write_req(rec)
    print(json.dumps({"ok": stats.ok, "err": stats.err, "last_err": stats.last_err}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urls", required=True)
    ap.add_argument("--model", default="moonshotai/Kimi-K3.1-Pro")
    ap.add_argument("--conc", type=int, default=128, help="streams per URL")
    ap.add_argument("--isl", type=int, default=1024)
    ap.add_argument("--osl", type=int, default=8000)
    ap.add_argument("--osl-jitter", type=float, default=0.5, help="output length uniform in osl * (1 -+ jitter)")
    ap.add_argument("--duration", type=float, default=3600)
    ap.add_argument("--ramp-s", type=float, default=60, help="spread the stream starts over this many seconds")
    ap.add_argument("--read-timeout", type=float, default=900)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True)
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
