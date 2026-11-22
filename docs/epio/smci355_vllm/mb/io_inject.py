#!/usr/bin/env python3
"""MoRI IO RDMA traffic between two nodes, shaped like PD KV transfers, with a
controllable burst pattern. One process per GPU/NIC pair (GPU i <-> ionic_i, same rail).

  passive side ("recv", holds the remote buffer):  io_inject.py recv --devs 0-7 --port-base 47500 --mem gpu
  initiator ("send"):                              io_inject.py send --peer 10.15.20.8 --devs 0-7 --port-base 47500 \
                                                       --mem gpu --chunk-mb 64 --period-ms 10 --duration 30 --out DIR
With --op write (default) the initiator's buffer is the source and data lands on the passive side; with
--op read the initiator pulls the passive side's buffer into its own (RDMA READ, the way a PD decode that
reads its KV would). --numa local|remote places either side's host buffer on or off its NIC's NUMA node.
On this cluster's ionic VFs a READ into GPU memory completes but leaves the buffer untouched (bench/io_rwcheck.py);
READs into host memory and WRITEs land correctly.

Sender loop per device: a burst ("chunk") of --chunk-mb is posted every --period-ms
(start-to-start); if a burst takes longer than the period, the next starts when it ends
(--period-ms 0 = back-to-back = saturate). A burst is one batch_write of segments of
--seg-kb (default 72 = one K3 KV page: 128 tokens x 576 B fp8 MLA latent), like the sglang
MoRI connector's per-layer page runs. Backend config = sglang's (4 QP/transfer, 4 workers,
CQ polling). --mem host registers pinned host memory on both ends: NIC DMA without the GPU.

Per burst: t_submit, t_done (wall clock), bytes -> DIR/io_dev<i>.npz; prints a summary.
"""
import argparse
import json
import multiprocessing as mp
import os
import pickle
import socket
import struct
import time


def parse_devs(s):
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def die_with_parent():
    # children must not outlive the launcher (a killed docker exec leaves them holding GPU
    # memory and NIC registrations); mori/HIP teardown can also hang at exit -> os._exit
    import ctypes
    import signal

    ctypes.CDLL("libc.so.6").prctl(1, signal.SIGKILL)  # PR_SET_PDEATHSIG


def send_msg(sock, obj):
    b = pickle.dumps(obj)
    sock.sendall(struct.pack("!I", len(b)) + b)


def recv_msg(sock):
    hdr = b""
    while len(hdr) < 4:
        c = sock.recv(4 - len(hdr))
        if not c:
            return None
        hdr += c
    n = struct.unpack("!I", hdr)[0]
    buf = b""
    while len(buf) < n:
        c = sock.recv(n - len(buf))
        if not c:
            return None
        buf += c
    return pickle.loads(buf)


def ctl_ip():
    # the hostname resolves to 127.0.1.1 on these VMs; use the address that routes to eth1
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.15.20.1", 9))
        return s.getsockname()[0]
    finally:
        s.close()


def phys(dev):
    """The physical device behind HIP device dev (HIP_VISIBLE_DEVICES may reorder them); ionic_<phys> is its NIC."""
    vis = os.environ.get("HIP_VISIBLE_DEVICES", "")
    return int(vis.split(",")[dev]) if vis else dev


def make_engine(name, dev, args):
    import mori.io as mio

    os.environ["MORI_RDMA_DEVICES"] = f"ionic_{phys(dev)}"
    os.environ.setdefault("MORI_IB_GID_INDEX", "1")
    os.environ.setdefault("MORI_IO_QP_MAX_SEND_WR", "32767")
    host = args.bind_ip or ctl_ip()
    eng = mio.IOEngine(f"inj-{name}-{dev}-{os.getpid()}", mio.IOEngineConfig(host=host, port=0))
    cfg = mio.RdmaBackendConfig(args.qp, -1, args.workers, mio.PollCqMode.POLLING, False)
    eng.create_backend(mio.BackendType.RDMA, cfg)
    return eng


_keep = []
_libc = None


def bind_numa(dev, how):
    """Pin this process (CPUs and MPOL_BIND memory policy) to the NUMA node of ionic_<dev> ("local")
    or to the other node ("remote")."""
    import ctypes

    global _libc
    _libc = _libc or ctypes.CDLL("libc.so.6", use_errno=True)
    node = int(open(f"/sys/class/infiniband/ionic_{phys(dev)}/device/numa_node").read())
    if how == "remote":
        node = 1 - node
    os.sched_setaffinity(0, parse_devs(open(f"/sys/devices/system/node/node{node}/cpulist").read().strip()))
    mask = ctypes.c_ulong(1 << node)
    if _libc.syscall(238, 2, ctypes.byref(mask), ctypes.c_ulong(64)) != 0:  # set_mempolicy(MPOL_BIND, ...)
        raise OSError(ctypes.get_errno(), "set_mempolicy")
    return node


def page_nodes(addr, size, n=16):
    """NUMA nodes of n pages spread over [addr, addr + size) (move_pages in query mode)"""
    import ctypes

    pages = (ctypes.c_void_p * n)(*[addr + i * (size // n) for i in range(n)])
    status = (ctypes.c_int * n)()
    _libc.syscall(279, 0, ctypes.c_ulong(n), pages, None, status, 0)
    return sorted(set(status))


def alloc(nbytes, dev, mem, numa=None):
    import torch

    if mem == "gpu":
        torch.cuda.set_device(dev)
        return torch.empty(nbytes, dtype=torch.uint8, device=f"cuda:{dev}")
    if numa:
        node = bind_numa(dev, numa)
    if mem == "hosthuge" or numa:
        # hosthuge: 2 MiB hugetlb pages (like the UMBP DRAM tier), needs vm.nr_hugepages reserved.
        # With numa, host buffers are plain mmap pages faulted in under the memory policy: pin_memory()
        # places them by HIP's own rule (the current device's node) regardless of the policy.
        import mmap

        import numpy as np

        pg = (2 << 20) if mem == "hosthuge" else 4096
        size = (nbytes + pg - 1) // pg * pg
        flags = mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS | (0x40000 if mem == "hosthuge" else 0)  # MAP_HUGETLB
        m = mmap.mmap(-1, size, flags=flags)
        _keep.append(m)
        a = np.frombuffer(m, dtype=np.uint8, count=size)
        a[:: 4096] = 0  # fault the pages in
        if numa:
            print(f"dev {dev}: {mem} buffer bound to node {node}, sampled pages on nodes {page_nodes(a.ctypes.data, size)}", flush=True)
        return torch.from_numpy(a)
    return torch.empty(nbytes, dtype=torch.uint8).pin_memory()


def receiver(dev, args):
    die_with_parent()
    _receiver(dev, args)
    os._exit(0)


def _receiver(dev, args):
    import mori.io as mio

    eng = make_engine("rx", dev, args)
    buf = alloc(int(args.buf_gb * (1 << 30)), dev, args.mem, args.numa)
    md = eng.register_torch_tensor(buf)
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", args.port_base + dev))
    srv.listen(1)
    srv.settimeout(args.accept_timeout)
    while True:
        try:
            conn, _ = srv.accept()
        except socket.timeout:
            return
        send_msg(conn, (eng.get_engine_desc().pack(), md.pack()))
        peer = recv_msg(conn)
        if peer is None:
            conn.close()
            continue
        eng.register_remote_engine(mio.EngineDesc.unpack(peer))
        send_msg(conn, "ok")
        recv_msg(conn)  # blocks until the sender says done / closes
        conn.close()
        if not args.serve_forever:
            return


def sender(dev, args, out_q):
    die_with_parent()
    try:
        _sender(dev, args, out_q)
    except Exception as e:  # noqa: BLE001
        out_q.put((dev, {"error": repr(e)}))
    time.sleep(0.5)  # let the queue feeder flush
    os._exit(0)


def _sender(dev, args, out_q):
    import numpy as np

    import mori.io as mio

    eng = make_engine("tx", dev, args)
    nbytes = int(args.buf_gb * (1 << 30))
    buf = alloc(nbytes, dev, args.mem, args.numa)
    xfer = eng.batch_read if args.op == "read" else eng.batch_write
    md = eng.register_torch_tensor(buf)
    sock = None
    for _ in range(200):
        try:
            sock = socket.create_connection((args.peer, args.port_base + dev), timeout=5)
            break
        except OSError:
            time.sleep(0.1)
    if sock is None:
        out_q.put((dev, {"error": "connect failed"}))
        return
    sock.settimeout(None)
    rdesc, rmd = recv_msg(sock)
    eng.register_remote_engine(mio.EngineDesc.unpack(rdesc))
    remote = mio.MemoryDesc.unpack(rmd)
    send_msg(sock, eng.get_engine_desc().pack())
    assert recv_msg(sock) == "ok"

    seg = int(args.seg_kb * 1024)
    nseg = max(1, int(args.chunk_mb * (1 << 20)) // seg)
    span = nseg * seg * (2 if args.scatter else 1)
    nslots = max(1, nbytes // span)

    def chunk(c):
        base = (c % nslots) * span
        step = seg * (2 if args.scatter else 1)
        offs = [base + i * step for i in range(nseg)]
        return [md], [offs], [remote], [offs], [[seg] * nseg]

    # connect the QPs outside the timings
    st = xfer(*chunk(0), [eng.allocate_transfer_uid()])
    assert eng.wait_all(st) == mio.StatusCode.SUCCESS
    if args.start_at:
        time.sleep(max(0.0, args.start_at - time.time()))
    t0 = time.time()
    t_sub, t_done, nb = [], [], []
    pending = []
    c = 0
    period = args.period_ms / 1e3
    next_t = t0
    while True:
        now = time.time()
        if now - t0 >= args.duration:
            break
        if period > 0 and now < next_t:
            time.sleep(min(next_t - now, 0.0005) if next_t - now < 0.002 else next_t - now - 0.001)
            continue
        ts = time.time()
        st = xfer(*chunk(c), [eng.allocate_transfer_uid()])
        pending.append((ts, st))
        c += 1
        next_t = (next_t + period) if period > 0 else ts
        if period > 0 and next_t < ts - period:   # fell behind: no catch-up bursts
            next_t = ts
        while len(pending) >= args.inflight:
            ts0, st0 = pending.pop(0)
            rc = eng.wait_all(st0)
            t_sub.append(ts0); t_done.append(time.time()); nb.append(nseg * seg)
            if rc != mio.StatusCode.SUCCESS:
                out_q.put((dev, {"error": f"transfer failed rc={rc}"}))
                return
    for ts0, st0 in pending:
        eng.wait_all(st0)
        t_sub.append(ts0); t_done.append(time.time()); nb.append(nseg * seg)
    t1 = time.time()
    send_msg(sock, "done")
    sock.close()
    t_sub, t_done, nb = np.array(t_sub), np.array(t_done), np.array(nb)
    os.makedirs(args.out, exist_ok=True)
    np.savez(os.path.join(args.out, f"io_dev{dev}.npz"), t_submit=t_sub, t_done=t_done, bytes=nb,
             meta=json.dumps({"dev": dev, "args": vars(args), "host": socket.gethostname()}))
    lat = (t_done - t_sub) * 1e3
    out_q.put((dev, dict(bursts=int(nb.size), gbps=float(nb.sum() / (t1 - t0) / 1e9),
                         burst_ms_p50=float(np.median(lat)) if lat.size else None,
                         burst_ms_p99=float(np.percentile(lat, 99)) if lat.size else None,
                         burst_gbps_p50=float(np.median(nb / (t_done - t_sub)) / 1e9) if lat.size else None)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("role", choices=["recv", "send"])
    ap.add_argument("--devs", default="0-7")
    ap.add_argument("--port-base", type=int, default=47500)
    ap.add_argument("--peer")
    ap.add_argument("--bind-ip")
    ap.add_argument("--mem", choices=["gpu", "host", "hosthuge"], default="gpu")
    ap.add_argument("--op", choices=["write", "read"], default="write",
                    help="initiator side: write = push its buffer to the peer, read = pull the peer's buffer")
    ap.add_argument("--numa", choices=["local", "remote"],
                    help="this side's host buffer on (off) its NIC's NUMA node")
    ap.add_argument("--buf-gb", type=float, default=1.0)
    ap.add_argument("--chunk-mb", type=float, default=64)
    ap.add_argument("--seg-kb", type=float, default=72)
    ap.add_argument("--scatter", action="store_true", help="every other segment (non-contiguous)")
    ap.add_argument("--period-ms", type=float, default=0.0)
    ap.add_argument("--inflight", type=int, default=1)
    ap.add_argument("--qp", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--start-at", type=float, default=0.0, help="wall-clock start time")
    ap.add_argument("--accept-timeout", type=float, default=600)
    ap.add_argument("--serve-forever", action="store_true")
    ap.add_argument("--out", default="/tmp/io_inject")
    args = ap.parse_args()
    devs = parse_devs(args.devs)
    ctx = mp.get_context("spawn")
    if args.role == "recv":
        ps = [ctx.Process(target=receiver, args=(d, args)) for d in devs]
        for p in ps:
            p.start()
        print(f"receivers up on devs {devs} ports {args.port_base}+dev", flush=True)
        for p in ps:
            p.join()
        os._exit(0)
    q = ctx.Queue()
    ps = [ctx.Process(target=sender, args=(d, args, q)) for d in devs]
    for p in ps:
        p.start()
    res = {}
    for _ in ps:
        try:
            d, r = q.get(timeout=args.duration + 300)
        except Exception:  # noqa: BLE001
            break
        res[d] = r
    for p in ps:
        p.join(timeout=10)
    tot = sum(r.get("gbps", 0) for r in res.values())
    print(json.dumps({"total_GBps": round(tot, 2), "per_dev": {d: {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()} for d, r in sorted(res.items())}}), flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
