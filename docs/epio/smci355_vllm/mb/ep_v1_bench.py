#!/usr/bin/env python3
"""EP dispatch + combine with MoRI's v1 internode kernels, configured as vLLM's MoRI all2all configures them
(vllm/distributed/device_communicators/all2all.py MoriAll2AllManager + fused_moe/prepare_finalize/mori.py in the
k31-ms image): InterNodeV1LL (--all2all-backend mori_low_latency) or InterNodeV1 (mori_high_throughput),
gfx950 launch config 8 warps x 64 blocks / 32 RDMA blocks, bf16 tokens without scales, one QP per PE,
max_num_inp_token_per_rank = the scheduler's max_num_batched_tokens.

  torchrun --nnodes 2 --nproc_per_node 8 --node_rank R --master_addr A --master_port P \
      ep_v1_bench.py --tokens 128 --out <dir>

Timing follows the DO microbenchmarks (test_dispatch_combine_v2_internode_local.py): one event before the loop and
three per round that tile it, nothing synchronises inside the loop. dispatch = end of the previous round .. end of
this dispatch; combine = start .. end of this combine, which with --big-n also holds the local GEMMs that space the
rounds out to a decode's duty cycle (combine_ex = combine minus those GEMMs, timed on their own).
Every rank writes its per-round series to <out>/rank<r>.json; rank 0 prints one EPV1 summary line.
"""
import argparse
import json
import os
import statistics
import time

import torch
import torch.distributed as dist


def pct(v, q):
    s = sorted(v)
    return s[min(len(s) - 1, int(q * len(s)))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", choices=["v1_ll", "v1"], default="v1_ll")
    ap.add_argument("--tokens", type=int, default=128, help="tokens per rank per round")
    ap.add_argument("--max-tokens", type=int, default=1024, help="max_num_inp_token_per_rank")
    ap.add_argument("--hidden", type=int, default=3584)
    ap.add_argument("--topk", type=int, default=16)
    ap.add_argument("--experts-per-rank", type=int, default=56)
    ap.add_argument("--num-qp", type=int, default=1)
    ap.add_argument("--block-num", type=int, default=64)
    ap.add_argument("--rdma-block-num", type=int, default=32)
    ap.add_argument("--warp-per-block", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=2000)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--drop-rounds", type=int, default=10)
    ap.add_argument("--big-n", type=int, default=int(os.environ.get("EP_LOCAL_BIG_N", "0")),
                    help="GEMMs that each write 16 MB (~65 us) after every combine")
    ap.add_argument("--reps", type=int, default=1, help="timed loops in this process, each with its own series")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="cpu:gloo", rank=rank, world_size=world)
    import mori

    torch._C._distributed_c10d._register_process_group("mori", dist.group.WORLD)
    mori.shmem.shmem_torch_process_group_init("mori")
    kernel_type = {
        "v1_ll": mori.ops.EpDispatchCombineKernelType.InterNodeV1LL,
        "v1": mori.ops.EpDispatchCombineKernelType.InterNodeV1,
    }[args.kernel]
    cfg = mori.ops.EpDispatchCombineConfig(
        data_type=torch.bfloat16,
        rank=rank,
        world_size=world,
        hidden_dim=args.hidden,
        scale_dim=0,
        scale_type_size=0,
        max_token_type_size=torch.bfloat16.itemsize,
        max_num_inp_token_per_rank=args.max_tokens,
        num_experts_per_rank=args.experts_per_rank,
        num_experts_per_token=args.topk,
        warp_num_per_block=args.warp_per_block,
        block_num=args.block_num,
        kernel_type=kernel_type,
        rdma_block_num=args.rdma_block_num,
        gpu_per_node=min(8, world),
        num_qp_per_pe=args.num_qp,
    )
    op = mori.ops.EpDispatchCombineOp(cfg)

    rng = torch.Generator(device=device)
    rng.manual_seed(4242 + rank)
    n = args.tokens
    num_experts = args.experts_per_rank * world
    inp = torch.randn(n, args.hidden, generator=rng, device=device, dtype=torch.float32).to(torch.bfloat16)
    idx = torch.argsort(torch.rand(n, num_experts, generator=rng, device=device), dim=1)[:, : args.topk]
    idx = idx.to(torch.int32).contiguous()
    wts = torch.rand(n, args.topk, generator=rng, device=device, dtype=torch.float32)

    big_n = args.big_n
    ba = torch.randn(2048, 4096, device=device, dtype=torch.bfloat16) if big_n else None
    bb = torch.randn(4096, 4096, device=device, dtype=torch.bfloat16) if big_n else None
    bc = torch.empty(2048, 4096, device=device, dtype=torch.bfloat16) if big_n else None

    def round_trip():
        out = op.dispatch(inp, wts, None, idx)
        op.combine(out[0], None, idx)
        return out

    recv = 0
    for i in range(args.warmup):
        out = round_trip()
        if i == args.warmup - 1:
            torch.cuda.synchronize()
            recv = int(out[4][0].item())
    torch.cuda.synchronize()
    # A wrong configuration would still produce timings: one combine checked against its closed form. With combine
    # weights None every destination rank returns its received copy unweighted, so a token comes back multiplied by
    # the number of distinct ranks its top-k experts live on.
    out = op.dispatch(inp, wts, None, idx)
    comb = op.combine(out[0], None, idx)[0][:n].float()
    ranks_hit = torch.zeros(n, world, dtype=torch.int32, device=device)
    ranks_hit.scatter_add_(1, (idx.long() // args.experts_per_rank), torch.ones_like(idx))
    expect = inp.float() * (ranks_hit > 0).sum(1, keepdim=True).float()
    rel_err = float(((comb - expect).abs().max() / expect.abs().max().clamp_min(1e-6)).item())
    torch.cuda.synchronize()
    dist.barrier()

    os.makedirs(args.out, exist_ok=True)
    results = []
    for rep in range(args.reps):
        R = args.rounds
        ev = [torch.cuda.Event(enable_timing=True) for _ in range(3 * R + 1)]
        lev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(R)] if big_n else []
        dist.barrier()
        epoch0 = time.time()
        t0 = time.perf_counter()
        ev[0].record()
        for i in range(R):
            out = op.dispatch(inp, wts, None, idx)
            ev[3 * i + 1].record()
            ev[3 * i + 2].record()
            op.combine(out[0], None, idx)
            if big_n:
                lev[i][0].record()
                for _ in range(big_n):
                    torch.matmul(ba, bb, out=bc)
                lev[i][1].record()
            ev[3 * i + 3].record()
        torch.cuda.synchronize()
        wall_us = (time.perf_counter() - t0) * 1e6 / R
        keep = slice(args.drop_rounds, None)
        d = [ev[3 * i].elapsed_time(ev[3 * i + 1]) * 1e3 for i in range(R)][keep]
        c = [ev[3 * i + 2].elapsed_time(ev[3 * i + 3]) * 1e3 for i in range(R)][keep]
        loc = [lev[i][0].elapsed_time(lev[i][1]) * 1e3 for i in range(R)][keep] if big_n else [0.0] * len(c)
        cx = [a - b for a, b in zip(c, loc)]
        res = {
            "rank": rank, "rep": rep, "epoch0": epoch0, "epoch1": time.time(), "wall_us": wall_us,
            "dispatch_us": d, "combine_us": c, "combine_ex_us": cx, "local_us": loc,
        }
        results.append(res)
        stats = torch.tensor([statistics.mean(d), statistics.median(d), pct(d, 0.99),
                              statistics.mean(cx), statistics.median(cx), pct(cx, 0.99), wall_us], dtype=torch.float64)
        allst = [torch.zeros_like(stats) for _ in range(world)]
        dist.all_gather(allst, stats)
        if rank == 0:
            a = torch.stack(allst)
            gpn = min(8, world)
            nodes = " ".join(f"node{k}={a[k * gpn:(k + 1) * gpn, 0].mean():.1f}/{a[k * gpn:(k + 1) * gpn, 3].mean():.1f}"
                             for k in range(world // gpn))
            print(f"EPV1 rep={rep} kernel={args.kernel} world={world} tokens={n} hidden={args.hidden} topk={args.topk} "
                  f"epr={args.experts_per_rank} qp={args.num_qp} big_n={big_n} recv={recv} rel_err={rel_err:.2e} "
                  f"dispatch_mean={a[:, 0].mean():.2f} dispatch_med={a[:, 1].mean():.2f} dispatch_p99={a[:, 2].mean():.2f} "
                  f"combine_mean={a[:, 3].mean():.2f} combine_med={a[:, 4].mean():.2f} combine_p99={a[:, 5].mean():.2f} "
                  f"dispatch/combine_mean_by_node: {nodes} wall={a[:, 6].mean():.1f} t={time.time():.1f}", flush=True)
    with open(os.path.join(args.out, f"rank{rank}.json"), "w") as f:
        json.dump({"args": vars(args), "recv": recv, "rel_err": rel_err, "results": results}, f)
    torch.cuda.synchronize()
    dist.barrier()
    mori.shmem.shmem_finalize()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
