#!/usr/bin/env bash
# Runs INSIDE the yutongwu-epio container:  ep_half.sh <v1|v2> <node_rank> <nnodes> <master_ip> <port> <out.log> <bench args...>
# One node's part of an EP bench over nnodes nodes (EP = 8 x nnodes).
#   v1: mb/ep_v1_bench.py on the image's mori (vLLM's InterNodeV1LL path), torchrun with 8 processes per node
#   v2: ROCm/mori PR #996's EPv2 entry (the DO microbenchmarks; tests/python/ops/dispatch_combine_v2), torchrun with
#       one process per node that spawns the 8 ranks. EP_SCRIPT=test_dispatch_combine_v2_internode_local.py (a copy
#       of mb/'s, placed next to the entry because it imports _data from there) adds the EP_LOCAL_BIG_N duty cycle.
set -uo pipefail
kind=$1; rank=$2; nnodes=$3; master=$4; port=$5; out=$6; shift 6
W=/apps/yutongwu/store-vllm/k31-vllm-ep-io-smci355-20261119
mkdir -p "$(dirname "$out")"
if [ "$kind" = v1 ]; then
    MB_MORI=image source "$W/mb/env.sh"
    # vLLM passes its launch config explicitly and leaves the tuning tables alone
    unset MORI_EP_LAUNCH_CONFIG_MODE MORI_EP_DISABLE_RDMA_ATOMICS
    cd /tmp
    timeout "${EP_TIMEOUT:-900}" torchrun --nnodes="$nnodes" --node_rank="$rank" --nproc_per_node=8 \
        --master_addr="$master" --master_port="$port" "$W/mb/ep_v1_bench.py" "$@" > "$out" 2>&1
else
    MB_MORI=pr996 source "$W/mb/env.sh"
    cd "$MORI_SRC"
    timeout "${EP_TIMEOUT:-900}" torchrun --nnodes="$nnodes" --node_rank="$rank" --nproc_per_node=1 \
        --master_addr="$master" --master_port="$port" \
        "tests/python/ops/dispatch_combine_v2/${EP_SCRIPT:-test_dispatch_combine_v2_internode.py}" --cmd bench "$@" > "$out" 2>&1
fi
echo "rc=$?" >> "$out"
