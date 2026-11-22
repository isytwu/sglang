#!/usr/bin/env bash
# Runs INSIDE the yutongwu-epio container:  IOTC=104|0|off iobench_half.sh <node_rank> <master_ip> <my_ip> <port> <out.log> <benchmark args...>
# mori's own IO benchmark (tests/python/io/benchmark.py, PR #996) as in the DO microbenchmarks: node_rank 0 =
# initiator, 1 = target; run with --iters 1000000000 it never ends and is stopped with pkill.
set -uo pipefail
rank=$1; master=$2; myip=$3; port=$4; out=$5; shift 5
W=/apps/yutongwu/store-vllm/k31-vllm-ep-io-smci355-20261119
MB_MORI=pr996 source "$W/mb/env.sh"
mkdir -p "$(dirname "$out")"
cd "$MORI_SRC"
exec torchrun --nnodes=2 --node_rank="$rank" --nproc_per_node=1 --master_addr="$master" --master_port="$port" \
    tests/python/io/benchmark.py --host="$myip" "$@" > "$out" 2>&1
