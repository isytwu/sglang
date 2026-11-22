#!/usr/bin/env bash
# Runs on the LOGIN node, unattended:  nohup bash tools/smoke.sh <2-node test job> > /dev/null 2>&1 &
# Shakes the pipeline down on a short 2-node allocation before the 3-node runs: the containers, the v1 EP16 bench on
# the image's mori, the PR #996 build (on B, in the background), then the K3.1 DP2 x TP8 EP16 decode
# (e2e/serve.sh decode_ep) up to a short streamed load and one profiler capture. Log: logs/smoke_<job>.log, results
# under results/smoke_<job>/.
set -uo pipefail
W=/apps/yutongwu/store-vllm/k31-vllm-ep-io-smci355-20261119
JOB=${1:?test job id}
export PATH=/opt/slurm/bin:$PATH
exec > >(tee -a "$W/logs/smoke_$JOB.log") 2>&1
echo "smoke $JOB $(date -u +%FT%TZ)"
while :; do
    st=$(squeue -h -j "$JOB" -o %T 2> /dev/null)
    [ "$st" = RUNNING ] && break
    [ -z "$st" ] && { echo "job $JOB is gone"; exit 1; }
    sleep 30
done
NODES=($(scontrol show hostnames "$(squeue -h -j "$JOB" -o %N)" | sed 's/^smci355-ccs-aus-//'))
export JOB A=${NODES[0]} B=${NODES[1]} C=${NODES[1]}
source "$W/mb/lib.sh"
O=$W/results/smoke_$JOB
mkdir -p "$O"
echo "nodes A=$A B=$B ($(date -u +%FT%TZ))"
ctl_ips "$A" "$B"
finish() {
    for n in "$A" "$B"; do
        hx "$n" "pkill -u \$(id -u) -f '[t]orchrun|[e]p_v1_bench|[t]est_dispatch_combine_v2|[v]llm serve|[V]LLM::|[l]oadgen.py'; sleep 3; pkill -9 -u \$(id -u) -f '[t]orchrun|[e]p_v1_bench|[t]est_dispatch_combine_v2|[v]llm serve|[V]LLM::|[l]oadgen.py'; true"
    done
    echo "smoke done, cancelling $JOB ($(date -u +%FT%TZ))"
    scancel "$JOB"
}
trap finish EXIT

# every wait names its pids: a bare wait would also wait for the tee of the exec above (bash 5.1)
sp=()
for n in "$A" "$B"; do
    (cx "$n" "echo container \$(hostname -s) ok; /opt/runtime-venv/bin/python -c 'import torch, mori; print(torch.__version__, torch.cuda.device_count(), mori.__file__)'; /opt/runtime-venv/bin/python $W/mb/topo_check.py" > "$O/setup_$n.log" 2>&1) &
    sp+=($!)
done
wait "${sp[@]}"
cat "$O"/setup_*.log
for n in "$A" "$B"; do echo "$n: amdgpu $(hx "$n" 'cat /sys/module/amdgpu/version') kfd $(gpu_busy "$n")"; done

# the PR #996 build only needs CPUs: it runs on B next to the v1 bench and the vLLM start-up; its EPv2 run is left to
# the 3-node job's warm phase (it would compete with vLLM for GPU memory here)
echo "--- build PR #996 on $B, v1 EP16 on $A + $B ($(date -u +%T))"
(cx "$B" "bash $W/mb/build.sh" > "$O/build.log" 2>&1; echo "build rc=$?" >> "$O/build.log") &
bpid=$!
export EP_TIMEOUT=900
echo "v1: $(ep_run "$O" v1 v1 --kernel v1_ll --tokens 128 --max-tokens 1024 --rounds 500 --warmup 50 --out "$O/v1_json")"
echo "v1 r0 tail:"; tail -5 "$O/v1_r0.log"
unset EP_TIMEOUT

echo "--- K3.1 decode DP2 x TP8 EP16 ($(date -u +%T))"
P=()
P+=("$(cxbg "$A" "$O/serve_$A.log" "env PROFILE_DIR=$O/prof PROFILE_ITERS=10 bash $W/e2e/serve.sh decode_ep 0 ${CTL[$A]} 8000")")
P+=("$(cxbg "$B" "$O/serve_$B.log" "env PROFILE_DIR=$O/prof PROFILE_ITERS=10 bash $W/e2e/serve.sh decode_ep 1 ${CTL[$A]} 8000")")
t0=$(date +%s)
ready=0
while (( $(date +%s) - t0 < 2700 )); do
    ok=0
    for n in "$A" "$B"; do hx "$n" "curl -sf -m 5 http://127.0.0.1:8000/health > /dev/null" && ok=$((ok + 1)); done
    [ $ok = 2 ] && { ready=1; break; }
    dead=0; for p in "${P[@]}"; do kill -0 "$p" 2> /dev/null || dead=1; done
    [ $dead = 1 ] && break
    sleep 15
done
echo "ready=$ready after $(( $(date +%s) - t0 ))s"
for n in "$A" "$B"; do echo "== serve_$n.log tail"; grep -v -E '^\s*$' "$O/serve_$n.log" | tail -25; done
if [ $ready = 1 ]; then
    cx "$A" "MB_MORI=image bash $W/mb/py.sh $W/e2e/loadgen.py --urls http://${CTL[$A]}:8000,http://${CTL[$B]}:8000 --conc 64 --isl 512 --osl 1500 --duration 240 --ramp-s 20 --out $O/load" > "$O/loadgen.log" 2>&1 &
    lp=$!
    sleep 150
    cp=(); for n in "$A" "$B"; do hx "$n" "curl -s -m 30 -X POST http://127.0.0.1:8000/start_profile" > /dev/null & cp+=($!); done
    wait "${cp[@]}"
    sleep 6
    cp=(); for n in "$A" "$B"; do hx "$n" "curl -s -m 600 -X POST http://127.0.0.1:8000/stop_profile" > /dev/null & cp+=($!); done
    wait "${cp[@]}"
    wait "$lp"
    cat "$O/loadgen.log"; tail -4 "$O/load/live.jsonl"
    ls -la "$O/prof" | head
    grep -h -E "Avg generation throughput" "$O"/serve_*.log | tail -6
fi
wait "$bpid"; tail -3 "$O/build.log"
