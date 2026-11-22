# Sourced on the smci355 LOGIN node by the microbenchmark drivers. Needs JOB (the holder job) and the node roles
# A, B, C (short names such as n08-25): EP16 on A (EP rank 0, master) + B, the third node C is the IO source.
# Every step runs in the node's named pyxis container (k31-ms vLLM image, created on first use, kept for the job);
# everything writes straight to NFS under $W.
W=/apps/yutongwu/store-vllm/k31-vllm-ep-io-smci355-20261119
SQSH=/apps/yutongwu/store-vllm/k31-umbp-20261112/images/yukiozzz_vllm-openai-rocm_k31-ms-1114.sqsh
CNAME=yutongwu-epio
IONIC=/usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so
# /etc/slurm: slurmstepd runs the TaskProlog (/etc/slurm/task_prolog.sh) inside the container
MOUNTS="/apps:/apps,/etc/slurm:/etc/slurm,/dev/kfd:/dev/kfd,/dev/dri:/dev/dri,/dev/infiniband:/dev/infiniband,$IONIC:$IONIC:ro,/data/models:/data/models:ro,/perf_apps/data/models:/perf_apps/data/models:ro"
PRE="export RAIL_ALIGN=${RAIL_ALIGN:-0}; "'ulimit -S -l "$(ulimit -H -l)" 2>/dev/null; '
export PATH=/opt/slurm/bin:$PATH

host() { echo "smci355-ccs-aus-$1"; }
# hx <node> <cmd>: on the host (ssh), e.g. the read-only NIC counters, or pkill of our own processes
hx() { local n=$1; shift; ssh -o BatchMode=yes -o ConnectTimeout=10 "$(host "$n")" "$*" < /dev/null; }
# --cpu-bind=none: a step's tasks would otherwise be pinned to its CPU allocation
SRUN_C=(--overlap -N1 -n1 -c 128 --mem=0 --cpu-bind=none --job-name=yt-epio --kill-on-bad-exit=0
        --container-image="$SQSH" --container-name="$CNAME" --container-mounts="$MOUNTS"
        --container-writable --container-remap-root --container-workdir=/tmp)
# cx <node> <cmd>: one srun step in the node's container, foreground
cx() { local n=$1; shift; srun --jobid="$JOB" -w "$(host "$n")" "${SRUN_C[@]}" bash -c "$PRE$*" < /dev/null; }
# cxbg <node> <log> <cmd>: the same in the background; echoes the srun client's pid (SIGTERM to it ends the step)
cxbg() {
    local n=$1 log=$2; shift 2
    mkdir -p "$(dirname "$log")"
    srun --jobid="$JOB" -w "$(host "$n")" "${SRUN_C[@]}" bash -c "$PRE$*" < /dev/null > "$log" 2>&1 &
    echo $!
}

declare -A CTL
ctl_ips() { for n in "$@"; do CTL[$n]=$(hx "$n" "ip -4 -o addr show enp81s0f1 | awk '{print \$4}' | cut -d/ -f1"); done; }

# gpu_busy <node>: number of KFD processes on the node (0 = idle)
gpu_busy() { hx "$1" "ls /sys/class/kfd/kfd/proc 2>/dev/null | wc -l"; }
# wait_idle <secs> <nodes...>: until no KFD process is left and ports 8000 / 13345 are free on the nodes (a phase that
# starts while the previous one's processes are still exiting collides with their ports: e1x_6857)
wait_idle() {
    local to=$1 t0 n busy; shift
    t0=$(date +%s)
    while :; do
        busy=""
        for n in "$@"; do
            [ "$(hx "$n" "ls /sys/class/kfd/kfd/proc 2>/dev/null | wc -l; ss -ltn | grep -c -E ':(8000|13345) '" | paste -sd+ | bc)" = 0 ] || busy="$busy $n"
        done
        [ -z "$busy" ] && return 0
        (( $(date +%s) - t0 > to )) && { echo "still busy after ${to}s:$busy"; return 1; }
        sleep 5
    done
}

snap() { local d=$1 tag=$2; shift 2; mkdir -p "$d"; for n in "$@"; do hx "$n" "bash $W/mb/snap.sh $tag" >> "$d/snap.jsonl"; done; }
rate() { hx "$1" "bash $W/mb/rate.sh ${2:-2}"; }
# rates <file>: the EP nodes as A, B, D, E (in EP rank order) and the IO node as C
rates() {
    local f=$1 i=0 n labels=(A B D E F G H)
    { for n in $(ep_nodes); do echo "${labels[$i]} $(rate "$n" 2)"; i=$((i + 1)); done; echo "C $(rate "$C" 2)"; } | tee "$f"
}
all_nodes() { echo "$(ep_nodes) $C"; }

# ep_nodes: the EP nodes, rank-0 node first (EPN, default "A B")
ep_nodes() { echo "${EPN:-$A $B}"; }
# ep_run <outdir> <name> <v1|v2> <bench args...>: one EP bench over the EP nodes (EP16 on A + B by default; EPN="A B D E"
# for EP32); logs <outdir>/<name>_r<node rank>.log
ep_run() {
    local od=$1 name=$2 kind=$3; shift 3
    local port=$((39800 + RANDOM % 150)) args="$*" nodes=($(ep_nodes))
    local nn=${#nodes[@]} master=${CTL[${nodes[0]}]}
    mkdir -p "$od"
    local xe="EP_SCRIPT=${EP_SCRIPT:-} EP_LOCAL_BIG_N=${EP_LOCAL_BIG_N:-0} EP_TIMEOUT=${EP_TIMEOUT:-900}"
    [ -n "${EP_SCRIPT:-}" ] || xe="EP_LOCAL_BIG_N=${EP_LOCAL_BIG_N:-0} EP_TIMEOUT=${EP_TIMEOUT:-900}"
    local pids=() r
    for r in $(seq 1 $((nn - 1))); do
        pids+=("$(cxbg "${nodes[$r]}" "$od/${name}_r$r.step" "env $xe bash $W/mb/ep_half.sh $kind $r $nn $master $port $od/${name}_r$r.log $args")")
    done
    sleep 3
    cx "${nodes[0]}" "env $xe bash $W/mb/ep_half.sh $kind 0 $nn $master $port $od/${name}_r0.log $args" > "$od/${name}_r0.step" 2>&1
    local k=0 p alive
    while [ $k -lt 90 ]; do
        alive=0; for p in "${pids[@]}"; do kill -0 "$p" 2> /dev/null && alive=1; done
        [ $alive = 0 ] && break
        sleep 2; k=$((k + 1))
    done
    for p in "${pids[@]}"; do kill "$p" 2> /dev/null; done
    if [ "$kind" = v1 ]; then
        grep -h -E "^EPV1|rc=" "$od/${name}_r0.log" | tr '\n' ' ' | cut -c1-400
    else
        grep -h -E "^rc=|BENCH ABORTED" "$od/${name}_r0.log" | tr '\n' ' '
    fi
}
# ep_kill: whatever an aborted EP run left on the EP nodes
ep_kill() { for n in $(ep_nodes); do hx "$n" "pkill -u \$(id -u) -f '[e]p_v1_bench.py|[t]est_dispatch_combine_v2_internode'; true"; done; }

# iob_start <outdir> <IOTC> <initiator> <target> <benchmark args...>: mori's IO benchmark, one torchrun process per
# node (it spawns --num-initiator-dev / --num-target-dev engines). write: initiator -> target; read: target -> initiator.
IOB_PIDS=()
IOB_SEQ=$((RANDOM % 20))
iob_start() {
    local od=$1 iotc=$2 ini=$3 tgt=$4; shift 4
    local port=$((39900 + IOB_SEQ % 80)) args="$*"
    IOB_SEQ=$((IOB_SEQ + 1))
    mkdir -p "$od"
    IOB_PIDS+=("$(cxbg "$tgt" "$od/iobench_r1.step" "IOTC=$iotc bash $W/mb/iobench_half.sh 1 ${CTL[$ini]} ${CTL[$tgt]} $port $od/iobench_r1.log $args")")
    sleep 2
    IOB_PIDS+=("$(cxbg "$ini" "$od/iobench_r0.step" "IOTC=$iotc bash $W/mb/iobench_half.sh 0 ${CTL[$ini]} ${CTL[$ini]} $port $od/iobench_r0.log $args")")
}
iob_stop() {
    for n in "$@"; do hx "$n" "pkill -INT -u \$(id -u) -f '[t]ests/python/io/benchmark.py'; true"; done
    sleep 3
    for n in "$@"; do hx "$n" "pkill -9 -u \$(id -u) -f '[t]ests/python/io/benchmark.py'; true"; done
    for p in "${IOB_PIDS[@]}"; do kill "$p" 2> /dev/null; done
    IOB_PIDS=()
}

# inj_start <outdir> <IOTC> <op> <initiator> <passive> <io_inject send args...>: KV-shaped MoRI IO on all 8 rails
# (GPU i <-> GPU i). write: initiator -> passive; read: passive -> initiator (the initiator pulls). The buffers are GPU
# memory unless INJ_PMEM / INJ_IMEM give the passive / initiator side's io_inject memory options (e.g.
# "--mem host --numa remote": pinned host memory on the socket away from the NIC); INJ_PTASK prefixes the passive
# side's command (e.g. "taskset -c 64-127": a step's processes inherit the CPUs 0-63,128-191, the socket the vLLM
# workers are pinned to, and ConstrainCores=no lets them move). Every call gets its own port block.
INJ_PIDS=()
INJ_SEQ=$((RANDOM % 20))
inj_start() {
    local od=$1 iotc=$2 op=$3 ini=$4 pas=$5; shift 5
    local pb=$((32000 + (INJ_SEQ % 40) * 20)) sargs="$*"
    INJ_SEQ=$((INJ_SEQ + 1))
    mkdir -p "$od/inj"
    INJ_PIDS+=("$(cxbg "$pas" "$od/inj/recv.log" "MB_MORI=pr996 IOTC=$iotc ${INJ_PTASK:-} bash $W/mb/py.sh $W/mb/io_inject.py recv --devs 0-7 --port-base $pb ${INJ_PMEM:---mem gpu} --buf-gb ${INJ_BUF_GB:-1} --serve-forever --accept-timeout 3600 --out $od/inj/recv")")
    sleep 20
    INJ_PIDS+=("$(cxbg "$ini" "$od/inj/send.log" "MB_MORI=pr996 IOTC=$iotc bash $W/mb/py.sh $W/mb/io_inject.py send --op $op --peer ${CTL[$pas]} --devs 0-7 --port-base $pb ${INJ_IMEM:---mem gpu} --buf-gb ${INJ_BUF_GB:-1} $sargs --out $od/inj/send")")
}
inj_stop() {
    for n in "$@"; do hx "$n" "pkill -INT -u \$(id -u) -f '[i]o_inject.py'; true"; done
    sleep 3
    for n in "$@"; do hx "$n" "pkill -9 -u \$(id -u) -f '[i]o_inject.py'; true"; done
    for p in "${INJ_PIDS[@]}"; do kill "$p" 2> /dev/null; done
    INJ_PIDS=()
}

# fab_start <dir> <nodes...> / fab_stop <dir> <nodes...>: per-second NIC counters on the hosts
fab_start() { local d=$1; shift; for n in "$@"; do hx "$n" "mkdir -p $d; nohup bash $W/mb/fabmon.sh $d/fab_$n.tsv 1 > /dev/null 2>&1 & echo \$! > $d/fabmon_$n.pid"; done; }
fab_stop() { local d=$1; shift; for n in "$@"; do hx "$n" "kill \$(cat $d/fabmon_$n.pid) 2>/dev/null; rm -f $d/fabmon_$n.pid; true"; done; }
