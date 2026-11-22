#!/usr/bin/env bash
# Runs on the LOGIN node, unattended:  nohup bash tools/launch.sh <holder job> [phase ...] > /dev/null 2>&1 &
# Waits for the holder allocation (tools/holder.sbatch), gives the nodes their roles in node-list order (every node but
# the last runs EP / the decode: A = EP rank 0 / DP rank 0, B = the EP node that receives the microbenchmarks' IO; the
# last node C is the IO peer and the load generator; 3 nodes = EP16, 5 nodes = EP32), runs the phases that still fit
# in the allocation, cleans the nodes and cancels the holder. Log: logs/launch_<job>.log.
# Phases (default: setup build warm c1 e1 c3 c2):
#   setup   create the pyxis container on every node, record driver / firmware / GPU / PCIe state
#   build   ROCm/mori PR #996 (3a869d5f) in place under src/ (on C; skipped when already built)
#   warm    one EP run per kind to JIT-compile the kernels (not recorded)
#   c0      vLLM's kernel with the e2e decode's buffers (max 2048 tokens / rank) at 64 / 128 tokens, no IO, back to back
#           and at the duty cycle: how much of the e2e dispatch / combine time is the kernel
#   c1      microbenchmark, EP back to back: the DO scenarios x v2k48 v2k128 v1k128 (+ v2def)
#   c2      twice the EP load: v2k256 v1k256, base / IW8 / IWk / IRk
#   c3      the decode's duty cycle (EP_LOCAL_BIG_N=20): v2k128 v1k128, base / IW8 / IWk / IRk
#   c4      more IO forms: TC 0, host memory near / far, 2:1 incast, fan-out to every EP node, more QPs, 64 MB bursts
#   c5      heavy EP: 384 and 512 tokens / rank (v1, v2), base / IW8 / IWk / IRk
#   c6      vLLM's kernel with 4 QPs per PE and InterNodeV1 (mori_high_throughput), base / IW8 / IWk / IRk
#   c3h     the duty cycle at 384 tokens / rank
#   c7      the duty cycle (vLLM's kernel, 128 and 384 tokens / rank) against KV-shaped writes at 5 / 10 / 20 GB/s per NIC
#           and back to back
#   c9      READ: B's KV-shaped reads at 5 / 10 / 20 GB/s per NIC and back to back, IR8 with fewer READs in flight
#           (IR8s), one rail (IR1), and the EP node as the READ responder (OR8, ORk); c9d the same at the duty cycle
#   c9b     B's READs in flight: IR8 at 64 / 256 / 1024 / 2048 / 4096 / 8192 transfers of 32 KB per batch
#   apply   move mb/*.new, e2e/*.new over the scripts (edits made while a phase ran the old version)
#   c8      base / IW8 / IWk / IRk with RAIL_ALIGN=1 (mb/rail.sh: every cross-node GPU pair on one backend rail; the
#           default pairs cross rails between the n08-25 / n08-33 kind of node and the others)
#   e1ra    e1 with RAIL_ALIGN=1 (base / IWk / IRk)
#   rot     c1's receive-side scenarios with the EP16 roles rotated (A=C, B=A, C=B; 3-node jobs)
#   e1      vLLM K3.1 decode, DP x TP8 on every node but C (EP16 / EP32), KV-shaped IO windows from C (e2e/e1.sh);
#           windows of E1_WIN (120) s every E1_WIN + E1_GAP (60) s, E1_CYCLES (1) rounds of base IWk IRk IWk10
#   e1s     e1 with HSA_NO_SCRATCH_RECLAIM=1 (the colleague's recipe sets 0; the microbenchmarks run with 1). In job 6857
#           the decode hung at start-up with it (after the DSA multistream setup); do not schedule it again
#   e1m     the Moonshot case (5-node jobs): global batch 512 at qlen 8 over EP32, windows base IWk IWk1 IRb IRb1 IRk IWk10 IWb;
#           e1m2 a second such run
#   pd32    the Moonshot case as P/D (6-node jobs): 1 prefill, the EP32 decode, C; loadgen 512 streams (2k in, 2k out),
#           15 min natural, then IWk IWk1g IWkR64 IRk IWkS IWkT0;  pd32t0  the same with the KV in TC 0, natural only
#   pd32a   EP32 P/D under AgentX (concurrency 256), natural traffic only: 2 prefill + 4 decode, the proxy and the
#           client on the first prefill node
#   e1x     e1 at twice the EP load (256 sequences per DP rank), windows base IRb IRb1 IWb IRk
#   e1t1    the same with the IO on one decode node only (IWk1 / IRk1 / IWhR windows)
#   e2w/e2r     vLLM K3.1 PD through srtctl, EP16 decode (3-node jobs), MoRIIO WRITE / READ, IO windows (e2e/e2.sh)
#   e2w32/e2r32 the same with the EP32 decode (5-node jobs)
#   pd      P/D under the AgentX workload (e2e/pd.sh, 4+ nodes): every node before the last three a prefill, the two
#           before C the EP16 decode, MoRIIO READ with the KV in TC 104 (as EP); 20 min of natural KV traffic, then
#           IO windows into the decode nodes (logs/windows_pd_<job>.txt, default IWk IRk IWk10 IRk10 IWkS IWkT0)
#   pdt0    the same with the KV in TC 0, natural traffic only;  pdn  pd's natural traffic only (a repeat)
#   pdc     AgentX at concurrency 256 (KV in TC 104): 15 min natural, then IWk IRk IWkT0
#   pdh     the real P/D KV path under the synthetic client (loadgen through the proxy, 256 streams, 8k in, 2k out:
#           the decode batch near max-num-seqs, E1's EP load), 15 min natural, then pd's IO windows; pdht0 the same
#           with the KV in TC 0, natural traffic only;  pdhf / pdhft0  pdh / pdht0's natural traffic with the turnover
#           of logs/env_<run>.txt (e.g. ISL 1024, OSL 512: four times the new requests, and their KV, per second)
# The phases still to run live in logs/phases_<job>.txt (one line, space separated), read before every phase: edit it
# to change what a waiting or running job does next. The arguments seed it when it does not exist yet.
set -uo pipefail
W=/apps/yutongwu/store-vllm/k31-vllm-ep-io-smci355-20261119
JOB=${1:?holder job id}; shift
PF=$W/logs/phases_$JOB.txt
export PATH=/opt/slurm/bin:$PATH
mkdir -p "$W/logs"
exec > >(tee -a "$W/logs/launch_$JOB.log") 2>&1
# one launcher per job: a second one would run phases next to the first, and cancel the job when its list is empty
LOCK=$W/logs/launch_$JOB.pid
if [ -s "$LOCK" ] && kill -0 "$(cat "$LOCK")" 2> /dev/null; then
    echo "launcher $(cat "$LOCK") already drives job $JOB; exiting ($(date -u +%FT%TZ))"; exit 1
fi
echo $$ > "$LOCK"
[ -s "$PF" ] || echo "${*:-setup build warm c1 e1 c3 c2}" > "$PF"
echo "launch $JOB phases: $(cat "$PF") ($(date -u +%FT%TZ))"

while :; do
    st=$(squeue -h -j "$JOB" -o %T 2> /dev/null)
    [ "$st" = RUNNING ] && break
    [ -z "$st" ] && { echo "job $JOB is gone ($(date -u +%FT%TZ))"; exit 1; }
    sleep 30
done
NODES=($(scontrol show hostnames "$(squeue -h -j "$JOB" -o %N)" | sed 's/^smci355-ccs-aus-//'))
NN=${#NODES[@]}
export JOB C=${NODES[$((NN - 1))]} EPN="${NODES[*]:0:$((NN - 1))}"
export A=${NODES[0]} B=${NODES[1]}
source "$W/mb/lib.sh"
echo "job $JOB running on ${NODES[*]}: EP / decode nodes $EPN (A=$A B=$B), IO / load node C=$C ($(date -u +%FT%TZ))"

left_s() {
    awk -v t="$(squeue -h -j "$JOB" -o %L)" 'BEGIN { d = 0; if (index(t, "-")) { split(t, a, "-"); d = a[1]; t = a[2] }
        n = split(t, p, ":"); s = 0; for (i = 1; i <= n; i++) s = s * 60 + p[i]; print d * 86400 + s }'
}
# sweep <nodes...>: our processes that a phase left behind (never other users')
PAT='[t]orchrun|[e]p_v1_bench|[t]est_dispatch_combine_v2|[i]o_inject.py|[t]ests/python/io/benchmark.py|[v]llm serve|[V]LLM::|[l]oadgen.py|mb/fabmon.sh'
sweep() {
    for n in "$@"; do hx "$n" "pkill -u \$(id -u) -f '$PAT'; sleep 3; pkill -9 -u \$(id -u) -f '$PAT'; true"; done
    sleep 5
    for n in "$@"; do echo "  $n: kfd procs $(gpu_busy "$n")"; done
}
# phase <name> <min seconds left> <timeout s> <command...>
phase() {
    local name=$1 need=$2 to=$3; shift 3
    local left; left=$(left_s)
    if (( left < need )); then echo "skip $name: ${left}s left (< ${need}s)"; return 1; fi
    (( to > left - 240 )) && to=$((left - 240))
    echo "=== phase $name (${left}s left, timeout ${to}s) $(date -u +%FT%TZ)"
    timeout --signal=TERM --kill-after=60 "$to" "$@"
    local rc=$?
    echo "=== phase $name rc=$rc $(date -u +%FT%TZ)"
    sweep "${NODES[@]}" > /dev/null
    return $rc
}
# snap <script> <tag>: run a phase from a copy of its script (next to it, so its relative sources still resolve):
# bash reads a script as it runs, and editing the original under a running phase would corrupt it
snap() { local c; c="$(dirname "$1")/.snap_${JOB}_$2_$(basename "$1")"; cp "$1" "$c"; echo "$c"; }
mbsum() { python3 "$W/mb/analyze.py" "$W/results/mb/$1" > "$W/results/mb/$1/summary.md" 2>&1; }
e2esum() { python3 "$W/e2e/e1_analyze.py" "$W/results/e2e/$1" > "$W/results/e2e/$1/summary.md" 2>&1; }
MB4="base1 IW8 IR8 IWk IRk base2"

# next_phase: pop the first phase of the phase file (atomically rewritten)
next_phase() {
    local rest
    read -r ph rest < "$PF" || return 1
    [ -n "${ph:-}" ] || return 1
    echo "$rest" > "$PF.tmp" && mv "$PF.tmp" "$PF"
}
RAN=0
while next_phase; do
    RAN=1
    echo "--- next: $ph (then: $(cat "$PF"))"
    case $ph in
    after:*)
        # after:<pid>:<campaign>: a phase an earlier launcher left running; wait for it, then summarise it
        IFS=: read -r _ apid acamp <<< "$ph"
        while kill -0 "$apid" 2> /dev/null; do sleep 20; done
        echo "  $apid ($acamp) finished $(date -u +%FT%TZ)"
        if [ -d "$W/results/e2e/$acamp" ]; then e2esum "$acamp"; elif [ -n "$acamp" ]; then mbsum "$acamp"; fi
        sweep "${NODES[@]}" > /dev/null
        ;;
    setup)
        # a bare wait would also wait for the tee of the exec above (bash 5.1)
        sp=()
        for n in "${NODES[@]}"; do
            (cx "$n" "echo container \$(hostname -s) ok; /opt/runtime-venv/bin/python -c 'import torch; print(torch.__version__, torch.cuda.device_count())'; /opt/runtime-venv/bin/python $W/mb/topo_check.py; source $W/mb/rail.sh; echo rail_perm \$(rail_perm)" > "$W/logs/setup_${JOB}_$n.log" 2>&1) &
            sp+=($!)
        done
        wait "${sp[@]}"
        cat "$W"/logs/setup_${JOB}_*.log
        for n in "${NODES[@]}"; do
            echo "$n: amdgpu $(hx "$n" 'cat /sys/module/amdgpu/version') fw $(hx "$n" 'ethtool -i benic1p1 | awk "/firmware-version/{print \$2}"') kfd $(gpu_busy "$n") qos $(hx "$n" 'sudo -n nicctl show qos 2>/dev/null | grep -v -E "^NIC|^Port" | md5sum | cut -c1-8')"
        done
        sweep "${NODES[@]}"
        ;;
    build)
        phase build 1800 2700 bash -c "source $W/mb/lib.sh; cx $C 'bash $W/mb/build.sh'" ||
            echo "BUILD FAILED: the EPv2 / IO parts will fail"
        ;;
    warm)
        export EP_TIMEOUT=1500
        epr=$((896 / (8 * (NN - 1))))
        phase warm 1800 1800 bash -c "source $W/mb/lib.sh; ctl_ips ${NODES[*]}
            echo \"warm v1: \$(ep_run $W/results/mb/_warm_$JOB warm_v1 v1 --kernel v1_ll --experts-per-rank $epr --tokens 128 --max-tokens 1024 --rounds 200 --warmup 20 --out $W/results/mb/_warm_$JOB/warm_v1_json)\"
            echo \"warm v2: \$(ep_run $W/results/mb/_warm_$JOB warm_v2 v2 --kernel-type v2_ll --max-tokens 128 --hidden-dim 3584 --topk 16 --experts-per-rank $epr --dtype bf16 --combine-dtype bf16 --num-qp 8 --scale-dim 0 --rounds 200 --warmup 20)\""
        unset EP_TIMEOUT
        ;;
    apply)
        # replace scripts with their .new versions (prepared while a phase was running the old one)
        for f in "$W"/mb/*.new "$W"/e2e/*.new; do [ -f "$f" ] && mv -v "$f" "${f%.new}"; done ;;
    c9)
        CAMP=c9_$JOB SHAPES="v1k128 v2k128" ONLY="base1 IR8 IR8m IR8s IR8L IR1 IRk5 IRk10 IRk20 IRk OR8 ORk base2" \
            phase c9 1800 3300 bash "$(snap "$W/mb/campaign.sh" "$ph")"; mbsum "c9_$JOB" ;;
    c9b)
        CAMP=c9b_$JOB SHAPES="v1k128 v2k128" ONLY="base1 IR8s IR8b256 IR8b1024 IR8b2048 IR8b4096 IR8 base2" \
            phase c9b 1500 2700 bash "$(snap "$W/mb/campaign.sh" "$ph")"; mbsum "c9b_$JOB" ;;
    c9d)
        CAMP=c9d_$JOB SHAPES="v1k128 v2k128" ONLY="base1 IRk5 IRk10 IRk20 IRk IR8 base2" EP_LOCAL_BIG_N=20 \
            EP_SCRIPT=test_dispatch_combine_v2_internode_local.py phase c9d 1800 3000 bash "$(snap "$W/mb/campaign.sh" "$ph")"; mbsum "c9d_$JOB" ;;
    c0)
        CAMP=c0_$JOB SHAPES="v1k128 v1m2k64 v1m2k128" ONLY="base1" phase c0 900 900 bash "$(snap "$W/mb/campaign.sh" "$ph")"
        CAMP=c0d_$JOB SHAPES="v1k128 v1m2k64 v1m2k128" ONLY="base1" EP_LOCAL_BIG_N=20 phase c0d 900 900 bash "$(snap "$W/mb/campaign.sh" "$ph")"
        mbsum "c0_$JOB"; mbsum "c0d_$JOB" ;;
    c1)
        CAMP=c1_$JOB phase c1 3000 4200 bash "$(snap "$W/mb/campaign.sh" "$ph")"; mbsum "c1_$JOB" ;;
    c2)
        CAMP=c2_$JOB SHAPES="v2k256 v1k256" ONLY="$MB4" phase c2 1200 1800 bash "$(snap "$W/mb/campaign.sh" "$ph")"; mbsum "c2_$JOB" ;;
    c3)
        CAMP=c3_$JOB SHAPES="v2k128 v1k128" ONLY="$MB4" EP_LOCAL_BIG_N=20 \
            EP_SCRIPT=test_dispatch_combine_v2_internode_local.py phase c3 1200 1800 bash "$(snap "$W/mb/campaign.sh" "$ph")"; mbsum "c3_$JOB" ;;
    c3h)
        CAMP=c3h_$JOB SHAPES="v2k384 v1k384" ONLY="$MB4" EP_LOCAL_BIG_N=20 \
            EP_SCRIPT=test_dispatch_combine_v2_internode_local.py phase c3h 1200 1800 bash "$(snap "$W/mb/campaign.sh" "$ph")"; mbsum "c3h_$JOB" ;;
    c7)
        CAMP=c7_$JOB SHAPES="v1k128 v1k384" ONLY="base1 IWk5 IWk10 IWk20 IWk base2" EP_LOCAL_BIG_N=20 \
            phase c7 1200 2100 bash "$(snap "$W/mb/campaign.sh" "$ph")"; mbsum "c7_$JOB" ;;
    c8)
        CAMP=c8_$JOB RAIL_ALIGN=1 SHAPES="v2k128 v1k128 v1k384" ONLY="$MB4" phase c8 1500 2400 bash "$(snap "$W/mb/campaign.sh" "$ph")"; mbsum "c8_$JOB" ;;
    e1ra)
        WIN=${E1_WIN:-120} GAP=${E1_GAP:-60} CYCLES=${E1_CYCLES:-1} RUN=e1ra_$JOB DECODE="$EPN" RAIL_ALIGN=1 \
            phase e1ra 2400 $(( $(left_s) - 300 )) bash "$(snap "$W/e2e/e1.sh" "$ph")"; e2esum "e1ra_$JOB" ;;
    c4)
        CAMP=c4_$JOB SHAPES="v2k128 v1k128" ONLY="base1 IWkT0 IR8T0 IWhL IWhR IRhR I2Wk IWkA IWkQ IWk64 base2" \
            phase c4 1800 3000 bash "$(snap "$W/mb/campaign.sh" "$ph")"; mbsum "c4_$JOB" ;;
    c5)
        CAMP=c5_$JOB SHAPES="v1k384 v1k512 v2k384 v2k512" ONLY="$MB4" phase c5 1500 2400 bash "$(snap "$W/mb/campaign.sh" "$ph")"; mbsum "c5_$JOB" ;;
    c6)
        CAMP=c6_$JOB SHAPES="v1q4k128 v1htk128" ONLY="$MB4" phase c6 1200 1800 bash "$(snap "$W/mb/campaign.sh" "$ph")"; mbsum "c6_$JOB" ;;
    rot)
        [ "$NN" = 3 ] || { echo "skip rot: EP16 roles only"; continue; }
        CAMP=rot_$JOB A=$C B=$A C=$B EPN="$C $A" ONLY="$MB4" SHAPES="v2k128 v1k128" phase rot 1200 1800 bash "$(snap "$W/mb/campaign.sh" "$ph")"
        mbsum "rot_$JOB" ;;
    e1)
        WIN=${E1_WIN:-120} GAP=${E1_GAP:-60} CYCLES=${E1_CYCLES:-1} RUN=e1_$JOB DECODE="$EPN" phase e1 2400 $(( $(left_s) - 300 )) bash "$(snap "$W/e2e/e1.sh" "$ph")"; e2esum "e1_$JOB" ;;
    e1s)
        WIN=${E1_WIN:-120} GAP=${E1_GAP:-60} CYCLES=${E1_CYCLES:-1} RUN=e1s_$JOB DECODE="$EPN" SERVE_ENV="HSA_NO_SCRATCH_RECLAIM=1" \
            WINDOWS="base IRb IRb1 IWb IRk" \
            phase e1s 2400 $(( $(left_s) - 300 )) bash "$(snap "$W/e2e/e1.sh" "$ph")"; e2esum "e1s_$JOB" ;;
    e1m|e1m2)
        # the Moonshot case on a 5-node job: DP4 x 128 sequences = global batch 512 at qlen 8 over EP32; IO from C into
        # every decode node (C's NICs split four ways) and into one of them (all of C's NICs on it). e1m2: a second run
        # (its own logs/windows_e1m2_<job>.txt / env file)
        WIN=${E1_WIN:-120} GAP=${E1_GAP:-60} CYCLES=${E1_CYCLES:-1} RUN=${ph}_$JOB DECODE="$EPN" \
            WINDOWS="base IWk IWk1 IRb IRb1 IRk IWk10 IWb" WINDOWS_ADD="" \
            phase "$ph" 2400 $(( $(left_s) - 300 )) bash "$(snap "$W/e2e/e1.sh" "$ph")"; e2esum "${ph}_$JOB" ;;
    e1x)
        # twice the EP load: 256 sequences per DP rank (256 tokens per EP rank per dispatch at qlen 8)
        WIN=${E1_WIN:-120} GAP=${E1_GAP:-60} CYCLES=${E1_CYCLES:-1} RUN=e1x_$JOB DECODE="$EPN" CONC=256 \
            SERVE_ENV="MAX_SEQS=256 MAX_BATCHED=3072" WINDOWS="base IRb IRb1 IWb IRk" \
            phase e1x 2400 $(( $(left_s) - 300 )) bash "$(snap "$W/e2e/e1.sh" "$ph")"; e2esum "e1x_$JOB" ;;
    e1t1)
        WIN=${E1_WIN:-120} GAP=${E1_GAP:-60} CYCLES=${E1_CYCLES:-1} RUN=e1t1_$JOB DECODE="$EPN" TARGETS="$B" WINDOWS="base IWk1 IRk1 IWhR" \
            phase e1t1 2400 $(( $(left_s) - 300 )) bash "$(snap "$W/e2e/e1.sh" "$ph")"; e2esum "e1t1_$JOB" ;;
    pd32a)
        # EP32 P/D under AgentX, natural traffic only: two prefill nodes, the EP32 decode on the next four, no IO node
        # (the proxy and the AgentX client run on the first prefill node, on its CPUs)
        (( NN >= 6 )) || { echo "skip $ph: needs 6 nodes"; continue; }
        PREFILL="${NODES[0]} ${NODES[1]}" DECODE="${NODES[*]:2:4}" C="${NODES[0]}" RUN=${ph}_$JOB KVTC=104 DUR=1500 \
            NAT_S=1500 WINDOWS=none WINDOWS_ADD="" RAIL_ALIGN=1 CONC=${PD32A_CONC:-256} CLIENT=agentx \
            phase "$ph" 3000 $(( $(left_s) - 300 )) bash "$(snap "$W/e2e/pd.sh" "$ph")"
        e2esum "${ph}_$JOB" ;;
    pd32|pd32t0)
        # the Moonshot case as P/D on a 6-node job: one prefill, the EP32 decode (DP4 x TP8) on the next four, C last;
        # the synthetic client holds 128 sequences per DP rank (global batch 512, qlen 8) while the KV moves P -> D
        (( NN >= 6 )) || { echo "skip $ph: needs 6 nodes"; continue; }
        kvtc=104 wins="IWk IWk1g IWkR64 IRk IWkS IWkT0"
        [ "$ph" = pd32t0 ] && kvtc=0 wins=none
        PREFILL="${NODES[0]}" DECODE="${NODES[*]:1:4}" RUN=${ph}_$JOB KVTC=$kvtc NAT_S=900 WINDOWS="$wins" WINDOWS_ADD="" \
            RAIL_ALIGN=1 WIN=${E1_WIN:-120} GAP=${E1_GAP:-60} CONC=512 CLIENT=loadgen ISL=${PD32_ISL:-2048} OSL=${PD32_OSL:-2048} \
            phase "$ph" 2400 $(( $(left_s) - 300 )) bash "$(snap "$W/e2e/pd.sh" "$ph")"
        e2esum "${ph}_$JOB" ;;
    pd|pdt0|pdn|pdc|pdh|pdht0|pdhf|pdhft0)
        (( NN >= 4 )) || { echo "skip $ph: needs 4+ nodes"; continue; }
        kvtc=104 dur=2700 nat=1200 wins="IWk IRk IWk10 IRk10 IWkS IWkT0" conc=128 client=agentx
        [[ $ph == *t0 ]] && kvtc=0
        [ "$ph" != pd ] && dur=1500 nat=1500 wins=none
        [ "$ph" = pdc ] && conc=256 dur=2100 nat=900 wins="IWk IRk IWkT0"
        [[ $ph == pdh* ]] && client=loadgen conc=256 nat=900
        [ "$ph" = pdh ] && wins="IWk IRk IWk10 IRk10 IWkS IWkT0"
        PREFILL="${NODES[*]:0:$((NN - 3))}" DECODE="${NODES[*]:$((NN - 3)):2}" RUN=${ph}_$JOB KVTC=$kvtc DUR=$dur \
            NAT_S=$nat WINDOWS="$wins" WINDOWS_ADD="" RAIL_ALIGN=1 WIN=${E1_WIN:-120} GAP=${E1_GAP:-60} CONC=$conc \
            CLIENT=$client phase "$ph" 2400 $(( $(left_s) - 300 )) bash "$(snap "$W/e2e/pd.sh" "$ph")"
        e2esum "${ph}_$JOB" ;;
    e2w|e2r|e2w32|e2r32)
        m=write; [[ $ph == e2r* ]] && m=read
        rec=$W/configs/e2_pd_$m.yaml; [[ $ph == *32 ]] && rec=$W/configs/e2_pd32_$m.yaml
        conc=$((128 * (NN - 1)))
        RUN=${ph}_$JOB MODE=$m RECIPE=$rec CONC=$conc phase "$ph" 4200 $(( $(left_s) - 300 )) bash "$(snap "$W/e2e/e2.sh" "$ph")"
        e2esum "${ph}_$JOB" ;;
    *) echo "unknown phase $ph" ;;
    esac
done
rm -f "$LOCK"
# a launcher that found no phase left ran none: whoever popped them is still running them
[ "$RAN" = 1 ] || { echo "no phases left for job $JOB; leaving it alone ($(date -u +%FT%TZ))"; exit 0; }
sweep "${NODES[@]}"
echo "done; cancelling $JOB ($(date -u +%FT%TZ))"
scancel "$JOB"
