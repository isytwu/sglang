#!/usr/bin/env bash
# Runs on the LOGIN node:  JOB=<holder> PREFILL="<node> ..." DECODE="<node> <node> ..." C=<node> RUN=<name>
#                          [KVTC=104] [CONC=128] [DUR=2700] [NAT_S=1200] [WINDOWS="IWk IRk ..."] [WIN=120] [GAP=60]
#                          [SERVE_ENV="VAR=.."] [PSERVE_ENV="VAR=.."] [CLIENT=loadgen ISL=8192 OSL=2048 WARM=300]
#                          bash e2e/pd.sh
# P/D under the AgentX workload. Every PREFILL node serves the colleague's prefill (TP8 DCP8, port 8100); the DECODE
# nodes serve E1's wide-EP decode (DP x TP8, one rank per node, MoRI all2all, port 8000). The KV moves with vLLM's
# MoRIIO connector in READ mode (the decode RDMA-reads what its prefix cache lacks), in traffic class KVTC (EP stays
# in 104). C runs e2e/pd_proxy.py (port 8000), the AgentX client (agentx-harness 1.0.13: AIPerf replaying the
# semianalysis Claude Code traces as InferenceX's agentx profile call does, CONC lanes, DUR s of measurement after its
# warmup) and the injected IO. Once AgentX measures, NAT_S s run with the natural KV traffic only ("nat" in
# events.tsv), then the IO windows of e2e/windows.sh go to the decode nodes while AgentX keeps going (none when
# WINDOWS is "none"). fab_<node>.tsv has every node's per-second NIC bytes, PFC, CNPs and ECN marks; load/ has the
# proxy's per-request token times and per-second series.
# logs/env_<run>.txt and logs/windows_<run>.txt, when they exist, are sourced first / replace WINDOWS.
set -uo pipefail
[ -s "$(dirname "$0")/../logs/env_${RUN:-}.txt" ] && source "$(dirname "$0")/../logs/env_${RUN:-}.txt"
source "$(dirname "$0")/../mb/lib.sh"
: "${JOB:?}" "${C:?}" "${RUN:?}" "${PREFILL:?}" "${DECODE:?}"
PN=($PREFILL) DN=($DECODE)
KVTC=${KVTC:-104}; CONC=${CONC:-128}; DUR=${DUR:-2700}; NAT_S=${NAT_S:-1200}
WIN=${WIN:-120}; GAP=${GAP:-60}; CYCLES=1; IOTC=${IOTC:-104}
AGX=$W/agentx
OUT=$W/results/e2e/$RUN
mkdir -p "$OUT/prof" "$OUT/load"
exec > >(tee -a "$OUT/driver.log") 2>&1
ctl_ips "${PN[@]}" "${DN[@]}" "$C"
# every node once: C can be a prefill node too (no IO node; the proxy and the client only use its CPUs)
ALL=($(printf '%s\n' "${PN[@]}" "${DN[@]}" "$C" | awk '!seen[$0]++'))
echo "=== PD $RUN job $JOB prefill ${PN[*]} decode ${DN[*]} (DP${#DN[@]} x TP8 = EP$((8 * ${#DN[@]}))) C=$C MoRIIO READ, KV in TC $KVTC, AgentX conc $CONC dur ${DUR}s nat ${NAT_S}s RAIL_ALIGN=${RAIL_ALIGN:-0} $(date -u +%FT%TZ)"
source "$W/e2e/windows.sh"
PROF_URLS=$(for n in "${DN[@]}"; do printf '%s:8000 ' "$n"; done)

PIDS=()
cleanup() {
    inj_stop "${DN[@]}" "$C"
    iob_stop "${DN[@]}" "$C"
    for p in "${PIDS[@]}"; do kill "$p" 2> /dev/null; done
    sleep 10
    for p in "${INJ_PRE_PIDS[@]}"; do kill "$p" 2> /dev/null; done
    for n in "${ALL[@]}"; do
        hx "$n" "pkill -u \$(id -u) -f '[v]llm serve|[V]LLM::|[p]d_proxy.py|[a]gentx profile|[a]iperf|[i]o_inject.py'; true"
    done
    fab_stop "$OUT" "${ALL[@]}"
}
trap cleanup EXIT
wait_idle 180 "${ALL[@]}" || exit 1
fab_start "$OUT" "${ALL[@]}"
# PRE_RECV=1: the IO windows' write receivers come up on the targets before the decode does (mb/lib.sh inj_prerecv)
if [ -n "${PRE_RECV:-}" ]; then
    inj_prerecv "$OUT/prerecv" "$IOTC" ${TARGETS:-$DECODE}
    sleep 40
    ev prerecv "${TARGETS:-$DECODE}"
fi

# kvcfg <role> <node> <http port> <port base>: the MoRIIO config of one server. The connector needs its HTTP port,
# a handshake port and a notify port (it adds the DP / TP rank offsets) and the address it advertises (the control
# network's, the one the proxy uses); srtctl filled these in for the E2 runs
kvcfg() {
    printf '{"kv_connector":"MoRIIOConnector","kv_role":"%s","kv_load_failure_policy":"fail","kv_connector_extra_config":{"backend":"rdma","read_mode":true,"defer_timeout":600,"host_ip":"%s","http_port":"%s","handshake_port":"%s","notify_port":"%s"}}' \
        "$1" "${CTL[$2]}" "$3" "$4" "$(( $4 + 1000 ))"
}
SX="SPEC_K=${SPEC_K:-7} DP_SIZE=${#DN[@]} PROFILE_DIR=$OUT/prof PROFILE_ITERS=${PROFILE_ITERS:-20} RAIL_ALIGN=${RAIL_ALIGN:-0} KV_TC=$KVTC ${SERVE_ENV:-}"
# as in e1.sh: a server that exits before all are ready gets the whole group restarted, SERVE_TRIES (2) times in all
for try in $(seq 1 "${SERVE_TRIES:-2}"); do
    ev serve_start "KV TC $KVTC, try $try"
    SPIDS=()
    for i in "${!DN[@]}"; do
        SPIDS+=("$(cxbg "${DN[$i]}" "$OUT/serve_${DN[$i]}.log" "env $SX KV_CFG='$(kvcfg kv_consumer "${DN[$i]}" 8000 26100)' bash $W/e2e/serve.sh decode_ep $i ${CTL[${DN[0]}]} 8000")")
    done
    for n in "${PN[@]}"; do
        # fcfs: shortest-prefill-first needs the SimpleCPUOffloadConnector in this build
        SPIDS+=("$(cxbg "$n" "$OUT/prefill_$n.log" "env RAIL_ALIGN=${RAIL_ALIGN:-0} KV_TC=$KVTC P_SCHED=fcfs ${PSERVE_ENV:-} KV_CFG='$(kvcfg kv_producer "$n" 8100 26000)' bash $W/e2e/serve.sh prefill 0 - 8100")")
    done
    PIDS+=("${SPIDS[@]}")
    t0=$(date +%s) up=0
    while :; do
        ok=0
        for n in "${DN[@]}"; do hx "$n" "curl -sf -m 5 http://127.0.0.1:8000/health > /dev/null" && ok=$((ok + 1)); done
        for n in "${PN[@]}"; do hx "$n" "curl -sf -m 5 http://127.0.0.1:8100/health > /dev/null" && ok=$((ok + 1)); done
        [ $ok = $(( ${#DN[@]} + ${#PN[@]} )) ] && { up=1; break; }
        dead=""
        for p in "${SPIDS[@]}"; do kill -0 "$p" 2> /dev/null || dead=1; done
        [ -n "$dead" ] && { echo "a server exited (try $try):"; tail -n 30 "$OUT"/serve_*.log "$OUT"/prefill_*.log; break; }
        (( $(date +%s) - t0 > ${READY_S:-4200} )) && { echo "servers not ready after ${READY_S:-4200}s"; exit 1; }
        sleep 15
    done
    (( up )) && break
    (( try == ${SERVE_TRIES:-2} )) && exit 1
    for p in "${SPIDS[@]}"; do kill "$p" 2> /dev/null; done
    sleep 10
    for n in "${DN[@]}" "${PN[@]}"; do hx "$n" "pkill -u \$(id -u) -f '[v]llm serve|[V]LLM::'; true"; done
    for f in "$OUT"/serve_*.log "$OUT"/prefill_*.log; do case $f in *.try*) ;; *) mv "$f" "${f%.log}.try$try.log" ;; esac; done
    # the pre-started receivers hold the decode GPUs too, so wait_idle (no KFD process) would never pass there
    if [ -n "${PRE_RECV:-}" ]; then sleep 60; wait_idle 300 "${PN[@]}" || exit 1; else wait_idle 300 "${DN[@]}" "${PN[@]}" || exit 1; fi
done
ev serve_ready "$(( $(date +%s) - t0 ))s"

PURL=$(for n in "${PN[@]}"; do printf -- '--prefill http://%s:8100 ' "${CTL[$n]}"; done)
DURL=$(for n in "${DN[@]}"; do printf -- '--decode http://%s:8000 ' "${CTL[$n]}"; done)
PIDS+=("$(cxbg "$C" "$OUT/proxy.log" "$AGX/venv/bin/python $W/e2e/pd_proxy.py $PURL $DURL --port 8000 --out $OUT/load")")
for i in $(seq 30); do hx "$C" "curl -sf -m 5 http://127.0.0.1:8000/health > /dev/null" && break; sleep 2; done
# one request through both legs before the client starts: the first READ sets up the MoRIIO sessions
hx "$C" "curl -s -m 600 http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{\"model\":\"moonshotai/Kimi-K3.1-Pro\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hello in one word.\"}],\"max_tokens\":8}'" | head -c 600
echo
grep -q '"done": true' "$OUT/load/requests.jsonl" 2> /dev/null || sleep 12
grep -q '"done": true' "$OUT/load/requests.jsonl" || { echo "the probe request failed:"; tail -n 3 "$OUT/load/requests.jsonl" "$OUT/proxy.log"; exit 1; }
ev pd_ok -

if [ "${CLIENT:-agentx}" = loadgen ]; then
    # the synthetic client instead (CLIENT=loadgen): CONC streams in all through the proxy, ISL random tokens in,
    # OSL (+-50 %) out, to hold the decode batch near max-num-seqs (E1's load) while the KV still moves P -> D
    ISL=${ISL:-8192}; OSL=${OSL:-2048}
    nw=$( [ "${WINDOWS:-}" = none ] && echo 0 || echo ${WINDOWS:-x x x x x x} | wc -w)
    total=$(( ${WARM:-300} + NAT_S + (nw + 1) * (WIN + GAP) + 600 ))
    PIDS+=("$(cxbg "$C" "$OUT/loadgen.log" "MB_MORI=image bash $W/mb/py.sh $W/e2e/loadgen.py --urls http://127.0.0.1:8000 --conc $CONC --isl $ISL --osl $OSL --duration $total --out $OUT/client")")
    ev load_start "conc=$CONC isl=$ISL osl=$OSL"
    sleep "${WARM:-300}"
    ev measure_start "${WARM:-300}s of ramp"
else
    METRICS=$(for n in "${DN[@]}"; do printf 'http://%s:8000/metrics ' "${CTL[$n]}"; done; for n in "${PN[@]}"; do printf 'http://%s:8100/metrics ' "${CTL[$n]}"; done)
    AENV="HF_HOME=$AGX/hf_home HF_HUB_CACHE=$AGX/hf_home/hub AIPERF_HTTP_TCP_USER_TIMEOUT=900000 AIPERF_USE_DYNAMO_CONV_AWARE_ROUTING=0 AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID=false AIPERF_EXPERIMENTAL_FAST=0"
    UNSAFE=""; (( DUR < 900 )) && UNSAFE=--unsafe-override
    PIDS+=("$(cxbg "$C" "$OUT/agentx.log" "ulimit -n \$(ulimit -Hn) 2>/dev/null; env $AENV $AGX/venv/bin/agentx profile --url http://127.0.0.1:8000 --model moonshotai/Kimi-K3.1-Pro --tokenizer /perf_apps/data/models/Kimi-K3.1-Pro --concurrency $CONC --benchmark-duration $DUR --tokenizer-trust-remote-code --server-metrics $METRICS --output-artifact-dir $OUT/agentx --public-dataset semianalysis_cc_traces_weka_062126 --warmup-requests-per-lane ${WARMUP_PER_LANE:-1} --agentic-warmup-grace-period ${WARM_GRACE:-3600} --benchmark-grace-period 300 $UNSAFE")")
    AGPID=${PIDS[-1]}
    ev agentx_start "conc=$CONC dur=$DUR"
    # the measurement starts when AIPerf's Profiling phase does
    t0=$(date +%s)
    until grep -qE "Phase profiling .*started|Credit phase start: profiling" "$OUT/agentx/logs/aiperf.log" 2> /dev/null; do
        kill -0 "$AGPID" 2> /dev/null || { echo "AgentX exited before measuring:"; tail -30 "$OUT/agentx.log"; exit 1; }
        (( $(date +%s) - t0 > ${WARM_MAX:-3600} )) && { echo "no Profiling phase after ${WARM_MAX:-3600}s; going on"; break; }
        sleep 10
    done
    ev measure_start "$(( $(date +%s) - t0 ))s of warmup"
fi
ev nat start
sleep "$NAT_S"
ev nat end
tail -1 "$OUT/load/live.jsonl" 2> /dev/null
if [ "${WINDOWS:-}" != none ]; then
    WINDOWS_ADD="${WINDOWS_ADD-}" run_windows
fi
ev windows_done -
if [ "${CLIENT:-agentx}" != loadgen ]; then
    # AgentX ends DUR s after its measurement began, plus its grace period
    while kill -0 "$AGPID" 2> /dev/null; do sleep 15; done
    ev agentx_done "$(tail -c 300 "$OUT/agentx.log" | tr '\n' ' ')"
fi
echo "PD_DONE $(date -u +%T)"
