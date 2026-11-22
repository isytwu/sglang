#!/usr/bin/env bash
# Runs on the LOGIN node:  JOB=<holder> DECODE="<node> <node> [<node> <node>]" C=<node> RUN=<name> [SPEC_K=7]
#                          [CONC=128] [ISL=1024] [OSL=8000] [WIN=180] [GAP=120] [CYCLES=2] [IOTC=104] [TARGETS=".."]
#                          [WINDOWS="base IWk IRk IWk10"] [SERVE_ENV="VAR=.."] bash e2e/e1.sh
# E1: K3.1 decode with wide EP in vLLM, IO from another node. The DECODE nodes serve one DP x TP8 decode, one DP rank
# per node (two nodes = EP16, four = EP32: DP4 x 128 sequences = the Moonshot case, global batch 512 at qlen 8):
# e2e/serve.sh decode_ep, MoRI all2all InterNodeV1LL, qlen SPEC_K + 1, max-num-seqs 128 per DP rank, under CONC
# streams per DP rank (ISL random tokens in, OSL out, ignore_eos: the batch stays full and new prefills are rare).
# C runs the load generator and is the IO peer. After WARM s the windows of e2e/windows.sh run; fab_<node>.tsv has the
# per-second NIC counters. A / B (if DECODE is unset) give the classic two-node decode.
# logs/env_<run>.txt, when it exists, is sourced first (e.g. export RAIL_ALIGN=1 for a run whose launcher entry has none).
set -uo pipefail
[ -s "$(dirname "$0")/../logs/env_${RUN:-}.txt" ] && source "$(dirname "$0")/../logs/env_${RUN:-}.txt"
source "$(dirname "$0")/../mb/lib.sh"
: "${JOB:?}" "${C:?}" "${RUN:?}"
DECODE=${DECODE:-$A $B}
DN=($DECODE)
SPEC_K=${SPEC_K:-7}; CONC=${CONC:-128}; ISL=${ISL:-1024}; OSL=${OSL:-8000}
WIN=${WIN:-180}; GAP=${GAP:-120}; CYCLES=${CYCLES:-2}; IOTC=${IOTC:-104}; WARM=${WARM:-300}
OUT=$W/results/e2e/$RUN
mkdir -p "$OUT/prof" "$OUT/load"
exec > >(tee -a "$OUT/driver.log") 2>&1
ctl_ips "${DN[@]}" "$C"
echo "=== E1 $RUN job $JOB decode ${DN[*]} (DP${#DN[@]} x TP8 = EP$((8 * ${#DN[@]}))) leader ${CTL[${DN[0]}]}, IO/load C=$C (${CTL[$C]}) SPEC_K=$SPEC_K CONC=$CONC/DP ISL=$ISL OSL=$OSL targets ${TARGETS:-all} RAIL_ALIGN=${RAIL_ALIGN:-0} $(date -u +%FT%TZ)"
source "$W/e2e/windows.sh"
PROF_URLS=$(for n in "${DN[@]}"; do printf '%s:8000 ' "$n"; done)

PIDS=()
cleanup() {
    inj_stop "${DN[@]}" "$C"
    for p in "${PIDS[@]}"; do kill "$p" 2> /dev/null; done
    sleep 10
    for n in "${DN[@]}" "$C"; do hx "$n" "pkill -u \$(id -u) -f '[v]llm serve|[l]oadgen.py|[V]LLM::'; true"; done
    fab_stop "$OUT" "${DN[@]}" "$C"
}
trap cleanup EXIT
wait_idle 180 "${DN[@]}" "$C" || exit 1
fab_start "$OUT" "${DN[@]}" "$C"

SX="SPEC_K=$SPEC_K DP_SIZE=${#DN[@]} PROFILE_DIR=$OUT/prof PROFILE_ITERS=${PROFILE_ITERS:-20} RAIL_ALIGN=${RAIL_ALIGN:-0} ${SERVE_ENV:-}"
# a server that exits before every rank is ready (the EP32 e1m_6867: one TP worker of DP0 gone with exit code 255 a
# second after it was spawned, nothing logged) gets the whole group restarted, SERVE_TRIES (2) times in all; the
# failed attempt's logs stay as serve_<node>.try<n>.log
for try in $(seq 1 "${SERVE_TRIES:-2}"); do
    ev serve_start "try $try"
    SPIDS=()
    for i in "${!DN[@]}"; do
        SPIDS+=("$(cxbg "${DN[$i]}" "$OUT/serve_${DN[$i]}.log" "env $SX bash $W/e2e/serve.sh decode_ep $i ${CTL[${DN[0]}]} 8000")")
    done
    PIDS+=("${SPIDS[@]}")
    t0=$(date +%s) up=0
    while :; do
        ok=0
        for n in "${DN[@]}"; do hx "$n" "curl -sf -m 5 http://127.0.0.1:8000/health > /dev/null" && ok=$((ok + 1)); done
        [ $ok = ${#DN[@]} ] && { up=1; break; }
        dead=""
        for p in "${SPIDS[@]}"; do kill -0 "$p" 2> /dev/null || dead=1; done
        [ -n "$dead" ] && { echo "a server exited (try $try):"; tail -n 30 "$OUT"/serve_*.log; break; }
        (( $(date +%s) - t0 > ${READY_S:-4200} )) && { echo "servers not ready after ${READY_S:-4200}s"; exit 1; }
        sleep 15
    done
    (( up )) && break
    (( try == ${SERVE_TRIES:-2} )) && exit 1
    for p in "${SPIDS[@]}"; do kill "$p" 2> /dev/null; done
    sleep 10
    for n in "${DN[@]}"; do hx "$n" "pkill -u \$(id -u) -f '[v]llm serve|[V]LLM::'; true"; mv "$OUT/serve_$n.log" "$OUT/serve_$n.try$try.log"; done
    wait_idle 300 "${DN[@]}" || exit 1
done
ev serve_ready "$(( $(date +%s) - t0 ))s"

[ -s "$W/logs/windows_$RUN.txt" ] && { WINDOWS=$(cat "$W/logs/windows_$RUN.txt"); WINDOWS_ADD=""; }
nw=$(( $(echo ${WINDOWS:-base IRb IWk IRk IWb IWk10} | wc -w) + $(echo ${WINDOWS_ADD-IWk IWk10} | wc -w) ))
total=$(( WARM + (CYCLES * nw + 1) * (WIN + GAP) + 600 ))
URLS=$(for n in "${DN[@]}"; do printf 'http://%s:8000,' "${CTL[$n]}"; done)
PIDS+=("$(cxbg "$C" "$OUT/loadgen.log" "MB_MORI=image bash $W/mb/py.sh $W/e2e/loadgen.py --urls ${URLS%,} --conc $CONC --isl $ISL --osl $OSL --duration $total --out $OUT/load")")
ev load_start "conc=$CONC/url isl=$ISL osl=$OSL"
sleep "$WARM"
tail -1 "$OUT/load/live.jsonl" 2> /dev/null
run_windows
ev done -
echo "E1_DONE $(date -u +%T)"
