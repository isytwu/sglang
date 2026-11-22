#!/usr/bin/env bash
# Runs on the LOGIN node:  JOB=<holder> MODE=write|read RUN=<name> [RECIPE=configs/e2_pd32_write.yaml] [CONC=256]
#                          [ISL=1024] [OSL=8000] [WIN=180] [GAP=120] [CYCLES=1] [IOTC=104] [TARGETS=".."] bash e2e/e2.sh
# E2: the K3.1 PD pair with real MoRIIO KV transfers. srtctl (e2e/run_srtctl.sh, RECIPE, default configs/e2_pd_<MODE>.yaml)
# brings up the prefill (TP8 DCP8, first node, with the vLLM router) and the EP decode (DP x TP8 on the other nodes:
# EP16 on two, EP32 on four with the e2_pd32_* recipes) and
# holds them (benchmark: manual). This driver then puts CONC streams through the router (from the prefill node's
# container) and runs the windows of e2e/windows.sh with the prefill node as the IO peer: on top of the KV the
# decode receives anyway, KV-shaped writes from the prefill node into both decode nodes, or reads by them.
set -uo pipefail
source "$(dirname "$0")/../mb/lib.sh"
: "${JOB:?}" "${MODE:?}" "${RUN:?}"
CONC=${CONC:-256}; ISL=${ISL:-1024}; OSL=${OSL:-8000}
WIN=${WIN:-180}; GAP=${GAP:-120}; CYCLES=${CYCLES:-1}; IOTC=${IOTC:-104}; WARM=${WARM:-300}
OUT=$W/results/e2e/$RUN
mkdir -p "$OUT/load"
exec > >(tee -a "$OUT/driver.log") 2>&1
NODES=($(scontrol show hostnames "$(squeue -h -j "$JOB" -o %N)" | sed 's/^smci355-ccs-aus-//'))
head=${NODES[0]}
echo "=== E2 $RUN job $JOB mode $MODE nodes ${NODES[*]} CONC=$CONC ISL=$ISL OSL=$OSL $(date -u +%FT%TZ)"
source "$W/e2e/windows.sh"

SPID=
cleanup() {
    inj_stop "${NODES[@]}"
    for n in "${NODES[@]}"; do hx "$n" "pkill -u \$(id -u) -f '[l]oadgen.py'; true"; done
    if [ -n "$SPID" ] && kill -0 "$SPID" 2> /dev/null; then
        kill -TERM "$SPID"
        for i in $(seq 1 60); do kill -0 "$SPID" 2> /dev/null || break; sleep 5; done
        kill -KILL "$SPID" 2> /dev/null
    fi
    # srtctl's teardown can leave its router step behind (it then holds port 8000 on the head node)
    for s in $(squeue -h -s -j "$JOB" -o '%i %j' | awk '$2 !~ /^(batch|extern|yt-epio|yt-e2-srtctl)$/ {print $1}'); do
        echo "cancelling leftover step $s"; scancel "$s"
    done
    for n in "${NODES[@]}"; do hx "$n" "pkill -u \$(id -u) -f '[v]llm-router|[v]llm serve|[V]LLM::'; true"; done
    fab_stop "$OUT" "${NODES[@]}"
}
trap cleanup EXIT
fab_start "$OUT" "${NODES[@]}"

ev serve_start -
srun --jobid="$JOB" --overlap -N1 -n1 -c 128 --mem=0 -w "$(host "$head")" --job-name=yt-e2-srtctl \
    bash "$W/e2e/run_srtctl.sh" "${RECIPE:-$W/configs/e2_pd_$MODE.yaml}" "$OUT" > "$OUT/srtctl.log" 2>&1 < /dev/null &
SPID=$!
t0=$(date +%s)
until [ -f "$OUT/logs/server_ready.json" ]; do
    kill -0 "$SPID" 2> /dev/null || { echo "srtctl exited before the servers were ready:"; tail -40 "$OUT/srtctl.log"; exit 1; }
    (( $(date +%s) - t0 > ${READY_S:-4200} )) && { echo "servers not ready after ${READY_S:-4200}s"; exit 1; }
    sleep 20
done
ev serve_ready "$(( $(date +%s) - t0 ))s"

P=$(ls "$OUT"/logs/*_prefill_w0.out 2> /dev/null | head -1 | xargs -r basename | sed 's/^smci355-ccs-aus-//; s/_prefill_w0.out$//')
DN=($(ls "$OUT"/logs/*_decode_*.out 2> /dev/null | xargs -r -n1 basename | sed 's/^smci355-ccs-aus-//; s/_decode_.*$//' | sort -u))
[ -n "$P" ] && [ ${#DN[@]} -ge 2 ] || { echo "cannot tell the roles from $OUT/logs: P=$P D=${DN[*]}"; ls "$OUT/logs"; exit 1; }
export A=${DN[0]} B=${DN[1]} C=$P DECODE="${DN[*]}"
ctl_ips "${DN[@]}" "$C"
PROF_URLS=""
for n in "${DN[@]}"; do
    port=$(grep -h -o -E -- '--port[ =][0-9]+' "$OUT"/logs/smci355-ccs-aus-${n}_decode_*.out 2> /dev/null | head -1 | grep -o -E '[0-9]+$')
    [ -n "$port" ] && PROF_URLS="$PROF_URLS $n:$port"
done
echo "prefill + router $C (${CTL[$C]}), decode ${DN[*]}, profile at:${PROF_URLS:- none}"

[ -s "$W/logs/windows_$RUN.txt" ] && { WINDOWS=$(cat "$W/logs/windows_$RUN.txt"); WINDOWS_ADD=""; }
nw=$(( $(echo ${WINDOWS:-base IRb IWk IRk IWb IWk10} | wc -w) + $(echo ${WINDOWS_ADD-IWk IWk10} | wc -w) ))
total=$(( WARM + (CYCLES * nw + 1) * (WIN + GAP) + 600 ))
cxbg "$C" "$OUT/loadgen.log" "MB_MORI=image bash $W/mb/py.sh $W/e2e/loadgen.py --urls http://${CTL[$C]}:8000 --conc $CONC --isl $ISL --osl $OSL --duration $total --ramp-s 120 --out $OUT/load" > /dev/null
ev load_start "conc=$CONC isl=$ISL osl=$OSL"
sleep "$WARM"
tail -1 "$OUT/load/live.jsonl" 2> /dev/null
run_windows
ev done -
echo "E2_DONE $(date -u +%T)"
