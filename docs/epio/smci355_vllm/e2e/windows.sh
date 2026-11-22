# Sourced by e2e/e1.sh and e2e/e2.sh (login node, after mb/lib.sh): the measurement windows against a running decode.
# Needs DECODE (the decode nodes), C (the IO peer), OUT, IOTC, WIN, GAP, CYCLES and PROF_URLS ("host:port ..." of the
# decode API servers that take /start_profile, /stop_profile; empty = no captures). TARGETS (default: every decode
# node) are the decode nodes the IO goes to. Windows of WIN s alternate with GAP s of no IO; WINDOWS (default
# "base IWk IRk IWk10") picks them:
#   base    no IO
#   IWk     KV-shaped RDMA writes from C's GPUs into the targets' GPUs (io_inject, back to back; the PD WRITE mode)
#   IRk     the targets pull KV-shaped RDMA reads from C's GPUs (the PD READ mode)
#   IWk10   the writes at 10 GB/s per NIC
#   IWhR    the writes into the targets' pinned host memory on the far socket (KV landing in DRAM across NUMA)
#   IWkT0   IWk with the IO in TC 0 (the lossy class)
#   IWk1 / IRk1  IWk / IRk into the first target only: all of C's NICs on one decode node
#   IRk5 / IRk10 the reads at 5 / 10 GB/s per NIC; IRkT0 the reads in TC 0
#   IWk5 / IWk2 / IWk1g  the writes at 5 / 2 / 1 GB/s per NIC (in e1ra_6857 10 GB/s already cost +72 %)
#   IWk1gS / IWk1gB      1 GB/s per NIC in 1 MB bursts every ms / 64 MB bursts every 64 ms (IWk1g: 16 MB every 16 ms)
#   IWk10S / IWk5S / IWkS  10 / 5 GB/s per NIC in 1 MB bursts (every 0.1 / 0.2 ms), and 1 MB bursts back to back
#                          (io_inject keeps one burst per QP in flight: the burst is what each NIC has in flight)
#   IWk10c2 / c4 / c8      10 GB/s per NIC in 2 / 4 / 8 MB bursts (in e1_6866 1 MB cost nothing, 16 MB +72 %)
#   IWkSi4 / IWkSi16       1 MB bursts back to back with 4 / 16 of them in flight (4 / 16 MB in flight per NIC)
#   IRkS / IRk10S          the reads in 1 MB bursts back to back / at 10 GB/s per NIC
#   IWkR16 / IWkR64        one 16 / 64 MB burst per NIC per second: roughly a 4k / 16k-token prompt's KV (576 B per
#                          token and layer, DCP 8 spreads it over the DP rank's 8 GPUs) landing once a second
#   SPIN    no IO: 8 busy loops on each target in a job step, i.e. on the CPUs io_inject's processes get (each of
#           them spins a core polling its CQs; the vLLM workers are pinned to the same socket)
#   IWk0    io_inject's receivers set up on the targets (GPU buffers registered, CQs polled), one 16 MB burst per
#           rail, then nothing: the receivers without the traffic
#   IWkN1   IWk with the receivers moved to socket 1 (taskset -c 64-127), away from the vLLM workers
# logs/windows_<run>.txt (e.g. windows_e1m_6855.txt), when it exists, replaces WINDOWS (and WINDOWS_ADD) for that run.
#   IRb / IWb    mori's IO benchmark (8 GPU pairs, 32 KB transfers, batches of 8192) between C and every target: the
#                targets read C (IRb) or C writes them (IWb). In the microbenchmarks IRb's pattern (thousands of small
#                READs in flight) slowed EP 10-40x where KV-shaped reads at a higher rate cost 10-20 %
#   IRb1 / IWb1  the same on the first target only (all of C's NICs on one decode node)
# Every window gets a torch-profiler capture on the decode servers in its middle; events.tsv has the window and
# capture times. snaps.jsonl has every node's RDMA counters (mb/snap.sh: CNPs and ECN-marked packets received,
# retransmissions, PFC) once the IO runs (<window>_<cycle>_a) and before it stops (_b).
EV=$OUT/events.tsv
ev() { printf '%s\t%s\t%s\n' "$(date +%s.%N)" "$1" "$2" | tee -a "$EV"; }
IOB="--enable-batch-transfer --enable-sess --buffer-size 32768 --transfer-batch-size 8192 --iters 1000000000 --num-initiator-dev 8 --num-target-dev 8"

profile() {
    [ -n "${PROF_URLS:-}" ] || return 0
    local cp=() u
    ev "prof_$1" start
    for u in $PROF_URLS; do hx "${u%%:*}" "curl -s -m 30 -X POST http://127.0.0.1:${u##*:}/start_profile" > /dev/null & cp+=($!); done
    wait "${cp[@]}"
    sleep "${PROFILE_S:-8}"
    cp=()
    for u in $PROF_URLS; do hx "${u%%:*}" "curl -s -m 600 -X POST http://127.0.0.1:${u##*:}/stop_profile" > /dev/null & cp+=($!); done
    wait "${cp[@]}"
    ev "prof_$1" end
}

win_rates() { local n s=""; for n in $DECODE "$C"; do s="$s$n $(rate "$n" 2) | "; done; echo "$s"; }
snaps() { local n; for n in $DECODE "$C"; do hx "$n" "bash $W/mb/snap.sh $1" >> "$OUT/snaps.jsonl" 2> /dev/null; done; }

# alive: the decode streamed something in the last minute (the loadgen's 10 s records); a hung decode (e1_6857 IWb,
# e1x_6857 ramp-up) ends the run instead of idling through the remaining windows
alive() {
    local recent
    recent=$(tail -6 "$OUT/load/live.jsonl" 2> /dev/null | python3 -c "import sys, json; print(sum(json.loads(l)['gaps'] for l in sys.stdin))" 2> /dev/null)
    [ "${recent:-1}" != 0 ]
}

window() {
    local name=$1 cyc=$2 per=0 mb=16 inf=1 tc=$IOTC t tg=(${TARGETS:-$DECODE})
    alive || { ev stalled "no streamed chunks in the last minute before $name"; exit 3; }
    ev "$name" start
    case "$name" in
        IWk10|IRk10) per=1.6 ;;
        IRk5|IWk5) per=3.2 ;;
        IWk2) per=8 ;;
        IWk1g) per=16 ;;
        IWk1gS) per=1 mb=1 ;;
        IWk1gB) per=64 mb=64 ;;
        IWk10S|IRk10S) per=0.1 mb=1 ;;
        IWk5S) per=0.2 mb=1 ;;
        IWkS|IRkS) mb=1 ;;
        IWk10c2) per=0.2 mb=2 ;;
        IWk10c4) per=0.4 mb=4 ;;
        IWk10c8) per=0.8 mb=8 ;;
        IWkSi4) mb=1 inf=4 ;;
        IWkSi16) mb=1 inf=16 ;;
        IWkR16) per=1000 mb=16 ;;
        IWkR64) per=1000 mb=64 ;;
        IWkT0|IRkT0) tc=0 ;;
        IWk1|IRk1|IRb1|IWb1) tg=("${tg[0]}") ;;
    esac
    case "$name" in
        IWk|IWk10|IWkT0|IWk1|IWk5|IWk2|IWk1g|IWk1gS|IWk1gB|IWk10S|IWk5S|IWkS|IWk10c2|IWk10c4|IWk10c8|IWkSi4|IWkSi16|IWkR16|IWkR64)
            for t in "${tg[@]}"; do
                inj_start "$OUT/inj_${name}_$cyc/$t" "$tc" write "$C" "$t" --chunk-mb $mb --period-ms $per --inflight $inf --duration $((WIN + 120))
            done ;;
        IWhR)
            for t in "${tg[@]}"; do
                INJ_PMEM="--mem host --numa remote" inj_start "$OUT/inj_${name}_$cyc/$t" "$tc" write "$C" "$t" --chunk-mb 16 --period-ms 0 --duration $((WIN + 120))
            done ;;
        IWk0)
            for t in "${tg[@]}"; do
                inj_start "$OUT/inj_${name}_$cyc/$t" "$tc" write "$C" "$t" --chunk-mb 16 --period-ms 3600000 --duration $((WIN + 120))
            done ;;
        IWkN1)
            for t in "${tg[@]}"; do
                INJ_PTASK="taskset -c 64-127" inj_start "$OUT/inj_${name}_$cyc/$t" "$tc" write "$C" "$t" --chunk-mb 16 --period-ms 0 --duration $((WIN + 120))
            done ;;
        SPIN)
            # inj_stop kills the steps with the rest of INJ_PIDS
            for t in "${tg[@]}"; do
                INJ_PIDS+=("$(cxbg "$t" "$OUT/spin_${cyc}_$t.log" "for i in 1 2 3 4 5 6 7 8; do timeout $((WIN + 60)) bash -c 'while :; do :; done' & done; wait")")
            done ;;
        IRk|IRk1|IRk5|IRk10|IRkT0|IRkS|IRk10S)
            for t in "${tg[@]}"; do
                inj_start "$OUT/inj_${name}_$cyc/$t" "$tc" read "$t" "$C" --chunk-mb $mb --period-ms $per --inflight $inf --duration $((WIN + 120))
            done ;;
        IRb|IWb|IRb1|IWb1)
            local op=read; [[ $name == IWb* ]] && op=write
            for t in "${tg[@]}"; do
                if [ $op = read ]; then
                    iob_start "$OUT/iob_${name}_$cyc/$t" "$tc" "$t" "$C" $IOB --op-type read
                else
                    iob_start "$OUT/iob_${name}_$cyc/$t" "$tc" "$C" "$t" $IOB --op-type write
                fi
            done
            sleep 25 ;;
    esac
    [ "$name" = base ] || { sleep 20; ev "${name}_io_on" "$(win_rates)"; }
    snaps "${name}_${cyc}_a"
    sleep $((WIN / 2 - 20))
    profile "${name}_$cyc"
    sleep $((WIN / 2 - 20))
    snaps "${name}_${cyc}_b"
    [ "$name" = base ] || { ev "${name}_io_end" "$(win_rates)"; inj_stop $DECODE "$C"; iob_stop $DECODE "$C"; }
    ev "$name" end
    tail -1 "$OUT/load/live.jsonl" 2> /dev/null
}

run_windows() {
    local cyc name list
    if [ -s "$W/logs/windows_$RUN.txt" ]; then WINDOWS=$(cat "$W/logs/windows_$RUN.txt"); WINDOWS_ADD=""; fi
    # IWb last: in e1_6857 the decode hung in it (an RCCL all-gather never completed, 600 s watchdog), so whatever
    # follows it would be lost
    # WINDOWS_ADD (default IWk IWk10, the clean large effect of e1_6857 and its 10 GB/s version) joins every list that
    # lacks them, so every E1 run can be compared on them
    local all="${WINDOWS:-base IRb IWk IRk IWb IWk10}" w
    for w in ${WINDOWS_ADD-IWk IWk10}; do [[ " $all " == *" $w "* ]] || all="$all $w"; done
    list="$(for name in $all; do [[ $name == IWb* ]] || printf '%s ' "$name"; done)"
    list="$list$(for name in $all; do [[ $name == IWb* ]] && printf '%s ' "$name"; done)"
    for cyc in $(seq 1 "$CYCLES"); do
        for name in $list; do
            window "$name" "$cyc"
            sleep "$GAP"
        done
    done
    window base $((CYCLES + 1))
}
