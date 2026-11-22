#!/usr/bin/env bash
# Runs on the LOGIN node, from a copy:  cp tools/idle_run.sh logs/run_idle_<job>.sh
#                                       setsid nohup bash logs/run_idle_<job>.sh <held holder job> > /dev/null 2>&1 < /dev/null &
# Lets a held holder job (its launcher already waiting for it) use the cluster only while nobody else needs it.
# Every job here has the same priority (multifactor with every weight 0, nice cannot go below it), so a pending job
# runs in submit order, ahead of everything submitted after it: the job is only released when it can start at once.
# Every minute:
#   held      release it once NEED (5) nodes have been idle with no other job waiting for nodes for IDLE_MIN (10)
#             minutes in a row; hold it again if it has not started 3 minutes after the release
#   running   once another job has waited for nodes for 2 minutes, empty the job's phase file: its launcher finishes
#             the phase in progress, cleans up and cancels the job. The phases it did not reach go to a new held job
#             (setup build warm first, same windows files), watched the same way, at most ROUNDS (3) jobs in all
# Stops when the job ends any other way, or when logs/idle_<job>.off exists (job left as it is). Log: logs/idle_run.log
set -uo pipefail
W=/apps/yutongwu/store-vllm/k31-vllm-ep-io-smci355-20261119
export PATH=/opt/slurm/bin:$PATH
JOB=${1:?held holder job}
NEED=${NEED:-5} IDLE_MIN=${IDLE_MIN:-10} ROUNDS=${ROUNDS:-3}
exec >> "$W/logs/idle_run.log" 2>&1
log() { echo "$(date -u +%FT%TZ) [$JOB] $*"; }
# jobs other than this one waiting for nodes (held jobs, dependencies and begin times do not count)
waiting() { squeue -h -t PENDING -o '%i %r' | awk -v j="$JOB" '$1 != j && $2 ~ /^(Resources|Priority|None)$/' | wc -l; }
# nodes planned for a pending job show as "planned", not "idle"
idle_nodes() { sinfo -h -N -o '%N %T' | awk '$2 == "idle" { print $1 }' | sort -u | wc -l; }
state() { squeue -h -j "$JOB" -o %T 2> /dev/null; }

yield_job() {
    local pf=$W/logs/phases_$JOB.txt left nj st
    left=$(tr ' ' '\n' < "$pf" | grep -vxE 'setup|build|warm' | xargs)
    : > "$pf"
    log "$(waiting) job(s) waiting for nodes: ending after the phase in progress (not reached: ${left:-none})"
    while [ -n "$lpid" ] && kill -0 "$lpid" 2> /dev/null; do sleep 30; done
    # a launcher that had not popped a phase yet leaves the job alone
    sleep 30
    st=$(state)
    [[ $st == RUNNING || $st == PENDING ]] && { log "launcher ended, job still $st: cancelling it"; scancel "$JOB"; }
    [ -n "$left" ] || { log "nothing left to run; stopping"; exit 0; }
    if (( round >= ROUNDS )); then
        echo "$left" > "$W/logs/deferred_$JOB.txt"
        log "$ROUNDS jobs used; not reached, in logs/deferred_$JOB.txt: $left"; exit 0
    fi
    nj=$(cd "$W" && sbatch --parsable --hold --job-name="$NAME" --nodes="$NEED" --time=04:00:00 tools/holder.sbatch) ||
        { log "resubmitting failed"; exit 1; }
    echo "setup build warm $left" > "$W/logs/phases_$nj.txt"
    for f in "$W"/logs/windows_*_"$JOB".txt; do [ -e "$f" ] && cp "$f" "${f%_"$JOB".txt}_$nj.txt"; done
    cp "$W/tools/launch.sh" "$W/logs/run_launch_$nj.sh"
    (cd "$W" && setsid nohup bash "logs/run_launch_$nj.sh" "$nj" >> "logs/nohup_launch_$nj.out" 2>&1 < /dev/null &)
    log "not reached goes to held job $nj: setup build warm $left"
    JOB=$nj round=$((round + 1)) calm=0 busy=0 rel=0
    sleep 10
    lpid=$(cat "$W/logs/launch_$JOB.pid" 2> /dev/null)
}

NAME=$(squeue -h -j "$JOB" -o %j)
lpid=$(cat "$W/logs/launch_$JOB.pid" 2> /dev/null)
[ -n "$NAME" ] && [ -n "$lpid" ] && kill -0 "$lpid" 2> /dev/null || { log "no such job, or no launcher waiting for it"; exit 1; }
log "watching $NAME (launcher $lpid): release after $NEED nodes idle and nothing waiting for $IDLE_MIN min, yield when a job waits"
round=1 calm=0 busy=0 rel=0
while :; do
    [ -e "$W/logs/idle_$JOB.off" ] && { log "logs/idle_$JOB.off: stopping, job left as it is"; exit 0; }
    read -r st rs <<< "$(squeue -h -j "$JOB" -o '%T %r' 2> /dev/null)"
    case ${st:-} in
    '') log "job gone; stopping"; exit 0 ;;
    PENDING)
        if [[ $rs == JobHeld* ]]; then
            w=$(waiting) i=$(idle_nodes)
            if (( w == 0 && i >= NEED )); then calm=$((calm + 1)); else calm=0; fi
            if (( calm >= IDLE_MIN )); then
                log "$i nodes idle, nothing waiting for $IDLE_MIN min: releasing"; scontrol release "$JOB"; rel=$SECONDS calm=0
            fi
        elif (( rel && SECONDS - rel > 180 )); then
            log "still pending 3 min after the release ($rs): holding it again"; scontrol hold "$JOB"; rel=0
        fi ;;
    RUNNING)
        (( rel )) && { log "running on $(squeue -h -j "$JOB" -o %N)"; rel=0; }
        if (( $(waiting) > 0 )); then busy=$((busy + 1)); else busy=0; fi
        (( busy >= 2 )) && yield_job ;;
    COMPLETED | CANCELLED | FAILED | TIMEOUT | NODE_FAIL | PREEMPTED | OUT_OF_MEMORY | BOOT_FAIL | DEADLINE)
        log "job ended ($st); stopping"; exit 0 ;;
    esac
    sleep 60
done
