#!/usr/bin/env bash
# Runs on the LOGIN node:  JOB=<holder> A=<node> B=<node> C=<node> [EPN="A B D E"] CAMP=<name> [ONLY="scn ..."]
#                          [SHAPES="sh ..."] [SHAPES_ONLY="..."] [EP_LOCAL_BIG_N=20 EP_SCRIPT=test_dispatch_combine_v2_internode_local.py]
#                          [IOTC=104] [REPS=2] bash mb/campaign.sh
# EP over the EP nodes (EPN, default A + B = EP16; four nodes = EP32) next to MoRI IO from / to the IO node C, all in
# TC 104 unless IOTC or the scenario says otherwise. B is the EP node that receives the IO, A the one that sends it.
# The default scenario list is the DO microbenchmarks' (microbench/run/mb_campaign5.sh there):
#   I*:  IO into B from C: B's NICs receive IO data, as responder (W: C writes B) or requester (R: B reads C);
#        IW is the PD decode in WRITE mode (the prefill writes its KV), IR the decode in READ mode (it pulls its KV)
#   O*:  IO out of A to C: A's NICs transmit IO data (W: A writes C)
#   D2*: IO between A and B themselves
#   8 = all 8 rails (8 GPU pairs, mori's IO benchmark), 1 = rail 0 only; k = KV-shaped io_inject (back to back).
# More, picked with ONLY:
#   IWkT0 / IR8T0  IWk / IR8 with the IO in TC 0 (DSCP 0: the lossy class, away from EP's no-drop class)
#   IWhL / IWhR    KV-shaped writes from C's GPUs into pinned host memory of B on the NIC's socket / on the other one
#   IRhR           B reads C's GPU memory into its host memory on the far socket
#   I2Wk           KV-shaped writes into B from C and from A at once (2:1 incast into B)
#   IWkA           KV-shaped writes from C into every EP node (C's NICs split between them), the e2e E1's IWk;
#                  IWkA1g/2/5/10 the same at 1 / 2 / 5 / 10 GB/s per NIC (16 MB every 16 / 8 / 3.2 / 1.6 ms) and
#                  IWkAT0 in TC 0: in e1ra_6866 1 GB/s per NIC cost the decode 20 %, TC 0 at full rate nothing
#   IWkQ           IWk with 16 QPs and 8 workers per device; IWk64 with 64 MB bursts
#   IWk5/10/20     IWk at 5 / 10 / 20 GB/s per NIC (16 MB every 3.2 / 1.6 / 0.8 ms); IRk5/10/20 the same for B's reads
#   IR8s / IR8m    IR8 with batches of 64 / 1024 transfers instead of 8192 (fewer READs in flight); IR1 one pair (rail 0)
#   IR8L           IR8 with 128 KB transfers in batches of 2048 (the same bytes in flight, a quarter of the READs)
#   IR8b<n>        IR8 with batches of n = 256 / 1024 / 2048 / 4096 transfers of 32 KB (8 / 32 / 64 / 128 MB in flight per pair)
#   OR8 / ORk      C reads A: the EP node A is the READ responder (its NICs transmit the IO data)
# SHAPES replaces every scenario's shape list, SHAPES_ONLY filters it. logs/only_<camp>.txt / logs/shapes_<camp>.txt,
# when they exist, replace ONLY / SHAPES (a launcher's phase entry fixes them).
# EP shapes: v2k48 = the DO K3.1 shape (EPv2 v2_ll, 48 tokens / rank), v2k128 = the same at the Moonshot load (EP32,
# global batch 512, qlen 8 = 128 tokens / rank), v1k128 = vLLM's MoRI all2all (v1 InterNodeV1LL, the image's mori)
# at 128 tokens / rank, v?k256 / 384 / 512 = more EP traffic per NIC (at EP16, 384 tokens / rank put about the bytes of
# EP32 at 128 on each NIC), v1q4k128 = vLLM's kernel with 4 QPs per PE, v1htk128 = InterNodeV1 (mori_high_throughput),
# v1m2k64 / v1m2k128 = vLLM's kernel with the e2e decode's buffers (max-num-batched-tokens 2048) at 64 / 128 tokens,
# v2def = the DO DeepSeek shape (hidden 7168, top-8, 128 tokens).
# EP_LOCAL_BIG_N=20 spaces the EP rounds out to a decode's duty cycle (20 GEMMs of ~65 us after every combine).
set -uo pipefail
source "$(dirname "$0")/lib.sh"
: "${JOB:?}" "${A:?}" "${B:?}" "${C:?}"
CAMP="${CAMP:?}"; REPS="${REPS:-2}"; IOTC="${IOTC:-104}"; ROUNDS="${ROUNDS:-2000}"
[ -s "$W/logs/only_$CAMP.txt" ] && ONLY=$(cat "$W/logs/only_$CAMP.txt")
[ -s "$W/logs/shapes_$CAMP.txt" ] && SHAPES=$(cat "$W/logs/shapes_$CAMP.txt")
OUT="$W/results/mb/$CAMP"
mkdir -p "$OUT"
exec > >(tee -a "$OUT/campaign.log") 2>&1
NODES=($(all_nodes))
ctl_ips "${NODES[@]}"
echo "=== $CAMP job $JOB A=$A (${CTL[$A]}) B=$B (${CTL[$B]}) C=$C (${CTL[$C]}) EP nodes: $(ep_nodes) IOTC=$IOTC BIG_N=${EP_LOCAL_BIG_N:-0} RAIL_ALIGN=${RAIL_ALIGN:-0} $(date -u +%FT%TZ)"

# K3.1's 896 routed experts over the EP ranks: 56 per rank at EP16, 28 at EP32
EPR=$((896 / (8 * $(ep_nodes | wc -w))))
V2="--kernel-type v2_ll --hidden-dim 3584 --topk 16 --experts-per-rank $EPR --dtype bf16 --combine-dtype bf16 --num-qp 8 --scale-dim 0 --rounds $ROUNDS --warmup 100"
V1="--kernel v1_ll --experts-per-rank $EPR --max-tokens 1024 --rounds $ROUNDS --warmup 100"
declare -A SHAPE=(
    [v2k48]="v2|$V2 --max-tokens 48"
    [v2k128]="v2|$V2 --max-tokens 128"
    [v2k256]="v2|$V2 --max-tokens 256"
    [v2k384]="v2|$V2 --max-tokens 384"
    [v2k512]="v2|$V2 --max-tokens 512"
    [v1k128]="v1|$V1 --tokens 128"
    [v1k256]="v1|$V1 --tokens 256"
    [v1k384]="v1|$V1 --tokens 384"
    [v1k512]="v1|$V1 --tokens 512"
    [v1q4k128]="v1|$V1 --tokens 128 --num-qp 4"
    [v1m2k64]="v1|${V1/--max-tokens 1024/--max-tokens 2048} --tokens 64"
    [v1m2k128]="v1|${V1/--max-tokens 1024/--max-tokens 2048} --tokens 128"
    [v1htk128]="v1|${V1/v1_ll/v1} --tokens 128"
    [v2def]="v2|--kernel-type v2_ll --max-tokens 128 --hidden-dim 7168 --topk 8 --dtype bf16 --combine-dtype bf16 --num-qp 8 --scale-dim 0 --rounds $ROUNDS --warmup 100"
)
IO="--enable-batch-transfer --enable-sess --buffer-size 32768 --transfer-batch-size 8192 --iters 1000000000"
P8="--num-initiator-dev 8 --num-target-dev 8"
KV="--chunk-mb 16 --period-ms 0 --duration 1800"

# name | kind | initiator passive | io args | shapes | IO traffic class (default IOTC)
SCN=(
  "base1|none|-|-|v2k48 v2k128 v1k128 v2def"
  "IW8|iob|$C $B|$IO --op-type write $P8|v2k48 v2k128 v1k128 v2def"
  "IR8|iob|$B $C|$IO --op-type read $P8|v2k48 v2k128 v1k128 v2def"
  "IWk|inj_write|$C $B|$KV|v2k48 v2k128 v1k128"
  "IRk|inj_read|$B $C|$KV|v2k48 v2k128 v1k128"
  "OW8|iob|$A $C|$IO --op-type write $P8|v2k48 v2k128 v1k128"
  "D2W1|iob|$A $B|$IO --op-type write|v2k48 v2k128 v1k128"
  "base2|none|-|-|v2k48 v2k128 v1k128 v2def"
)
EXTRA=(
  "IWkT0|inj_write|$C $B|$KV|v2k128 v1k128|0"
  "IR8T0|iob|$B $C|$IO --op-type read $P8|v2k128 v1k128|0"
  "IWhL|inj_write_hostL|$C $B|$KV|v2k128 v1k128"
  "IWhR|inj_write_hostR|$C $B|$KV|v2k128 v1k128"
  "IRhR|inj_read_hostR|$B $C|$KV|v2k128 v1k128"
  "I2Wk|inj_write_2src|$C $B|$KV|v2k128 v1k128"
  "IWkA|inj_write_all|$C -|$KV|v2k128 v1k128"
  "IWkA1g|inj_write_all|$C -|--chunk-mb 16 --period-ms 16 --duration 1800|v2k128 v1k128"
  "IWkA2|inj_write_all|$C -|--chunk-mb 16 --period-ms 8 --duration 1800|v2k128 v1k128"
  "IWkA5|inj_write_all|$C -|--chunk-mb 16 --period-ms 3.2 --duration 1800|v2k128 v1k128"
  "IWkA10|inj_write_all|$C -|--chunk-mb 16 --period-ms 1.6 --duration 1800|v2k128 v1k128"
  "IWkAT0|inj_write_all|$C -|$KV|v2k128 v1k128|0"
  "IWkQ|inj_write|$C $B|$KV --qp 16 --workers 8|v2k128 v1k128"
  "IWk64|inj_write|$C $B|--chunk-mb 64 --period-ms 0 --duration 1800|v2k128 v1k128"
  "IWk5|inj_write|$C $B|--chunk-mb 16 --period-ms 3.2 --duration 1800|v2k128 v1k128"
  "IWk10|inj_write|$C $B|--chunk-mb 16 --period-ms 1.6 --duration 1800|v2k128 v1k128"
  "IWk20|inj_write|$C $B|--chunk-mb 16 --period-ms 0.8 --duration 1800|v2k128 v1k128"
  "IRk5|inj_read|$B $C|--chunk-mb 16 --period-ms 3.2 --duration 1800|v2k128 v1k128"
  "IRk10|inj_read|$B $C|--chunk-mb 16 --period-ms 1.6 --duration 1800|v2k128 v1k128"
  "IRk20|inj_read|$B $C|--chunk-mb 16 --period-ms 0.8 --duration 1800|v2k128 v1k128"
  "IR8s|iob|$B $C|${IO/--transfer-batch-size 8192/--transfer-batch-size 64} --op-type read $P8|v2k128 v1k128"
  "IR8m|iob|$B $C|${IO/--transfer-batch-size 8192/--transfer-batch-size 1024} --op-type read $P8|v2k128 v1k128"
  "IR8L|iob|$B $C|${IO/--buffer-size 32768 --transfer-batch-size 8192/--buffer-size 131072 --transfer-batch-size 2048} --op-type read $P8|v2k128 v1k128"
  "IR8b256|iob|$B $C|${IO/--transfer-batch-size 8192/--transfer-batch-size 256} --op-type read $P8|v2k128 v1k128"
  "IR8b1024|iob|$B $C|${IO/--transfer-batch-size 8192/--transfer-batch-size 1024} --op-type read $P8|v2k128 v1k128"
  "IR8b2048|iob|$B $C|${IO/--transfer-batch-size 8192/--transfer-batch-size 2048} --op-type read $P8|v2k128 v1k128"
  "IR8b4096|iob|$B $C|${IO/--transfer-batch-size 8192/--transfer-batch-size 4096} --op-type read $P8|v2k128 v1k128"
  "IR1|iob|$B $C|$IO --op-type read|v2k128 v1k128"
  "OR8|iob|$C $A|$IO --op-type read $P8|v2k128 v1k128"
  "ORk|inj_read|$C $A|$KV|v2k128 v1k128"
)
# extra scenarios run in the order ONLY names them, between the default list's scenarios they sit next to
if [ -n "${ONLY:-}" ]; then
    LIST=()
    for want in $ONLY; do
        for line in "${SCN[@]}" "${EXTRA[@]}"; do [ "${line%%|*}" = "$want" ] && LIST+=("$line"); done
    done
else
    LIST=("${SCN[@]}")
fi

wait_idle 180 "${NODES[@]}" || { echo "nodes not idle: not starting"; exit 1; }
fab_start "$OUT" "${NODES[@]}"
trap 'iob_stop ${NODES[*]}; inj_stop ${NODES[*]}; ep_kill; fab_stop "$OUT" ${NODES[*]}' EXIT
for line in "${LIST[@]}"; do
    IFS='|' read -r name kind sides ioargs shapes tc <<< "$line"
    tc=${tc:-$IOTC}
    [ -f "$OUT/$name/done" ] && { echo "skip $name"; continue; }
    mkdir -p "$OUT/$name"
    echo "=== $name ($kind $sides, TC $tc) $(date -u +%T)"
    snap "$OUT/$name" pre "${NODES[@]}"
    set -- $sides; ini=${1:-}; pas=${2:-}
    case "$kind" in
        iob) iob_start "$OUT/$name" "$tc" "$ini" "$pas" $ioargs; sleep 45 ;;
        inj_write) inj_start "$OUT/$name" "$tc" write "$ini" "$pas" $ioargs; sleep 20 ;;
        inj_read) inj_start "$OUT/$name" "$tc" read "$ini" "$pas" $ioargs; sleep 20 ;;
        inj_write_hostL) INJ_PMEM="--mem host --numa local" inj_start "$OUT/$name" "$tc" write "$ini" "$pas" $ioargs; sleep 20 ;;
        inj_write_hostR) INJ_PMEM="--mem host --numa remote" inj_start "$OUT/$name" "$tc" write "$ini" "$pas" $ioargs; sleep 20 ;;
        inj_read_hostR) INJ_IMEM="--mem host --numa remote" inj_start "$OUT/$name" "$tc" read "$ini" "$pas" $ioargs; sleep 20 ;;
        inj_write_2src) inj_start "$OUT/$name/from_C" "$tc" write "$C" "$B" $ioargs
                        inj_start "$OUT/$name/from_A" "$tc" write "$A" "$B" $ioargs; sleep 20 ;;
        inj_write_all) for n in $(ep_nodes); do inj_start "$OUT/$name/to_$n" "$tc" write "$C" "$n" $ioargs; done; sleep 20 ;;
    esac
    rates "$OUT/$name/rate_before.txt"
    for sh in ${SHAPES:-$shapes}; do
        if [ -n "${SHAPES_ONLY:-}" ] && [[ " $SHAPES_ONLY " != *" $sh "* ]]; then continue; fi
        IFS='|' read -r ek eargs <<< "${SHAPE[$sh]}"
        for rep in $(seq 1 "$REPS"); do
            [ "$ek" = v1 ] && eargs_r="$eargs --out $OUT/$name/ep_${sh}_${rep}_json" || eargs_r="$eargs"
            echo "  ep $sh rep$rep: $(ep_run "$OUT/$name" "ep_${sh}_${rep}" "$ek" $eargs_r)"
        done
    done
    rates "$OUT/$name/rate_after.txt"
    snap "$OUT/$name" post "${NODES[@]}"
    case "$kind" in
        iob) iob_stop "${NODES[@]}" ;;
        inj_*) inj_stop "${NODES[@]}" ;;
    esac
    sleep 5
    echo "  after stop: $(for n in "${NODES[@]}"; do printf '%s %s / ' "$n" "$(rate "$n" 1)"; done)"
    touch "$OUT/$name/done"
done
echo "CAMPAIGN_DONE $(date -u +%T)"
