#!/usr/bin/env bash
# DCQCN on the DO VMs: does the switch ECN-mark, does the receiver send CNPs, does the sender slow down?
# Runs LOCALLY:  [NODES_SH=nodes.sh] S=05 R=06 RAILS="0 1 2 3" DUR=20 CONFIGS="104:0 106:0 104:1 106:1" bash dcqcn_test.sh <out_dir>
# NODES_SH defines PUB[node]=ssh address, SUF[node]=last octet of the rail IPs (10.201.<rail>.<SUF>) and SSH_OPTS.
# Sender S writes (ib_write_bw, 1 MiB, 4 QPs per rail) into host memory of R on the NUMA node away from R's NICs,
# so R's NICs cannot drain at line rate and push back (PFC), which builds the switch queue towards R.
# Each config is TC:match_default; match_default is set on the NICs used on both nodes and put back to 0 at exit.
set -u
HERE=$(cd "$(dirname "$0")" && pwd); source "${NODES_SH:-$HERE/../run/nodes.sh}"
OUT="${1:?out_dir}"; mkdir -p "$OUT"
S="${S:-05}"; R="${R:-06}"; RAILS="${RAILS:-0 1 2 3}"; DUR="${DUR:-20}"; CONFIGS="${CONFIGS:-104:0 106:0 104:1 106:1}"
RMEM="${RMEM:-1}"; SMEM="${SMEM:-0}"; CPU="${CPU:-0}"
r() { local n=$1; shift; timeout $((DUR + 120)) ssh "${SSH_OPTS[@]}" root@"${PUB[$n]}" "$@"; }

set_md() {  # value
  for n in $S $R; do r $n "for i in $RAILS; do echo $1 > /sys/class/infiniband/ionic_\$i/dcqcn/match_default; done; for i in $RAILS; do echo -n \"ionic_\$i=\$(cat /sys/class/infiniband/ionic_\$i/dcqcn/match_default) \"; done"; echo " <- node$n match_default"; done
}
restore() { echo "restoring match_default=0"; set_md 0; }
trap restore EXIT

snap() {  # node tag
  r $1 "for i in $RAILS; do d=/sys/class/infiniband/ionic_\$i/ports/1/hw_counters; nd=\$(ls /sys/class/infiniband/ionic_\$i/device/net); for c in rx_rdma_ecn_pkts rx_rdma_cnp_pkts tx_rdma_cnp_pkts tx_rdma_retx_pkts tx_rdma_ack_timeout rx_rdma_ucast_bytes tx_rdma_ucast_bytes req_rx_pkt_seq_err resp_rx_outouf_seq; do echo \"ionic_\$i \$c \$(cat \$d/\$c)\"; done; ethtool -S \$nd | awk -v i=ionic_\$i '/frames_(tx|rx)_pripause:|(tx|rx)_pripause_[03]_1us_count:/ {sub(\":\",\"\",\$1); print i, \$1, \$2}'; done" > "$OUT/$2.node$1.txt"
}

qp_profiles() {  # node tag
  r $1 "for i in $RAILS; do b=\$(basename \$(readlink -f /sys/class/infiniband/ionic_\$i/device)); for q in /sys/kernel/debug/ionic/\$b/lif0/rdma/qp/*; do echo \"ionic_\$i \$(basename \$q) \$(grep -iE 'dcqcn|tos|dscp|tclass|traffic|ecn' \$q/info 2>/dev/null | tr '\n' ' ')\"; done; done" > "$OUT/$2.qp.node$1.txt"
}

for cfg in $CONFIGS; do
  tc=${cfg%%:*}; md=${cfg##*:}; tag="tc${tc}_md${md}"
  echo "=== $tag $(date -u +%T)"
  set_md "$md"
  snap $S "$tag.before"; snap $R "$tag.before"
  for i in $RAILS; do
    r $R "numactl --cpunodebind=$CPU --membind=$RMEM ib_write_bw -d ionic_$i -x 1 --tclass=$tc -s 1048576 -q 4 -D $DUR -F --report_gbits -p $((19100 + i))" > "$OUT/$tag.server$i.log" 2>&1 &
  done
  sleep 3
  for i in $RAILS; do
    r $S "numactl --cpunodebind=$CPU --membind=$SMEM ib_write_bw -d ionic_$i -x 1 --tclass=$tc -s 1048576 -q 4 -D $DUR -F --report_gbits -p $((19100 + i)) 10.201.$i.${SUF[$R]}" > "$OUT/$tag.client$i.log" 2>&1 &
  done
  sleep $((DUR / 2 + 3)); qp_profiles $S "$tag"; qp_profiles $R "$tag"
  wait
  snap $S "$tag.after"; snap $R "$tag.after"
  sleep 3
done
