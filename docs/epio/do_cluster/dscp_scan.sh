#!/usr/bin/env bash
# Which DSCP does the fabric ECN-mark, and on which priority does each DSCP get paused?
# Runs LOCALLY:  [NODES_SH=nodes.sh] S1=01 S2=03 R=02 RAIL=0 DUR=15 DSCPS="0 8 ... 56" bash dscp_scan.sh <out_dir>
# S1 and S2 both write (ib_write_bw, 1 MiB, 4 QPs each) into R's host memory on the NIC's own NUMA node over one
# rail: a 2:1 incast on the switch port towards R. Per DSCP, counters of the rail's NIC on all three nodes are
# taken before and after, and the IP TOS of R's QPs is read from debugfs during the run.
set -u
HERE=$(cd "$(dirname "$0")" && pwd); source "${NODES_SH:-$HERE/../run/nodes.sh}"
OUT="${1:?out_dir}"; mkdir -p "$OUT"
S1="${S1:-01}"; S2="${S2:-03}"; R="${R:-02}"; RAIL="${RAIL:-0}"; DUR="${DUR:-15}"
DSCPS="${DSCPS:-0 8 16 24 26 32 40 46 48 56}"
r() { local n=$1; shift; timeout $((DUR + 90)) ssh "${SSH_OPTS[@]}" root@"${PUB[$n]}" "$@"; }
NUMA=$(r $R "cat /sys/class/infiniband/ionic_$RAIL/device/numa_node")

snap() {  # node tag
  r $1 "d=/sys/class/infiniband/ionic_$RAIL/ports/1/hw_counters; nd=\$(ls /sys/class/infiniband/ionic_$RAIL/device/net); for c in rx_rdma_ecn_pkts rx_rdma_cnp_pkts tx_rdma_cnp_pkts tx_rdma_retx_pkts tx_rdma_ack_timeout rx_rdma_ucast_bytes tx_rdma_ucast_bytes req_rx_pkt_seq_err resp_rx_outouf_seq resp_tx_pkt_seq_err; do echo \"\$c \$(cat \$d/\$c)\"; done; ethtool -S \$nd | awk '/frames_(tx|rx)_(pri)?pause:|(tx|rx)_pripause_[0-7]_1us_count:/ {sub(\":\",\"\",\$1); print \$1, \$2}'" > "$OUT/$2.node$1.txt"
}

k=0
for dscp in $DSCPS; do
  tc=$((dscp * 4)); tag="dscp$dscp"; p1=$((19300 + 2 * k)); p2=$((p1 + 1)); k=$((k + 1))
  echo "=== $tag (tclass $tc) $(date -u +%T)"
  for n in $S1 $S2 $R; do snap $n "$tag.before"; done
  for p in $p1 $p2; do
    r $R "numactl --cpunodebind=$NUMA --membind=$NUMA ib_write_bw -d ionic_$RAIL -x 1 --tclass=$tc -s 1048576 -q 4 -D $DUR -F --report_gbits -p $p" > "$OUT/$tag.server$p.log" 2>&1 &
  done
  sleep 3
  r $S1 "numactl --cpunodebind=$NUMA --membind=$NUMA ib_write_bw -d ionic_$RAIL -x 1 --tclass=$tc -s 1048576 -q 4 -D $DUR -F --report_gbits -p $p1 10.201.$RAIL.${SUF[$R]}" > "$OUT/$tag.client_s1.log" 2>&1 &
  r $S2 "numactl --cpunodebind=$NUMA --membind=$NUMA ib_write_bw -d ionic_$RAIL -x 1 --tclass=$tc -s 1048576 -q 4 -D $DUR -F --report_gbits -p $p2 10.201.$RAIL.${SUF[$R]}" > "$OUT/$tag.client_s2.log" 2>&1 &
  sleep $((DUR / 2 + 3))
  r $R "b=\$(basename \$(readlink -f /sys/class/infiniband/ionic_$RAIL/device)); for q in /sys/kernel/debug/ionic/\$b/lif0/rdma/qp/*; do awk '/hdr_ip4_tos|dcqcn_profile/ {printf \"%s=%s \", \$1, \$2}' \$q/info; echo; done" > "$OUT/$tag.qp.txt" 2>&1
  wait
  for n in $S1 $S2 $R; do snap $n "$tag.after"; done
  sleep 2
done
for n in $S1 $S2 $R; do r $n "pkill -f 'ib_write_bw .* -p 193[0-9][0-9]' && echo leftover-killed-on-node$n" 2>/dev/null; done; true
