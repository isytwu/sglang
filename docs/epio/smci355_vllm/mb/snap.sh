#!/usr/bin/env bash
# Runs ON a smci355 host:  snap.sh <tag>  ->  one JSON line on stdout
# RDMA bytes and loss-recovery counters of ionic_0..7, PFC summed over their ports. Read-only.
tag="${1:-snap}"
{
    printf '{"tag":"%s","host":"%s","ts":%s,"dev":{' "$tag" "$(hostname -s)" "$(date +%s.%N)"
    for i in 0 1 2 3 4 5 6 7; do
        d=/sys/class/infiniband/ionic_$i/ports/1/hw_counters
        [ "$i" -gt 0 ] && printf ','
        printf '"%d":{' "$i"
        first=1
        for c in tx_rdma_ucast_bytes rx_rdma_ucast_bytes tx_rdma_retx_pkts tx_rdma_ack_timeout \
                 req_rx_pkt_seq_err resp_rx_outouf_seq req_rx_impl_nak_seq_err rx_rdma_cnp_pkts rx_rdma_ecn_pkts; do
            [ $first = 1 ] || printf ','; first=0
            printf '"%s":%s' "$c" "$(cat "$d/$c" 2>/dev/null || echo 0)"
        done
        printf '}'
    done
    printf '},"pfc":'
    for i in 0 1 2 3 4 5 6 7; do ethtool -S "$(ls /sys/class/infiniband/ionic_$i/device/net/)" 2>/dev/null; done | awk '
        $1 == "frames_tx_pripause:" {a += $2}  $1 == "tx_pripause_0_1us_count:" {b += $2}  $1 == "tx_pripause_3_1us_count:" {c += $2}
        $1 == "frames_rx_pripause:" {d += $2}  $1 == "rx_pripause_0_1us_count:" {e += $2}  $1 == "rx_pripause_3_1us_count:" {f += $2}
        END {printf "{\"tx_frames\":%.0f,\"tx_us_p0\":%.0f,\"tx_us_p3\":%.0f,\"rx_frames\":%.0f,\"rx_us_p0\":%.0f,\"rx_us_p3\":%.0f}", a, b, c, d, e, f}'
    printf '}\n'
}
