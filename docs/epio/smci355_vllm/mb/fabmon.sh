#!/usr/bin/env bash
# Runs ON a smci355 host:  fabmon.sh <out.tsv> [interval_s]
# Port counters of the 8 backend NICs, in ionic_0..7 order, every interval (default 1 s): PFC summed over the 8 ports
# (pause times in us; tx_* = pause frames this host sent because its receive side was congested, rx_* = pause frames
# the switch sent to this host), the bytes of each port (all traffic, RDMA included), and the CNPs and ECN-marked
# packets the 8 RDMA devices received (a CNP received = one of this host's sending QPs was told to slow down).
# Read-only.
out="${1:?out.tsv}"; iv="${2:-1}"
mkdir -p "$(dirname "$out")"
ports=""
for i in 0 1 2 3 4 5 6 7; do ports="$ports $(ls /sys/class/infiniband/ionic_$i/device/net/)"; done
hw=/sys/class/infiniband/ionic_[0-7]/ports/1/hw_counters
{ printf '# ports (ionic_0..7):%s\n' "$ports"
  printf 'ts\ttx_pfc_frames\ttx_pause_us_p0\ttx_pause_us_p3\trx_pfc_frames\trx_pause_us_p0\trx_pause_us_p3'
  for i in 0 1 2 3 4 5 6 7; do printf '\ttx_B_%d\trx_B_%d' "$i" "$i"; done; printf '\tcnp_rx\tecn_rx\n'; } > "$out"
while :; do
    t=$(date +%s.%N)
    cnp=$(cat $hw/rx_rdma_cnp_pkts 2>/dev/null | awk '{s += $1} END {printf "%.0f", s}')
    ecn=$(cat $hw/rx_rdma_ecn_pkts 2>/dev/null | awk '{s += $1} END {printf "%.0f", s}')
    p=0
    for d in $ports; do echo "PORT $p"; ethtool -S "$d" 2>/dev/null; p=$((p + 1)); done | awk -v t="$t" -v cnp="$cnp" -v ecn="$ecn" '
        $1 == "PORT" {p = $2}
        $1 == "frames_tx_pripause:" {a += $2}  $1 == "tx_pripause_0_1us_count:" {b += $2}  $1 == "tx_pripause_3_1us_count:" {c += $2}
        $1 == "frames_rx_pripause:" {d += $2}  $1 == "rx_pripause_0_1us_count:" {e += $2}  $1 == "rx_pripause_3_1us_count:" {f += $2}
        $1 == "octets_tx_ok:" {tx[p] = $2}  $1 == "octets_rx_ok:" {rx[p] = $2}
        END {printf "%s\t%.0f\t%.0f\t%.0f\t%.0f\t%.0f\t%.0f", t, a, b, c, d, e, f
             for (i = 0; i < 8; i++) printf "\t%.0f\t%.0f", tx[i], rx[i]; printf "\t%s\t%s\n", cnp, ecn}' >> "$out"
    sleep "$iv"
done
