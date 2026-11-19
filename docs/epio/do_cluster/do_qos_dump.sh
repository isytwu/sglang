#!/usr/bin/env bash
# Read-only QoS / PFC / DCQCN snapshot of one DO MI355X VM, one line per item, for comparing the 8 nodes.
# Runs ON a node (as root):  ssh root@node 'bash -s' < do_qos_dump.sh
export LC_ALL=C
for ib in $(ls /sys/class/infiniband | sort -V); do
  nd=$(ls /sys/class/infiniband/$ib/device/net | head -1)
  bdf=$(basename "$(readlink -f /sys/class/infiniband/$ib/device)")
  d=/sys/class/infiniband/$ib
  for s in pfc ets app buffer maxrate dcbx; do
    echo "$ib dcb_$s $(timeout 5 dcb $s show dev $nd 2>&1 | tr '\n' ' ' | sed 's/  */ /g')"
  done
  echo "$ib pause $(ethtool -a $nd 2>&1 | tail -n +2 | tr '\n' ' ' | sed 's/\s\+/ /g')"
  echo "$ib priv_flags $(ethtool --show-priv-flags $nd 2>&1 | tail -n +2 | tr '\n' ' ' | sed 's/\s\+/ /g')"
  echo "$ib mtu $(cat /sys/class/net/$nd/mtu) active_mtu $(ibv_devinfo -d $ib 2>/dev/null | awk '/active_mtu/ {print $2}')"
  echo "$ib fec $(ethtool --show-fec $nd 2>&1 | awk -F': ' '/Active FEC/ {print $2}')"
  echo "$ib mad_sa_cc $(cat $d/mad_sa_cc 2>/dev/null) puec_nports $(cat $d/puec_nports 2>/dev/null)"
  echo "$ib dcqcn_match_default $(cat $d/dcqcn/match_default)"
  echo "$ib dcqcn_match_rules [$(cat $d/dcqcn/match_rules | tr '\n' ';')]"
  echo "$ib dcqcn_profiles_md5 $(cd $d/dcqcn/profiles && for p in $(ls | sort -n); do for s in roce_rp roce_np; do for f in $(ls $p/$s | sort); do echo "$p/$s/$f=$(cat $p/$s/$f)"; done; done; done | md5sum | cut -c1-12)"
  for p in 0 1 2 3 4 5 6 7; do
    echo "$ib pripause_prio$p tx_us=$(ethtool -S $nd | awk -v k="tx_pripause_${p}_1us_count:" '$1==k {print $2}') rx_us=$(ethtool -S $nd | awk -v k="rx_pripause_${p}_1us_count:" '$1==k {print $2}')"
  done
  echo "$ib pause_frames $(ethtool -S $nd | awk '/frames_(tx|rx)_(pri)?pause:/ {printf "%s%s ", $1, $2}')"
  hc=$d/ports/1/hw_counters
  echo "$ib ecn_cnp rx_ecn=$(cat $hc/rx_rdma_ecn_pkts) rx_cnp=$(cat $hc/rx_rdma_cnp_pkts) tx_cnp=$(cat $hc/tx_rdma_cnp_pkts)"
  # QPs alive right now: which IP TOS they put on the wire and which DCQCN profile they use
  echo "$ib live_qps $(for q in /sys/kernel/debug/ionic/$bdf/lif0/rdma/qp/*; do t=$(awk '/hdr_ip4_tos/ {print $2}' $q/info 2>/dev/null); p=$(awk '/dcqcn_profile/ {print $2}' $q/info 2>/dev/null); [ -n "$t" ] && echo "tos=$t,profile=$p"; done | sort | uniq -c | awk '{printf "%s x%s; ", $2, $1}')"
done
