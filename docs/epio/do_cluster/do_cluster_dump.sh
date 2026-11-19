#!/usr/bin/env bash
# Read-only dump of everything a DO MI355X VM exposes about its host, GPUs, AINICs, QoS and DCQCN.
# Runs ON a node (as root):  ssh root@node 'bash -s' < do_cluster_dump.sh > nodeXX.txt
# Changes nothing: only reads sysfs / debugfs / procfs and runs query commands.
export LC_ALL=C
sec() { printf '\n########## %s\n' "$*"; }
run() { printf '\n$ %s\n' "$*"; timeout 90 bash -c "$*" 2>&1; }

sec SYSTEM
run date -u
run hostname
run uptime
run cat /etc/os-release
run uname -a
run cat /proc/cmdline
run systemd-detect-virt
run dmidecode -t system
run dmidecode -t bios
run dmidecode -t baseboard
run dmidecode -t chassis
run dmidecode -t processor
run dmidecode -t memory
run lscpu
run numactl -H
run free -g
run "grep -E 'MemTotal|MemFree|MemAvailable|Huge|Hugepagesize' /proc/meminfo"
run "for f in /sys/kernel/mm/transparent_hugepage/enabled /sys/kernel/mm/transparent_hugepage/defrag; do echo \$f: \$(cat \$f); done"
run "for n in /sys/devices/system/node/node*/hugepages/hugepages-*/nr_hugepages; do echo \$n: \$(cat \$n); done"
run "lsmod | grep -E 'amdgpu|amdxcp|ionic|ib_|rdma|iw_|kfd|drm' "
run "sysctl -a 2>/dev/null | grep -E 'rp_filter|arp_ignore|arp_announce|arp_filter|ip_forward|net.core.(rmem|wmem)_(max|default)|net.ipv4.tcp_(rmem|wmem)|kernel.numa_balancing|vm.nr_hugepages|kernel.randomize_va_space'"
run "cat /sys/kernel/mm/numa/demotion_enabled 2>/dev/null; cat /proc/sys/kernel/numa_balancing"
run "chronyc tracking; chronyc sources -v | tail -15"
run timedatectl
run lsblk
run "df -h / /scratch 2>/dev/null"
run "docker version --format 'client {{.Client.Version}} server {{.Server.Version}}'"
run "ulimit -a"
run "cat /sys/module/vfio*/parameters/* 2>/dev/null | head; ls /sys/kernel/iommu_groups | wc -l"
run "dmesg | grep -iE 'iommu|DMAR|AMD-Vi|SEV|x2apic' | head -40"

sec PCI
run lspci -nn
run lspci -tv
run "for d in /sys/bus/pci/devices/*; do c=\$(cat \$d/class); case \$c in 0x0302*|0x0380*|0x0200*|0x0207*|0x1200*) echo \"\$(basename \$d) class=\$c vendor=\$(cat \$d/vendor) device=\$(cat \$d/device) subsys=\$(cat \$d/subsystem_vendor):\$(cat \$d/subsystem_device) numa=\$(cat \$d/numa_node) link=\$(cat \$d/current_link_speed 2>/dev/null)x\$(cat \$d/current_link_width 2>/dev/null) max=\$(cat \$d/max_link_speed 2>/dev/null)x\$(cat \$d/max_link_width 2>/dev/null) driver=\$(basename \$(readlink \$d/driver) 2>/dev/null)\";; esac; done"
for b in $(lspci -D -d 1002: | awk '/Display|Processing|accelerator/ {print $1}') $(lspci -D -d 1dd8: | awk '{print $1}'); do
  run lspci -vvv -s "$b"
done

sec GPU_DRIVER_AND_ROCM
run "cat /sys/module/amdgpu/version"
run "modinfo amdgpu | grep -vE '^(alias|firmware):'"
run "for p in /sys/module/amdgpu/parameters/*; do printf '%s=%s\n' \$(basename \$p) \"\$(cat \$p 2>/dev/null)\"; done"
run "ls -d /opt/rocm* 2>/dev/null; cat /opt/rocm/.info/version 2>/dev/null; dpkg -l | grep -iE 'amdgpu|rocm-core|amd-smi|rocm-smi' | head -20"
run "amd-smi version"
run "amd-smi list"
run "amd-smi static"
run "amd-smi firmware"
run "amd-smi partition"
run "amd-smi topology"
run "amd-smi xgmi"
run "amd-smi metric --power --clock --temperature --usage --mem-usage --pcie --ecc"
run "rocm-smi --showdriverversion --showproductname --showvbios --showbus --showmeminfo vram --showtopo --showcomputepartition --showmemorypartition --showmaxpower --showpids"
run "cat /sys/kernel/debug/kfd/mem_limit 2>/dev/null"
run "for n in /sys/class/kfd/kfd/topology/nodes/*; do echo \"== \$n\"; cat \$n/properties | grep -E 'cpu_cores_count|simd_count|gfx_target_version|vendor_id|device_id|location_id|domain|drm_render_minor|num_xcc|hive_id|unique_id|local_mem_size|mem_banks_count|io_links_count|p2p_links_count'; cat \$n/mem_banks/*/properties 2>/dev/null | grep -E 'heap_type|size_in_bytes'; done"
run "for n in /sys/class/kfd/kfd/topology/nodes/1/io_links/* /sys/class/kfd/kfd/topology/nodes/1/p2p_links/*; do echo \"== \$n\"; cat \$n/properties 2>/dev/null; done"
run "for c in /sys/class/drm/card*/device; do [ -e \$c/mem_info_vram_total ] && echo \"\$c vram_total=\$(cat \$c/mem_info_vram_total) vis_vram_total=\$(cat \$c/mem_info_vis_vram_total) gtt_total=\$(cat \$c/mem_info_gtt_total) vbios=\$(cat \$c/vbios_version 2>/dev/null) serial=\$(cat \$c/serial_number 2>/dev/null) xgmi_hive=\$(cat \$c/xgmi_hive_info/xgmi_hive_id 2>/dev/null) compute_part=\$(cat \$c/current_compute_partition 2>/dev/null) mem_part=\$(cat \$c/current_memory_partition 2>/dev/null)\"; done"
run "dmesg | grep -iE 'amdgpu' | grep -iE 'fw|firmware|version|VRAM|xgmi|psp|smu|ras|partition|BAR|aperture' | head -120"

sec NIC_DRIVER
run "modinfo ionic | grep -vE '^alias:'"
run "modinfo ionic_rdma 2>/dev/null | grep -vE '^alias:'"
run "for p in /sys/module/ionic*/parameters/*; do printf '%s=%s\n' \$p \"\$(cat \$p 2>/dev/null)\"; done"
run "dpkg -l | grep -iE 'ionic|rdma-core|ibverbs|perftest|libibumad|rdmacm|infiniband'"
run "ls -l /usr/lib/x86_64-linux-gnu/libibverbs/ | grep -i ionic; cat /etc/libibverbs.d/*ionic* 2>/dev/null"
run "ib_write_bw --version 2>&1 | head -3"
run "rdma system show"
run "rdma link show"
run "rdma dev show -d"
run "rdma resource show"
run "show_gids"
run "ibv_devinfo -v"
run "dmesg | grep -iE 'ionic' | head -120"

sec ROUTING
run "ip -br addr"
run "ip -d link show"
run "ip rule"
run "ip route show table all | grep -vE '^(local|broadcast|anycast|multicast) '"
run "ip -6 route show table all | grep -vE '^(local|broadcast|anycast|multicast) ' | head -60"
run "ip neigh | head -60"

for ib in $(ls /sys/class/infiniband | sort -V); do
  nd=$(ls /sys/class/infiniband/$ib/device/net 2>/dev/null | head -1)
  bdf=$(basename "$(readlink -f /sys/class/infiniband/$ib/device)")
  sec "NIC $ib netdev=$nd bdf=$bdf numa=$(cat /sys/class/infiniband/$ib/device/numa_node)"
  run "for f in fw_ver hca_type hw_rev node_type node_guid sys_image_guid node_desc mad_sa_cc puec_nports; do printf '%s=%s\n' \$f \"\$(cat /sys/class/infiniband/$ib/\$f 2>/dev/null)\"; done"
  run "for f in state phys_state rate link_layer lid sm_lid cap_mask; do printf 'port1.%s=%s\n' \$f \"\$(cat /sys/class/infiniband/$ib/ports/1/\$f 2>/dev/null)\"; done"
  run "for g in /sys/class/infiniband/$ib/ports/1/gids/*; do i=\$(basename \$g); v=\$(cat \$g 2>/dev/null); [ \"\$v\" != 0000:0000:0000:0000:0000:0000:0000:0000 ] && echo \"gid\$i=\$v type=\$(cat /sys/class/infiniband/$ib/ports/1/gid_attrs/types/\$i 2>/dev/null) ndev=\$(cat /sys/class/infiniband/$ib/ports/1/gid_attrs/ndevs/\$i 2>/dev/null)\"; done"
  run "ls /sys/class/infiniband/$ib/ports/1/hw_counters 2>/dev/null | head -0; for c in /sys/class/infiniband/$ib/ports/1/hw_counters/*; do printf '%s=%s\n' \$(basename \$c) \"\$(cat \$c 2>/dev/null)\"; done"
  run "echo match_default=\$(cat /sys/class/infiniband/$ib/dcqcn/match_default); echo match_rules=[\$(cat /sys/class/infiniband/$ib/dcqcn/match_rules | tr '\n' ';')]"
  run "cd /sys/class/infiniband/$ib/dcqcn/profiles && for p in \$(ls | sort -n); do for s in roce_rp roce_np; do for f in \$(ls \$p/\$s | sort); do printf 'profile%s.%s.%s=%s\n' \$p \$s \$f \"\$(cat \$p/\$s/\$f 2>/dev/null)\"; done; done; done"
  run "cat /sys/kernel/debug/ionic/$bdf/identity; echo; cat /sys/kernel/debug/ionic/$bdf/lif0/identity; echo; cat /sys/kernel/debug/ionic/$bdf/lif0/rdma/info"
  run "cat /sys/kernel/debug/ionic/$bdf/nlifs /sys/kernel/debug/ionic/$bdf/nintrs /sys/kernel/debug/ionic/$bdf/nrxqs_per_lif /sys/kernel/debug/ionic/$bdf/ntxqs_per_lif /sys/kernel/debug/ionic/$bdf/lif0/state 2>/dev/null"
  [ -n "$nd" ] || continue
  run ethtool -i "$nd"
  run ethtool "$nd"
  run ethtool -a "$nd"
  run ethtool --show-priv-flags "$nd"
  run ethtool --show-fec "$nd"
  run ethtool -m "$nd"
  run ethtool -g "$nd"
  run ethtool -l "$nd"
  run ethtool -c "$nd"
  run ethtool -k "$nd"
  run "ip -d -s link show $nd"
  run "ip -4 addr show $nd; ip -6 addr show $nd"
  run ethtool -S "$nd"
done
sec END
run date -u
