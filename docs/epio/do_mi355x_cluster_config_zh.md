# DO MI355X 集群（8 台虚拟机）配置记录：系统、GPU、AINIC、QoS / PFC、DCQCN

2026-11-19 13:05 UTC 对 8 台机器做了一次只读采集（脚本见 [do_cluster/do_cluster_dump.sh](do_cluster/do_cluster_dump.sh)），集群即将下线，留作以后参考。本文已脱敏：不含公网 IP、主机名、序列号、MAC、GUID。

## 要点

- **8 台的软硬件配置完全一致，只有两处不同**：网卡固件（node01 / 03 / 04 / 07 是 `1.117.5-a-196`，node02 / 05 / 06 / 08 是 `1.117.5-a-147`）和 Docker 版本。
- **QoS、PFC、DCQCN 的配置 8 台也一致**（第 6 节）：虚拟机里能看到的每一项，64 张网卡都相同；宿主机上的 QoS 看不到，但各台表现一致。
- **这是 DigitalOcean 的 KVM 虚拟机**：GPU 和网卡都是 SR-IOV 的 VF 直通进来的。AINIC 固件里的 QoS、PFC、DCQCN 配置在宿主机的 PF 上，虚拟机里读不到：`dcb` 返回 Operation not supported；`nicctl` 只认 PF，拷进虚拟机运行报 `No AMD NICs detected`。
- **DCQCN 没有起作用**（第 8 节）：
  - 交换机从来没给我们的包打过 ECN 标记：64 张网卡开机以来收到的 ECN 标记、收发的 CNP 全是 0；加压到接收端一半时间都在发 PFC，也还是 0。包本身带 ECT(0)，不是 ECN 位的问题。
  - 很可能是因为流量没进交换机配了 ECN 的那个队列：DO 的网卡发出的 PFC 记在优先级 0 上；而 AINIC 的参考配置（smci355，`nicctl` 能看到）是 DSCP 26 → 优先级 3、只在优先级 3 上开 PFC，那边的 PFC 就记在优先级 3 上，ECN 和 CNP 都有几十亿（7.1 节）。
  - AINIC 上不是没配 QoS：DSCP 26 拥塞时发 PFC、不丢包，DSCP 0 丢包、不发 PFC，网卡确实按 DSCP 分了无损类和有损类，只是无损类看起来在优先级 0 上。
  - 驱动 sysfs 里的 DCQCN 参数是驱动的默认值，不代表固件的配置；固件里 DCQCN 有没有启用，虚拟机里证明不了。QP 默认也没绑定 DCQCN profile（`match_default` 为 0）。
  - 要直接确认，得请 DO 在宿主机上跑 `nicctl show qos` / `nicctl show dcqcn`，并查交换机的队列和 ECN 配置（8.4 节）。

## 1. 机器

| 节点 | 网卡固件 | Docker | rail IP | 控制网 IP |
|---|---|---|---|---|
| node01 | 1.117.5-a-196 | 29.9.3 | 10.201.\<rail\>.1 | 10.15.20.2 |
| node02 | 1.117.5-a-147 | 29.9.3 | 10.201.\<rail\>.2 | 10.15.20.3 |
| node03 | 1.117.5-a-196 | 29.9.3 | 10.201.\<rail\>.3 | 10.15.20.4 |
| node04 | 1.117.5-a-196 | 29.9.3 | 10.201.\<rail\>.4 | 10.15.20.5 |
| node05 | 1.117.5-a-147 | 29.1.3 | 10.201.\<rail\>.5 | 10.15.20.6 |
| node06 | 1.117.5-a-147 | 29.1.3 | 10.201.\<rail\>.6 | 10.15.20.7 |
| node07 | 1.117.5-a-196 | 29.1.3 | 10.201.\<rail\>.7 | 10.15.20.8 |
| node08 | 1.117.5-a-147 | 29.1.3 | 10.201.\<rail\>.8 | 10.15.20.9 |

采集时 node04 已运行约 6 天，其余约 1 周；第 6、7 节的计数都是从开机（驱动加载）开始累计的。

## 2. 虚拟机和主机

| 项 | 内容 |
|---|---|
| 虚拟化 | KVM（`systemd-detect-virt`），DMI：DigitalOcean / Droplet，BIOS "DigitalOcean 20171212"；芯片组是 QEMU 的 ICH9 / Q35，系统盘、管理网卡是 virtio；虚拟机里没有 IOMMU |
| CPU | 2 颗 AMD EPYC 9575F 64-Core；虚拟机里是 2 socket × 96 vCPU，每核 1 线程，共 192 个 vCPU |
| NUMA | 2 个节点：node 0 是 vCPU 0–95、约 1,032,093 MB，node 1 是 vCPU 96–191、约 1,032,107 MB；距离 10 / 20 |
| 内存 | MemTotal 2,113,742,552–2,113,742,620 kB（约 2016 GiB，各台差几十 kB）；大页默认 0（我们跑 UMBP 时临时设），THP `madvise`，`kernel.numa_balancing = 0` |
| 系统 | Ubuntu 24.04.4 LTS，内核 6.8.0-137-generic |
| 内核参数 | `ro console=tty1 console=ttyS0 net.ifnames=0 biosdevname=0` |
| 时钟 | systemd-timesyncd，NTP 已同步；没装 chrony |
| Docker | 29.9.3（node01–04）/ 29.1.3（node05–08） |

## 3. GPU

| 项 | 内容 |
|---|---|
| 型号 | AMD Instinct MI355X **VF**（SR-IOV），device 0x75b3，subsystem 0x75a3，gfx950，256 CU，OAM |
| 显存 | HBM3E（Micron），294,592 MB = 308,902,100,992 B，位宽 2048，峰值 8192 GB/s；`vis_vram_total` 与总量相同（BAR 覆盖全部显存） |
| 分区 | 计算 SPX，内存 NPS1（8 张卡都是） |
| 功耗上限 | 1400 W |
| PCIe | Gen5 x16（32 GT/s x16），虚拟机里 MaxPayload 128 B、MaxReadReq 128 B |
| XGMI | 8 张卡两两直连，1 跳，weight 15 |
| IFWI | 00175784（2026/01/22 17:08），part number 113-M355-01-1K1-030A |
| VBIOS | 113-M355-01-1K1-030A |
| RAS | UMC、SDMA、GFX、MMHUB、ATHUB、PCIE_BIF、HDP、XGMI_WAFL、DF、SMN、MP0、MP1、MCA、VCN、JPEG、IH、MPIO 的 ECC 都开；poison 开；process isolation 关 |
| 驱动 | amdgpu-dkms 1:6.19.14.31400000（`amdgpu-install` 31.40.0.0），amdgpu-dkms-firmware 31.40.0.0，模块版本 6.19.14.31400000 |
| 主机上的 ROCm | 7.14.0~pre3（`amdrocm-core7.14`），amd-smi 26.5.0，amdgpu-exporter 1.5.1。容器里用的是各自镜像的 ROCm，主机只提供内核驱动 |
| KFD | system 内存上限 2,030,412 MB，TTM 1,032,100 MB；GTT 总量约 1,082 GB |

GPU 固件（64 张卡相同）：

| 固件 | 版本 | 固件 | 版本 |
|---|---|---|---|
| CP_MEC1 / CP_MEC2 | 44 / 44 | TA_RAS | 1B.45.00.0A |
| RLC | 43 | TA_XGMI | 20.00.00.14 |
| SDMA0 / SDMA1 | 14 / 14 | PM | 04.86.16.03 |
| VCN | 09.10.D0.00 | PLDM_BUNDLE | 01.26.00.02 |
| RLC_RESTORE_LIST_GPM_MEM / SRM_MEM / CNTL | 4 / 4 / 4 | | |

<details>
<summary>amdgpu 的 99 个模块参数（8 台相同）</summary>

```
abmlevel=-1 agp=-1 aspm=-1 async_gfx_ring=1 audio=-1 backlight=-1 bad_page_threshold=-1 bapm=-1
cg_mask=18446744073709551615 cik_support=-1 compute_multipipe=-1 cwsr_enable=1 damageclips=-1 dc=-1
dcdebugmask=0 dcfeaturemask=2 debug_evictions=N debug_mask=0 deep_color=0 disable_cu=(null) discovery=-1
disp_priority=0 dpm=-1 emu_mode=0 enforce_isolation=-1 exp_hw_support=0 force_asic_type=-1 forcelongtraining=0
freesync_video=0 fw_load_type=-1 gartsize=4294967295 gpu_recovery=-1 gttsize=-1 halt_if_hws_hang=0
hdmi_hpd_debounce_delay_ms=0 hw_i2c=0 hws_gws_support=N hws_max_conc_proc=-1 ip_block_mask=4294967295
keep_idle_process_evicted=N lbpw=-1 lockup_timeout= max_num_of_queues_per_device=4096 mcbp=-1 mes_log_enable=0
modeset=-1 moverate=-1 msi=-1 mtype_local=-1 no_evict=0 no_queue_eviction_on_vm_fault=0 no_system_mem_limit=N
noretry=-1 num_kcq=2 pcie_gen2=-1 pcie_gen_cap=0 pcie_lane_cap=0 pcie_p2p=Y pg_mask=4294967295
ppfeaturemask=0xfff7bfff priv_cp_queues=0 ptl=-1 queue_preemption_timeout_ms=9000 ras_enable=-1
ras_mask=4294967295 rebar=-1 reset_method=-1 runpm=-1 sched_hw_submission=2 sched_jobs=32 sched_policy=0
sdma_phase_quantum=32 seamless=-1 send_sigterm=0 sg_display=-1 si_support=-1 smu_memory_pool_size=0
smu_pptable_id=-1 svm_default_granularity=9 timeout_fatal_disable=N timeout_period=0 tmz=-1 umsch_mm=0
umsch_mm_fwlog=0 uni_mes=1 use_xgmi_p2p=1 user_partt_mode=4294967294 user_queue=-1 vcnfw_log=0
virtual_display=(null) vis_vramlimit=0 visualconfirm=0 vm_block_size=-1 vm_fault_stop=0 vm_fragment_size=-1
vm_size=-1 vm_update_mode=-1 vramlimit=-1 wbrf=-1
```

</details>

## 4. 网卡（AMD Pensando AINIC，SR-IOV VF）

| 项 | 内容 |
|---|---|
| 设备 | AMD Pensando DSC Ethernet Controller **VF**，vendor 0x1dd8，part 4099，debugfs 里 asic_type 0x4；每台 8 张 |
| 固件 | `1.117.5-a-196`（node01 / 03 / 04 / 07）、`1.117.5-a-147`（node02 / 05 / 06 / 08）；同一台的 8 张卡相同 |
| 驱动 | ionic / ionic_rdma 25.12.15.001（`ionic-dkms`、`ionic-common` 包，DKMS 编译）；ionic_rdma 依赖 `mlx_compat` |
| 用户态 | libionic1 54.0-184（`libionic-rdmav34.so` → `libionic.so.1.1.54.0-184`）；rdma-core / libibverbs / librdmacm 2410mlnx54（MLNX OFED 24.10 的用户态）；perftest 24.10.0（6.23） |
| 链路 | 400GBASE-DR4，单端口，自协商关，FEC RS |
| 光模块 | ColorChip CTF4XDR4CS2-11，CMIS rev 5.1，MPO，1310 nm DFB，power class 8（8 W） |
| MTU | netdev 4200（最大 9144）；RoCE `active_mtu` 4096 |
| PCIe | Gen5 x16（32 GT/s x16），虚拟机里 MaxPayload 128 B、MaxReadReq 512 B |
| NUMA | ionic_0–3 在 NUMA 0，ionic_4–7 在 NUMA 1 |
| 和 GPU 的关系 | GPU i（BDF 83/8b/93/9b/a3/ab/b3/bb:00.0）和网卡 i（84/8c/94/9c/a4/ac/b4/bc:00.0）挂在同一个 PCIe 交换节点下 |
| 网卡名 | ionic_i ↔ eth(i+2)（eth2–eth9） |
| 以太网队列 | 2 个 combined 通道；ring 1024 / 1024（最大 RX 16384 / TX 8192）；RX 自适应中断合并，rx-usecs 8 |
| 私有开关 | `rdma-sniffer` off，`cmb-rings` off |
| pause 设置（`ethtool -a`） | Autonegotiate off，RX on，TX on |
| RDMA 能力（`ibv_devinfo -v`） | max_qp 4096，max_qp_wr 65535，max_sge 8，max_cq 3968，max_cqe 65435，max_mr 16384，max_pd 1024，max_qp_rd_atom / init_rd_atom 16，atomic ATOMIC_GLOB，max_srq 256，max_srq_wr 65535，max_msg_sz 2 GB，active_width 4X × 100 Gb/s；8 个 DCQCN profile |
| 模块参数 | ionic：`devcmd_timeout=5 rx_copybreak=256 rx_fill_threshold=64 tx_budget=256 asic_addr_len=52 affinity_mask_override=0`；ionic_rdma：`admincmd_timeout=2 aq_dbell=Y qp_dbell=Y default_ats=N` |

## 5. 网络

- **rail-only，8 条 rail**：网卡 i（eth(i+2)）在 rail i，网段 10.201.i.0/24，node N 的地址是 10.201.i.N；8 个网段都是直连，没有跨 rail 的路由，rail 之间不通。
- **GID**：index 0 是 RoCEv2 的 IPv6 link-local，index 1 是 RoCEv2 的 IPv4（`::ffff:10.201.i.N`）。我们的 `MORI_IB_GID_INDEX=1`、`NCCL_IB_GID_INDEX=1` 用的就是它。
- **控制网 eth1**：10.15.20.0/24（node N 是 .N+1）；10.0.0.0/8、172.16.0.0/12、192.168.0.0/16 经 10.15.20.1 走 eth1（MTU 1500）。torch / gloo / NCCL 的 socket 都走它（`*_SOCKET_IFNAME=eth1`）。
- **eth0**：公网地址加一个 DO 内网地址，默认路由在这里。
- 没有策略路由（`ip rule` 只有默认三条）。sysctl：`rp_filter=2`，`arp_ignore=2`，`arp_announce=1`，`ip_forward=1`，`net.core.rmem/wmem_max = 212992`。

## 6. 8 台的 QoS、PFC、DCQCN 配置对比

虚拟机里能看到的 QoS、PFC、DCQCN 配置，**8 台（64 张网卡）完全一致**，唯一的差别是网卡固件。宿主机 PF 上的 QoS 配置看不到，从行为看各台也一致。2026-11-19 14:50 UTC 用 [do_cluster/do_qos_dump.sh](do_cluster/do_qos_dump.sh) 又在 8 台上采了一遍，结果如下。

| 项 | 怎么看 | 64 张网卡的情况 |
|---|---|---|
| DCB：PFC、ETS、DSCP 映射、缓冲、限速、DCBX | `dcb pfc / ets / app / buffer / maxrate / dcbx show` | 都返回 Operation not supported：VF 上读不到，配置在宿主机的 PF 上 |
| 链路 pause 设置 | `ethtool -a` | 相同：Autonegotiate off，RX on，TX on |
| 私有开关 | `ethtool --show-priv-flags` | 相同：`rdma-sniffer` off，`cmb-rings` off |
| MTU | `/sys/class/net/*/mtu`、`ibv_devinfo` | 相同：4200；RoCE 4096 |
| FEC | `ethtool --show-fec` | 相同：RS |
| DCQCN 8 个 profile 的参数 | `/sys/class/infiniband/ionic_*/dcqcn/profiles/*/roce_{rp,np}/*` | 相同：全部参数拼起来的 md5 都是 `2a08bc7db987`；8 个 profile 之间也相同，取值见 8.1 |
| DCQCN 绑定 | `dcqcn/match_default`、`dcqcn/match_rules` | 相同：0、空，QP 都不做 DCQCN |
| 暂停帧的优先级 | `ethtool -S` 的 `tx_pripause_<p>_1us_count`、`rx_pripause_<p>_1us_count` | 相同：发出的暂停只在优先级 0；收到过暂停的 48 张网卡，都是优先级 0 和 3 同时、累计时长完全相等；另外 16 张开机以来没收到过暂停。配置已知的 smci355 上，网卡发出的暂停记在优先级 3（7.1 节） |
| ECN 标记、CNP | `hw_counters` 的 `rx_rdma_ecn_pkts`、`rx_rdma_cnp_pkts`、`tx_rdma_cnp_pkts` | 开机以来都是 0 |
| RoCE 包的 ECN 位 | debugfs 里每个 QP 的 `hdr_ip4_tos` | 两种固件都置 ECT(0)：node05 / 06（a-147）上用 TC 104 建的 QP 是 TOS 106；采集时 node05–08 上正在跑的 2,304 个 QP（TC 0）都是 TOS 2，其中 node07 是 a-196 |
| 网卡固件 | `ethtool -i` | 两种：`1.117.5-a-196`（node01 / 03 / 04 / 07）、`1.117.5-a-147`（node02 / 05 / 06 / 08） |
| 固件里的 QoS、DCQCN | `nicctl show qos`、`nicctl show dcqcn` | 虚拟机里没装；把 `nicctl` 拷进 node05 试过（2026-11-20），每条命令都报 `No AMD NICs detected`：它只认 PF，VF 上用不了 |
| 宿主机侧：DSCP → 优先级、PFC 优先级、无损队列、DCQCN 开关 | 要在宿主机上用 `nicctl` | 看不到。从行为看：DSCP 26 是无损类，在 node01–04（K3.1 端到端，有 PFC、几乎不重传）和 node05 / 06（8.3 的测试，PFC 很重、0 重传）上验证过；DSCP 0 是有损类，在 node01–04（K3.1 的 TC 0 对照，2% 重传）上验证过；node07 / 08 没有单独验证 |

## 7. QoS 和 PFC

**虚拟机里看不到 QoS 配置**：VF 上 `dcb pfc / ets / app / buffer` 都返回 Operation not supported；`nicctl` 没装，拷进来运行报 `No AMD NICs detected`。下面是从计数和测试推出来的：

- **DSCP 26（traffic class 104）是无损类**：接收端一跟不上就发 PFC，全程不丢包。**DSCP 0（traffic class 0）是有损类**：接收端跟不上时直接丢包，靠重传。mori 的参考配置是 DSCP 26 / 优先级 3。
- **暂停帧的优先级**：
  - 网卡发给交换机的暂停帧，计数都记在**优先级 0**（`tx_pripause_0_1us_count`）。
  - 交换机发给网卡的暂停帧，在**优先级 0 和 3 上同时计数**，两个数完全相等，也就是交换机的暂停帧同时置了 0 和 3 两位。所以无损类触发的暂停也会连带停掉 DSCP 0 的流量（之前的 v1 测试测到过这种连带）。
- **RoCE 包的 ECN 位**：网卡把 traffic class 的低 2 位置成 ECT(0)。我们用 traffic class 104 建的 QP，发出去的 IP TOS 是 106；同时在跑的别人的任务（traffic class 用默认的 0）发出去是 2（ionic debugfs 里每个 QP 的 `hdr_ip4_tos`）。
- **开机以来的计数**（每台 8 张网卡合计；计数从驱动加载开始，主要来自这期间各人的测试）：

| 节点 | 网卡发出的暂停帧 | 发出暂停的累计时长（优先级 0） | 网卡收到的暂停帧 | 收到暂停的累计时长（优先级 0 / 3） | RDMA 重传包 | ACK 超时 | 收到 ECN 标记 | 收 / 发 CNP |
|---|---|---|---|---|---|---|---|---|
| node01 | 12,700 | 0 s | 38,960,379 | 121 / 121 s | 23,438,017,114 | 467,592 | 0 | 0 / 0 |
| node02 | 128,530,035 | 223 s | 504,959 | 1 / 1 s | 3,848,347,736 | 586,691 | 0 | 0 / 0 |
| node03 | 428,835 | 1 s | 32,329,586 | 108 / 108 s | 16,907,782,679 | 544,418 | 0 | 0 / 0 |
| node04 | 74,482,607 | 181 s | 0 | 0 / 0 s | 1,285,275,618 | 394,119 | 0 | 0 / 0 |
| node05 | 49,209,049 | 58 s | 60,196 | 0 / 0 s | 1,608,006,352 | 7,324 | 0 | 0 / 0 |
| node06 | 24,952,129 | 26 s | 37,074 | 0 / 0 s | 1,581,907,179 | 3,955 | 0 | 0 / 0 |
| node07 | 6,270,594 | 12 s | 0 | 0 / 0 s | 83,343,956 | 3,459 | 0 | 0 / 0 |
| node08 | 46,929,909 | 50 s | 175,202 | 1 / 1 s | 111,946,021 | 104,779 | 0 | 0 / 0 |

计数来源：暂停帧和暂停时长来自 `ethtool -S`（`frames_tx_pripause`、`tx_pripause_<p>_1us_count` 等，单位 1 µs），RDMA 相关的来自 `/sys/class/infiniband/ionic_*/ports/1/hw_counters/`。

### 7.1 和配置已知的 AINIC 对比（smci355）

smci355 是我们另一套 MI355X + AINIC 的裸机集群，能用 `nicctl` 直接看网卡固件里的配置（2026-11-20 在其中一台上看的，7 张正常的卡配置相同）。同样的计数放在一起比：

| 项 | smci355（裸机 PF，固件 1.117.5-a-77） | DO（VF，固件 a-147 / a-196） |
|---|---|---|
| 分类方式（`nicctl show qos`） | DSCP | 看不到 |
| DSCP → 优先级 | 26 → 3，48 → 6，其余 → 0 | 看不到 |
| PFC 不丢包的优先级 | 3（PFC priority bitmap 0x8） | 看不到 |
| 调度 | 优先级 0 DWRR 10%，3 DWRR 90%，6 严格优先级、限速 10 Gbps | 看不到 |
| 固件的 DCQCN（`nicctl show dcqcn`） | profile 1 Enabled；CNP 用 DSCP 48 | 看不到 |
| 网卡发出的 PFC 记在 | `tx_pripause_3`（和配置一致） | `tx_pripause_0` |
| 每张卡开机以来收到的 ECN 标记 | 约 82 亿–84 亿 | 0 |
| 每张卡发出 / 收到的 CNP | 约 38 亿–40 亿 / 0.4 亿–0.7 亿 | 0 / 0 |
| 驱动 sysfs 的 DCQCN | `match_default` 0，profile 1 的 `cnp_dscp` 46 | 一样 |

说明了什么：
- **计数器是可信的**：smci355 配的是优先级 3，PFC 就记在 `tx_pripause_3`。DO 记在优先级 0，说明 DO 的 AINIC 把 DSCP 26 的无损类放在了优先级 0 上，和参考配置不同（前提是 VF 上的计数和 PF 含义一样）。
- **AINIC 上不是没配**：DO 上 DSCP 26 不丢包、DSCP 0 丢包，网卡确实按 DSCP 分了无损类和有损类，只是无损类的优先级不一样。
- **交换机侧**：DO 的交换机发来的暂停同时带优先级 0 和 3，我们的流量在交换机上也被当成能暂停的流量，没有丢包。如果交换机只在优先级 3 的队列上开了 ECN，而我们的流量按优先级 0 处理，就进不了那个队列，这和 ECN 一直是 0 对得上。
- **驱动 sysfs 里的 DCQCN 参数不代表固件配置**：两边的 sysfs 一样，但 smci355 的固件里 CNP 用的是 DSCP 48，sysfs 里写的却是 46。

## 8. DCQCN 的问题

**问题**：这套集群上 DCQCN 从来没有工作过，无损类（DSCP 26）只靠 PFC 做流控。接收端一跟不上就发 PFC，交换机再暂停发送端；交换机发出的暂停帧同时置了优先级 0 和 3，还会连带暂停 DSCP 0 的流量。所以 incast 或者接收端变慢时，会出现大面积、长时间的暂停（8.3 的测试里，接收端一半以上的时间在暂停），发送端却从来不会主动降速。

### 8.1 网卡上的配置

驱动在 `/sys/class/infiniband/ionic_*/dcqcn/` 下给出 8 个 profile（`profiles/1`–`profiles/8`），每个分发送端（reaction point，`roce_rp`）和接收端（notification point，`roce_np`）两组参数。**64 张网卡、8 个 profile 的参数完全一样**：

| 参数 | 值 | 参数 | 值 |
|---|---|---|---|
| `roce_rp/ai_rate` | 160 | `roce_rp/min_rate` | 1 |
| `roce_rp/hai_rate` | 300 | `roce_rp/qp_rate` | 100000 |
| `roce_rp/gd` | 11 | `roce_rp/rate_reduce_monitor_period` | 1 |
| `roce_rp/min_dec_fac` | 50 | `roce_rp/rate_to_set_on_first_cnp` | 0 |
| `roce_rp/initial_alpha_value` | 64 | `roce_rp/threshold` | 1 |
| `roce_rp/byte_reset` | 431068 | `roce_rp/time_reset` | 1 |
| `roce_rp/token_bucket_size` | 800000 | `roce_rp/clamp_tgt_rate` | 0 |
| `roce_rp/dce_tcp_g` | 512 | `roce_rp/clamp_tgt_rate_ati` | 1 |
| `roce_rp/dce_tcp_rtt` | 1 | | |
| `roce_np/cnp_dscp` | 46 | `roce_np/icnp_802p_prio` | 6 |

**这些是驱动里的值，不代表固件的配置**：虚拟机里能看到的只有驱动 sysfs 里的这 8 个 profile。固件里 DCQCN 有没有启用、用哪个 profile、CNP 用哪个 DSCP，要在宿主机上用 `nicctl show dcqcn` 看，VF 上 `nicctl` 用不了。对照：smci355 上 `nicctl show dcqcn` 显示 profile 1 Enabled，参数和上表一样（AI 160、HAI 300、byte count 431068、G 512、初始 alpha 64、token bucket 800000 等），但 CNP 用的是 DSCP 48，它的驱动 sysfs 里也写着 46。

**QP 默认不绑定任何 profile**：`match_default` 是 0，`match_rules` 为空，所以新建的 QP 都用 profile 0，即不做 DCQCN。每个 QP 实际用的 profile 可以在 debugfs 里看：`/sys/kernel/debug/ionic/<bdf>/lif0/rdma/qp/<qpn>/info` 的 `dcqcn_profile`。

### 8.2 为什么没生效

DCQCN 的链路是：交换机队列堆起来时给包打 ECN 标记，接收端网卡收到标记后给发送端回 CNP，发送端收到 CNP 才降速。这里卡在了第一步：

1. **交换机不打 ECN 标记**：64 张网卡开机以来收到的 ECN 标记（`rx_rdma_ecn_pkts`）全是 0，收发的 CNP 也全是 0；同期每台都有几百万到上亿个暂停帧，队列显然堆起来过。8.3 的测试里专门造了拥塞，也是 0。
   - 不是包不带 ECT 的问题：网卡发出的 RoCE 包都带 ECT(0)，两种固件都是（第 6、7 节）。
   - 剩下三种可能，只能在交换机上确认：交换机根本没开 ECN；ECN 只开在优先级 3 的队列上，而我们的流量按优先级 0 处理，进不了那个队列（7.1 节的对比支持这一种）；ECN 的门限比 PFC 的门限还高，PFC 总是先触发。
2. **QP 没绑定 DCQCN profile**（8.1）：就算有 CNP，发送端也不会降速。这一层在虚拟机里就能改。

### 8.3 测试

在 node05 → node06 上测（另外 6 台在跑别的任务，没用）。

- **怎么造拥塞**：node05 的 rail 0–3 用 `ib_write_bw`（1 MiB 消息，每条 rail 4 个 QP，20 s）写进 node06 的主机内存。接收缓冲放在**另一个 NUMA**（网卡在 NUMA 0，缓冲在 NUMA 1），网卡往内存写不动，只能发 PFC 让交换机停下，交换机发往 node06 的队列随之堆积。
- **对比**：traffic class 104 / 106（106 显式带 ECT(0)）× `match_default` 0 / 1（两台的 ionic_0–3 都改，测完改回 0）。

| 配置 | 每条 rail 带宽 | 接收端发出暂停帧（暂停累计） | 发送端收到暂停帧 | 接收端收到 ECN 标记 | 接收端发 CNP / 发送端收 CNP | QP 的 profile |
|---|---|---|---|---|---|---|
| TC 104，`match_default` 0（现状） | 230–245 Gb/s | 2,789 万（42 s，4 张卡合计 20 s 内） | 1,498 万 | 0 | 0 / 0 | 0 |
| TC 106，`match_default` 0 | 232–246 Gb/s | 2,891 万（42 s） | 1,501 万 | 0 | 0 / 0 | 0 |
| TC 104，`match_default` 1 | 281–384 Gb/s（rail 3 没连上，只有 3 条） | 1,628 万（17 s） | 728 万 | 0 | 0 / 0 | 1 |
| TC 106，`match_default` 1 | 218–231 Gb/s | 2,721 万（44 s） | 1,435 万 | 0 | 0 / 0 | 1 |

- 接收端 4 张卡合计 20 s 里暂停了 42–44 s，即每张卡一半以上的时间在暂停，交换机队列已经堆满到要暂停发送端，但**没有一个包被打上 ECN 标记**。
- traffic class 104 和 106 发出去的 IP TOS 都是 106（debugfs `hdr_ip4_tos`），说明 ECT(0) 是网卡自己加的，不是 ECN 位的问题。
- `match_default` 改成 1 以后，新建的 QP 都变成 `dcqcn_profile 1`，绑定在虚拟机里就能生效；但没有 CNP，降速无从触发，带宽和暂停没有变化（第 3 行只有 3 条 rail 在跑，跨 NUMA 的总带宽由 3 张卡分，所以每条更高）。

### 8.4 怎么让 DCQCN 生效

1. **先确认配置（需要 DO 的宿主机和网络侧）**：
   - 宿主机，每张 AINIC 的 PF 上：`nicctl show qos`，看 Classification type、DSCP-to-priority、PFC no-drop priorities、Scheduling；`nicctl show dcqcn`，看 Status 和 CNP 用的 DSCP。参考配置（smci355，7.1 节）：按 DSCP 分类，DSCP 26 → 优先级 3、DSCP 48 → 优先级 6，PFC 不丢包的优先级是 3，DCQCN profile 1 Enabled、CNP 用 DSCP 48。如果 DO 的无损类确实在优先级 0，要么改成和参考配置一致，要么让交换机按同样的优先级处理。
   - 交换机：DSCP → 队列的映射、PFC 开在哪些优先级、每个队列的 WRED / ECN 配置；拥塞时看每个队列的 ECN 标记和 PFC 计数，确认 DSCP 26 的流量进了哪个队列。另外查一下为什么交换机发出的暂停同时置了优先级 0 和 3。
2. **交换机打开 ECN**：在承载 DSCP 26 的无损队列上打开 ECN 标记（WRED + ECN），门限设在 PFC 的 Xoff 之下，让 ECN 先于 PFC 触发；CNP 的 DSCP（参考配置是 48，对应优先级 6）要进一个高优先级队列。
3. **虚拟机**：在应用建 QP 之前把 QP 绑到一个 profile。
   - 全部绑：`for d in /sys/class/infiniband/ionic_*; do echo 1 > $d/dcqcn/match_default; done`
   - 按优先级绑：写 `match_rules`，例如 `prio 0x8 1`（优先级掩码 0x8 即优先级 3 → profile 1）。
   - 只对之后新建的 QP 生效，已有的 QP 不变；改动是全机的，会影响这台机器上其他人新建的 QP。
   - 用 debugfs 里 QP 的 `dcqcn_profile` 确认绑上了。
4. **验证**：拥塞时接收端 `rx_rdma_ecn_pkts`、`tx_rdma_cnp_pkts` 和发送端 `rx_rdma_cnp_pkts` 都要增长，PFC 暂停时长应该明显下降；每个 QP 的 DCQCN 状态（当前限速）要在宿主机上用 `nicctl show rdma queue-pair --detail` 看。
5. **还要确认固件真的会降速**：另一套裸机集群（smci355，固件 1.117.5-a-45 / a-77）上，交换机打了 ECN、发送端也收到了 CNP、QP 也绑了 profile，但 QP 的 DCQCN 状态一直是 0，没有降速。这里的 a-147 / a-196 因为没有 ECN 标记，还没法验证。

## 9. 采集方法和原始数据

- QoS / DCQCN 对比脚本 [do_cluster/do_qos_dump.sh](do_cluster/do_qos_dump.sh)：只读，每张网卡每项一行，方便逐台对比（第 6 节）。
- 采集脚本 [do_cluster/do_cluster_dump.sh](do_cluster/do_cluster_dump.sh)：只读，在每台机器上以 root 运行，输出约 1.6 万行；内容包括系统、DMI、CPU / NUMA / 内存、`lspci -vvv`、`amd-smi static / firmware / partition / topology / xgmi / metric`、`rocm-smi`、KFD 拓扑、amdgpu / ionic 的 modinfo 和模块参数、软件包版本、`ibv_devinfo -v`、路由、每张网卡的 sysfs / debugfs / DCQCN 全部参数 / `ethtool`（`-i -a -m -g -l -c -k -S` 等）。
- DCQCN 测试脚本 [do_cluster/dcqcn_test.sh](do_cluster/dcqcn_test.sh)：用 `NODES_SH=` 指定一个定义了 `PUB`（ssh 地址）、`SUF`（rail IP 的末位）、`SSH_OPTS` 的文件；测完把 `match_default` 改回 0。
- 8 台的完整原始输出（未脱敏）和 DCQCN 测试的原始计数在我们本地的任务目录 `k31-agentx-pd-ep-io-umbp-20261117/cluster_config/`。
