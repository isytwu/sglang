# 复现指南：smci355 上 K3.1 vLLM 的 IO 对 EP 影响（给接手的同事和 agent）

结论文档：[k31_smci355_vllm_ep_io_zh.md](k31_smci355_vllm_ep_io_zh.md)。脚本：[smci355_vllm/](smci355_vllm/)，下文称“工具包”。

这份指南写给接手的同事，以及同事用的 agent：照着做，能在 smci355 上复现结论文档里的几个 case，再在这个基础上调配置，看能不能缓解 IO 对 EP 的影响。
所有 case 都是在 smci355 上跑出来的，同事在 smci355 上可以直接照做。每个 case 都给出了要多少节点、要跑哪些命令、预期结果是什么，以及结果在哪里看。

| case | 内容 | 节点 | 一次约多久 | 能否复现 IO 拖慢 EP | 预期 |
|---|---|---|---|---|---|
| A | **只有 decode，加注入**（EP16，每个 rank 128 条，qlen 8） | 3 | 50–60 分钟 | **能，最明显、最快** | 16 MB 一批的写或读慢约 66%，每卡 1 GB/s 也有 +15–21%；TC 0 或 1 MB 一批没有影响 |
| B | **真实 PD**（MoRIIO READ）加高批量合成负载，再加注入 | 4（1P）或 5（2P） | 每段 30–50 分钟 | **注入部分能**；自然 KV 部分不能 | 注入：16 MB 一批慢 62–70%，TC 0 或 1 MB 一批没有影响，和 A 一样。自然 KV：TC 104 对 TC 0 差 ±0.6% |
| C | 真实 PD 加 AgentX（并发 128） | 5 | 60–80 分钟 | **不能** | decode 每个 rank 只有 10–40 条在解码，EP 占比太小，自然 KV 和注入的影响都在约 14% 的噪声以内 |
| D | 单测：只跑 mori EP 内核，加 IO | 3 | 20–50 分钟 | **能，但只在内核层面** | 大批量 READ 让 EP 内核慢 10–40 倍；KV 形状的 IO 只慢 5–20%，会低估端到端的影响 |
| E | **EP32（Moonshot 那个 case），只有 decode，加注入**（每个 rank 128 条，全局 512，qlen 8） | 5 | 60–90 分钟 | **能** | 1 台 decode 线速读 +28%，4 台各收每股 5–10 GB/s 的突发 +25–27%，线速写进 1 台 +18%；TC 0 没有影响，拆成 1–2 MB 一批还剩 +1–7% |
| E2 | EP32 真实 PD（MoRIIO READ）加合成负载 512 条，再加注入；AgentX | 6 | 每段 30–55 分钟 | **注入部分能**；自然 KV 不能；AgentX 那次挂死了 | 注入：1 台线速读 +25%，每股 10 GB/s +22%，TC 0 +0.3%。自然 KV：TC 104 对 TC 0 差 0.06% |

**推荐顺序：**
1. **先跑 A。** 它能确认环境和复现都正常，之后试缓解配置也主要用它：信号大（+66%），每个窗口 3 分钟，一次作业能比较十来种配置。
2. **有效的办法再到 B 上验证。** B 走的是真实 PD 的 KV 路径，能确认这个配置对 KV 传输本身没有副作用，比如 TTFT 和吞吐。
   要确认在 Moonshot 的配置上也有效，再到 E 上跑一次（5 台，信号是 +18–28%，比 A 小）。
3. **D 用来分析内核层面的原因。** 比如 READ 的在途量、主机内存这类端到端不好单独控制的因素。
4. **C 不能用来复现。** 它只能说明在真实的 AgentX 负载下影响很小，不适合拿来调配置。

## 0. 准备（只做一次）

### 0.1 环境

- **集群：** smci355，Slurm 分区 `Compute-Group01`，账号 `default`。
  - 可调度节点：n08-21/25/29/33、n09-21/29；n06-21 是登录节点，n09-25/33 是 drain 状态。
  - 每台 8 × MI355X、8 × ionic 400G，rail-only。
  - 有一个插件把作业时长限制在 4 小时以内。
- **镜像：** `/apps/yutongwu/store-vllm/k31-umbp-20261112/images/yukiozzz_vllm-openai-rocm_k31-ms-1114.sqsh`。所有人可读，直接用，不用复制。
- **模型：**
  - 主模型：节点本地 `/data/models/Kimi-K3.1-Pro`，没有就用 NFS 上的 `/perf_apps/data/models/Kimi-K3.1-Pro`；
  - 草稿模型：`/perf_apps/data/models/Kimi-K3.1-Pro-DSpark-preview`。
- **QoS（6 台一致）：** DSCP 26（TC 104）是唯一的无损类（PFC，DWRR 90%），其他都是优先级 0（有损，DWRR 10%）。
  EP（`MORI_RDMA_TC`）、KV（`MORI_IO_TC`）、注入的 IO（`IOTC`）默认都是 104。

### 0.2 复制工具包，替换路径

脚本里写死了工作目录和容器名，复制后要换成自己的。目录要放在 `/apps` 下，因为容器只挂载了 `/apps`。

```bash
K=/apps/$USER/k31-epio-repro                       # 自己的工作目录
git clone --depth 1 -b k31-epio-smci355-vllm https://github.com/isytwu/sglang.git /tmp/epio-doc
mkdir -p $K && cp -r /tmp/epio-doc/docs/epio/smci355_vllm/* $K/ && mkdir -p $K/logs $K/results
grep -rl /apps/yutongwu/store-vllm/k31-vllm-ep-io-smci355-20261119 $K | xargs sed -i "s#/apps/yutongwu/store-vllm/k31-vllm-ep-io-smci355-20261119#$K#g"
grep -rl yutongwu-epio $K | xargs sed -i "s#yutongwu-epio#$USER-epio#g"   # pyxis 容器名
sed -i "s#yutongwu-k31-epio#$USER-k31-epio#" $K/tools/holder.sbatch
```

### 0.3 AgentX 客户端（只有 case C 和 pd32a 要用）

```bash
R=$K/agentx; mkdir -p $R/uv/bin $R/hf_home/hub
curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=$R/uv/bin UV_NO_MODIFY_PATH=1 sh
export UV_CACHE_DIR=$R/uv-cache UV_PYTHON_INSTALL_DIR=$R/uv-python
$R/uv/bin/uv venv --python 3.11 $R/venv
printf 'agentx-harness==1.0.13\ndatasets>=4.7.0\n' > $R/requirements.txt
$R/uv/bin/uv pip install --python $R/venv/bin/python --no-config -r $R/requirements.txt
# 轨迹数据集：共享缓存里有一份，也可以让第一次运行在线下载（登录节点和计算节点都能上网）
cp -a /apps/InferenceX_CI/hf-hub-cache/datasets--semianalysisai--cc-traces-weka-062126 $R/hf_home/hub/ 2>/dev/null || true
```

tokenizer 用 `/perf_apps/data/models/Kimi-K3.1-Pro`，已经写在 `pd.sh` 里。HF 缓存里那份 Kimi tokenizer 缺 `tiktoken.model`，不能用。

### 0.4 怎么跑：持有作业加启动器

所有测试都按同一个模式来：
1. 提交一个只占节点的持有作业（`tools/holder.sbatch`，里面只有 sleep）；
2. 在登录节点上起启动器 `tools/launch.sh`。它等作业开始，按顺序跑“阶段”，所有步骤都通过 `srun --overlap` 进作业；
3. 跑完以后，启动器清理节点并取消作业。

```bash
cd $K
J=$(sbatch --parsable --nodes=3 --time=04:00:00 tools/holder.sbatch)     # 节点数按 case 定
echo "setup e1" > logs/phases_$J.txt                                      # 要跑的阶段，按顺序
cp tools/launch.sh logs/run_launch_$J.sh                                  # 从副本运行，之后改原脚本不影响它
setsid nohup bash logs/run_launch_$J.sh $J >> logs/nohup_launch_$J.out 2>&1 < /dev/null &
tail -f logs/launch_$J.log
```

- **角色：** 节点按名字排序，最后一台是 C（负载、代理、注入），前面的是 decode（以及 PD 时的 prefill）。第一台是 A，第二台是 B。
- **阶段：** `setup` 在每台节点上创建容器，并检查驱动、固件、QoS 和 GPU 与网卡的对应关系，**每个作业都要先跑它**。
  - `build` 和 `warm` 只有单测需要，端到端不用跑。
  - 每个阶段的含义写在 `tools/launch.sh` 的开头。
- **按运行覆盖设置：** 运行名是 `<阶段>_<作业号>`，比如 `e1_7001`。下面两个文件放在 `logs/` 里，有就生效：
  - `windows_<运行名>.txt`：注入窗口列表；
  - `env_<运行名>.txt`：在启动脚本开头 source，例如 `export RAIL_ALIGN=1`。
- **改后面的阶段：** `logs/phases_<作业号>.txt` 在每个阶段开始前才读，作业运行中直接改这个文件就行。
  - 想暂停：在登录节点起一个 `sleep 7200 &`，然后把 `after:<它的 pid>:` 写到阶段文件最前面，启动器会等这个进程退出。
- **结果：** `results/e2e/<运行名>/`。阶段结束后，启动器会自动生成 `summary.md`。

## A. EP16，只有 decode，加注入（最核心的 case）

**用途：** 最快看到 IO 拖慢 EP，以及验证缓解办法。之后调配置主要用这个 case。

- **节点：** 3 台。A 和 B 是 decode，DP2 × TP8 = EP16，每台一个 DP rank；C 跑负载和注入。
- **负载：** `e2e/loadgen.py`，每个 rank 128 条流。随机 token，输入 1024、输出 8000 ± 50%，忽略 EOS，所以 batch 一直是满的。
  每个 EP rank 每次 dispatch 128 个 token，每步约 300 ms。
- **注入：** `mb/io_inject.py`（MoRI IO，KV 形状：72 KiB 一段，每批 N MB，GPU i ↔ GPU i，8 张网卡都用），从 C 往两台 decode 写或读。
  每个窗口 120 秒、间隔 60 秒；最后一个窗口固定是 base（无 IO）。

```bash
J=$(sbatch --parsable --nodes=3 --time=04:00:00 tools/holder.sbatch)
echo "setup e1" > logs/phases_$J.txt
echo "export RAIL_ALIGN=1" > logs/env_e1_$J.txt                           # 一定要设，见“坑”的第 1 条
echo "base IWk IWkT0 IWk1g IWk10c4 IWkS IRk IRkT0 IWk1" > logs/windows_e1_$J.txt
cp tools/launch.sh logs/run_launch_$J.sh && setsid nohup bash logs/run_launch_$J.sh $J >> logs/nohup_launch_$J.out 2>&1 < /dev/null &
```

**窗口一览**（定义在 `e2e/windows.sh`，可以自己加）：

| 名字 | IO |
|---|---|
| `base` | 无 |
| `IWk` / `IRk` | 写 / 读，16 MB 一批，满速（C 的每张网卡分给两台 decode） |
| `IWk1g` `IWk2` `IWk5` `IWk10` | 写，16 MB 一批，每卡 1 / 2 / 5 / 10 GB/s |
| `IWk10c2` `IWk10c4` `IWk10c8` | 写，每卡 10 GB/s，2 / 4 / 8 MB 一批 |
| `IWkS` `IWk10S` `IWk5S` `IWk1gS` | 写，1 MB 一批：连续发 / 每卡 10、5、1 GB/s |
| `IWkSi4` `IWkSi16` | 1 MB 一批，同时 4 / 16 批在途 |
| `IWk1gB` | 每卡 1 GB/s，64 MB 一批 |
| `IWkR16` `IWkR64` | 每张网卡每秒一次 16 / 64 MB 的突发（模拟每秒到一个请求的 KV） |
| `IWkT0` `IRkT0` | 同 IWk / IRk，IO 放在 TC 0 |
| `IWk1` `IRk1` | 只打第一台 decode（C 的 8 张网卡全给它） |
| `IWhR` | 写进 decode 另一个 NUMA 节点的主机内存 |
| `IRb` `IWb`（`IRb1` `IWb1`） | mori IO benchmark 读 / 写，32 KB × 8192 一批；`IWb` 固定排在最后 |

窗口的速率、单批大小、在途批数，在 `window()` 开头的 `case` 里用 `per`（周期，ms）、`mb`（每批 MB）、`inf`（在途批数）来设，加新窗口照着写就行。

**预期**（e1ra_6866、e1t1_6866、e1_6881，base 约 302 ms）：

| 窗口 | 步时变化 | decode 收到的 CNP（每秒） |
|---|---|---|
| IWk / IRk | +66% | 约 2.5 万 |
| IWk1g | +15–21% | 约 3,300 |
| IWk10c4 | +14% | 约 1.4 万 |
| IWk1 | +44%（没收到 IO 的那台 decode 每秒收到 6.4 万个 CNP） | — |
| IWkT0 / IRkT0 | +0.6% / +0.9% | 0 |
| IWkS / IWk1gS | +0.4% / +0.1% | 0 |

**看结果：**
- `results/e2e/e1_<J>/summary.md` 里是每个窗口的步时（客户端 token 间隔的中位数、p90、p99）、网卡速率，以及 profiler 里的 EP 内核时间。
- `snaps.jsonl` 是每个窗口开始和结束时的 RDMA 计数器（CNP、ECN、重传、PFC），读法见第 F.2 节。
- `fab_<node>.tsv` 是每秒的网卡字节、PFC、CNP 和 ECN 标记包。

## B. EP16 真实 PD 加高批量合成负载，再加注入

**用途：** 在真实的 PD KV 路径上看自然的 KV 流量，并在高 EP 负载下注入 IO。

- **流程：**
  1. prefill（TP8 · DCP8）把请求的 KV 留在本地；
  2. 代理 `e2e/pd_proxy.py` 把 prefill 返回的 `kv_transfer_params` 转给某个 decode DP rank；
  3. decode 用 MoRIIO READ 拉 KV，然后开始生成。
- **负载：** `loadgen.py` 经代理发请求，总并发 256，decode 每个 rank 一直是 128 条。
- **节点：**
  - 5 台时：前 2 台 prefill，接下来 2 台 decode，最后一台 C。
  - 4 台时：只有 1 台 prefill。这时 prefill 撑不住 8k 输入，要用 2k（见下面）。
- **阶段：**
  - `pdh`：自然流量 15 分钟，然后跑窗口 IWk IRk IWk10 IRk10 IWkS IWkT0；
  - `pdht0`：KV 放进 TC 0，只看自然流量；
  - `pdhf` / `pdhft0`：同上，但用 `env_<运行名>.txt` 设 `export ISL=1024 OSL=512`，新请求多 4 倍。

```bash
J=$(sbatch --parsable --nodes=5 --time=04:00:00 tools/holder.sbatch)
echo "setup pdh pdht0" > logs/phases_$J.txt
# 4 台节点时，prefill 只有 1 台，要把输入降到 2k：
# for r in pdh pdht0; do echo "export ISL=2048" > logs/env_${r}_$J.txt; done
cp tools/launch.sh logs/run_launch_$J.sh && setsid nohup bash logs/run_launch_$J.sh $J >> logs/nohup_launch_$J.out 2>&1 < /dev/null &
```

**预期**（pdh_6977、pdh_6976 等）：
- 自然流量：TC 104 对 TC 0，步时差 ±0.6%，decode 每秒只收到 2–20 个 CNP；
- 注入：IWk / IRk / IWk10 / IRk10 慢 62–70%，IWkS 和 IWkT0 只慢 0.1–0.7%。

**看结果：** 除了 `summary.md`，还要跑

```bash
python3 e2e/pd_analyze.py results/e2e/pdh_$J results/e2e/pdht0_$J
```

它会输出：
- 按运行请求数和 KV 用量拟合后，每个窗口的步时变化（剔除 profiler 采集前后的点）；
- 自然流量期间按秒对齐的 KV 流入、CNP 和步时，并把多次运行放在一起比较。

## C. EP16 真实 PD 加 AgentX

**用途：** 在真实负载下看 KV 和注入的影响。需要先做第 0.3 节。

- **节点：** 5 台，前 2 台 prefill，2 台 decode，C 跑代理、AgentX 客户端和注入。
- **客户端：** `agentx profile` 回放 `semianalysis_cc_traces_weka_062126`，并发 128。
  - 先做 warmup：快照预热请求，加每条通道 1 个缓存压力请求，大约 20 分钟；
  - 然后测量 2700 秒：前 20 分钟是自然流量，后面是注入窗口 IWk IRk IWk10 IRk10 IWkS IWkT0。
- **阶段：** `pd`；`pdt0` 是 KV 放进 TC 0，只看自然流量，测 1500 秒。

```bash
J=$(sbatch --parsable --nodes=5 --time=04:00:00 tools/holder.sbatch)
echo "setup pd pdt0" > logs/phases_$J.txt
cp tools/launch.sh logs/run_launch_$J.sh && setsid nohup bash logs/run_launch_$J.sh $J >> logs/nohup_launch_$J.out 2>&1 < /dev/null &
```

**预期**（pd_6954 / pdt0_6954）：
- decode 每个 rank 只有 10–40 条在解码，步时 115–150 ms，EP 内核每次 150–350 µs；
- 注入的影响在约 14% 的噪声以内；
- 自然流量下 TC 104 对 TC 0 是 131.5 对 133.7 ms。

**注意：**
- 并发 256 时，2 台 prefill 撑不住，warmup 会拖很久。`pd.sh` 里 warmup 的宽限时间已经改成 3600 秒（`WARM_GRACE`）；当初是 600 秒，AIPerf 取消了还在排队的请求，整轮被中止。
- AgentX 的 TTFT 主要花在 prefill 排队上。

## D. 单测：只跑 EP 内核，加 IO

**用途：** 不起模型，快速测某种 IO 对 EP 内核本身的影响，比如 READ 在途量、主机内存、incast。

- **节点：** 3 台。A 和 B 跑 EP16，B 是接收 IO 的节点，C 是 IO 来源。
- **内核：**
  - v1：vLLM 用的 InterNodeV1LL，直接用镜像里的 mori；
  - v2：EPv2 v2_ll，需要先 `build`。工具包期望在 `$K/src/mori-pr996` 下有一份 ROCm/mori 3a869d5f，见 `mb/build.sh`；Cython 和 prettytable 装在 `$K/pylib`。
- **阶段：** `c1` 是默认场景；`c4` 是更多 IO 形态；`c9` / `c9b` 是 READ 的在途量；`c7` 加了 decode 的占空比。场景定义在 `mb/campaign.sh`。

```bash
J=$(sbatch --parsable --nodes=3 --time=04:00:00 tools/holder.sbatch)
echo "setup build warm c1 c9b" > logs/phases_$J.txt
# 只跑 v1、不编 mori：echo "base1 IW8 IR8 IWk IRk base2" > logs/only_c1_$J.txt; echo "v1k128" > logs/shapes_c1_$J.txt
cp tools/launch.sh logs/run_launch_$J.sh && setsid nohup bash logs/run_launch_$J.sh $J >> logs/nohup_launch_$J.out 2>&1 < /dev/null &
```

结果在 `results/mb/<campaign>/summary.md`。注意单测会**低估**端到端的影响：KV 形状的写入，单测里 v1 只慢 5–11%，端到端慢 66%。所以调缓解配置时，最后要回到 case A 或 B 验证。

## E. EP32（Moonshot 那个 case）：只有 decode，加注入

**用途：** 在 Moonshot 的配置上复现 IO 拖慢 EP，并在 EP32 上验证缓解办法。

- **节点：** 5 台。前 4 台是 decode（DP4 × TP8 = EP32，每台一个 DP rank），最后一台 C 跑负载和注入。
  每个 rank 128 条，全局 batch 512，qlen 8。负载和注入的方式同 case A。
- **三个一定要设的：**
  - `RAIL_ALIGN=1`：见“坑”第 1 条。
  - `PRE_RECV=1`：注入的写入接收进程在 decode 启动前就起好，整轮不关；窗口里只在 C 上起停发送端。见“坑”第 11 条。
  - `OSL=64000`：输出 64k，整轮没有请求结束，batch 一直是 512。默认的 8k，开始 13 分钟后就有请求结束，新请求在 decode 上做 prefill，会把一部分步拉到约 720 ms，混进比较。
- **阶段：** `e1m`；同一个作业里要再跑一轮就用 `e1m2`，它有自己的 env 和 windows 文件。
  每个窗口约 3.6 分钟，加上启动和加压约 11 分钟。

```bash
J=$(sbatch --parsable --nodes=5 --time=04:00:00 tools/holder.sbatch)
echo "setup e1m" > logs/phases_$J.txt
printf 'export RAIL_ALIGN=1\nexport PRE_RECV=1\nexport OSL=64000\n' > logs/env_e1m_$J.txt
echo "base IRk1 IWk1 IWk1p10 IWk10 IWk5 IWk1g IWk IWkS IWk1S IWk10c4 IWk10c2 IWk1T0 IWkT0" > logs/windows_e1m_$J.txt
cp tools/launch.sh logs/run_launch_$J.sh && setsid nohup bash logs/run_launch_$J.sh $J >> logs/nohup_launch_$J.out 2>&1 < /dev/null &
```

**EP32 多加的窗口**（其余见 case A 的窗口一览）：

| 名字 | IO |
|---|---|
| `IWk1p20` `IWk1p10` `IWk1p5` `IWk1p2` | 只写第一台 decode，16 MB 一批，每股 20 / 10 / 5 / 2 GB/s |
| `IWk1S` | 只写第一台，1 MB 一批，连续 |
| `IWk1T0` | 同 `IWk1`，IO 在 TC 0 |
| `IWk0` | 运行中新起接收进程，每台只来一批 16 MB，用来检查起进程会不会让 decode 挂死 |

注意，4 台 decode 时 `IWk`、`IRk` 这类“满速”窗口里，C 的每张网卡同时带 4 股流，每股的瞬时速率只有线速的 1/4 左右，影响反而比每股 5–10 GB/s 的突发小。要打出线速，用只打一台的 `IWk1`、`IRk1`。

**预期**（e1m2_7005，base 528 ms）：

| 窗口 | 步时变化 | decode 收到的 CNP（每秒，每台） |
|---|---|---|
| base | — | 4–6 千（EP32 没有 IO 时就有） |
| IRk1 | +28%（同一轮两次 27.8% / 27.9%） | 被读那台约 0.3 万，另外 3 台 16–21 万 |
| IWk10 / IWk5 | +25% / +27% | 6–8 万 |
| IWk1 | +18%（三次 17.7–18.4%） | 被写那台约 0.6 万，另外 3 台 14–18 万 |
| IWk1g | +13% | 2.4–2.7 万 |
| IWk / IRk | +10.5% / +12.5% | 4.4–5.0 万 |
| IWk10c4 / IWk10c2 | +11.0% / +6.7%（EP16 上 2 MB 已经没有影响） | 5.0–5.5 万 / 3.2–3.9 万 |
| IWkS / IWk1S | +3.3% / +1.3% | 约 2 万 |
| IWkT0 / IWk1T0 / IRkT0 | +0.3% / +0.3% / +0.3% | 和 base 一样 |

**看结果：**

```bash
python3 tools/ep32_win.py results/e2e/e1m_$J    # 每个窗口：IO 开着对前后关着的步时，以及每台 decode 每秒收到的 CNP
```

`summary.md` 里还有 profiler 采到的 EP 内核时间：base 时 dispatch / combine 约 1,550 / 1,100 µs，IWk1 时都 +37%，IRk1 时 +70% / +48%。

**注意：**
- 如果 decode 挂死，会看到 4 台 decode 的 `fab_*.tsv` 发送字节同时不再增长，`load/live.jsonl` 的 `gaps` 变成 0。
  这时窗口会卡在 profiler 的 stop 请求上，最多 600 秒，下一个窗口检测到 stalled 才退出。可以直接结束这次运行，它退出时会清理节点：
  `kill $(pgrep -f "e2e/.snap_${J}_e1m_e1.sh" | head -1)`（启动器跑的是 `e1.sh` 的快照）。
- 每轮 profiler 的 trace 有约 2 GB，`e1_analyze.py` 已经改成并行解析，约 20 秒。

## E2. EP32 真实 PD

**用途：** 在 Moonshot 的配置上走真实的 PD KV 路径，看自然 KV 和注入 IO 的影响。

- **节点：** 6 台（整个集群）。
- **阶段：**
  - `pd32` / `pd32t0`：1 台 prefill、4 台 decode、1 台 C，512 条合成负载，decode 每个 rank 一直 128 条。`pd32` 先看 15 分钟自然流量再注入，`pd32t0` 是 KV 放进 TC 0 的对照，只看自然流量。
  - `pd32a`：2 台 prefill、4 台 decode，AgentX 并发 256，只看自然流量。代理和客户端放在第一台 prefill 上。

```bash
J=$(sbatch --parsable --nodes=6 --time=04:00:00 tools/holder.sbatch)
echo "setup pd32 pd32t0" > logs/phases_$J.txt
printf 'export PRE_RECV=1\nexport ISL=2048\nexport OSL=512\n' > logs/env_pd32_$J.txt     # 输出 512：新请求多，自然 KV 多
printf 'export ISL=2048\nexport OSL=512\n' > logs/env_pd32t0_$J.txt
echo "IWkR64 IWk1g IWk10 IWk1 IWk1T0 IWkS IRk1" > logs/windows_pd32_$J.txt
cp tools/launch.sh logs/run_launch_$J.sh && setsid nohup bash logs/run_launch_$J.sh $J >> logs/nohup_launch_$J.out 2>&1 < /dev/null &
```

**预期**（pd32_7006、pd32t0_7006）：
- 自然 KV：KV 平均 3.6 GB/s、p99 9 GB/s，TC 104 和 TC 0 的步时是 545.3 对 545.0 ms，看不出影响。
- 注入：IRk1 +25%，IWk10 +22%，IWk1 +17%，IWk1g +10%，IWkS +2%，IWk1T0 +0.3%，比只有 decode（case E）低 1.5–3 个百分点。
- `pd32a`（AgentX）在 7006 上预热 3 分钟后 decode 挂死，没有数据，见“坑”第 11 条。

**看结果：**

```bash
python3 e2e/pd_analyze.py results/e2e/pd32_$J results/e2e/pd32t0_$J
```

## F. 读结果，以及可以调的旋钮

### F.1 文件

- `events.tsv`：每个窗口的开始、IO 起来、IO 结束、结束，profiler 采集的时间，以及 PD 运行的 `nat start/end`。
- `load/requests.jsonl`：每个请求的到达时间、首个 token 时间和每个 chunk 的时间。PD 运行还有 prefill 那一路的耗时、prompt 长度和 KV block 数。
- `load/live.jsonl`：每 10 秒的 token 间隔中位数和 p90。
- `load/series.jsonl`：PD 运行每秒一条，记录每个 decode rank 的 token 间隔和转交请求数。
- `fab_<node>.tsv`：每秒的 8 张网卡收发字节、PFC 暂停时间（p3 是 TC 104），以及最后两列：收到的 CNP、收到的 ECN 标记包。
- `snaps.jsonl`：每个窗口在 IO 起来后（`_a`）和结束前（`_b`）各记一次 RDMA 计数器。
- `serve_<node>.log` 是 decode 的日志，`prefill_<node>.log` 是 prefill 的，`proxy.log` 是代理的，`agentx/` 是 AIPerf 的完整输出。

### F.2 判断 EP 有没有被 DCQCN 降速

decode 节点只发 EP 数据（以及 READ 请求）。它收到 CNP，就说明它的发送 QP 被要求降速了：

```bash
for i in 0 1 2 3 4 5 6 7; do cat /sys/class/infiniband/ionic_$i/ports/1/hw_counters/rx_rdma_cnp_pkts; done | paste -sd+ | bc
```

每秒几千个以上，步时就会明显变慢。

### F.3 可以调的旋钮

| 想改什么 | 在哪里改 |
|---|---|
| 注入 IO 的流量类 | 窗口名带 `T0` 就是 TC 0；全局改用 `IOTC`；`mb/env.sh` 里的 `MORI_IO_TC` |
| KV 传输的流量类（PD） | `pd.sh` 的 `KVTC`，传给 `serve.sh` 的 `KV_TC`，最终是 `MORI_IO_TC` |
| EP 的流量类 | `e2e/k31_env.sh` 里的 `MORI_RDMA_TC=104` |
| IO 的单批大小、速率、在途批数 | `e2e/windows.sh` 的 `window()`：`mb`、`per`、`inf` |
| EP 负载 | 每个 rank 的序列数（`MAX_SEQS`，`CONC`）；投机草稿 token 数（`SPEC_K`，qlen = `SPEC_K` + 1） |
| EP 内核 | `serve.sh` 的 `A2A`（`--all2all-backend`，默认 `mori_low_latency`） |
| rail 对齐 | `RAIL_ALIGN=1`，见 `mb/rail.sh` |
| 交换机、网卡的 ECN / DCQCN | 不在工具包里。smci355 上一个来源就能让 EP 被降速，而 DO 上不会，最值得查的就是这里 |

## G. 踩过的坑

1. **一定要 `RAIL_ALIGN=1`。** n08-25 和 n08-33 与其他节点的 GPU 编号顺序不同，`RAIL_ALIGN=0` 时同序号 GPU 有 6 对跨 rail，大批量 decode 挂死过 3 次。
2. **容器要挂载 `/etc/slurm`，** 否则 TaskProlog 失败，`lib.sh` 里已经挂了。srun 要带 `--cpu-bind=none -c 128 --mem=0 --overlap`。
3. **MoRIIO 的 `kv_connector_extra_config` 必须有 `http_port`、`handshake_port`、`notify_port`，** 最好再带 `host_ip`（控制网 IP），否则 worker 初始化时报 KeyError。`pd.sh` 的 `kvcfg()` 已经按节点填好。
4. **prefill 不要用 `shortest-prefill-first`，** 除非同时挂 SimpleCPUOffloadConnector。否则第一个请求进来引擎就报 `Experimental SPF requires SimpleCPU read-only preview` 退出。
5. **代理到 prefill 和 decode 要每路新建连接（`force_close`）。** uvicorn 5 秒就关闭空闲连接，复用到被关掉的连接会 broken pipe，AgentX 的 warmup 就会中止。
6. **AgentX 的 warmup 宽限要够长，** 否则还在 prefill 排队的请求会被取消，整轮中止。
7. **启动失败会自动重试：** 某台服务就绪前退出，`e1.sh` 和 `pd.sh` 会保存日志（`serve_<node>.try1.log`），然后整组重启一次。
8. **srtctl 加 vllm-router 的 PD 在 decode 是 DP2 时卡在交接上。** router 是自动发现模式，只认到一个 decode。所以这里不用它，用自己写的代理。
9. **profiler 导出 trace 时引擎会停几秒，** 统计窗口时要剔掉采集前后的点。`e1_analyze.py` 和 `pd_analyze.py` 都已经剔了。
10. **AgentX 跑的时间长，又会占满 prefill。** 测注入时，用合成负载（case A、B）更干净。
11. **EP32 挂死过一次，就在注入脚本往 decode 上起 io_inject 接收进程之后几秒。** 当时 C 还没开始发 IO，所有 rank 停在 EP 内核里，没有恢复。
    之后用 `PRE_RECV=1` 把写入的接收进程提前到 decode 启动前，没再挂。运行中新起进程（`IWk0`、每次 `IRk1`）也没再复现，触发条件没确认。读类窗口（`IRk*`）的发起端还是在窗口里新起，建议排在后面。
    EP32 真实 PD 加 AgentX（`pd32a`）也挂过一次：预热 3 分钟后所有 rank 停住，10 分钟后 RCCL watchdog 杀掉进程。那台 decode 有进程到作业结束都没杀掉，节点被 Slurm 置成 drain（`Kill task failed`），要管理员 resume。
    挂死时 `load/live.jsonl` 里 `open` 不变、`gaps` 一直是 0，可以早点结束运行，不用等 watchdog。
12. **EP32 用默认的 8k 输出，后段的比较会被 prefill 干扰。** 见 case E，用 `OSL=64000`。看 p90 时尤其要注意：后段有请求结束以后，没有 IO 的 p90 也会涨到约 720 ms。
