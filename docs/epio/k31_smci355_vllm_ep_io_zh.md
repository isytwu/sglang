# K3.1 vLLM 在 smci355 上：IO 对 EP 的影响（已移走）

这组测试跑的是 vLLM，文档、图和脚本已经移到 isytwu/vllm 的 `k31-epio-smci355` 分支：
[docs/epio/k31_smci355_vllm_ep_io_zh.md](https://github.com/isytwu/vllm/blob/k31-epio-smci355/docs/epio/k31_smci355_vllm_ep_io_zh.md)。

移过去以后更正了机制：smci355 上 DCQCN 实际没有降速，EP 变慢是排在 IO 后面排队，不是被 DCQCN 降速（新文档第 4 节）。
这个分支上的旧版本见 commit ac59eb4 及之前。
