# Sourced INSIDE the yutongwu-epio container (k31-ms vLLM image, pyxis) on a smci355 node.
#   MB_MORI=pr996  -> ROCm/mori PR #996 (3a869d5f) built in place under src/ (the mori of the DO microbenchmarks)
#   MB_MORI=image  -> the image's own mori (1.2.5.dev94, what vLLM's MoRI all2all and MoRIIO run on)
# EP (shmem QPs) goes out on MORI_RDMA_TC, IO processes on MORI_IO_TC. smci355's NICs map DSCP 26 (TC 104)
# to priority 3, the only PFC no-drop priority (DWRR 90 %); every other DSCP is lossy priority 0.
W=/apps/yutongwu/store-vllm/k31-vllm-ep-io-smci355-20261119
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
# enroot's 90-amd.env sets ROCM_PATH=/opt/rocm, which this image does not have.
export ROCM_PATH=/usr/local/lib/python3.12/dist-packages/_rocm_sdk_devel ROCM_HOME=/usr/local/lib/python3.12/dist-packages/_rocm_sdk_devel
export HIP_PATH=$ROCM_PATH HIP_CLANG_PATH=$ROCM_PATH/lib/llvm/bin
export PATH=/opt/runtime-venv/bin:$ROCM_PATH/bin:$ROCM_PATH/lib/llvm/bin:/usr/local/lib/python3.12/dist-packages/_rocm_sdk_core/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
unset CUDA_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES GPU_DEVICE_ORDINAL
# RAIL_ALIGN=1: HIP device order that keeps every cross-node GPU pair on one backend rail (mb/rail.sh)
source "$W/mb/rail.sh"
rail_align
PY=/opt/runtime-venv/bin/python
# pylib: prettytable + wcwidth for the EPv2 entry's tables (not in the image)
case "${MB_MORI:-pr996}" in
    pr996) MORI_SRC=$W/src/mori-pr996; export PYTHONPATH="$MORI_SRC:$MORI_SRC/python:$W/pylib" ;;
    image) MORI_SRC=$W/src/mori-pr996; export PYTHONPATH="$W/pylib" ;;
    *) echo "bad MB_MORI=$MB_MORI" >&2; return 2 ;;
esac
export MORI_JIT_CACHE_DIR=/tmp/yutongwu_mori_jit_${MB_MORI:-pr996}
export GLOO_SOCKET_IFNAME=enp81s0f1 MORI_SOCKET_IFNAME=enp81s0f1 NCCL_SOCKET_IFNAME=enp81s0f1 TP_SOCKET_IFNAME=enp81s0f1
export MORI_IB_GID_INDEX=1 MORI_DISABLE_IONIC_CCQE=1 MORI_IB_ENABLE_RELAXED_ORDERING=0
export MORI_EP_DISABLE_RDMA_ATOMICS=1 MORI_ENABLE_RAIL_ONLY=1 MORI_DEVICE_NIC=ionic
export MORI_RDMA_DEVICES=ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7
export MORI_ENABLE_SDMA=0 MORI_EP_LAUNCH_CONFIG_MODE=AUTO HSA_NO_SCRATCH_RECLAIM=1
export MORI_SHMEM_HEAP_SIZE="${MORI_SHMEM_HEAP_SIZE:-8G}"
export MORI_IO_RAIL_AFFINITY=1 MORI_IO_QP_MAX_SEND_WR=32767
export MORI_RDMA_TC="${EPTC:-104}"
IOTC="${IOTC:-104}"
case "$IOTC" in
    104|0) export MORI_IO_TC="$IOTC"; unset MORI_IO_TC_DISABLE ;;
    off)   unset MORI_IO_TC; export MORI_IO_TC_DISABLE=1 ;;
    *) echo "bad IOTC=$IOTC" >&2; return 2 ;;
esac
