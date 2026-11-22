# Sourced INSIDE the yutongwu-epio container (k31-ms vLLM image) by the K3.1 vLLM workers.
# The decode env of the colleague's 1P1D recipe (recipes/colleague/pd-1114/disagg-ms.yaml of k31-umbp-20261112, decode
# role) unchanged, plus what wide EP needs on smci355: MoRI EP over the 8 ionic NICs with the RDMA settings of the
# microbenchmarks (mb/env.sh), a larger MoRI symmetric heap, and ROCM_PATH back to the image's SDK (enroot's 90-amd.env
# points it at /opt/rocm, which this image does not have: FlyDSL's JIT then cannot find ld.lld).
W=/apps/yutongwu/store-vllm/k31-vllm-ep-io-smci355-20261119
export ROCM_PATH=/usr/local/lib/python3.12/dist-packages/_rocm_sdk_devel ROCM_HOME=/usr/local/lib/python3.12/dist-packages/_rocm_sdk_devel
export HIP_PATH=$ROCM_PATH HIP_CLANG_PATH=$ROCM_PATH/lib/llvm/bin
export PATH=/opt/runtime-venv/bin:$ROCM_PATH/bin:$ROCM_PATH/lib/llvm/bin:/usr/local/lib/python3.12/dist-packages/_rocm_sdk_core/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
unset CUDA_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES GPU_DEVICE_ORDINAL PYTHONPATH
# RAIL_ALIGN=1: HIP device order that keeps every cross-node GPU pair on one backend rail (mb/rail.sh)
source "$W/mb/rail.sh"
rail_align
export HOME=/tmp/yt-k31-home HF_HOME=/tmp/yt-k31-home/hf HF_HUB_OFFLINE=1
mkdir -p "$HOME"
export VLLM_USE_V2_MODEL_RUNNER=1 VLLM_ROCM_USE_AITER_MOE_SITUV2=1 VLLM_ROCM_USE_AITER=1
export AITER_BF16_FP8_MOE_BOUND=0 AITER_FLYDSL_STAGE2_FP8=0
export VLLM_ROCM_AITER_MLA_ASM_PADDING=asm VLLM_ROCM_AITER_MLA_DCP_VERIFY=asm VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=NONE
export SAFETENSORS_FAST_GPU=1 VLLM_SSM_CONV_STATE_LAYOUT=DS VLLM_KV_CACHE_LAYOUT=BLHNC
export NCCL_DMABUF_ENABLE=0 HSA_ENABLE_IPC_MODE_LEGACY=1 HIP_FORCE_DEV_KERNARG=1 PYTHONHASHSEED=42
export VLLM_USE_DIRECT_DCP_A2A=0 VLLM_USE_DIRECT_DCP_Q_GATHER=0 VLLM_USE_DIRECT_DCP_KV_GATHER=0
export VLLM_ENGINE_READY_TIMEOUT_S=7200 VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1200 PYTHONNOUSERSITE=1 VLLM_SERVER_DEV_MODE=0
export MORI_IO_SQ_BACKOFF_TIMEOUT_US=500000 MORI_IO_QP_MAX_SEND_WR=8192 MORI_IO_QP_MAX_CQE=16384 MORI_IO_QP_MAX_SGE=2
export MORI_RDMA_TC=104 MORI_IO_TC=104 MORI_IO_TC_DISABLE=0 HSA_NO_SCRATCH_RECLAIM=0
export VLLM_FASTSAFETENSORS_QUEUE_SIZE=-1 VLLM_USE_BREAKABLE_CUDAGRAPH=0
export VLLM_ROCM_USE_AITER_FLYDSL_SPARSE_MLA=1 AITER_FLYDSL_SPARSE_MLA=1 VLLM_ROCM_DCP_AITER_ALLGATHER=1
export VLLM_ROCM_DSA_DCP_GLOBAL_KEY_PREFILL=1 VLLM_ROCM_DSA_DCP_ROWSHARD_PREFILL=1
export MORI_IO_ENABLE_NOTIFICATION=0 THP_MEM_ALLOC_ENABLE=1 MORI_IO_LOG_LEVEL=info
export PYTORCH_CUDA_ALLOC_CONF=backend:native,expandable_segments:False
export VLLM_K3_FUSED_INDEXER_PREP=1 VLLM_K3_DSA_FUSED_NORM_GATE=1 VLLM_K3_KDA_FB_STRIDED=1 VLLM_K3_ATTN_RES_REGTILE=1
export VLLM_K3_LATENT_NORM_UP_PROJ=1 VLLM_ROCM_MEM_HEADROOM_GIB=8 VLLM_ROCM_DCP_VUP_RS=1 VLLM_ROCM_DSA_MULTISTREAM=1
export TRITON_CACHE_DIR=/tmp/yt-k31-cache/triton TORCHINDUCTOR_CACHE_DIR=/tmp/yt-k31-cache/inductor
export VLLM_DCP_Q_REPLICATE=1 VLLM_K31_CPRR_SPLIT96=0 VLLM_K3_SP_PREFILL=0
export K31_RUNTIME_ROLE="${K31_RUNTIME_ROLE:-D}"
# wide EP (MoRI all2all) and the cross-node DP / EP groups
export GLOO_SOCKET_IFNAME=enp81s0f1 NCCL_SOCKET_IFNAME=enp81s0f1 MORI_SOCKET_IFNAME=enp81s0f1 TP_SOCKET_IFNAME=enp81s0f1
export NCCL_IB_HCA=ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7 NCCL_IB_GID_INDEX=1 NCCL_IB_TC=104
export MORI_RDMA_DEVICES=ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7 MORI_DEVICE_NIC=ionic
export MORI_IB_GID_INDEX=1 MORI_DISABLE_IONIC_CCQE=1 MORI_IB_ENABLE_RELAXED_ORDERING=0 MORI_ENABLE_RAIL_ONLY=1
export MORI_SHMEM_HEAP_SIZE="${MORI_SHMEM_HEAP_SIZE:-8G}" MORI_ENABLE_SDMA=0
export MORI_JIT_CACHE_DIR=/tmp/yutongwu_mori_jit_image
