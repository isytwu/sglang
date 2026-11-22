#!/usr/bin/env bash
# Runs INSIDE the yutongwu-epio container:  serve.sh <decode_ep|prefill> <dp_rank> <leader_ip> <port> [extra vllm args...]
# One K3.1 vLLM server, in the foreground (its log is the step's stdout).
#   decode_ep  one DP rank (TP8 DCP8, the colleague's decode) of a DP_SIZE (2) x TP8 decode, one rank per node: MoE
#              through MoRI all2all (--all2all-backend mori_low_latency = InterNodeV1LL), hybrid LB (an API server on
#              every node for its own DP rank), DSpark with SPEC_K draft tokens (qlen SPEC_K + 1)
#   prefill    the colleague's prefill (TP8 DCP8), dp_rank / leader unused
# Env knobs: MODEL (default: local copy when complete, else NFS), SPEC_K (7), SPEC_ACC (synthetic acceptance, 3.0),
#   MAX_SEQS (128 per DP rank), MAX_BATCHED (2048), GPU_UTIL (0.85), KV_CFG (kv-transfer-config JSON, none by default),
#   PROFILE_DIR (enables the torch profiler, driven by /start_profile + /stop_profile), A2A (mori_low_latency),
#   DCP (decode-context-parallel-size, 8), EXTRA_ARGS (appended), KV_TC (the MoRIIO KV transfer's traffic class,
#   MORI_IO_TC: 104 like EP unless set), P_SCHED (the prefill's scheduling policy, shortest-prefill-first: this
#   build's SPF needs the SimpleCPUOffloadConnector next to the KV connector, as the colleague's recipe has it).
set -uo pipefail
mode=$1; dp_rank=$2; leader=$3; port=$4; shift 4
W=/apps/yutongwu/store-vllm/k31-vllm-ep-io-smci355-20261119
[ "$mode" = prefill ] && export K31_RUNTIME_ROLE=P
source "$W/e2e/k31_env.sh"
[ -n "${KV_TC:-}" ] && export MORI_IO_TC=$KV_TC
if [ "$mode" = prefill ]; then
    export VLLM_USE_BREAKABLE_CUDAGRAPH=1 VLLM_ROCM_DCP_VUP_RS=0 VLLM_ROCM_DSA_MULTISTREAM=0 VLLM_K3_SP_PREFILL=1
    export VLLM_K3_SP_PREFILL_MIN_TOKENS=4096
    unset VLLM_DCP_Q_REPLICATE VLLM_K31_CPRR_SPLIT96
fi
ulimit -S -l "$(ulimit -H -l)" 2>/dev/null
LOCAL=/data/models/Kimi-K3.1-Pro; NFS=/perf_apps/data/models/Kimi-K3.1-Pro
if [ -z "${MODEL:-}" ]; then
    if [ "$(ls $LOCAL/*.safetensors 2>/dev/null | wc -l)" = 96 ] && cmp -s $LOCAL/model.safetensors.index.json $NFS/model.safetensors.index.json; then
        MODEL=$LOCAL
    else
        MODEL=$NFS
    fi
fi
SPEC_K=${SPEC_K:-7}; SPEC_ACC=${SPEC_ACC:-3.0}
SPEC="{\"model\":\"/perf_apps/data/models/Kimi-K3.1-Pro-DSpark-preview\",\"num_speculative_tokens\":$SPEC_K,\"method\":\"dspark\",\"attention_backend\":\"ROCM_AITER_MLA\",\"kv_cache_dtype\":\"fp8\",\"draft_sample_method\":\"probabilistic\",\"rejection_sample_method\":\"synthetic\",\"synthetic_acceptance_length\":$SPEC_ACC}"
ARGS=(--served-model-name moonshotai/Kimi-K3.1-Pro --trust-remote-code --moe-backend auto --tensor-parallel-size 8
      --load-format fastsafetensors --language-model-only --max-model-len 1048576 --enable-prefix-caching
      --kv-cache-dtype fp8 --stream-interval 1 --seed 42 --decode-context-parallel-size "${DCP:-8}" --dcp-comm-backend a2a
      --additional-config '{"kda_prefill_backend":"fused"}' --attention-config '{"indexer_kv_dtype":"mxfp4"}'
      --hf-overrides '{"text_config":{"dsa_config":{"indexer_use_fp8":true}}}' --prefix-match-unit 512
      --cp-kv-cache-interleave-size 1 --quantization-config '{"targets":{"*in_proj_qkvgfab":"fp8_per_channel"}}'
      --mamba-ssm-cache-dtype float32 --mamba-cache-mode align --block-size 64 --speculative-config "$SPEC"
      --host 0.0.0.0 --port "$port")
if [ "$mode" = decode_ep ]; then
    q=$((SPEC_K + 1)); mb=${MAX_BATCHED:-2048}; ms=${MAX_SEQS:-128}
    # decode batches are multiples of qlen up to max-num-seqs x qlen; FULL graphs for those, piecewise above
    sizes=$(python3 -c "q=$q; top=$ms*q; s=sorted({q*i for i in (1,2,4,8,16,24,32,48,64,80,96,112,128,144,160,192,224,256) if q*i<=top} | {top}); print(','.join(map(str,s)))")
    cap=${sizes##*,}
    ARGS+=(--gpu-memory-utilization "${GPU_UTIL:-0.85}" --max-num-seqs "$ms" --max-num-batched-tokens "$mb"
           --compilation-config "{\"mode\":3,\"cudagraph_mode\":\"FULL_AND_PIECEWISE\",\"max_cudagraph_capture_size\":$cap,\"custom_ops\":[\"+fused_rms_norm_gated\"],\"cudagraph_capture_sizes\":[$sizes]}"
           --scheduling-policy fcfs --use-replayssm --watermark 0.02
           --data-parallel-size "${DP_SIZE:-2}" --data-parallel-size-local 1 --data-parallel-start-rank "$dp_rank"
           --data-parallel-address "$leader" --data-parallel-rpc-port 13345 --data-parallel-hybrid-lb
           --enable-expert-parallel --all2all-backend "${A2A:-mori_low_latency}")
else
    ARGS+=(--gpu-memory-utilization "${GPU_UTIL:-0.87}" --max-num-seqs "${MAX_SEQS:-48}" --max-num-batched-tokens "${MAX_BATCHED:-16384}"
           --compilation-config '{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE","max_cudagraph_capture_size":4096,"custom_ops":["+fused_rms_norm_gated"],"cudagraph_capture_sizes":[256,512,768,1024,1536,2048,3072,4096]}'
           --scheduling-policy "${P_SCHED:-shortest-prefill-first}")
fi
[ -n "${KV_CFG:-}" ] && ARGS+=(--kv-transfer-config "$KV_CFG")
# EXTRA_ARGS: more vllm serve arguments, word-split
[ -n "${EXTRA_ARGS:-}" ] && ARGS+=($EXTRA_ARGS)
[ -n "${PROFILE_DIR:-}" ] && ARGS+=(--profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"$PROFILE_DIR\",\"torch_profiler_with_stack\":false,\"torch_profiler_use_gzip\":true,\"torch_profiler_dump_cuda_time_total\":false,\"max_iterations\":${PROFILE_ITERS:-40}}")
echo "serve $mode dp_rank=$dp_rank leader=$leader port=$port model=$MODEL host=$(hostname -s) $(date -u +%FT%TZ)"
printf ' %q' vllm serve "$MODEL" "${ARGS[@]}" "$@"; echo
/opt/runtime-venv/bin/python /opt/k31-pr26/prepare_role.py || exit 3
# srt-slurm's k31-ms.sh setup step: the image's prebuilt aiter fmha module instead of a JIT build
/opt/runtime-venv/bin/python /opt/k31-r10/reuse_aiter_module.py || echo "reuse_aiter_module failed"
exec /opt/runtime-venv/bin/vllm serve "$MODEL" "${ARGS[@]}" "$@"
