#!/usr/bin/env bash
# srtctl's orchestrator (do_sweep) for one E2 recipe (configs/e2_pd_*.yaml, benchmark: manual) inside the 3-node
# allocation: prefill on the first node (with the router), the EP16 decode on the other two. Launch from the login node
# as a one-node step on the allocation's first node (e2e/e2.sh does):
#   srun --jobid=<job> --overlap -N1 -n1 -c 128 -w <first node> bash e2e/run_srtctl.sh <recipe> <out dir>
# Adapted from k31-umbp-20261112/tools/run_k31_pd.sh.
set -uo pipefail
RECIPE=${1:?recipe}
OUT=${2:?output dir}
W=/apps/yutongwu/store-vllm/k31-vllm-ep-io-smci355-20261119
S=/apps/yutongwu/store-vllm/k3-vllm-umbp-connector-20261014/src/srt-slurm
mkdir -p "$OUT/logs"
CFG=$OUT/$(basename "$RECIPE")
cp "$RECIPE" "$CFG"

# A one-node step started with `srun --jobid` seeds every nested srun with its task layout; restore the job-level
# view srtctl expects.
JOB_NODELIST=$(/opt/slurm/bin/squeue -h -j "$SLURM_JOB_ID" -o %N)
[[ -n $JOB_NODELIST ]] || { echo "cannot read the node list of job $SLURM_JOB_ID" >&2; exit 1; }
while read -r v; do
    unset "$v"
done < <(compgen -e | grep -E '^SLURM_(STEP_|NTASKS|NPROCS|NNODES|TASKS_PER_NODE|CPUS_PER_TASK|CPU_BIND|DISTRIBUTION|SRUN_COMM_|STEPID|PROCID|LOCALID|NODEID|GTIDS|LAUNCH_NODE_IPADDR|TRES_PER_TASK|OVERLAP|EXACT|PTY_|UMASK|GPUS_ON_NODE|MEM_PER|NODELIST|JOB_NODELIST|JOB_NUM_NODES)')
NN=$(/opt/slurm/bin/scontrol show hostnames "$JOB_NODELIST" | wc -l)
export SLURM_NODELIST=$JOB_NODELIST SLURM_JOB_NODELIST=$JOB_NODELIST
export SLURM_NNODES=$NN SLURM_JOB_NUM_NODES=$NN SLURM_NTASKS=$NN SLURM_NPROCS=$NN
export SLURM_NTASKS_PER_NODE=1 SLURM_CPUS_PER_TASK=128

# All nodes load the same path: their local copies when all are complete (minutes), otherwise NFS (~25 min).
LOCAL_MODEL=/data/models/Kimi-K3.1-Pro
NFS_MODEL=/perf_apps/data/models/Kimi-K3.1-Pro
complete=0
for node in $(/opt/slurm/bin/scontrol show hostnames "$JOB_NODELIST"); do
    if /opt/slurm/bin/srun --jobid="$SLURM_JOB_ID" --overlap -N1 -n1 -w "$node" bash -c \
        "[[ \$(ls $LOCAL_MODEL/*.safetensors 2>/dev/null | wc -l) == 96 ]] &&
         cmp -s $LOCAL_MODEL/model.safetensors.index.json $NFS_MODEL/model.safetensors.index.json"; then
        complete=$((complete + 1))
    fi
done
if [[ $complete == "$NN" ]]; then
    sed -i "s#^  path: Kimi-K3.1-Pro\$#  path: $LOCAL_MODEL#" "$CFG"
fi
grep -m1 "^  path:" "$CFG"
cp "$CFG" "$OUT/config.yaml"

unset PYTHONPATH PYTHONHOME VIRTUAL_ENV CONDA_PREFIX CONDA_DEFAULT_ENV LD_LIBRARY_PATH
# /usr/bin carries an older Slurm client that cannot parse this cluster's slurm.conf.
export PATH=$S/bin:/opt/slurm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export UV_PROJECT_ENVIRONMENT=$S/.venv-compute
export SRTCTL_OUTPUT_DIR=$OUT SRTCTL_SOURCE_DIR=$S SRTSLURM_CONFIG=$W/config/srtslurm.yaml
export INFMAX_WORKSPACE=/apps/yutongwu/store-vllm/k31-umbp-20261112/src/imini-pd-60c8b1f0/inferencex-e2e
export MODEL=moonshotai/Kimi-K3.1-Pro MODEL_PREFIX=kimik3.1 FRAMEWORK=vllm PRECISION=fp4 IS_MULTINODE=true
export RUNNER_TYPE=cluster:mi355x-amds KV_P2P_TRANSFER=moriio IMAGE=yukiozzz/vllm-openai-rocm:k31-ms-1114
export VLLM_ENGINE_READY_TIMEOUT_S=7200 MORI_RDMA_TC=104
export HF_HOME=/hf_hub_cache HF_HUB_CACHE=/hf_hub_cache/hub

echo "e2 recipe=$(basename "$RECIPE") job=$SLURM_JOB_ID nodes=$SLURM_NODELIST start=$(date -u +%FT%TZ)"
cd "$S" || exit 1
uv run --python 3.12 --no-dev --no-sync -m srtctl.cli.do_sweep "$CFG"
rc=$?
echo "e2 rc=$rc end=$(date -u +%FT%TZ)"
exit $rc
