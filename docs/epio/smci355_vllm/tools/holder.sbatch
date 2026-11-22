#!/bin/bash
#SBATCH --job-name=yutongwu-k31-epio
#SBATCH --partition=Compute-Group01
#SBATCH --account=default
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=06:00:00
#SBATCH --exclude=smci355-ccs-aus-n06-21,smci355-ccs-aus-n09-25,smci355-ccs-aus-n09-33
#SBATCH --output=/apps/yutongwu/store-vllm/k31-vllm-ep-io-smci355-20261119/logs/holder_%j.log
# Holds the nodes; everything runs as `srun --jobid=<this job> --overlap` steps launched from the login node
# (tools/launch.sh), which cancels this job when it is done.
echo "holder $SLURM_JOB_ID on $SLURM_JOB_NODELIST from $(date -u +%FT%TZ)"
sleep 21600
