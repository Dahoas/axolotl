#!/bin/bash
#SBATCH --job-name=rlhf
#SBATCH --account=synth-rlhf
#SBATCH --nodes=2
#SBATCH --partition=a40x
#SBATCH --mem=0
#SBATCH --output=/weka/home-alex/repos/axolotl/examples/MATH/slurm_logs/%j_%x.out
#SBATCH --exclusive

export LD_LIBRARY_PATH=/opt/aws-ofi-nccl/lib:/opt/amazon/efa/lib64:/usr/local/cuda-11.0/efa/lib:/usr/local/cuda-11.0/lib:/usr/local/cuda-11.0/lib64:/usr/local/cuda-11.0:/opt/nccl/build/lib:/opt/aws-ofi-nccl-install/lib:/opt/aws-ofi-nccl/lib:$LD_LIBRARY_PATH
export PATH=/opt/amazon/efa/bin:/opt/amazon/openmpi/bin:$PATH

export NCCL_DEBUG=WARN
export NCCL_PROTO=simple
export FI_EFA_FORK_SAFE=1
export FI_LOG_LEVEL=1
export FI_EFA_USE_DEVICE_RDMA=1 # use for p4dn
export FI_EFA_ENABLE_SHM_TRANSFER=0
export FI_PROVIDER=efa
export FI_EFA_TX_MIN_CREDITS=64
# export CUDA_LAUNCH_BLOCKING=1

export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
echo $MASTER_ADDR
export MASTER_PORT=1234
export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`
# export TRITON_HOST=ip-26-0-133-127:8001/gptj-rm-static

srun --account synth-rlhf /weka/home-alex/repos/axolotl/examples/MATH/scripts/on.sh
