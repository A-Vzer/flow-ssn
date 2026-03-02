#!/bin/bash
#SBATCH --job-name=gauss_lidc
#SBATCH --output=checkpoints/%x_%j.out
#SBATCH --error=checkpoints/%x_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# Parse experiment name
exp_name="${1:-gauss_lidc_$(date +%Y%m%d_%H%M%S)}"

# Create checkpoint directory
mkdir -p "checkpoints/$exp_name"

# Load modules (adjust based on your cluster setup)
# module load cuda/11.8
# module load cudnn/8.6

# Activate virtual environment
source .venv/bin/activate

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TQDM_MININTERVAL=1

# WandB configuration
export WANDB_PROJECT="gssn"
export WANDB_ENTITY="${WANDB_ENTITY:-$USER}"
export WANDB_NAME="$exp_name"
# Uncomment if running offline or want to sync later
# export WANDB_MODE=offline

# Log system info
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "Experiment Name: $exp_name"
echo "=========================================="

# Print GPU info
nvidia-smi

# Training hyperparameters
res=128
num_classes=2

ARGS=(
# DATA:
    --dataset='lidc'
    --data_dir=../datasets/lidc/data_lidc.hdf5
    --resolution=$res
    --img_channels=1
    --num_classes=$num_classes
# TRAIN:
    --exp_name=$exp_name
    --seed=8
    --epochs=1001
    --bs=32
    --mc_samples=1
    --lr=1e-4
    --lr_warmup=2000
    --wd=1e-4
    --ema_rate=0.9999
# EVAL:
    --eval_samples=32
    --eval_freq=16
# MODEL:
    --model="gauss"
    --band_width=1
# BASE:
    --base_net='unet'
    --base_input_shape 1 $res $res
    --base_model_channels=32
    --base_out_channels=$((num_classes * 2))
    --base_num_res_blocks=1
    --base_attention_resolutions 0
    --base_dropout=0.1
    --base_channel_mult 1 2 4 8
    --base_num_heads=1
    --base_num_head_channels=64
)

# Run training
echo "Starting training..."
uv run python -m gssn.train "${ARGS[@]}" 2>&1 | tee "checkpoints/$exp_name/log.out"

# Capture exit status
exit_status=$?

echo "=========================================="
echo "End Time: $(date)"
echo "Exit Status: $exit_status"
echo "=========================================="

exit $exit_status
