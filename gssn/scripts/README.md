# Training Scripts

This directory contains training scripts for both local execution and SLURM cluster submission.

## Scripts Overview

### Local Execution
- `run_gauss_lidc.sh` - Train Gaussian Segmentation Network locally
- `run_continuous_lidc.sh` - Train Continuous Flow-SSN locally

### SLURM Submission
- `slurm_gauss_lidc.sh` - Submit Gaussian model training to SLURM
- `slurm_continuous_lidc.sh` - Submit Flow-SSN training to SLURM

## Usage

### Local Training

```bash
cd gssn

# Train Gaussian model
./scripts/run_gauss_lidc.sh my_experiment_name

# Train Flow-SSN model
./scripts/run_continuous_lidc.sh my_experiment_name
```

### SLURM Training

```bash
cd gssn

# Submit Gaussian model job
sbatch scripts/slurm_gauss_lidc.sh my_experiment_name

# Submit Flow-SSN job
sbatch scripts/slurm_continuous_lidc.sh my_experiment_name

# Check job status
squeue -u $USER

# View job output (while running)
tail -f checkpoints/my_experiment_name/log.out

# Cancel job
scancel <job_id>
```

If you don't provide an experiment name, it will auto-generate one with timestamp:
```bash
sbatch scripts/slurm_gauss_lidc.sh
# Creates experiment: gauss_lidc_20260302_143022
```

## SLURM Configuration

The SLURM scripts request:
- **Time**: 48 hours
- **Memory**: 32GB
- **CPUs**: 8 cores
- **GPU**: 1 GPU
- **Partition**: `gpu` (adjust for your cluster)

### Customizing SLURM Parameters

Edit the `#SBATCH` directives at the top of the scripts:

```bash
#SBATCH --time=72:00:00        # Request 72 hours instead of 48
#SBATCH --mem=64G              # Request 64GB memory
#SBATCH --gres=gpu:a100:1      # Request specific GPU type
#SBATCH --partition=long       # Use different partition
```

### Loading Modules

If your cluster requires module loading, uncomment and adjust these lines:

```bash
module load cuda/11.8
module load cudnn/8.6
```

## WandB Integration

All training scripts integrate with Weights & Biases for experiment tracking.

### Setup

1. **Login to WandB** (one-time):
   ```bash
   wandb login
   ```

2. **Set your entity** (optional):
   ```bash
   export WANDB_ENTITY="your_team_or_username"
   ```
   Or edit the scripts to set it permanently.

3. **Configure project** (optional):
   The default project is `"gssn"`. To change it:
   ```bash
   export WANDB_PROJECT="my_custom_project"
   ```

### Offline Mode

For clusters without internet access, use offline mode:

```bash
# In the SLURM script, uncomment:
export WANDB_MODE=offline

# After job completes, sync from login node:
wandb sync checkpoints/my_experiment_name/wandb/
```

### WandB Environment Variables

The scripts configure:
- `WANDB_PROJECT="gssn"` - Project name
- `WANDB_ENTITY="${WANDB_ENTITY:-$USER}"` - Team or username (defaults to $USER)
- `WANDB_NAME="$exp_name"` - Run name (auto-generated or provided)

## Training Hyperparameters

### Gaussian Model
- **Model**: Gaussian Segmentation Network with banded covariance
- **Band width**: 1 (3×3 spatial correlation)
- **Base network**: UNet (32 channels, [1,2,4,8] multipliers)
- **Batch size**: 32
- **Learning rate**: 1e-4 with 2000-step warmup
- **Epochs**: 1001
- **Evaluation**: Every 16 epochs with 32 samples

### Flow-SSN Model
- **Model**: Continuous Flow-SSN
- **Flow network**: UNet (16 channels, [1,1,1] multipliers)
- **Base network**: UNet (32 channels, [1,2,4,8] multipliers)
- **ODE steps**: 10 (evaluation)
- **Batch size**: 32
- **Learning rate**: 1e-4 with 2000-step warmup
- **Epochs**: 1001
- **Evaluation**: Every 16 epochs with 32 samples

## Output

All experiments save to `checkpoints/<exp_name>/`:
- `log.out` - Training logs
- `checkpoint_dice.pkl` - Best Dice score checkpoint
- `checkpoint_ged.pkl` - Best energy distance checkpoint
- `wandb/` - WandB logs (if configured)

## Monitoring

### View logs
```bash
# Real-time
tail -f checkpoints/my_experiment_name/log.out

# Search for metrics
grep "valid" checkpoints/my_experiment_name/log.out
```

### Check GPU usage
```bash
# On SLURM job
srun --jobid=<job_id> nvidia-smi

# Or connect to node
ssh <node_name>
nvidia-smi
```

### WandB Dashboard
Visit https://wandb.ai/<entity>/gssn to view:
- Training curves
- Validation metrics
- System metrics (GPU/CPU/Memory)
- Hyperparameters

## Troubleshooting

### Out of Memory
- Reduce `--bs` (batch size) in the script
- Reduce `--eval_samples` (evaluation samples)
- Request more memory: `#SBATCH --mem=64G`

### Job Timeout
- Increase time limit: `#SBATCH --time=72:00:00`
- Reduce `--epochs` for faster completion

### WandB Not Syncing
```bash
# Check WandB status
wandb verify

# Manually sync
wandb sync checkpoints/my_experiment_name/wandb/
```

### Module Not Found
- Ensure virtual environment is activated: `source .venv/bin/activate`
- Install dependencies: `uv sync`

## Comparing Models

After training both models:

```bash
# Check best metrics
grep "model saved" checkpoints/gauss_*/log.out
grep "model saved" checkpoints/c_flowssn_*/log.out

# Compare test results
grep "test" checkpoints/gauss_*/log.out | tail -5
grep "test" checkpoints/c_flowssn_*/log.out | tail -5
```
