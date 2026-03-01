# GSSN — Flow-SSN in JAX

JAX/Flax reimplementation of the Continuous Flow Stochastic Segmentation Network (Flow-SSN).

## Setup

```bash
uv sync
```

### For GPU (CUDA 12):

```bash
uv sync --extra gpu
```

## Training

```bash
cd scripts
bash run_continuous_lidc.sh my_experiment
```

## Project Structure

```
gssn/
├── data/lidc.py              # LIDC-IDRI dataset loading
├── eval/metrics.py            # Evaluation metrics (GED, HM-IoU, Dice)
├── models/continuous/
│   ├── model.py               # ContinuousFlowSSN
│   └── solvers.py             # Euler ODE solver
├── nn/
│   ├── unet.py                # UNet backbone
│   └── utils.py               # Timestep embedding, etc.
├── factory.py                 # Model/arg building
├── train.py                   # Training loop
└── utils.py                   # EMA, seeding, LR schedule
```
