#!/bin/bash
exp_name="$1"
mkdir -p "checkpoints/$exp_name"
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

export TQDM_MININTERVAL=1

uv run python -m gssn.train "${ARGS[@]}" 2>&1 | tee "checkpoints/$exp_name/log.out"
