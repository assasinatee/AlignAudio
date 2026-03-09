#!/bin/bash
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1

accelerate launch --config_file configs/accelerate/nvidia/1gpu.yaml \
 train.py \
 exp_name=noise_distillation \
 optimizer.lr=2.5e-5 \
 warmup_params.warmup_steps=3000 \
 epoch_length=Null \
 epochs=30 \
 +model.pretrained_ckpt="/hpc_stor03/sjtu_home/yixuan.li/work/accel_hydra/examples/noise_distillation/experiments/noise_distillation/checkpoints/epoch_1/model.safetensors"