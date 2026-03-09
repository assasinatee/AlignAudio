#!/bin/bash
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1

accelerate launch --config_file configs/accelerate/nvidia/1gpu.yaml \
 train.py \
 exp_name=earlyt+cos \
 model=single_task_flow_matching_star_direct \
 data@data_dict=star_audiocaps_speecht5 \
 optimizer.lr=5e-6 \
 train_dataloader.collate_fn.pad_keys='["waveform", "duration"]' \
 val_dataloader.collate_fn.pad_keys='["waveform", "duration"]' \
 warmup_params.warmup_steps=1000 \
 epoch_length=Null \
 epochs=20 \
 +model.pretrained_ckpt="/hpc_stor03/sjtu_home/yixuan.li/work/x_to_audio_generation/experiments/audiocaps_star_ft_speecht5/checkpoints/best/model.safetensors"