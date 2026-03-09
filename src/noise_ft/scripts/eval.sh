python evaluation/path_star.py \
  --gen_audio_dir /hpc_stor03/sjtu_home/yixuan.li/work/x_to_audio_generation/experiments/audiocaps_star_ft_speecht5/inference/none+cos/clean_step_50 \
  -o clean \
  -c 16

python evaluation/path_star.py \
  --gen_audio_dir /hpc_stor03/sjtu_home/yixuan.li/work/x_to_audio_generation/experiments/audiocaps_star_ft_speecht5/inference/none+cos/noisy_step_50 \
  -o noisy \
  -c 16