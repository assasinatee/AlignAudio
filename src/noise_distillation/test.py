import os
import shutil
import torchaudio
from pathlib import Path
from tqdm import tqdm

def filter_audio_by_duration(src_dir, dst_dir, max_duration_sec=10.0):
    """
    遍历 src_dir 内的音频文件，将超过 max_duration_sec 的文件复制到 dst_dir。
    """
    os.makedirs(dst_dir, exist_ok=True)

    audio_files = list(Path(src_dir).glob("*.wav"))
    print(f"总音频数量: {len(audio_files)}")

    filtered_count = 0
    for file in tqdm(audio_files):
        try:
            info = torchaudio.info(file)
            duration = info.num_frames / info.sample_rate
            if duration > max_duration_sec:
                # 拷贝或移动文件到 dst_dir
                shutil.move(file, Path(dst_dir) / file.name)
                filtered_count += 1
        except Exception as e:
            print(f"跳过 {file}: {e}")

    print(f"筛选完成，超过 {max_duration_sec}s 的文件数量: {filtered_count}")

def transfer(src_dir, dst_dir):
    """
    遍历 src_dir 内的音频文件，将超过 max_duration_sec 的文件复制到 dst_dir。
    """
    os.makedirs(dst_dir, exist_ok=True)
    audio_files = list(Path(src_dir).glob("*.wav"))

    filtered_count = 0
    for file in tqdm(audio_files):
        try:
            shutil.move(file, Path(dst_dir) / file.name)
            filtered_count += 1
        except Exception as e:
            print(f"跳过 {file}: {e}")

    print(f"转移文件数量: {filtered_count}")
    
# 使用示例
src_dir = "/hpc_stor03/sjtu_home/yixuan.li/work/TTS/speecht5_res/CLB/train"
dst_dir = "/hpc_stor03/sjtu_home/yixuan.li/work/TTS/speecht5_res/CLB/train_too_long"
filter_audio_by_duration(src_dir, dst_dir, max_duration_sec=9.0)
