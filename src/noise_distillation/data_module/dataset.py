import os
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import json
import glob
from torch.nn.utils.rnn import pad_sequence
import torchaudio
from collections import defaultdict
import random
from transformers import SpeechT5Processor, SpeechT5ForSpeechToSpeech

# from augment import (add_environmental_noise,
#                      add_music_noise,
#                      add_reverb)

from accel_hydra.data_module.collate_function import PaddingCollate

# class AudioDataset(Dataset):
#     def __init__(self, audio_dir, env, music, reverb, model_name_or_path, device):
#         self.audio_files = sorted(glob.glob(os.path.join(audio_dir, "*.wav")))
#         self.env = sorted(glob.glob(os.path.join(env, "*.wav")))
#         self.music = sorted(glob.glob(os.path.join(music, "*.wav")))
#         self.reverb = sorted(glob.glob(os.path.join(reverb, "*.wav")))
        
#         self.processor = SpeechT5Processor.from_pretrained(model_name_or_path)
#         self.model = SpeechT5ForSpeechToSpeech.from_pretrained(model_name_or_path).eval()
#         self.device = device  # 传入设备，避免硬编码

#     def __len__(self):
#         return len(self.audio_files)

#     def __getitem__(self, idx):
#         wav, sr = torchaudio.load(self.audio_files[idx]) 
#         env, _ = torchaudio.load(self.env[idx])  
#         music, _ = torchaudio.load(self.music[idx])
#         reverb, _ = torchaudio.load(self.reverb[idx]) 
        
#         if sr != 16000:
#             resample = torchaudio.transforms.Resample(sr, 16000)
#             wav = resample(wav)
#             env = resample(env)
#             music = resample(music)
#             reverb = resample(reverb)

#         wav = wav.squeeze(0)
#         env = env.squeeze(0)
#         music = music.squeeze(0)
#         reverb = reverb.squeeze(0)

#         inputs = self.processor(
#             audio=wav,  
#             sampling_rate=16000, 
#             return_tensors="pt", 
#             padding=False 
#         )
#         inputs = {k: v.to(self.device) for k, v in inputs.items()}

#         # (1, T, D) → (T, D)
#         with torch.no_grad():
#             x = self.model.speecht5.encoder(**inputs)
#             enc = x.last_hidden_state.squeeze(0)  # (T_enc, D)

        
#         attn_mask = inputs.get("attention_mask", None)
#         if attn_mask is not None:
#             attn_mask = attn_mask.squeeze(0).cpu()  # (T_enc,)
#         enc = enc.cpu()  

#         return {
#             "clean_wav": wav.cpu(),          # (1, T_clean)
#             "env_noise": env.cpu(),          # (1, T_env)
#             "music_noise": music.cpu(),      # (1, T_music)
#             "reverb_noise": reverb.cpu(),    # (1, T_reverb)
#             "encoder_feat": enc,             # (T_enc, D)
#             "attention_mask": attn_mask      # (T_enc,)
#         }

# def collate_fn(batch_list):
#     """
#     batch_list: List[dict], 每个 dict 由 dataset.__getitem__ 返回
#     """
#     clean_wav   = [b["clean_wav"].squeeze(0).numpy()  for b in batch_list]
#     env_noise   = [b["env_noise"].squeeze(0).numpy()  for b in batch_list]
#     music_noise = [b["music_noise"].squeeze(0).numpy()for b in batch_list]
#     reverb_noise= [b["reverb_noise"].squeeze(0).numpy()for b in batch_list]

#     encoder_feat = [b["encoder_feat"].squeeze(0).numpy()for b in batch_list]
#     attention_mask=[b["attention_mask"].squeeze(0).numpy()for b in batch_list]
#     return {
#         "clean_wav": clean_wav,
#         "env_noise": env_noise,
#         "music_noise": music_noise,
#         "reverb_noise": reverb_noise,
#         "encoder_feat": encoder_feat,
#         "attention_mask": attention_mask
#     }

import random
import numpy as np
import torch
from torch.utils.data import Dataset
import torchaudio
from transformers import SpeechT5Processor, SpeechT5ForSpeechToSpeech
import os
import glob
from pathlib import Path

class AudioDataset(Dataset):
    def __init__(self, audio_dir, musan_dir, rirs_dir, device, 
                 env_prob=0.3, music_prob=0.3, reverb_prob=0.3, snr_range=(0, 20),
                 seed=42, h5_file=None):
        """
        参数说明:
            audio_dir: 干净音频目录
            musan_dir: MUSAN目录，包含music/和noise/子目录
            rirs_dir: RIRS目录，包含simulated_rirs/音频文件
            env_prob: 添加环境音的概率
            music_prob: 添加背景音乐的概率
            reverb_prob: 添加混响的概率
            snr_range: 信噪比范围(dB)
            seed: 随机种子，确保可复现
            h5_file: str, 如果提供，直接从h5读取encoder_feat
        """
        # 设置随机种子
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        self.audio_files = sorted(glob.glob(os.path.join(audio_dir, "*.wav")))

        self.env_files = sorted(glob.glob(os.path.join(musan_dir, "noise", "**/*.wav"), recursive = True))
        self.music_files = sorted(glob.glob(os.path.join(musan_dir, "music", "**/*.wav"), recursive = True))
        self.reverb_files = sorted(glob.glob(os.path.join(rirs_dir, "**/*.wav"), recursive = True))

        self.env_prob = env_prob
        self.music_prob = music_prob
        self.reverb_prob = reverb_prob
        self.snr_range = snr_range
        
        self.device = device
        self.h5_file = h5_file
        self.h5f = None
        
        print(f"数据集初始化完成:")
        print(f"  干净音频: {len(self.audio_files)} 个文件")
        print(f"  环境音: {len(self.env_files)} 个文件")
        print(f"  背景音乐: {len(self.music_files)} 个文件")
        print(f"  混响: {len(self.reverb_files)} 个文件")
        print(f"  加噪概率 - 环境音: {env_prob}, 音乐: {music_prob}, 混响: {reverb_prob}")
        print(f"  SNR范围: {snr_range} dB")
        print(f"  使用h5_file: {h5_file is not None}")

    def __len__(self):
        return len(self.audio_files)
    
    def _load_and_resample(self, audio_path, target_sr=16000):
        """加载音频并重采样到目标采样率"""
        wav, sr = torchaudio.load(audio_path)
        if sr != target_sr:
            resample_transform = torchaudio.transforms.Resample(sr, target_sr)
            wav = resample_transform(wav)
        return wav.squeeze(0), target_sr
    
    def _random_crop_or_pad(self, audio, target_length):
        """随机裁剪或填充音频到目标长度"""
        audio_len = audio.shape[0]
        
        if audio_len >= target_length:
            # 随机裁剪
            start = random.randint(0, audio_len - target_length)
            return audio[start:start + target_length]
        else:
            # 填充
            padded = torch.zeros(target_length)
            padded[:audio_len] = audio
            return padded
    
    def _add_noise(self, clean_audio, noise_audio, snr_db):
        """按照指定SNR添加噪声"""
        # 计算能量
        clean_energy = torch.sum(clean_audio ** 2)
        noise_energy = torch.sum(noise_audio ** 2)
        
        if noise_energy == 0:
            return clean_audio
        
        # 计算需要的噪声缩放因子
        target_snr_linear = 10 ** (snr_db / 10.0)
        scale_factor = torch.sqrt(clean_energy / (noise_energy * target_snr_linear))
        
        # 添加噪声
        noisy_audio = clean_audio + scale_factor * noise_audio
        
        # 归一化防止过载
        max_val = torch.max(torch.abs(noisy_audio))
        if max_val > 1.0:
            noisy_audio = noisy_audio / max_val
            
        return noisy_audio
    
    def _apply_reverb(self, audio, impulse_response):
        """应用混响（卷积）"""
        # 确保都是1D tensor
        audio = audio.unsqueeze(0) if audio.dim() == 1 else audio
        impulse_response = impulse_response.unsqueeze(0) if impulse_response.dim() == 1 else impulse_response
        
        # 卷积操作
        reverb_audio = torch.nn.functional.conv1d(
            audio.unsqueeze(0), 
            impulse_response.flip(-1).unsqueeze(0), 
            padding=impulse_response.shape[-1] - 1
        ).squeeze(0)
        
        # 裁剪到原始长度
        reverb_audio = reverb_audio[:, :audio.shape[-1]]
        
        # 归一化
        max_val = torch.max(torch.abs(reverb_audio))
        if max_val > 0:
            reverb_audio = reverb_audio / max_val
            
        return reverb_audio.squeeze(0)
    
    def __getitem__(self, idx):
        # 为每个样本设置确定性随机种子
        sample_seed = self.seed + idx
        random_state = random.getstate()
        np_state = np.random.get_state()
        torch_state = torch.random.get_rng_state()
        
        random.seed(sample_seed)
        np.random.seed(sample_seed)
        torch.manual_seed(sample_seed)
        
        try:
            # 加载干净音频
            clean_wav, sr = self._load_and_resample(self.audio_files[idx])
            clean_length = clean_wav.shape[0]
            
            # 初始化加噪后的音频
            noisy_wav = clean_wav.clone()
            
            # 存储噪声信息
            noise_info = {
                "env_used": False,
                "music_used": False,
                "reverb_used": False,
                "snr": None
            }
            
            # 1. 随机添加环境音
            if self.env_files and random.random() < self.env_prob:
                env_idx = random.randint(0, len(self.env_files) - 1)
                env_wav, _ = self._load_and_resample(self.env_files[env_idx])
                
                # 裁剪或填充到目标长度
                if env_wav.shape[0] != clean_length:
                    env_wav = self._random_crop_or_pad(env_wav, clean_length)
                
                # 随机SNR
                snr = random.uniform(*self.snr_range)
                noisy_wav = self._add_noise(noisy_wav, env_wav, snr)
                noise_info["env_used"] = True
                noise_info["snr_env"] = snr
            
            # 2. 随机添加背景音乐
            if self.music_files and random.random() < self.music_prob:
                music_idx = random.randint(0, len(self.music_files) - 1)
                music_wav, _ = self._load_and_resample(self.music_files[music_idx])
                
                # 裁剪或填充到目标长度
                if music_wav.shape[0] != clean_length:
                    music_wav = self._random_crop_or_pad(music_wav, clean_length)
                
                # 随机SNR（如果已经添加了环境音，使用同一个SNR或重新生成）
                snr = noise_info.get("snr_env", random.uniform(*self.snr_range))
                noisy_wav = self._add_noise(noisy_wav, music_wav, snr)
                noise_info["music_used"] = True
                noise_info["snr_music"] = snr
            
            # 3. 随机添加混响
            if self.reverb_files and random.random() < self.reverb_prob:
                reverb_idx = random.randint(0, len(self.reverb_files) - 1)
                impulse_response, _ = self._load_and_resample(self.reverb_files[reverb_idx])
                
                # 应用混响
                noisy_wav = self._apply_reverb(noisy_wav, impulse_response)
                noise_info["reverb_used"] = True
            
            # 确保长度一致
            min_len = min(clean_wav.shape[0], noisy_wav.shape[0])
            clean_wav = clean_wav[:min_len]
            noisy_wav = noisy_wav[:min_len]
            
            if self.h5_file is not None:
                key = Path(self.audio_files[idx]).stem
                self.h5f = h5py.File(self.h5_file, "r")
                enc = torch.from_numpy(self.h5f[key][:]).cpu()  # (T, D)
            
            return {
                "clean_wav": clean_wav.cpu(),          # 原始干净音频
                "noisy_wav": noisy_wav.cpu(),          # 加噪后的音频
                "encoder_feat": enc,                   # 编码器特征
                "noise_info": noise_info,              # 噪声信息
                "audio_path": self.audio_files[idx]    # 音频路径（用于调试）
            }
            
        finally:
            # 恢复随机状态
            random.setstate(random_state)
            np.random.set_state(np_state)
            torch.set_rng_state(torch_state)

# file: data_module/dataset.py
from typing import Any, Dict, List, Optional
import numpy as np
import torch

class CollateFn:
    def __init__(self, to_numpy: bool = True):
        self.to_numpy = bool(to_numpy)

    def _maybe_numpy(self, x: Any) -> Any:
        """Convert torch.Tensor -> numpy.ndarray if requested; leave other types unchanged."""
        if x is None:
            return None
        if self.to_numpy and isinstance(x, torch.Tensor):
            # ensure tensor is on CPU
            if x.device.type != "cpu":
                x = x.detach().cpu()
            return x.numpy()
        # already numpy or not a tensor — return as-is
        return x

    def __call__(self, batch_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        # defensive access using .get to avoid KeyError for missing fields
        clean_wav = [self._maybe_numpy(b.get("clean_wav")) for b in batch_list]
        noisy_wav = [self._maybe_numpy(b.get("noisy_wav")) for b in batch_list]
        encoder_feat = [b.get("encoder_feat") for b in batch_list]
        noise_info = [b.get("noise_info") for b in batch_list]
        audio_paths = [b.get("audio_path") for b in batch_list]

        return {
            "clean_wav": clean_wav,
            "noisy_wav": noisy_wav,
            "encoder_feat": encoder_feat,
            "noise_info": noise_info,
            "audio_paths": audio_paths,
        }

def save_reverb_sample_only():
    """
    获取一个只带混响的样本并保存
    """
    # 设置路径
    audio_dir = "/hpc_stor03/sjtu_home/yixuan.li/work/TTS/speecht5_res/CLB/test"
    musan_dir = "/hpc_stor03/public/shared/data/raa/musan"
    rirs_dir = "/hpc_stor03/public/shared/data/raa/RIRS_NOISES/simulated_rirs/mediumroom"
    model_name_or_path = "/hpc_stor03/sjtu_home/yixuan.li/model_ckpt/speecht5_vc"
    
    # 创建只加混响的数据集
    dataset = AudioDataset(
        audio_dir=audio_dir,
        musan_dir=musan_dir,
        rirs_dir=rirs_dir,
        model_name_or_path=model_name_or_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        env_prob=0.0,    # 不加环境音
        music_prob=0.0,  # 不加音乐
        reverb_prob=1.0, # 只加混响
        snr_range=(5, 20),
        seed=42
    )
    
    # 获取第一个样本
    sample = dataset[0]
    
    print(f"干净音频形状: {sample['clean_wav'].shape}")
    print(f"加混响音频形状: {sample['noisy_wav'].shape}")
    print(f"噪声信息: {sample['noise_info']}")
    
    # 保存音频文件
    output_dir = "/hpc_stor03/sjtu_home/yixuan.li"
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存干净音频
    clean_path = os.path.join(output_dir, "clean.wav")
    torchaudio.save(clean_path, sample['clean_wav'].unsqueeze(0), 16000)
    print(f"保存干净音频到: {clean_path}")
    
    # 保存混响音频
    reverb_path = os.path.join(output_dir, "reverb_only.wav")
    torchaudio.save(reverb_path, sample['noisy_wav'].unsqueeze(0), 16000)
    print(f"保存混响音频到: {reverb_path}")
    
    # 打印对比信息
    print(f"\n对比信息:")
    print(f"音频长度: {sample['clean_wav'].shape[0]/16000:.2f} 秒")
    print(f"是否添加了混响: {sample['noise_info']['reverb_used']}")
    
    return clean_path, reverb_path

# 使用示例
if __name__ == "__main__":
    # save_reverb_sample_only()
    # 设置路径
    audio_dir = "/hpc_stor03/sjtu_home/yixuan.li/work/TTS/speecht5_res/CLB/val"
    musan_dir = "/hpc_stor03/public/shared/data/raa/musan"
    rirs_dir = "/hpc_stor03/public/shared/data/raa/RIRS_NOISES/simulated_rirs/mediumroom"
    
    # 创建数据集
    dataset = AudioDataset(
        audio_dir=audio_dir,
        musan_dir=musan_dir,
        rirs_dir=rirs_dir,
        device="cuda" if torch.cuda.is_available() else "cpu",
        env_prob=0.5,
        music_prob=0.3,
        reverb_prob=0.2,
        snr_range=(5, 20),
        seed=42,
        h5_file = "/hpc_stor03/sjtu_home/yixuan.li/work/audio_embeds/cavp_features/speecht5_features/CLB_val.h5"
    )
    
    # 测试一个样本
    sample = dataset[0]
    print(f"干净音频形状: {sample['clean_wav'].shape}")
    print(f"加噪音频形状: {sample['noisy_wav'].shape}")
    print(f"feature形状: {sample['encoder_feat'].shape}")
    print(f"噪声信息: {sample['noise_info']}")


# if __name__ == '__main__':
#     dataset = AudioDataset(
#         "/hpc_stor03/sjtu_home/yixuan.li/work/TTS/speecht5_res/CLB/test",
#         "/hpc_stor03/sjtu_home/yixuan.li/work/TTS/speecht5_res/augmented_CLB_test/env_noise",
#         "/hpc_stor03/sjtu_home/yixuan.li/work/TTS/speecht5_res/augmented_CLB_test/music_noise",
#         "/hpc_stor03/sjtu_home/yixuan.li/work/TTS/speecht5_res/augmented_CLB_test/reverb",
#         "/hpc_stor03/sjtu_home/yixuan.li/model_ckpt/speecht5_vc",
#         "cpu"
#     )
#     loader = DataLoader(dataset,
#                         batch_size=2,
#                         shuffle=False,
#                         num_workers=1,
#                         pin_memory=True,
#                         persistent_workers=True,
#                         collate_fn = collate_fn
#                     )
#     for batch in loader:
#         print(batch["clean_wav"])
#         print(batch["env_noise"])
#         print(batch["encoder_feat"])
#         print(batch["attention_mask"])
#         break
        
    