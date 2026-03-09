import os
import glob
import random
from pathlib import Path

import torch
import torchaudio
import numpy as np
from tqdm import tqdm


class AudioAugmentor:
    def __init__(
        self,
        noise_dir,                 # DEMAND 噪声目录
        noise_prob=0.5,
        snr_range=(5, 20),
        seed=42,
        target_sr=16000,
    ):
        self.noise_files = sorted(
            glob.glob(os.path.join(noise_dir, "**/*.wav"), recursive=True)
        )

        self.noise_prob = noise_prob
        self.snr_range = snr_range
        self.seed = seed
        self.target_sr = target_sr

        print("AudioAugmentor initialized:")
        print(f"  DEMAND noise files: {len(self.noise_files)}")

    # ===== 基础工具函数 =====

    def _load_and_resample(self, path):
        wav, sr = torchaudio.load(path)
        if sr != self.target_sr:
            wav = torchaudio.transforms.Resample(sr, self.target_sr)(wav)
        return wav.squeeze(0)

    def _random_crop_high_energy(self, audio, target_len, top_ratio=0.2):
        """
        从能量最高的 top_ratio 区间中随机裁剪
        """
        if audio.shape[0] <= target_len:
            padded = torch.zeros(target_len)
            padded[:audio.shape[0]] = audio
            return padded

        # 用滑窗估计局部能量
        stride = target_len // 4
        energies = []
        positions = []

        for start in range(0, audio.shape[0] - target_len, stride):
            seg = audio[start:start + target_len]
            energies.append(torch.sum(seg ** 2).item())
            positions.append(start)

        energies = np.array(energies)
        thresh = np.percentile(energies, 100 * (1 - top_ratio))

        candidate_starts = [
            pos for pos, e in zip(positions, energies) if e >= thresh
        ]

        start = random.choice(candidate_starts)
        return audio[start:start + target_len]

    def _add_noise(self, clean, noise, snr_db):
        clean_energy = torch.sum(clean ** 2)
        noise_energy = torch.sum(noise ** 2)

        if noise_energy == 0:
            return clean

        snr_linear = 10 ** (snr_db / 10.0)
        scale = torch.sqrt(clean_energy / (noise_energy * snr_linear))

        noisy = clean + scale * noise

        max_val = torch.max(torch.abs(noisy))
        if max_val > 1.0:
            noisy = noisy / max_val

        return noisy

    # ===== 核心增强函数 =====

    def augment(self, clean_wav, file_idx):
        # 固定随机性（与 Dataset 对齐）
        seed = self.seed + file_idx
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        noisy = clean_wav.clone()
        length = clean_wav.shape[0]

        if self.noise_files and random.random() < self.noise_prob:
            noise = self._load_and_resample(random.choice(self.noise_files))
            noise = self._random_crop_high_energy(noise, length)
            snr = random.uniform(*self.snr_range)
            noisy = self._add_noise(noisy, noise, snr)

        return noisy


def augment_folder(
    clean_dir,
    noisy_dir,
    noise_dir,
    **augment_kwargs,
):
    os.makedirs(noisy_dir, exist_ok=True)

    audio_files = sorted(glob.glob(os.path.join(clean_dir, "*.wav")))
    augmentor = AudioAugmentor(noise_dir, **augment_kwargs)

    for idx, wav_path in enumerate(tqdm(audio_files)):
        clean_wav = augmentor._load_and_resample(wav_path)
        noisy_wav = augmentor.augment(clean_wav, idx)

        out_path = os.path.join(noisy_dir, Path(wav_path).name)
        torchaudio.save(
            out_path,
            noisy_wav.unsqueeze(0),
            augmentor.target_sr,
        )


if __name__ == "__main__":
    augment_folder(
        clean_dir="/hpc_stor03/sjtu_home/yixuan.li/work/TTS/speecht5_res/CLB/test",
        noisy_dir="/hpc_stor03/sjtu_home/yixuan.li/work/TTS/speecht5_res/CLB_test_demand",
        noise_dir="/hpc_stor03/sjtu_home/yixuan.li/DEMAND",
        noise_prob=1,
        snr_range=(10, 20),
        seed=42,
    )
