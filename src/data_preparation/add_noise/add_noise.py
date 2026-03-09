import os
import glob
import random
import argparse
from pathlib import Path
import torch
import torchaudio
import numpy as np
from tqdm import tqdm


class AudioAugmentor:
    def __init__(
        self,
        musan_dir,
        rirs_dir,
        env_prob=0.5,
        music_prob=0.3,
        reverb_prob=0.2,
        snr_db=10,
        drr_db=10,
        seed=42,
        target_sr=16000,
    ):
        self.env_files = sorted(
            glob.glob(os.path.join(musan_dir, "noise", "**/*.wav"), recursive=True)
        )
        self.music_files = sorted(
            glob.glob(os.path.join(musan_dir, "music", "**/*.wav"), recursive=True)
        )
        self.reverb_files = sorted(
            glob.glob(os.path.join(rirs_dir, "**/*.wav"), recursive=True)
        )
        self.env_prob = env_prob
        self.music_prob = music_prob
        self.reverb_prob = reverb_prob
        self.snr_db = snr_db
        self.drr_db = drr_db
        self.seed = seed
        self.target_sr = target_sr
        print(f"Loaded: {len(self.env_files)} noise, {len(self.music_files)} music, {len(self.reverb_files)} RIRs")

    def _load_and_resample(self, path):
        wav, sr = torchaudio.load(path)
        if sr != self.target_sr:
            wav = torchaudio.transforms.Resample(sr, self.target_sr)(wav)
        return wav.squeeze(0)

    def _random_crop_or_pad(self, audio, target_len):
        if audio.shape[0] >= target_len:
            start = random.randint(0, audio.shape[0] - target_len)
            return audio[start:start + target_len]
        else:
            padded = torch.zeros(target_len)
            padded[:audio.shape[0]] = audio
            return padded

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

    def _apply_reverb(self, audio, rir, target_drr_db):
        direct_samples = int(50 / 1000 * self.target_sr)
        rir_direct = rir[:direct_samples]
        rir_reverb = rir[direct_samples:]
        energy_direct = torch.sum(rir_direct ** 2) + 1e-12
        energy_reverb = torch.sum(rir_reverb ** 2) + 1e-12
        current_drr_db = 10 * torch.log10(energy_direct / energy_reverb)
        scale_reverb = 10 ** ((current_drr_db - target_drr_db) / 20)
        rir_reverb *= scale_reverb
        rir_scaled = torch.cat([rir_direct, rir_reverb], dim=0)
        audio = audio.unsqueeze(0).unsqueeze(0)
        rir_scaled = rir_scaled.unsqueeze(0).unsqueeze(0)
        out = torch.nn.functional.conv1d(audio, rir_scaled.flip(-1), padding=rir_scaled.shape[-1] - 1)
        out = out[..., :audio.shape[-1]]
        max_val = torch.max(torch.abs(out))
        if max_val > 0:
            out = out / max_val
        return out.squeeze()

    def augment(self, clean_wav, file_idx):
        seed = self.seed + file_idx
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        noisy = clean_wav.clone()
        length = clean_wav.shape[0]
        snr_env = None

        if self.env_files and random.random() < self.env_prob:
            env = self._load_and_resample(random.choice(self.env_files))
            env = self._random_crop_or_pad(env, length)
            snr_env = self.snr_db
            noisy = self._add_noise(noisy, env, snr_env)

        if self.music_files and random.random() < self.music_prob:
            music = self._load_and_resample(random.choice(self.music_files))
            music = self._random_crop_or_pad(music, length)
            snr = snr_env if snr_env is not None else self.snr_db
            noisy = self._add_noise(noisy, music, snr)

        if self.reverb_files and random.random() < self.reverb_prob:
            rir = self._load_and_resample(random.choice(self.reverb_files))
            noisy = self._apply_reverb(noisy, rir, self.drr_db)

        return noisy


def augment_folder(clean_dir, noisy_dir, musan_dir, rirs_dir, **kwargs):
    os.makedirs(noisy_dir, exist_ok=True)
    audio_files = sorted(glob.glob(os.path.join(clean_dir, "*.wav")))
    augmentor = AudioAugmentor(musan_dir, rirs_dir, **kwargs)
    for idx, wav_path in enumerate(tqdm(audio_files)):
        clean_wav = augmentor._load_and_resample(wav_path)
        noisy_wav = augmentor.augment(clean_wav, idx)
        out_path = os.path.join(noisy_dir, Path(wav_path).name)
        torchaudio.save(out_path, noisy_wav.unsqueeze(0), augmentor.target_sr)


def main():
    parser = argparse.ArgumentParser(description="Audio augmentation with MUSAN and RIRs")
    parser.add_argument("--clean_dir", required=True, help="Input clean audio directory")
    parser.add_argument("--noisy_dir", required=True, help="Output noisy audio directory")
    parser.add_argument("--musan_dir", required=True, help="MUSAN dataset root")
    parser.add_argument("--rirs_dir", required=True, help="RIRs dataset root")
    parser.add_argument("--env_prob", type=float, default=0.5, help="Environmental noise probability")
    parser.add_argument("--music_prob", type=float, default=0.3, help="Background music probability")
    parser.add_argument("--reverb_prob", type=float, default=0.2, help="Reverb probability")
    parser.add_argument("--snr_db", type=float, default=10, help="SNR in dB")
    parser.add_argument("--drr_db", type=float, default=10, help="DRR in dB")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--target_sr", type=int, default=16000, help="Target sample rate")

    args = parser.parse_args()
    augment_folder(
        clean_dir=args.clean_dir,
        noisy_dir=args.noisy_dir,
        musan_dir=args.musan_dir,
        rirs_dir=args.rirs_dir,
        env_prob=args.env_prob,
        music_prob=args.music_prob,
        reverb_prob=args.reverb_prob,
        snr_db=args.snr_db,
        drr_db=args.drr_db,
        seed=args.seed,
        target_sr=args.target_sr,
    )


if __name__ == "__main__":
    main()