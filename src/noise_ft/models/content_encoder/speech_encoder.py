import torch
import torch.nn as nn
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import librosa
import numpy as np
import yaml
import safetensors

import torch
import torch.nn as nn
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import librosa
import numpy as np
import yaml
import safetensors

# -------------------------------
# Speaker Embedding Extractor (trainable projection)
# -------------------------------
class SpeakerExtractor(nn.Module):
    def __init__(self, onnx_path: str, proj_dim: int = 1024):
        super().__init__()
        import onnxruntime as ort
        self.session = ort.InferenceSession(onnx_path)
        self.proj = nn.Linear(192, proj_dim)  # 可训练投影
        nn.init.xavier_uniform_(self.proj.weight)

    @staticmethod
    def compute_fbank(wav_path, num_mel_bins=80, frame_length=25, frame_shift=10, dither=0.0):
        waveform, sample_rate = torchaudio.load(wav_path)
        waveform = waveform * (1 << 15)
        mat = kaldi.fbank(
            waveform,
            num_mel_bins=num_mel_bins,
            frame_length=frame_length,
            frame_shift=frame_shift,
            dither=dither,
            sample_frequency=sample_rate,
            window_type='hamming',
            use_energy=False
        )
        # CMN
        mat = mat - torch.mean(mat, dim=0)
        return mat

    def forward(self, wav_path: str):
        feats = self.compute_fbank(wav_path)
        feats = feats.unsqueeze(0).numpy()  # (1, T, D)
        embs = self.session.run(output_names=['embs'], input_feed={'feats': feats})
        spk_emb = torch.tensor(embs[0], dtype=torch.float32)
        spk_emb = self.proj(spk_emb)  # (1, proj_dim), 可训练
        return spk_emb


# -------------------------------
# Content Token Extractor (trainable token embedding)
# -------------------------------
class ContentTokenExtractor(nn.Module):
    def __init__(self, 
                 hubert_model,
                 content_tokenizer,
                 hubert_mean: torch.Tensor,
                 hubert_std: torch.Tensor,
                 vocab_size: int = 32,
                 embed_dim: int = 1024,
                 token_type: str = "hubert_vevo_codec",
                 output_layer: int = 18,
                 device="cuda"):
        super().__init__()
        self.hubert = hubert_model
        self.tokenizer = content_tokenizer
        self.hubert_mean = hubert_mean
        self.hubert_std = hubert_std
        self.token_type = token_type
        self.output_layer = output_layer
        self.device = device
        self.token_embed = nn.Embedding(vocab_size, embed_dim)  # 可训练 token embedding

    @staticmethod
    def load_wav_16k(wav_path: str, device):
        wav, _ = librosa.load(wav_path, sr=16000)
        wav = torch.tensor(wav).unsqueeze(0).to(device)
        return wav

    @torch.no_grad()
    def extract_hubert_features(self, wav_16k: torch.Tensor):
        lengths = torch.tensor([wav_16k.shape[1]], device=wav_16k.device, dtype=torch.long)
        feats, feat_lengths = self.hubert.extract_features(
            wav_16k,
            lengths=lengths,
            num_layers=self.output_layer
        )
        feats = feats[-1]
        return feats, feat_lengths

    def forward(self, wav_path: str):
        wav_16k = self.load_wav_16k(wav_path, self.device)
        feats, _ = self.extract_hubert_features(wav_16k)
        feats = (feats - self.hubert_mean.to(feats)) / self.hubert_std.to(feats)

        # quantize
        if self.token_type == "hubert_codec":
            tokens, _ = self.tokenizer.quantize(feats)
        elif self.token_type == "hubert_vevo_codec":
            x = self.tokenizer.encoder(feats.transpose(1, 2))
            z = self.tokenizer.projector(x)
            _, idx = self.tokenizer.quantizer.codebook.forward_index(z.transpose(2, 1))
            tokens = idx[0]
        else:
            raise ValueError(f"Unknown token_type: {self.token_type}")

        # token embedding，可训练
        token_emb = self.token_embed(tokens)  # (1, L, embed_dim)
        return token_emb


# -------------------------------
# Unified VC Feature Extractor
# -------------------------------
class VCFeatureExtractor(nn.Module):
    def __init__(self,
                 onnx_spk_path: str,
                 hubert_model,
                 content_tokenizer,
                 hubert_mean: torch.Tensor,
                 hubert_std: torch.Tensor,
                 spk_proj_dim: int = 1024,
                 token_vocab_size: int = 32,
                 token_embed_dim: int = 1024,
                 device="cuda"):
        super().__init__()
        self.device = device
        self.spk_extractor = SpeakerExtractor(onnx_spk_path, proj_dim=spk_proj_dim)
        self.content_extractor = ContentTokenExtractor(
            hubert_model=hubert_model,
            content_tokenizer=content_tokenizer,
            hubert_mean=hubert_mean,
            hubert_std=hubert_std,
            vocab_size=token_vocab_size,
            embed_dim=token_embed_dim,
            device=device
        )

    def forward(self, wav_path: str):
        spk_emb = self.spk_extractor(wav_path).to(self.device)       # (1, 1024)
        content_emb = self.content_extractor(wav_path).to(self.device)  # (1, L, 1024)
        return content_emb, spk_emb

# -------------------------------
# Example usage
# -------------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- paths ----
    wav_path = "/hpc_stor03/sjtu_home/yixuan.li/reverb_only.wav"
    onnx_spk_path = "/hpc_stor03/sjtu_home/yixuan.li/project/voxceleb_ECAPA512_LM/voxceleb_ECAPA1024_LM/voxceleb_ECAPA1024_LM.onnx"

    hubert_stat_path = "/hpc_stor03/sjtu_home/yixuan.li/project/Vevo/tokenizer/vq32/hubert_large_l18_mean_std.npz"
    repcodec_cfg_path = "/hpc_stor03/sjtu_home/yixuan.li/project/Vevo/tokenizer/vq32/hubert_large_l18_c32.yaml"
    repcodec_ckpt_path = "/hpc_stor03/sjtu_home/yixuan.li/project/Vevo/tokenizer/vq32/hubert_large_l18_c32.pkl"

    # ---- load models ----
    import torchaudio
    hubert_model = torchaudio.pipelines.HUBERT_LARGE.get_model().to(device).eval()

    with open(hubert_stat_path, "rb") as f:
        stat = np.load(f)
        hubert_mean = torch.tensor(stat["mean"]).to(device)
        hubert_std = torch.tensor(stat["std"]).to(device)

    from models.codec.vevo.vevo_repcodec import VevoRepCodec
    with open(repcodec_cfg_path) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    content_tokenizer = VevoRepCodec(**cfg)
    content_tokenizer.quantizer.initial()
    content_tokenizer.eval()
    import torch
    state = torch.load(repcodec_ckpt_path, map_location="cpu")
    content_tokenizer.load_state_dict(state["model"]["repcodec"])
    content_tokenizer.to(device)

    # ---- unified extractor ----
    extractor = VCFeatureExtractor(
        onnx_spk_path=onnx_spk_path,
        hubert_model=hubert_model,
        content_tokenizer=content_tokenizer,
        hubert_mean=hubert_mean,
        hubert_std=hubert_std,
        device=device
    )

    content_emb, spk_emb = extractor(wav_path)
    print("Content token embedding shape:", content_emb.shape)
    print("Speaker embedding shape:", spk_emb.shape)
