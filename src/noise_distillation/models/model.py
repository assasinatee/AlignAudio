import torch
import torch.nn as nn
import types

class DepthwiseConv1d(nn.Module):
    def __init__(self, in_ch, kernel_size=3, dilation=1):
        super().__init__()
        self.pad = (kernel_size - 1) // 2 * dilation
        self.conv = nn.Conv1d(in_ch, in_ch, kernel_size, padding=self.pad, dilation=dilation, groups=in_ch, bias=True)

    def forward(self, x):
        # x: [B, T, D] -> conv over T: need [B, D, T]
        x = x.transpose(1, 2)  # [B, D, T]
        y = self.conv(x)
        return y.transpose(1, 2)  # [B, T, D]

class ChannelMLP(nn.Module):
    def __init__(self, dim, bottleneck=128, dropout=0.1):
        super().__init__()
        self.down = nn.Linear(dim, bottleneck)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, T, D]
        h = self.down(x)         # [B, T, d]
        h = self.act(h)
        h = self.up(h)           # [B, T, D]
        h = self.dropout(h)
        return h

class DualPathAdapter(nn.Module):
    def __init__(self, dim, bottleneck=128, kernel_size=3, dropout=0.1, use_layernorm=True):
        super().__init__()
        self.time_path = nn.Sequential(
            DepthwiseConv1d(dim, kernel_size=kernel_size),
            nn.GELU(),
            nn.LayerNorm(dim)    # optional to stabilize
        )
        self.chan_path = ChannelMLP(dim, bottleneck=bottleneck, dropout=dropout)
        self.alpha = nn.Parameter(torch.tensor(0.1))  # learnable scale for time
        self.beta  = nn.Parameter(torch.tensor(0.1))  # learnable scale for chan
        self.use_layernorm = use_layernorm
        if use_layernorm:
            self.ln = nn.LayerNorm(dim)

    def forward(self, x):
        # x: [B, T, D]
        t = self.time_path(x)   # [B, T, D]
        c = self.chan_path(x)   # [B, T, D]
        out = x + self.alpha * t + self.beta * c
        if self.use_layernorm:
            out = self.ln(out)
        return out

# ---------- Adapter module ----------
class Adapter(nn.Module):
    def __init__(self, embed_dim, bottleneck_dim=128, dropout=0.1, use_layernorm=True):
        super().__init__()
        self.down = nn.Linear(embed_dim, bottleneck_dim)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.use_layernorm = use_layernorm
        if use_layernorm:
            self.ln = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: [B, T, D] or [T, D] -> apply on last dim
        h = self.down(x)
        h = self.act(h)
        h = self.up(h)
        h = self.dropout(h)
        out = x + h
        if self.use_layernorm:
            out = self.ln(out)
        return out

# ---------- Layer wrapper that composes original layer + adapter ----------
class LayerWithAdapter(nn.Module):
    def __init__(self, orig_layer, embed_dim, bottleneck_dim=128, kernel_size=3, dropout=0.1, use_layernorm=True):
        super().__init__()
        self.orig = orig_layer
        self.adapter = DualPathAdapter(embed_dim, bottleneck_dim, kernel_size, dropout, use_layernorm)

    def forward(self, *args, **kwargs):
        """
        Call original layer, then apply adapter to the tensor outputs.
        The wrapper is defensive: if original layer returns a tuple or a BaseModelOutput-like dict,
        it will try to find and replace the main hidden-state tensor.
        """
        out = self.orig(*args, **kwargs)

        # helper to apply adapter on tensor-like output
        def apply_adapter_to_tensor(t):
            # t might be [B, T, D] or [T, D]
            return self.adapter(t)

        # case 1: tuple (common for some internal layer outputs)
        if isinstance(out, tuple):
            # assume first element is the hidden states
            first = out[0]
            try:
                first = apply_adapter_to_tensor(first)
                return (first,) + out[1:]
            except Exception:
                return out

        # case 2: dict / BaseModelOutput-like
        if isinstance(out, dict):
            # common key names:
            for k in ("last_hidden_state", "hidden_states", "hidden_state"):
                if k in out and isinstance(out[k], torch.Tensor):
                    try:
                        out[k] = apply_adapter_to_tensor(out[k])
                    except Exception:
                        pass
            return out

        # case 3: plain Tensor
        if isinstance(out, torch.Tensor):
            try:
                return apply_adapter_to_tensor(out)
            except Exception:
                return out

        # fallback: unknown type, return as-is
        return out

def _get_module_and_attr(root, dotted_name):
    """Given root module and dotted attribute name, return (parent_module, attr_name) if exists"""
    parts = dotted_name.split('.')
    parent = root
    for p in parts[:-1]:
        if not hasattr(parent, p):
            return None, None
        parent = getattr(parent, p)
    last = parts[-1]
    if hasattr(parent, last):
        return parent, last
    return None, None

def find_encoder_layer_list(model):
    """
    Try common locations to find encoder layer list/modulelist for SpeechT5 in HF transformers.
    Returns (parent_module, attr_name, layers_list) or (None, None, None) if not found.
    """
    tries = [
        "speecht5.encoder.encoder.layers",   # some layouts
        "speecht5.encoder.encoder.layer",
        "speecht5.encoder.layers",
        "speecht5.encoder.layer",
        "speecht5.encoder",  # fallback: maybe encoder directly exposes children
    ]
    for name in tries:
        parent, attr = _get_module_and_attr(model, name)
        if parent is not None:
            candidate = getattr(parent, attr)
            # if it's ModuleList or list-like, return
            if isinstance(candidate, (torch.nn.ModuleList, list, tuple)):
                return parent, attr, list(candidate)
            # if it's a module container, return its children as list
            if isinstance(candidate, torch.nn.Module):
                return parent, attr, list(candidate.children())

    # Generic fallback: search for a module under model named 'encoder' and then search for ModuleList inside it
    for nm, mod in model.named_modules():
        if nm.endswith("encoder") and isinstance(mod, torch.nn.Module):
            # look for a ModuleList or attribute containing multiple 'layer' children
            for sub_name, sub_mod in mod.named_children():
                if isinstance(sub_mod, (torch.nn.ModuleList, list)) or 'layer' in sub_name:
                    return mod, sub_name, list(sub_mod) if isinstance(sub_mod, (list, tuple)) else list(sub_mod)
    return None, None, None

def replace(model, k=4, bottleneck_dim=128, kernel_size=3, dropout=0.1, use_layernorm=True):
    """
    Replace last `k` transformer layers inside encoder, robust to whether wrapped_encoder
    is already LayerWithAdapter or raw SpeechT5Encoder.
    """
    encoder = model.speecht5.encoder  # SpeechT5EncoderWithSpeechPrenet

    # 获取实际 encoder
    wrapped_encoder = encoder.wrapped_encoder

    # 如果是 LayerWithAdapter, 取 .orig
    if isinstance(wrapped_encoder, LayerWithAdapter):
        orig_encoder = wrapped_encoder.orig
    else:
        orig_encoder = wrapped_encoder

    # 检查是否有 layers
    if not hasattr(orig_encoder, "layers"):
        raise RuntimeError(f"Cannot find 'layers' in encoder: {type(orig_encoder)}")

    layers = orig_encoder.layers
    if not isinstance(layers, (nn.ModuleList, list, tuple)):
        raise RuntimeError(f"orig_encoder.layers is not ModuleList/list/tuple, got {type(layers)}")

    total = len(layers)
    start_idx = 0 if k is None or k >= total else total - k
    replaced = 0

    for i in range(start_idx, total):
        orig_layer = layers[i]
        embed_dim = getattr(orig_layer, "hidden_size", None)
        if embed_dim is None:
            embed_dim = getattr(getattr(model, "config", None), "d_model", 768)

        layers[i] = LayerWithAdapter(orig_layer, embed_dim, bottleneck_dim, kernel_size, dropout, use_layernorm)
        replaced += 1

    # assign back
    orig_encoder.layers = layers
    print(f"[replace_last_k_safe] Replaced {replaced}/{total} transformer layers (last {k}) with LayerWithAdapter.")
    return replaced

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def freeze_all(model):
    for p in model.parameters():
        p.requires_grad = False

def unfreeze_adapters_by_name(model, name_keyword='adapter'):
    """按参数名包含关键字解冻（通用但依赖命名）"""
    for n, p in model.named_parameters():
        if name_keyword in n.lower():
            p.requires_grad = True

def show_param_counts(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"params total: {total:,}, trainable: {trainable:,} ({trainable/total:.4%})")
    return total, trainable

from transformers import SpeechT5ForSpeechToSpeech, SpeechT5Processor
if __name__ == '__main__':
    device = "cpu"
    processor = SpeechT5Processor.from_pretrained("/hpc_stor03/sjtu_home/yixuan.li/model_ckpt/speecht5_vc")
    model = SpeechT5ForSpeechToSpeech.from_pretrained("/hpc_stor03/sjtu_home/yixuan.li/model_ckpt/speecht5_vc").to(device).eval()
    print(count_parameters(model))
    # attach adapters to last 4 layers only:
    replace(model, k=4, bottleneck_dim=128, dropout=0.1)
    print(count_parameters(model))
    # print(model.speecht5.encoder.wrapped_encoder.layers[-4:])

    freeze_all(model)
    unfreeze_adapters_by_name(model, name_keyword='adapter')
    show_param_counts(model)
    show_param_counts(model.speecht5.encoder)