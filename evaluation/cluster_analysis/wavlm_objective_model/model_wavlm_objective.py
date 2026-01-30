#!/usr/bin/env python
"""
Standalone WavLM model for inference

Matches the architecture in train_wavlm_fair.py exactly.

Usage:
    from model_wavlm_fair import load_encoder
    
    model = load_encoder("checkpoint.pt", device="cuda")
    z = model.encode(wav)  # wav: [B, 1, T_samples] -> z: [B, code_dim, T_frames]
    logits = model.get_logits(z)  # z: [B, code_dim, T] -> logits: [B, T, K]
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------
# Shared Components (from model.py)
# ------------------------------

class ResBlock(nn.Module):
    def __init__(self, ch, k=3, dilation=(1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList([nn.Conv1d(ch, ch, k, dilation=d, padding=(k*d-d)//2) for d in dilation])
        self.convs2 = nn.ModuleList([nn.Conv1d(ch, ch, k, padding=(k-1)//2) for _ in dilation])

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            x = x + c2(F.leaky_relu(c1(F.leaky_relu(x, 0.1)), 0.1))
        return x


class SnakeBeta(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.a = nn.Parameter(torch.zeros(1, ch, 1))

    def forward(self, x):
        a = F.softplus(self.a) + 1e-2
        return x + (torch.sin(a * x) ** 2) / a.clamp_min(1e-2)


# ------------------------------
# Gaussian Adaptive Attention
# ------------------------------

class GaussianAdaptiveAttention(nn.Module):
    def __init__(self, norm_axis, num_heads, num_gaussians, padding_value=None, eps=1e-8):
        super().__init__()
        self.norm_axis = norm_axis
        self.eps = eps
        self.num_heads = num_heads
        self.padding_value = padding_value
        self.num_gaussians = num_gaussians
        self.mean_offsets = nn.Parameter(torch.zeros(num_gaussians))
        self.c = nn.Parameter(torch.exp(torch.randn(num_gaussians)))

    def forward(self, x):
        mask = x != self.padding_value if self.padding_value is not None else None
        x_f = x.float()
        x_masked = torch.where(mask, x_f, torch.zeros_like(x_f)) if mask is not None else x_f

        mean = x_masked.mean(dim=self.norm_axis, keepdim=True)
        var = x_masked.var(dim=self.norm_axis, keepdim=True) + self.eps
        std = torch.sqrt(var)

        c = self.c.float().clamp(min=1e-2, max=10.0)
        log_mixture = None

        for i in range(self.num_gaussians):
            adjusted_mean = mean + self.mean_offsets[i]
            y_norm = ((x_f - adjusted_mean) / std).clamp(-10.0, 10.0)
            c_i = c[i]
            c2 = c_i * c_i + self.eps
            log_gauss = -(y_norm ** 2) / (2.0 * c2) - 0.5 * math.log(2.0 * math.pi) - 0.5 * torch.log(c2)
            log_mixture = log_gauss if log_mixture is None else log_mixture + log_gauss

        if mask is not None:
            log_mixture = torch.where(mask, log_mixture, torch.full_like(log_mixture, -1e9))

        log_mixture = torch.nan_to_num(log_mixture, nan=-1e9, posinf=-1e9, neginf=-1e9)
        log_denom = torch.logsumexp(log_mixture, dim=self.norm_axis, keepdim=True)
        mixture = torch.exp(log_mixture - log_denom).to(x.dtype)
        mixture = torch.nan_to_num(mixture, nan=0.0, posinf=0.0, neginf=0.0)

        out = x * mixture
        if mask is not None:
            out = torch.where(mask, out, x)
        return out


class MultiHeadGaussianAdaptiveAttention(nn.Module):
    def __init__(self, norm_axis, num_heads, num_gaussians, padding_value=None, eps=1e-8):
        super().__init__()
        self.norm_axis = norm_axis
        self.num_heads = num_heads
        self.attention_heads = nn.ModuleList([
            GaussianAdaptiveAttention(norm_axis, num_heads, num_gaussians, padding_value, eps)
            for _ in range(num_heads)
        ])

    def forward(self, x):
        chunk_size = x.shape[self.norm_axis] // self.num_heads
        outputs = []
        for i, head in enumerate(self.attention_heads):
            start = i * chunk_size
            end = start + chunk_size if i < self.num_heads - 1 else x.shape[self.norm_axis]
            chunk = x.narrow(self.norm_axis, start, end - start)
            outputs.append(head(chunk))
        return torch.cat(outputs, dim=self.norm_axis)


class GaussianBlock(nn.Module):
    def __init__(self, norm_axes, num_heads, num_gaussians, num_layers, padding_value=None, eps=1e-8, residual=True):
        super().__init__()
        self.layers = nn.ModuleList([
            MultiHeadGaussianAdaptiveAttention(norm_axes[i], num_heads[i], num_gaussians[i], padding_value, eps)
            for i in range(num_layers)
        ])
        self.residual = residual

    def forward(self, x):
        for layer in self.layers:
            out = layer(x)
            x = out + x if self.residual else out
        return x


# ------------------------------
# Encoder Block
# ------------------------------

class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride, n_res=2, use_gaatn=False):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, 2*stride, stride=stride, padding=stride//2)
        self.res = nn.ModuleList([ResBlock(out_ch, 3, (1, 3**i, 5**i)) for i in range(n_res)])
        self.snake = SnakeBeta(out_ch)
        self.use_gaatn = use_gaatn
        if use_gaatn:
            self.gaatn = GaussianBlock(
                norm_axes=[2], num_heads=[4], num_gaussians=[4],
                num_layers=1, padding_value=None, eps=1e-8, residual=True
            )

    def forward(self, x):
        x = self.snake(self.conv(x))
        for r in self.res:
            x = r(x)
        if self.use_gaatn:
            x = self.gaatn(x)
        return x


# ------------------------------
# Relative Position Bias (WavLM-style)
# ------------------------------

class RelativePositionBias(nn.Module):
    """Gated Relative Position Bias (WavLM-style)."""
    
    def __init__(self, num_heads, head_dim, num_buckets=320, max_distance=800):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        
        self.rel_pos_embed = nn.Embedding(num_buckets, num_heads)
        self.gate_u_vec = nn.Parameter(torch.randn(num_heads, head_dim) * 0.02)
        self.gate_w_vec = nn.Parameter(torch.randn(num_heads, head_dim) * 0.02)
        self.gate_scale_scalar = nn.Parameter(torch.ones(num_heads))
    
    def _relative_position_bucket(self, relative_position):
        num_buckets = self.num_buckets
        max_distance = self.max_distance
        half_buckets = num_buckets // 2
        
        sign = (relative_position >= 0).long()
        rel_pos_abs = relative_position.abs()
        
        small_threshold = half_buckets // 2
        is_small = rel_pos_abs < small_threshold
        rel_pos_abs_clamped = rel_pos_abs.clamp(min=1)
        
        log_ratio = torch.log(rel_pos_abs_clamped.float() / small_threshold) / math.log(max_distance / small_threshold)
        log_pos = small_threshold + (log_ratio * (half_buckets - small_threshold)).long()
        log_pos = log_pos.clamp(max=half_buckets - 1)
        
        bucket = torch.where(is_small, rel_pos_abs, log_pos)
        bucket = bucket + sign * half_buckets
        bucket = bucket.clamp(0, num_buckets - 1)
        
        return bucket

    def forward(self, q, seq_len):
        device = q.device
        dtype = q.dtype
        B, H, T, D = q.shape
        
        positions = torch.arange(seq_len, device=device)
        relative_pos = positions.unsqueeze(0) - positions.unsqueeze(1)
        
        buckets = self._relative_position_bucket(relative_pos)
        d = self.rel_pos_embed(buckets).to(dtype)
        d = d.permute(2, 0, 1)
        
        g_update = torch.sigmoid(torch.einsum('bhtd,hd->bht', q, self.gate_u_vec.to(dtype)))
        g_reset = torch.sigmoid(torch.einsum('bhtd,hd->bht', q, self.gate_w_vec.to(dtype)))
        
        g_update = g_update.unsqueeze(-1)
        g_reset = g_reset.unsqueeze(-1)
        
        d = d.unsqueeze(0)
        scale = self.gate_scale_scalar.view(1, H, 1, 1).to(dtype)
        
        r_tilde = scale * g_reset * d
        r = d + g_update * d + (1 - g_update) * r_tilde
        
        return r


# ------------------------------
# Conformer Block
# ------------------------------

class ConformerBlock(nn.Module):
    def __init__(self, dim, heads=8, ff_mult=4, conv_k=31, drop=0.1, use_rel_pos=True):
        super().__init__()
        self.heads, self.head_dim = heads, dim // heads
        self.ff1 = nn.Sequential(nn.Linear(dim, dim*ff_mult), nn.GELU(), nn.Dropout(drop),
                                  nn.Linear(dim*ff_mult, dim), nn.Dropout(drop))
        self.qkv = nn.Linear(dim, 3*dim)
        self.out = nn.Linear(dim, dim)
        self.conv = nn.Sequential(
            nn.GroupNorm(1, dim), nn.Conv1d(dim, 2*dim, 1), nn.GLU(dim=1),
            nn.Conv1d(dim, dim, conv_k, padding=conv_k//2, groups=dim),
            nn.GroupNorm(1, dim), nn.SiLU(), nn.Conv1d(dim, dim, 1), nn.Dropout(drop))
        self.ff2 = nn.Sequential(nn.Linear(dim, dim*ff_mult), nn.GELU(), nn.Dropout(drop),
                                  nn.Linear(dim*ff_mult, dim), nn.Dropout(drop))
        self.drop = drop
        # Each conformer has its own rel_pos_bias (not shared)
        self.rel_pos_bias = RelativePositionBias(heads, self.head_dim) if use_rel_pos else None
        self.attn_scale = 32.0

    def forward(self, x):
        B, D, T = x.shape
        s = x.transpose(1, 2)
        s = s + 0.5 * self.ff1(s)
        qkv = self.qkv(s).view(B, T, 3, self.heads, self.head_dim)
        q, k, v = [t.transpose(1, 2) for t in qkv.unbind(2)]
        
        if self.rel_pos_bias is not None:
            scale = self.attn_scale
            q_scaled = q / scale
            attn_scores = torch.matmul(q_scaled, k.transpose(-2, -1))
            attn_scores = (attn_scores - attn_scores.max(dim=-1, keepdim=True).values) * scale
            attn_scores = attn_scores / math.sqrt(self.head_dim)
            
            rel_bias = self.rel_pos_bias(q, T)
            attn_scores = attn_scores + rel_bias
            
            attn_probs = F.softmax(attn_scores, dim=-1)
            attn_probs = F.dropout(attn_probs, p=self.drop, training=self.training)
            attn_out = torch.matmul(attn_probs, v)
            attn_out = attn_out.transpose(1, 2).reshape(B, T, D)
        else:
            attn_out = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.drop if self.training else 0
            ).transpose(1, 2).reshape(B, T, D)
        
        s = s + self.out(attn_out)
        c = s.transpose(1, 2)
        c = c + self.conv(c)
        s = c.transpose(1, 2) + 0.5 * self.ff2(c.transpose(1, 2))
        return s.transpose(1, 2)


# ------------------------------
# WavLM Encoder
# ------------------------------

class WavLMEncoder(nn.Module):
    """
    WavLM-style masked prediction encoder.
    Uses same CNN+Conformer architecture as JEPA but with:
    - Simple linear classification head (instead of StrongClusterHead)
    - No layer aggregation
    - Mask embedding for masked positions
    - Each conformer has its own rel_pos_bias (not shared)
    """
    
    def __init__(
        self,
        code_dim=768,
        channels=[64, 128, 256, 512, 768],
        strides=[4, 2, 8, 5],
        n_res=8,
        n_conformer=8,
        heads=16,
        num_classes=1024,
        use_gaatn=False,
        use_rel_pos=True,
        dropout=0.1,
    ):
        super().__init__()
        self.hop = math.prod(strides)
        self.code_dim = code_dim
        self.num_classes = num_classes
        
        # CNN frontend
        self.input_conv = nn.Conv1d(1, channels[0], 7, padding=3)
        self.encoder = nn.ModuleList([
            EncoderBlock(channels[i], channels[i+1], strides[i], n_res, use_gaatn)
            for i in range(len(strides))
        ])
        self.proj = nn.Conv1d(channels[-1], code_dim, 1)
        
        # Standalone rel_pos_bias (for compatibility - may not be used if each conformer has its own)
        head_dim = code_dim // heads
        self.rel_pos_bias = RelativePositionBias(heads, head_dim) if use_rel_pos else None
        
        # Conformer layers - each has its own rel_pos_bias
        self.conformers = nn.ModuleList([
            ConformerBlock(code_dim, heads, use_rel_pos=use_rel_pos)
            for _ in range(n_conformer)
        ])
        
        # WavLM-specific: mask embedding and linear classification head
        self.mask_emb = nn.Parameter(torch.zeros(code_dim))
        nn.init.uniform_(self.mask_emb, -0.1, 0.1)
        
        self.final_proj = nn.Linear(code_dim, num_classes)
        self.dropout = nn.Dropout(dropout)
    
    def encode(self, wav):
        """
        Encode waveform to latent representation.
        
        Args:
            wav: [B, 1, T] raw waveform
            
        Returns:
            z: [B, code_dim, T'] encoded features
        """
        x = self.input_conv(wav)
        for enc in self.encoder:
            x = enc(x)
        x = self.proj(x)  # [B, code_dim, T]
        
        # Transpose for dropout: [B, T, code_dim]
        x = x.transpose(1, 2)
        x = self.dropout(x)
        x = x.transpose(1, 2)  # [B, code_dim, T]
        
        # Conformer layers
        for conf in self.conformers:
            x = conf(x)
        
        return x  # [B, code_dim, T]
    
    def get_logits(self, z):
        """
        Get classification logits from encoded features.
        
        Args:
            z: [B, code_dim, T] encoded features
            
        Returns:
            logits: [B, T, num_classes] classification logits
        """
        x = z.transpose(1, 2)  # [B, T, code_dim]
        return self.final_proj(x)  # [B, T, num_classes]
    
    def forward(self, wav, mask=None):
        """
        Forward pass with optional masking.
        
        Args:
            wav: [B, 1, T] raw waveform
            mask: [B, T_frames] binary mask (1 = keep, 0 = mask), optional
            
        Returns:
            logits: [B, T, num_classes] classification logits
            features: [B, code_dim, T] encoder features
        """
        # CNN frontend
        x = self.input_conv(wav)
        for enc in self.encoder:
            x = enc(x)
        x = self.proj(x)  # [B, code_dim, T]
        
        # Transpose for masking: [B, T, code_dim]
        x = x.transpose(1, 2)
        T = x.shape[1]
        
        # Apply masking if provided
        if mask is not None:
            mask_expanded = mask[:, :T].unsqueeze(-1).to(x.dtype)  # [B, T, 1]
            x = x * mask_expanded + self.mask_emb.to(x.dtype) * (1 - mask_expanded)
        
        x = self.dropout(x)
        
        # Transpose back for Conformers: [B, code_dim, T]
        x = x.transpose(1, 2)
        
        # Conformer layers
        for conf in self.conformers:
            x = conf(x)
        
        # Output features and logits
        features = x  # [B, code_dim, T]
        logits = self.get_logits(x)  # [B, T, num_classes]
        
        return logits, features


# ------------------------------
# Loading utilities
# ------------------------------

def get_default_config():
    """Default config matching train_wavlm_fair.py."""
    return {
        "sample_rate": 16000,
        "code_dim": 512,
        "channels": [32, 64, 128, 256],
        "strides": [8, 8, 5],
        "n_res": 4,
        "n_conformer": 4,
        "heads": 32,
        "K": 1024,
        "use_gaatn": True,
        "use_rel_pos": True,
        "dropout": 0.1,
    }


def load_encoder(
    checkpoint_path: str,
    config: Optional[dict] = None,
    device: str = "cuda",
    strict: bool = False,
) -> WavLMEncoder:
    """
    Load a trained WavLMEncoder from checkpoint.
    
    Args:
        checkpoint_path: path to .pt checkpoint file
        config: model config dict (uses default if None)
        device: device to load model to
        strict: whether to strictly enforce state dict matching
        
    Returns:
        Loaded WavLMEncoder model in eval mode
    """
    if config is None:
        config = get_default_config()
    
    model = WavLMEncoder(
        code_dim=config.get('code_dim', 512),
        channels=config.get('channels', [32, 64, 128, 256]),
        strides=config.get('strides', [8, 8, 5]),
        n_res=config.get('n_res', 4),
        n_conformer=config.get('n_conformer', 4),
        heads=config.get('heads', 32),
        num_classes=config.get('K', 1024),
        use_gaatn=config.get('use_gaatn', True),
        use_rel_pos=config.get('use_rel_pos', True),
        dropout=config.get('dropout', 0.1),
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'online' in checkpoint:
            state_dict = checkpoint['online']
        elif 'module' in checkpoint:
            state_dict = checkpoint['module']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    # Clean up module. prefix from DDP/DeepSpeed
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            cleaned_state_dict[k[7:]] = v
        else:
            cleaned_state_dict[k] = v
    
    model.load_state_dict(cleaned_state_dict, strict=strict)
    model.to(device)
    model.eval()
    
    return model


def print_model_stats(model: nn.Module, name: str = "Model"):
    """Print model parameter statistics."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n{'='*60}")
    print(f"{name}: {total/1e6:.1f}M params ({trainable/1e6:.1f}M trainable)")
    print(f"{'='*60}")
    
    for n, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        print(f"  {n:<20} {params/1e6:>6.2f}M ({params/total*100:>5.1f}%)")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--test_audio", type=str, default=None)
    args = parser.parse_args()
    
    print(f"Loading model from {args.checkpoint}...")
    model = load_encoder(args.checkpoint, device=args.device)
    print_model_stats(model, "WavLMEncoder")
    
    # Test with random input
    print("\nTesting with random input...")
    with torch.no_grad():
        dummy_wav = torch.randn(1, 1, 16000 * 2).to(args.device)  # 2 seconds
        z = model.encode(dummy_wav)
        logits = model.get_logits(z)
        print(f"Input: {dummy_wav.shape}")
        print(f"Features: {z.shape}")
        print(f"Logits: {logits.shape}")
        print(f"Hop: {model.hop}, Expected frames: {dummy_wav.shape[-1] // model.hop}")
    
    # Test with real audio if provided
    if args.test_audio:
        import torchaudio
        wav, sr = torchaudio.load(args.test_audio)
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        wav = wav.unsqueeze(0).to(args.device)
        
        with torch.no_grad():
            z = model.encode(wav)
            logits = model.get_logits(z)
            codes = logits.argmax(dim=-1)
            print(f"\nAudio: {wav.shape[-1]/16000:.2f}s")
            print(f"  Features: {z.shape}")
            print(f"  Codes: {codes.shape}, unique: {codes.unique().numel()}")
    
    print("\nDone!")
