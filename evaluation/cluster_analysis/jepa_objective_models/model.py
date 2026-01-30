"""
Model definitions for cluster analysis evaluation.

Matches the architecture in train/jepa/model.py exactly.

Usage:
    from model import load_encoder
    
    model = load_encoder("checkpoint.pt", device="cuda", use_gaatn=True)
    z = model.encode(wav)  # wav: [B, 1, T_samples] -> z: [B, code_dim, T_frames]
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Gaussian Adaptive Attention
# =============================================================================

class GaussianAdaptiveAttention(nn.Module):
    def __init__(self, norm_axis: int, num_gaussians: int, eps: float = 1e-8):
        super().__init__()
        self.norm_axis = norm_axis
        self.eps = eps
        self.num_gaussians = num_gaussians
        self.mean_offsets = nn.Parameter(torch.zeros(num_gaussians))
        self.c = nn.Parameter(torch.exp(torch.randn(num_gaussians)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_f = x.float()
        mean = x_f.mean(dim=self.norm_axis, keepdim=True)
        std = torch.sqrt(x_f.var(dim=self.norm_axis, keepdim=True) + self.eps)

        c = self.c.float().clamp(min=1e-2, max=10.0)
        log_mixture = None

        for i in range(self.num_gaussians):
            y_norm = ((x_f - mean - self.mean_offsets[i]) / std).clamp(-10.0, 10.0)
            c2 = c[i] ** 2 + self.eps
            log_gauss = -(y_norm ** 2) / (2.0 * c2) - 0.5 * math.log(2.0 * math.pi) - 0.5 * torch.log(c2)
            log_mixture = log_gauss if log_mixture is None else log_mixture + log_gauss

        log_mixture = torch.nan_to_num(log_mixture, nan=-1e9, posinf=-1e9, neginf=-1e9)
        mixture = torch.softmax(log_mixture, dim=self.norm_axis).to(x.dtype)
        return x * mixture


class GaussianBlock(nn.Module):
    def __init__(self, norm_axis: int = 2, num_heads: int = 4, num_gaussians: int = 4):
        super().__init__()
        self.heads = nn.ModuleList([
            GaussianAdaptiveAttention(norm_axis, num_gaussians)
            for _ in range(num_heads)
        ])
        self.norm_axis = norm_axis
        self.num_heads = num_heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        chunk_size = x.shape[self.norm_axis] // self.num_heads
        outputs = []
        for i, head in enumerate(self.heads):
            start = i * chunk_size
            end = start + chunk_size if i < self.num_heads - 1 else x.shape[self.norm_axis]
            outputs.append(head(x.narrow(self.norm_axis, start, end - start)))
        return x + torch.cat(outputs, dim=self.norm_axis)


# =============================================================================
# Cross-Attention Layer Aggregation
# =============================================================================

class AttentionLayerAggregation(nn.Module):
    """Cross-attention layer aggregation: query from final layer, keys/values from all layers."""

    def __init__(self, code_dim: int, num_layers: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.num_layers = num_layers
        self.code_dim = code_dim
        self.num_heads = num_heads
        self.head_dim = code_dim // num_heads

        self.q_proj = nn.Linear(code_dim, code_dim)
        self.k_proj = nn.Linear(code_dim, code_dim)
        self.v_proj = nn.Linear(code_dim, code_dim)
        self.out_proj = nn.Linear(code_dim, code_dim)

        self.scale = self.head_dim ** -0.5
        self.dropout = dropout

    def forward(self, layers: list) -> torch.Tensor:
        B, C, T = layers[0].shape
        stacked = torch.stack(layers, dim=1)  # [B, num_layers, C, T]
        pooled = stacked.mean(dim=-1)  # [B, num_layers, C]

        q = self.q_proj(pooled[:, -1:, :])  # [B, 1, C]
        k = self.k_proj(pooled)  # [B, num_layers, C]
        v = self.v_proj(pooled)  # [B, num_layers, C]

        q = q.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, self.num_layers, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, self.num_layers, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)

        layer_weights = attn.mean(dim=1).squeeze(1)  # [B, num_layers]
        z_agg = (stacked * layer_weights[:, :, None, None]).sum(dim=1)
        return z_agg


# =============================================================================
# Relative Position Bias (WavLM-style)
# =============================================================================

class RelativePositionBias(nn.Module):
    """Gated Relative Position Bias."""

    def __init__(self, num_heads: int, head_dim: int, num_buckets: int = 320, max_distance: int = 800):
        super().__init__()
        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.max_distance = max_distance

        self.rel_pos_embed = nn.Embedding(num_buckets, num_heads)
        self.gate_u = nn.Parameter(torch.randn(num_heads, head_dim) * 0.02)
        self.gate_w = nn.Parameter(torch.randn(num_heads, head_dim) * 0.02)
        self.scale = nn.Parameter(torch.ones(num_heads))

    def _bucket(self, rel_pos: torch.Tensor) -> torch.Tensor:
        half = self.num_buckets // 2
        sign = (rel_pos >= 0).long()
        abs_pos = rel_pos.abs()
        threshold = half // 2

        is_small = abs_pos < threshold
        log_ratio = torch.log(abs_pos.clamp(min=1).float() / threshold) / math.log(self.max_distance / threshold)
        log_pos = (threshold + log_ratio * (half - threshold)).long().clamp(max=half - 1)

        bucket = torch.where(is_small, abs_pos, log_pos) + sign * half
        return bucket.clamp(0, self.num_buckets - 1)

    def forward(self, q: torch.Tensor, seq_len: int) -> torch.Tensor:
        device, dtype = q.device, q.dtype
        B, H, T, D = q.shape

        positions = torch.arange(seq_len, device=device)
        rel_pos = positions.unsqueeze(0) - positions.unsqueeze(1)
        d = self.rel_pos_embed(self._bucket(rel_pos)).to(dtype).permute(2, 0, 1).unsqueeze(0)

        g_u = torch.sigmoid(torch.einsum('bhtd,hd->bht', q, self.gate_u.to(dtype))).unsqueeze(-1)
        g_w = torch.sigmoid(torch.einsum('bhtd,hd->bht', q, self.gate_w.to(dtype))).unsqueeze(-1)
        s = self.scale.view(1, H, 1, 1).to(dtype)

        return d + g_u * d + (1 - g_u) * s * g_w * d


# =============================================================================
# Encoder Components
# =============================================================================

class SnakeBeta(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.a = nn.Parameter(torch.zeros(1, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = F.softplus(self.a) + 1e-2
        return x + (torch.sin(a * x) ** 2) / a.clamp_min(1e-2)


class ResBlock(nn.Module):
    def __init__(self, channels: int, kernel: int = 3, dilation: tuple = (1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList([
            nn.Conv1d(channels, channels, kernel, dilation=d, padding=(kernel * d - d) // 2)
            for d in dilation
        ])
        self.convs2 = nn.ModuleList([
            nn.Conv1d(channels, channels, kernel, padding=(kernel - 1) // 2)
            for _ in dilation
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for c1, c2 in zip(self.convs1, self.convs2):
            x = x + c2(F.leaky_relu(c1(F.leaky_relu(x, 0.1)), 0.1))
        return x


class EncoderBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int, n_res: int = 2, use_gaatn: bool = False):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, 2 * stride, stride=stride, padding=stride // 2)
        self.res = nn.ModuleList([ResBlock(out_ch, 3, (1, 3**i, 5**i)) for i in range(n_res)])
        self.snake = SnakeBeta(out_ch)
        self.gaatn = GaussianBlock() if use_gaatn else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.snake(self.conv(x))
        for r in self.res:
            x = r(x)
        if self.gaatn is not None:
            x = self.gaatn(x)
        return x


# =============================================================================
# Conformer Block
# =============================================================================

class ConformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int = 8, ff_mult: int = 4, conv_k: int = 31,
                 drop: float = 0.1, rel_pos_bias: RelativePositionBias = None):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.attn_scale = 32.0

        self.ff1 = nn.Sequential(
            nn.Linear(dim, dim * ff_mult), nn.GELU(), nn.Dropout(drop),
            nn.Linear(dim * ff_mult, dim), nn.Dropout(drop)
        )
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
        self.conv = nn.Sequential(
            nn.GroupNorm(1, dim), nn.Conv1d(dim, 2 * dim, 1), nn.GLU(dim=1),
            nn.Conv1d(dim, dim, conv_k, padding=conv_k // 2, groups=dim),
            nn.GroupNorm(1, dim), nn.SiLU(), nn.Conv1d(dim, dim, 1), nn.Dropout(drop)
        )
        self.ff2 = nn.Sequential(
            nn.Linear(dim, dim * ff_mult), nn.GELU(), nn.Dropout(drop),
            nn.Linear(dim * ff_mult, dim), nn.Dropout(drop)
        )
        self.drop = drop
        self.rel_pos_bias = rel_pos_bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, D, T = x.shape
        s = x.transpose(1, 2)
        s = s + 0.5 * self.ff1(s)

        qkv = self.qkv(s).view(B, T, 3, self.heads, self.head_dim)
        q, k, v = [t.transpose(1, 2) for t in qkv.unbind(2)]

        if self.rel_pos_bias is not None:
            q_scaled = q / self.attn_scale
            attn = torch.matmul(q_scaled, k.transpose(-2, -1))
            attn = (attn - attn.max(dim=-1, keepdim=True).values) * self.attn_scale
            attn = attn / math.sqrt(self.head_dim) + self.rel_pos_bias(q, T)
            attn = F.dropout(F.softmax(attn, dim=-1), p=self.drop, training=self.training)
            out = torch.matmul(attn, v).transpose(1, 2).reshape(B, T, D)
        else:
            out = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.drop if self.training else 0
            ).transpose(1, 2).reshape(B, T, D)

        s = s + self.out(out)
        c = s.transpose(1, 2)
        c = c + self.conv(c)
        s = c.transpose(1, 2) + 0.5 * self.ff2(c.transpose(1, 2))
        return s.transpose(1, 2)


# =============================================================================
# Cluster Head
# =============================================================================

class StrongClusterHead(nn.Module):
    def __init__(self, dim: int, K: int, hidden_mult: int = 2, n_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        hidden = dim * hidden_mult
        self.in_proj = nn.Linear(dim, hidden)
        self.in_norm = nn.LayerNorm(hidden)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, hidden),
                nn.Dropout(dropout),
            )
            for _ in range(n_layers)
        ])
        self.out_norm = nn.LayerNorm(hidden)
        self.out_proj = nn.Linear(hidden, K)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = z.permute(0, 2, 1)  # [B, T, C]
        x = F.gelu(self.in_norm(self.in_proj(x)))
        for block in self.blocks:
            x = x + block(x)
        return self.out_proj(self.out_norm(x))


# =============================================================================
# Online Encoder
# =============================================================================

class OnlineEncoder(nn.Module):
    def __init__(
        self,
        code_dim: int = 512,
        channels: list = [32, 64, 128, 256],
        strides: list = [8, 8, 5],
        n_res: int = 4,
        n_conformer: int = 4,
        heads: int = 32,
        K: int = 1024,
        use_gaatn: bool = False,
        use_rel_pos: bool = True,
    ):
        super().__init__()
        self.hop = math.prod(strides)
        self.code_dim = code_dim
        self.use_gaatn = use_gaatn
        self.use_rel_pos = use_rel_pos

        self.layer_attention = AttentionLayerAggregation(
            code_dim=code_dim,
            num_layers=n_conformer + 1,
            num_heads=4,
        )

        self.input_conv = nn.Conv1d(1, channels[0], 7, padding=3)
        self.encoder = nn.ModuleList([
            EncoderBlock(channels[i], channels[i + 1], strides[i], n_res, use_gaatn)
            for i in range(len(strides))
        ])
        self.proj = nn.Conv1d(channels[-1], code_dim, 1)

        head_dim = code_dim // heads
        self.rel_pos_bias = RelativePositionBias(heads, head_dim) if use_rel_pos else None

        self.conformers = nn.ModuleList([
            ConformerBlock(code_dim, heads, rel_pos_bias=self.rel_pos_bias)
            for _ in range(n_conformer)
        ])

        self.cluster_head = StrongClusterHead(code_dim, K, hidden_mult=2, n_layers=3, dropout=0.1)
        self.mask_token = nn.Parameter(torch.randn(1, code_dim, 1) * 0.02)
        self.predictor = nn.Sequential(
            nn.Conv1d(code_dim, code_dim, 1), nn.GELU(),
            ConformerBlock(code_dim, heads, rel_pos_bias=self.rel_pos_bias),
            nn.Conv1d(code_dim, code_dim, 1),
        )

    def encode(self, wav: torch.Tensor, return_layers: bool = False):
        x = self.input_conv(wav)
        for enc in self.encoder:
            x = enc(x)
        z = self.proj(x)

        if not return_layers:
            for conf in self.conformers:
                z = conf(z)
            return z

        layers = [z]
        for conf in self.conformers:
            z = conf(z)
            layers.append(z)
        return z, layers

    def forward(self, wav: torch.Tensor, mask: torch.Tensor):
        z_final, layers = self.encode(wav, return_layers=True)
        z_agg = self.layer_attention(layers)

        B, C, T = z_final.shape
        mask_3d = mask.unsqueeze(1)
        z_masked = z_agg * mask_3d + self.mask_token.expand(B, -1, T) * (1 - mask_3d)
        z_pred = self.predictor(z_masked)

        return z_agg, z_pred, self.cluster_head(z_agg)


# =============================================================================
# Target Encoder
# =============================================================================

class TargetEncoder(nn.Module):
    def __init__(
        self,
        code_dim: int = 512,
        channels: list = [32, 64, 128, 256],
        strides: list = [8, 8, 5],
        n_res: int = 4,
        n_conformer: int = 4,
        heads: int = 32,
        use_gaatn: bool = False,
        use_rel_pos: bool = True,
    ):
        super().__init__()

        self.layer_attention = AttentionLayerAggregation(
            code_dim=code_dim,
            num_layers=n_conformer + 1,
            num_heads=4,
        )

        self.input_conv = nn.Conv1d(1, channels[0], 7, padding=3)
        self.encoder = nn.ModuleList([
            EncoderBlock(channels[i], channels[i + 1], strides[i], n_res, use_gaatn)
            for i in range(len(strides))
        ])
        self.proj = nn.Conv1d(channels[-1], code_dim, 1)

        head_dim = code_dim // heads
        self.rel_pos_bias = RelativePositionBias(heads, head_dim) if use_rel_pos else None

        self.conformers = nn.ModuleList([
            ConformerBlock(code_dim, heads, rel_pos_bias=self.rel_pos_bias)
            for _ in range(n_conformer)
        ])

        for p in self.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, wav: torch.Tensor, return_layers: bool = False):
        x = self.input_conv(wav)
        for enc in self.encoder:
            x = enc(x)
        z = self.proj(x)

        layers = [z]
        for conf in self.conformers:
            z = conf(z)
            layers.append(z)

        z_agg = self.layer_attention(layers)

        if return_layers:
            return z_agg, layers
        return z_agg


# =============================================================================
# Loading Utilities
# =============================================================================

def get_default_config(use_gaatn: bool = True, use_rel_pos: bool = True) -> dict:
    return {
        "sample_rate": 16000,
        "code_dim": 512,
        "channels": [32, 64, 128, 256],
        "strides": [8, 8, 5],
        "n_res": 4,
        "n_conformer": 4,
        "heads": 32,
        "K": 1024,
        "use_gaatn": use_gaatn,
        "use_rel_pos": use_rel_pos,
    }


def load_encoder(
    checkpoint_path: str,
    config: Optional[dict] = None,
    device: str = "cuda",
    strict: bool = False,
    use_gaatn: bool = True,
    use_rel_pos: bool = True,
) -> OnlineEncoder:
    """Load encoder from checkpoint."""
    if config is None:
        config = get_default_config(use_gaatn, use_rel_pos)

    model = OnlineEncoder(
        code_dim=config.get('code_dim', 512),
        channels=config.get('channels', [32, 64, 128, 256]),
        strides=config.get('strides', [8, 8, 5]),
        n_res=config.get('n_res', 4),
        n_conformer=config.get('n_conformer', 4),
        heads=config.get('heads', 32),
        K=config.get('K', 1024),
        use_gaatn=config.get('use_gaatn', use_gaatn),
        use_rel_pos=config.get('use_rel_pos', use_rel_pos),
    )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

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

    # Remove 'module.' prefix if present
    cleaned = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}

    model.load_state_dict(cleaned, strict=strict)
    model.to(device)
    model.eval()

    return model


def load_target_encoder(
    checkpoint_path: str,
    config: Optional[dict] = None,
    device: str = "cuda",
    use_gaatn: bool = True,
    use_rel_pos: bool = True,
) -> TargetEncoder:
    """Load target encoder from checkpoint."""
    if config is None:
        config = get_default_config(use_gaatn, use_rel_pos)

    model = TargetEncoder(
        code_dim=config.get('code_dim', 512),
        channels=config.get('channels', [32, 64, 128, 256]),
        strides=config.get('strides', [8, 8, 5]),
        n_res=config.get('n_res', 4),
        n_conformer=config.get('n_conformer', 4),
        heads=config.get('heads', 32),
        use_gaatn=config.get('use_gaatn', use_gaatn),
        use_rel_pos=config.get('use_rel_pos', use_rel_pos),
    )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['target'])
    model.to(device)
    model.eval()

    return model


def print_model_stats(model: nn.Module, name: str = "Model"):
    """Print model parameter statistics."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n{'=' * 60}")
    print(f"{name}: {total / 1e6:.1f}M params ({trainable / 1e6:.1f}M trainable)")
    print(f"{'=' * 60}")

    for n, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        print(f"  {n:<20} {params / 1e6:>6.2f}M ({params / total * 100:>5.1f}%)")


# Aliases
JEPAEncoder = OnlineEncoder
load_jepa_encoder = load_encoder


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use_gaatn", action="store_true")
    parser.add_argument("--use_rel_pos", action="store_true", default=True)
    args = parser.parse_args()

    print(f"Loading model from {args.checkpoint}...")
    model = load_encoder(args.checkpoint, device=args.device, use_gaatn=args.use_gaatn, use_rel_pos=args.use_rel_pos)
    print_model_stats(model, "OnlineEncoder")

    print("\nTesting with random input...")
    with torch.no_grad():
        dummy = torch.randn(1, 1, 16000 * 2).to(args.device)
        z = model.encode(dummy)
        print(f"Input: {dummy.shape} -> Output: {z.shape}")

    print("\nDone!")
