"""WavLM encoder using Conformer architecture."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Normalization (bf16 safe)
# =============================================================================

class Fp32GroupNorm(nn.GroupNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.group_norm(
            x.float(), self.num_groups,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps
        ).to(x.dtype)


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
# Relative Position Bias
# =============================================================================

class RelativePositionBias(nn.Module):
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

        g_u = torch.sigmoid(torch.einsum('bhtd,hd->bht', q, self.gate_u)).unsqueeze(-1)
        g_w = torch.sigmoid(torch.einsum('bhtd,hd->bht', q, self.gate_w)).unsqueeze(-1)
        s = self.scale.view(1, H, 1, 1)

        return d + g_u * d + (1 - g_u) * s * g_w * d


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
# WavLM Encoder
# =============================================================================

class WavLMEncoder(nn.Module):
    """WavLM-style encoder with masked prediction head."""

    def __init__(
        self,
        code_dim: int = 512,
        channels: list = [32, 64, 128, 256],
        strides: list = [8, 8, 5],
        n_res: int = 4,
        n_conformer: int = 4,
        heads: int = 32,
        num_classes: int = 1024,
        use_gaatn: bool = False,
        use_rel_pos: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hop = math.prod(strides)
        self.code_dim = code_dim

        # CNN frontend
        self.input_conv = nn.Conv1d(1, channels[0], 7, padding=3)
        self.encoder = nn.ModuleList([
            EncoderBlock(channels[i], channels[i + 1], strides[i], n_res, use_gaatn)
            for i in range(len(strides))
        ])
        self.proj = nn.Conv1d(channels[-1], code_dim, 1)

        # Conformer layers
        head_dim = code_dim // heads
        self.rel_pos_bias = RelativePositionBias(heads, head_dim) if use_rel_pos else None
        self.conformers = nn.ModuleList([
            ConformerBlock(code_dim, heads, rel_pos_bias=self.rel_pos_bias)
            for _ in range(n_conformer)
        ])

        # Mask embedding
        self.mask_emb = nn.Parameter(torch.zeros(code_dim))
        nn.init.uniform_(self.mask_emb, -0.1, 0.1)

        # Classification head
        self.final_proj = nn.Linear(code_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, wav: torch.Tensor, mask: torch.Tensor = None, return_features: bool = False):
        """
        Args:
            wav: [B, 1, T] waveform
            mask: [B, T_frames] where 1=keep, 0=mask
            return_features: return features without classification
        Returns:
            logits: [B, T, num_classes]
            features: [B, T, code_dim]
        """
        x = self.input_conv(wav)
        for enc in self.encoder:
            x = enc(x)
        x = self.proj(x).transpose(1, 2)  # [B, T, C]
        T = x.shape[1]

        # Apply masking
        if mask is not None:
            mask_exp = mask[:, :T].unsqueeze(-1).to(x.dtype)
            x = x * mask_exp + self.mask_emb.to(x.dtype) * (1 - mask_exp)

        x = self.dropout(x).transpose(1, 2)

        for conf in self.conformers:
            x = conf(x)

        x = x.transpose(1, 2)  # [B, T, C]

        if return_features:
            return x

        return self.final_proj(x), x
