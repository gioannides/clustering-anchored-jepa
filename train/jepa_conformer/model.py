"""Encoder architectures for JEPA training."""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Gaussian Adaptive Attention
# =============================================================================

class GaussianAdaptiveAttention(nn.Module):
    """Single-head Gaussian adaptive attention."""
    
    def __init__(self, norm_axis: int, num_heads: int, num_gaussians: int, eps: float = 1e-8):
        super().__init__()
        self.norm_axis = norm_axis
        self.eps = eps
        self.num_gaussians = num_gaussians
        self.mean_offsets = nn.Parameter(torch.zeros(num_gaussians))
        self.c = nn.Parameter(torch.exp(torch.randn(num_gaussians)))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_f = x.float()
        mean = x_f.mean(dim=self.norm_axis, keepdim=True)
        var = x_f.var(dim=self.norm_axis, keepdim=True) + self.eps
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
        
        log_mixture = torch.nan_to_num(log_mixture, nan=-1e9, posinf=-1e9, neginf=-1e9)
        log_denom = torch.logsumexp(log_mixture, dim=self.norm_axis, keepdim=True)
        mixture = torch.exp(log_mixture - log_denom).to(x.dtype)
        mixture = torch.nan_to_num(mixture, nan=0.0, posinf=0.0, neginf=0.0)
        
        return x * mixture


class MultiHeadGaussianAdaptiveAttention(nn.Module):
    """Multi-head Gaussian adaptive attention."""
    
    def __init__(self, norm_axis: int, num_heads: int, num_gaussians: int, eps: float = 1e-8):
        super().__init__()
        self.norm_axis = norm_axis
        self.num_heads = num_heads
        self.heads = nn.ModuleList([
            GaussianAdaptiveAttention(norm_axis, num_heads, num_gaussians, eps)
            for _ in range(num_heads)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        chunk_size = x.shape[self.norm_axis] // self.num_heads
        outputs = []
        
        for i, head in enumerate(self.heads):
            start = i * chunk_size
            end = start + chunk_size if i < self.num_heads - 1 else x.shape[self.norm_axis]
            chunk = x.narrow(self.norm_axis, start, end - start)
            outputs.append(head(chunk))
        
        return torch.cat(outputs, dim=self.norm_axis)


class GaussianBlock(nn.Module):
    """Block of Gaussian adaptive attention layers."""
    
    def __init__(
        self,
        norm_axes: list,
        num_heads: list,
        num_gaussians: list,
        num_layers: int,
        eps: float = 1e-8,
        residual: bool = True
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            MultiHeadGaussianAdaptiveAttention(norm_axes[i], num_heads[i], num_gaussians[i], eps)
            for i in range(num_layers)
        ])
        self.residual = residual
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            out = layer(x)
            x = out + x if self.residual else out
        return x


# =============================================================================
# Layer Aggregation
# =============================================================================

class AttentionLayerAggregation(nn.Module):
    """Cross-attention layer aggregation."""
    
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
        
        stacked = torch.stack(layers, dim=1)
        pooled = stacked.mean(dim=-1)
        
        q = self.q_proj(pooled[:, -1:, :])
        k = self.k_proj(pooled)
        v = self.v_proj(pooled)
        
        q = q.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, self.num_layers, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, self.num_layers, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)
        
        layer_weights = attn.mean(dim=1).squeeze(1)
        z_agg = (stacked * layer_weights[:, :, None, None]).sum(dim=1)
        
        return z_agg


# =============================================================================
# Encoder Components
# =============================================================================

class SnakeBeta(nn.Module):
    """Snake activation with learnable frequency."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.a = nn.Parameter(torch.zeros(1, channels, 1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = F.softplus(self.a) + 1e-2
        return x + (torch.sin(a * x) ** 2) / a.clamp_min(1e-2)


class ResBlock(nn.Module):
    """Residual block with dilated convolutions."""
    
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
    """Strided encoder block."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        n_res: int = 2,
        use_gaatn: bool = False
    ):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, 2 * stride, stride=stride, padding=stride // 2)
        self.res = nn.ModuleList([ResBlock(out_channels, 3, (1, 3**i, 5**i)) for i in range(n_res)])
        self.snake = SnakeBeta(out_channels)
        self.use_gaatn = use_gaatn
        
        if use_gaatn:
            self.gaatn = GaussianBlock(
                norm_axes=[2],
                num_heads=[4],
                num_gaussians=[4],
                num_layers=1,
                residual=True
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.snake(self.conv(x))
        for r in self.res:
            x = r(x)
        if self.use_gaatn:
            x = self.gaatn(x)
        return x


class RelativePositionBias(nn.Module):
    """Gated relative position bias (WavLM-style)."""
    
    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        num_buckets: int = 320,
        max_distance: int = 800
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        
        self.rel_pos_embed = nn.Embedding(num_buckets, num_heads)
        self.gate_u_vec = nn.Parameter(torch.randn(num_heads, head_dim) * 0.02)
        self.gate_w_vec = nn.Parameter(torch.randn(num_heads, head_dim) * 0.02)
        self.gate_scale_scalar = nn.Parameter(torch.ones(num_heads))
    
    def _relative_position_bucket(self, relative_position: torch.Tensor) -> torch.Tensor:
        """Convert relative position to bucket index."""
        half_buckets = self.num_buckets // 2
        
        sign = (relative_position >= 0).long()
        rel_pos_abs = relative_position.abs()
        
        small_threshold = half_buckets // 2
        is_small = rel_pos_abs < small_threshold
        
        rel_pos_abs_clamped = rel_pos_abs.clamp(min=1)
        log_ratio = torch.log(rel_pos_abs_clamped.float() / small_threshold) / math.log(self.max_distance / small_threshold)
        log_pos = small_threshold + (log_ratio * (half_buckets - small_threshold)).long()
        log_pos = log_pos.clamp(max=half_buckets - 1)
        
        bucket = torch.where(is_small, rel_pos_abs, log_pos)
        bucket = bucket + sign * half_buckets
        bucket = bucket.clamp(0, self.num_buckets - 1)
        
        return bucket
    
    def forward(self, q: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Compute gated relative position bias."""
        device = q.device
        dtype = q.dtype
        B, H, T, D = q.shape
        
        positions = torch.arange(seq_len, device=device)
        relative_pos = positions.unsqueeze(0) - positions.unsqueeze(1)
        
        buckets = self._relative_position_bucket(relative_pos)
        d = self.rel_pos_embed(buckets).to(dtype).permute(2, 0, 1).unsqueeze(0)
        
        g_update = torch.sigmoid(torch.einsum('bhtd,hd->bht', q, self.gate_u_vec)).unsqueeze(-1)
        g_reset = torch.sigmoid(torch.einsum('bhtd,hd->bht', q, self.gate_w_vec)).unsqueeze(-1)
        
        scale = self.gate_scale_scalar.view(1, H, 1, 1)
        
        r_tilde = scale * g_reset * d
        r = d + g_update * d + (1 - g_update) * r_tilde
        
        return r


class ConformerBlock(nn.Module):
    """Conformer block with optional relative position bias."""
    
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        ff_mult: int = 4,
        conv_kernel: int = 31,
        dropout: float = 0.1,
        rel_pos_bias: RelativePositionBias = None
    ):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        
        self.ff1 = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ff_mult, dim),
            nn.Dropout(dropout)
        )
        
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
        
        self.conv = nn.Sequential(
            nn.GroupNorm(1, dim),
            nn.Conv1d(dim, 2 * dim, 1),
            nn.GLU(dim=1),
            nn.Conv1d(dim, dim, conv_kernel, padding=conv_kernel // 2, groups=dim),
            nn.GroupNorm(1, dim),
            nn.SiLU(),
            nn.Conv1d(dim, dim, 1),
            nn.Dropout(dropout)
        )
        
        self.ff2 = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ff_mult, dim),
            nn.Dropout(dropout)
        )
        
        self.dropout = dropout
        self.rel_pos_bias = rel_pos_bias
        self.attn_scale = 32.0
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, D, T = x.shape
        s = x.transpose(1, 2)
        
        # First FFN
        s = s + 0.5 * self.ff1(s)
        
        # Self-attention
        qkv = self.qkv(s).view(B, T, 3, self.heads, self.head_dim)
        q, k, v = [t.transpose(1, 2) for t in qkv.unbind(2)]
        
        if self.rel_pos_bias is not None:
            scale = self.attn_scale
            q_scaled = q / scale
            attn_scores = torch.matmul(q_scaled, k.transpose(-2, -1))
            attn_scores = (attn_scores - attn_scores.max(dim=-1, keepdim=True).values) * scale
            attn_scores = attn_scores / math.sqrt(self.head_dim)
            attn_scores = attn_scores + self.rel_pos_bias(q, T)
            attn_probs = F.softmax(attn_scores, dim=-1)
            attn_probs = F.dropout(attn_probs, p=self.dropout, training=self.training)
            attn_out = torch.matmul(attn_probs, v).transpose(1, 2).reshape(B, T, D)
        else:
            attn_out = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.dropout if self.training else 0
            ).transpose(1, 2).reshape(B, T, D)
        
        s = s + self.out(attn_out)
        
        # Convolution
        c = s.transpose(1, 2)
        c = c + self.conv(c)
        
        # Second FFN
        s = c.transpose(1, 2) + 0.5 * self.ff2(c.transpose(1, 2))
        
        return s.transpose(1, 2)


# =============================================================================
# Cluster Head
# =============================================================================

class ClusterHead(nn.Module):
    """MLP cluster prediction head."""
    
    def __init__(
        self,
        dim: int,
        K: int,
        hidden_mult: int = 2,
        n_layers: int = 3,
        dropout: float = 0.1
    ):
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
        x = z.permute(0, 2, 1)
        x = F.gelu(self.in_norm(self.in_proj(x)))
        
        for block in self.blocks:
            x = x + block(x)
        
        return self.out_proj(self.out_norm(x))


# =============================================================================
# Main Encoders
# =============================================================================

class OnlineEncoder(nn.Module):
    """Online encoder with predictor and cluster head."""
    
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
        use_rel_pos: bool = True
    ):
        super().__init__()
        self.hop = math.prod(strides)
        self.code_dim = code_dim
        
        # Layer aggregation
        self.layer_attention = AttentionLayerAggregation(
            code_dim=code_dim,
            num_layers=n_conformer + 1,
            num_heads=4,
        )
        
        # Encoder
        self.input_conv = nn.Conv1d(1, channels[0], 7, padding=3)
        self.encoder = nn.ModuleList([
            EncoderBlock(channels[i], channels[i + 1], strides[i], n_res, use_gaatn)
            for i in range(len(strides))
        ])
        self.proj = nn.Conv1d(channels[-1], code_dim, 1)
        
        # Relative position bias
        head_dim = code_dim // heads
        self.rel_pos_bias = RelativePositionBias(heads, head_dim) if use_rel_pos else None
        
        # Conformer stack
        self.conformers = nn.ModuleList([
            ConformerBlock(code_dim, heads, rel_pos_bias=self.rel_pos_bias)
            for _ in range(n_conformer)
        ])
        
        # Heads
        self.cluster_head = ClusterHead(code_dim, K)
        self.mask_token = nn.Parameter(torch.randn(1, code_dim, 1) * 0.02)
        
        # Predictor
        self.predictor = nn.Sequential(
            nn.Conv1d(code_dim, code_dim, 1),
            nn.GELU(),
            ConformerBlock(code_dim, heads, rel_pos_bias=self.rel_pos_bias),
            nn.Conv1d(code_dim, code_dim, 1),
        )
    
    def encode(self, wav: torch.Tensor, return_layers: bool = False):
        """Encode waveform to latent representation."""
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
        """Forward pass with masking."""
        z_final, layers = self.encode(wav, return_layers=True)
        z_agg = self.layer_attention(layers)
        
        B, C, T = z_final.shape
        mask_3d = mask.unsqueeze(1)
        z_masked = z_agg * mask_3d + self.mask_token.expand(B, -1, T) * (1 - mask_3d)
        z_pred = self.predictor(z_masked)
        
        return z_agg, z_pred, self.cluster_head(z_agg)


class TargetEncoder(nn.Module):
    """Target encoder (EMA copy, no gradients)."""
    
    def __init__(
        self,
        code_dim: int = 512,
        channels: list = [32, 64, 128, 256],
        strides: list = [8, 8, 5],
        n_res: int = 4,
        n_conformer: int = 4,
        heads: int = 32,
        use_gaatn: bool = False,
        use_rel_pos: bool = True
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
        
        # Freeze parameters
        for p in self.parameters():
            p.requires_grad = False
    
    @torch.no_grad()
    def forward(self, wav: torch.Tensor, return_layers: bool = False):
        """Forward pass (no gradients)."""
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
# Frozen GMM
# =============================================================================

class FrozenGMM:
    """Frozen GMM for soft cluster targets."""
    
    def __init__(self, K: int, dim: int, device: str = 'cuda', temperature: float = 1.0):
        self.K = K
        self.dim = dim
        self.device = device
        self.temperature = temperature
        self.means = None
        self.covariances = None
        self.weights = None
    
    @torch.no_grad()
    def soft_assign(self, x: torch.Tensor, batch_size: int = 500) -> torch.Tensor:
        """Compute soft posterior probabilities."""
        if x.dim() == 3:
            B, C, T = x.shape
            x_flat = x.permute(0, 2, 1).reshape(-1, C).float()
        else:
            x_flat = x.float()
        
        N = x_flat.shape[0]
        probs = []
        
        for i in range(0, N, batch_size):
            probs.append(self._soft_assign_batch(x_flat[i:i + batch_size]))
        
        return torch.cat(probs)
    
    def _soft_assign_batch(self, x: torch.Tensor, chunk_k: int = 64) -> torch.Tensor:
        """Chunked soft assignment."""
        log_weights = torch.log(self.weights + 1e-10)
        log_prob_chunks = []
        
        for k_start in range(0, self.K, chunk_k):
            k_end = min(k_start + chunk_k, self.K)
            
            means_chunk = self.means[k_start:k_end]
            cov_chunk = self.covariances[k_start:k_end]
            weights_chunk = log_weights[k_start:k_end]
            
            diff = x.unsqueeze(1) - means_chunk.unsqueeze(0)
            inv_cov = 1.0 / (cov_chunk + 1e-6)
            mahal = ((diff ** 2) * inv_cov.unsqueeze(0)).sum(dim=-1)
            log_det = torch.log(cov_chunk + 1e-6).sum(dim=-1)
            
            log_prob_chunk = -0.5 * (self.dim * np.log(2 * np.pi) + log_det.unsqueeze(0) + mahal)
            log_prob_chunk = log_prob_chunk + weights_chunk.unsqueeze(0)
            log_prob_chunks.append(log_prob_chunk)
        
        log_prob = torch.cat(log_prob_chunks, dim=1)
        return F.softmax(log_prob / self.temperature, dim=1)
    
    @classmethod
    def load(cls, path: str, device: str = 'cuda', temperature: float = None) -> 'FrozenGMM':
        """Load GMM from checkpoint."""
        data = torch.load(path, map_location=device, weights_only=False)
        temp = temperature if temperature is not None else data.get('temperature', 1.0)
        
        gmm = cls(data['K'], data['dim'], device=device, temperature=temp)
        gmm.means = data['means'].to(device)
        gmm.covariances = data['covariances'].to(device)
        gmm.weights = data['weights'].to(device)
        
        return gmm
