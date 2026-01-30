"""WavLM-style Transformer encoder for JEPA training."""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# FP32 Normalization (for bf16 stability)
# =============================================================================

class Fp32GroupNorm(nn.GroupNorm):
    """GroupNorm that casts to float32 internally."""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        return F.group_norm(
            x.float(),
            self.num_groups,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps
        ).to(orig_dtype)


class Fp32LayerNorm(nn.LayerNorm):
    """LayerNorm that casts to float32 internally."""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        return F.layer_norm(
            x.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps
        ).to(orig_dtype)


# =============================================================================
# CNN Feature Extractor
# =============================================================================

class ConvFeatureExtractor(nn.Module):
    """
    WavLM/HuBERT CNN frontend.
    
    7 conv layers: first with k=10,s=5, then 6 with k=3,s=2.
    Total stride = 5 * 2^6 = 320 (20ms @ 16kHz)
    """
    
    def __init__(self, conv_dim: int = 512):
        super().__init__()
        
        self.conv_layers = nn.ModuleList()
        
        # First conv: large kernel
        self.conv_layers.append(nn.Sequential(
            nn.Conv1d(1, conv_dim, kernel_size=10, stride=5, bias=False),
            Fp32GroupNorm(1, conv_dim),
            nn.GELU(),
        ))
        
        # Remaining 6 convs: smaller kernel, stride 2
        for _ in range(6):
            self.conv_layers.append(nn.Sequential(
                nn.Conv1d(conv_dim, conv_dim, kernel_size=3, stride=2, bias=False),
                Fp32GroupNorm(1, conv_dim),
                nn.GELU(),
            ))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 1, T] waveform
        Returns:
            [B, conv_dim, T'] features
        """
        for conv in self.conv_layers:
            x = conv(x)
        return x


# =============================================================================
# Gated Relative Position Bias
# =============================================================================

class GatedRelativePositionBias(nn.Module):
    """WavLM-style gated relative position bias."""
    
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
        self.gate_ur = nn.Linear(head_dim, num_heads, bias=False)
        self.gate_i = nn.Linear(head_dim, num_heads, bias=False)
        self.scale = nn.Parameter(torch.ones(num_heads))
        
        nn.init.xavier_uniform_(self.gate_ur.weight, gain=0.1)
        nn.init.xavier_uniform_(self.gate_i.weight, gain=0.1)
    
    def _relative_position_bucket(self, relative_position: torch.Tensor) -> torch.Tensor:
        """Logarithmic bucketing for relative positions."""
        half_buckets = self.num_buckets // 2
        
        sign = (relative_position >= 0).long()
        rel_pos_abs = relative_position.abs()
        
        small_threshold = half_buckets // 2
        is_small = rel_pos_abs < small_threshold
        
        log_ratio = torch.log(rel_pos_abs.float().clamp(min=1) / small_threshold) / math.log(self.max_distance / small_threshold)
        log_pos = small_threshold + (log_ratio * (half_buckets - small_threshold)).long()
        log_pos = log_pos.clamp(max=half_buckets - 1)
        
        bucket = torch.where(is_small, rel_pos_abs, log_pos)
        bucket = bucket + sign * half_buckets
        return bucket.clamp(0, self.num_buckets - 1)
    
    def forward(self, q: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        Args:
            q: [B, heads, T, head_dim]
        Returns:
            [B, heads, T, T] position bias
        """
        device = q.device
        B, H, T, D = q.shape
        
        # Relative position matrix
        positions = torch.arange(seq_len, device=device)
        rel_pos = positions.unsqueeze(0) - positions.unsqueeze(1)
        buckets = self._relative_position_bucket(rel_pos)
        
        # Position embeddings: [T, T, heads] -> [1, heads, T, T]
        pos_embed = self.rel_pos_embed(buckets)
        pos_embed = pos_embed.permute(2, 0, 1).unsqueeze(0)
        
        # Gating based on query
        q_mean = q.mean(dim=2)  # [B, H, D]
        q_for_gate = q_mean.transpose(1, 2)  # [B, D, H]
        
        gate_r = torch.sigmoid(self.gate_ur(q_for_gate.mean(dim=-1)))  # [B, H]
        gate_u = torch.sigmoid(self.gate_i(q_for_gate.mean(dim=-1)))   # [B, H]
        
        gate_r = gate_r.view(B, H, 1, 1)
        gate_u = gate_u.view(B, H, 1, 1)
        scale = self.scale.view(1, H, 1, 1)
        
        # Gated position bias
        gated_pos = scale * gate_r * pos_embed
        output = pos_embed + gate_u * gated_pos
        
        return output


# =============================================================================
# Transformer Layer
# =============================================================================

class TransformerLayer(nn.Module):
    """Transformer layer with gated relative position bias."""
    
    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        ff_dim: int = 3072,
        dropout: float = 0.1,
        rel_pos_bias: GatedRelativePositionBias = None
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Self-attention
        self.self_attn_layer_norm = Fp32LayerNorm(embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # FFN
        self.final_layer_norm = Fp32LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.rel_pos_bias = rel_pos_bias
        self.scale = self.head_dim ** -0.5
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, C]
        Returns:
            [B, T, C]
        """
        B, T, C = x.shape
        
        # Self-attention with pre-norm
        residual = x
        x = self.self_attn_layer_norm(x)
        
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Add relative position bias
        if self.rel_pos_bias is not None:
            attn = attn + self.rel_pos_bias(q, T)
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, T, C)
        out = self.out_proj(out)
        out = self.dropout(out)
        x = residual + out
        
        # FFN with pre-norm
        residual = x
        x = self.final_layer_norm(x)
        x = F.gelu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = residual + x
        
        return x


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
        """
        Args:
            z: [B, C, T]
        Returns:
            [B, T, K] logits
        """
        x = z.permute(0, 2, 1)  # [B, T, C]
        x = F.gelu(self.in_norm(self.in_proj(x)))
        
        for block in self.blocks:
            x = x + block(x)
        
        return self.out_proj(self.out_norm(x))


# =============================================================================
# Main Encoders
# =============================================================================

class OnlineEncoder(nn.Module):
    """WavLM-based online encoder for JEPA."""
    
    def __init__(
        self,
        code_dim: int = 768,
        conv_dim: int = 512,
        num_heads: int = 12,
        ff_dim: int = 3072,
        num_layers: int = 12,
        dropout: float = 0.1,
        K: int = 1024
    ):
        super().__init__()
        
        self.hop = 320  # WavLM fixed stride
        self.code_dim = code_dim
        
        # CNN frontend
        self.feature_extractor = ConvFeatureExtractor(conv_dim)
        
        # Project to transformer dim
        self.post_extract_proj = nn.Linear(conv_dim, code_dim)
        self.layer_norm = Fp32LayerNorm(code_dim)
        self.dropout_module = nn.Dropout(dropout)
        
        # Mask embedding
        self.mask_emb = nn.Parameter(torch.zeros(code_dim))
        nn.init.uniform_(self.mask_emb, -0.1, 0.1)
        
        # Shared relative position bias
        head_dim = code_dim // num_heads
        self.rel_pos_bias = GatedRelativePositionBias(num_heads, head_dim)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(code_dim, num_heads, ff_dim, dropout, self.rel_pos_bias)
            for _ in range(num_layers)
        ])
        
        # Cluster head
        self.cluster_head = ClusterHead(code_dim, K, hidden_mult=2, n_layers=3, dropout=0.1)
        
        # Predictor
        self.mask_token = nn.Parameter(torch.randn(1, code_dim, 1) * 0.02)
        self.predictor = nn.Sequential(
            nn.Conv1d(code_dim, code_dim, 1),
            nn.GELU(),
            nn.Conv1d(code_dim, code_dim, 1),
        )
    
    def encode(self, wav: torch.Tensor) -> torch.Tensor:
        """
        Args:
            wav: [B, 1, T]
        Returns:
            [B, code_dim, T']
        """
        x = self.feature_extractor(wav)  # [B, conv_dim, T']
        x = x.transpose(1, 2)  # [B, T', conv_dim]
        x = self.post_extract_proj(x)  # [B, T', code_dim]
        x = self.layer_norm(x)
        x = self.dropout_module(x)
        
        for layer in self.layers:
            x = layer(x)
        
        return x.transpose(1, 2)  # [B, code_dim, T']
    
    def forward(self, wav: torch.Tensor, mask: torch.Tensor):
        """
        Args:
            wav: [B, 1, T]
            mask: [B, T'] where 1=keep, 0=mask
        Returns:
            z: [B, code_dim, T']
            z_pred: [B, code_dim, T']
            cluster_logits: [B, T', K]
        """
        z = self.encode(wav)
        B, C, T = z.shape
        
        # Apply mask
        mask_3d = mask.unsqueeze(1)
        z_masked = z * mask_3d + self.mask_token.expand(B, -1, T) * (1 - mask_3d)
        z_pred = self.predictor(z_masked)
        
        return z, z_pred, self.cluster_head(z)


class TargetEncoder(nn.Module):
    """WavLM-based target encoder (EMA copy, no gradients)."""
    
    def __init__(
        self,
        code_dim: int = 768,
        conv_dim: int = 512,
        num_heads: int = 12,
        ff_dim: int = 3072,
        num_layers: int = 12,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.hop = 320
        
        self.feature_extractor = ConvFeatureExtractor(conv_dim)
        self.post_extract_proj = nn.Linear(conv_dim, code_dim)
        self.layer_norm = Fp32LayerNorm(code_dim)
        self.dropout_module = nn.Dropout(dropout)
        
        head_dim = code_dim // num_heads
        self.rel_pos_bias = GatedRelativePositionBias(num_heads, head_dim)
        
        self.layers = nn.ModuleList([
            TransformerLayer(code_dim, num_heads, ff_dim, dropout, self.rel_pos_bias)
            for _ in range(num_layers)
        ])
        
        # Freeze parameters
        for p in self.parameters():
            p.requires_grad = False
    
    @torch.no_grad()
    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """
        Args:
            wav: [B, 1, T]
        Returns:
            [B, code_dim, T']
        """
        x = self.feature_extractor(wav)
        x = x.transpose(1, 2)
        x = self.post_extract_proj(x)
        x = self.layer_norm(x)
        x = self.dropout_module(x)
        
        for layer in self.layers:
            x = layer(x)
        
        return x.transpose(1, 2)


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
    
    @torch.no_grad()
    def assign(self, x: torch.Tensor, batch_size: int = 50000) -> torch.Tensor:
        """Hard assignment to most likely component."""
        if x.dim() == 3:
            B, C, T = x.shape
            x_flat = x.permute(0, 2, 1).reshape(-1, C).float()
        else:
            x_flat = x.float()
        
        N = x_flat.shape[0]
        if N <= batch_size:
            return self._assign_batch(x_flat)
        
        labels = []
        for i in range(0, N, batch_size):
            labels.append(self._assign_batch(x_flat[i:i + batch_size]))
        return torch.cat(labels)
    
    def _assign_batch(self, x: torch.Tensor) -> torch.Tensor:
        log_weights = torch.log(self.weights + 1e-10)
        diff = x.unsqueeze(1) - self.means.unsqueeze(0)
        inv_cov = 1.0 / (self.covariances + 1e-6)
        mahal = ((diff ** 2) * inv_cov.unsqueeze(0)).sum(dim=-1)
        log_det = torch.log(self.covariances + 1e-6).sum(dim=-1)
        log_prob = -0.5 * (self.dim * np.log(2 * np.pi) + log_det.unsqueeze(0) + mahal)
        log_prob = log_prob + log_weights.unsqueeze(0)
        return log_prob.argmax(dim=1)
    
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
