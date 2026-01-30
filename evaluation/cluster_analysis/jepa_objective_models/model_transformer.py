"""
Model definitions for Transformer-based encoder (WavLM architecture).

Matches the architecture in train/jepa_transformer/model.py.

Usage:
    from model_transformer import load_encoder
    
    model = load_encoder("checkpoint.pt", device="cuda")
    z = model.encode(wav)  # wav: [B, 1, T_samples] -> z: [B, code_dim, T_frames]
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# FP32 Norms (bf16 compatibility)
# =============================================================================

class Fp32GroupNorm(nn.GroupNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.group_norm(
            x.float(), self.num_groups,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps
        ).to(x.dtype)


class Fp32LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x.float(), self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps
        ).to(x.dtype)


# =============================================================================
# WavLM CNN Frontend
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
        for conv in self.conv_layers:
            x = conv(x)
        return x


# =============================================================================
# Gated Relative Position Bias (WavLM)
# =============================================================================

class GatedRelativePositionBias(nn.Module):
    """WavLM-style gated relative position bias."""

    def __init__(self, num_heads: int, head_dim: int, num_buckets: int = 320, max_distance: int = 800):
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
        half = self.num_buckets // 2
        sign = (relative_position >= 0).long()
        abs_pos = relative_position.abs()
        threshold = half // 2

        is_small = abs_pos < threshold
        log_ratio = torch.log(abs_pos.float().clamp(min=1) / threshold) / math.log(self.max_distance / threshold)
        log_pos = (threshold + log_ratio * (half - threshold)).long().clamp(max=half - 1)

        bucket = torch.where(is_small, abs_pos, log_pos) + sign * half
        return bucket.clamp(0, self.num_buckets - 1)

    def forward(self, q: torch.Tensor, seq_len: int) -> torch.Tensor:
        device = q.device
        B, H, T, D = q.shape

        positions = torch.arange(seq_len, device=device)
        rel_pos = positions.unsqueeze(0) - positions.unsqueeze(1)
        buckets = self._relative_position_bucket(rel_pos)

        pos_embed = self.rel_pos_embed(buckets).permute(2, 0, 1).unsqueeze(0)

        q_mean = q.mean(dim=2).transpose(1, 2)
        gate_r = torch.sigmoid(self.gate_ur(q_mean.mean(dim=-1))).view(B, H, 1, 1)
        gate_u = torch.sigmoid(self.gate_i(q_mean.mean(dim=-1))).view(B, H, 1, 1)
        scale = self.scale.view(1, H, 1, 1)

        return pos_embed + gate_u * scale * gate_r * pos_embed


# =============================================================================
# Transformer Layer
# =============================================================================

class TransformerLayer(nn.Module):
    """Single Transformer layer with gated relative position bias."""

    def __init__(self, embed_dim: int = 768, num_heads: int = 12, ff_dim: int = 3072,
                 dropout: float = 0.1, rel_pos_bias: GatedRelativePositionBias = None):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.self_attn_layer_norm = Fp32LayerNorm(embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.final_layer_norm = Fp32LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.rel_pos_bias = rel_pos_bias
        self.scale = self.head_dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        # Self-attention with pre-norm
        residual = x
        x = self.self_attn_layer_norm(x)

        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if self.rel_pos_bias is not None:
            attn = attn + self.rel_pos_bias(q, T)

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, T, C)
        out = self.dropout(self.out_proj(out))
        x = residual + out

        # FFN with pre-norm
        residual = x
        x = self.final_layer_norm(x)
        x = self.dropout(self.fc2(F.gelu(self.fc1(x))))
        return residual + x


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
        return (stacked * layer_weights[:, :, None, None]).sum(dim=1)


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
        x = z.permute(0, 2, 1)
        x = F.gelu(self.in_norm(self.in_proj(x)))
        for block in self.blocks:
            x = x + block(x)
        return self.out_proj(self.out_norm(x))


# =============================================================================
# Online Encoder (Transformer-based)
# =============================================================================

class OnlineEncoder(nn.Module):
    """WavLM-style Transformer encoder."""

    def __init__(
        self,
        code_dim: int = 512,
        conv_dim: int = 256,
        num_heads: int = 8,
        ff_dim: int = 2048,
        num_layers: int = 10,
        dropout: float = 0.1,
        K: int = 1024,
        **kwargs
    ):
        super().__init__()

        self.hop = 320  # Fixed WavLM stride
        self.code_dim = code_dim
        self.conv_dim = conv_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

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

        # Layer aggregation
        self.layer_attention = AttentionLayerAggregation(
            code_dim=code_dim,
            num_layers=num_layers + 1,
            num_heads=4,
        )

        # Cluster head
        self.cluster_head = StrongClusterHead(code_dim, K)

        # Predictor
        self.mask_token = nn.Parameter(torch.randn(1, code_dim, 1) * 0.02)
        self.predictor = nn.Sequential(
            nn.Conv1d(code_dim, code_dim, 1), nn.GELU(),
            nn.Conv1d(code_dim, code_dim, 1),
        )

    def encode(self, wav: torch.Tensor, return_layers: bool = False):
        x = self.feature_extractor(wav).transpose(1, 2)
        x = self.dropout_module(self.layer_norm(self.post_extract_proj(x)))

        if not return_layers:
            for layer in self.layers:
                x = layer(x)
            return x.transpose(1, 2)

        layers = [x.transpose(1, 2)]
        for layer in self.layers:
            x = layer(x)
            layers.append(x.transpose(1, 2))

        return x.transpose(1, 2), layers

    def forward(self, wav: torch.Tensor, mask: torch.Tensor):
        z_final, layers = self.encode(wav, return_layers=True)
        z_agg = self.layer_attention(layers)

        B, C, T = z_final.shape
        mask_3d = mask.unsqueeze(1)
        z_masked = z_agg * mask_3d + self.mask_token.expand(B, -1, T) * (1 - mask_3d)
        z_pred = self.predictor(z_masked)

        return z_agg, z_pred, self.cluster_head(z_agg)


# =============================================================================
# Target Encoder (Transformer-based)
# =============================================================================

class TargetEncoder(nn.Module):
    """WavLM-style Transformer target encoder (EMA, frozen)."""

    def __init__(
        self,
        code_dim: int = 512,
        conv_dim: int = 256,
        num_heads: int = 8,
        ff_dim: int = 2048,
        num_layers: int = 10,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__()

        self.hop = 320
        self.code_dim = code_dim

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

        self.layer_attention = AttentionLayerAggregation(
            code_dim=code_dim,
            num_layers=num_layers + 1,
            num_heads=4,
        )

        for p in self.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, wav: torch.Tensor, return_layers: bool = False):
        x = self.feature_extractor(wav).transpose(1, 2)
        x = self.layer_norm(self.post_extract_proj(x))

        layers = [x.transpose(1, 2)]
        for layer in self.layers:
            x = layer(x)
            layers.append(x.transpose(1, 2))

        z_agg = self.layer_attention(layers)

        if return_layers:
            return z_agg, layers
        return z_agg


# =============================================================================
# Loading Utilities
# =============================================================================

def get_default_config() -> dict:
    """Default config for Transformer encoder."""
    return {
        "sample_rate": 16000,
        "code_dim": 512,
        "conv_dim": 256,
        "num_heads": 8,
        "ff_dim": 2048,
        "num_layers": 10,
        "dropout": 0.1,
        "K": 1024,
    }


def get_base_config() -> dict:
    """WavLM Base-like config."""
    return {
        "sample_rate": 16000,
        "code_dim": 768,
        "conv_dim": 512,
        "num_heads": 12,
        "ff_dim": 3072,
        "num_layers": 12,
        "dropout": 0.1,
        "K": 1024,
    }


def load_encoder(
    checkpoint_path: str,
    config: Optional[dict] = None,
    device: str = "cuda",
    strict: bool = False,
) -> OnlineEncoder:
    """Load encoder from checkpoint."""
    if config is None:
        config = get_default_config()

    model = OnlineEncoder(
        code_dim=config.get('code_dim', 512),
        conv_dim=config.get('conv_dim', 256),
        num_heads=config.get('num_heads', 8),
        ff_dim=config.get('ff_dim', 2048),
        num_layers=config.get('num_layers', 10),
        dropout=config.get('dropout', 0.1),
        K=config.get('K', 1024),
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

    cleaned = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}

    model.load_state_dict(cleaned, strict=strict)
    model.to(device)
    model.eval()

    return model


def load_target_encoder(
    checkpoint_path: str,
    config: Optional[dict] = None,
    device: str = "cuda",
) -> TargetEncoder:
    """Load target encoder from checkpoint."""
    if config is None:
        config = get_default_config()

    model = TargetEncoder(
        code_dim=config.get('code_dim', 512),
        conv_dim=config.get('conv_dim', 256),
        num_heads=config.get('num_heads', 8),
        ff_dim=config.get('ff_dim', 2048),
        num_layers=config.get('num_layers', 10),
        dropout=config.get('dropout', 0.1),
    )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if 'target' in checkpoint:
        model.load_state_dict(checkpoint['target'])
    else:
        raise KeyError("Checkpoint does not contain 'target' key")

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
        print(f"  {n:<25} {params / 1e6:>6.2f}M ({params / total * 100:>5.1f}%)")


# Aliases
TransformerEncoder = OnlineEncoder
WavLMEncoder = OnlineEncoder
load_transformer_encoder = load_encoder


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--config", type=str, default="small", choices=["small", "base"])
    args = parser.parse_args()

    config = get_default_config() if args.config == "small" else get_base_config()

    print(f"Loading model from {args.checkpoint}...")
    model = load_encoder(args.checkpoint, config=config, device=args.device)
    print_model_stats(model, "TransformerEncoder")

    print("\nTesting with random input...")
    with torch.no_grad():
        dummy = torch.randn(1, 1, 16000 * 2).to(args.device)
        z = model.encode(dummy)
        print(f"Input: {dummy.shape} -> Output: {z.shape}")
        print(f"Hop: {model.hop}, Expected frames: {dummy.shape[-1] // model.hop}")

    print("\nDone!")
