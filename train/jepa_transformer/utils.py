"""Training utilities for JEPA."""

import hashlib
import random

import torch
import torch.distributed as dist


def rank0() -> bool:
    """Check if current process is rank 0."""
    return (not dist.is_initialized()) or (dist.get_rank() == 0)


def unwrap(model):
    """Unwrap model from DeepSpeed/DDP wrapper."""
    return model.module if hasattr(model, "module") else model


def stable_hash(s: str) -> str:
    """Compute stable hash for deduplication."""
    return hashlib.md5(s.encode()).hexdigest()


@torch.no_grad()
def ema_update(target, online, decay: float = 0.996):
    """Update target encoder with EMA of online encoder."""
    online_state = unwrap(online).state_dict()
    
    for name, p_t in target.named_parameters():
        if name in online_state:
            p_s = online_state[name]
            if p_s.shape == p_t.shape:
                p_t.data.mul_(decay).add_(p_s.to(p_t.dtype), alpha=1 - decay)


def create_mask(
    B: int,
    T: int,
    device: torch.device,
    frame_lengths: list = None,
    min_span: int = 10,
    max_span: int = 25,
    mask_ratio: tuple = (0.4, 0.65)
) -> torch.Tensor:
    """
    Create block-wise temporal mask.
    
    Args:
        B: Batch size
        T: Sequence length
        device: Torch device
        frame_lengths: Actual lengths (excludes padding)
        min_span: Minimum mask span
        max_span: Maximum mask span
        mask_ratio: (min_ratio, max_ratio) of frames to mask
    
    Returns:
        Mask tensor [B, T] where 1 = keep, 0 = mask
    """
    masks = torch.ones(B, T, device=device)
    
    for b in range(B):
        real_T = frame_lengths[b] if frame_lengths else T
        
        ratio = random.uniform(*mask_ratio)
        to_mask = int(real_T * ratio)
        masked = 0
        
        while masked < to_mask:
            span = random.randint(min_span, min(max_span, max(min_span, real_T // 3)))
            start = random.randint(0, max(1, real_T - span))
            end = min(start + span, real_T)
            masks[b, start:end] = 0
            masked += (end - start)
        
        # Don't mask padding
        if frame_lengths and real_T < T:
            masks[b, real_T:] = 1
    
    return masks


def create_padding_mask(
    B: int,
    T: int,
    frame_lengths: list,
    device: torch.device
) -> torch.Tensor:
    """
    Create padding mask.
    
    Returns:
        Mask tensor [B, T] where 1 = real audio, 0 = padding
    """
    pad_mask = torch.zeros(B, T, device=device)
    for b in range(B):
        pad_mask[b, :frame_lengths[b]] = 1
    return pad_mask
