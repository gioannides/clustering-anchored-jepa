"""Utilities for WavLM training."""

import random

import torch
import torch.distributed as dist


def rank0() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def unwrap(m):
    return m.module if hasattr(m, "module") else m


def create_mask(B: int, T: int, device: torch.device, frame_lengths: list = None,
                mask_prob: float = 0.65, mask_length: int = 10) -> torch.Tensor:
    """WavLM-style span masking."""
    masks = torch.ones(B, T, device=device)

    for b in range(B):
        real_T = frame_lengths[b] if frame_lengths else T
        num_mask = int(real_T * mask_prob)

        if num_mask == 0:
            continue

        num_spans = max(1, num_mask // mask_length)
        span_starts = torch.randint(0, max(1, real_T - mask_length), (num_spans,))

        for start in span_starts:
            length = random.randint(mask_length // 2, mask_length * 2)
            end = min(start + length, real_T)
            masks[b, start:end] = 0

        if frame_lengths and real_T < T:
            masks[b, real_T:] = 1

    return masks


def create_padding_mask(B: int, T: int, frame_lengths: list, device: torch.device) -> torch.Tensor:
    """Create padding mask (1=valid, 0=padding)."""
    mask = torch.zeros(B, T, device=device)
    for b in range(B):
        mask[b, :frame_lengths[b]] = 1
    return mask
