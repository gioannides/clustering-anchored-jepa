"""Distributed training utilities."""

import os

import torch
import torch.distributed as dist


def setup_distributed():
    """Initialize distributed training if available."""
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        
        return rank, local_rank, world_size
    else:
        return 0, 0, 1


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    """Check if current process is the main process."""
    return not dist.is_initialized() or dist.get_rank() == 0


def get_rank() -> int:
    """Get current process rank."""
    return dist.get_rank() if dist.is_initialized() else 0


def get_world_size() -> int:
    """Get total number of processes."""
    return dist.get_world_size() if dist.is_initialized() else 1


def broadcast_tensor(tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
    """Broadcast tensor from src to all processes."""
    if not dist.is_initialized():
        return tensor
    dist.broadcast(tensor, src=src)
    return tensor


def all_gather_with_sizes(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Gather tensors of different sizes from all ranks.
    
    Handles padding/unpadding automatically.
    """
    if not dist.is_initialized():
        return tensor
    
    world_size = get_world_size()
    
    # Gather sizes
    local_size = torch.tensor(tensor.shape[0], device=device)
    all_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(all_sizes, local_size)
    
    # Pad to max size
    max_size = max(s.item() for s in all_sizes)
    if tensor.shape[0] < max_size:
        padding = torch.zeros(max_size - tensor.shape[0], tensor.shape[1], device=device)
        tensor = torch.cat([tensor, padding], dim=0)
    
    # Gather
    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor)
    
    # Remove padding
    real_data = []
    for i, g in enumerate(gathered):
        real_size = all_sizes[i].item()
        real_data.append(g[:real_size])
    
    return torch.cat(real_data, dim=0)
