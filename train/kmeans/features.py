"""Streaming dataset for K-means fitting."""

import glob
import json
import os
import random

import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import IterableDataset

from .utils import get_rank, get_world_size


class StreamingDataset(IterableDataset):
    """Streaming dataset with distributed sharding."""
    
    def __init__(
        self,
        root: str,
        sample_rate: int = 16000,
        max_seconds: float = 15.0,
        base_path: str = "/scratch/gioannides/granary_data"
    ):
        self.root = root
        self.sr = sample_rate
        self.max_samples = int(sample_rate * max_seconds)
        self.base_path = base_path

    def __iter__(self):
        rank = get_rank()
        world_size = get_world_size()
        
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            total_shards = world_size * num_workers
            shard_id = rank * num_workers + worker_id
        else:
            total_shards = world_size
            shard_id = rank
        
        all_files = []
        for root_dir in self.root.split(','):
            root_dir = root_dir.strip()
            all_files.extend(sorted(glob.glob(os.path.join(root_dir, "*.jsonl"))))
        
        files = all_files[shard_id::total_shards]
        random.shuffle(files)
        
        for fp in files:
            try:
                with open(fp) as f:
                    for line in f:
                        try:
                            obj = json.loads(line.strip())
                            wav_path = obj["wav_path"]
                            
                            if not os.path.isabs(wav_path):
                                wav_path = os.path.join(self.base_path, wav_path)
                            
                            wav, sr = torchaudio.load(wav_path)
                            if wav.shape[0] > 1:
                                wav = wav.mean(0, keepdim=True)
                            if sr != self.sr:
                                wav = torchaudio.functional.resample(wav, sr, self.sr)
                            if wav.shape[-1] > self.max_samples:
                                start = random.randint(0, wav.shape[-1] - self.max_samples)
                                wav = wav[..., start:start + self.max_samples]
                            
                            yield wav.squeeze(0)
                        except:
                            continue
            except:
                continue


def make_collate(hop: int):
    """Create collate function for k-means."""
    def collate(batch):
        if not batch:
            return None, None
        
        lengths = [x.shape[0] for x in batch]
        T = max(lengths)
        T = ((max(T, 4 * hop) + hop - 1) // hop) * hop
        stacked = torch.stack([F.pad(x, (0, T - x.shape[0])) for x in batch]).unsqueeze(1)
        frame_lengths = [l // hop for l in lengths]
        
        return stacked, frame_lengths
    return collate
