"""Streaming audio dataset for JEPA training."""

import glob
import json
import os
import random

import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import IterableDataset

from .utils import stable_hash


class StreamingDataset(IterableDataset):
    """
    Streaming audio dataset with deduplication.
    
    Args:
        data_dir: Comma-separated directories containing JSONL files
        sample_rate: Target sample rate
        max_seconds: Maximum audio length
        seen_file: Path to save/load seen audio hashes
        base_path: Base path for relative audio paths
        preload_paths: Preload all paths into memory
    """
    
    def __init__(
        self,
        data_dir: str,
        sample_rate: int = 16000,
        max_seconds: float = 15.0,
        seen_file: str = None,
        base_path: str = None,
        preload_paths: bool = True
    ):
        self.data_dir = data_dir
        self.sample_rate = sample_rate
        self.max_samples = int(sample_rate * max_seconds)
        self.seen_file = seen_file
        self.base_path = base_path
        
        self._all_paths = None
        self._seen = None
        
        if preload_paths:
            self._preload_all_paths()
            self._load_seen()
    
    def _preload_all_paths(self):
        """Load all wav paths into memory."""
        all_paths = []
        
        for root_dir in self.data_dir.split(','):
            root_dir = root_dir.strip()
            jsonl_files = glob.glob(os.path.join(root_dir, "*.jsonl"))
            
            for jf in sorted(jsonl_files):
                try:
                    with open(jf, 'r') as f:
                        for line in f:
                            try:
                                obj = json.loads(line.strip())
                                wav_path = obj.get("wav_path") or obj.get("audio_path") or obj.get("path")
                                
                                if not wav_path:
                                    continue
                                
                                if not wav_path.startswith('/') and self.base_path:
                                    wav_path = os.path.join(self.base_path, wav_path)
                                
                                all_paths.append(wav_path)
                            except:
                                continue
                except:
                    continue
        
        self._all_paths = all_paths
        print(f"[StreamingDataset] Preloaded {len(all_paths)} paths")
    
    def _load_seen(self):
        """Load seen hashes and filter paths."""
        self._seen = set()
        
        if self.seen_file and os.path.exists(self.seen_file):
            try:
                self._seen = set(torch.load(self.seen_file, weights_only=True))
                print(f"[StreamingDataset] Loaded {len(self._seen)} seen hashes")
            except:
                pass
        
        if self._all_paths:
            before = len(self._all_paths)
            self._all_paths = [p for p in self._all_paths if stable_hash(p) not in self._seen]
            random.shuffle(self._all_paths)
            print(f"[StreamingDataset] Filtered {before} -> {len(self._all_paths)} unseen paths")
    
    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        rank = int(os.environ.get("RANK", 0))
        world = int(os.environ.get("WORLD_SIZE", 1))
        
        if worker:
            worker_id = rank * worker.num_workers + worker.id
            total_workers = world * worker.num_workers
        else:
            worker_id = rank
            total_workers = world
        
        my_paths = list(reversed(self._all_paths[worker_id::total_workers]))
        seen_buffer = []
        
        for wav_path in my_paths:
            try:
                wav, sr = torchaudio.load(wav_path)
                
                # Mono
                if wav.shape[0] > 1:
                    wav = wav.mean(0, keepdim=True)
                
                # Resample
                if sr != self.sample_rate:
                    wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
                
                # Trim
                if wav.shape[-1] > self.max_samples:
                    start = random.randint(0, wav.shape[-1] - self.max_samples)
                    wav = wav[..., start:start + self.max_samples]
                
                seen_buffer.append(stable_hash(wav_path))
                
                if self.seen_file and len(seen_buffer) >= 500:
                    self._save_seen_buffer(seen_buffer)
                    seen_buffer = []
                
                yield wav.squeeze(0), wav_path
            
            except Exception:
                continue
        
        if self.seen_file and seen_buffer:
            self._save_seen_buffer(seen_buffer)
        
        # Signal exhaustion
        while True:
            yield None, None
    
    def _save_seen_buffer(self, buffer: list):
        """Append buffer to seen file with locking."""
        import fcntl
        lock_path = self.seen_file + ".lock"
        
        try:
            with open(lock_path, 'w') as lf:
                fcntl.flock(lf, fcntl.LOCK_EX | fcntl.LOCK_NB)
                try:
                    existing = set()
                    if os.path.exists(self.seen_file):
                        existing = set(torch.load(self.seen_file, weights_only=True))
                    merged = existing | set(buffer)
                    torch.save(list(merged), self.seen_file + ".tmp")
                    os.replace(self.seen_file + ".tmp", self.seen_file)
                finally:
                    fcntl.flock(lf, fcntl.LOCK_UN)
        except BlockingIOError:
            pass


def make_collate(hop: int):
    """Create collate function for DataLoader."""
    def collate(batch):
        batch = [(w, p) for w, p in batch if w is not None]
        
        if not batch:
            return None, [], 0.0, None
        
        wavs, paths = zip(*batch)
        total_secs = sum(x.shape[0] for x in wavs)
        lengths = [x.shape[0] for x in wavs]
        
        T = max(lengths)
        T = ((max(T, 4 * hop) + hop - 1) // hop) * hop
        
        stacked = torch.stack([F.pad(x, (0, T - x.shape[0])) for x in wavs]).unsqueeze(1)
        frame_lengths = [l // hop for l in lengths]
        
        return stacked, list(paths), total_secs, frame_lengths
    
    return collate
