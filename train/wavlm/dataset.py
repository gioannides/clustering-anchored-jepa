"""Streaming dataset for WavLM training."""

import fcntl
import glob
import hashlib
import json
import os
import random

import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import IterableDataset


def stable_hash(s: str) -> str:
    return hashlib.md5(s.encode()).hexdigest()


class StreamingDataset(IterableDataset):
    """Streaming dataset with deduplication."""

    def __init__(
        self,
        root: str,
        sample_rate: int = 16000,
        max_seconds: float = 15.0,
        seen_file: str = None,
        base_path: str = "/scratch/gioannides/granary_data",
        preload_paths: bool = True
    ):
        self.root = root
        self.sr = sample_rate
        self.max_samples = int(sample_rate * max_seconds)
        self.seen_file = seen_file
        self.base_path = base_path

        self._all_paths = None
        self._seen = None

        if preload_paths:
            self._preload_paths()
            self._load_seen()

    def _preload_paths(self):
        paths = []
        for root_dir in self.root.split(','):
            for jf in sorted(glob.glob(os.path.join(root_dir.strip(), "*.jsonl"))):
                try:
                    with open(jf) as f:
                        for line in f:
                            obj = json.loads(line.strip())
                            wav_path = obj["wav_path"]
                            if not wav_path.startswith('/'):
                                wav_path = os.path.join(self.base_path, wav_path)
                            paths.append(wav_path)
                except:
                    continue
        self._all_paths = paths
        print(f"[StreamingDataset] Preloaded {len(paths)} paths")

    def _load_seen(self):
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
            print(f"[StreamingDataset] Filtered {before} -> {len(self._all_paths)} unseen")

    def _save_seen(self, buffer: list):
        if not self.seen_file:
            return
        try:
            with open(self.seen_file + ".lock", 'w') as lf:
                fcntl.flock(lf, fcntl.LOCK_EX | fcntl.LOCK_NB)
                try:
                    existing = set()
                    if os.path.exists(self.seen_file):
                        existing = set(torch.load(self.seen_file, weights_only=True))
                    torch.save(list(existing | set(buffer)), self.seen_file + ".tmp")
                    os.replace(self.seen_file + ".tmp", self.seen_file)
                finally:
                    fcntl.flock(lf, fcntl.LOCK_UN)
        except BlockingIOError:
            pass

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

        my_paths = self._all_paths[worker_id::total_workers]
        seen_buffer = []

        for wav_path in my_paths:
            try:
                wav, sr = torchaudio.load(wav_path)
                if wav.shape[0] > 1:
                    wav = wav.mean(0, keepdim=True)
                if sr != self.sr:
                    wav = torchaudio.functional.resample(wav, sr, self.sr)
                if wav.shape[-1] > self.max_samples:
                    start = random.randint(0, wav.shape[-1] - self.max_samples)
                    wav = wav[..., start:start + self.max_samples]

                seen_buffer.append(stable_hash(wav_path))
                if len(seen_buffer) >= 500:
                    self._save_seen(seen_buffer)
                    seen_buffer = []

                yield wav.squeeze(0), wav_path
            except:
                continue

        if seen_buffer:
            self._save_seen(seen_buffer)

        while True:
            yield None, None


def make_collate(stride: int = 320):
    """Create collate function."""
    def collate(batch):
        batch = [(w, p) for w, p in batch if w is not None]
        if not batch:
            return None, [], 0.0, None

        wavs, paths = zip(*batch)
        lengths = [x.shape[0] for x in wavs]
