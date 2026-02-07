"""Streaming audio dataset with deduplication and path-level sharding."""

import fcntl
import glob
import hashlib
import json
import os
import random
from typing import Optional, Set

import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import IterableDataset

from .utils import get_rank, get_world_size, is_main_process


def stable_hash(s: str) -> str:
    """Deterministic hash for deduplication."""
    return hashlib.md5(s.encode()).hexdigest()


class StreamingAudioDataset(IterableDataset):
    """
    Stream audio from JSONL files with:
      - Multi-directory support (comma-separated data_dir)
      - Path-level sharding across workers/GPUs
      - Seen-file deduplication across restarts
    """

    def __init__(
        self,
        data_dir: str,
        sample_rate: int = 16000,
        max_seconds: float = 15.0,
        base_path: Optional[str] = None,
        seen_file: Optional[str] = None,
    ):
        self.data_dir = data_dir
        self.sample_rate = sample_rate
        self.max_samples = int(sample_rate * max_seconds)
        self.base_path = base_path
        self.seen_file = seen_file

        # Preload and filter paths once on construction
        self._all_paths = self._collect_all_paths()
        self._seen = self._load_seen()
        self._all_paths = [p for p in self._all_paths if stable_hash(p) not in self._seen]
        random.shuffle(self._all_paths)

        if is_main_process():
            print(f"[Dataset] {len(self._all_paths)} unseen paths "
                  f"({len(self._seen)} previously seen)")

    # ------------------------------------------------------------------
    # Path collection
    # ------------------------------------------------------------------

    def _collect_all_paths(self) -> list[str]:
        """Collect all wav paths from comma-separated directories."""
        all_paths = []
        for root_dir in self.data_dir.split(","):
            root_dir = root_dir.strip()
            if not root_dir:
                continue
            for jf in sorted(glob.glob(os.path.join(root_dir, "*.jsonl"))):
                all_paths.extend(self._paths_from_jsonl(jf))

        if is_main_process():
            print(f"[Dataset] Collected {len(all_paths)} total paths")
        return all_paths

    def _paths_from_jsonl(self, filepath: str) -> list[str]:
        """Extract wav paths from a single JSONL file."""
        paths = []
        try:
            with open(filepath) as f:
                for line in f:
                    try:
                        obj = json.loads(line.strip())
                        wav_path = (
                            obj.get("wav_path")
                            or obj.get("audio_path")
                            or obj.get("path")
                        )
                        if not wav_path:
                            continue
                        if not os.path.isabs(wav_path) and self.base_path:
                            wav_path = os.path.join(self.base_path, wav_path)
                        paths.append(wav_path)
                    except Exception:
                        continue
        except Exception:
            pass
        return paths

    # ------------------------------------------------------------------
    # Seen-file handling
    # ------------------------------------------------------------------

    def _load_seen(self) -> Set[str]:
        """Load previously-seen hashes from disk."""
        if not self.seen_file or not os.path.exists(self.seen_file):
            return set()
        try:
            seen = set(torch.load(self.seen_file, weights_only=True))
            if is_main_process():
                print(f"[Dataset] Loaded {len(seen)} seen hashes")
            return seen
        except Exception:
            return set()

    def _flush_seen_buffer(self, buffer: list[str]):
        """Append hashes to seen file with file-lock safety."""
        if not self.seen_file or not buffer:
            return

        lock_path = self.seen_file + ".lock"
        try:
            with open(lock_path, "w") as lf:
                fcntl.flock(lf, fcntl.LOCK_EX | fcntl.LOCK_NB)
                try:
                    existing: set = set()
                    if os.path.exists(self.seen_file):
                        existing = set(torch.load(self.seen_file, weights_only=True))
                    merged = existing | set(buffer)
                    tmp = self.seen_file + ".tmp"
                    torch.save(list(merged), tmp)
                    os.replace(tmp, self.seen_file)
                finally:
                    fcntl.flock(lf, fcntl.LOCK_UN)
        except BlockingIOError:
            pass  # Another worker holds the lock; skip this flush

    # ------------------------------------------------------------------
    # Iteration (path-level sharding)
    # ------------------------------------------------------------------

    def __iter__(self):
        rank = get_rank()
        world_size = get_world_size()

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            total_shards = world_size * worker_info.num_workers
            shard_id = rank * worker_info.num_workers + worker_info.id
        else:
            total_shards = world_size
            shard_id = rank

        # Shard at the *path* level for even distribution
        my_paths = self._all_paths[shard_id::total_shards]

        seen_buffer: list[str] = []

        for wav_path in my_paths:
            wav = self._load_audio(wav_path)
            if wav is None:
                continue

            seen_buffer.append(stable_hash(wav_path))
            if len(seen_buffer) >= 500:
                self._flush_seen_buffer(seen_buffer)
                seen_buffer = []

            yield wav

        # Flush remainder
        self._flush_seen_buffer(seen_buffer)

    # ------------------------------------------------------------------
    # Audio I/O
    # ------------------------------------------------------------------

    def _load_audio(self, path: str) -> Optional[torch.Tensor]:
        """Load, mono-mix, resample, and crop a single audio file."""
        try:
            wav, sr = torchaudio.load(path)

            if wav.shape[0] > 1:
                wav = wav.mean(0, keepdim=True)
            if sr != self.sample_rate:
                wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
            if wav.shape[-1] > self.max_samples:
                start = random.randint(0, wav.shape[-1] - self.max_samples)
                wav = wav[..., start : start + self.max_samples]

            return wav.squeeze(0)
        except Exception:
            return None


# --------------------------------------------------------------------------
# Collate
# --------------------------------------------------------------------------

def collate_with_lengths(hop: int):
    """Collate that tracks original lengths (avoids padding contamination)."""

    def collate(batch):
        batch = [x for x in batch if x is not None]
        if not batch:
            return None, None

        lengths = [x.shape[0] for x in batch]
        T = max(lengths)
        T = ((max(T, 4 * hop) + hop - 1) // hop) * hop

        stacked = torch.stack(
            [F.pad(x, (0, T - x.shape[0])) for x in batch]
        ).unsqueeze(1)
        frame_lengths = [l // hop for l in lengths]

        return stacked, frame_lengths

    return collate
