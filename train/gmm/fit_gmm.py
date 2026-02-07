#!/usr/bin/env python3
"""
Fit GMM on log-mel spectrograms using gradient-based optimization.

Supports single-GPU and multi-GPU (via torchrun).
"""

import argparse
import glob
import json
import math
import os
import random
import re

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchaudio
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm



from gmm import GradientGMM, compute_gmm_metrics
from dataset import StreamingAudioDataset, collate_with_lengths

from utils import (
    setup_distributed,
    cleanup_distributed,
    is_main_process,
    get_rank,
    get_world_size,
    all_gather_with_sizes,
)



# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------

class GMMTrainer:
    """GMM fitting trainer with multi-GPU support."""
    
    def __init__(self, args):
        self.args = args
        
        # Setup distributed
        self.rank, self.local_rank, self.world_size = setup_distributed()
        self.device = f'cuda:{self.local_rank}' if torch.cuda.is_available() else 'cpu'
        
        # Create output directory
        if is_main_process():
            os.makedirs(args.output_dir, exist_ok=True)
        if dist.is_initialized():
            dist.barrier()
        
        # Compute hop from strides
        strides = [int(x) for x in args.strides.split(',')]
        self.hop = math.prod(strides)
        
        # Initialize mel transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=args.sample_rate,
            n_fft=400,
            hop_length=self.hop,
            n_mels=args.n_mels,
        ).to(self.device)
        
        # Initialize or load GMM
        self.gmm, self.start_frames = self._init_gmm()
        
        # Wrap with DDP
        if dist.is_initialized():
            self.gmm = DDP(self.gmm, device_ids=[self.local_rank], output_device=self.local_rank)
        
        self.gmm_module = self.gmm.module if hasattr(self.gmm, 'module') else self.gmm
        
        # Optimizer and scheduler
        self.optimizer = torch.optim.Adam(self.gmm.parameters(), lr=args.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=args.n_sgd_steps, eta_min=args.lr * 0.01
        )
        
        # Initialize metrics file
        if is_main_process():
            self._init_metrics_file()
    
    def _init_gmm(self):
        """Initialize or resume GMM."""
        start_frames = 0
        
        if self.args.resume and os.path.exists(self.args.resume):
            gmm = GradientGMM.load(self.args.resume, device=self.device)
            match = re.search(r'(\d+)M\.pt$', self.args.resume)
            if match:
                start_frames = int(match.group(1)) * 1_000_000
            if is_main_process():
                print(f"[Resume] Loaded from {self.args.resume}, starting at {start_frames/1e6:.0f}M frames")
        else:
            gmm = GradientGMM(self.args.K, self.args.n_mels, device=self.device)
        
        return gmm, start_frames
    
    def _init_metrics_file(self):
        """Initialize metrics TSV file."""
        metrics_file = os.path.join(self.args.output_dir, "metrics.tsv")
        if not os.path.exists(metrics_file):
            with open(metrics_file, "w") as f:
                f.write("frames\tupdate\tavg_loss\talive\tbalance_cv\tweight_entropy\t"
                        "nll\tposterior_entropy\tmean_delta\tvar_delta\tavg_var\t"
                        "min_var\tmax_var\tinter_cluster_dist\tintra_cluster_dist\n")
    
    def fit(self):
        """Run GMM fitting."""
        if is_main_process():
            self._print_config()
        
        # Create dataloader        
        dataset = StreamingAudioDataset(
            self.args.data_dir,
            self.args.sample_rate,
            self.args.max_seconds,
            self.args.base_path,
            seen_file=self.args.seen_file,
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_with_lengths(self.hop),
            prefetch_factor=4,
        )
        
        # Training state
        buffer = []
        buffer_frames = 0
        total_frames = self.start_frames
        last_ckpt_frames = self.start_frames
        n_updates = 0
        
        pbar = tqdm(dataloader, desc="Fitting GMM", disable=not is_main_process())
        
        for wav, frame_lengths in pbar:
            if wav is None:
                continue
            
            # Extract features
            feat_flat, n_frames = self._extract_features(wav, frame_lengths)
            
            buffer.append(feat_flat)
            buffer_frames += feat_flat.shape[0]
            total_frames += n_frames
            
            # Update GMM when buffer is full
            if buffer_frames >= self.args.batch_update_size // self.world_size:
                batch_data = self._gather_buffer(buffer)
                
                # Initialize on first batch
                if n_updates == 0 and not self.args.resume:
                    self.gmm_module.init_from_data(batch_data, method='kmeans++')
                
                # Run SGD update
                avg_loss = self._sgd_update(batch_data, n_updates)
                n_updates += 1
                
                # Log metrics
                if is_main_process():
                    self._log_metrics(batch_data, total_frames, n_updates, avg_loss, pbar)
                
                # Reset buffer
                buffer = []
                buffer_frames = 0
                
                if dist.is_initialized():
                    dist.barrier()
            
            # Checkpoint
            if self.args.save_every > 0 and (total_frames - last_ckpt_frames) >= self.args.save_every:
                ckpt_path = os.path.join(self.args.output_dir, f"gmm_ckpt_{total_frames // 1_000_000}M.pt")
                self.gmm_module.save(ckpt_path)
                last_ckpt_frames = total_frames
                
                if dist.is_initialized():
                    dist.barrier()
            
            # Check termination
            if total_frames >= self.args.target_frames:
                break
        
        # Final update with remaining buffer
        if buffer:
            batch_data = self._gather_buffer(buffer)
            self._sgd_update(batch_data, n_updates, final=True)
        
        pbar.close()
        
        # Save final GMM
        if is_main_process():
            gmm_path = os.path.join(self.args.output_dir, "gmm.pt")
            self.gmm_module.save(gmm_path)
            self._print_summary(total_frames, n_updates, gmm_path)
        
        cleanup_distributed()
    
    def _extract_features(self, wav, frame_lengths):
        """Extract log-mel features from waveform."""
        wav = wav.to(self.device)
        
        with torch.no_grad():
            mel = self.mel_transform(wav.squeeze(1))
            log_mel = torch.log(mel + 1e-6)
            
            # Only keep real frames (no padding)
            real_frames = []
            for b in range(log_mel.shape[0]):
                real_T = frame_lengths[b]
                real_frames.append(log_mel[b, :, :real_T].permute(1, 0))
            
            feat_flat = torch.cat(real_frames, dim=0)
            
            # Count frames across all GPUs
            local_frames = torch.tensor(feat_flat.shape[0], device=self.device)
            if dist.is_initialized():
                dist.all_reduce(local_frames)
            
            return feat_flat, local_frames.item()
    
    def _gather_buffer(self, buffer):
        """Gather buffer from all ranks."""
        batch_data = torch.cat(buffer, dim=0)
        
        if dist.is_initialized():
            batch_data = all_gather_with_sizes(batch_data, self.device)
        
        # Subsample if too large
        if batch_data.shape[0] > self.args.max_samples:
            indices = torch.randperm(batch_data.shape[0], device=self.device)[:self.args.max_samples]
            batch_data = batch_data[indices]
        
        return batch_data
    
    def _sgd_update(self, batch_data, n_updates, final=False):
        """Run SGD steps on batch."""
        # Store previous params for delta tracking
        prev_means = self.gmm_module.means.clone()
        prev_vars = self.gmm_module.covariances.clone()
        
        self.gmm.train()
        losses = []
        
        n_steps = self.args.n_sgd_steps // 2 if final else self.args.n_sgd_steps
        
        sgd_pbar = tqdm(
            range(n_steps),
            desc=f"SGD update {n_updates + 1}",
            leave=False,
            disable=not is_main_process()
        )
        
        for step in sgd_pbar:
            indices = torch.randperm(batch_data.shape[0], device=self.device)[:self.args.sgd_batch_size]
            mini_batch = batch_data[indices]
            
            self.optimizer.zero_grad()
            loss = self.gmm(mini_batch, chunk_size=self.args.chunk_size)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.gmm.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
            
            losses.append(loss.item())
            
            if step % 100 == 0 and is_main_process():
                sgd_pbar.set_postfix(loss=f"{loss.item():.2f}", lr=f"{self.scheduler.get_last_lr()[0]:.2e}")
        
        # Reset scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.args.n_sgd_steps, eta_min=self.args.lr * 0.01
        )
        
        self._prev_means = prev_means
        self._prev_vars = prev_vars
        
        return sum(losses) / len(losses)
    
    def _log_metrics(self, batch_data, total_frames, n_updates, avg_loss, pbar):
        """Compute and log metrics."""
        metrics = compute_gmm_metrics(self.gmm, batch_data, self._prev_means, self._prev_vars)
        
        pbar.set_postfix(
            frames=f"{total_frames / 1e6:.1f}M",
            updates=n_updates,
            alive=f"{metrics['alive']}/{self.args.K}",
            nll=f"{metrics['nll']:.2f}",
        )
        
        # Write to file
        metrics_file = os.path.join(self.args.output_dir, "metrics.tsv")
        with open(metrics_file, "a") as f:
            f.write(f"{total_frames}\t{n_updates}\t{avg_loss:.6f}\t"
                    f"{metrics['alive']}\t{metrics['balance_cv']:.4f}\t"
                    f"{metrics['weight_entropy']:.4f}\t{metrics['nll']:.4f}\t"
                    f"{metrics['posterior_entropy']:.4f}\t{metrics['mean_delta']:.6f}\t"
                    f"{metrics['var_delta']:.6f}\t{metrics['avg_var']:.4f}\t"
                    f"{metrics['min_var']:.6f}\t{metrics['max_var']:.4f}\t"
                    f"{metrics['inter_cluster_dist']:.4f}\t{metrics['intra_cluster_dist']:.4f}\n")
        
        # Periodic detailed logging
        if n_updates % 5 == 0:
            print(f"\n[Metrics @ {total_frames / 1e6:.1f}M frames, update {n_updates}]")
            print(f"  alive={metrics['alive']}/{self.args.K}, balance_cv={metrics['balance_cv']:.3f}")
            print(f"  nll={metrics['nll']:.3f}, posterior_ent={metrics['posterior_entropy']:.3f}")
            print(f"  Δmean={metrics['mean_delta']:.5f}, Δvar={metrics['var_delta']:.5f}")
    
    def _print_config(self):
        """Print training configuration."""
        print("=" * 60)
        print("[Phase 1] Fit GMM on Log-Mel (Gradient-based)")
        print("=" * 60)
        print(f"  K={self.args.K}, n_mels={self.args.n_mels}")
        print(f"  target_frames={self.args.target_frames / 1e6:.0f}M")
        print(f"  batch_update_size={self.args.batch_update_size / 1e6:.0f}M")
        print(f"  world_size={self.world_size}, device={self.device}")
    
    def _print_summary(self, total_frames, n_updates, gmm_path):
        """Print final summary."""
        print(f"\n[GMM] Total: {total_frames / 1e6:.1f}M frames, {n_updates} updates")
        print("\n" + "=" * 60)
        print(f"[Done] GMM saved to: {gmm_path}")
        print(f"\nNext step - run JEPA training:")
        print(f"  python train/jepa/train.py --gmm_path {gmm_path}")
        print("=" * 60)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Fit GMM on log-mel spectrograms",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Required

    parser.add_argument('--data_dir', required=True, help='Comma-separated directories containing JSONL files')
    
    parser.add_argument('--output_dir', required=True, help='Output directory')
    
    # GMM config
    parser.add_argument('--K', type=int, default=1024, help='Number of GMM components')
    parser.add_argument('--n_mels', type=int, default=80, help='Number of mel bins')
    
    # Audio config
    parser.add_argument('--sample_rate', type=int, default=16000, help='Audio sample rate')
    parser.add_argument('--strides', default='8,8,5', help='Encoder strides (for hop calculation)')
    parser.add_argument('--max_seconds', type=float, default=15.0, help='Max audio length')
    parser.add_argument('--base_path', default=None, help='Base path for relative audio paths')
    
    # Training config
    parser.add_argument('--target_frames', type=int, default=100_000_000, help='Total frames to process')
    parser.add_argument('--batch_update_size', type=int, default=5_000_000, help='Frames per update')
    parser.add_argument('--max_samples', type=int, default=50_000_000, help='Max samples per update')
    parser.add_argument('--n_sgd_steps', type=int, default=2000, help='SGD steps per update')
    parser.add_argument('--sgd_batch_size', type=int, default=16384, help='SGD mini-batch size')
    parser.add_argument('--chunk_size', type=int, default=4096, help='Chunk size for log-prob')
    parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate')
    
    # Dataloader config
    parser.add_argument('--batch_size', type=int, default=64, help='Audio batch size')
    
    # Checkpointing
    parser.add_argument('--save_every', type=int, default=50_000_000, help='Save every N frames')

    
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--seen_file', type=str, default=None, help='Path to seen-hashes file for dedup')
    

    
    return parser.parse_args()


def main():
    args = parse_args()
    if args.seen_file is None:
        args.seen_file = os.path.join(args.output_dir, "seen_hashes.pt")
    trainer = GMMTrainer(args)
    trainer.fit()


if __name__ == '__main__':
    main()
