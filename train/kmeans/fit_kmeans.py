#!/usr/bin/env python3
"""
Fit K-Means on log-mel features (Multi-GPU).

Usage:
    # Single GPU
    python fit_kmeans.py --data_dir /path/to/jsonl --output_dir ./KMeans --K 512
    
    # Multi-GPU
    torchrun --nproc_per_node=8 fit_kmeans.py --data_dir /path/to/jsonl --output_dir ./KMeans --K 512
"""

import argparse
import os
import re

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from tqdm import tqdm

from kmeans import MiniBatchKMeans, compute_kmeans_metrics
from features import FeatureExtractor
from dataset import StreamingDataset, make_collate
from utils import setup_distributed, cleanup_distributed, is_main_process, get_world_size


def main(args):
    rank, local_rank, world_size = setup_distributed()
    device = f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu'
    
    if is_main_process():
        os.makedirs(args.output_dir, exist_ok=True)
    
    if dist.is_initialized():
        dist.barrier()
    
    feature_extractor = FeatureExtractor(
        sample_rate=args.sample_rate,
        hop_length=args.hop_length,
        n_mels=args.n_mels,
        device=device,
    )
    
    if is_main_process():
        print("=" * 60)
        print("[Phase 1] Fit K-Means on Log-Mel")
        print("=" * 60)
        print(f"  K={args.K}, n_mels={args.n_mels}")
        print(f"  target_frames={args.target_frames / 1e6:.0f}M")
        print(f"  batch_update_size={args.batch_update_size / 1e6:.0f}M")
        print(f"  world_size={world_size}, device={device}")
        
        metrics_file = os.path.join(args.output_dir, "kmeans_metrics.txt")
        if not os.path.exists(metrics_file):
            with open(metrics_file, "w") as f:
                f.write("frames\tupdate\talive\tbalance_cv\tinertia\tinter_dist\t"
                        "centroid_delta\tsilhouette\tmin_size\tmax_size\n")
    
    start_frames = 0
    n_updates = 0
    
    if args.resume_ckpt and os.path.exists(args.resume_ckpt):
        kmeans = MiniBatchKMeans.load(args.resume_ckpt, device=device)
        match = re.search(r'(\d+)M\.pt$', args.resume_ckpt)
        if match:
            start_frames = int(match.group(1)) * 1_000_000
        if is_main_process():
            print(f"[Resume] Loaded from {args.resume_ckpt}, starting at {start_frames / 1e6:.0f}M frames")
    else:
        kmeans = MiniBatchKMeans(args.K, feature_extractor.dim, device=device)
    
    dataloader = DataLoader(
        StreamingDataset(args.data_dir, args.sample_rate, args.max_seconds),
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True,
        collate_fn=make_collate(args.hop_length),
        prefetch_factor=4,
    )
    
    buffer = []
    buffer_frames = 0
    total_frames = start_frames
    last_ckpt_frames = start_frames
    
    pbar = tqdm(dataloader, desc="Fitting K-Means", disable=not is_main_process())
    
    for wav, frame_lengths in pbar:
        if wav is None:
            continue
        
        wav = wav.to(device)
        
        with torch.no_grad():
            feat_flat = feature_extractor(wav, frame_lengths)
            
            buffer.append(feat_flat)
            buffer_frames += feat_flat.shape[0]
            
            local_frames = torch.tensor(feat_flat.shape[0], device=device)
            if dist.is_initialized():
                dist.all_reduce(local_frames)
            total_frames += local_frames.item()
        
        if buffer_frames >= args.batch_update_size // world_size:
            batch_data = torch.cat(buffer, dim=0)
            
            if dist.is_initialized():
                local_size = torch.tensor(batch_data.shape[0], device=device)
                all_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
                dist.all_gather(all_sizes, local_size)
                
                max_size = max(s.item() for s in all_sizes)
                
                if batch_data.shape[0] < max_size:
                    padding = torch.zeros(max_size - batch_data.shape[0], feature_extractor.dim, device=device)
                    batch_data = torch.cat([batch_data, padding], dim=0)
                
                gathered = [torch.zeros_like(batch_data) for _ in range(world_size)]
                dist.all_gather(gathered, batch_data)
                
                real_data = []
                for i, g in enumerate(gathered):
                    real_size = all_sizes[i].item()
                    real_data.append(g[:real_size])
                
                batch_data = torch.cat(real_data, dim=0)
            
            if batch_data.shape[0] > args.max_samples:
                indices = torch.randperm(batch_data.shape[0], device=device)[:args.max_samples]
                batch_data = batch_data[indices]
            
            if n_updates == 0 and not (args.resume_ckpt and os.path.exists(args.resume_ckpt)):
                kmeans.init_centroids(batch_data, method=args.init_method)
            
            prev_centroids = kmeans.centroids.clone()
            
            if args.update_method == 'minibatch':
                n_minibatches = max(1, batch_data.shape[0] // args.minibatch_size)
                indices = torch.randperm(batch_data.shape[0], device=device)
                
                for i in range(n_minibatches):
                    start = i * args.minibatch_size
                    end = min(start + args.minibatch_size, batch_data.shape[0])
                    minibatch = batch_data[indices[start:end]]
                    kmeans.partial_fit(minibatch)
            else:
                kmeans.fit_batch(batch_data, n_iterations=args.lloyd_iterations)
            
            n_updates += 1
            buffer = []
            buffer_frames = 0
            
            if is_main_process():
                metrics = compute_kmeans_metrics(kmeans, batch_data, prev_centroids)
                
                pbar.set_postfix(
                    frames=f"{total_frames / 1e6:.1f}M",
                    updates=n_updates,
                    alive=f"{metrics['alive']}/{args.K}",
                    inertia=f"{metrics['inertia']:.2f}",
                    delta=f"{metrics['centroid_delta']:.4f}",
                )
                
                with open(os.path.join(args.output_dir, "kmeans_metrics.txt"), "a") as f:
                    f.write(f"{total_frames}\t{n_updates}\t{metrics['alive']}\t"
                            f"{metrics['balance_cv']:.4f}\t{metrics['inertia']:.4f}\t"
                            f"{metrics['inter_cluster_dist']:.4f}\t{metrics['centroid_delta']:.6f}\t"
                            f"{metrics['silhouette']:.4f}\t{metrics['min_cluster_size']}\t"
                            f"{metrics['max_cluster_size']}\n")
                
                if n_updates % 5 == 0:
                    print(f"\n[Metrics @ {total_frames / 1e6:.1f}M frames, update {n_updates}]")
                    print(f"  alive={metrics['alive']}/{args.K}, balance_cv={metrics['balance_cv']:.3f}")
                    print(f"  inertia={metrics['inertia']:.3f}, silhouette={metrics['silhouette']:.3f}")
            
            if dist.is_initialized():
                dist.barrier()
        
        if args.save_every_frames > 0 and (total_frames - last_ckpt_frames) >= args.save_every_frames:
            ckpt_path = os.path.join(args.output_dir, f"kmeans_ckpt_{total_frames // 1_000_000}M.pt")
            kmeans.save(ckpt_path)
            last_ckpt_frames = total_frames
            
            if dist.is_initialized():
                dist.barrier()
        
        if total_frames >= args.target_frames:
            break
    
    # Final update with remaining buffer
    if buffer:
        batch_data = torch.cat(buffer, dim=0)
        
        if dist.is_initialized():
            local_size = torch.tensor(batch_data.shape[0], device=device)
            all_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
            dist.all_gather(all_sizes, local_size)
            
            max_size = max(s.item() for s in all_sizes)
            
            if batch_data.shape[0] < max_size:
                padding = torch.zeros(max_size - batch_data.shape[0], feature_extractor.dim, device=device)
                batch_data = torch.cat([batch_data, padding], dim=0)
            
            gathered = [torch.zeros_like(batch_data) for _ in range(world_size)]
            dist.all_gather(gathered, batch_data)
            
            real_data = []
            for i, g in enumerate(gathered):
                real_size = all_sizes[i].item()
                real_data.append(g[:real_size])
            
            batch_data = torch.cat(real_data, dim=0)
        
        if batch_data.shape[0] > args.max_samples:
            indices = torch.randperm(batch_data.shape[0], device=device)[:args.max_samples]
            batch_data = batch_data[indices]
        
        kmeans.fit_batch(batch_data, n_iterations=args.lloyd_iterations)
    
    pbar.close()
    
    if is_main_process():
        print(f"\n[KMeans] Total: {total_frames / 1e6:.1f}M frames, {n_updates} updates")
        
        kmeans_path = os.path.join(args.output_dir, "kmeans.pt")
        kmeans.save(kmeans_path)
        
        print("\n" + "=" * 60)
        print(f"[Done] K-Means saved to: {kmeans_path}")
        print("=" * 60)
    
    cleanup_distributed()


def parse_args():
    parser = argparse.ArgumentParser(description="Fit K-Means on log-mel features")
    
    parser.add_argument('--data_dir', required=True, help='Comma-separated JSONL directories')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    
    # Audio
    parser.add_argument('--sample_rate', type=int, default=16000)
    parser.add_argument('--hop_length', type=int, default=320)
    parser.add_argument('--max_seconds', type=float, default=15.0)
    parser.add_argument('--n_mels', type=int, default=80)
    
    # K-Means
    parser.add_argument('--K', type=int, default=512, help='Number of clusters')
    parser.add_argument('--init_method', type=str, default='kmeans++', choices=['kmeans++', 'random'])
    parser.add_argument('--update_method', type=str, default='lloyd', choices=['minibatch', 'lloyd'])
    parser.add_argument('--lloyd_iterations', type=int, default=20)
    parser.add_argument('--minibatch_size', type=int, default=10000)
    
    # Data
    parser.add_argument('--target_frames', type=int, default=100_000_000)
    parser.add_argument('--batch_update_size', type=int, default=5_000_000)
    parser.add_argument('--max_samples', type=int, default=10_000_000)
    parser.add_argument('--batch_size', type=int, default=64)
    
    # Checkpointing
    parser.add_argument('--save_every_frames', type=int, default=50_000_000)
    parser.add_argument('--resume_ckpt', type=str, default=None)
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
