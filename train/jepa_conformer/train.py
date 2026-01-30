#!/usr/bin/env python3
"""
JEPA training with frozen GMM supervision.

Usage:
    deepspeed train.py \
        --data_dir /path/to/jsonl \
        --output_dir ./checkpoints/jepa \
        --gmm_path ./checkpoints/gmm/gmm.pt \
        --ds_config ../configs/ds_config.json
"""

import argparse
import json
import math
import os
import re
from collections import deque

import deepspeed
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import OnlineEncoder, TargetEncoder, FrozenGMM
from augment import DenoiseAugmentor
from dataset import StreamingDataset, make_collate
from utils import rank0, unwrap, ema_update, create_mask, create_padding_mask


def train(args):
    """Main training loop."""
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Parse architecture config
    channels = [int(x) for x in args.channels.split(',')]
    strides = [int(x) for x in args.strides.split(',')]
    hop = math.prod(strides)
    
    # Create dataloader
    seen_file = os.path.join(args.output_dir, "seen_hashes.pt")
    dataset = StreamingDataset(
        args.data_dir,
        args.sample_rate,
        args.max_seconds,
        seen_file=seen_file,
        preload_paths=True
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True,
        collate_fn=make_collate(hop),
        prefetch_factor=8,
        persistent_workers=False,
        timeout=60,
    )
    
    # Initialize model
    online = OnlineEncoder(
        args.code_dim, channels, strides,
        args.n_res, args.n_conformer, args.heads, args.K,
        args.use_gaatn, args.use_rel_pos
    )
    
    if rank0():
        enc_params = sum(
            p.numel() for n, p in online.named_parameters()
            if not any(x in n for x in ['cluster_head', 'predictor', 'mask_token'])
        )
        total_params = sum(p.numel() for p in online.parameters())
        print(f"[Model] Encoder: {enc_params / 1e6:.1f}M, Total: {total_params / 1e6:.1f}M params")
    
    optimizer = torch.optim.AdamW(online.parameters(), lr=args.lr, weight_decay=1e-3, betas=(0.8, 0.99))
    
    # Load DeepSpeed config
    with open(args.ds_config) as f:
        ds_cfg = json.load(f)
    
    if torch.cuda.is_available() and args.local_rank >= 0:
        torch.cuda.set_device(args.local_rank)
    
    # Load checkpoint if specified
    global_step = 0
    target_state_dict = None
    
    if args.pt_ckpt and os.path.exists(args.pt_ckpt):
        if rank0():
            print(f"[Resume] Loading from .pt checkpoint: {args.pt_ckpt}")
        
        ckpt = torch.load(args.pt_ckpt, map_location='cpu')
        
        if 'online' in ckpt:
            state_dict = ckpt['online']
            target_state_dict = ckpt.get('target', None)
            global_step = ckpt.get('step', 0)
        else:
            state_dict = ckpt
            match = re.search(r'(\d+)k?\.pt', args.pt_ckpt)
            if match:
                num = int(match.group(1))
                global_step = num * 1000 if 'k' in args.pt_ckpt.lower() else num
        
        if args.reinit_cluster_head:
            state_dict = {k: v for k, v in state_dict.items() if 'cluster_head' not in k}
            if rank0():
                print("[Resume] Reinitializing cluster_head")
        
        online.load_state_dict(state_dict, strict=False)
        
        if rank0():
            print(f"[Resume] Loaded step {global_step}")
    
    # Initialize DeepSpeed
    engine, _, _, _ = deepspeed.initialize(
        args=args,
        model=online,
        optimizer=optimizer,
        model_parameters=online.parameters(),
        config=ds_cfg
    )
    device = engine.device
    dtype = next(engine.module.parameters()).dtype
    
    # Initialize target encoder
    target = TargetEncoder(
        args.code_dim, channels, strides,
        args.n_res, args.n_conformer, args.heads,
        args.use_gaatn, args.use_rel_pos
    ).to(device, dtype)
    
    if target_state_dict is not None:
        target.load_state_dict(target_state_dict)
        if rank0():
            print("[Resume] Loaded target encoder from checkpoint")
    else:
        ema_update(target, engine.module, decay=0.0)
        if rank0():
            print("[Resume] Initialized target from online encoder")
    
    # Mel transform
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=args.sample_rate,
        n_fft=400,
        hop_length=hop,
        n_mels=args.n_mels
    ).to(device)
    
    # Augmentor
    augmentor = DenoiseAugmentor(
        p_noise=args.p_noise,
        p_mix=args.p_mix,
        snr_range_noise=(args.snr_low_noise, args.snr_high_noise),
        snr_range_speech=(args.snr_low_speech, args.snr_high_speech)
    )
    
    if rank0():
        print(f"[Augment] p_noise={args.p_noise}, p_mix={args.p_mix}")
    
    # Load GMM(s)
    gmms = []
    gmm_weights = []
    
    if args.gmm_paths:
        for item in args.gmm_paths.split(','):
            if ':' in item:
                path, weight = item.rsplit(':', 1)
                gmm_weights.append(float(weight))
            else:
                path = item
                gmm_weights.append(1.0)
            gmms.append(FrozenGMM.load(path.strip(), device=device, temperature=args.cluster_temperature))
        
        total_w = sum(gmm_weights)
        gmm_weights = [w / total_w for w in gmm_weights]
        
        if rank0():
            print(f"[GMM] Loaded {len(gmms)} GMMs, weights={gmm_weights}")
    
    elif args.gmm_path and os.path.exists(args.gmm_path):
        gmms = [FrozenGMM.load(args.gmm_path, device=device, temperature=args.cluster_temperature)]
        gmm_weights = [1.0]
        if rank0():
            print(f"[GMM] Loaded from {args.gmm_path}, K={gmms[0].K}")
    else:
        raise ValueError("GMM path required. Use --gmm_path or --gmm_paths")
    
    if rank0():
        print(f"[Schedule] cluster_weight: {args.cluster_weight_start} -> {args.cluster_weight_end}")
    
    # Checkpoint directory
    ckpt_dir = os.path.join(args.output_dir, "ckpts")
    os.makedirs(ckpt_dir, exist_ok=True)
    
    total_seconds = 0.0
    
    # Resume from DeepSpeed checkpoint
    if args.resume:
        try:
            _, client_sd = engine.load_checkpoint(ckpt_dir, tag=None)
            if client_sd and 'step' in client_sd:
                global_step = client_sd['step']
                if rank0():
                    print(f"[Resume] Step {global_step}")
                
                portable_path = os.path.join(ckpt_dir, f"portable_step{global_step}.pt")
                if os.path.exists(portable_path):
                    portable = torch.load(portable_path, map_location=device)
                    target.load_state_dict(portable['target'])
                    total_seconds = portable.get('total_seconds', 0.0)
                    if rank0():
                        print(f"[Resume] Loaded target, total_seconds={total_seconds / 3600:.1f}h")
        except Exception as e:
            if rank0():
                print(f"[Resume] Failed: {e}")
    
    # Training loop
    metrics = {k: deque(maxlen=10) for k in ['jepa', 'cluster', 'cw']}
    pbar = tqdm(total=args.max_steps, disable=not rank0(), initial=global_step, desc="Training")
    
    dl_iter = iter(dataloader)
    
    while global_step < args.max_steps:
        try:
            wav, paths, batch_secs, frame_lengths = next(dl_iter)
        except StopIteration:
            dl_iter = iter(dataloader)
            wav, paths, batch_secs, frame_lengths = next(dl_iter)
        
        if wav is None:
            continue
        
        # Track total audio
        batch_secs_tensor = torch.tensor(batch_secs, device=device, dtype=torch.float32)
        if dist.is_initialized():
            dist.all_reduce(batch_secs_tensor)
        total_seconds += batch_secs_tensor.item() / args.sample_rate
        
        wav = wav.to(device, dtype)
        B = wav.shape[0]
        
        if torch.isnan(wav).any() or torch.isinf(wav).any():
            if rank0():
                print("[WARNING] NaN/Inf in input, skipping")
            continue
        
        # Augmentation
        wav_aug, wav_clean = augmentor(wav)
        
        # Target forward
        with torch.no_grad():
            z_target_agg, target_layers = target(wav_clean, return_layers=True)
            T_z = z_target_agg.shape[-1]
            
            # Compute log-mel for GMM
            mel = mel_transform(wav_clean.squeeze(1).float())
            log_mel = torch.log(mel + 1e-6)
            if log_mel.shape[-1] > T_z:
                log_mel = log_mel[..., :T_z]
            elif log_mel.shape[-1] < T_z:
                log_mel = F.pad(log_mel, (0, T_z - log_mel.shape[-1]))
            
            # Weighted GMM posteriors
            soft_labels = None
            for gmm, w in zip(gmms, gmm_weights):
                if gmm.dim == args.code_dim:
                    sl = gmm.soft_assign(z_target_agg)
                else:
                    sl = gmm.soft_assign(log_mel)
                
                if soft_labels is None:
                    soft_labels = w * sl
                else:
                    soft_labels = soft_labels + w * sl
        
        # Create masks
        mask = create_mask(B, T_z, device, frame_lengths).to(dtype)
        pad_mask = create_padding_mask(B, T_z, frame_lengths, device).to(dtype)
        
        # Online forward
        z_online_agg, z_pred, cluster_logits = engine.module(wav_aug, mask)
        
        # JEPA loss
        m = (1 - mask).unsqueeze(1) * pad_mask.unsqueeze(1)
        jepa_loss = ((z_pred - z_target_agg.detach()).pow(2) * m).sum() / (m.sum() * z_pred.shape[1]).clamp_min(1)
        
        # Cluster loss
        mask_flat = ((1 - mask) * pad_mask).view(-1).bool()
        
        if mask_flat.any():
            pred_log_probs = F.log_softmax(cluster_logits.view(-1, args.K)[mask_flat], dim=-1)
            soft_tgt = soft_labels[mask_flat]
            cluster_loss = F.kl_div(pred_log_probs, soft_tgt, reduction='batchmean')
        else:
            cluster_loss = torch.tensor(0.0, device=device)
        
        # Total loss
        progress = global_step / args.max_steps
        cluster_weight = args.cluster_weight_start + (args.cluster_weight_end - args.cluster_weight_start) * progress
        loss = jepa_loss + cluster_weight * cluster_loss
        
        if torch.isnan(loss):
            if rank0():
                print(f"[DEBUG] NaN loss! jepa={jepa_loss.item()}, cluster={cluster_loss.item()}")
            continue
        
        # Backward
        engine.backward(loss)
        engine.step()
        ema_update(target, engine.module, decay=args.ema_decay)
        
        # Metrics
        metrics['jepa'].append(jepa_loss.item())
        metrics['cluster'].append(cluster_loss.item())
        metrics['cw'].append(cluster_weight)
        
        global_step += 1
        
        if rank0():
            pbar.update(1)
        
        # Cleanup
        del z_online_agg, z_pred, cluster_logits, z_target_agg, soft_labels, target_layers
        del wav_aug, wav_clean, mel, log_mel
        del loss, jepa_loss, cluster_loss, mask, m, mask_flat, pad_mask
        
        if global_step % 50 == 0:
            torch.cuda.empty_cache()
        
        # Logging
        if global_step % args.log_every == 0:
            avgs = {k: sum(v) / len(v) for k, v in metrics.items()}
            hours = total_seconds / 3600
            
            if rank0():
                pbar.set_postfix(
                    jepa=f"{avgs['jepa']:.4f}",
                    cluster=f"{avgs['cluster']:.4f}",
                    cw=f"{avgs['cw']:.3f}",
                    hrs=f"{hours:.1f}"
                )
                
                with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                    f.write(f"{global_step}\t{avgs['jepa']:.6f}\t{avgs['cluster']:.6f}\t{avgs['cw']:.4f}\t{hours:.2f}\n")
        
        # Checkpointing
        if args.save_every > 0 and global_step % args.save_every == 0:
            engine.save_checkpoint(ckpt_dir, tag=f"step{global_step}", client_state={'step': global_step})
            
            if rank0():
                torch.save({
                    'online': unwrap(engine.module).state_dict(),
                    'target': target.state_dict(),
                    'step': global_step,
                    'total_seconds': total_seconds,
                }, os.path.join(ckpt_dir, f"portable_step{global_step}.pt"))
    
    # Final checkpoint
    engine.save_checkpoint(ckpt_dir, tag="final", client_state={'step': global_step})
    
    if rank0():
        print(f"[Done] {global_step} steps, {total_seconds / 3600:.1f}h audio")


def parse_args():
    parser = argparse.ArgumentParser(description="JEPA training with GMM supervision")
    
    # Required
    parser.add_argument('--data_dir', required=True, help='Comma-separated JSONL directories')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--ds_config', required=True, help='DeepSpeed config path')
    parser.add_argument('--local_rank', type=int, default=-1, help='Local rank for distributed')
    
    # GMM
    parser.add_argument('--gmm_path', type=str, default=None, help='Single GMM path')
    parser.add_argument('--gmm_paths', type=str, default=None, help='Multiple GMMs: "path1:w1,path2:w2"')
    parser.add_argument('--cluster_temperature', type=float, default=1.0, help='GMM temperature')
    
    # Architecture
    parser.add_argument('--code_dim', type=int, default=512, help='Latent dimension')
    parser.add_argument('--channels', default='32,64,128,256', help='Encoder channels')
    parser.add_argument('--strides', default='8,8,5', help='Encoder strides')
    parser.add_argument('--n_res', type=int, default=4, help='Residual blocks per stage')
    parser.add_argument('--n_conformer', type=int, default=4, help='Conformer layers')
    parser.add_argument('--heads', type=int, default=32, help='Attention heads')
    parser.add_argument('--K', type=int, default=1024, help='Cluster classes')
    parser.add_argument('--n_mels', type=int, default=80, help='Mel bins')
    parser.add_argument('--use_gaatn', action='store_true', help='Use Gaussian Adaptive Attention')
    parser.add_argument('--use_rel_pos', action='store_true', help='Use relative position bias')
    
    # Training
    parser.add_argument('--max_steps', type=int, default=50000, help='Total steps')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size per GPU')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--ema_decay', type=float, default=0.996, help='EMA decay')
    parser.add_argument('--sample_rate', type=int, default=16000, help='Sample rate')
    parser.add_argument('--max_seconds', type=float, default=15.0, help='Max audio length')
    
    # Supervision schedule
    parser.add_argument('--cluster_weight_start', type=float, default=1.0, help='Initial cluster weight')
    parser.add_argument('--cluster_weight_end', type=float, default=0.01, help='Final cluster weight')
    
    # Augmentation
    parser.add_argument('--p_noise', type=float, default=0.25, help='Noise probability')
    parser.add_argument('--p_mix', type=float, default=0.25, help='Mix probability')
    parser.add_argument('--snr_low_noise', type=float, default=-5.0, help='Min noise SNR')
    parser.add_argument('--snr_high_noise', type=float, default=20.0, help='Max noise SNR')
    parser.add_argument('--snr_low_speech', type=float, default=-5.0, help='Min speech SNR')
    parser.add_argument('--snr_high_speech', type=float, default=5.0, help='Max speech SNR')
    
    # Checkpointing
    parser.add_argument('--save_every', type=int, default=1000, help='Checkpoint interval')
    parser.add_argument('--log_every', type=int, default=10, help='Log interval')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--pt_ckpt', type=str, default=None, help='Portable checkpoint path')
    parser.add_argument('--reinit_cluster_head', action='store_true', help='Reinitialize cluster head')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    if rank0():
        print("=" * 60)
        print(f"[JEPA + GMM] K={args.K}")
        print(f"  use_gaatn={args.use_gaatn}, use_rel_pos={args.use_rel_pos}")
        print(f"  cluster_weight: {args.cluster_weight_start} -> {args.cluster_weight_end}")
        print("=" * 60)
    
    train(args)


if __name__ == '__main__':
    main()
