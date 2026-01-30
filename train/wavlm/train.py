#!/usr/bin/env python3
"""
WavLM training with K-means pseudo-labels.

Usage:
    deepspeed train.py \
        --data_dir /path/to/jsonl \
        --output_dir ./checkpoints/wavlm \
        --ds_config ../configs/ds_config.json \
        --kmeans_path ./checkpoints/kmeans/kmeans.pt
"""

import argparse
import json
import math
import os
from collections import deque

import deepspeed
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import WavLMEncoder
from labeler import KMeansLabeler
from augment import DenoiseAugmentor
from dataset import StreamingDataset, make_collate
from utils import rank0, unwrap, create_mask, create_padding_mask


def train(args):
    os.makedirs(args.output_dir, exist_ok=True)

    channels = [int(x) for x in args.channels.split(',')]
    strides = [int(x) for x in args.strides.split(',')]
    hop = math.prod(strides)

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
        prefetch_factor=4,
        persistent_workers=False,
        timeout=60,
    )

    model = WavLMEncoder(
        code_dim=args.code_dim,
        channels=channels,
        strides=strides,
        n_res=args.n_res,
        n_conformer=args.n_conformer,
        heads=args.heads,
        num_classes=args.K,
        use_gaatn=args.use_gaatn,
        use_rel_pos=args.use_rel_pos,
        dropout=args.dropout,
    )

    if rank0():
        total_params = sum(p.numel() for p in model.parameters())
        print(f"[WavLM] Total params: {total_params / 1e6:.1f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.98))

    with open(args.ds_config) as f:
        ds_cfg = json.load(f)

    if torch.cuda.is_available() and args.local_rank >= 0:
        torch.cuda.set_device(args.local_rank)

    engine, _, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        optimizer=optimizer,
        model_parameters=model.parameters(),
        config=ds_cfg,
    )
    device = engine.device
    dtype = next(engine.module.parameters()).dtype

    # Load K-means labeler
    if not args.kmeans_path or not os.path.exists(args.kmeans_path):
        raise ValueError(f"K-means path required: {args.kmeans_path}")

    labeler = KMeansLabeler.from_file(args.kmeans_path, device)
    if rank0():
        print(f"[KMeans] Loaded K={labeler.K}, dim={labeler.dim}")

    # Feature transform (log-mel to match k-means)
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=args.sample_rate,
        n_fft=400,
        hop_length=hop,
        n_mels=labeler.dim,
    ).to(device)

    if rank0():
        print(f"[Features] Using log-mel (dim={labeler.dim})")

    augmentor = DenoiseAugmentor(p_noise=args.p_noise, p_mix=args.p_mix)

    ckpt_dir = os.path.join(args.output_dir, "ckpts")
    os.makedirs(ckpt_dir, exist_ok=True)

    global_step = 0
    total_seconds = 0.0

    if args.resume:
        try:
            _, client_sd = engine.load_checkpoint(ckpt_dir, tag=None)
            if client_sd and 'step' in client_sd:
                global_step = client_sd['step']
                total_seconds = client_sd.get('total_seconds', 0.0)
                if rank0():
                    print(f"[Resume] Step {global_step}, hours={total_seconds / 3600:.1f}h")
        except Exception as e:
            if rank0():
                print(f"[Resume] Failed: {e}")

    metrics = {k: deque(maxlen=50) for k in ['loss', 'acc']}
    pbar = tqdm(total=args.max_steps, disable=not rank0(), initial=global_step, desc="WavLM")

    dl_iter = iter(dataloader)

    while global_step < args.max_steps:
        try:
            wav, paths, batch_secs, frame_lengths = next(dl_iter)
        except StopIteration:
            dl_iter = iter(dataloader)
            wav, paths, batch_secs, frame_lengths = next(dl_iter)

        if wav is None:
            continue

        batch_secs_tensor = torch.tensor(batch_secs, device=device, dtype=torch.float32)
        if dist.is_initialized():
            dist.all_reduce(batch_secs_tensor)
        total_seconds += batch_secs_tensor.item() / args.sample_rate

        wav = wav.to(device, dtype)
        B = wav.shape[0]

        if torch.isnan(wav).any() or torch.isinf(wav).any():
            continue

        wav_aug, wav_clean = augmentor(wav)

        # Get pseudo-labels from log-mel features
        with torch.no_grad():
            mel = mel_transform(wav_clean.squeeze(1))
            log_mel = torch.log(mel + 1e-6).transpose(1, 2)  # [B, T, D]
            T_feat = log_mel.shape[1]
            labels = labeler.assign(log_mel)  # [B, T]

        mask = create_mask(B, T_feat, device, frame_lengths)
        pad_mask = create_padding_mask(B, T_feat, frame_lengths, device)

        logits, features = engine.module(wav_aug, mask)

        # Align dimensions
        T_out = logits.shape[1]
        if T_out != T_feat:
            min_T = min(T_out, T_feat)
            logits = logits[:, :min_T]
            labels = labels[:, :min_T]
            mask = mask[:, :min_T]
            pad_mask = pad_mask[:, :min_T]

        # Masked prediction loss
        mask_positions = ((1 - mask) * pad_mask).bool()

        if mask_positions.any():
            pred = logits[mask_positions]
            tgt = labels[mask_positions]
            loss = F.cross_entropy(pred, tgt)

            with torch.no_grad():
                acc = (pred.argmax(dim=-1) == tgt).float().mean()
        else:
            loss = torch.tensor(0.0, device=device)
            acc = torch.tensor(0.0, device=device)

        if torch.isnan(loss):
            continue

        engine.backward(loss)
        engine.step()

        metrics['loss'].append(loss.item())
        metrics['acc'].append(acc.item())
        global_step += 1

        if rank0():
            pbar.update(1)

        if global_step % args.log_every == 0:
            avgs = {k: sum(v) / len(v) for k, v in metrics.items()}
            hours = total_seconds / 3600
            if rank0():
                pbar.set_postfix(loss=f"{avgs['loss']:.4f}", acc=f"{avgs['acc']:.3f}", hrs=f"{hours:.1f}")
                with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                    f.write(f"{global_step}\t{avgs['loss']:.6f}\t{avgs['acc']:.4f}\t{hours:.2f}\n")

        if args.save_every > 0 and global_step % args.save_every == 0:
            engine.save_checkpoint(
                ckpt_dir,
                tag=f"step{global_step}",
                client_state={'step': global_step, 'total_seconds': total_seconds},
            )
            if rank0():
                torch.save({
                    'model': unwrap(engine.module).state_dict(),
                    'step': global_step,
                    'total_seconds': total_seconds,
                }, os.path.join(ckpt_dir, f"portable_step{global_step}.pt"))

        del logits, features, labels, mask, pad_mask
        if global_step % 50 == 0:
            torch.cuda.empty_cache()

    engine.save_checkpoint(
        ckpt_dir, tag="final",
        client_state={'step': global_step, 'total_seconds': total_seconds},
    )
    if rank0():
        torch.save({
            'model': unwrap(engine.module).state_dict(),
            'step': global_step,
            'total_seconds': total_seconds,
        }, os.path.join(ckpt_dir, "portable_final.pt"))
        print(f"[Done] {global_step} steps, {total_seconds / 3600:.1f}h audio")


def parse_args():
    parser = argparse.ArgumentParser(description="WavLM training")

    parser.add_argument('--data_dir', required=True, help='Comma-separated JSONL directories')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--ds_config', required=True)
    parser.add_argument('--local_rank', type=int, default=-1)

    # K-means
    parser.add_argument('--kmeans_path', required=True, help='K-means checkpoint')
    parser.add_argument('--K', type=int, default=1024)

    # Audio
    parser.add_argument('--sample_rate', type=int, default=16000)
    parser.add_argument('--max_seconds', type=float, default=15.0)

    # Architecture
    parser.add_argument('--code_dim', type=int, default=512)
    parser.add_argument('--channels', default='32,64,128,256')
    parser.add_argument('--strides', default='8,8,5')
    parser.add_argument('--n_res', type=int, default=4)
    parser.add_argument('--n_conformer', type=int, default=4)
    parser.add_argument('--heads', type=int, default=32)
    parser.add_argument('--use_gaatn', action='store_true')
    parser.add_argument('--use_rel_pos', action='store_true')
    parser.add_argument('--dropout', type=float, default=0.1)

    # Training
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_steps', type=int, default=50000)
    parser.add_argument('--lr', type=float, default=1e-5)

    # Augmentation
    parser.add_argument('--p_noise', type=float, default=0.25)
    parser.add_argument('--p_mix', type=float, default=0.25)

    # Logging
    parser.add_argument('--log_every', type=int, default=10)
    parser.add_argument('--save_every', type=int, default=1000)
    parser.add_argument('--resume', action='store_true')

    return parser.parse_args()


def main():
    args = parse_args()

    if rank0():
        print("=" * 60)
        print("[WavLM Training]")
        print(f"  K={args.K}, code_dim={args.code_dim}")
        print(f"  kmeans_path={args.kmeans_path}")
        print("=" * 60)

    train(args)


if __name__ == '__main__':
    main()
