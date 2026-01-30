#!/usr/bin/env python3
"""
Pure JEPA training - no cluster loss, just masked prediction.

Usage:
    deepspeed train.py \
        --data_dir /path/to/jsonl \
        --output_dir ./checkpoints/jepa_pure \
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
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import OnlineEncoder, TargetEncoder
from augment import DenoiseAugmentor
from dataset import StreamingDataset, make_collate
from utils import rank0, unwrap, ema_update, create_mask, create_padding_mask


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
        prefetch_factor=8,
        persistent_workers=False,
        timeout=60,
    )

    online = OnlineEncoder(
        args.code_dim, channels, strides,
        args.n_res, args.n_conformer, args.heads,
        args.use_gaatn, args.use_rel_pos
    )

    if rank0():
        enc_params = sum(
            p.numel() for n, p in online.named_parameters()
            if not any(x in n for x in ['predictor', 'mask_token'])
        )
        total_params = sum(p.numel() for p in online.parameters())
        print(f"[Model] Encoder: {enc_params / 1e6:.1f}M, Total: {total_params / 1e6:.1f}M params")

    optimizer = torch.optim.AdamW(online.parameters(), lr=args.lr, weight_decay=1e-3, betas=(0.8, 0.99))

    with open(args.ds_config) as f:
        ds_cfg = json.load(f)

    if torch.cuda.is_available() and args.local_rank >= 0:
        torch.cuda.set_device(args.local_rank)

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

        online.load_state_dict(state_dict, strict=False)

        if rank0():
            print(f"[Resume] Loaded step {global_step}")

    engine, _, _, _ = deepspeed.initialize(
        args=args,
        model=online,
        optimizer=optimizer,
        model_parameters=online.parameters(),
        config=ds_cfg
    )
    device = engine.device
    dtype = next(engine.module.parameters()).dtype

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

    augmentor = DenoiseAugmentor(
        p_noise=args.p_noise,
        p_mix=args.p_mix,
        snr_range_noise=(args.snr_low_noise, args.snr_high_noise),
        snr_range_speech=(args.snr_low_speech, args.snr_high_speech)
    )

    if rank0():
        print(f"[Augment] p_noise={args.p_noise}, p_mix={args.p_mix}")
        print(f"[GAATN] {'Enabled' if args.use_gaatn else 'Disabled'}")
        print(f"[RelPos] {'Enabled' if args.use_rel_pos else 'Disabled'}")

    ckpt_dir = os.path.join(args.output_dir, "ckpts")
    os.makedirs(ckpt_dir, exist_ok=True)

    total_seconds = 0.0

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

    metrics = {'jepa': deque(maxlen=50)}
    pbar = tqdm(total=args.max_steps, disable=not rank0(), initial=global_step, desc="Training")

    dl_iter = iter(dataloader)

    while global_step < args.max_steps:
        try:
            wav, paths, batch_secs, lengths = next(dl_iter)
        except StopIteration:
            dl_iter = iter(dataloader)
            wav, paths, batch_secs, lengths = next(dl_iter)

        if wav is None:
            continue

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

        wav_aug, wav_clean = augmentor(wav)

        with torch.no_grad():
            z_target = target(wav_clean)
            T_z = z_target.shape[-1]

        # Compute frame lengths from sample lengths
        wav_lens = torch.tensor(lengths, device=device)
        frame_lengths = (wav_lens // hop).clamp(max=T_z).tolist()

        mask = create_mask(B, T_z, device, frame_lengths).to(dtype)
        pad_mask = (torch.arange(T_z, device=device)[None, :] < torch.tensor(frame_lengths, device=device)[:, None]).to(dtype)

        z_online, z_pred = engine.module(wav_aug, mask)

        # JEPA loss
        m = (1 - mask).unsqueeze(1) * pad_mask.unsqueeze(1)
        jepa_loss = ((z_pred - z_target.detach()).pow(2) * m).sum() / (m.sum() * z_pred.shape[1]).clamp_min(1)

        if torch.isnan(jepa_loss):
            if rank0():
                print("[DEBUG] NaN loss!")
            continue

        engine.backward(jepa_loss)
        engine.step()
        ema_update(target, engine.module, decay=args.ema_decay)

        metrics['jepa'].append(jepa_loss.item())
        global_step += 1

        if rank0():
            pbar.update(1)

        del z_online, z_pred, z_target, wav_aug, wav_clean
        del jepa_loss, mask, m, pad_mask

        if global_step % 50 == 0:
            torch.cuda.empty_cache()

        if global_step % args.log_every == 0:
            avg_jepa = sum(metrics['jepa']) / len(metrics['jepa'])
            hours = total_seconds / 3600

            if rank0():
                pbar.set_postfix(jepa=f"{avg_jepa:.4f}", hrs=f"{hours:.1f}")
                with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                    f.write(f"{global_step}\t{avg_jepa:.6f}\t{hours:.2f}\n")

        if args.save_every > 0 and global_step % args.save_every == 0:
            engine.save_checkpoint(ckpt_dir, tag=f"step{global_step}", client_state={'step': global_step})

            if rank0():
                torch.save({
                    'online': unwrap(engine.module).state_dict(),
                    'target': target.state_dict(),
                    'step': global_step,
                    'total_seconds': total_seconds,
                }, os.path.join(ckpt_dir, f"portable_step{global_step}.pt"))

    engine.save_checkpoint(ckpt_dir, tag="final", client_state={'step': global_step})

    if rank0():
        print(f"[Done] {global_step} steps, {total_seconds / 3600:.1f}h audio")


def parse_args():
    parser = argparse.ArgumentParser(description="Pure JEPA training")

    parser.add_argument('--data_dir', required=True, help='Comma-separated JSONL directories')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--ds_config', required=True, help='DeepSpeed config path')
    parser.add_argument('--local_rank', type=int, default=-1)

    # Architecture
    parser.add_argument('--code_dim', type=int, default=512)
    parser.add_argument('--channels', default='32,64,128,256')
    parser.add_argument('--strides', default='8,8,5')
    parser.add_argument('--n_res', type=int, default=4)
    parser.add_argument('--n_conformer', type=int, default=4)
    parser.add_argument('--heads', type=int, default=32)
    parser.add_argument('--use_gaatn', action='store_true')
    parser.add_argument('--use_rel_pos', action='store_true')

    # Training
    parser.add_argument('--max_steps', type=int, default=50000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--ema_decay', type=float, default=0.996)
    parser.add_argument('--sample_rate', type=int, default=16000)
    parser.add_argument('--max_seconds', type=float, default=15.0)

    # Augmentation
    parser.add_argument('--p_noise', type=float, default=0.25)
    parser.add_argument('--p_mix', type=float, default=0.25)
    parser.add_argument('--snr_low_noise', type=float, default=-5.0)
    parser.add_argument('--snr_high_noise', type=float, default=20.0)
    parser.add_argument('--snr_low_speech', type=float, default=-5.0)
    parser.add_argument('--snr_high_speech', type=float, default=5.0)

    # Checkpointing
    parser.add_argument('--save_every', type=int, default=1000)
    parser.add_argument('--log_every', type=int, default=10)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--pt_ckpt', type=str, default=None)

    return parser.parse_args()


def main():
    args = parse_args()

    if rank0():
        print("=" * 60)
        print("[JEPA Pure] No cluster loss")
        print(f"  use_gaatn={args.use_gaatn}, use_rel_pos={args.use_rel_pos}")
        print(f"  denoise: p_noise={args.p_noise}, p_mix={args.p_mix}")
        print("=" * 60)

    train(args)


if __name__ == '__main__':
    main()
