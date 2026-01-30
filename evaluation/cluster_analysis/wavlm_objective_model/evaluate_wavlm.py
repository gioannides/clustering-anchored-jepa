#!/usr/bin/env python3
"""
Evaluate WavLM speech representations

For models trained with train_wavlm_fair.py

Computes:
1. Cluster entropy (utilization)
2. Adjacent frame consistency (local smoothness)
3. Visualizations for paper (UMAP, transitions, spectrograms)

Usage:
    python eval_representations_wavlm_fair.py \
        --checkpoint ./WavLM_fair/ckpts/portable_step10000.pt \
        --data "/path/to/jsonl" \
        --output_dir ./eval_wavlm

Default config matches train_wavlm_fair.py:
    --code_dim 512 --channels 32,64,128,256 --strides 8,8,5 
    --n_res 4 --n_conformer 4 --heads 32 --K 1024 --use_gaatn --use_rel_pos
"""

import os
import json
import glob
import random
import argparse
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm
import matplotlib.pyplot as plt

#plt.rcParams.update({
#    'font.size': 14,
#    'axes.titlesize': 16,
#    'axes.labelsize': 14,
#    'xtick.labelsize': 12,
#    'ytick.labelsize': 12,
#    'legend.fontsize': 12,
#    'font.family': 'sans-serif',
#})

plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'font.family': 'sans-serif',
    'axes.linewidth': 1.5,
})

from model_wavlm_objective import load_encoder


# ============================================================================
# Data Loading
# ============================================================================

def load_utterances(jsonl_dirs, n_utterances, sr=16000, max_sec=10.0, base_path="/scratch/gioannides/granary_data"):
    """Load audio utterances from JSONL files."""
    all_paths = []
    for root_dir in jsonl_dirs.split(','):
        root_dir = root_dir.strip()
        if not root_dir:
            continue
        for jf in glob.glob(os.path.join(root_dir, "*.jsonl")):
            try:
                with open(jf) as f:
                    for line in f:
                        try:
                            obj = json.loads(line.strip())
                            wp = obj.get("wav_path") or obj.get("audio_path") or obj.get("path")
                            if wp:
                                all_paths.append(wp if wp.startswith('/') else os.path.join(base_path, wp))
                        except:
                            pass
            except:
                pass
    
    print(f"[Data] Found {len(all_paths)} paths")
    random.shuffle(all_paths)
    
    utterances, max_samples = [], int(sr * max_sec)
    for wp in tqdm(all_paths, desc="Loading"):
        if len(utterances) >= n_utterances:
            break
        try:
            wav, fsr = torchaudio.load(wp)
            if wav.shape[0] > 1:
                wav = wav.mean(0, keepdim=True)
            if fsr != sr:
                wav = torchaudio.functional.resample(wav, fsr, sr)
            if wav.shape[-1] < sr:
                continue
            if wav.shape[-1] > max_samples:
                s = random.randint(0, wav.shape[-1] - max_samples)
                wav = wav[..., s:s + max_samples]
            utterances.append(wav.squeeze(0))
        except:
            pass
    
    print(f"[Data] Loaded {len(utterances)} utterances")
    return utterances


# ============================================================================
# Feature Extraction
# ============================================================================

@torch.no_grad()
def extract_representations(model, utterances, device, batch_size=8):
    """Extract continuous representations from model."""
    model.eval()
    all_z = []
    hop = model.hop
    
    for i in tqdm(range(0, len(utterances), batch_size), desc="Extracting representations"):
        batch = utterances[i:i + batch_size]
        max_len = max(w.shape[-1] for w in batch)
        max_len = ((max_len + hop - 1) // hop) * hop
        
        padded = torch.stack([F.pad(w, (0, max_len - w.shape[-1])) for w in batch])
        padded = padded.unsqueeze(1).to(device)  # [B, 1, T]
        
        z = model.encode(padded)  # [B, code_dim, T']
        
        for j, w in enumerate(batch):
            T = w.shape[-1] // hop
            all_z.append(z[j, :, :T].cpu().permute(1, 0))  # [T, code_dim]
    
    return all_z


@torch.no_grad()
def extract_cluster_assignments(model, utterances, device, batch_size=8):
    """Extract cluster assignments using model's final_proj (linear classification head)."""
    model.eval()
    all_labels = []
    all_probs = []
    hop = model.hop
    
    for i in tqdm(range(0, len(utterances), batch_size), desc="Extracting clusters"):
        batch = utterances[i:i + batch_size]
        max_len = max(w.shape[-1] for w in batch)
        max_len = ((max_len + hop - 1) // hop) * hop
        
        padded = torch.stack([F.pad(w, (0, max_len - w.shape[-1])) for w in batch])
        padded = padded.unsqueeze(1).to(device)  # [B, 1, T]
        
        z = model.encode(padded)  # [B, code_dim, T']
        logits = model.get_logits(z)  # [B, T', K]
        probs = F.softmax(logits, dim=-1)
        
        for j, w in enumerate(batch):
            T = w.shape[-1] // hop
            all_labels.append(logits[j, :T, :].argmax(dim=-1).cpu())  # [T]
            all_probs.append(probs[j, :T, :].cpu())  # [T, K]
    
    return all_labels, all_probs


# ============================================================================
# Metrics
# ============================================================================

def compute_metrics(all_labels, all_probs, k):
    """Compute entropy and adjacent consistency metrics."""
    # Flatten labels
    labels_flat = torch.cat(all_labels)
    
    # Entropy
    counts = np.bincount(labels_flat.numpy(), minlength=k)
    p = counts / counts.sum()
    p = p[p > 0]
    entropy = -np.sum(p * np.log(p))
    
    # Adjacent consistency
    same, total = 0, 0
    for labels in all_labels:
        if len(labels) < 2:
            continue
        same += (labels[:-1] == labels[1:]).sum().item()
        total += len(labels) - 1
    
    adj_cons = same / total if total > 0 else 0
    
    return entropy, entropy / np.log(k), adj_cons, labels_flat


def compute_cooccurrence(all_labels, k):
    """Compute cluster co-occurrence matrix."""
    cooccur = np.zeros((k, k))
    
    for labels in all_labels:
        labs = labels.numpy()
        for t in range(len(labs) - 1):
            cooccur[labs[t], labs[t + 1]] += 1
    
    return cooccur


# ============================================================================
# Visualizations
# ============================================================================
def plot_umap(z_flat, labels, k, path, n_samples=10000):
    """UMAP visualization of representations - no legend, larger points."""
    try:
        import umap
    except ImportError:
        print("[Skip] pip install umap-learn for UMAP visualization")
        return
    
    print("[Viz] Computing UMAP...")
    N = z_flat.shape[0]
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()
    
    if N > n_samples:
        idx = np.random.choice(N, n_samples, replace=False)
        z_sub = z_flat[idx].numpy() if isinstance(z_flat, torch.Tensor) else z_flat[idx]
        labels_sub = labels[idx]
    else:
        z_sub = z_flat.numpy() if isinstance(z_flat, torch.Tensor) else z_flat
        labels_sub = labels
    
    z_2d = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42).fit_transform(z_sub)
    
    # Top 10 clusters by frequency
    unique, counts = np.unique(labels_sub, return_counts=True)
    top = unique[np.argsort(counts)[-10:]]
    
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Plot other clusters in gray
    other = ~np.isin(labels_sub, top)
    ax.scatter(z_2d[other, 0], z_2d[other, 1], c='lightgray', s=8, alpha=0.3, rasterized=True)
    
    # Plot top clusters with larger points, no legend
    colors = plt.cm.tab10(np.arange(10))
    for i, cid in enumerate(top):
        m = labels_sub == cid
        ax.scatter(z_2d[m, 0], z_2d[m, 1], s=12, alpha=0.6, c=[colors[i]], rasterized=True)
    
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Saved] {path}")

def plot_distribution(labels, k, path, model_name='Model'):
    """Plot cluster distribution as line (for single model, or call plot_distribution_overlay for multiple)."""
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()
    
    counts = np.bincount(labels, minlength=k)
    sorted_counts = np.sort(counts)[::-1]
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(k), sorted_counts, color='steelblue', linewidth=2, alpha=0.8)
    ax.set_yscale('log')
    ax.set_xlabel('Cluster Rank')
    ax.set_ylabel('Frame Count (log)')
    ax.set_xlim(0, k)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Saved] {path}")


def plot_distribution_overlay(results_dict, k, path):
    """
    Plot overlaid cluster distributions for multiple methods.
    
    results_dict: {'Pure JEPA': labels1, 'GMM-JEPA': labels2, ...}
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    
    colors = {
        'Pure JEPA': '#d62728',
        'GMM-JEPA': '#1f77b4',
        'WavLM-style': '#7f7f7f',
        'GMM-JEPA-T': '#2ca02c',
    }
    linestyles = {
        'Pure JEPA': '-',
        'GMM-JEPA': '-',
        'WavLM-style': '--',
        'GMM-JEPA-T': '-.',
    }
    
    for name, labels in results_dict.items():
        if isinstance(labels, torch.Tensor):
            labels = labels.numpy()
        
        counts = np.bincount(labels, minlength=k)
        sorted_counts = np.sort(counts)[::-1]
        
        color = colors.get(name, 'black')
        ls = linestyles.get(name, '-')
        
        ax.plot(range(k), sorted_counts, color=color, linestyle=ls, 
                linewidth=2, alpha=0.8, label=name)
    
    ax.set_yscale('log')
    ax.set_xlabel('Cluster Rank')
    ax.set_ylabel('Frame Count (log)')
    ax.set_xlim(0, k)
    ax.legend(frameon=False, loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Saved] {path}")


def plot_transitions(cooccur, path, top_k=10):
    """Self-transition stability + top cross-transitions."""
    cluster_counts = cooccur.sum(axis=1)
    top = np.argsort(cluster_counts)[-top_k:][::-1]
    
    row_sums = cooccur.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    probs = cooccur / row_sums
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = plt.cm.tab10(np.arange(top_k))
    
    # Left: self-transition
    ax1 = axes[0]
    self_t = np.diag(probs)[top]
    bars = ax1.barh(range(top_k), self_t, color=colors)
    ax1.set_yticks(range(top_k))
    ax1.set_yticklabels([f'Cluster {c}' for c in top])
    ax1.set_xlabel('Self-Transition Probability')
    ax1.set_title('Cluster Stability')
    ax1.set_xlim(0, 1)
    ax1.invert_yaxis()
    for bar, val in zip(bars, self_t):
        ax1.text(val + 0.02, bar.get_y() + bar.get_height()/2, f'{val:.0%}', va='center')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Right: cross-transitions
    ax2 = axes[1]
    trans = [(c1, c2, probs[c1, c2]) for c1 in top for c2 in top if c1 != c2 and probs[c1, c2] > 0.02]
    trans.sort(key=lambda x: -x[2])
    trans = trans[:12]
    
    if trans:
        labels_list = [f'{c1}→{c2}' for c1, c2, _ in trans]
        vals = [p for _, _, p in trans]
        bar_colors = [colors[list(top).index(c1)] for c1, _, _ in trans]
        
        ax2.barh(range(len(trans)), vals, color=bar_colors)
        ax2.set_yticks(range(len(trans)))
        ax2.set_yticklabels(labels_list)
        ax2.set_xlabel('Transition Probability')
        ax2.set_title('Top Cross-Transitions')
        ax2.invert_yaxis()
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Saved] {path}")


def plot_utterance(all_labels, all_probs, idx, path, hop_ms, k):
    """Plot cluster assignments over time for a single utterance."""
    labels = all_labels[idx].numpy()
    probs = all_probs[idx].numpy()
    T = len(labels)
    
    time_sec = np.arange(T) * hop_ms / 1000
    unique = np.unique(labels)
    cmap = plt.cm.tab20(np.arange(len(unique)))
    c2col = {l: cmap[i] for i, l in enumerate(unique)}
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 4), sharex=True, 
                              gridspec_kw={'height_ratios': [1.5, 1], 'hspace': 0.05})
    
    ax1 = axes[0]
    for t in range(T - 1):
        ax1.axvspan(time_sec[t], time_sec[t+1], color=c2col[labels[t]], alpha=0.85)
    ax1.set_ylabel('Cluster')
    ax1.set_yticks([])
    ax1.set_xlim(0, time_sec[-1])
    for s in ['top', 'right', 'bottom']:
        ax1.spines[s].set_visible(False)
    
    ax2 = axes[1]
    ent = -np.sum(probs * np.log(probs + 1e-10), axis=1)
    conf = 1 - ent / np.log(k)
    ax2.fill_between(time_sec, 0, conf, alpha=0.7, color='steelblue')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Confidence')
    ax2.set_ylim(0, 1)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Saved] {path}")

def plot_spectrograms(utterances, all_labels, path, n_clusters=3, n_examples=5, sr=16000, hop=320,
                      cluster_labels=None):
    """
    Plot mel spectrograms grouped by cluster with acoustic type labels.
    
    cluster_labels: dict mapping cluster_id -> acoustic_type label (e.g., {0: 'Silence', 5: 'Voiced'})
                    If None, uses generic labels.
    """
    from torchaudio.transforms import MelSpectrogram
    mel = MelSpectrogram(sample_rate=sr, n_fft=1024, hop_length=hop, n_mels=80)
    
    # Collect frames by cluster
    cluster_frames = defaultdict(list)
    for ui, labels in enumerate(all_labels):
        labs = labels.numpy() if isinstance(labels, torch.Tensor) else labels
        for t, l in enumerate(labs):
            if 5 < t < len(labs) - 5:
                cluster_frames[l].append((ui, t))
    
    # Select clusters to plot
    if cluster_labels is not None:
        clusters_to_plot = list(cluster_labels.keys())[:n_clusters]
    else:
        top = sorted(cluster_frames.items(), key=lambda x: -len(x[1]))[:n_clusters]
        clusters_to_plot = [cid for cid, _ in top]
        cluster_labels = {cid: f'Cluster {i+1}' for i, cid in enumerate(clusters_to_plot)}
    
    n_rows = len(clusters_to_plot)
    fig, axes = plt.subplots(n_rows, n_examples, figsize=(n_examples * 2, n_rows * 1.5))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    ctx = 8  # Context frames on each side
    
    for i, cid in enumerate(clusters_to_plot):
        frames = cluster_frames.get(cid, [])
        random.shuffle(frames)
        
        for j in range(n_examples):
            ax = axes[i, j]
            
            if j >= len(frames):
                ax.axis('off')
                continue
            
            ui, t = frames[j]
            wav = utterances[ui]
            
            center = t * hop
            seg = wav[max(0, center - ctx*hop):min(len(wav), center + (ctx+1)*hop)]
            
            if len(seg) < hop:
                ax.axis('off')
                continue
            
            m = torch.log(mel(seg.unsqueeze(0)) + 1e-6).squeeze().numpy()
            ax.imshow(m, aspect='auto', origin='lower', cmap='magma')
            
            # White band marking the target frame
            cx = m.shape[1] // 2
            ax.axvspan(cx - 0.5, cx + 0.5, color='white', alpha=0.4)
            
            ax.set_xticks([])
            ax.set_yticks([])
            
            for s in ax.spines.values():
                s.set_visible(False)
            
            # Row label (acoustic type)
            if j == 0:
                ax.set_ylabel(cluster_labels.get(cid, f'C{cid}'), fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Saved] {path}")

# ============================================================================
# Main
# ============================================================================

def main(args):
    device = args.device if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Build config (matches train_wavlm_fair.py)
    config = {
        "code_dim": args.code_dim,
        "channels": [int(x) for x in args.channels.split(',')],
        "strides": [int(x) for x in args.strides.split(',')],
        "n_res": args.n_res,
        "n_conformer": args.n_conformer,
        "heads": args.heads,
        "K": args.K,
        "use_gaatn": args.use_gaatn,
        "use_rel_pos": args.use_rel_pos,
        "dropout": args.dropout,
    }
    
    print(f"[Config] {config}")
    print(f"[Model] Loading WavLM encoder from {args.checkpoint}")
    
    model = load_encoder(args.checkpoint, config=config, device=device, strict=False)
    hop_ms = model.hop / args.sample_rate * 1000
    
    print(f"[Model] Hop: {model.hop} samples ({hop_ms:.1f}ms)")
    print(f"[Model] Code dim: {model.code_dim}, K: {model.num_classes}")
    
    # Load data
    utterances = load_utterances(args.data, args.n_utterances, args.sample_rate, 
                                  args.max_seconds, args.base_path)
    
    if len(utterances) == 0:
        print("[Error] No utterances loaded!")
        return
    
    # Extract representations and cluster assignments
    all_z = extract_representations(model, utterances, device, args.batch_size)
    all_labels, all_probs = extract_cluster_assignments(model, utterances, device, args.batch_size)
    
    z_flat = torch.cat(all_z, dim=0)
    print(f"[Data] {z_flat.shape[0]:,} frames, dim={z_flat.shape[1]}")
    
    # Compute metrics
    K = args.K
    entropy, norm_ent, adj_cons, labels_flat = compute_metrics(all_labels, all_probs, K)
    cooccur = compute_cooccurrence(all_labels, K)
    
    print(f"\n{'='*60}")
    print(f"[Metrics] WavLM Model")
    print(f"  Entropy: {entropy:.3f} (normalized: {norm_ent:.1%})")
    print(f"  Adjacent consistency: {adj_cons:.3f}")
    print(f"  Used clusters: {(np.bincount(labels_flat.numpy(), minlength=K) > 0).sum()}/{K}")
    print(f"{'='*60}\n")
    
    # Visualizations
    print("[Viz] Generating visualizations...")
    
    plot_umap(z_flat, labels_flat, K, os.path.join(args.output_dir, 'umap.png'), args.viz_samples)
    plot_distribution(labels_flat, K, os.path.join(args.output_dir, 'distribution.png'))
    plot_transitions(cooccur, os.path.join(args.output_dir, 'transitions.png'))
    
    for i in range(min(3, len(all_labels))):
        plot_utterance(all_labels, all_probs, i, 
                      os.path.join(args.output_dir, f'utterance_{i}.png'), hop_ms, K)
    
    # Optional: define acoustic labels for top clusters (requires manual inspection)
    # cluster_labels = {0: 'Silence', 5: 'Voiced', 12: 'Fricative'}
    cluster_labels = None  # Uses automatic top-3 clusters with generic labels
    
    plot_spectrograms(utterances, all_labels, os.path.join(args.output_dir, 'spectrograms.png'),
                      n_clusters=3, n_examples=5, sr=args.sample_rate, hop=model.hop,
                      cluster_labels=cluster_labels)
    
    # Save results
    results = {
        'model_type': 'wavlm_fair',
        'entropy': float(entropy),
        'entropy_normalized': float(norm_ent),
        'adjacent_consistency': float(adj_cons),
        'n_frames': int(z_flat.shape[0]),
        'n_clusters': K,
        'used_clusters': int((np.bincount(labels_flat.numpy(), minlength=K) > 0).sum()),
        'config': config,
        'hop_ms': hop_ms,
    }

    # Save labels for combined plotting
    torch.save(labels_flat, os.path.join(args.output_dir, 'labels.pt'))
    
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n[Done] Results saved to {args.output_dir}")


if __name__ == '__main__':
    p = argparse.ArgumentParser(description="Evaluate WavLM speech representations")
    
    # Required
    p.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    p.add_argument('--data', required=True, help='Comma-separated JSONL directories')
    
    # Output
    p.add_argument('--output_dir', default='./eval_wavlm_results', help='Output directory')
    
    # Model config (defaults match train_wavlm_fair.py example command)
    p.add_argument('--code_dim', type=int, default=512, help='Code dimension')
    p.add_argument('--channels', default='32,64,128,256', help='Encoder channels')
    p.add_argument('--strides', default='8,8,5', help='Encoder strides')
    p.add_argument('--n_res', type=int, default=4, help='Number of residual blocks')
    p.add_argument('--n_conformer', type=int, default=4, help='Number of conformer blocks')
    p.add_argument('--heads', type=int, default=32, help='Number of attention heads')
    p.add_argument('--K', type=int, default=1024, help='Number of cluster codes')
    p.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    p.add_argument('--use_gaatn', action='store_true', default=True, help='Use Gaussian Adaptive Attention')
    p.add_argument('--no_gaatn', action='store_false', dest='use_gaatn', help='Disable GAATN')
    p.add_argument('--use_rel_pos', action='store_true', default=True, help='Use relative position bias')
    p.add_argument('--no_rel_pos', action='store_false', dest='use_rel_pos', help='Disable rel pos bias')
    
    # Data
    p.add_argument('--base_path', default='/scratch/gioannides/granary_data',
                   help='Base path for relative audio paths')
    p.add_argument('--n_utterances', type=int, default=1000, help='Number of utterances to load')
    p.add_argument('--max_seconds', type=float, default=10.0, help='Max utterance length')
    p.add_argument('--sample_rate', type=int, default=16000, help='Audio sample rate')
    
    # Runtime
    p.add_argument('--batch_size', type=int, default=8, help='Batch size for extraction')
    p.add_argument('--device', default='cuda', help='Device')
    p.add_argument('--viz_samples', type=int, default=10000, help='Samples for UMAP')
    
    args = p.parse_args()
    main(args)

