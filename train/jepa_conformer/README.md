# Phase 2: JEPA Training

Train a Joint Embedding Predictive Architecture with frozen GMM supervision.

## Overview

The encoder learns speech representations through two objectives:
1. **JEPA Loss**: Predict masked latent representations from an EMA teacher
2. **Cluster Loss**: Match cluster head outputs to frozen GMM posteriors

A decaying supervision schedule allows GMM regularization to dominate early training before gradually yielding to the JEPA objective.

## Prerequisites

1. Fitted GMM checkpoint from Phase 1:
```bash
   python train/gmm/fit_gmm.py --data_dir /path/to/data --output_dir ./checkpoints/gmm
```

2. DeepSpeed configuration file (see `configs/ds_config.json`)

## Usage

### Basic Training
```bash
deepspeed train.py \
  --data_dir /path/to/jsonl \
  --output_dir ./checkpoints/jepa \
  --gmm_path ./checkpoints/gmm/gmm.pt \
  --ds_config ../configs/ds_config.json \
  --max_steps 50000
```

### Multi-GPU Training
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 deepspeed train.py \
  --data_dir /path/to/jsonl \
  --output_dir ./checkpoints/jepa \
  --gmm_path ./checkpoints/gmm/gmm.pt \
  --ds_config ../configs/ds_config.json \
  --max_steps 50000 \
  --batch_size 96
```

### Resume Training
```bash
deepspeed train.py \
  --data_dir /path/to/jsonl \
  --output_dir ./checkpoints/jepa \
  --gmm_path ./checkpoints/gmm/gmm.pt \
  --ds_config ../configs/ds_config.json \
  --resume
```

### Resume from Portable Checkpoint
```bash
deepspeed train.py \
  --data_dir /path/to/jsonl \
  --output_dir ./checkpoints/jepa \
  --gmm_path ./checkpoints/gmm/gmm.pt \
  --ds_config ../configs/ds_config.json \
  --pt_ckpt ./checkpoints/jepa/ckpts/portable_step10000.pt
```

### Multiple GMMs (Ensemble)
```bash
deepspeed train.py \
  --data_dir /path/to/jsonl \
  --output_dir ./checkpoints/jepa \
  --gmm_paths "gmm1.pt:0.5,gmm2.pt:0.3,gmm3.pt:0.2" \
  --ds_config ../configs/ds_config.json
```

## Arguments

### Required

| Argument | Description |
|----------|-------------|
| `--data_dir` | Comma-separated directories containing JSONL files |
| `--output_dir` | Output directory for checkpoints |
| `--ds_config` | Path to DeepSpeed config JSON |
| `--gmm_path` | Path to fitted GMM (or use `--gmm_paths`) |

### Model Architecture

| Argument | Default | Description |
|----------|---------|-------------|
| `--code_dim` | 512 | Latent dimension |
| `--channels` | 32,64,128,256 | Encoder channel progression |
| `--strides` | 8,8,5 | Encoder strides (hop = product) |
| `--n_res` | 4 | Residual blocks per encoder stage |
| `--n_conformer` | 4 | Number of Conformer layers |
| `--heads` | 32 | Attention heads |
| `--K` | 1024 | Number of cluster classes |
| `--use_gaatn` | flag | Enable Gaussian Adaptive Attention |
| `--use_rel_pos` | flag | Enable gated relative position bias |

### Training

| Argument | Default | Description |
|----------|---------|-------------|
| `--max_steps` | 50000 | Total training steps |
| `--batch_size` | 32 | Batch size per GPU |
| `--lr` | 1e-5 | Learning rate |
| `--ema_decay` | 0.996 | EMA decay for target encoder |
| `--max_seconds` | 15.0 | Max audio length |

### GMM Supervision

| Argument | Default | Description |
|----------|---------|-------------|
| `--cluster_weight_start` | 1.0 | Initial cluster loss weight |
| `--cluster_weight_end` | 0.01 | Final cluster loss weight |
| `--cluster_temperature` | 1.0 | Temperature for soft labels |

### Augmentation

| Argument | Default | Description |
|----------|---------|-------------|
| `--p_noise` | 0.25 | Noise addition probability |
| `--p_mix` | 0.25 | Utterance mixing probability |
| `--snr_low_noise` | -5.0 | Min SNR for noise (dB) |
| `--snr_high_noise` | 20.0 | Max SNR for noise (dB) |
| `--snr_low_speech` | -5.0 | Min SNR for speech mixing (dB) |
| `--snr_high_speech` | 5.0 | Max SNR for speech mixing (dB) |

### Checkpointing

| Argument | Default | Description |
|----------|---------|-------------|
| `--save_every` | 1000 | Save checkpoint every N steps |
| `--log_every` | 10 | Log metrics every N steps |
| `--resume` | flag | Resume from latest checkpoint |
| `--pt_ckpt` | None | Resume from portable .pt checkpoint |

## Output
```
output_dir/
├── ckpts/
│   ├── step1000/              # DeepSpeed checkpoint
│   ├── step2000/
│   ├── portable_step1000.pt   # Portable checkpoint
│   ├── portable_step2000.pt
│   └── ...
├── log.txt                    # Training metrics
└── seen_hashes.pt             # Processed audio tracking
```

### Portable Checkpoint Format
```python
{
    'online': dict,        # Online encoder state_dict
    'target': dict,        # Target encoder state_dict
    'step': int,           # Training step
    'total_seconds': float # Total audio processed (seconds)
}
```

## DeepSpeed Configuration

Example `ds_config.json`:
```json
{
  "train_batch_size": "auto",
  "gradient_accumulation_steps": 1,
  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "initial_scale_power": 16
  },
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {
      "device": "none"
    }
  },
  "gradient_clipping": 1.0
}
```

## Architecture Overview
```
┌─────────────────────────────────────────────────────────────┐
│                      Online Encoder                          │
├─────────────────────────────────────────────────────────────┤
│  Input Conv → [Encoder Blocks] → Proj → [Conformers] → Agg  │
│                                                              │
│  Encoder Block: Conv + Snake-Beta + ResBlocks + DAAM        │
│  Conformer: FFN + MHSA (rel pos) + ConvModule + FFN         │
│  Aggregation: Cross-attention over layers                    │
├─────────────────────────────────────────────────────────────┤
│                      Outputs                                 │
│  ├── z_online: Aggregated representations                   │
│  ├── z_pred: Predictor output (for JEPA loss)              │
│  └── cluster_logits: Cluster head output (for GMM loss)    │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                      Target Encoder                          │
├─────────────────────────────────────────────────────────────┤
│  Same architecture as Online (no gradient)                   │
│  Updated via EMA: θ_target ← τ·θ_target + (1-τ)·θ_online   │
└─────────────────────────────────────────────────────────────┘
```

## Loss Functions

### JEPA Loss
```
L_jepa = MSE(predictor(masked_z), z_target) over masked positions
```

### Cluster Loss
```
L_cluster = KL(GMM_posteriors || softmax(cluster_logits)) over masked positions
```

### Total Loss
```
L_total = L_jepa + λ(t) · L_cluster

where λ(t) decays linearly from cluster_weight_start to cluster_weight_end
```

## File Structure
```
train/jepa/
├── README.md       # This file
├── __init__.py     # Package exports
├── train.py        # Main training script
├── model.py        # Encoder architectures
├── augment.py      # Denoising augmentation
├── dataset.py      # Streaming dataset
└── utils.py        # Utilities (masking, EMA, etc.)
```
