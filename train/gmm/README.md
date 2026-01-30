```markdown
# Phase 1: GMM Fitting

Fit a diagonal-covariance Gaussian Mixture Model on log-mel spectrograms using gradient-based optimization with multi-GPU support.

## Overview

The GMM is fitted **once** on input features and remains frozen during JEPA training. This provides stable acoustic targets that prevent representation collapse.

**Key features:**
- Diagonal covariance (equivalent to `sklearn.GaussianMixture(covariance_type='diag')`)
- K-means++ initialization
- Multi-GPU support via `torchrun`
- Streaming data loading for large datasets
- Automatic checkpointing and resume

## Usage

### Single GPU

```bash
python fit_gmm.py \
  --data_dir /path/to/jsonl \
  --output_dir ./checkpoints/gmm \
  --K 1024 \
  --target_frames 9600000000
```

### Multi-GPU

```bash
torchrun --nproc_per_node=8 fit_gmm.py \
  --data_dir /path/to/jsonl \
  --output_dir ./checkpoints/gmm \
  --K 1024 \
  --target_frames 9600000000
```

### Resume from Checkpoint

```bash
torchrun --nproc_per_node=8 fit_gmm.py \
  --data_dir /path/to/jsonl \
  --output_dir ./checkpoints/gmm \
  --K 1024 \
  --target_frames 9600000000 \
  --resume ./checkpoints/gmm/gmm_ckpt_350M.pt
```

## Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_dir` | *required* | Directory containing JSONL files |
| `--output_dir` | *required* | Output directory for checkpoints |
| `--K` | 1024 | Number of GMM components |
| `--n_mels` | 80 | Number of mel bins |
| `--sample_rate` | 16000 | Audio sample rate |
| `--strides` | 8,8,5 | Encoder strides (for hop calculation) |
| `--max_seconds` | 15.0 | Max audio length per utterance |
| `--base_path` | None | Base path for relative audio paths |
| `--target_frames` | 100M | Total frames to process |
| `--batch_update_size` | 5M | Frames per SGD update cycle |
| `--max_samples` | 50M | Max samples per update |
| `--n_sgd_steps` | 2000 | SGD steps per update |
| `--sgd_batch_size` | 16384 | SGD mini-batch size |
| `--lr` | 1e-2 | Learning rate |
| `--batch_size` | 64 | Audio batch size |
| `--save_every` | 50M | Checkpoint every N frames |
| `--resume` | None | Resume from checkpoint path |

## Data Format

The script expects JSONL files with audio paths:

```json
{"wav_path": "/path/to/audio.wav"}
{"wav_path": "relative/path/audio.wav"}
```

Relative paths are resolved using `--base_path` if provided.

## Output

```
output_dir/
├── gmm.pt              # Final GMM checkpoint
├── gmm_ckpt_*M.pt      # Intermediate checkpoints
└── metrics.tsv         # Training metrics
```

### Checkpoint Format

```python
{
    'means': torch.Tensor,       # [K, dim]
    'covariances': torch.Tensor, # [K, dim] (diagonal)
    'weights': torch.Tensor,     # [K]
    'log_vars': torch.Tensor,    # [K, dim]
    'log_weights': torch.Tensor, # [K]
    'K': int,
    'dim': int,
}
```

### Loading a Trained GMM

```python
from gmm import GradientGMM

gmm = GradientGMM.load('./checkpoints/gmm/gmm.pt', device='cuda')

# Hard assignment
labels = gmm.assign(features)  # [N]

# Soft assignment (posteriors)
posteriors = gmm.soft_assign(features)  # [N, K]
```

## Metrics

The following metrics are logged to `metrics.tsv`:

| Metric | Description |
|--------|-------------|
| `alive` | Number of active clusters |
| `balance_cv` | Coefficient of variation of cluster sizes |
| `weight_entropy` | Normalized entropy of mixture weights |
| `nll` | Negative log-likelihood |
| `posterior_entropy` | Average entropy of posteriors |
| `mean_delta` | Mean parameter change |
| `var_delta` | Variance parameter change |
| `inter_cluster_dist` | Average distance between centroids |
| `intra_cluster_dist` | Average distance within clusters |

## Next Step

After fitting, run JEPA training:

```bash
python train/jepa/train.py \
  --data_dir /path/to/jsonl \
  --gmm_path ./checkpoints/gmm/gmm.pt \
  --output_dir ./checkpoints/jepa
```

## File Structure

```
train/gmm/
├── README.md       # This file
├── __init__.py     # Package exports
├── fit_gmm.py      # Main training script
├── gmm.py          # GradientGMM class
└── utils.py        # Distributed utilities
```
```
