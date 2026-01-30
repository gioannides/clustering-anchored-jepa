# WavLM Training

WavLM-style masked prediction with K-means pseudo-labels.

## Overview

Trains an encoder using:
1. K-means pseudo-labels from log-mel features
2. Span masking of encoder inputs
3. Cross-entropy loss on masked positions
4. WavLM-style denoising augmentation

## Usage

### 1. Fit K-Means (see train/kmeans/)
```bash
python ../kmeans/fit_kmeans.py \
  --data_dir /path/to/jsonl \
  --output_dir ./KMeans \
  --K 1024
```

### 2. Train WavLM
```bash
deepspeed train.py \
  --data_dir /path/to/jsonl \
  --output_dir ./WavLM \
  --ds_config ../configs/ds_config.json \
  --kmeans_path ./KMeans/kmeans.pt \
  --use_rel_pos \
  --use_gaatn \
  --max_steps 50000
```

## Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| --kmeans_path | required | Path to K-means checkpoint |
| --K | 1024 | Number of classes |
| --code_dim | 512 | Encoder dimension |
| --n_conformer | 4 | Conformer layers |
| --use_gaatn | false | Use Gaussian attention |
| --use_rel_pos | false | Use relative position bias |

## Comparison with JEPA

| Feature | WavLM | JEPA |
|---------|-------|------|
| Target | K-means labels | EMA encoder |
| Loss | Cross-entropy | MSE |
| Supervision | Pseudo-labels | Self-supervised |
| Teacher | Fixed (K-means) | EMA updated |
