# Phase 2c: Pure JEPA Training

JEPA training without cluster supervision - just masked prediction with EMA target.

## Overview

This is a baseline/ablation that trains JEPA without any GMM cluster loss:
- Online encoder predicts masked representations
- Target encoder (EMA) provides targets
- No external supervision signal

## Usage
```bash
deepspeed train.py \
  --data_dir /path/to/jsonl \
  --output_dir ./checkpoints/jepa_pure \
  --ds_config ../configs/ds_config.json \
  --use_rel_pos \
  --use_gaatn \
  --max_steps 50000
```

## Key Differences from GMM-supervised JEPA

| Feature | JEPA + GMM | Pure JEPA |
|---------|------------|-----------|
| Cluster loss | KL(GMM posteriors) | None |
| GMM required | Yes | No |
| Collapse risk | Low (GMM anchors) | Higher |
| Loss schedule | Decaying cluster weight | N/A |

## Arguments

Same as `train/jepa/` except:
- No `--gmm_path` / `--gmm_paths`
- No `--cluster_weight_*`
- No `--cluster_temperature`
- No `--K`
