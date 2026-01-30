# Phase 1: K-Means Clustering

GPU-accelerated mini-batch K-means on log-mel features.

## Usage

### Single GPU
```bash
python fit_kmeans.py \
  --data_dir /path/to/jsonl \
  --output_dir ./KMeans \
  --K 512 \
  --target_frames 100000000
```

### Multi-GPU
```bash
torchrun --nproc_per_node=8 fit_kmeans.py \
  --data_dir /path/to/jsonl \
  --output_dir ./KMeans \
  --K 512 \
  --target_frames 500000000
```

## Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| --K | 512 | Number of clusters |
| --n_mels | 80 | Mel frequency bins |
| --init_method | kmeans++ | Initialization method |
| --update_method | lloyd | Update: lloyd (batch) or minibatch |
| --target_frames | 100M | Target frames to process |
| --batch_update_size | 5M | Frames per update |

## Output
```
output_dir/
├── kmeans.pt              # Final centroids
├── kmeans_ckpt_XXM.pt     # Checkpoints
└── kmeans_metrics.txt     # Training metrics
```
