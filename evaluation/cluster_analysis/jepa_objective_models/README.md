# Cluster Analysis Evaluation

Evaluate speech representations with cluster analysis metrics and visualizations.

## Metrics

- **Entropy**: Cluster utilization (higher = more uniform usage)
- **Adjacent Consistency**: Temporal smoothness (higher = more stable assignments)
- **Used Clusters**: Number of active clusters

## Usage

```bash
python evaluate.py \
    --checkpoint /path/to/model.pt \
    --data /path/to/jsonl \
    --output_dir ./results \
    --use_gaatn \
    --use_rel_pos
```

## Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| --checkpoint | required | Model checkpoint path |
| --data | required | Comma-separated JSONL directories |
| --output_dir | ./eval_results | Output directory |
| --code_dim | 512 | Encoder dimension |
| --channels | 32,64,128,256 | Encoder channels |
| --strides | 8,8,5 | Encoder strides |
| --n_conformer | 4 | Conformer layers |
| --num_classes | 1024 | Number of clusters (K) |
| --use_gaatn | false | Use Gaussian attention |
| --use_rel_pos | true | Use relative position bias |
| --n_utterances | 1000 | Utterances to evaluate |
| --viz_samples | 10000 | Samples for UMAP |

## Outputs

```
output_dir/
├── results.json        # Metrics summary
├── labels.pt           # Cluster assignments
├── umap.png            # UMAP visualization
├── distribution.png    # Cluster frequency
├── transitions.png     # Transition probabilities
├── spectrograms.png    # Spectrograms by cluster
├── layer_weights.png   # Layer attention weights
└── utterance_*.png     # Per-utterance analysis
```

## Comparing Multiple Models

```python
from evaluate import plot_distribution_overlay
import torch

results = {
    'GMM-JEPA': torch.load('gmm_jepa/labels.pt'),
    'Pure JEPA': torch.load('pure_jepa/labels.pt'),
    'WavLM-style': torch.load('wavlm/labels.pt'),
}

plot_distribution_overlay(results, k=1024, path='comparison.png')
```

## Example

```bash
# Evaluate GMM-JEPA
python evaluate.py \
    --checkpoint ./checkpoints/jepa/encoder.pt \
    --data /path/to/librispeech \
    --output_dir ./eval_gmm_jepa \
    --use_gaatn \
    --use_rel_pos

# Evaluate WavLM-style
python evaluate.py \
    --checkpoint ./checkpoints/wavlm/encoder.pt \
    --data /path/to/librispeech \
    --output_dir ./eval_wavlm \
    --use_rel_pos
```
