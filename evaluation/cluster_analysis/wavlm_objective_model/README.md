# WavLM Evaluation

Evaluate WavLM-style speech representations trained with K-means pseudo-labels.

## Files

- `model_wavlm_objective.py` - WavLM encoder (Conformer + linear head)
- `evaluate_wavlm.py` - Evaluation script with metrics and visualizations

## Usage

```bash
python evaluate_wavlm.py \
    --checkpoint ./WavLM/model.pt \
    --data "/path/to/jsonl" \
    --output_dir ./eval_results
```

## Model Config

Default config matches `train_wavlm_fair.py`:

```
--code_dim 512
--channels 32,64,128,256
--strides 8,8,5
--n_res 4
--n_conformer 4
--heads 32
--K 1024
--use_gaatn
--use_rel_pos
```

## Outputs

```
output_dir/
├── results.json      # Metrics summary
├── labels.pt         # Cluster assignments
├── umap.png          # UMAP visualization
├── distribution.png  # Cluster frequency (Zipf)
├── transitions.png   # Self/cross transitions
├── spectrograms.png  # Mel specs by cluster
└── utterance_*.png   # Per-utterance timelines
```

## Metrics

- **Entropy**: Cluster utilization (higher = more uniform)
- **Adjacent consistency**: P(c_t = c_{t+1}) temporal smoothness
- **Used clusters**: Number of active clusters

## Comparing Models

```python
from evaluate_wavlm import plot_distribution_overlay
import torch

results = {
    'GMM-JEPA': torch.load('gmm_jepa/labels.pt'),
    'WavLM-style': torch.load('wavlm/labels.pt'),
}

plot_distribution_overlay(results, k=1024, path='comparison.png')
```
