# Phase 2b: JEPA Training (Transformer Architecture)

WavLM-style Transformer encoder for JEPA training.

## Architecture

- **CNN Frontend**: 7-layer conv (stride 320 = 20ms @ 16kHz)
- **Transformer**: Standard pre-norm with gated relative position bias
- **Fixed hop**: 320 samples

## Usage
```bash
deepspeed train.py \
  --data_dir /path/to/jsonl \
  --output_dir ./checkpoints/jepa_transformer \
  --gmm_path ./checkpoints/gmm/gmm.pt \
  --ds_config ../configs/ds_config.json \
  --code_dim 768 \
  --conv_dim 512 \
  --num_heads 12 \
  --ff_dim 3072 \
  --num_layers 12 \
  --max_steps 50000
```

## Model Sizes

| Config | conv_dim | code_dim | layers | heads | ff_dim | Params |
|--------|----------|----------|--------|-------|--------|--------|
| Small  | 256      | 512      | 10     | 8     | 2048   | ~30M   |
| Base   | 512      | 768      | 12     | 12    | 3072   | ~95M   |
| Large  | 512      | 1024     | 24     | 16    | 4096   | ~315M  |

## Differences from Conformer Version

| Feature | Conformer | Transformer |
|---------|-----------|-------------|
| Encoder | Conv + ResBlocks + DAAM | 7-layer CNN |
| Backbone | Conformer (FFN-Attn-Conv-FFN) | Standard Transformer |
| Hop | Configurable (product of strides) | Fixed 320 |
| Layer Agg | Cross-attention | None (final layer) |
| Predictor | Conformer-based | 2 conv layers |
