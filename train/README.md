# DeepSpeed Checkpoint Converter

Convert DeepSpeed (ZeRO) checkpoint directories into portable `.pt` state dicts.

## File

- `ds_ckpt_to_pt.py` - Generic checkpoint conversion utility

## Usage

```bash
python ds_ckpt_to_pt.py \
    --ds_dir ./JEPA/ckpts \
    --out_pt ./JEPA/model.pt \
    --tag step45000
```

## Arguments

| Argument | Description |
|----------|-------------|
| `--ds_dir` | DeepSpeed checkpoint directory |
| `--out_pt` | Output `.pt` file path |
| `--tag` | Checkpoint tag (e.g., `step45000`, `final`). `None` = latest |

## How It Works

1. **Primary**: Uses DeepSpeed's `get_fp32_state_dict_from_zero_checkpoint()` to merge ZeRO shards
2. **Fallback**: If ZeRO merge fails, searches for consolidated files (`pytorch_model.bin`, `*.pt`)

## Supported Formats

- ZeRO Stage 1/2/3 sharded checkpoints
- Consolidated `pytorch_model.bin`
- Standard `.pt` checkpoints
- Wrapped state dicts (`module`, `model_state_dict` keys)

## Examples

```bash
# Latest checkpoint
python ds_ckpt_to_pt.py \
    --ds_dir ./WavLM/ckpts \
    --out_pt ./WavLM/wavlm_model.pt

# Specific step
python ds_ckpt_to_pt.py \
    --ds_dir ./JEPA/ckpts \
    --out_pt ./JEPA/jepa_step100k.pt \
    --tag step100000

# Final checkpoint
python ds_ckpt_to_pt.py \
    --ds_dir ./model/ckpts \
    --out_pt ./model/final.pt \
    --tag final
```

## Output

Single `.pt` file containing the full FP32 state dict, loadable with:

```python
state_dict = torch.load("model.pt", map_location="cpu")
model.load_state_dict(state_dict)
```
