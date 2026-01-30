#!/usr/bin/env python
# ds_ckpt_to_pt.py

import os
import glob
import argparse
import torch

from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint


def load_from_zero(checkpoint_dir: str, tag: str | None):
    """
    Try to merge ZeRO shards into a single FP32 state_dict using DeepSpeed helper.
    """
    sd = get_fp32_state_dict_from_zero_checkpoint(
        checkpoint_dir=checkpoint_dir,
        tag=tag,   # e.g. 'final', 'step100000'; None => latest
    )
    if not isinstance(sd, dict) or len(sd) == 0:
        raise RuntimeError(
            f"Empty state_dict from ZeRO checkpoint in {checkpoint_dir} (tag={tag})"
        )
    return sd


def maybe_load_from_consolidated_files(checkpoint_dir: str):
    """
    Fallback: look for any existing consolidated checkpoint (.bin or .pt)
    and load it directly with torch.load.

    Returns:
        state_dict (dict) or None if nothing usable is found.
    """
    cand: list[str] = []
    cand += glob.glob(
        os.path.join(checkpoint_dir, "**", "pytorch_model.bin"), recursive=True
    )
    cand += glob.glob(
        os.path.join(checkpoint_dir, "**", "*.pt"), recursive=True
    )

    if not cand:
        return None

    # Pick the largest file as best guess for "full" checkpoint
    cand.sort(key=lambda p: os.path.getsize(p), reverse=True)
    ckpt_path = cand[0]
    print(f"[info] fallback: loading {ckpt_path!r}")

    try:
        obj = torch.load(ckpt_path, map_location="cpu")
    except Exception as e:
        print(f"[warn] torch.load failed on {ckpt_path!r}: {e}")
        return None

    # If obj is already a state_dict-like mapping of tensors, return as-is.
    if isinstance(obj, dict):
        # Many training setups wrap state_dicts under 'module' or 'model_state_dict'
        if "module" in obj and isinstance(obj["module"], dict):
            print("[info] unwrapping 'module' subkey")
            return obj["module"]
        if "model_state_dict" in obj and isinstance(obj["model_state_dict"], dict):
            print("[info] unwrapping 'model_state_dict' subkey")
            return obj["model_state_dict"]

        # Otherwise assume it's already a usable state_dict
        return obj

    # For weird formats (e.g. plain nn.Module), try getattr .state_dict
    if hasattr(obj, "state_dict"):
        print("[info] object has state_dict(), extracting")
        return obj.state_dict()

    print("[warn] fallback object is not a dict and has no state_dict()")
    return None


def main():
    ap = argparse.ArgumentParser(
        description="Convert a DeepSpeed checkpoint directory (ZeRO or consolidated) "
                    "to a single .pt state_dict."
    )
    ap.add_argument(
        "--ds_dir",
        required=True,
        help="DeepSpeed checkpoint dir (e.g. ./jepa_encoder_ds or ./decoder_ds)",
    )
    ap.add_argument(
        "--out_pt",
        required=True,
        help="Output .pt path for the merged state_dict",
    )
    ap.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Checkpoint tag for ZeRO shards (e.g. 'final', 'step100000'); "
             "None => latest",
    )
    args = ap.parse_args()

    out_dir = os.path.dirname(os.path.abspath(args.out_pt))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # 1) Try ZeRO merge path (preferred)
    try:
        sd = load_from_zero(args.ds_dir, args.tag)
        print(f"[info] loaded FP32 state_dict from ZeRO shards: {len(sd)} keys")
    except Exception as e:
        print(f"[warn] zero_to_fp32 merge failed: {e}")
        # 2) Fallback to any consolidated file we can find
        sd = maybe_load_from_consolidated_files(args.ds_dir)
        if sd is None:
            raise RuntimeError(
                f"Could not read any usable state_dict from {args.ds_dir}"
            )
        print(f"[info] loaded fallback state_dict: {len(sd)} keys")

    # Final save
    torch.save(sd, args.out_pt)
    print(f"[OK] wrote state_dict with {len(sd)} keys to {args.out_pt!r}")


if __name__ == "__main__":
    main()


