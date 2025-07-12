#!/usr/bin/env python3
"""
evaluate_error.py

Usage:
    python evaluate_error.py \
        --checkpoint path/to/checkpoint.pth \
        --template TemplateMNI152NLin2009cSym.nii.gz \
        [--device cuda] \
        [--return_weights]

Example:
    python evaluate_error.py \
        --checkpoint ./checkpoints/best_model.pth \
        --template /home/so_salar/Desktop/Templates/TemplateMNI152NLin2009cSym.nii.gz \
        --device cuda \
        --return_weights
"""

import argparse
from pathlib import Path

import torch

# assumes validate.py defines `validate`
from validate import validate
# from utills.py
from utills import ConvNetCoM
# from Main.py
from Main import load_fixed_image

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate model mean residual error on LandmarkTests")
    p.add_argument(
        "--checkpoint", "-c", required=True,
        help="Path to model checkpoint (.pt or .pth) containing either state_dict or a dict with 'model_state_dict'"
    )
    p.add_argument(
        "--template", "-t", default="TemplateMNI152NLin2009cSym.nii.gz",
        help="Path to fixed/template image (NIfTI)"
    )
    p.add_argument(
        "--device", "-d", default=None,
        help="Torch device (e.g. 'cuda' or 'cpu'). If omitted, auto‐selects cuda if available."
    )
    p.add_argument(
        "--return_weights", "-r", action="store_true",
        help="If set, calls network(input)[0] instead of network(input)"
    )
    return p.parse_args()

def main():
    args = parse_args()

    # decide device
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[INFO] Using device: {device}")

    # instantiate network
    network = ConvNetCoM(dim=3, input_ch=1, out_dim=32, norm_type="instance").to(device)

    # load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    else:
        state = ckpt
    network.load_state_dict(state)
    print(f"[INFO] Loaded checkpoint from {args.checkpoint}")

    # load fixed/template image
    tpl_path = Path(args.template)
    fixed_tensor, fixed_affine = load_fixed_image(tpl_path, device, use_intensity=True)
    print(f"[INFO] Loaded fixed image: {tpl_path}")

    # run validation
    print("[INFO] Running validate() ...")
    mean_error = validate(network, fixed_tensor, fixed_affine, device, args.return_weights)

    print(f"\n=== Mean Residual Error: {mean_error:.4f} ===")

if __name__ == "__main__":
    main()
