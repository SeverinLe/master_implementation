"""
precompute_reps.py

Run the frozen SSL encoder over every training image in Tiny ImageNet
and save the representations to disk.

Usage:
    python scripts/precompute_reps.py \
        --data_dir  data/tiny-imagenet-200/train \
        --out_file  data/tiny-imagenet-200/train_reps.pt \
        --image_size 64 \
        --batch_size 128 \
        --device cpu

Output:
    A .pt file containing a dict:
    {
        "paths" : list of str         — relative image paths, length N
        "reps"  : torch.Tensor        — shape (N, 2048), float32
    }

    The index alignment is exact:
        reps[i]  is the representation of  paths[i]
    This pairing is what the training dataloader uses.
"""

import argparse
import sys
import os
from pathlib import Path

import torch

# Make sure we can import from project root
sys.path.append(str(Path(__file__).resolve().parent.parent))

from rcdm.encoder import load_encoder, build_transform, encode_batch


def collect_image_paths(data_dir):
    """
    Walk the training directory and collect every image path.

    Tiny ImageNet training structure:
        train/
            n01443537/
                images/
                    n01443537_0.JPEG
                    n01443537_1.JPEG
                    ...
            n01629819/
                images/
                    ...

    Returns a sorted list of absolute path strings.
    We sort so the order is deterministic across runs.
    """
    valid_extensions = {".jpeg", ".jpg", ".png", ".JPEG"}
    paths = []

    data_dir = Path(data_dir)
    for img_path in sorted(data_dir.rglob("*")):
        if img_path.suffix in valid_extensions:
            paths.append(str(img_path))

    print(f"Found {len(paths)} images in {data_dir}")
    return paths


def main():
    parser = argparse.ArgumentParser(description="Precompute SSL representations")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/tiny-imagenet-200/train",
        help="Root of the Tiny ImageNet training split",
    )
    parser.add_argument(
        "--out_file",
        type=str,
        default="data/tiny-imagenet-200/train_reps.pt",
        help="Where to save the output .pt file",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=64,
        help="Resize images to this before encoding (match RCDM training size)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Images per encoder forward pass — reduce if you run out of memory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="cpu or cuda",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    # Step 1 — collect all image paths
    # ------------------------------------------------------------------ #
    print("\n[1/3] Collecting image paths...")
    paths = collect_image_paths(args.data_dir)

    if len(paths) == 0:
        raise RuntimeError(
            f"No images found in {args.data_dir}. "
            "Check that the path points to the Tiny ImageNet train/ folder."
        )

    # ------------------------------------------------------------------ #
    # Step 2 — load encoder and run over all images
    # ------------------------------------------------------------------ #
    print(f"\n[2/3] Loading encoder on {args.device}...")
    encoder  = load_encoder(device=args.device)
    transform = build_transform(image_size=args.image_size)

    print(f"Running encoder over {len(paths)} images "
          f"(batch_size={args.batch_size})...")

    reps = encode_batch(
        image_paths=paths,
        encoder=encoder,
        transform=transform,
        device=args.device,
        batch_size=args.batch_size,
    )

    # reps shape: (N, 2048)
    print(f"\nRepresentations shape : {reps.shape}")
    print(f"Representations dtype : {reps.dtype}")
    print(f"Sample norm (first 5) : {reps[:5].norm(dim=1).tolist()}")

    # ------------------------------------------------------------------ #
    # Step 3 — save to disk
    # ------------------------------------------------------------------ #
    print(f"\n[3/3] Saving to {args.out_file}...")
    os.makedirs(Path(args.out_file).parent, exist_ok=True)

    torch.save(
        {
            "paths": paths,   # list[str], length N
            "reps":  reps,    # Tensor (N, 2048)
        },
        args.out_file,
    )

    # Quick sanity check on the saved file
    loaded = torch.load(args.out_file)
    assert len(loaded["paths"]) == loaded["reps"].shape[0], \
        "Path count and rep count don't match — something went wrong"
    assert loaded["reps"].shape[1] == 2048, \
        f"Expected 2048-dim reps, got {loaded['reps'].shape[1]}"

    print(f"\nDone. Saved {len(paths)} representations to {args.out_file}")
    print(f"File size: {Path(args.out_file).stat().st_size / 1e6:.1f} MB")
    print("\nVerification passed — paths and reps are aligned.")


if __name__ == "__main__":
    main()