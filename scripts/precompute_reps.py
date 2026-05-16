"""
precompute_reps.py

Run the frozen DinoV3 ViT-S/16 encoder over every training image and save
the CLS-token representations to disk.

What changed from the original RCDM:
  - Encoder   : ResNet-50 (2048-dim avgpool) → DinoV3 ViT-S/16 (384-dim CLS token)
  - image_size: 64 (Tiny ImageNet) → 224 (Messidor-2 canonical)
  - data_dir  : data/tiny-imagenet-200/train → data/messidor2/train
  - Rep shape : (N, 2048) → (N, 384)

Usage:
    python scripts/precompute_reps.py \\
        --data_dir  data/messidor2/train \\
        --out_file  data/messidor2/train_reps.pt \\
        --image_size 224 \\
        --batch_size 64 \\
        --device cpu

Output:
    A .pt file containing a dict:
    {
        "paths" : list of str         — image paths, length N
        "reps"  : torch.Tensor        — shape (N, 384), float32
    }

    Index alignment is exact:  reps[i] is the representation of paths[i].
    This pairing is what RepresentationDataset uses during training.
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

    Messidor-2 training structure (flat or nested — any layout works):
        train/
            img001.jpg
            img002.png
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
        default="data/messidor2/train",
        help="Root of the Messidor-2 training split",
    )
    parser.add_argument(
        "--out_file",
        type=str,
        default="data/messidor2/train_reps.pt",
        help="Where to save the output .pt file",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=224,
        help="Resize images before encoding (224 = DinoV3 ViT-S/16 canonical size)",
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
            "Check that the path points to the Messidor-2 train/ folder."
        )

    # ------------------------------------------------------------------ #
    # Step 2 — load encoder and run over all images
    # ------------------------------------------------------------------ #
    print(f"\n[2/3] Loading encoder on {args.device}...")
    encoder  = load_encoder(device=args.device)
    # ── JiT-RCDM: always 224 for DinoV3 ViT-S/16 — fixed pos-embed grid (224/16=14×14) ──
    # args.image_size controls the generative model resolution, not the encoder.
    transform = build_transform(image_size=224)

    print(f"Running encoder over {len(paths)} images "
          f"(batch_size={args.batch_size})...")

    reps = encode_batch(
        image_paths=paths,
        encoder=encoder,
        transform=transform,
        device=args.device,
        batch_size=args.batch_size,
    )

    # reps shape: (N, 384) — DinoV3 ViT-S/16 CLS token
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
            "reps":  reps,    # Tensor (N, 384)
        },
        args.out_file,
    )

    # Quick sanity check on the saved file
    loaded = torch.load(args.out_file)
    assert len(loaded["paths"]) == loaded["reps"].shape[0], \
        "Path count and rep count don't match — something went wrong"
    assert loaded["reps"].shape[1] == 384, \
        f"Expected 384-dim DinoV3 ViT-S reps, got {loaded['reps'].shape[1]}"

    print(f"\nDone. Saved {len(paths)} representations to {args.out_file}")
    print(f"File size: {Path(args.out_file).stat().st_size / 1e6:.1f} MB")
    print("\nVerification passed — paths and reps are aligned.")


if __name__ == "__main__":
    main()