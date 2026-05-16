"""
scripts/sampling.py

Generate images from a trained JiT-RCDM checkpoint.

For each conditioning image supplied via --cond_images, the script:
  1. Loads the image and encodes it with DinoV3 to get h (384-dim CLS token).
  2. Runs the 50-step Heun ODE sampler to generate --n_samples images per h.
  3. Saves a side-by-side grid: [conditioning image | generated samples].
  4. Logs all grids to Weights & Biases (unless --no_wandb is set).

Usage (from project root):
    python scripts/sampling.py \\
        --checkpoint  checkpoints/jit_rcdm_final.pt \\
        --cond_images data/messidor2/test/img1.png data/messidor2/test/img2.png \\
        --out_dir     samples/ \\
        --n_samples   4 \\
        --num_steps   50 \\
        --device      cuda \\
        --wandb_project jit-rcdm

Add --no_wandb to skip W&B logging.

Output:
    samples/sample_img1.png   — grid saved to disk
    samples/sample_img2.png
    W&B run with all grids logged as images
"""

import argparse
import sys
import os
from pathlib import Path

import torch
import torchvision.utils as vutils
from torchvision import transforms
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent.parent))

from rcdm.encoder import load_encoder, build_transform, DINOV3_CHECKPOINT
from rcdm.jit import create_jit_model, FlowMatching

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def load_model(checkpoint_path: str, device: torch.device):
    """
    Restore a JiT model from a training checkpoint.
    Architecture args are read from model_cfg stored inside the checkpoint.

    [fix-1] If the checkpoint contains an EMA shadow dict, apply EMA weights
    instead of the raw model weights — EMA consistently produces better
    sample quality (JiT paper Tab. 9).
    """
    state = torch.load(checkpoint_path, map_location=device)
    cfg   = state["model_cfg"]
    model = create_jit_model(
        image_size=cfg["image_size"],
        patch_size=cfg.get("patch_size", 16),
        hidden_dim=cfg["hidden_dim"],
        depth=cfg["depth"],
        num_heads=cfg["num_heads"],
        h_dim=cfg["h_dim"],
        cond_dim=cfg.get("cond_dim"),
    )
    model.load_state_dict(state["model"])

    # ── JiT-RCDM [fix-1]: prefer EMA weights for inference ──
    ema_state = state.get("ema")
    if ema_state is not None:
        shadow = ema_state.get("shadow", {})
        for name, param in model.named_parameters():
            if name in shadow:
                param.data.copy_(shadow[name].to(device))
        print("  [EMA] loaded EMA weights for inference")
    else:
        print("  [EMA] no EMA found in checkpoint — using raw weights")

    model.eval()
    model.to(device)
    return model, cfg, state


def unnorm(x: torch.Tensor) -> torch.Tensor:
    """Diffusion space [-1, 1] → display space [0, 1]."""
    return (x.clamp(-1, 1) + 1.0) / 2.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",   type=str, required=True,
                        help="Path to trained JiT-RCDM .pt checkpoint")
    parser.add_argument("--cond_images",  type=str, nargs="+", required=True,
                        help="Conditioning image paths (one output grid per image)")
    parser.add_argument("--out_dir",      type=str, default="samples/",
                        help="Directory to write output grids")
    parser.add_argument("--n_samples",    type=int, default=4,
                        help="Number of images to generate per conditioning image")
    parser.add_argument("--num_steps",    type=int,   default=50,
                        help="Heun ODE steps (50 recommended)")
    parser.add_argument("--cfg_scale",   type=float, default=3.0,
                        help="Classifier-free guidance scale. "
                             "1.0 = no guidance (conditional only). "
                             "3.0 = recommended for retinal detail. "
                             "Requires model trained with --cfg_dropout > 0.")
    parser.add_argument("--device",       type=str,   default="cpu")
    parser.add_argument("--encoder_ckpt", type=str, default=DINOV3_CHECKPOINT,
                        help="Path to the local DinoV3 checkpoint directory")

    # Weights & Biases
    parser.add_argument("--no_wandb",       action="store_true",
                        help="Disable W&B logging")
    parser.add_argument("--wandb_project",  type=str, default="jit-rcdm",
                        help="W&B project to log samples into")
    parser.add_argument("--wandb_entity",   type=str, default=None,
                        help="W&B entity (team or username)")
    parser.add_argument("--wandb_run_name", type=str, default="sampling",
                        help="W&B run display name")
    parser.add_argument("--wandb_run_id",   type=str, default=None,
                        help="Link to an existing W&B run (e.g. the training run)")

    args = parser.parse_args()

    use_wandb = (not args.no_wandb) and WANDB_AVAILABLE
    if not args.no_wandb and not WANDB_AVAILABLE:
        print("Warning: wandb not installed. Continuing without logging.")

    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Load JiT (read model_cfg first for image_size / cond_dim)
    # ------------------------------------------------------------------ #
    print(f"Loading JiT-RCDM from {args.checkpoint}...")
    model, cfg, state = load_model(args.checkpoint, device)
    flow       = FlowMatching()
    image_size = cfg["image_size"]
    train_step = state.get("step", "unknown")

    print(f"  image_size={image_size}, "
          f"hidden_dim={cfg['hidden_dim']}, "
          f"cond_dim={cfg.get('cond_dim')}, "
          f"h_dim={cfg['h_dim']}, "
          f"trained_steps={train_step}")

    print("Loading DinoV3 encoder...")
    encoder = load_encoder(device=device, checkpoint_path=args.encoder_ckpt)
    # ── JiT-RCDM [fix-2]: always encode at 224 px — DINOv3 ViT-S/16 has fixed
    # pos-embeds for a 14×14 patch grid (224/16). Using the generative model's
    # image_size here would silently feed DINOv3 a wrong-resolution input.
    enc_transform = build_transform(image_size=224)

    # Diffusion normalisation for the conditioning thumbnail in the output grid
    diffusion_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    # ------------------------------------------------------------------ #
    # W&B initialisation
    # ------------------------------------------------------------------ #
    if use_wandb:
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            id=args.wandb_run_id,
            resume="allow",
            config={
                "checkpoint":    args.checkpoint,
                "n_samples":     args.n_samples,
                "num_steps":     args.num_steps,
                "cfg_scale":     args.cfg_scale,
                "trained_steps": train_step,
                **{f"model/{k}": v for k, v in cfg.items()},
            },
        )
        print(f"  W&B run: {run.url}")

    # ------------------------------------------------------------------ #
    # Generate samples for each conditioning image
    # ------------------------------------------------------------------ #
    all_wandb_images = []

    for img_path in args.cond_images:
        stem = Path(img_path).stem
        print(f"\nConditioning on {img_path} → generating {args.n_samples} samples...")

        # Encode conditioning image → h (384-dim CLS token)
        img_pil = Image.open(img_path).convert("RGB")
        x_enc   = enc_transform(img_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            out = encoder(pixel_values=x_enc)
        h_single = out.last_hidden_state[:, 0, :]               # (1, 384)
        h        = h_single.expand(args.n_samples, -1)          # (n_samples, 384)

        # Run Heun ODE (with CFG when cfg_scale > 1.0)
        noise = torch.randn(args.n_samples, 3, image_size, image_size, device=device)
        with torch.no_grad():
            generated = flow.sample(
                model, noise, h=h,
                num_steps=args.num_steps,
                cfg_scale=args.cfg_scale,
            )

        # Build output grid: [cond_thumb | gen_1 | gen_2 | ...]
        cond_thumb = diffusion_transform(img_pil).unsqueeze(0).to(device)
        all_images = torch.cat([cond_thumb, generated], dim=0)   # (1+n, 3, H, W)
        grid = vutils.make_grid(
            unnorm(all_images),
            nrow=args.n_samples + 1,
            padding=4,
            normalize=False,
        )

        # Save to disk
        out_path = Path(args.out_dir) / f"sample_{stem}.png"
        vutils.save_image(grid, out_path)
        print(f"  saved → {out_path}")

        # Collect for W&B
        if use_wandb:
            all_wandb_images.append(
                wandb.Image(
                    grid.cpu(),
                    caption=(
                        f"{stem} | "
                        f"left: conditioning  right: {args.n_samples} generated  "
                        f"({args.num_steps}-step Heun, cfg={args.cfg_scale}, "
                        f"trained_steps={train_step})"
                    ),
                )
            )

    # Log all grids in a single W&B call so they appear in the same panel
    if use_wandb and all_wandb_images:
        wandb.log({"samples": all_wandb_images})
        wandb.finish()
        print(f"\nAll grids logged to W&B: {run.url}")

    print("\nDone.")


if __name__ == "__main__":
    main()
