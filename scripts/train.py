"""
scripts/train.py

Train JiT-RCDM on Messidor-2 fundus images using precomputed DinoV3
representations.

What changed from the original RCDM train.py
---------------------------------------------
  - Model      : UNetModel → JiT (plain ViT with adaLN-Zero)
  - Encoder h  : 2048-dim ResNet-50 avgpool → 384-dim DinoV3 CLS token
  - Objective  : ε-prediction + DDPM loss → x-prediction + flow-matching MSE
  - Timesteps  : discrete t ~ Uniform[0,T], 1000 steps → continuous t ~
                 logit-normal(0,1), no schedule sampler needed
  - Sampler    : p_sample_loop (1000 DDPM steps) → 50-step Heun ODE
  - image_size : 64 (Tiny ImageNet) → 224 (Messidor-2 / DinoV3 canonical)
  - Monitoring : wandb — loss, grad norm, LR, sample image grids

Usage (from project root):
    python scripts/train.py \\
        --reps_file   data/messidor2/train_reps.pt \\
        --save_dir    checkpoints/ \\
        --image_size  224 \\
        --batch_size  8 \\
        --lr          1e-4 \\
        --total_steps 100000 \\
        --device      cuda \\
        --wandb_project jit-rcdm

Add --no_wandb to run without Weights & Biases.
"""

import argparse
import sys
import os
from pathlib import Path

import torch
import torchvision.utils as vutils
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parent.parent))

from rcdm.jit import create_jit_model, JiT_S_16, JiT_S_32, FlowMatching
from rcdm.dataset import RepresentationDataset

# ── JiT-RCDM [fix-1]: EMA (Exponential Moving Average of model weights) ──
# JiT paper (Tab. 9) shows EMA at decay=0.9999 gives the best FID.
# Pattern adapted from https://github.com/LTH14/JiT/blob/main/denoiser.py
class EMA:
    """
    Maintains a shadow copy of model parameters updated as:
        shadow = decay * shadow + (1 - decay) * param
    Apply shadow weights to model for evaluation / sampling, then restore.
    """
    def __init__(self, model: torch.nn.Module, decay: float = 0.9999):
        self.decay  = decay
        self.shadow: dict = {}
        self.backup: dict = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.data.clone().float()

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for name, p in model.named_parameters():
            if p.requires_grad and name in self.shadow:
                self.shadow[name] = (
                    self.decay * self.shadow[name]
                    + (1.0 - self.decay) * p.data.float()
                )

    def apply_shadow(self, model: torch.nn.Module) -> None:
        """Temporarily swap in EMA weights (undo with restore())."""
        for name, p in model.named_parameters():
            if p.requires_grad and name in self.shadow:
                self.backup[name] = p.data
                p.data = self.shadow[name].to(p.data.dtype)

    def restore(self, model: torch.nn.Module) -> None:
        """Restore original weights after apply_shadow()."""
        for name, p in model.named_parameters():
            if name in self.backup:
                p.data = self.backup[name]
        self.backup.clear()

    def state_dict(self) -> dict:
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, sd: dict) -> None:
        self.decay  = sd["decay"]
        self.shadow = sd["shadow"]

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# ── JiT-RCDM [fix-4c]: linear LR warmup ──
def _lr_at_step(step: int, warmup_steps: int, base_lr: float) -> float:
    """Linear warmup then constant. JiT uses 5 epochs of warmup."""
    if warmup_steps <= 0 or step >= warmup_steps:
        return base_lr
    return base_lr * (step + 1) / warmup_steps


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def unnorm(x: torch.Tensor) -> torch.Tensor:
    """Diffusion space [-1, 1] → display space [0, 1]."""
    return (x.clamp(-1, 1) + 1.0) / 2.0


def make_sample_grid(
    model: torch.nn.Module,
    flow:  FlowMatching,
    probe_x: torch.Tensor,
    probe_h: torch.Tensor,
    device:  torch.device,
    num_steps: int = 20,
) -> torch.Tensor:
    """
    Generate one sample per probe conditioning vector and return a
    comparison grid.

    Layout (2 rows, N columns):
        row 0 — conditioning images from the dataset (ground-truth)
        row 1 — images generated conditioned on those same h vectors

    Using 20 Heun steps instead of 50 keeps visualization fast during
    training; full quality requires 50 steps at inference time.
    """
    model.eval()
    with torch.no_grad():
        noise = torch.randn_like(probe_x)
        generated = flow.sample(model, noise, h=probe_h, num_steps=num_steps)
    model.train()

    comparison = torch.cat([probe_x, generated], dim=0)   # (2N, 3, H, W)
    grid = vutils.make_grid(
        unnorm(comparison),
        nrow=probe_x.shape[0],   # N columns → 2-row grid
        padding=2,
        normalize=False,
    )
    return grid   # (3, H', W') in [0, 1]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _model_cfg(model, args) -> dict:
    """Read architecture dims from the live model so presets are recorded correctly."""
    return {
        "image_size": model.image_size,
        "patch_size": model.patch_size,
        "hidden_dim": model.hidden_dim,
        "depth":      len(model.blocks),
        "num_heads":  model.blocks[0].attn.num_heads,
        "h_dim":      args.h_dim,
        "cond_dim":   model.cond_dim,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    # Data / training
    parser.add_argument("--reps_file",     type=str,   default="data/messidor2/train_reps.pt")
    parser.add_argument("--save_dir",      type=str,   default="checkpoints/")
    parser.add_argument("--image_size",    type=int,   default=224)
    parser.add_argument("--batch_size",    type=int,   default=8)
    parser.add_argument("--lr",            type=float, default=1e-4)
    parser.add_argument("--total_steps",   type=int,   default=100000)
    parser.add_argument("--save_interval", type=int,   default=5000)
    parser.add_argument("--log_interval",  type=int,   default=100)
    parser.add_argument("--device",        type=str,   default="cpu")
    parser.add_argument("--num_workers",   type=int,   default=0,
                        help="DataLoader workers (0=main process; use 4 on GPU)")
    parser.add_argument("--resume",        type=str,   default=None,
                        help="Path to checkpoint to resume from")
    # ── JiT-RCDM [fix-4c]: LR warmup ──
    parser.add_argument("--warmup_steps",  type=int,   default=1000,
                        help="Linear LR warmup steps (JiT uses ~5 epochs). "
                             "Set 0 to disable.")
    # ── JiT-RCDM [fix-4d]: gradient accumulation ──
    parser.add_argument("--grad_accum",    type=int,   default=1,
                        help="Gradient accumulation steps. Effective batch = "
                             "batch_size × grad_accum. Use to simulate larger "
                             "batches on memory-constrained hardware.")
    # ── JiT-RCDM [fix-1]: EMA decay ──
    parser.add_argument("--ema_decay",     type=float, default=0.9999,
                        help="EMA decay rate (JiT paper Tab. 9: 0.9999 best).")

    # JiT architecture
    parser.add_argument("--model",         type=str,   default=None,
                        help="Preset model variant: 'S16' or 'S32'. "
                             "Overrides --hidden_dim/--num_heads/--patch_size/--cond_dim. "
                             "S16 ≈ 25 M params, 196 tokens. "
                             "S32 ≈ 25 M params, 49 tokens (faster on MPS/CPU).")
    parser.add_argument("--hidden_dim",    type=int,   default=768,
                        help="ViT hidden dimension (768=JiT-B, 1024=JiT-L). "
                             "Ignored when --model is set.")
    parser.add_argument("--depth",         type=int,   default=12,
                        help="Number of JiT transformer blocks")
    parser.add_argument("--num_heads",     type=int,   default=12,
                        help="Attention heads (hidden_dim // num_heads = head_dim). "
                             "Ignored when --model is set.")
    parser.add_argument("--patch_size",    type=int,   default=16,
                        help="Patch size in pixels (image_size % patch_size == 0). "
                             "Ignored when --model is set.")
    parser.add_argument("--h_dim",         type=int,   default=384,
                        help="DinoV3 CLS token dimension (384 for ViT-S/16)")
    parser.add_argument("--cond_dim",      type=int,   default=None,
                        help="Conditioning bottleneck width (default = hidden_dim, "
                             "i.e. no bottleneck). Preset S16/S32 use 64. "
                             "Ignored when --model is set.")

    # Weights & Biases
    parser.add_argument("--no_wandb",          action="store_true",
                        help="Disable Weights & Biases logging")
    parser.add_argument("--wandb_project",     type=str, default="jit-rcdm",
                        help="W&B project name")
    parser.add_argument("--wandb_entity",      type=str, default=None,
                        help="W&B entity (team or username); None = your default")
    parser.add_argument("--wandb_run_name",    type=str, default=None,
                        help="W&B run display name; None = auto-generated")
    parser.add_argument("--wandb_run_id",      type=str, default=None,
                        help="Resume an existing W&B run by its ID")
    parser.add_argument("--sample_interval",   type=int,   default=2000,
                        help="Log a sample image grid every N steps")
    parser.add_argument("--n_sample_images",   type=int,   default=4,
                        help="Number of conditioning images shown in sample grids")
    parser.add_argument("--cfg_dropout",       type=float, default=0.1,
                        help="Probability of replacing h with zeros during training "
                             "(null-h CFG dropout). Set to 0.0 to disable CFG entirely. "
                             "A model trained with 0.0 cannot use cfg_scale > 1 at inference.")

    args = parser.parse_args()

    use_wandb = (not args.no_wandb) and WANDB_AVAILABLE
    if not args.no_wandb and not WANDB_AVAILABLE:
        print("Warning: wandb not installed. Run `pip install wandb`. Continuing without logging.")

    device = torch.device(args.device)
    os.makedirs(args.save_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 1 — Build JiT model
    # ------------------------------------------------------------------ #
    PRESETS = {"S16": JiT_S_16, "S32": JiT_S_32}
    print("\n[1/4] Building JiT model...")
    if args.model:
        key = args.model.upper()
        if key not in PRESETS:
            raise ValueError(f"--model must be one of {list(PRESETS)}; got '{args.model}'")
        print(f"  Using preset JiT_{key}")
        model = PRESETS[key](image_size=args.image_size, h_dim=args.h_dim)
    else:
        model = create_jit_model(
            image_size=args.image_size,
            patch_size=args.patch_size,
            hidden_dim=args.hidden_dim,
            depth=args.depth,
            num_heads=args.num_heads,
            h_dim=args.h_dim,
            cond_dim=args.cond_dim,
        )
    model.to(device)

    flow = FlowMatching()

    # Restore checkpoint (read wandb run_id for seamless run resumption)
    wandb_run_id = args.wandb_run_id
    if args.resume:
        print(f"  Resuming from {args.resume}")
        state      = torch.load(args.resume, map_location=device)
        model.load_state_dict(state["model"])
        start_step = state["step"]
        wandb_run_id = wandb_run_id or state.get("wandb_run_id")
        print(f"  Resuming from step {start_step}")
    else:
        start_step = 0

    total_params = sum(p.numel() for p in model.parameters())
    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters  : {total_params / 1e6:.1f}M")
    print(f"  Trainable         : {trainable / 1e6:.1f}M")

    # ------------------------------------------------------------------ #
    # 2 — Dataset and dataloader
    # ------------------------------------------------------------------ #
    print("\n[2/4] Loading dataset...")
    dataset = RepresentationDataset(
        reps_file=args.reps_file,
        image_size=args.image_size,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(args.device == "cuda"),
        drop_last=True,
    )
    print(f"  {len(dataset)} samples, "
          f"{len(dataloader)} batches/epoch at batch_size={args.batch_size}")

    # Fixed probe batch for sample-grid visualisation.
    # NOTE: shuffle=True means these ARE training images — the visualisation
    # shows how well the model reconstructs its own training data, which is
    # a valid proxy for convergence even if not a held-out set.
    n_probe = min(args.n_sample_images, args.batch_size)
    probe_x, probe_h = next(iter(dataloader))
    probe_x = probe_x[:n_probe].to(device)
    probe_h = probe_h[:n_probe].to(device)

    # ------------------------------------------------------------------ #
    # 3 — Optimiser + EMA
    # ------------------------------------------------------------------ #
    print("\n[3/4] Setting up optimiser...")
    # ── JiT-RCDM [fix-4a]: betas=(0.9, 0.95) per JiT paper (vs PyTorch default (0.9, 0.999)) ──
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.0,
        betas=(0.9, 0.95),
    )

    if args.resume:
        optimiser.load_state_dict(state["optimiser"])

    # ── JiT-RCDM [fix-1]: build EMA after model and optimiser are ready ──
    ema = EMA(model, decay=args.ema_decay)
    if args.resume and "ema" in state:
        ema.load_state_dict(state["ema"])
        print(f"  EMA restored (decay={ema.decay})")
    else:
        print(f"  EMA initialised (decay={args.ema_decay})")

    # ------------------------------------------------------------------ #
    # W&B initialisation
    #
    # wandb.init is called after the model and dataset are ready so that
    # the config logged to the run includes the parameter count.
    # Resuming uses the run_id stored in the checkpoint so that all metrics
    # appear on the same W&B run timeline.
    # ------------------------------------------------------------------ #
    if use_wandb:
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            id=wandb_run_id,
            resume="allow",
            config={
                # Training
                "reps_file":      args.reps_file,
                "batch_size":     args.batch_size,
                "effective_batch": args.batch_size * args.grad_accum,
                "lr":             args.lr,
                "total_steps":    args.total_steps,
                "warmup_steps":   args.warmup_steps,
                "grad_accum":     args.grad_accum,
                "ema_decay":      args.ema_decay,
                "cfg_dropout":    args.cfg_dropout,
                # Architecture (read from model so presets are correct)
                **{f"model/{k}": v for k, v in _model_cfg(model, args).items()},
                # Derived
                "total_params_M": round(total_params / 1e6, 1),
                "n_train_images": len(dataset),
            },
        )
        # Track gradients and parameter histograms every log_interval steps
        wandb.watch(model, log="gradients", log_freq=args.log_interval)
        print(f"  W&B run: {run.url}")

    # ------------------------------------------------------------------ #
    # 4 — Training loop
    # ------------------------------------------------------------------ #
    print("\n[4/4] Starting training...\n")
    model.train()

    step        = start_step
    data_iter   = iter(dataloader)
    accum_loss  = 0.0
    accum_gnorm = 0.0

    # ── JiT-RCDM [fix-4d]: gradient accumulation ──
    # We zero gradients once per optimizer step (every grad_accum micro-steps).
    optimiser.zero_grad()

    while step < args.total_steps:

        # ── inner micro-step loop for gradient accumulation ──
        for _micro in range(args.grad_accum):
            try:
                x, h = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                x, h = next(data_iter)

            x = x.to(device)   # (B, 3, H, W)  images in [-1, 1]
            h = h.to(device)   # (B, 384)       DinoV3 CLS representations

            # Forward — flow-matching x-prediction loss with CFG null-h dropout.
            # Divide by grad_accum so the accumulated gradient equals the mean
            # loss over all micro-batches (same as a single large batch).
            loss = flow.training_loss(model=model, x=x, h=h,
                                      p_uncond=args.cfg_dropout)
            (loss / args.grad_accum).backward()

        # ── JiT-RCDM [fix-4c]: linear LR warmup ──
        new_lr = _lr_at_step(step, args.warmup_steps, args.lr)
        for g in optimiser.param_groups:
            g["lr"] = new_lr

        # clip_grad_norm_ returns the total norm *before* clipping.
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimiser.step()
        optimiser.zero_grad()

        # ── JiT-RCDM [fix-1]: EMA update after every optimizer step ──
        ema.update(model)

        accum_loss  += loss.item()
        accum_gnorm += grad_norm.item()
        step += 1

        # ---- scalar logging ------------------------------------------ #
        if step % args.log_interval == 0:
            avg_loss  = accum_loss  / args.log_interval
            avg_gnorm = accum_gnorm / args.log_interval
            current_lr = optimiser.param_groups[0]["lr"]

            print(f"step {step:>7d}/{args.total_steps} "
                  f"| loss {avg_loss:.4f} "
                  f"| grad_norm {avg_gnorm:.3f} "
                  f"| lr {current_lr:.2e}")

            if use_wandb:
                wandb.log({
                    "train/loss":      avg_loss,
                    "train/grad_norm": avg_gnorm,
                    "train/lr":        current_lr,
                    "step":            step,
                })

            accum_loss  = 0.0
            accum_gnorm = 0.0

        # ---- sample image grid (use EMA weights) --------------------- #
        if use_wandb and step % args.sample_interval == 0:
            # ── JiT-RCDM [fix-1]: sample with EMA model ──
            ema.apply_shadow(model)
            grid = make_sample_grid(model, flow, probe_x, probe_h, device)
            ema.restore(model)
            wandb.log({
                "samples/grid": wandb.Image(
                    grid.cpu(),
                    caption=(
                        f"step {step} | "
                        "top: conditioning images  "
                        "bottom: EMA-generated (20-step Heun)"
                    ),
                ),
                "step": step,
            })

        # ---- checkpoint ---------------------------------------------- #
        if step % args.save_interval == 0:
            ckpt_path = Path(args.save_dir) / f"jit_rcdm_step{step:07d}.pt"
            model_cfg = _model_cfg(model, args)
            torch.save(
                {
                    "step":         step,
                    "model":        model.state_dict(),
                    "optimiser":    optimiser.state_dict(),
                    "model_cfg":    model_cfg,
                    # ── JiT-RCDM [fix-1]: persist EMA shadow weights ──
                    "ema":          ema.state_dict(),
                    "wandb_run_id": wandb.run.id if use_wandb else None,
                },
                ckpt_path,
            )
            print(f"  → checkpoint: {ckpt_path}")

            if use_wandb:
                wandb.save(str(ckpt_path), base_path=str(Path(args.save_dir)))

    # ---- final save ---------------------------------------------------- #
    final_path = Path(args.save_dir) / "jit_rcdm_final.pt"
    torch.save(
        {
            "step":         step,
            "model":        model.state_dict(),
            "optimiser":    optimiser.state_dict(),
            "model_cfg":    _model_cfg(model, args),
            # ── JiT-RCDM [fix-1]: persist EMA shadow weights ──
            "ema":          ema.state_dict(),
            "wandb_run_id": wandb.run.id if use_wandb else None,
        },
        final_path,
    )
    print(f"\nTraining complete. Final model saved to {final_path}")

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
