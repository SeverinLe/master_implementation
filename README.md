# JiT-RCDM — Representation-Conditioned Diffusion for Retinal Images

A re-implementation of **RCDM** (Representation-Conditioned Diffusion Model,
Bordes et al. 2022) that replaces the original UNet + DDPM backbone with a
**JiT ViT denoiser + flow matching**, conditioned on **DinoV3** SSL
representations, applied to the **Messidor-2** diabetic retinopathy dataset.

---

## Architecture overview

```
 Messidor-2 image
        │
        ▼
 ┌─────────────────────────────┐
 │  DinoV3 ViT-S/16 (frozen)  │   → CLS token h  (B, 384)
 └─────────────────────────────┘
        │
        ▼  ConditioningProjector
        │  Linear(384 → 64) + SiLU
        │
        h_proj  (B, 64)
        │
        │          sinusoidal(t, 64)
        │         → time_embed MLP
        │         →  t_emb  (B, 64)
        │
        └──── c = h_proj + t_emb   (B, 64)   ←── conditioning bottleneck
                       │
                       ▼  (shared across all blocks)
        ┌──────────────────────────────────────┐
        │  JiT ViT Denoiser                    │
        │                                      │
        │  PatchEmbed  z_t (B,3,224,224)        │
        │    → tokens  (B, 196, 1024)           │
        │                                      │
        │  × 12 JiTBlocks                      │
        │    adaLN-Zero:  c (64) → 6·1024      │
        │    Self-attention + FFN               │
        │                                      │
        │  FinalLayer → unpatchify             │
        │    → x_pred  (B, 3, 224, 224)        │
        └──────────────────────────────────────┘
```

**Training objective (flow matching):**
```
z_t  = t · x  +  (1−t) · ε       linear interpolation, t ~ logit-normal(0,1)
loss = ||JiT(z_t, t, h) − x||²   x-prediction MSE
```

**Sampling (50-step Heun ODE):**
```
start: z₀ ~ N(0, I)
for i = 0 … 49:
    v₁ = (x_pred(zᵢ, tᵢ) − zᵢ) / (1 − tᵢ)
    v₂ = (x_pred(zᵢ + dt·v₁, tᵢ₊₁) − …) / (1 − tᵢ₊₁)
    z_{i+1} = zᵢ + dt · (v₁ + v₂) / 2
end: z₅₀ ≈ clean image conditioned on h
```

---

## Prerequisites

| Requirement | Tested version |
|---|---|
| Python | 3.10 + |
| PyTorch | 2.1 + |
| torchvision | 0.16 + |
| Hugging Face `transformers` | 4.38 + |
| Pillow | 10 + |

---

## Installation

```bash
# 1. Clone / enter the project
cd master_implementation

# 2. Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install core dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers pillow

# 4. Install the guided_diffusion package (editable, for legacy UNet support)
pip install -e guided_diffusion/
```

> The `rcdm/` package does **not** have a `setup.py` — scripts add it to
> `sys.path` automatically. If you import from outside the `scripts/`
> directory, add the project root to `PYTHONPATH`:
> ```bash
> export PYTHONPATH=$PYTHONPATH:/path/to/master_implementation
> ```

---

## Project layout

```
master_implementation/
├── checkpoints/
│   ├── dinov3_vits16_tmp/       ← DinoV3 encoder weights go here
│   └── jit_rcdm_final.pt        ← produced by train.py
│
├── data/
│   └── messidor2/
│       ├── train/               ← training images (any flat or nested layout)
│       ├── test/                ← test images used during sampling
│       └── train_reps.pt        ← produced by precompute_reps.py
│
├── rcdm/
│   ├── encoder.py               ← DinoV3 loader + encode_batch()
│   ├── conditioning.py          ← ConditioningProjector, AdaLNZero, (legacy cBN)
│   ├── jit.py                   ← JiT ViT model + FlowMatching
│   └── dataset.py               ← RepresentationDataset (image, h) pairs
│
├── scripts/
│   ├── precompute_reps.py       ← step 1: encode all training images
│   ├── train.py                 ← step 2: train JiT-RCDM
│   └── sampling.py              ← step 3: generate images
│
├── guided_diffusion/            ← legacy ADM framework (UNet, DDPM — kept for reference)
│
├── CHANGES.md                   ← full architectural change log
└── README.md                    ← this file
```

---

## Step 0 — Place the DinoV3 checkpoint

The encoder is loaded **only from local files** (no internet connection
required). The checkpoint directory must contain HuggingFace model files:

```
checkpoints/dinov3_vits16_tmp/
    config.json
    model.safetensors       (or pytorch_model.bin)
```

If you have the weights as a raw PyTorch `.pth` file, convert them first:

```python
# convert_to_hf.py  (run once)
from transformers import Dinov2Config, Dinov2Model
import torch, os

cfg   = Dinov2Config(hidden_size=384, num_attention_heads=6, num_hidden_layers=12,
                     patch_size=16, image_size=224)
model = Dinov2Model(cfg)

# Load and map your weights
state = torch.load("dinov3_vits16.pth", map_location="cpu")
model.load_state_dict(state, strict=False)

model.save_pretrained("checkpoints/dinov3_vits16_tmp")
print("Saved.")
```

Verify the encoder loads correctly:

```bash
python -c "
from rcdm.encoder import load_encoder, build_transform, encode_image
enc = load_encoder(device='cpu')
print('Encoder loaded. Parameters:', sum(p.numel() for p in enc.parameters()))
"
```

Expected output: `Encoder loaded. Parameters: ~22M` (ViT-S/16).

---

## Step 1 — Prepare Messidor-2 data

Messidor-2 images can be organised in any directory structure `precompute_reps.py`
can walk (flat folder or nested by grade/set). A typical layout:

```
data/messidor2/train/
    20051020_55225_0100_PP.tif
    20051020_55226_0100_PP.tif
    ...
```

Supported extensions: `.jpg`, `.jpeg`, `.png`, `.JPEG`, `.tif`, `.tiff`
(add others to `collect_image_paths()` in `precompute_reps.py` if needed).

---

## Step 2 — Precompute DinoV3 representations

This step runs the frozen encoder over every training image **once** and saves
a `(N, 384)` tensor to disk. Training reads from this file every epoch — the
encoder never runs again during training.

```bash
python scripts/precompute_reps.py \
    --data_dir   data/messidor2/train \
    --out_file   data/messidor2/train_reps.pt \
    --image_size 224 \
    --batch_size 64 \
    --device     cuda
```

| Argument | Default | Description |
|---|---|---|
| `--data_dir` | `data/messidor2/train` | Root directory of training images |
| `--out_file` | `data/messidor2/train_reps.pt` | Output path for the `.pt` file |
| `--image_size` | `224` | Resize images to this size before encoding |
| `--batch_size` | `64` | Images per encoder forward pass |
| `--device` | `cpu` | `cpu` or `cuda` |

Expected output:
```
[1/3] Collecting image paths...
Found 1748 images in data/messidor2/train

[2/3] Loading encoder on cuda...
Running encoder over 1748 images (batch_size=64)...
  encoded 0/1748 images
  encoded 640/1748 images
  encoded 1280/1748 images

Representations shape : torch.Size([1748, 384])
Representations dtype : torch.float32
Sample norm (first 5) : [14.2, 13.8, 15.1, 14.6, 13.9]

[3/3] Saving to data/messidor2/train_reps.pt...
Done. Saved 1748 representations to data/messidor2/train_reps.pt
File size: 2.7 MB
Verification passed — paths and reps are aligned.
```

> **Important:** if you ever change the encoder or the image preprocessing,
> delete the old `.pt` file and re-run this script. Training with stale
> representations will silently produce wrong results.

---

## Step 3 — Train

```bash
python scripts/train.py \
    --reps_file    data/messidor2/train_reps.pt \
    --save_dir     checkpoints/ \
    --image_size   224 \
    --batch_size   8 \
    --lr           1e-4 \
    --total_steps  100000 \
    --save_interval 5000 \
    --log_interval  100 \
    --device       cuda
```

### All training arguments

| Argument | Default | Description |
|---|---|---|
| `--reps_file` | `data/messidor2/train_reps.pt` | Precomputed representation file |
| `--save_dir` | `checkpoints/` | Directory for checkpoints |
| `--image_size` | `224` | Must match `--image_size` used in precompute step |
| `--batch_size` | `8` | Images per gradient step |
| `--lr` | `1e-4` | AdamW learning rate |
| `--total_steps` | `100000` | Total gradient steps |
| `--save_interval` | `5000` | Save checkpoint every N steps |
| `--log_interval` | `100` | Print loss every N steps |
| `--device` | `cpu` | `cpu` or `cuda` |
| `--num_workers` | `0` | DataLoader workers (use 4 on GPU) |
| `--resume` | `None` | Path to checkpoint to resume from |
| `--hidden_dim` | `1024` | JiT ViT hidden dimension (1024=ViT-L, 768=ViT-B) |
| `--depth` | `12` | Number of JiT transformer blocks |
| `--num_heads` | `16` | Attention heads (`hidden_dim / num_heads = head_dim`) |
| `--patch_size` | `16` | Patch size in pixels (`image_size % patch_size == 0`) |
| `--h_dim` | `384` | DinoV3 CLS token dimension |
| `--cond_dim` | `64` | Conditioning bottleneck dimension |

Expected training output:
```
[1/4] Building JiT model...
  Total parameters  : 86.1M
  Trainable         : 86.1M

[2/4] Loading dataset...
Loading representations from data/messidor2/train_reps.pt...
  1748 image-representation pairs loaded
  1748 samples, 218 batches per epoch at batch_size=8

[3/4] Setting up optimiser...

[4/4] Starting training...

step     100/100000 | loss 0.4821
step     200/100000 | loss 0.4103
...
step    5000/100000 | loss 0.2847
  → saved checkpoint: checkpoints/jit_rcdm_step0005000.pt
```

### Resuming a run

```bash
python scripts/train.py \
    --reps_file data/messidor2/train_reps.pt \
    --resume    checkpoints/jit_rcdm_step0005000.pt \
    --device    cuda
```

The checkpoint stores all architecture parameters in `model_cfg`, so you do
not need to repeat `--hidden_dim`, `--depth`, etc.

### GPU memory guide

| `--hidden_dim` | `--depth` | `--batch_size` | VRAM |
|---|---|---|---|
| 1024 (ViT-L) | 12 | 8 | ~22 GB |
| 1024 (ViT-L) | 12 | 4 | ~14 GB |
| 768 (ViT-B) | 12 | 8 | ~12 GB |
| 768 (ViT-B) | 8 | 8 | ~8 GB |

For a single 16 GB GPU use `--hidden_dim 768 --depth 8 --batch_size 8`.
For a single 8 GB GPU use `--hidden_dim 768 --depth 4 --batch_size 4`.

---

## Step 4 — Generate images

```bash
python scripts/sampling.py \
    --checkpoint  checkpoints/jit_rcdm_final.pt \
    --cond_images data/messidor2/test/img1.png \
                  data/messidor2/test/img2.png \
    --out_dir     samples/ \
    --n_samples   4 \
    --num_steps   50 \
    --device      cuda
```

Each conditioning image produces one PNG grid:
```
samples/
    sample_img1.png     ← [cond | gen_1 | gen_2 | gen_3 | gen_4]
    sample_img2.png
```

The leftmost image is the conditioning input (resized to `image_size`).
The remaining images are independently generated but all conditioned on the
same DinoV3 representation.

### All sampling arguments

| Argument | Default | Description |
|---|---|---|
| `--checkpoint` | required | Trained `.pt` checkpoint path |
| `--cond_images` | required | One or more conditioning image paths |
| `--out_dir` | `samples/` | Output directory |
| `--n_samples` | `4` | Samples to generate per conditioning image |
| `--num_steps` | `50` | Heun ODE steps (more steps = slightly better quality) |
| `--device` | `cpu` | `cpu` or `cuda` |
| `--encoder_ckpt` | auto | DinoV3 checkpoint path (defaults to `checkpoints/dinov3_vits16_tmp`) |

### Programmatic usage

```python
import torch
from rcdm.encoder import load_encoder, build_transform
from rcdm.jit import create_jit_model, FlowMatching
from PIL import Image

device = torch.device("cuda")

# Load encoder
encoder   = load_encoder(device=device)
transform = build_transform(image_size=224)

# Load model
state = torch.load("checkpoints/jit_rcdm_final.pt", map_location=device)
cfg   = state["model_cfg"]
model = create_jit_model(**cfg)
model.load_state_dict(state["model"])
model.eval().to(device)

flow = FlowMatching()

# Encode a conditioning image
img = Image.open("data/messidor2/test/some_image.png").convert("RGB")
with torch.no_grad():
    out = encoder(pixel_values=transform(img).unsqueeze(0).to(device))
    h = out.last_hidden_state[:, 0, :]          # (1, 384)
    h = h.expand(4, -1)                         # repeat for 4 samples

# Generate
noise = torch.randn(4, 3, 224, 224, device=device)
with torch.no_grad():
    samples = flow.sample(model, noise, h=h, num_steps=50)
    # samples: (4, 3, 224, 224) in [-1, 1]

# Convert to [0, 1] for display / saving
samples_display = (samples.clamp(-1, 1) + 1) / 2
```

---

## Adjusting the conditioning strength

The conditioning bottleneck dimension `cond_dim` controls how much of the
DinoV3 representation is preserved in the conditioning signal:

| `--cond_dim` | Effect |
|---|---|
| 32 | Stronger regularisation; generated images share only coarse-level structure with the conditioning image |
| **64** | **Default.** Balanced between fidelity and diversity |
| 128 | More conditioning detail; generated images more closely resemble the input |
| 256 | Strong conditioning; use when you want near-deterministic reconstruction |

`--cond_dim` must be set to the **same value** at precompute, train, and
sample time. It is saved in the checkpoint's `model_cfg`.

---

## Module reference

### `rcdm/encoder.py`

```python
from rcdm.encoder import load_encoder, build_transform, encode_image, encode_batch

encoder   = load_encoder(device="cuda")               # frozen DinoV3-S
transform = build_transform(image_size=224)
h         = encode_image("image.png", encoder, transform, device="cuda")
# h: (1, 384)
```

### `rcdm/jit.py`

```python
from rcdm.jit import create_jit_model, FlowMatching

model = create_jit_model(
    image_size=224,   # spatial resolution
    patch_size=16,    # pixels per patch
    hidden_dim=1024,  # ViT token dimension
    depth=12,         # transformer blocks
    num_heads=16,     # attention heads
    h_dim=384,        # encoder output dim
    cond_dim=128,      # conditioning bottleneck
)

flow = FlowMatching()
loss    = flow.training_loss(model, x_clean, h)            # training
samples = flow.sample(model, noise, h, num_steps=50)       # inference
```

### `rcdm/conditioning.py`

```python
from rcdm.conditioning import ConditioningProjector, AdaLNZero

# Used internally by JiT — you rarely need to import these directly.
proj = ConditioningProjector(h_dim=384, cond_dim=64)
ada  = AdaLNZero(hidden_dim=1024, cond_dim=64)
```

### `rcdm/dataset.py`

```python
from rcdm.dataset import RepresentationDataset
from torch.utils.data import DataLoader

dataset    = RepresentationDataset("data/messidor2/train_reps.pt", image_size=224)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

x, h = next(iter(dataloader))
# x: (8, 3, 224, 224) — images in [-1, 1]
# h: (8, 384)         — DinoV3 representations
```

---

## Troubleshooting

**`OSError: Can't load tokenizer for 'checkpoints/dinov3_vits16_tmp'`**
The encoder checkpoint directory is empty or missing `config.json`. See
[Step 0](#step-0--place-the-dinov3-checkpoint).

**`AssertionError: Expected 384-dim DinoV3 ViT-S reps`**
Your `.pt` file was generated with the old ResNet-50 or DinoV2 encoder.
Delete it and re-run `precompute_reps.py`.

**`RuntimeError: mat1 and mat2 shapes cannot be multiplied`**
The `--h_dim` or `--cond_dim` passed to `train.py` does not match what the
checkpoint was trained with. Load the checkpoint first and read `model_cfg`:
```python
import torch
print(torch.load("checkpoints/jit_rcdm_final.pt", map_location="cpu")["model_cfg"])
```

**CUDA out of memory during training**
Reduce `--batch_size` or use a smaller model (`--hidden_dim 768 --depth 8`).
See the [GPU memory guide](#gpu-memory-guide).

**Loss is not decreasing after 10k steps**
- Check that the representations file was generated with the *current*
  encoder (not a stale file from a previous run).
- Try a higher learning rate (`--lr 3e-4`) for the first 20k steps.
- Make sure `--image_size` matches between precompute, train, and the actual
  image dimensions in the dataset.

---

## Citation

If you use this code, please cite the original RCDM paper and the JiT paper:

```bibtex
@article{bordes2022high,
  title   = {High Fidelity Visualization of What Your Self-Supervised Representation Knows About},
  author  = {Bordes, Florian and Balestriero, Randall and Vincent, Pascal},
  journal = {Transactions on Machine Learning Research},
  year    = {2022}
}

@article{peebles2023scalable,
  title   = {Scalable Diffusion Models with Transformers},
  author  = {Peebles, William and Xie, Saining},
  journal = {ICCV},
  year    = {2023}
}
```
