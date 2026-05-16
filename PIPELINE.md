# JiT-RCDM Pipeline Documentation

**Task:** Conditional retinal fundus image generation on Messidor-2  
**Conditioning:** Frozen DinoV3 ViT-S/16 CLS token (384-dim) from the same patient's fundus image  
**Generator:** Plain ViT denoiser (JiT) trained with flow matching, producing 224×224 RGB images

---

## 1 — Overview

### Original frameworks

This codebase merges two independent open-source works:

| Framework | Repo | What it is |
|---|---|---|
| **RCDM** | [facebookresearch/RCDM](https://github.com/facebookresearch/RCDM) | Representation-Conditioned Diffusion Model. Trains a conditional UNet denoiser (ADM-style) where the conditioning signal is an SSL representation `h` from a frozen encoder. Original encoder: ResNet-50 (2048-dim). Original denoiser: UNet with Conditional Batch Norm. Trained as DDPM on 64×64 Tiny ImageNet. |
| **JiT** | [LTH14/JiT](https://github.com/LTH14/JiT) | "Just in Time" — a plain ViT denoiser trained with flow matching. Replaces the UNet backbone with a standard Vision Transformer conditioned via adaLN-Zero. Class-conditional on ImageNet. |

### What this codebase does

RCDM's conditioning idea (use an SSL representation instead of a class label) is applied to JiT's architecture (ViT + flow matching). The result is a pipeline that:

1. Encodes a fundus image into a 384-dim semantic vector `h` via frozen **DinoV3 ViT-S/16**
2. Trains a **JiT ViT denoiser** conditioned on `h` to generate new 224×224 fundus images
3. At inference, generates diverse plausible images that match the semantic content (pathology pattern, vessel structure, optic disc location) of the conditioning image

### What was kept, what was changed

| Component | From RCDM | From JiT | Changed in JiT-RCDM |
|---|---|---|---|
| Conditioning idea (SSL `h` instead of class label) | ✓ | — | `class_emb(y)` → `cond_proj(h)` |
| `ConditioningProjector` (Linear + SiLU) | ✓ | — | Input dim 2048 → 384; output dim configurable |
| `ConditionalBatchNorm2d` | ✓ (UNet path) | — | Kept for compat; unused in JiT path |
| Frozen encoder, `eval()` + `requires_grad=False` | ✓ | — | Encoder switched to DinoV3 local ckpt |
| CFG null-h dropout (training) | ✓ | — | Null vector → learnable `nn.Parameter` |
| Plain ViT block structure | — | ✓ | RMSNorm, SwiGLU, RoPE, qk-norm added |
| adaLN-Zero conditioning per block | — | ✓ | LayerNorm → RMSNorm inside adaLN |
| Sinusoidal timestep embedding | — | ✓ | Unchanged |
| Flow matching (x-prediction, linear path) | — | ✓ | t-sampler μ corrected to −0.8 |
| Heun ODE sampler | — | ✓ | CFG two-pass blending added at x-pred level |
| EMA shadow weights | — | ✓ | Added to checkpoints (was missing) |
| AdamW β₂ = 0.95 | — | ✓ | Was defaulting to PyTorch's 0.999 |

---

## 2 — Data flow (shapes at each stage)

```
Conditioning image (PIL, any resolution)
    │
    ▼  build_transform(224) — Resize + CenterCrop + ImageNet normalise
Encoder input : (1, 3, 224, 224)
    │
    ▼  DinoV3 ViT-S/16  [frozen]
h : (B, 384)   ← CLS token from last_hidden_state[:, 0, :]
    │
    ├─────────────────────────────────────────────┐
    │                                             │
    ▼  ConditioningProjector                      │ Training:
h_proj : (B, cond_dim)                            │ z_t = t·x + (1−t)·ε
    │                                             │ t ~ sigmoid(N(−0.8, 0.8))
    ▼  + time_embed(sinusoidal(t))                │
c : (B, cond_dim)   ← shared across all blocks   │
    │                                             │
    ▼  PatchEmbed (Conv2d, patch=16)              │
tokens : (B, 196, hidden_dim)                     │
    │                                             │
    ▼  JiTBlock × depth                           │
       adaLN-Zero(c) → 6 modulation params        │
       Attention(RoPE + qk-norm) + SwiGLU         │
tokens : (B, 196, hidden_dim)                     │
    │                                             │
    ▼  FinalLayer (adaLN shift/scale + Linear)    │
patches : (B, 196, 16×16×3)                       │
    │                                             │
    ▼  unpatchify                                 │
x_pred : (B, 3, 224, 224)   ← predicted clean x  │
    │                                             │
    └──── MSE loss vs x ──────────────────────────┘

Inference:
    noise ~ N(0,I) : (B, 3, 224, 224)
    │
    ▼  50-step Heun ODE  (t: 0 → 1)
       each step: x_pred_cond   = model(z, t, h)
                  x_pred_uncond = model(z, t, null_h)
                  x_pred = x_pred_uncond + cfg_scale·(x_pred_cond − x_pred_uncond)
                  v = (x_pred − z) / (1 − t)
    │
    ▼
generated image : (B, 3, 224, 224)
```

---

## 3 — Component details

### 3.1 — Encoder: DinoV3 ViT-S/16

**File:** `rcdm/encoder.py`

| Property | Value |
|---|---|
| Architecture | ViT-S/16 (patch size 16, hidden_dim 384, depth 12) |
| Training | DINO self-supervised learning, fine-tuned for medical/retinal imagery |
| Checkpoint | `checkpoints/dinov3_vits16_tmp/` (local, HuggingFace format) |
| Output used | `last_hidden_state[:, 0, :]` — the CLS token |
| Output dimension | **384** |
| Input image size | **Always 224×224** (fixed by the model's positional embedding grid: 224/16 = 14×14 patches) |
| Normalisation | ImageNet mean/std: `[0.485, 0.456, 0.406]` / `[0.229, 0.224, 0.225]` |
| Frozen | Yes — `eval()` + `requires_grad=False` throughout |

**What RCDM used:** `torchvision.models.resnet50`, `avgpool` output, 2048-dim, trained on ImageNet for classification.

**Why we changed it:**

- ResNet-50 avgpool vectors are trained with classification pressure — they encode "which class" but lose spatial texture and fine structure. DINO-style SSL learns representations where nearby vectors correspond to perceptually similar images, which is what RCDM conditioning requires.
- DinoV3 is fine-tuned on medical/retinal data, so its CLS token is semantically aligned with Messidor-2's feature space (vessel patterns, disc morphology, pathology stage).
- ViT-S/16 keeps the representation compact (384 vs 1024 for DinoV2-L), making the conditioning path lighter.

**Important:** the encoder runs at 224×224 regardless of what `image_size` the generative model uses. These are two independent configurations that must not be conflated.

---

### 3.2 — Precomputed representations

**File:** `scripts/precompute_reps.py`  
**Output:** `data/messidor2/train_reps.pt` — dict `{"paths": [...], "reps": Tensor(N, 384)}`

Representations are computed once before training and stored on disk. The generative model never calls the encoder during training — it reads `h` directly from the `.pt` file.

**Why precompute:** A single DinoV3 forward pass at 224×224 is cheap, but doing it every training step adds latency, complicates batching, and makes experiments harder to reproduce. Precomputing lets training run as fast as the generative model allows.

---

### 3.3 — Dataset

**File:** `rcdm/dataset.py` — `RepresentationDataset`

Returns `(x, h)` pairs:

| Tensor | Shape | Normalisation | Purpose |
|---|---|---|---|
| `x` | `(3, 224, 224)` | `[-1, 1]` via `(pixel − 0.5) / 0.5` | Target for flow-matching MSE loss |
| `h` | `(384,)` | None (raw CLS token) | Conditioning vector for the denoiser |

**Dual normalisation:** `x` and `h` use different normalisation on purpose.  
- `h` was computed with ImageNet normalisation — changing `x`'s normalisation does not affect it.  
- The denoiser expects `x ∈ [−1, 1]` (standard diffusion convention); the encoder expects ImageNet-normalised inputs (ViT convention). These are independent and must not be mixed.

---

### 3.4 — Conditioning projector

**File:** `rcdm/conditioning.py` — `ConditioningProjector`

```
h : (B, 384)
  → Linear(384, cond_dim)
  → SiLU
→ h_proj : (B, cond_dim)
```

| Parameter | Value in S16 preset | Reasoning |
|---|---|---|
| `h_dim` | 384 | DinoV3 ViT-S/16 CLS token dimension |
| `cond_dim` | 128 | Bottleneck: compresses 384-dim signal to 128-dim before adding to timestep embedding. Regularises the conditioning path for smaller models. |

**What RCDM used:** `Linear(2048 → 512) + SiLU`. The same one-layer design is kept. Only the dimensions change.

**cond_dim vs hidden_dim:** The JiT paper uses `cond_dim == hidden_dim` (no bottleneck). The S presets use `cond_dim=128 < hidden_dim=384`. The bottleneck forces the model to compress `h` to its dominant axes before modulating each block — acts as a light regulariser on a small dataset. At `hidden_dim=768` (JiT-B) the paper-faithful `cond_dim=768` (no bottleneck) is the right choice.

---

### 3.5 — Timestep embedding

**File:** `rcdm/jit.py` — `timestep_embedding` + `time_embed` MLP

```
t : (B,)   scalar ∈ [0, 1]
  → sinusoidal(t × 1000, cond_dim)        # (B, cond_dim)  — scaled to DDPM freq range
  → Linear(cond_dim → 4·cond_dim)
  → SiLU
  → Linear(4·cond_dim → cond_dim)
→ t_emb : (B, cond_dim)
```

The sinusoidal basis uses `t × 1000` to map the continuous [0,1] flow-matching time into the frequency range originally designed for DDPM's discrete 0–1000 integer steps. This is taken directly from JiT.

**What JiT used:** Identical. Unchanged.

---

### 3.6 — Shared conditioning signal c

```
c = t_emb + h_proj        (B, cond_dim)
```

This single vector drives all adaLN-Zero blocks. It is computed once per forward pass and shared — each block has its own small MLP that maps `c → 6·hidden_dim` modulation parameters, but the input `c` is identical for all blocks.

**What JiT used:** `c = t_emb + class_emb(y)` where `y` is an integer class label. We replace `class_emb(y)` with `cond_proj(h)` where `h` is the continuous DinoV3 CLS token. Same addition, different input.

**Why add (not concatenate):** Adding timestep and conditioning keeps the `c` dimension at `cond_dim`. Concatenation would double it, increasing adaLN MLP parameters by 2× in every block. The additive design also allows the timestep signal to dominate early in training (when `cond_proj` weights are near-zero from `trunc_normal` init), which provides natural curriculum: the model first learns the diffusion trajectory, then learns to steer it with `h`.

---

### 3.7 — JiT denoiser (plain ViT)

**File:** `rcdm/jit.py` — `JiT`, `JiTBlock`, `Attention`, `SwiGLU`, `PatchEmbed`, `FinalLayer`

#### Presets

| Preset | `hidden_dim` | `num_heads` | `head_dim` | `patch_size` | `cond_dim` | Tokens @ 224px | Params |
|---|---|---|---|---|---|---|---|
| `JiT_S_16` | 384 | 6 | 64 | 16 | 128 | 196 | ~25 M |
| `JiT_S_32` | 384 | 6 | 64 | 32 | 128 | 49 | ~25 M |

`head_dim = hidden_dim / num_heads = 384 / 6 = 64` — standard ViT-S head dimension. 2D RoPE requires `head_dim % 4 == 0`; 64 satisfies this.

#### Patch embedding

```
z_t : (B, 3, 224, 224)
  → Conv2d(3, hidden_dim, kernel=patch_size, stride=patch_size)   [no bias]
  → flatten spatial → transpose
→ tokens : (B, 196, hidden_dim)      for patch=16
           (B,  49, hidden_dim)      for patch=32
```

**What JiT used:** same Conv2d patchify + a learned `pos_embed` nn.Parameter added after the projection.  
**What we changed:** `pos_embed` is removed. Positional information is injected via 2D RoPE inside each attention block instead (see §3.8).

#### Transformer blocks

12 × `JiTBlock`, each:

```
input : (B, N, hidden_dim), conditioning c : (B, cond_dim)

adaLN-Zero(c) → shift_a, scale_a, gate_a, shift_f, scale_f, gate_f   (each: B, hidden_dim)

x ← x + gate_a · Attention( (1 + scale_a) · RMSNorm(x) + shift_a )
x ← x + gate_f · SwiGLU(   (1 + scale_f) · RMSNorm(x) + shift_f )
```

**What JiT used:** Same adaLN-Zero block structure with `nn.LayerNorm`, `nn.MultiheadAttention`, GELU FFN.  
**What we changed:** LayerNorm → RMSNorm; nn.MultiheadAttention → custom Attention (qk-norm + RoPE); GELU FFN → SwiGLU. See §3.8–3.10.

#### Final layer

```
tokens : (B, 196, hidden_dim)
  → adaLN (shift + scale from c, no gate)
  → RMSNorm
  → Linear(hidden_dim → patch_size² × 3)    [zero-init]
  → unpatchify
→ x_pred : (B, 3, 224, 224)
```

Zero-initialising the final linear layer means the model outputs `x_pred ≈ 0` at training step 0, which gives a loss of `≈ E[||x||²]` — a finite, predictable starting point rather than a random large loss.

---

### 3.8 — Normalisation: RMSNorm

**File:** `rcdm/conditioning.py` — `RMSNorm`

```python
x_out = x / sqrt(mean(x²) + eps)   # optionally scaled by learned weight
```

**What JiT used:** `nn.LayerNorm` (mean-centering + RMS scaling).  
**What we changed:** RMSNorm drops the mean-centering step.

Why RMSNorm:
- Standard in all modern transformer-based diffusion models that follow JiT (MAR, SiT, etc.)
- Faster: one fewer reduction operation per normalisation
- Numerically more stable in mixed-precision training
- Affine weight `γ` is kept where the block needs it (qk-norm in Attention); removed (`affine=False`) inside adaLN-Zero since the adaLN modulation already provides scale and shift

---

### 3.9 — Positional encoding: 2D RoPE

**File:** `rcdm/jit.py` — `compute_2d_rope_freqs`, `apply_rotary_emb`

Rotary Position Embedding encodes relative positions by rotating query and key vectors before the dot product. For a 2D patch grid:

- The `head_dim` is split into four quarters
- Quarters 1–2 encode the row position (which patch row, 0–13 for a 14×14 grid)
- Quarters 3–4 encode the column position (which patch column)
- Each position `(r, c)` gets a unique rotation applied to Q and K

```
freqs_cis : (196, head_dim//2)  complex64  ← precomputed once, stored as buffer
apply_rotary_emb(q, freqs_cis) → rotated q
apply_rotary_emb(k, freqs_cis) → rotated k
```

**What JiT used:** Learned absolute `pos_embed: nn.Parameter(B, N, hidden_dim)` added to patch tokens before the first block.  
**What we changed:** Learned pos_embed removed from `PatchEmbed`. 2D RoPE frequencies registered as a buffer on `JiT` and passed to every `JiTBlock.forward` → `Attention.forward`.

Why RoPE over learned positional embedding:
- Learned pos_embed is a fixed lookup table — it cannot generalise to image sizes not used during training
- RoPE encodes *relative* position: the dot product `q·k` after rotation depends on the offset `(r1−r2, c1−c2)`, not the absolute position. Two tokens that are 3 patches apart encode the same spatial relationship at any location in the image.
- Requires zero additional parameters (computed analytically)

---

### 3.10 — Attention: custom MHA with qk-norm

**File:** `rcdm/jit.py` — `Attention`

```python
qkv = Linear(hidden_dim → 3·hidden_dim, bias=False)(x)   # (B, N, 3·hidden_dim)
q, k, v = split into heads                               # each (B, N, heads, head_dim)
q = RMSNorm(head_dim, affine=True)(q)                    # per-head normalisation
k = RMSNorm(head_dim, affine=True)(k)
q = apply_rotary_emb(q, freqs_cis)                       # 2D RoPE
k = apply_rotary_emb(k, freqs_cis)
out = scaled_dot_product_attention(q, k, v)              # Flash Attention when available
out = Linear(hidden_dim → hidden_dim, bias=False)(out)
```

**What JiT used:** `nn.MultiheadAttention` (standard PyTorch module; includes bias, no qk-norm, no RoPE).  
**What we changed:** Replaced with custom class that adds qk-norm and RoPE.

Why qk-norm:
- Attention logits = `q·k / sqrt(head_dim)`. If Q and K grow large during training, logits overflow → attention collapses to one-hot (all weight on one token) → gradient vanishes
- Per-head RMSNorm on Q and K bounds the logit scale regardless of representation magnitude
- Cost: two extra `RMSNorm(head_dim)` per block — negligible

---

### 3.11 — FFN: SwiGLU

**File:** `rcdm/jit.py` — `SwiGLU`

```python
inner_dim = round_to_256(hidden_dim × mlp_ratio × 2/3)   # ≈ 341 for hidden=384, ratio=4 → 512

out = Linear(inner_dim → hidden_dim, bias=False)(
    SiLU( Linear(hidden_dim → inner_dim, bias=False)(x) )   # gate
    ×     Linear(hidden_dim → inner_dim, bias=False)(x)     # value
)
```

The `2/3` factor keeps total parameter count equal to a standard GELU FFN at the same `mlp_ratio`. The `round_to_256` aligns `inner_dim` to a hardware-friendly multiple.

**What JiT used:** `nn.Sequential(Linear, GELU, Linear)` — two projections.  
**What we changed:** Three projections (gate + value + output), SiLU gating, no bias.

Why SwiGLU:
- Gated linear units empirically outperform GELU FFNs at the same parameter count in all modern ViT-scale models (LLaMA, GPT-4, PaLM, MAR, JiT)
- The gate mechanism allows the FFN to selectively route information: tokens that don't match the learned feature pattern produce near-zero output from `SiLU(gate)`, leaving those token representations unchanged

---

### 3.12 — adaLN-Zero conditioning

**File:** `rcdm/conditioning.py` — `AdaLNZero`

The shared vector `c : (B, cond_dim)` drives a small per-block MLP:

```
c → SiLU → Linear(cond_dim → 6·hidden_dim)
  → chunk into 6 vectors: shift_a, scale_a, gate_a, shift_f, scale_f, gate_f
```

Applied to the token sequence:
```
x ← x + gate_a · Attn( (1 + scale_a) · RMSNorm(x) + shift_a )
x ← x + gate_f · FFN(  (1 + scale_f) · RMSNorm(x) + shift_f )
```

**Zero-init:** The `Linear(cond_dim → 6·hidden_dim)` output projection has weights and biases initialised to zero. At training step 0:
- `scale_a = scale_f = 0` → norm outputs are unchanged
- `shift_a = shift_f = 0` → no bias offset
- `gate_a = gate_f = 0` → **entire block is an identity function**

The network stabilises in unconditioned mode first. Conditioning gradually takes effect as the gates depart from zero.

**What RCDM used:** `ConditionalBatchNorm2d` — scalar `γ/β` per channel applied to 2D spatial feature maps. This only works for CNN feature maps `(B, C, H, W)`.  
**What we changed:** Replaced with adaLN-Zero for token sequences `(B, N, D)`, taken directly from JiT (which took it from DiT).

---

### 3.13 — Learnable null-h (CFG)

**File:** `rcdm/jit.py` — `JiT.null_h`, `FlowMatching`

```python
self.null_h = nn.Parameter(torch.zeros(h_dim))   # (384,) — trained jointly
```

**Training (CFG dropout):** With probability `p_uncond` (default 0.1), a batch element's `h` is replaced by `null_h`:

```python
mask = torch.rand(B) < p_uncond          # 10% of batch
h_used = torch.where(mask, null_h.expand(B,-1), h)
```

**Inference (two-pass CFG):**

```python
x_pred_cond   = model(z, t, h)
x_pred_uncond = model(z, t, null_h.expand(B,-1))
x_pred = x_pred_uncond + cfg_scale × (x_pred_cond − x_pred_uncond)
```

**What RCDM used:** `torch.zeros_like(h)` — hard-coded zero vector as null conditioning.  
**What we changed:** Replaced with a learnable `nn.Parameter`. The null vector is now trained to represent "no conditioning" in the model's own learned representation space, rather than the arbitrary point `h=0`.

**What JiT used:** Learnable `null_class` embedding (integer class label version of the same idea). Our `null_h` is the continuous-h analogue.

---

### 3.14 — Flow matching objective

**File:** `rcdm/jit.py` — `FlowMatching.training_loss`

Linear flow path:
```
z_t = t·x + (1−t)·ε          t ∈ [0, 1],  ε ~ N(0, I)
```
- At `t=0`: `z_t = ε` (pure noise)
- At `t=1`: `z_t = x` (clean image)

x-prediction:
```
loss = MSE(model(z_t, t, h), x)
```

Logit-normal t-sampler:
```python
u = -0.8 + 0.8 * torch.randn(B)
t = sigmoid(u)
```

| Parameter | RCDM (DDPM) | JiT (original) | JiT-RCDM |
|---|---|---|---|
| Objective | ε-prediction | x-prediction | x-prediction |
| Noise path | cosine/linear β schedule | linear `z_t = t·x + (1−t)·ε` | same |
| t distribution | Uniform integer [0, 1000] | logit-normal(0, 1) | **logit-normal(−0.8, 0.8)** |

**Why μ=−0.8 (JiT paper Tab. 3):** Shifting the logit-normal mode left concentrates more training on intermediate t values (t ≈ 0.3–0.6) where the model must reason about image structure. Uniform t wastes compute on t≈0 (near-random noise, trivial to denoise) and t≈1 (near-clean, trivial to predict).

---

### 3.15 — Heun ODE sampler

**File:** `rcdm/jit.py` — `FlowMatching.sample`

Steps from `t=0` (noise) to `t≈1` (image):

```
v₁ = (x_pred(z, t) − z) / (1 − t)          ← velocity at current step
z* = z + dt · v₁                             ← Euler predictor
v₂ = (x_pred(z*, t+dt) − z*) / (1 − t−dt)  ← velocity at predicted step
z  = z + dt · (v₁ + v₂) / 2                 ← Heun correction (2nd order)
```

Pure Euler at the last step avoids the `1/(1−t)` singularity at `t=1`.

**What RCDM used:** `p_sample_loop` — stochastic Markov chain, 1000 DDPM steps, fresh noise injected at every step.  
**What we changed:** Deterministic ODE solver, 50 steps, no stochastic noise. Same conditioning interface.

Why 50 Heun steps beat 1000 DDPM steps: DDPM injects fresh noise every step to maintain the Markov chain; the quality bottleneck is the discrete schedule. Flow matching follows a smooth ODE — a 2nd-order solver achieves higher accuracy with far fewer function evaluations.

---

### 3.16 — EMA (Exponential Moving Average)

**File:** `scripts/train.py` — `EMA` class

```python
shadow[name] = 0.9999 × shadow[name] + 0.0001 × param.data
```

Updated after every optimizer step. At inference, shadow weights are swapped in before sampling and restored afterward.

| Property | Value | Source |
|---|---|---|
| Decay | 0.9999 | JiT paper Tab. 9 — ablated as best |
| Stored in checkpoint | Yes (`"ema": ema.state_dict()`) | Fixed in JiT-RCDM (was missing) |
| Used at inference | Yes — `sampling.py` loads EMA shadow | Fixed in JiT-RCDM (was loading raw) |

**What the original had:** The EMA class existed and updated the shadow during training, but the shadow was never written to the checkpoint dict. Resuming discarded all EMA history.

---

### 3.17 — Training recipe

**File:** `scripts/train.py`

| Hyperparameter | Original JiT | JiT-RCDM default | Notes |
|---|---|---|---|
| Optimiser | AdamW | AdamW | Unchanged |
| β₁, β₂ | 0.9, **0.95** | 0.9, **0.95** | PyTorch default was 0.999; corrected |
| Weight decay | 0.0 | 0.0 | Unchanged |
| LR | 1×10⁻⁴ | 1×10⁻⁴ | Sensible default |
| LR schedule | Warmup + cosine | **Linear warmup** + constant | `--warmup_steps 1000` |
| Gradient clipping | max norm 1.0 | max norm 1.0 | Unchanged |
| Gradient accumulation | — | `--grad_accum 4` | Simulates batch 32 on MPS with batch_size 8 |
| EMA decay | 0.9999 | 0.9999 | Now persisted |
| CFG dropout | `p_uncond` | `--cfg_dropout 0.1` | 10% null-h during training |

**Recommended training duration:**

| Steps | Effective updates (batch=8, accum=4) | What you typically see |
|---|---|---|
| 5 k | 40 k | Mean color, coarse brightness — orange blobs |
| 15 k | 120 k | Coarse spatial structure — disc, rough vessel regions |
| 30 k | 240 k | Vessel topology becomes visible; CFG scale 1.5–2.0 usable |
| 50 k | 400 k | Fine vessel branches; CFG scale 2.0–3.0 usable |
| 100 k+ | 800 k+ | Micro-aneurysm-level detail; full CFG range |

---

## 4 — Configuration summary (JiT_S_16, recommended local run)

```
Encoder          DinoV3 ViT-S/16  — frozen, 384-dim CLS token
                 input: 224×224, ImageNet normalised

Denoiser         JiT_S_16
  hidden_dim     384
  depth          12  (transformer blocks)
  num_heads      6   (head_dim = 64)
  patch_size     16  (196 tokens at 224px)
  cond_dim       128 (conditioning bottleneck: 384→128→conditioning c)
  mlp_ratio      4   (SwiGLU inner_dim = 512)
  h_dim          384 (matches encoder output)
  Parameters     ~25 M

Conditioning     c = time_embed(sinusoidal(t, 128)) + cond_proj(h)
                 shape: (B, 128)  → per-block adaLN → (B, 6×384)

Training
  objective      x-prediction MSE
  t-sampler      logit-normal(μ=−0.8, σ=0.8)
  batch_size     8  ×  grad_accum 4  =  effective 32
  lr             1e-4 with 1000-step linear warmup
  betas          (0.9, 0.95)
  ema_decay      0.9999
  cfg_dropout    0.1

Sampling
  steps          50 Heun ODE steps
  cfg_scale      1.0 until step 15k → 1.5–2.0 until step 30k → 2.0–3.0 beyond
```

---

## 5 — File map

| File | Origin | Role |
|---|---|---|
| `rcdm/encoder.py` | Written for JiT-RCDM | Load + freeze DinoV3; `build_transform(224)` |
| `rcdm/dataset.py` | Adapted from RCDM | `RepresentationDataset` — serves `(x, h)` pairs |
| `rcdm/conditioning.py` | Adapted from RCDM + JiT | `RMSNorm`, `ConditioningProjector`, `AdaLNZero` |
| `rcdm/jit.py` | Adapted from JiT | Full denoiser + flow-matching utilities |
| `scripts/precompute_reps.py` | Adapted from RCDM | Batch-encode all images → `train_reps.pt` |
| `scripts/train.py` | Adapted from RCDM + JiT | Training loop with EMA, warmup, grad accum |
| `scripts/sampling.py` | Written for JiT-RCDM | Inference: load checkpoint, run Heun ODE, save grid |
| `guided_diffusion/` | From RCDM (unchanged) | Legacy UNet + DDPM — not used by JiT path |
