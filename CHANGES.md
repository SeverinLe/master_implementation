# JiT-RCDM — Architecture Change Log

Migrating RCDM from **ResNet-50 + UNet + DDPM** to **DinoV3 + JiT ViT + Flow Matching**
for fundus retinal image generation on **Messidor-2**.

Changes are listed in the order they were made. Where a later change supersedes
an earlier one, the earlier entry is annotated accordingly.

---

## Final state at a glance

| Component | Original RCDM | JiT-RCDM (final) |
|---|---|---|
| **Encoder** | ResNet-50, 2048-dim avgpool | DinoV3 ViT-S/16, 384-dim CLS token |
| **Conditioning projector** | Linear(2048→512) + SiLU | Linear(384→cond_dim) + SiLU |
| **Conditioning dim** | 512 | `cond_dim` — equals `hidden_dim` by default; 128 in S presets |
| **Denoiser** | UNet + ConditionalBatchNorm2d | Plain ViT + adaLN-Zero per block |
| **Normalisation** | BatchNorm / LayerNorm | RMSNorm throughout (Change 11 fix-5a) |
| **Positional encoding** | Learned absolute embedding | 2D Rotary Position Embedding / RoPE (Change 11 fix-5b) |
| **FFN** | Linear → GELU → Linear | SwiGLU, no bias (Change 11 fix-5d) |
| **Attention** | `nn.MultiheadAttention` | Custom MHA + qk-norm + RoPE (Change 11 fix-5c) |
| **Prediction target** | ε (noise) | x (clean image) |
| **Noise / flow schedule** | DDPM cosine/linear, 1000 steps | Linear flow, logit-normal(−0.8, 0.8) t (Change 11 fix-4b) |
| **Sampler** | DDPM `p_sample_loop` | 50-step Heun ODE |
| **`y_emb`** | Not used | Replaced by `cond_proj(h)` |
| **Null conditioning** | `zeros_like(h)` | Learnable `nn.Parameter` `null_h` (Change 11 fix-3) |
| **EMA** | Not persisted | Saved in checkpoints; applied at inference (Change 11 fix-1) |
| **AdamW β₂** | 0.999 (default) | 0.95 per JiT paper (Change 11 fix-4a) |
| **LR schedule** | Fixed LR | Linear warmup + constant (Change 11 fix-4c) |
| **Grad accumulation** | Not supported | `--grad_accum N` (Change 11 fix-4d) |
| **Preset variants** | — | `JiT_S_16` (196 tokens, ~24 M), `JiT_S_32` (49 tokens, ~26 M) |
| **Key new file** | — | `rcdm/jit.py` |

---

## Change 1 — ResNet-50 encoder replaced by DinoV2 ViT-L/14

> **Superseded by Change 6** — DinoV2 was later replaced by DinoV3.
> This entry documents the reasoning for moving away from ResNet-50, which still applies.

**File:** `rcdm/encoder.py`

### What changed

| | Before | After |
|---|---|---|
| Backbone | `torchvision.models.resnet50` | `AutoModel.from_pretrained("facebook/dinov2-large")` |
| Forward call | `encoder(x)` | `encoder(pixel_values=x).last_hidden_state[:, 0, :]` |
| Output dim | 2048 (avgpool vector) | 1024 (CLS token) |
| Default image size | 64 px | 224 px |

```python
# OLD
encoder = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
encoder.fc = nn.Identity()
h = encoder(x)                              # (B, 2048)

# INTERMEDIATE (DinoV2)
encoder = AutoModel.from_pretrained("facebook/dinov2-large")
out = encoder(pixel_values=x)
h = out.last_hidden_state[:, 0, :]          # CLS token → (B, 1024)
```

### Why ResNet-50 was replaced

- ViT-based encoders with DINO training produce representations that encode
  *what the image looks like* without label pressure. ResNet-50 avgpool vectors
  conflate classification-discriminative features with the representation.
- ViT attention is spatially aware of long-range retinal structures (vessel
  trees, optic disc, micro-aneurysms) that a CNN pyramid compresses away.
- The CLS token forward call (`pixel_values=x`, structured output object) is
  retained through all subsequent encoder changes.

### Invariants preserved

- Encoder is always **frozen**: `requires_grad = False`, `eval()` mode.
- `encode_image()` and `encode_batch()` signatures are unchanged.
- Precomputed `.pt` representation files must be regenerated whenever the
  encoder changes.

---

## Change 2 — Conditioning projector: input 2048→1024, output 512→1024

> **Superseded by Change 7** — output was later changed to 64 (bottleneck).
> Input change (2048→1024) follows directly from Change 1 and still applies.

**File:** `rcdm/conditioning.py` — `ConditioningProjector`

### What changed

```python
# OLD
ConditioningProjector(h_dim=2048, cond_dim=512)   # Linear(2048→512) + SiLU

# INTERMEDIATE
ConditioningProjector(h_dim=1024, cond_dim=1024)  # Linear(1024→1024) + SiLU
```

### Why

`h_dim 2048→1024` follows directly from the encoder dimension change.
`cond_dim 512→1024` was set to match the JiT hidden dimension so the projected
vector could be added to the timestep embedding without shape mismatch. This
pairing was later reconsidered (see Change 7).

---

## Change 3 — ConditionalBatchNorm2d replaced by AdaLNZero

**Files:** `rcdm/conditioning.py` (new `AdaLNZero` class), `rcdm/jit.py` (usage)

### What changed

`ConditionalBatchNorm2d` (cBN) modulated 2-D spatial feature maps `(B, C, H, W)`
with per-channel scale/shift derived from `h`. The JiT ViT operates on 1-D
token sequences `(B, N, D)` — cBN has no meaning in that space.

`AdaLNZero` (from DiT, Peebles & Xie 2022) produces **6 modulation scalars**
from the fused conditioning signal `c = timestep_emb(t) + cond_proj(h)`:

```
shift_attn, scale_attn, gate_attn,   ← attention branch
shift_ffn,  scale_ffn,  gate_ffn     ← FFN branch
```

Applied per block:
```python
x = x + gate_attn * Attn((1 + scale_attn) * LayerNorm(x) + shift_attn)
x = x + gate_ffn  * FFN( (1 + scale_ffn)  * LayerNorm(x) + shift_ffn)
```

The adaLN output projection is **zero-initialised**:
```python
nn.init.zeros_(self.adaLN_modulation[-1].weight)
nn.init.zeros_(self.adaLN_modulation[-1].bias)
```

### Why

- **Architecture match.** cBN is a CNN primitive; it has no spatial dimensions
  to normalise over a token sequence. adaLN-Zero is the canonical conditioning
  mechanism for ViT-based diffusion models (DiT, JiT).
- **Zero-init stability.** All gate values start at 0 → every block is an
  identity function at training step 0. The network stabilises in unconditioned
  mode first, then conditioning gradually takes effect — same reasoning as
  cBN's `gamma=1, beta=0` initialisation.
- **One `c` per forward pass.** The fused vector `c` is computed once and
  shared across all blocks. Each block has its own small adaLN MLP but inputs
  the same `c`, giving per-block flexibility at low cost.

`ConditionalBatchNorm2d` is **kept** in `rcdm/conditioning.py` for backward
compatibility with the legacy UNet path.

---

## Change 4 — Prediction target: ε-prediction → x-prediction with flow matching

**Files:** `rcdm/jit.py` (`FlowMatching`), `scripts/train.py`

### Training objective

```python
# OLD — DDPM, ε-prediction
z_t = sqrt(ᾱ_t) * x + sqrt(1 − ᾱ_t) * ε
loss = ||model(z_t, t) − ε||²              # predict noise

# NEW — flow matching, x-prediction
z_t = t * x + (1 − t) * ε                 # linear interpolation, t ∈ [0, 1]
t ~ sigmoid(N(0, 1))                       # logit-normal, not uniform
loss = ||model(z_t, t, h) − x||²          # predict clean image
```

### Timestep schedule comparison

| Property | DDPM | Flow Matching |
|---|---|---|
| t domain | discrete integers 0…T (T=1000) | continuous float ∈ (0, 1) |
| t distribution | Uniform[0, T] | logit-normal(0, 1) |
| Noise path | cosine/linear β schedule | linear: `z_t = t·x + (1−t)·ε` |
| Reverse process | stochastic Markov chain | deterministic ODE |

### Sampler: 50-step Heun ODE

```python
# OLD
x = diffusion.p_sample_loop(model, shape, model_kwargs={"h": h})  # 1000 steps

# NEW — velocity from x-prediction: v = (x_pred − z_t) / (1 − t)
x = flow.sample(model, noise, h=h, num_steps=50)
```

Heun is a 2nd-order predictor–corrector:
```
v₁ = (x_pred(z_t, t) − z_t) / (1 − t)    # velocity at current step
z* = z_t + dt · v₁                         # Euler predictor
v₂ = (x_pred(z*, t+dt) − z*) / (1 − t−dt) # velocity at predicted step
z_{t+dt} = z_t + dt · (v₁ + v₂) / 2       # Heun correction
```

### Why

- **x-prediction keeps the target on the data manifold.** ε-prediction asks
  the model to reconstruct unit Gaussian noise from a noisy image — the
  target lies off the image manifold at large patch sizes. Predicting x
  directly keeps the target in image space regardless of t.
- **Logit-normal t.** Uniform t wastes capacity at t≈0 (near-pure noise,
  trivial) and t≈1 (near-clean, also trivial). Logit-normal concentrates
  effort at intermediate t values where the model must reason about structure.
- **50 Heun steps > 1000 DDPM steps.** DDPM injects fresh noise every step to
  maintain a Markov chain. Flow matching follows a deterministic ODE, so a
  2nd-order solver achieves higher quality with far fewer function evaluations.
- **No `schedule_sampler` needed.** The DDPM-era importance-weighted discrete
  timestep sampler is gone; `t` is drawn inside `FlowMatching.training_loss()`.

---

## Change 5 — `y_emb` (class label) replaced by `cond_proj(h)`

**File:** `rcdm/jit.py` — `JiT.forward`

### What changed

```python
# Original JiT paper (class-conditional):
c = timestep_emb(t) + class_emb(y)     # y is an integer class label

# JiT-RCDM:
c = timestep_emb(t) + cond_proj(h)     # h is a continuous SSL vector
```

There is no `class_emb` embedding table. It is replaced entirely by
`ConditioningProjector`, which projects the continuous DinoV3 CLS token `h`
into the conditioning dimension before adding to the timestep embedding.

### Why

Messidor-2 carries severity grade labels (0–4), but using integer labels would
collapse all images within a grade into a single point in conditioning space —
losing the fine-grained variation between patients, cameras, and imaging
conditions within the same grade. The DinoV3 CLS token `h` is a *continuous*
signal: nearby `h` vectors correspond to perceptually similar images,
regardless of their grade label. This is the same design decision RCDM
originally made for the UNet, now carried forward into JiT.

---

## Change 6 — DinoV2-L replaced by DinoV3 ViT-S/16 (local checkpoint)

**File:** `rcdm/encoder.py`

> This change supersedes Change 1. DinoV2-L (remote, 1024-dim) is replaced
> by a local domain-specific DinoV3 ViT-S/16 (384-dim).

### What changed

```python
# OLD (Change 1 — DinoV2)
encoder = AutoModel.from_pretrained("facebook/dinov2-large")
# output: (B, 1024) — general-domain ViT-L

# NEW (DinoV3 — local)
DINOV3_CHECKPOINT = "checkpoints/dinov3_vits16_tmp"
encoder = AutoModel.from_pretrained(DINOV3_CHECKPOINT, local_files_only=True)
# output: (B, 384) — domain-specific ViT-S
```

`local_files_only=True` prevents any network request. The checkpoint directory
must contain `config.json` and `model.safetensors` before any encoder call.

A module-level constant `ENCODER_OUTPUT_DIM = 384` is exported so calling
code can reference the output dimension without hardcoding.

### Why

| Reason | Detail |
|---|---|
| **Domain specificity** | DinoV3 ViT-S/16 is trained or fine-tuned on medical/retinal imagery; its representations are more semantically aligned with Messidor-2 fundus images than a general-domain ViT-L |
| **Leaner conditioning** | 384-dim → 64-dim bottleneck (Change 7) is a 6× compression; from 1024-dim → 64 would be 16× |
| **Reproducibility** | A local checkpoint guarantees identical weights across runs and environments |

**Output dimension change:** 1024 → **384**. All defaults downstream updated.

---

## Change 7 — JiT conditioning bottleneck: cond_dim decoupled, set to 64

> **Superseded by Change 8** — the 64-dim bottleneck deviates from the JiT
> paper and was reverted. This entry documents the rationale for the interim
> design; Change 8 restores paper-faithful conditioning.

**Files:** `rcdm/jit.py`, `rcdm/conditioning.py`, `scripts/train.py`

An independent `cond_dim=64` parameter was introduced, compressing both the
timestep embedding and the DinoV3 projection to 64 dimensions before each
adaLN-Zero block. This created a 6× bottleneck (384→64) that is not present
in the original JiT paper.

---

## Change 8 — Restore paper-faithful conditioning: remove cond_dim bottleneck

**Files:** `rcdm/jit.py`, `rcdm/conditioning.py`, `scripts/train.py`, `scripts/sampling.py`

### What changed

The 64-dim conditioning bottleneck introduced in Change 7 is removed. The
conditioning dimension is always `hidden_dim` — exactly as specified in the
JiT paper (arxiv 2511.13720).

| Component | Change 7 (wrong) | Change 8 (paper-faithful) |
|---|---|---|
| Sinusoidal embedding dim | 64 | `hidden_dim` |
| `time_embed` MLP | Linear(64→256)→SiLU→Linear(256→64) | Linear(D→4D)→SiLU→Linear(4D→D) |
| `ConditioningProjector` output | 64 | `hidden_dim` |
| `c = t_emb + cond_proj(h)` shape | (B, 64) | (B, hidden_dim) |
| `JiTBlock` adaLN input | 64 | `hidden_dim` |
| `FinalLayer` adaLN input | 64 | `hidden_dim` |
| `--cond_dim` CLI arg | present | **removed** |

```python
# Conditioning path (final, paper-faithful)
sinusoidal(t, dim=hidden_dim)
  → time_embed  Linear(D → 4D) → SiLU → Linear(4D → D)        # (B, D)
  +
h (384-dim DinoV3)
  → cond_proj   Linear(384 → D) → SiLU                         # (B, D)
  = c                                                           # (B, D)

# Inside each JiTBlock
adaLN_modulation  Linear(D → 6·D)                              # D = hidden_dim
```

A shape assertion is added in `JiT.forward()` to catch regressions:

```python
assert c.shape == (B, self.hidden_dim)
```

Default `hidden_dim` is updated to 768 (JiT-B) and `num_heads` to 12
(ViT-B convention: 768 / 64 = 12 heads).

### Why

The 64-dim bottleneck was introduced without paper justification and reduces
the expressivity of the conditioning signal — the model has far fewer
parameters to route `h` information into each block's modulation. The JiT
paper specifies `c = timestep_emb(t) + class_emb(y)` where both live at
`hidden_dim`. Matching the paper exactly ensures the architecture is a valid
drop-in replacement for the class-conditional JiT with continuous SSL
conditioning.

---

## Change 9 — Local-training presets: JiT_S_16 and JiT_S_32

**Files:** `rcdm/jit.py`, `scripts/train.py`

### What changed

Two preset factory functions were added to `rcdm/jit.py` for use on machines
without an H100. Both target small training budgets (local GPU, Apple MPS, CPU):

| Preset | `hidden_dim` | `num_heads` | `patch_size` | `cond_dim` | Tokens @ 224px | Params |
|---|---|---|---|---|---|---|
| `JiT_S_16` | 384 | 6 | 16 | 128 | 196 | ~25 M |
| `JiT_S_32` | 384 | 6 | 32 | 128 | 49 | ~26 M |

```python
# rcdm/jit.py — new presets
def JiT_S_16(image_size=224, h_dim=384, **kwargs) -> JiT:
    return JiT(hidden_dim=384, num_heads=6, depth=12,
               patch_size=16, cond_dim=64, ...)

def JiT_S_32(image_size=224, h_dim=384, **kwargs) -> JiT:
    return JiT(hidden_dim=384, num_heads=6, depth=12,
               patch_size=32, cond_dim=64, ...)
```

To make presets selectable without re-entering every arch flag, `scripts/train.py`
gains a `--model` argument and a `_model_cfg()` helper that reads architecture
dimensions from the live model object (not from `args`) so that checkpoints are
always accurate even when a preset overrides the manual CLI flags:

```bash
# Fastest local run — use JiT_S_16
python scripts/train.py \
    --model        S16 \
    --reps_file    data/messidor2/train_reps.pt \
    --save_dir     checkpoints/ \
    --image_size   224 \
    --total_steps  5000 \
    --save_interval 1000 \
    --log_interval  100 \
    --device       mps

# Even fewer tokens (49 vs 196) — use JiT_S_32
python scripts/train.py --model S32 ...
```

`cond_dim` was also re-exposed as an optional parameter on `JiT`, `JiTBlock`,
and `create_jit_model` (default = `hidden_dim`, preserving the Change 8
paper-faithful no-bottleneck path). The presets pass `cond_dim=64` explicitly.

### Why

| Reason | Detail |
|---|---|
| **Width scaling** | `hidden_dim 384` vs 768 (JiT-B) cuts FFN and projection FLOPs by ~(384/768)² ≈ 4× per layer |
| **Patch-32 token reduction** | 49 tokens vs 196 means attention is ~(196/49)² ≈ 16× cheaper — the largest saving on MPS where attention is slow |
| **Head-dim invariant** | 384 / 6 = 64 head dim — same as the standard ViT-S recipe; no degradation from head-dim mismatch |
| **64-dim `cond_dim`** | Re-introduces the bottleneck only for the S presets. `hidden_dim=384` makes `cond_dim=hidden_dim` (no bottleneck) only 6× wider than 64, so the bottleneck is lighter here than it was at `hidden_dim=768` |
| **Checkpoint accuracy** | `_model_cfg()` reads `model.hidden_dim`, `model.cond_dim` etc. directly — a preset that silently overrides CLI args can no longer produce a checkpoint that lies about its own architecture |

---

## Complete list of modified / created files

| File | Change # | What changed |
|---|---|---|
| `rcdm/encoder.py` | 1, 6 | ResNet-50 → DinoV2 → DinoV3 local; output 2048→1024→384 |
| `rcdm/conditioning.py` | 2, 3, 8, **11** | ConditioningProjector output = hidden_dim; AdaLNZero added; bottleneck removed; **RMSNorm added; AdaLNZero norms switched to RMSNorm** |
| `rcdm/jit.py` | 3, 4, 5, 8, 9, 10, **11** | **New file.** JiT ViT denoiser with adaLN-Zero; FlowMatching utilities; `cond_dim` bottleneck; presets; CFG null-h dropout + two-pass sampler; **learnable null_h; 2D RoPE; SwiGLU; qk-norm; RMSNorm; logit-normal t fix** |
| `scripts/train.py` | 4, 8, 9, 10, **11** | UNet+DDPM → JiT+FlowMatching; h_dim 2048→384; `--model S16/S32` flag; `--cfg_dropout`; **EMA state saved in checkpoints; betas=(0.9,0.95); warmup; grad_accum** |
| `scripts/sampling.py` | 4, 8, 10, **11** | Inference script; `--cfg_scale` two-pass guidance; **EMA weights loaded; enc_transform hard-coded to 224** |
| `scripts/precompute_reps.py` | 1, 6 | Data dir defaults → Messidor-2; image_size 64→224; assertion 2048→384 |
| `guided_diffusion/guided_diffusion/script_util.py` | 1 | h_dim default 2048→1024 |

### Files unchanged (kept for backward compatibility)

| File | Note |
|---|---|
| `rcdm/dataset.py` | Loads `(image, h)` pairs generically; `h` dimension is inferred from the `.pt` file |
| `rcdm/conditioning.py` — `ConditionalBatchNorm2d` | Still imported by the legacy UNet path |
| `guided_diffusion/guided_diffusion/unet.py` | Legacy UNet + cBN remains functional |
| `guided_diffusion/guided_diffusion/gaussian_diffusion.py` | DDPM machinery retained; unused by the JiT training path |

---

## Data migration checklist

Any precomputed representation files created before these changes are **invalid**
and must be regenerated:

```bash
# 1. Place DinoV3 checkpoint at checkpoints/dinov3_vits16_tmp/
#    (requires config.json + model.safetensors)

# 2. Regenerate representations — output shape will be (N, 384)
python scripts/precompute_reps.py \
    --data_dir  data/messidor2/train \
    --out_file  data/messidor2/train_reps.pt \
    --image_size 224 \
    --device cuda

# 3a. Train with a preset (recommended for local machines)
python scripts/train.py \
    --model     S16 \
    --reps_file data/messidor2/train_reps.pt

# 3b. Train JiT-B (paper default: hidden_dim=768, no bottleneck)
python scripts/train.py \
    --reps_file data/messidor2/train_reps.pt \
    --h_dim 384
```

Old `.pt` files with shape `(N, 2048)` or `(N, 1024)` will be rejected by the
assertion in `precompute_reps.py` and will cause a shape mismatch in the model.

Checkpoints saved under Change 7 (with `cond_dim=64` in `model_cfg`) are
**incompatible** with the Change 8 architecture and must be retrained.

---

## Change 10 — Patch size fixed to 16 and classifier-free guidance (CFG)

**Files:** `rcdm/jit.py` (`FlowMatching`), `scripts/train.py`, `scripts/sampling.py`

---

### 10a — Patch size committed to 16

`patch_size=16` is set as the fixed configuration for all training going forward
(`--model S16`). The S32 preset (patch=32) is kept in the code for reference but
is no longer the recommended path.

#### Why patch=16, not 32

| Property | patch=16 | patch=32 |
|---|---|---|
| Tokens at 224×224 | 196 | 49 |
| Spatial resolution | 16 px/patch — fine vessel detail captured | 32 px/patch — coarse, loses thin vessels |
| Attention cost | 196² ≈ 38 k pairs | 49² ≈ 2.4 k pairs (~16× cheaper) |
| Best for | **Quality** — the goal | Speed on memory-limited hardware |

Retinal fundus images contain diagnostically critical fine structures —
micro-aneurysms (~10–20 px), thin vessel branches, optic disc margin. At
patch_size=32, a single patch covers 32×32 pixels: two neighbouring vessel
branches that are each 3–4 px wide can land in the same patch and become
indistinguishable to the model. At patch_size=16 each patch covers 16×16 px,
halving the spatial compression and preserving the signal needed to reconstruct
those structures.

The additional compute cost is acceptable: on Apple MPS a `JiT_S_16` step with
batch_size=4 at 224×224 fits in memory and runs in a practical time.

---

### 10b — Classifier-free guidance (CFG)

#### What changed

**Training** (`FlowMatching.training_loss`): a new `p_uncond` parameter
(default 0.1) randomly replaces `h` with a zero vector for a fraction of the
batch before the forward pass:

```python
# p_uncond fraction of samples train with h = 0 (null conditioning)
mask = torch.rand(B, device=device) < p_uncond
h = h.clone()
h[mask] = 0.0
loss = F.mse_loss(model(z_t, t, h=h), x)
```

`train.py` exposes this as `--cfg_dropout 0.1` (default active; pass 0.0 to
disable for a pure conditional baseline).

**Inference** (`FlowMatching.sample`): a new `cfg_scale` parameter (default 1.0
= no guidance) runs two forward passes per ODE step and extrapolates:

```python
x_pred_cond   = model(z, t, h=h)
x_pred_uncond = model(z, t, h=zeros_like(h))
x_pred = x_pred_uncond + cfg_scale * (x_pred_cond - x_pred_uncond)
```

The blending happens at the **x_pred level** (before velocity computation), not
at the velocity level. This is equivalent but numerically cleaner because
`x_pred` is bounded in image space while velocity `v = (x_pred − z)/(1−t)` can
be large near t=1.

`sampling.py` exposes `--cfg_scale` (default 3.0). Sensible range: 2.0–5.0.

#### Why CFG produces sharper, more faithful images

Without CFG, the model must cover the full conditional distribution
`p(x | h)` in a single forward pass. The learned score averages over all
modes consistent with `h`, which manifests as blurry or over-smoothed
outputs — the classic "regression to the mean" problem.

CFG extrapolates *away from the unconditional* prediction toward the conditional
one. Formally, it approximates sampling from a sharpened distribution:

```
p_cfg(x | h) ∝ p(x | h)^cfg_scale / p(x)^(cfg_scale − 1)
```

This increases the weight of image details that are specifically explained by `h`
(the patient's retinal structure) and down-weights features that appear
regardless of conditioning (generic texture, average lighting). The practical
result is that generated images reproduce the specific vascular pattern, optic
disc size, and pathology distribution of the conditioning image more faithfully.

#### Cost

Each ODE step now runs the model **twice** when `cfg_scale > 1.0`. With 50 Heun
steps (100 model evaluations per sample) this doubles to 200. The wall-clock
cost scales exactly 2× vs. no-guidance inference. Training cost is unchanged
(the conditional and null branches share a single batch forward pass with
masked `h`).

#### Recommended settings

| Use case | `--cfg_dropout` (train) | `--cfg_scale` (sample) |
|---|---|---|
| Strong guidance (retinal detail) | 0.10 | 3.0–5.0 |
| Balanced (fidelity + diversity) | 0.10 | 2.0 |
| Diversity / unconditional mix | 0.15 | 1.5 |
| Disable CFG entirely | 0.00 | 1.0 |

**Note:** A model trained with `cfg_dropout=0.0` cannot be used with
`cfg_scale > 1.0` at inference — it never learned the null-h distribution.
Always train with `cfg_dropout > 0` if you intend to use guidance at inference.

---

## Soundness review

A full cross-file audit was performed after Change 9. Findings below.

### y_emb / label_emb — status: dead code for the JiT path ✓

The original ADM UNet in `guided_diffusion/unet.py` constructs
`self.label_emb = nn.Embedding(num_classes, time_embed_dim)` when
`num_classes is not None`, and adds it to the timestep embedding in
`forward(x, timesteps, y=None)`.

**None of this is reachable from the JiT training path.**
`scripts/train.py` builds a `JiT` model via `create_jit_model()` and calls
`flow.training_loss(model, x, h)` — the UNet and `label_emb` are never
instantiated or called.

In the JiT model (`rcdm/jit.py`), `class_emb(y)` is replaced entirely by
`cond_proj(h)`, where `h` is the DinoV3 CLS token:

```python
# JiT.forward
t_emb = self.time_embed(timestep_embedding(t, self.cond_dim))   # (B, cond_dim)
c     = t_emb + self.cond_proj(h)                               # (B, cond_dim)
```

There is no `y` argument anywhere in `JiT.forward`. The substitution is complete.

### Flow-matching conditioning — mathematically verified ✓

Linear flow path: `z_t = t·x + (1−t)·ε`, t ∈ [0,1], t=0→pure noise, t=1→data.

Analytical velocity: `dz_t/dt = x − ε = (x − z_t)/(1−t)`.

The model predicts `x_pred = f_θ(z_t, t, h)`.
Velocity estimate: `v = (x_pred − z_t) / (1−t)` — exactly what the Heun sampler
uses. The ODE integrates forward from t=0 to t≈1.

Pure Euler at the last step avoids the `1/(1−t)` singularity at t=1. ✓

### Dual normalisation — correctly separated ✓

| Transform | Formula | Used for |
|---|---|---|
| Encoder (ImageNet) | mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225] | Computing `h` in `precompute_reps.py` and `sampling.py` |
| Diffusion | mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5] → [−1, 1] | Training images `x` in `dataset.py` and the conditioning thumbnail in `sampling.py` |

These are intentionally different and never mixed.

### `cond_dim` threading — consistent after Change 9 ✓

`cond_dim` flows from `JiT.__init__` → `time_embed` MLP → `cond_proj` →
`JiTBlock` → `AdaLNZero` → `FinalLayer`. All five components use `self.cond_dim`.

`sampling.py` reads `cfg.get("cond_dim")` which returns `None` for pre-Change-9
checkpoints → `JiT.__init__` maps `None → hidden_dim` (no bottleneck). ✓

`_model_cfg()` in `train.py` reads `model.cond_dim` directly from the live
model, so presets that override CLI flags are always recorded correctly. ✓

### Bugs fixed during this review

| File | Bug | Fix |
|---|---|---|
| `rcdm/dataset.py` | `image_size=64` default — silently resizes Messidor-2 to 64×64 | Changed default to `224` |
| `rcdm/dataset.py` | Docstring/comments say `(2048,)` | Updated to `(384,)` |
| `scripts/precompute_reps.py` | Module docstring, comments say DinoV2-L / (N,1024) / Tiny ImageNet | Updated to DinoV3 ViT-S/16 / (N,384) / Messidor-2 |
| `rcdm/conditioning.py` | `AdaLNZero` docstring says `1024 for ViT-L` | Updated to list S/B/L sizes |

---

## Image quality improvement roadmap

*What needs to change to produce better / more detailed images when generating
in the RCDM framework with a CLS-token h.*

The CLS token is a single 384-dim vector — a global semantic summary of the
image. This is the dominant bottleneck. The improvements below are ranked
by expected impact vs. implementation cost.

### 1 — Classifier-free guidance (CFG) ★★★ highest impact, low cost

Standard in all modern flow-matching / diffusion models. During training, drop
`h` to a zero vector (or a learned null embedding) with probability `p_uncond`
(typically 10–15 %):

```python
# in FlowMatching.training_loss
if random.random() < 0.1:
    h = torch.zeros_like(h)   # null conditioning
loss = F.mse_loss(model(z_t, t, h), x)
```

At inference, run two forward passes and extrapolate:

```python
x_pred_cond   = model(z, t, h)
x_pred_uncond = model(z, t, torch.zeros_like(h))
x_pred = x_pred_uncond + cfg_scale * (x_pred_cond - x_pred_uncond)
```

`cfg_scale` ∈ [1.5, 5.0]. Higher values increase sharpness/fidelity but reduce
diversity. This requires zero changes to the model architecture — only training
and sampling logic.

**Why it works:** at cfg_scale > 1, the model "overshoots" in the direction of
the conditioning signal, trading diversity for per-sample quality. The CLS-token
conditioning signal is amplified beyond what gradient descent alone achieves.

### 2 — Deeper conditioning projector ★★ moderate impact, trivial cost

`ConditioningProjector` is currently `Linear(384 → cond_dim) + SiLU` — a single
layer. A 2-layer MLP can better align the encoder's representation space with
the diffusion model's internal space:

```python
self.proj = nn.Sequential(
    nn.Linear(h_dim, h_dim * 2),
    nn.SiLU(),
    nn.Linear(h_dim * 2, cond_dim),
    nn.SiLU(),
)
```

No shape changes required anywhere else.

### 3 — Spatial cross-attention to DinoV3 patch tokens ★★★ highest structural impact

The CLS token discards spatial layout. DinoV3 produces one patch token per 16×16
pixel region (196 tokens for 224×224). Cross-attending the JiT's denoising tokens
to those 196 encoder patch tokens gives the model explicit per-region cues:
where vessels are, where the optic disc is, etc.

```
# In JiTBlock.forward — add a cross-attention layer
encoder_tokens : (B, 196, 384)  ← all DinoV3 patch tokens, not just CLS
x = x + gate_cross · CrossAttn(Q=x, KV=encoder_tokens)
```

This requires:
- Passing the full encoder sequence (not just CLS) through the pipeline
- Adding one `nn.MultiheadAttention` per block for cross-attention
- `precompute_reps.py` must store `last_hidden_state[:, 1:, :]` (196 × 384) in
  addition to or instead of the CLS token
- Significant memory increase (~8× more conditioning tokens)

This is the single change that produces the most visually detailed outputs in
comparable architectures (IP-Adapter, ControlNet, cross-attention DiT).

### 4 — EMA (Exponential Moving Average) of model weights ★★ moderate impact

Sample from a shadow EMA copy of the model weights rather than the live weights.
EMA smooths training noise and consistently produces sharper samples.

```python
ema_model = copy.deepcopy(model)
# after each optimiser.step():
for ema_p, p in zip(ema_model.parameters(), model.parameters()):
    ema_p.data.mul_(ema_decay).add_(p.data, alpha=1 - ema_decay)
# sample from ema_model, not model
```

`ema_decay = 0.9999` is standard. Adds one full model copy to memory.

### 5 — LR warmup + cosine decay schedule ★ small but reliable gain

Flow-matching models benefit from a short linear warmup (500–1000 steps) followed
by cosine decay to 0. Currently `train.py` uses a fixed LR with AdamW.

```python
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimiser, max_lr=args.lr,
    total_steps=args.total_steps, pct_start=0.01,
)
```

### 6 — Higher resolution ★ costly but straightforward

Training at 256×256 or 512×512 directly increases fine-detail capacity. Requires
`patch_size=16` (not 32) and significantly more VRAM. For MPS/local training,
256×256 with `JiT_S_16` is feasible if batch_size is reduced to 4.

### Summary table

| Change | Impact | Code complexity | Architecture change? |
|---|---|---|---|
| Classifier-free guidance | ★★★ | Low — training + sampling only | No |
| Deeper conditioning projector | ★★ | Trivial | No |
| Spatial cross-attention (patch tokens) | ★★★ | High | Yes |
| EMA weights | ★★ | Low | No |
| LR warmup / cosine decay | ★ | Trivial | No |
| Higher resolution (256/512) | ★ | Trivial (change args) | No |

**Recommended first step for a quick quality boost:** implement CFG (#1).
It requires no architecture changes, works with any existing checkpoint after
retraining with null-conditioning dropout, and is the de-facto standard in all
state-of-the-art flow-matching / diffusion models.

---

## Change 11 — Soundness fixes: EMA, encoder resize, learnable null-h, JiT training recipe, architecture alignment

**Files:** `scripts/train.py`, `scripts/sampling.py`, `rcdm/jit.py`, `rcdm/conditioning.py`

A systematic cross-file audit identified five deviations from the published JiT
and RCDM designs. All five are fixed here. The changes are tagged in the source
with `# ── JiT-RCDM [fix-N]: description ──` comments so they can be located
without reading this document.

We follow the principle: *change as little as possible from the original
[facebookresearch/RCDM](https://github.com/facebookresearch/RCDM) and
[LTH14/JiT](https://github.com/LTH14/JiT) implementations.* Every deviation is
documented here.

---

### What was taken directly from the original repos

#### From [facebookresearch/RCDM](https://github.com/facebookresearch/RCDM)

| Component | File in RCDM | How used here |
|---|---|---|
| `RepresentationDataset` concept | `dataset.py` | `rcdm/dataset.py` — loads `(image, h)` pairs; image_size default updated to 224 |
| `ConditioningProjector` structure | `guided_diffusion/condition_helper.py` | `rcdm/conditioning.py` — `Linear(h_dim, cond_dim) + SiLU` |
| `ConditionalBatchNorm2d` | `guided_diffusion/condition_helper.py` | Kept for UNet backward compat; not used in JiT path |
| Classifier-free guidance (null-h dropout) | `scripts/image_train.py` | `FlowMatching.training_loss` — `p_uncond` fraction replaces `h` with null vector |
| Dual normalisation (encoder ImageNet vs diffusion [−1,1]) | implicit in RCDM data pipeline | `rcdm/dataset.py` + `rcdm/encoder.py` |
| Frozen encoder (no gradients) | `guided_diffusion/condition_helper.py` | `rcdm/encoder.py` — `encoder.eval(); requires_grad=False` |

#### From [LTH14/JiT](https://github.com/LTH14/JiT)

| Component | File in JiT | How used here |
|---|---|---|
| `JiT` ViT denoiser class | `denoiser.py` | `rcdm/jit.py` — `JiT` class; block structure, patch embed, final layer |
| `AdaLNZero` conditioning | `denoiser.py` | `rcdm/conditioning.py` + `rcdm/jit.py` — 6-param modulation, zero-init output |
| Sinusoidal timestep embedding | `denoiser.py` | `rcdm/jit.py` — `timestep_embedding()` |
| `time_embed` MLP (Linear→SiLU→Linear) | `denoiser.py` | `rcdm/jit.py` — width `hidden_dim → 4·hidden_dim → hidden_dim` |
| Heun ODE sampler | `sample.py` | `rcdm/jit.py` — `FlowMatching.sample`; 50 steps; pure Euler at last step |
| CFG two-pass x-pred blending | `sample.py` | `rcdm/jit.py` — blending at `x_pred` level (not velocity) |
| `EMA` shadow-weight pattern | `denoiser.py` | `scripts/train.py` — `EMA` class; `apply_shadow` / `restore` |

---

### Fix 1 — EMA not persisted in checkpoints `[fix-1]`

**Problem:** The JiT paper (Tab. 9) ablates EMA at decay=0.9999 and reports it as
the single highest-impact training improvement. The EMA class existed and weights
were updated during training, but the shadow weights were never saved to the
checkpoint dict, so resuming from a checkpoint discarded all EMA history.
`sampling.py` also loaded raw model weights instead of the EMA shadow.

**Original JiT behaviour:** The JiT reference code writes EMA state to disk and
loads it at inference; samples are always drawn from the EMA model.

**Fix:**

In `scripts/train.py` — both the periodic checkpoint and the final save now include:
```python
"ema": ema.state_dict(),   # ── JiT-RCDM [fix-1]
```

In `scripts/sampling.py` — `load_model` now applies EMA shadow weights after
loading the raw model:
```python
ema_state = state.get("ema")
if ema_state is not None:
    shadow = ema_state.get("shadow", {})
    for name, param in model.named_parameters():
        if name in shadow:
            param.data.copy_(shadow[name].to(device))
```

Old checkpoints (no `"ema"` key) fall through gracefully — raw weights are used
and a warning is printed. This makes the fix backward-compatible.

---

### Fix 2 — DinoV3 encoder fed wrong image size in `sampling.py` `[fix-2]`

**Problem:** `sampling.py` built the encoder transform with
`build_transform(image_size=image_size)` where `image_size` came from the
*generative model's* checkpoint config (e.g. 256 or 512). DINOv3 ViT-S/16 uses
fixed sinusoidal position embeddings for a 14×14 patch grid (`224 / 16 = 14`);
feeding it any other resolution silently corrupts the CLS token.

**Original RCDM behaviour:** RCDM always ran the ResNet-50 encoder at the same
resolution as the generative model (both 64×64 in the Tiny ImageNet experiments).
This was coincidentally correct. The JiT path has two independent image sizes —
the encoder's fixed 224 px requirement vs. the generative model's configurable
`image_size` — which the original code conflated.

**Fix:** Hard-coded to 224 px regardless of `cfg["image_size"]`:
```python
# ── JiT-RCDM [fix-2]: always 224 for DinoV3 ViT-S/16 ──
enc_transform = build_transform(image_size=224)
```

---

### Fix 3 — Null-h should be a learnable parameter `[fix-3]`

**Problem:** CFG null conditioning used a literal `torch.zeros_like(h)` as the
unconditional input. A hard-coded zero vector forces the null branch to share the
same embedding as the mean of all conditioning vectors that happen to have zero
components — an arbitrary coincidence, not a learned representation of "no
conditioning."

**Original JiT behaviour (class-conditional):** JiT uses a learned `null_class`
embedding. RCDM's original null conditioning for continuous `h` is not defined;
using `h=0` is a common but suboptimal approximation.

**Fix:** `JiT.__init__` now registers a learnable parameter:
```python
# ── JiT-RCDM [fix-3]: learnable null-h ──
self.null_h = nn.Parameter(torch.zeros(h_dim))
```

`FlowMatching.training_loss` uses `torch.where` to substitute it:
```python
null_h      = model.null_h.unsqueeze(0).expand(B, -1)
h_conditioned = torch.where(mask.unsqueeze(1), null_h, h)
```

`FlowMatching.sample` uses `model.null_h.expand(B, -1)` instead of
`torch.zeros_like(h)` for the unconditioned pass.

Backward compatibility: when `null_h` is absent (old checkpoints), both methods
fall back to `torch.zeros_like(h)`.

---

### Fix 4 — JiT training recipe deviations `[fix-4a/b/c/d]`

Four training hyperparameter deviations from the JiT paper:

#### 4a — AdamW β₂: 0.999 → 0.95

**Problem:** PyTorch's `AdamW` default `betas=(0.9, 0.999)`. The JiT paper
(and most ViT diffusion papers following it) uses `betas=(0.9, 0.95)`. A higher
β₂ makes the second moment estimate sluggish, causing over-damped LR updates at
the start of training and under-damped updates when the loss landscape changes.

**Fix:** `optimiser = torch.optim.AdamW(..., betas=(0.9, 0.95))`.

#### 4b — Logit-normal t-sampler: μ=0 → μ=−0.8, σ=1 → σ=0.8

**Problem:** The implemented sampler used `sigmoid(N(0, 1))`. The JiT paper
(Tab. 3) reports that `logit-normal(μ=−0.8, σ=0.8)` gives the best FID, shifting
more training steps to intermediate timesteps (t ≈ 0.3–0.6) where the model must
reason about image structure, rather than wasting budget on near-noise (t≈0) and
near-clean (t≈1) extremes.

**Fix:**
```python
# ── JiT-RCDM [fix-4b]: μ=−0.8, σ=0.8 per JiT paper Tab. 3 ──
u = -0.8 + 0.8 * torch.randn(B, device=device)
t = torch.sigmoid(u)
```

#### 4c — LR warmup absent

**Problem:** ViT-based models routinely use a short linear warmup before reaching
full LR. Without warmup, the first steps fire large gradients into randomly
initialised attention weights, creating bad attractors that slow convergence.

**Original JiT behaviour:** JiT uses 5 epochs of linear LR warmup.

**Fix:** `_lr_at_step(step, warmup_steps, base_lr)` in `train.py` implements
linear warmup; the default is 1000 steps (`--warmup_steps 1000`). The LR is set
on the optimizer group directly at each step (no scheduler object needed).

#### 4d — Gradient accumulation absent

**Problem:** Effective batch size of 8 is too small for flow-matching on 224×224
images — gradient variance is high and the null-h branch sees only ~0.8 samples
per step (10 % of 8). The JiT paper trains with batch 256–512.

**Fix:** `--grad_accum N` divides each loss by N before `.backward()` and calls
`optimizer.step()` once every N micro-steps. This simulates a batch of
`batch_size × grad_accum` without additional VRAM. The CLI default is 1 (no
accumulation) so existing run commands are unaffected.

---

### Fix 5 — Architecture deviations from JiT `[fix-5a/b/c/d]`

Four components in `rcdm/jit.py` diverged from the JiT ViT specification:

#### 5a — LayerNorm → RMSNorm

**Problem:** The JiT denoiser uses RMSNorm (Zhang & Sennrich 2019) throughout.
The implementation used `nn.LayerNorm`, which computes both mean-centering and
RMS scaling. RMSNorm drops the mean-centering step; it is faster, numerically
more stable, and is used by every major ViT-based diffusion model that followed
the JiT paper.

**Fix:**
- `RMSNorm` class added to `rcdm/conditioning.py` and imported by `jit.py`.
- `AdaLNZero.norm1`, `norm2` changed to `RMSNorm(hidden_dim, affine=False)`.
- `FinalLayer.norm_final` changed to `RMSNorm(hidden_dim, affine=False)`.
- Q/K norms in `Attention` use `RMSNorm(head_dim, affine=True)` (fix-5c).

```python
# ── JiT-RCDM [fix-5a]: RMSNorm throughout ──
class RMSNorm(nn.Module):
    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm * self.weight).to(x.dtype)
```

#### 5b — Learned positional embedding → 2D RoPE

**Problem:** The original code used a learned `nn.Parameter` positional embedding
added to patch tokens in `PatchEmbed.forward`. JiT uses 2D Rotary Position
Embedding (RoPE, Su et al. 2021) applied to the query and key matrices inside
each attention block. Learned absolute pos-embeds generalise poorly to image
sizes not seen during training; RoPE provides relative position encoding that
extrapolates naturally.

**Original JiT behaviour:** RoPE frequencies are precomputed once as a buffer at
model construction time and applied inside each attention block.

**Fix:**
- `PatchEmbed` no longer adds a positional embedding (`pos_embed` parameter
  removed).
- `compute_2d_rope_freqs(grid_size, head_dim)` is added to `rcdm/jit.py` —
  returns a `(N, head_dim//2)` complex tensor of rotation frequencies.
- `apply_rotary_emb(x, freqs_cis)` rotates the last `head_dim` of Q and K.
- `freqs_cis` is registered as a buffer on `JiT` and passed to every
  `JiTBlock.forward`.

#### 5c — No qk-norm

**Problem:** JiT applies per-head RMSNorm to Q and K before the dot-product
attention (`qk_norm`). Without qk-norm, attention logits can diverge early in
training when the model learns large-magnitude representations, causing gradient
spikes and slow convergence.

**Fix:** The custom `Attention` class applies `self.q_norm` and `self.k_norm`
(both `RMSNorm(head_dim, affine=True)`) to the reshaped Q and K tensors before
calling `F.scaled_dot_product_attention`.

```python
# ── JiT-RCDM [fix-5c]: qk-norm ──
q = self.q_norm(q)   # (B, heads, N, head_dim)
k = self.k_norm(k)
```

#### 5d — GELU FFN → SwiGLU

**Problem:** The original `JiTBlock` used
`nn.Sequential(Linear, GELU, Linear)` for the feed-forward network. JiT uses
SwiGLU (Shazeer 2020): a gated linear unit where the gate is computed by a
separate projection. SwiGLU requires no bias and its inner dimension is
`round_to_256(hidden_dim × mlp_ratio × 2/3)` to maintain the same parameter
count as a standard FFN at the same `mlp_ratio`.

**Fix:** A `SwiGLU` module is added to `rcdm/jit.py`:
```python
# ── JiT-RCDM [fix-5d]: SwiGLU FFN, no bias ──
class SwiGLU(nn.Module):
    def forward(self, x):
        x1, x2 = self.fc1(x).chunk(2, dim=-1)
        return self.proj(F.silu(x1) * x2)
```
`fc1` outputs `2 × inner_dim` (both gate and value), `proj` maps back to
`hidden_dim`. All Linear layers have `bias=False`, consistent with JiT.

---

### Backward compatibility

Old checkpoints (pre-fix-5) used `attn.in_proj_weight` (nn.MultiheadAttention)
and had a `pos_embed` parameter. The new architecture uses `attn.qkv.weight` and
no `pos_embed`. These checkpoints are **incompatible** and must be retrained.
No attempt at weight surgery is made; training from scratch at 50 k+ steps is
the correct path.

Old checkpoints that pre-date fixes 1–4 but used the same architecture are
compatible: the new keys (`null_h`, `freqs_cis` buffer) will be absent and
fall back to zero-h CFG gracefully.

---

### Updated "Final state at a glance" table

| Component | Original RCDM/JiT | JiT-RCDM (after fix-5) |
|---|---|---|
| **Normalisation** | `nn.LayerNorm` | `RMSNorm` (Zhang & Sennrich 2019) |
| **Positional encoding** | Learned absolute `pos_embed` | 2D RoPE (Su et al. 2021) |
| **FFN** | Linear → GELU → Linear | SwiGLU (Shazeer 2020); no bias |
| **Attention** | `nn.MultiheadAttention` | Custom MHA with qk-norm + RoPE |
| **AdamW β₂** | 0.999 (PyTorch default) | **0.95** (JiT paper) |
| **t-sampler** | logit-normal(0, 1) | **logit-normal(−0.8, 0.8)** (JiT Tab. 3) |
| **LR schedule** | Fixed LR | Linear warmup (`--warmup_steps 1000`) |
| **Grad accumulation** | Not supported | `--grad_accum N` |
| **EMA** | Updated but not saved | Saved to checkpoint; loaded at inference |
| **Null-h** | `torch.zeros_like(h)` | Learnable `nn.Parameter(zeros(h_dim))` |
