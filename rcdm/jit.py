"""
rcdm/jit.py

JiT (ViT-based) denoiser + flow-matching utilities for JiT-RCDM.

Architecture overview
---------------------
  PatchEmbed      : image (B, C, H, W) → token sequence (B, N, hidden_dim)
                    [fix-5b] No learned pos_embed — 2D RoPE is applied inside Attention.
  Attention       : [fix-5c] Custom MHA with QK-RMSNorm and 2D RoPE.
  JiTBlock        : Transformer block with adaLN-Zero conditioning.
                    [fix-5a] RMSNorm replaces LayerNorm (via AdaLNZero).
                    [fix-5d] SwiGLU FFN replaces GELU FFN.
  FinalLayer      : token sequence → patch pixels (B, N, patch_size²·C)
                    [fix-5a] RMSNorm replaces LayerNorm.
  JiT             : full model — forward(z_t, t, h) → x_pred
                    [fix-3]  learnable null_h parameter for CFG.
                    [fix-5b] 2D RoPE buffer (freqs_cis) registered on the model.

Conditioning
------------
  h (h_dim DinoV3 CLS token)
    → ConditioningProjector  Linear(h_dim → cond_dim) + SiLU
    → (B, cond_dim)
  sinusoidal(t, cond_dim)
    → time_embed MLP  Linear(cond_dim → 4·cond_dim) → SiLU
                    → Linear(4·cond_dim → cond_dim)
    → (B, cond_dim)
  c = time_embed_out + cond_proj_out               (B, cond_dim)

  Each JiTBlock: adaLN MLP  Linear(cond_dim → 6·hidden_dim) → 6 params.
  FinalLayer:    adaLN MLP  Linear(cond_dim → 2·hidden_dim) → shift, scale.

  cond_dim == hidden_dim gives the paper-faithful "no bottleneck" path.
  cond_dim < hidden_dim (e.g. 64) adds a regularising bottleneck between
  the encoder representation and every transformer block.

Preset factory functions
------------------------
  JiT_S_16   hidden=384, heads=6, cond=128, patch=16  (~25 M params at 224px)
  JiT_S_32   hidden=384, heads=6, cond=128, patch=32  (~25 M, 4× fewer tokens)

Flow-matching utilities
-----------------------
  FlowMatching.training_loss  : x-prediction MSE with logit-normal t sampling
                                [fix-4b] mu=-0.8, sigma=0.8 (JiT paper Tab. 3)
                                [fix-3]  uses learnable null_h for CFG dropout
  FlowMatching.sample         : 50-step Heun ODE solver from noise to image
                                [fix-3]  uses learnable null_h for unconditional pass

Paper references
----------------
  JiT               : arxiv 2511.13720  https://github.com/LTH14/JiT
  RMSNorm           : Zhang & Sennrich 2019 (arxiv 1910.07467)
  SwiGLU            : Shazeer 2020 (arxiv 2002.05202)
  RoPE              : Su et al. 2021 (arxiv 2104.09864)
  2D RoPE for ViTs  : adopted from LLaMA / MAR / JiT codebase
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conditioning import AdaLNZero, ConditioningProjector, RMSNorm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Sinusoidal embedding for a continuous scalar t ∈ [0, 1].

    We scale t to [0, 1000] to match the frequency range sinusoidal
    embeddings were designed for in discrete DDPM.  This gives the MLP a
    rich, smooth signal across the full range of flow-matching timesteps.

    Args:
        t   : (B,) float tensor, values in [0, 1]
        dim : embedding dimensionality (must be even)

    Returns:
        (B, dim) sinusoidal embedding
    """
    assert dim % 2 == 0, "timestep embedding dim must be even"
    half = dim // 2
    freqs = torch.pow(
        10000.0,
        -torch.arange(half, dtype=torch.float32, device=t.device) / half,
    )
    args = t[:, None].float() * 1000.0 * freqs[None]   # (B, half)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, dim)


# ---------------------------------------------------------------------------
# ── JiT-RCDM [fix-5b]: 2D RoPE helper functions ──
# ---------------------------------------------------------------------------

def compute_2d_rope_freqs(grid_size: int, head_dim: int, theta: float = 10000.0) -> torch.Tensor:
    """
    Precompute 2D RoPE complex frequencies for a square (grid_size × grid_size) patch grid.
    Each token at position (r, c) gets row-RoPE applied to the first half of head_dim
    and col-RoPE applied to the second half.
    Returns: (N, head_dim//2) complex64, where N = grid_size^2.
    """
    assert head_dim % 4 == 0, "head_dim must be divisible by 4 for 2D RoPE"
    quarter = head_dim // 4          # each spatial direction uses head_dim/4 complex dims

    freqs = 1.0 / (theta ** (torch.arange(quarter, dtype=torch.float32) / quarter))

    row_pos = torch.arange(grid_size, dtype=torch.float32)
    col_pos = torch.arange(grid_size, dtype=torch.float32)

    row_emb = torch.outer(row_pos, freqs)   # (G, Q)
    col_emb = torch.outer(col_pos, freqs)   # (G, Q)

    # Tile to (G*G, Q): each patch (r, c) gets row_emb[r] and col_emb[c]
    row_emb = row_emb.unsqueeze(1).expand(-1, grid_size, -1).reshape(grid_size**2, quarter)
    col_emb = col_emb.unsqueeze(0).expand(grid_size, -1, -1).reshape(grid_size**2, quarter)

    freqs_2d = torch.cat([row_emb, col_emb], dim=-1)   # (N, head_dim//2)
    return torch.polar(torch.ones_like(freqs_2d), freqs_2d)   # (N, head_dim//2) complex64


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    Apply 2D RoPE to queries or keys.
    x         : (B, N, heads, head_dim)
    freqs_cis : (N, head_dim//2) complex
    Returns   : (B, N, heads, head_dim)
    """
    B, N, H, D = x.shape
    x_c = x.float().reshape(B, N, H, D // 2, 2)
    x_c = torch.view_as_complex(x_c.contiguous())        # (B, N, H, D//2) complex
    freqs = freqs_cis[:N].unsqueeze(0).unsqueeze(2)       # (1, N, 1, D//2)
    x_out = torch.view_as_real(x_c * freqs)               # (B, N, H, D//2, 2)
    return x_out.reshape(B, N, H, D).to(x.dtype)


# ---------------------------------------------------------------------------
# ── JiT-RCDM [fix-5c]: Custom Attention with QK-norm and 2D RoPE ──
# ---------------------------------------------------------------------------

class Attention(nn.Module):
    """
    Multi-head self-attention with QK-norm and 2D RoPE (JiT paper recipe).
    Replaces nn.MultiheadAttention.
    """

    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = hidden_dim // num_heads
        self.qkv  = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)
        self.proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        # ── JiT-RCDM [fix-5c]: per-head RMSNorm on Q and K ──
        self.q_norm = RMSNorm(self.head_dim, affine=True)
        self.k_norm = RMSNorm(self.head_dim, affine=True)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)          # each (B, N, heads, head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)

        # ── JiT-RCDM [fix-5b]: apply 2D RoPE ──
        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        q = q.transpose(1, 2)            # (B, heads, N, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(B, N, D)
        return self.proj(out)


# ---------------------------------------------------------------------------
# ── JiT-RCDM [fix-5d]: SwiGLU FFN replaces GELU FFN ──
# ---------------------------------------------------------------------------

class SwiGLU(nn.Module):
    """
    SwiGLU feed-forward network (Shazeer 2020) as used in JiT.
    inner_dim = round_to_256(hidden_dim * mlp_ratio * 2/3) to keep
    parameter count comparable to a standard GELU FFN at the same ratio.
    No bias: JiT recipe.
    """

    def __init__(self, hidden_dim: int, mlp_ratio: int = 4):
        super().__init__()
        raw   = int(hidden_dim * mlp_ratio * 2 / 3)
        inner = ((raw + 255) // 256) * 256              # round up to nearest 256
        self.fc1  = nn.Linear(hidden_dim, inner, bias=False)
        self.fc2  = nn.Linear(hidden_dim, inner, bias=False)
        self.proj = nn.Linear(inner, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(F.silu(self.fc1(x)) * self.fc2(x))


# ---------------------------------------------------------------------------
# Patch embedding
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    """
    Patchify an image into a flat sequence of linear projections.

    image (B, C, H, W)  →  tokens (B, N, hidden_dim)
    where N = (H // patch_size) * (W // patch_size).

    [fix-5b] Learned positional embedding removed; 2D RoPE is applied
    inside the Attention module via the freqs_cis buffer on JiT.
    """

    def __init__(self, image_size: int, patch_size: int, in_channels: int, hidden_dim: int):
        super().__init__()
        assert image_size % patch_size == 0, (
            f"image_size {image_size} must be divisible by patch_size {patch_size}"
        )
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_channels, hidden_dim,
            kernel_size=patch_size, stride=patch_size,
        )
        # ── JiT-RCDM [fix-5b]: pos_embed removed; RoPE replaces it ──

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)                        # (B, D, H/p, W/p)
        x = x.flatten(2).transpose(1, 2)        # (B, N, D)
        # ── JiT-RCDM [fix-5b]: no pos_embed added; RoPE applied in Attention ──
        return x


# ---------------------------------------------------------------------------
# Final output layer
# ---------------------------------------------------------------------------

class FinalLayer(nn.Module):
    """
    Maps the last token sequence back to patch pixels.

    Uses adaLN (shift + scale, no gate) driven by c (shape cond_dim),
    then a zero-initialised linear projection to patch pixels.

    [fix-5a] Uses RMSNorm instead of LayerNorm for norm_final.
    """

    def __init__(self, hidden_dim: int, patch_size: int, out_channels: int, cond_dim: int):
        super().__init__()
        # ── JiT-RCDM [fix-5a]: RMSNorm instead of LayerNorm ──
        self.norm_final = RMSNorm(hidden_dim, affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_dim, patch_size * patch_size * out_channels)
        # c is cond_dim-dimensional → 2 * hidden_dim modulation params
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 2 * hidden_dim),
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # c: (B, cond_dim)
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = (1 + scale.unsqueeze(1)) * self.norm_final(x) + shift.unsqueeze(1)
        return self.linear(x)


# ---------------------------------------------------------------------------
# Transformer block with adaLN-Zero
# ---------------------------------------------------------------------------

class JiTBlock(nn.Module):
    """
    Standard ViT Transformer block with adaLN-Zero conditioning.

    The 6 modulation scalars come from a per-block adaLN MLP:
        Linear(cond_dim → 6·hidden_dim)
    cond_dim == hidden_dim gives the paper-faithful path; cond_dim < hidden_dim
    adds a conditioning bottleneck.

    Structure:
        x ← x + gate_attn · Attn( (1+scale_a)·RMSNorm(x) + shift_a )   [fix-5a,5b,5c]
        x ← x + gate_ffn  · SwiGLU( (1+scale_f)·RMSNorm(x) + shift_f ) [fix-5a,5d]

    Args:
        hidden_dim : token dimension D
        num_heads  : attention heads (D % num_heads == 0)
        mlp_ratio  : FFN expansion ratio (default 4)
        cond_dim   : dimension of the shared conditioning vector c (default = hidden_dim)
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads:  int,
        mlp_ratio:  int = 4,
        cond_dim:   int = None,
    ):
        super().__init__()
        cond_dim = cond_dim or hidden_dim
        self.ada = AdaLNZero(hidden_dim, cond_dim=cond_dim)
        # ── JiT-RCDM [fix-5c]: custom Attention with QK-norm replaces nn.MultiheadAttention ──
        self.attn = Attention(hidden_dim, num_heads)
        # ── JiT-RCDM [fix-5d]: SwiGLU replaces GELU FFN ──
        self.ffn = SwiGLU(hidden_dim, mlp_ratio)

    def forward(self, x: torch.Tensor, c: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x         : token sequence  (B, N, hidden_dim)
            c         : fused condition  (B, cond_dim)
            freqs_cis : 2D RoPE freqs   (N, head_dim//2) complex  [fix-5b]
        Returns:
            x : (B, N, hidden_dim)
        """
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = self.ada.forward_pre(x, c)

        # Attention branch
        x_normed = self.ada.modulate(self.ada.norm1(x), shift_a, scale_a)
        # ── JiT-RCDM [fix-5b,5c]: pass freqs_cis to custom Attention ──
        attn_out = self.attn(x_normed, freqs_cis)
        x = x + gate_a.unsqueeze(1) * attn_out

        # FFN branch
        x_normed = self.ada.modulate(self.ada.norm2(x), shift_f, scale_f)
        x = x + gate_f.unsqueeze(1) * self.ffn(x_normed)

        return x


# ---------------------------------------------------------------------------
# Full JiT model
# ---------------------------------------------------------------------------

class JiT(nn.Module):
    """
    JiT denoiser: plain ViT conditioned via adaLN-Zero.

    Follows the JiT paper (arxiv 2511.13720). class_emb(y) is replaced by a
    continuous cond_proj(h) where h is the DinoV3 CLS token.

    Fixes applied relative to the original codebase:
      [fix-3]  Learnable null_h parameter for classifier-free guidance.
      [fix-4b] Logit-normal t-sampler: mu=-0.8, sigma=0.8 (JiT Tab. 3).
      [fix-5a] RMSNorm replaces LayerNorm in AdaLNZero and FinalLayer.
      [fix-5b] 2D RoPE replaces learned positional embedding.
      [fix-5c] Custom Attention with per-head QK-RMSNorm.
      [fix-5d] SwiGLU FFN replaces GELU FFN.

    cond_dim controls the width of the shared conditioning signal c.
    Setting cond_dim == hidden_dim (default) gives the paper-faithful path.
    Setting cond_dim < hidden_dim (e.g. 128) adds a regularising bottleneck —
    see preset factory functions JiT_S_16 / JiT_S_32.

    Inputs:
        z_t : (B, C, H, W)   noisy image at flow-matching time t
        t   : (B,)            continuous timestep in [0, 1]
        h   : (B, h_dim)      DinoV3 CLS-token representation (384 for ViT-S)

    Output:
        x_pred : (B, C, H, W)  predicted *clean* image (x-prediction)

    Conditioning chain:
        1. sinusoidal(t, cond_dim) → time_embed MLP → (B, cond_dim)
        2. h → cond_proj Linear(h_dim→cond_dim) + SiLU → (B, cond_dim)
        3. c = time_embed_out + cond_proj_out              (B, cond_dim)
        4. Each JiTBlock: adaLN MLP(cond_dim → 6·hidden_dim) → 6 params
        5. FinalLayer:    adaLN MLP(cond_dim → 2·hidden_dim) → shift, scale

    Args:
        image_size  : spatial size of input/output images (square)
        patch_size  : patch size in pixels (image_size % patch_size == 0)
        in_channels : image channels (3 for RGB fundus)
        hidden_dim  : ViT token dimension D
        depth       : number of JiTBlocks
        num_heads   : attention heads (hidden_dim % num_heads == 0)
        mlp_ratio   : FFN expansion ratio
        h_dim       : encoder CLS token dim (384 for DinoV3 ViT-S/16)
        cond_dim    : width of the shared conditioning vector c (default = hidden_dim)
    """

    def __init__(
        self,
        image_size:  int = 224,
        patch_size:  int = 16,
        in_channels: int = 3,
        hidden_dim:  int = 768,
        depth:       int = 12,
        num_heads:   int = 12,
        mlp_ratio:   int = 4,
        h_dim:       int = 384,
        cond_dim:    int = None,
    ):
        super().__init__()
        self.image_size  = image_size
        self.patch_size  = patch_size
        self.in_channels = in_channels
        self.hidden_dim  = hidden_dim
        self.cond_dim    = cond_dim or hidden_dim   # bottleneck width (== hidden_dim → no bottleneck)

        self.patch_embed = PatchEmbed(image_size, patch_size, in_channels, hidden_dim)

        # Timestep MLP: sinusoidal(cond_dim) → cond_dim.
        self.time_embed = nn.Sequential(
            nn.Linear(self.cond_dim, self.cond_dim * 4),
            nn.SiLU(),
            nn.Linear(self.cond_dim * 4, self.cond_dim),
        )

        # Projects DinoV3 CLS token h → cond_dim.
        self.cond_proj = ConditioningProjector(h_dim=h_dim, cond_dim=self.cond_dim)

        # ── JiT-RCDM [fix-3]: learnable null_h for unconditional CFG pass ──
        self.null_h = nn.Parameter(torch.zeros(h_dim))

        # Each block receives the cond_dim-dimensional c
        self.blocks = nn.ModuleList([
            JiTBlock(hidden_dim, num_heads, mlp_ratio, cond_dim=self.cond_dim)
            for _ in range(depth)
        ])

        self.final_layer = FinalLayer(hidden_dim, patch_size, in_channels, cond_dim=self.cond_dim)

        # ── JiT-RCDM [fix-5b]: register 2D RoPE frequencies as a buffer ──
        grid = image_size // patch_size
        freqs = compute_2d_rope_freqs(grid, hidden_dim // num_heads)
        self.register_buffer("freqs_cis", freqs)   # (N, head_dim//2) complex, moves with .to(device)

        self._init_weights()

    def _init_weights(self):
        def _basic_init(module):
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        self.apply(_basic_init)
        nn.init.trunc_normal_(self.patch_embed.proj.weight, std=0.02)
        # ── JiT-RCDM [fix-3]: null_h stays at zeros init (nn.Parameter already zeros) ──
        # ── JiT-RCDM [fix-5c]: q_norm/k_norm are RMSNorm with ones init — not touched here ──

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        Rearrange flat patch predictions back to a spatial image.

        x : (B, N, patch_size² · C)  →  (B, C, H, W)
        """
        p = self.patch_size
        c = self.in_channels
        h = self.image_size // p
        w = self.image_size // p
        x = x.reshape(x.shape[0], h, w, p, p, c)
        x = x.permute(0, 5, 1, 3, 2, 4)            # (B, C, h, p, w, p)
        return x.reshape(x.shape[0], c, h * p, w * p)

    def forward(
        self,
        z_t: torch.Tensor,
        t:   torch.Tensor,
        h:   torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict the clean image x from the noisy image z_t.

        Args:
            z_t : (B, C, H, W)   noisy interpolation at timestep t
            t   : (B,)            flow-matching timestep ∈ [0, 1]
            h   : (B, h_dim)      DinoV3 CLS representation

        Returns:
            x_pred : (B, C, H, W)  predicted clean image
        """
        B = z_t.shape[0]

        # Build the cond_dim-dimensional conditioning signal c.
        t_emb = self.time_embed(timestep_embedding(t, self.cond_dim))   # (B, cond_dim)
        c     = t_emb + self.cond_proj(h)                               # (B, cond_dim)

        assert c.shape == (B, self.cond_dim), (
            f"conditioning shape mismatch: got {c.shape}, expected ({B}, {self.cond_dim})"
        )

        # Patchify noisy image
        x = self.patch_embed(z_t)       # (B, N, hidden_dim)

        # ── JiT-RCDM [fix-5b]: retrieve RoPE buffer (already on the right device) ──
        freqs_cis = self.freqs_cis   # (N, head_dim//2) complex, moves with .to(device)

        # Apply all JiT blocks — each block maps c → 6·hidden_dim modulation params
        for block in self.blocks:
            x = block(x, c, freqs_cis)

        # Project tokens → patch pixels
        x = self.final_layer(x, c)      # (B, N, p²·C)

        return self.unpatchify(x)       # (B, C, H, W)


# ---------------------------------------------------------------------------
# Flow-matching utilities
# ---------------------------------------------------------------------------

class FlowMatching:
    """
    Flow-matching training objective and Heun ODE sampler with CFG support.

    Training (x-prediction + null-h dropout):
        z_t = t·x + (1−t)·ε         linear interpolation (flow path)
        t ~ logit-normal(mu, sigma)  [fix-4b] mu=-0.8, sigma=0.8 (JiT Tab. 3)
        h → null_h  with prob p_uncond  [fix-3] learnable null token for CFG
        loss = ||f_θ(z_t, t, h) − x||²   MSE in *image* space

    Sampling (50-step Heun ODE + optional CFG):
        v_θ = (x_pred − z_t) / (1 − t)   velocity from x-prediction
        z_{t+dt} ≈ z_t + dt · (v(z_t,t) + v(z*,t+dt)) / 2

        With cfg_scale > 1:
            x_pred = x_uncond + cfg_scale * (x_cond − x_uncond)
        blending happens at x_pred level (before velocity) — bounded in image
        space, numerically cleaner than blending velocities directly.
    """

    def training_loss(
        self,
        model:    nn.Module,
        x:        torch.Tensor,
        h:        torch.Tensor,
        p_uncond: float = 0.0,
    ) -> torch.Tensor:
        """
        Args:
            model    : JiT instance
            x        : (B, C, H, W) clean images in [-1, 1]
            h        : (B, h_dim)   DinoV3 representations
            p_uncond : probability of replacing h with null_h (CFG dropout)

        Returns:
            loss : scalar MSE
        """
        B, device = x.shape[0], x.device

        # ── JiT-RCDM [fix-3]: use learnable null_h instead of zeros for CFG dropout ──
        if p_uncond > 0.0:
            mask = (torch.rand(B, device=device) < p_uncond).unsqueeze(-1)  # (B, 1)
            null = getattr(model, 'null_h', None)
            if null is not None:
                # ── JiT-RCDM [fix-3]: use learnable null_h instead of zeros ──
                # ── JiT-RCDM [fix-3]: no .detach() — null_h must receive gradients ──
                null_expanded = null.unsqueeze(0).expand(B, -1)
                h = torch.where(mask, null_expanded, h)
            else:
                h = h.masked_fill(mask, 0.0)

        # ── JiT-RCDM [fix-4b]: logit-normal t-sampler: mu=-0.8, sigma=0.8 (JiT paper Tab. 3) ──
        u = -0.8 + 0.8 * torch.randn(B, device=device)   # JiT paper Tab. 3: mu=-0.8, sigma=0.8
        t = torch.sigmoid(u)               # logit-normal in (0, 1)

        epsilon = torch.randn_like(x)
        t_view  = t.view(B, 1, 1, 1)
        z_t     = t_view * x + (1.0 - t_view) * epsilon

        x_pred = model(z_t, t, h=h)
        return F.mse_loss(x_pred, x)

    @torch.no_grad()
    def sample(
        self,
        model:     nn.Module,
        noise:     torch.Tensor,
        h:         torch.Tensor,
        num_steps: int   = 50,
        cfg_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Generate images via Heun ODE (t=0 → t≈1) with optional CFG.

        Args:
            model     : trained JiT (eval mode)
            noise     : (B, C, H, W) starting Gaussian noise
            h         : (B, h_dim) DinoV3 representations
            num_steps : ODE steps (default 50)
            cfg_scale : guidance strength (1.0 = no guidance, 3.0 = recommended).
                        Requires the model to have been trained with p_uncond > 0.

        Returns:
            x_gen : (B, C, H, W)
        """
        device  = noise.device
        B       = noise.shape[0]
        z       = noise.clone()
        dt      = 1.0 / num_steps
        use_cfg = cfg_scale > 1.0

        # ── JiT-RCDM [fix-3]: use learnable null_h for unconditional pass ──
        if use_cfg:
            null = getattr(model, 'null_h', None)
            h_null = (null.unsqueeze(0).expand(B, -1).to(device)
                      if null is not None else torch.zeros_like(h))
        else:
            h_null = None

        for i in range(num_steps):
            t_val   = i / num_steps
            t_batch = torch.full((B,), t_val, dtype=torch.float32, device=device)

            x_pred1 = self._cfg_pred(model, z, t_batch, h, h_null, cfg_scale)
            denom1  = max(1.0 - t_val, 1e-8)
            v1      = (x_pred1 - z) / denom1

            if i < num_steps - 1:
                z_euler      = z + dt * v1
                t_val_next   = (i + 1) / num_steps
                t_batch_next = torch.full((B,), t_val_next, dtype=torch.float32, device=device)
                x_pred2      = self._cfg_pred(model, z_euler, t_batch_next, h, h_null, cfg_scale)
                denom2       = max(1.0 - t_val_next, 1e-8)
                v2           = (x_pred2 - z_euler) / denom2
                z            = z + dt * (v1 + v2) / 2.0
            else:
                z = z + dt * v1   # pure Euler at last step to avoid 1/(1−t) at t=1

        return z

    @staticmethod
    def _cfg_pred(
        model:     nn.Module,
        z:         torch.Tensor,
        t_batch:   torch.Tensor,
        h:         torch.Tensor,
        h_null,                       # torch.Tensor or None
        cfg_scale: float,
    ) -> torch.Tensor:
        """Single x_pred call with optional CFG blending at the x_pred level."""
        x_cond = model(z, t_batch, h=h)
        if cfg_scale <= 1.0 or h_null is None:
            return x_cond
        x_uncond = model(z, t_batch, h=h_null)
        return x_uncond + cfg_scale * (x_cond - x_uncond)


def create_jit_model(
    image_size: int = 224,
    patch_size: int = 16,
    hidden_dim: int = 768,
    depth:      int = 12,
    num_heads:  int = 12,
    h_dim:      int = 384,
    cond_dim:   int = None,
) -> JiT:
    """
    Generic factory for the JiT-RCDM denoiser.

    Defaults match JiT-B (hidden_dim=768, num_heads=12) with DinoV3 ViT-S/16
    (h_dim=384). cond_dim defaults to hidden_dim (no bottleneck). Use the
    preset factories JiT_S_16 / JiT_S_32 for local-training–friendly configs.
    """
    return JiT(
        image_size=image_size,
        patch_size=patch_size,
        in_channels=3,
        hidden_dim=hidden_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=4,
        h_dim=h_dim,
        cond_dim=cond_dim,
    )


# ---------------------------------------------------------------------------
# Preset model configs — local-training–friendly small variants
# ---------------------------------------------------------------------------

def JiT_S_16(image_size: int = 224, h_dim: int = 384, **kwargs) -> JiT:
    """
    JiT-Small / patch-16  — recommended default for local training.

    hidden_dim=384, num_heads=6 (64-dim heads), depth=12, cond_dim=128.
    At 224×224: 196 tokens, ~25 M parameters.
    Roughly 4× fewer multiply-adds in the large linear layers vs JiT-B_16
    at the same depth and patch count.
    cond_dim=128 is a mild bottleneck (384→128): wider than 64 to retain more
    conditioning expressivity while still being leaner than no-bottleneck (384).

    Shape verification:
      freqs_cis : (196, 32) complex  [head_dim//2 = 64//2 = 32]
      PatchEmbed: (B, 196, 384) — no pos_embed added
      Attention.qkv: (B, 196, 3*384) → q/k/v: (B, 196, 6, 64)
      apply_rotary_emb: x(B,196,6,64), freqs_cis(196,32) → out(B,196,6,64)
      SwiGLU inner: round_256(384*4*2/3) = round_256(1024) = 1024
      null_h: Parameter(384,)
    """
    return JiT(
        image_size=image_size,
        patch_size=16,
        in_channels=3,
        hidden_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        h_dim=h_dim,
        cond_dim=128,
        **kwargs,
    )


def JiT_S_32(image_size: int = 224, h_dim: int = 384, **kwargs) -> JiT:
    """
    JiT-Small / patch-32  — fastest option for MPS / CPU-limited machines.

    Same width as JiT_S_16 (hidden_dim=384, cond_dim=128) but patch_size=32
    → 49 tokens instead of 196 at 224×224. Attention cost scales with N²,
    so this gives ~16× cheaper attention blocks. Recommended when memory or
    compute is very tight.
    """
    assert image_size % 32 == 0, f"image_size {image_size} must be divisible by 32"
    return JiT(
        image_size=image_size,
        patch_size=32,
        in_channels=3,
        hidden_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        h_dim=h_dim,
        cond_dim=128,
        **kwargs,
    )
