import torch
import torch.nn as nn


# ── JiT-RCDM [fix-5a]: RMSNorm defined here; imported by jit.py ──
class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich 2019). Used by JiT instead of LayerNorm."""

    def __init__(self, dim: int, eps: float = 1e-6, affine: bool = True):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) if affine else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        x_out = x.float() * norm
        if self.weight is not None:
            x_out = x_out * self.weight
        return x_out.to(x.dtype)


class ConditionalBatchNorm2d(nn.Module):
    """
    Conditional Batch Normalization — used by the legacy UNet denoiser.

    Kept for backward compatibility with the UNet path. The JiT denoiser
    uses AdaLNZero instead (see below).
    """

    def __init__(self, num_features: int, cond_dim: int):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features, affine=False, eps=1e-5, momentum=0.1)
        self.gamma_fc = nn.Linear(cond_dim, num_features)
        self.beta_fc  = nn.Linear(cond_dim, num_features)
        nn.init.ones_(self.gamma_fc.weight)
        nn.init.zeros_(self.gamma_fc.bias)
        nn.init.zeros_(self.beta_fc.weight)
        nn.init.zeros_(self.beta_fc.bias)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        x_norm = self.bn(x)
        gamma = self.gamma_fc(h).unsqueeze(-1).unsqueeze(-1)
        beta  = self.beta_fc(h).unsqueeze(-1).unsqueeze(-1)
        return gamma * x_norm + beta


class ConditioningProjector(nn.Module):
    """
    Projects the DinoV3 CLS token directly to hidden_dim.

    Following the JiT paper exactly: cond_dim == hidden_dim.
    There is no compression bottleneck — h is projected to the same
    dimension as the ViT hidden state so it can be added to the timestep
    embedding without any shape mismatch.

    Why keep a projector at all: the CLS token carries global image semantics
    at a fixed scale. A learned linear + SiLU layer lets the model warp that
    semantic space to align with the diffusion model's internal representation
    without touching the frozen encoder.

    Args:
        h_dim    : dimension of raw DinoV3 CLS token (384 for ViT-S/16)
        cond_dim : target dimension = hidden_dim of the JiT ViT
    """

    def __init__(self, h_dim: int = 384, cond_dim: int = 768):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(h_dim, cond_dim),
            nn.SiLU(),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h : (B, h_dim) → (B, cond_dim)"""
        return self.proj(h)


class AdaLNZero(nn.Module):
    """
    Adaptive Layer Norm Zero (adaLN-Zero) from DiT (Peebles & Xie 2022),
    adopted directly by JiT.

    Replaces Conditional Batch Norm in the JiT denoiser. Unlike cBN which
    conditions a spatial 2-D feature map, adaLN-Zero conditions a sequence
    of 1-D token embeddings — the natural choice for a ViT backbone.

    How it works:
      A single MLP takes the fused conditioning signal
          c = timestep_embedding(t) + cond_proj(h)
      and produces 6 modulation scalars per token dimension:
          [shift_attn, scale_attn, gate_attn, shift_ffn, scale_ffn, gate_ffn]

      Applied as:
          x ← x + gate_attn · Attn((1+scale_attn)·RMSNorm(x) + shift_attn)
          x ← x + gate_ffn  · FFN((1+scale_ffn) ·RMSNorm(x) + shift_ffn)

    Zero-init rationale:
      The output projection of adaLN_modulation is zero-initialised.
      At the start of training all gate values are 0, so every block is an
      identity function — identical reasoning to cBN initialising gamma=1,
      beta=0. The network stabilises before conditioning starts having effect.

    Args:
        hidden_dim : ViT hidden dimension (384 for ViT-S, 768 for ViT-B, 1024 for ViT-L)
        cond_dim   : dimension of the fused conditioning vector c
    """

    def __init__(self, hidden_dim: int, cond_dim: int):
        super().__init__()
        # ── JiT-RCDM [fix-5a]: RMSNorm instead of LayerNorm ──
        self.norm1 = RMSNorm(hidden_dim, affine=False, eps=1e-6)
        self.norm2 = RMSNorm(hidden_dim, affine=False, eps=1e-6)

        # MLP: SiLU first so gradients flow cleanly through zero-init output
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 6 * hidden_dim),
        )
        # CRITICAL: zero-init — gates start at 0 → identity at init
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def modulate(self, x, shift, scale):
        return (1 + scale.unsqueeze(1)) * x + shift.unsqueeze(1)

    def forward_pre(self, x: torch.Tensor, c: torch.Tensor):
        """
        Return the 6 modulation params from conditioning c.

        Called by JiTBlock which supplies its own attention and FFN modules,
        keeping AdaLNZero as a pure conditioning helper.

        Args:
            x : token sequence  (B, N, hidden_dim)
            c : fused condition  (B, cond_dim)

        Returns:
            (shift_a, scale_a, gate_a, shift_f, scale_f, gate_f)
            each shape (B, hidden_dim)
        """
        return self.adaLN_modulation(c).chunk(6, dim=-1)
