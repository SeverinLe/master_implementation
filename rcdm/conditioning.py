import torch
import torch.nn as nn


class ConditionalBatchNorm2d(nn.Module):
    """
    Conditional Batch Normalization.

    Replaces GroupNorm in the ADM UNet. Instead of fixed learned
    scale (gamma) and shift (beta), produce them dynamically
    from the conditioning vector h — the SSL representation.

    Args:
        num_features : number of channels C in the feature map (B, C, H, W)
        cond_dim     : dimension of the conditioning vector h (we'll use 512,
                       after projecting from the raw SSL representation)
    """

    def __init__(self, num_features: int, cond_dim: int):
        super().__init__()

        # plain batchnorm with NO learned affine parameters
        # supply gamma and beta ourselves from h
        self.bn = nn.BatchNorm2d(num_features, affine=False, eps=1e-5, momentum=0.1)

        # two small linear layers: h → gamma, h → beta
        # output size matches the number of channels to scale/shift each channel independently
        self.gamma_fc = nn.Linear(cond_dim, num_features)
        self.beta_fc  = nn.Linear(cond_dim, num_features)

        # gamma starts at 1  → identity scale (like normal batchnorm at init)
        # beta  starts at 0  → no shift at init
        # start of training: cBN behaves exactly like plain batchnorm
        # network stabilises first, then h starts steering
        nn.init.ones_(self.gamma_fc.weight)
        nn.init.zeros_(self.gamma_fc.bias)
        nn.init.zeros_(self.beta_fc.weight)
        nn.init.zeros_(self.beta_fc.bias)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : feature map  — shape (B, C, H, W)
            h : conditioning  — shape (B, cond_dim)  [the projected SSL vector]

        Returns:
            out : shape (B, C, H, W)  — normalized, then scaled and shifted by h
        """
        # Step 1: normalize across the batch (removes mean, sets variance to 1)
        # result is still (B, C, H, W)
        x_norm = self.bn(x)

        # Step 2: produce per-channel gamma and beta from h
        # gamma_fc(h) shape: (B, C)
        # need (B, C, 1, 1) to broadcast across H and W
        gamma = self.gamma_fc(h).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        beta  = self.beta_fc(h).unsqueeze(-1).unsqueeze(-1)   # (B, C, 1, 1)

        # Step 3: scale and shift — this is where h actually steers the network
        return gamma * x_norm + beta


class ConditioningProjector(nn.Module):
    """
    Projects the raw SSL representation down to cond_dim before
    it enters any cBN layer.

    The raw backbone output is 2048-dim (ResNet-50); project it down
    to 512 once, then reuse that 512-dim vector at every cBN layer.
    This keeps the cBN layers small and avoids overfitting on h.

    Args:
        h_dim    : dimension of raw SSL representation (default 2048 for ResNet-50)
        cond_dim : target dimension passed to all cBN layers (default 512)
    """

    def __init__(self, h_dim: int = 2048, cond_dim: int = 512):
        super().__init__()

        self.proj = nn.Sequential(
            nn.Linear(h_dim, cond_dim),
            nn.SiLU(),           # smooth activation, same as ADM uses elsewhere
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h : raw SSL representation — shape (B, h_dim)
        Returns:
            projected h — shape (B, cond_dim)
        """
        return self.proj(h)