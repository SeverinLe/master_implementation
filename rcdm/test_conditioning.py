import torch
import sys
sys.path.append("..")      # so we can import rcdm from the project root

from rcdm.conditioning import ConditionalBatchNorm2d, ConditioningProjector

# Simulate a batch of 4 images, 64 channels, 32x32 spatial resolution
B, C, H, W = 4, 64, 32, 32
x = torch.randn(B, C, H, W)

# Simulate the raw SSL representation (2048-dim, like a ResNet-50 backbone)
h_raw = torch.randn(B, 2048)

# Step 1: project h down to 512
projector = ConditioningProjector(h_dim=2048, cond_dim=512)
h_proj = projector(h_raw)
print(f"h_raw shape : {h_raw.shape}")    # expect torch.Size([4, 2048])
print(f"h_proj shape: {h_proj.shape}")   # expect torch.Size([4, 512])

# Step 2: run through cBN
cbn = ConditionalBatchNorm2d(num_features=C, cond_dim=512)
out = cbn(x, h_proj)
print(f"x shape     : {x.shape}")        # expect torch.Size([4, 64, 32, 32])
print(f"out shape   : {out.shape}")      # expect torch.Size([4, 64, 32, 32])

# Check that different h values actually produce different outputs
h_other = torch.randn(B, 2048)
h_other_proj = projector(h_other)
out_other = cbn(x, h_other_proj)
print(f"\nOutputs differ with different h: {not torch.allclose(out, out_other)}")
# expect: True