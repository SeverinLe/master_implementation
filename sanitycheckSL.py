import torch, sys
sys.path.append(".")

from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults

args = model_and_diffusion_defaults()
args.update({
    "image_size": 64,
    "num_channels": 128,
    "num_res_blocks": 2,
    "learn_sigma": True,
    "diffusion_steps": 1000,
    "noise_schedule": "linear",
    "h_dim": 2048,
})

model, diffusion = create_model_and_diffusion(**args)
model.eval()

x = torch.randn(2, 3, 64, 64)
t = torch.tensor([500, 200])
h = torch.randn(2, 2048)

out = model(x, t, h=h)
print(f"output shape: {out.shape}")
# expect: torch.Size([2, 6, 64, 64])
