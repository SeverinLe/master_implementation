import torch, sys
import rcdm
sys.path.append(".")

from rcdm.encoder import load_encoder, build_transform, encode_image

device = "cpu"
encoder   = load_encoder(device=device)
transform = build_transform(image_size=64)

# Grab any image from your Tiny ImageNet training set
test_image = "data/tiny-imagenet-200/train/n01443537/images/n01443537_0.JPEG"

h = encode_image(test_image, encoder, transform, device=device)
print(f"h shape : {h.shape}")       # expect torch.Size([1, 2048])
print(f"h dtype : {h.dtype}")       # expect torch.float32
print(f"h norm  : {h.norm():.2f}")  # expect some positive number, typically 20–60

# Confirm encoder is truly frozen
trainable = sum(p.requires_grad for p in encoder.parameters())
print(f"trainable params: {trainable}")   # expect 0