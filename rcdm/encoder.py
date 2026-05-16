import os
import torch
from torchvision import transforms
from transformers import AutoModel
from PIL import Image

# Path to the local DinoV3 checkpoint directory.
# Expected contents: config.json + model.safetensors (HuggingFace format).
# ViT-S/16 architecture → hidden_dim=384 → CLS token is 384-dim.
DINOV3_CHECKPOINT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "checkpoints", "dinov3_vits16_tmp",
)

# CLS-token dimension for DinoV3 ViT-S/16
ENCODER_OUTPUT_DIM = 384


def load_encoder(device="cpu", checkpoint_path=DINOV3_CHECKPOINT):
    """
    Load DinoV3 ViT-S/16 from a local checkpoint as the frozen SSL encoder.

    Why DinoV3 over DinoV2-L:
      - DinoV3 ViT-S/16 is a custom-trained DINO-family SSL model fine-tuned
        for retinal / medical imaging, making its representations more domain-
        appropriate for Messidor-2 fundus images than a general-domain ViT-L.
      - ViT-S/16 (hidden_dim=384) paired with a 64-dim conditioning bottleneck
        keeps the conditioning path lean without losing semantic content.
      - Loading from a local checkpoint ensures reproducibility: the weights
        are fixed and not tied to a remote HuggingFace model version.

    The checkpoint directory must contain HuggingFace-compatible files:
        config.json
        model.safetensors  (or pytorch_model.bin)

    Output: CLS token from last_hidden_state[:,0,:] — shape (B, 384).

    Args:
        device          : "cpu" or "cuda"
        checkpoint_path : path to the local DinoV3 checkpoint directory

    Returns:
        encoder : frozen DinoV3-S backbone, eval mode, on device
    """
    encoder = AutoModel.from_pretrained(checkpoint_path, local_files_only=True)

    for param in encoder.parameters():
        param.requires_grad = False

    encoder.eval()
    encoder.to(device)
    return encoder


# DinoV3 uses standard ImageNet normalisation (same as DinoV2)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def build_transform(image_size=224):
    """
    Preprocessing pipeline for DinoV3 ViT-S/16.

    ViT-S/16 has patch_size=16; any input size divisible by 16 works.
    224×224 is the canonical choice and aligns with the JiT patch grid.
    """
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


@torch.no_grad()
def encode_image(image_path, encoder, transform, device="cpu"):
    """
    Extract the 384-dim CLS token from a single image via DinoV3.

    Args:
        image_path : path to image file (jpg, png, anything PIL can open)
        encoder    : frozen DinoV3-S from load_encoder()
        transform  : preprocessing pipeline from build_transform()
        device     : must match encoder's device

    Returns:
        h : torch.Tensor of shape (1, 384)
    """
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)        # (1, 3, H, W)
    out = encoder(pixel_values=x)
    h = out.last_hidden_state[:, 0, :]                # CLS token → (1, 384)
    return h


@torch.no_grad()
def encode_batch(image_paths, encoder, transform, device="cpu", batch_size=64):
    """
    Extract representations for a list of image paths efficiently.

    Used by precompute_reps.py to cache all training representations before
    training starts — so we do not re-run the encoder every step.

    Args:
        image_paths : list of file paths
        encoder     : frozen DinoV3-S from load_encoder()
        transform   : from build_transform()
        device      : must match encoder's device
        batch_size  : how many images to process at once

    Returns:
        reps : torch.Tensor of shape (N, 384)
    """
    all_reps = []

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]

        imgs = []
        for p in batch_paths:
            img = Image.open(p).convert("RGB")
            imgs.append(transform(img))

        x = torch.stack(imgs).to(device)              # (B, 3, H, W)

        out = encoder(pixel_values=x)
        h = out.last_hidden_state[:, 0, :]            # CLS token → (B, 384)
        all_reps.append(h.cpu())

        if i % (batch_size * 10) == 0:
            print(f"  encoded {i}/{len(image_paths)} images")

    return torch.cat(all_reps, dim=0)                 # (N, 384)
