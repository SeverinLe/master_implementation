import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


def load_encoder(device="cpu"):
    """
    Load a pretrained ResNet-50 backbone. 

    VISSL is a large framework built for distributed training clusters and its model zoo requires specific config 
    files to load. For Tiny ImageNet at demonstration scale, we use torchvision directly, which gives us the same 
    ResNet-50 backbone with pretrained ImageNet weights in three lines. The representations it produces are identical
    in structure to what VISSL would give — when you move to full ImageNet scale later, swapping in a VISSL checkpoint 
    is one function call.

    The final classification layer (fc) is removed — we want the
    2048-dim feature vector before that layer, not class probabilities.

    This is exactly what the RCDM paper does: use the ResNet-50
    trunk/backbone output as the conditioning vector h.

    Args:
        device : "cpu" or "cuda"

    Returns:
        encoder : frozen ResNet-50 backbone, eval mode, on device
    """
    # Load ResNet-50 with ImageNet pretrained weights
    encoder = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    # Remove the classification head — output is now (B, 2048)
    # avgpool gives (B, 2048, 1, 1), Flatten gives (B, 2048)
    encoder.fc = nn.Identity()

    # Freeze all weights — we never train the encoder
    for param in encoder.parameters():
        param.requires_grad = False

    encoder.eval()
    encoder.to(device)
    return encoder


# Standard ImageNet normalisation — must match what ResNet-50 was trained with
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def build_transform(image_size=64):
    """
    Preprocessing pipeline for any input image before encoding.

    Resize → CenterCrop → Tensor → Normalize.
    image_size should match what RCDM was trained on (64 for Tiny ImageNet).
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
    Extract the 2048-dim backbone representation from a single image file.

    This is the h vector that gets passed to RCDM as conditioning.
    Works on ANY image — in-distribution (Tiny ImageNet) or completely
    out-of-distribution (your own photos).

    Args:
        image_path : path to image file (jpg, png, anything PIL can open)
        encoder    : the frozen ResNet-50 from load_encoder()
        transform  : the preprocessing pipeline from build_transform()
        device     : must match encoder's device

    Returns:
        h : torch.Tensor of shape (1, 2048)
    """
    img = Image.open(image_path).convert("RGB")  # convert handles grayscale/RGBA
    x = transform(img).unsqueeze(0).to(device)   # (1, 3, H, W)
    h = encoder(x)                               # (1, 2048)
    return h


@torch.no_grad()
def encode_batch(image_paths, encoder, transform, device="cpu", batch_size=64):
    """
    Extract representations for a list of image paths efficiently.

    Used by precompute_reps.py to cache all training representations
    before training starts — so we don't re-run the encoder every step.

    Args:
        image_paths : list of file paths
        encoder     : frozen ResNet-50 from load_encoder()
        transform   : from build_transform()
        device      : must match encoder's device
        batch_size  : how many images to process at once

    Returns:
        reps : torch.Tensor of shape (N, 2048)
    """
    all_reps = []

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]

        # Load and preprocess each image in the batch
        imgs = []
        for p in batch_paths:
            img = Image.open(p).convert("RGB")
            imgs.append(transform(img))

        # Stack into a single tensor (B, 3, H, W)
        x = torch.stack(imgs).to(device)

        # Run through encoder
        h = encoder(x)   # (B, 2048)
        all_reps.append(h.cpu())

        if i % (batch_size * 10) == 0:
            print(f"  encoded {i}/{len(image_paths)} images")

    return torch.cat(all_reps, dim=0)   # (N, 2048)