"""
rcdm/dataset.py

A dataset that returns (image_tensor, h) pairs.
Images come from disk; h vectors come from the precomputed .pt file.
The index alignment from precompute_reps.py guarantees they match.
"""

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


class RepresentationDataset(Dataset):
    """
    Returns (x, h) where:
        x : image tensor (3, image_size, image_size), normalised to [-1, 1]
            — this is what the diffusion model expects
        h : representation tensor (2048,)
            — this is the conditioning vector

    Note on normalisation:
        The encoder uses ImageNet mean/std normalisation.
        The diffusion model expects pixels in [-1, 1] (centre around 0).
        These are two different transforms for two different purposes —
        x uses the diffusion normalisation, h was computed with encoder normalisation.
    """

    def __init__(self, reps_file, image_size=64):
        """
        Args:
            reps_file  : path to the .pt file from precompute_reps.py
            image_size : spatial size to resize images to
        """
        print(f"Loading representations from {reps_file}...")
        data = torch.load(reps_file)

        self.paths = data["paths"]   # list[str], length N
        self.reps  = data["reps"]    # Tensor (N, 2048)

        print(f"  {len(self.paths)} image-representation pairs loaded")

        # Diffusion model expects images in [-1, 1]
        # This is standard for all ADM/DDPM training
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),                         # [0, 1]
            transforms.Normalize([0.5, 0.5, 0.5],
                                  [0.5, 0.5, 0.5]),        # → [-1, 1]
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        # Load image and apply diffusion normalisation
        img = Image.open(self.paths[idx]).convert("RGB")
        x = self.transform(img)        # (3, image_size, image_size)

        # Load precomputed representation — already a tensor
        h = self.reps[idx]             # (2048,)

        return x, h