"""
IndicSent-M: Image Preprocessing and Transforms Module
Handles image verification, RGB conversion, and model-specific transforms.
Strictly ensures:
- Data augmentation applied ONLY on training split.
- Deterministic, unaugmented normalization on validation and test splits.
"""

from pathlib import Path
from typing import Optional, Tuple, Callable
from PIL import Image
import torch
from torchvision import transforms

# Standard ImageNet normalization constants (ResNet, EfficientNet, ViT)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# OpenAI CLIP normalization constants
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

def load_and_verify_image(image_path: Path) -> Optional[Image.Image]:
    """
    Safely opens an image, verifies it, converts to 3-channel RGB.
    Returns None if the image is missing or corrupted.
    """
    try:
        with Image.open(image_path) as img:
            img.verify()
        with Image.open(image_path) as img:
            return img.convert("RGB")
    except Exception:
        return None

def get_image_transforms(
    model_type: str = "resnet50",
    image_size: int = 224,
    is_training: bool = False
) -> transforms.Compose:
    """
    Returns appropriate PyTorch torchvision transform pipeline.
    
    Args:
        model_type: 'resnet50', 'efficientnet', 'clip', 'vit'
        image_size: Target square pixel dimension (default 224)
        is_training: If True, applies mild data augmentation. If False, strictly deterministic.
    """
    mean = CLIP_MEAN if "clip" in model_type.lower() else IMAGENET_MEAN
    std = CLIP_STD if "clip" in model_type.lower() else IMAGENET_STD

    if is_training:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

if __name__ == "__main__":
    t_train = get_image_transforms("resnet50", is_training=True)
    t_val = get_image_transforms("resnet50", is_training=False)
    print("Train transform:", t_train)
    print("Val transform:", t_val)
