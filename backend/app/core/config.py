"""
IndicSent-M: Backend Application Configuration
"""

import os
from pathlib import Path
from typing import List, Set
import torch

# Base directories
BACKEND_DIR = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BACKEND_DIR.parent
MODEL_TRAINING_DIR = PROJECT_ROOT / "model_training"
CHECKPOINTS_DIR = MODEL_TRAINING_DIR / "checkpoints"
MODEL_SRC_DIR = MODEL_TRAINING_DIR / "src"

class Settings:
    PROJECT_NAME: str = "IndicSent-M Sentiment Analysis API"
    VERSION: str = "1.0.0"
    DESCRIPTION: str = "High-performance multimodal sentiment classification API for Hinglish memes using MuRIL, XLM-RoBERTa, ResNet-50, and Gated Fusion."
    API_PREFIX: str = "/api"

    # CORS configuration
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ]

    # File upload constraints
    MAX_FILE_SIZE_BYTES: int = 10 * 1024 * 1024  # 10 MB
    ALLOWED_EXTENSIONS: Set[str] = {".jpg", ".jpeg", ".png", ".webp"}
    ALLOWED_MIME_TYPES: Set[str] = {
        "image/jpeg",
        "image/pjpeg",
        "image/png",
        "image/webp"
    }

    # Model defaults
    DEFAULT_MULTIMODAL_MODEL: str = "gated_fusion"
    DEFAULT_TEXT_MODEL: str = "logistic_regression"
    DEFAULT_CLASSICAL_MODEL: str = "logistic_regression"
    DEFAULT_IMAGE_MODEL: str = "resnet50"

    # Hardware device
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Paths
    PROJECT_ROOT: Path = PROJECT_ROOT
    MODEL_TRAINING_DIR: Path = MODEL_TRAINING_DIR
    CHECKPOINTS_DIR: Path = CHECKPOINTS_DIR
    MODEL_SRC_DIR: Path = MODEL_SRC_DIR

settings = Settings()
