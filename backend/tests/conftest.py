"""
IndicSent-M Backend: Pytest Fixtures & Configuration
"""

import sys
import io
import os
from pathlib import Path
import pytest
from PIL import Image
from fastapi.testclient import TestClient

# Ensure backend and model_training/src are in sys.path
TESTS_DIR = Path(__file__).resolve().parent
BACKEND_DIR = TESTS_DIR.parent
PROJECT_ROOT = BACKEND_DIR.parent
MODEL_SRC_DIR = PROJECT_ROOT / "model_training" / "src"

for p in [str(BACKEND_DIR), str(MODEL_SRC_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from app.main import app
from app.core.config import settings


@pytest.fixture(scope="session")
def client():
    """FastAPI TestClient fixture."""
    test_client = TestClient(app)
    yield test_client


@pytest.fixture(scope="session")
def valid_image_bytes() -> bytes:
    """Generates valid JPEG image bytes in memory."""
    img = Image.new("RGB", (224, 224), color=(73, 109, 137))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture(scope="session")
def valid_png_bytes() -> bytes:
    """Generates valid PNG image bytes in memory."""
    img = Image.new("RGB", (100, 100), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture(scope="session")
def corrupt_image_bytes() -> bytes:
    """Returns invalid, corrupted image data."""
    return b"NOT_A_VALID_IMAGE_HEADER_OR_DATA_XYZ_99999"


@pytest.fixture(scope="session")
def oversized_image_bytes() -> bytes:
    """Generates data exceeding the 10 MB limit."""
    # 10 MB + 1 KB
    return b"A" * (settings.MAX_FILE_SIZE_BYTES + 1024)
