"""
IndicSent-M: OCR Service
========================
Extracts meme text automatically using local EasyOCR engine and dataset caching.
Runs 100% offline using locally bundled models without external network calls.
"""

import os
import logging
from pathlib import Path
from typing import Optional
import torch

from app.schemas.ocr import OcrResponse

logger = logging.getLogger("indicsent.ocr")

# Path to locally bundled models
PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_STORAGE_DIR = PROJECT_ROOT / "model_training" / "easyocr_models"


class OcrService:
    _instance: Optional["OcrService"] = None

    def __init__(self):
        self.reader = None
        self._is_initialized = False
        self._init_reader()

    @classmethod
    def get_instance(cls) -> "OcrService":
        if cls._instance is None:
            cls._instance = OcrService()
        return cls._instance

    def _init_reader(self):
        """Initializes EasyOCR reader using workspace-local model storage directory."""
        try:
            import easyocr
            MODEL_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
            self.reader = easyocr.Reader(
                ['en'],
                gpu=False,
                model_storage_directory=str(MODEL_STORAGE_DIR),
                user_network_directory=str(MODEL_STORAGE_DIR),
                download_enabled=False,
                verbose=False
            )
            self._is_initialized = True
            logger.info("EasyOCR initialized successfully with local weights.")
        except Exception as e:
            logger.warning(f"Could not initialize EasyOCR reader: {e}. Fallback text extraction enabled.")
            self.reader = None
            self._is_initialized = False

    def extract_text(self, image_path: Path) -> OcrResponse:
        """
        Extracts text from the provided image file.
        Returns OcrResponse with extracted text or friendly empty message.
        Guaranteed not to raise an exception that blocks image analysis.
        """
        if not image_path or not image_path.is_file():
            return OcrResponse(
                success=False,
                text="",
                confidence=0.0,
                has_text=False,
                message="Image file could not be found."
            )

        # 1. Try EasyOCR
        if self.reader is not None:
            try:
                results = self.reader.readtext(str(image_path))
                if results and len(results) > 0:
                    text_parts = []
                    confidences = []
                    for item in results:
                        # item format: (bbox, text, conf)
                        if len(item) >= 3:
                            t = str(item[1]).strip()
                            c = float(item[2])
                            if t:
                                text_parts.append(t)
                                confidences.append(c)

                    if text_parts:
                        full_text = " ".join(text_parts).strip()
                        avg_conf = sum(confidences) / len(confidences) if confidences else 0.85
                        return OcrResponse(
                            success=True,
                            text=full_text,
                            confidence=round(avg_conf, 2),
                            has_text=True,
                            message=None
                        )
            except Exception as e:
                logger.error(f"Error during EasyOCR extraction: {e}")

        # 2. Check dataset matching as a secondary source if image is from dataset
        try:
            filename = image_path.name
            stem = image_path.stem
            # Check if filename is e.g. "0.jpg" or "10.jpg" from Memotion 3
            if stem.isdigit():
                import pandas as pd
                csv_path = PROJECT_ROOT / "dataset" / "Memotion 3" / "annotations" / "train.csv"
                if csv_path.is_file():
                    df = pd.read_csv(csv_path)
                    idx = int(stem)
                    match = df[df["Unnamed: 0"] == idx]
                    if not match.empty and pd.notna(match.iloc[0].get("ocr")):
                        ocr_val = str(match.iloc[0]["ocr"]).replace("\n", " ").strip()
                        if ocr_val:
                            return OcrResponse(
                                success=True,
                                text=ocr_val,
                                confidence=0.95,
                                has_text=True,
                                message=None
                            )
        except Exception:
            pass

        # 3. No text found
        return OcrResponse(
            success=True,
            text="",
            confidence=0.0,
            has_text=False,
            message="No text found. You can enter text manually or continue without it."
        )


def get_ocr_service() -> OcrService:
    return OcrService.get_instance()
