"""
IndicSent-M: Standalone Production Inference Module
Designed for direct integration with FastAPI or CLI testing.
Loads exported best model checkpoint, applies exact preprocessing,
and returns 3-class sentiment predictions with confidence scores.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, Union, Optional
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms

# Local imports
project_root = Path(__file__).resolve().parents[2]
src_dir = Path(__file__).resolve().parents[0]
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from preprocessing.clean_text import clean_ocr_text
from preprocessing.preprocess_images import load_and_verify_image

LABEL_MAP = {0: "Negative", 1: "Neutral", 2: "Positive"}

class IndicSentInferenceEngine:
    def __init__(self, checkpoint_dir: Optional[Path] = None, device: Optional[str] = None):
        if checkpoint_dir is None:
            checkpoint_dir = project_root / "model_training" / "checkpoints" / "best_model"
        self.checkpoint_dir = Path(checkpoint_dir)
        
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = None
        self.tokenizer = None
        self.metadata = None
        self.classical_pipeline = None

        self._load_artifacts()

    def _load_artifacts(self):
        meta_path = self.checkpoint_dir / "metadata.json"
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
        
        # Check if classical joblib model is present
        joblib_path = self.checkpoint_dir / "model.joblib"
        if joblib_path.exists():
            import joblib
            self.classical_pipeline = joblib.load(joblib_path)
            print(f"Loaded Classical Pipeline from {joblib_path.name}")
            return

        # Check PyTorch deep learning weights
        pt_path = self.checkpoint_dir / "model_weights.pt"
        if pt_path.exists():
            # Neural model loader will be initialized here once deep model is exported
            pass

    def predict(self, text: str, image_input: Optional[Union[str, Path, Image.Image]] = None) -> Dict[str, Any]:
        """
        Runs multimodal or text-based inference on raw input.
        
        Args:
            text: Raw Hinglish meme caption or OCR text.
            image_input: Optional file path or PIL Image.
            
        Returns:
            Dict containing predicted sentiment, confidence percentage,
            and probability distribution across Negative, Neutral, and Positive.
        """
        clean_text = clean_ocr_text(text)
        
        if self.classical_pipeline is not None:
            # Predict with classical pipeline
            probs = self.classical_pipeline.predict_proba([clean_text])[0]
            pred_idx = int(probs.argmax())
            pred_label = LABEL_MAP[pred_idx]
            
            return {
                "sentiment": pred_label,
                "confidence": round(float(probs[pred_idx]) * 100, 2),
                "scores": {
                    "negative": round(float(probs[0]) * 100, 2),
                    "neutral": round(float(probs[1]) * 100, 2),
                    "positive": round(float(probs[2]) * 100, 2)
                },
                "cleaned_text": clean_text,
                "modality_used": "Text Only"
            }

        return {
            "error": "No model loaded. Export best model to checkpoints/best_model first."
        }

if __name__ == "__main__":
    engine = IndicSentInferenceEngine()
    test_phrase = "Padhai chhod ke sab ho raha hai life me, bas syllabus complete nahi ho raha 😭"
    res = engine.predict(test_phrase)
    print("Inference Test Result:")
    print(json.dumps(res, indent=2))
