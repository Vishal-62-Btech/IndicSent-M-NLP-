"""
IndicSent-M: Unified Multimodal Sentiment Predictor Module
==========================================================
Production-ready inference engine for IndicSent-M (Multimodal Sentiment Analysis of Hinglish Memes).

Supported Modes:
1. 'classical'        : Classical ML models (TF-IDF + Logistic Regression, SVM, Naive Bayes, Random Forest)
2. 'transformer_text': Deep text models (Fine-tuned XLM-RoBERTa, MuRIL)
3. 'image'           : Vision transfer learning models (ResNet-50, EfficientNet-B0, MobileNet-V3)
4. 'multimodal'      : Multimodal fusion models (MuRIL + ResNet-50 Gated Fusion, Concat Fusion)

Features:
- Configurable pretrained loading with zero unnecessary network calls during inference.
- Rigorous input validation (rejection of empty text, corrupt images, missing modalities).
- Fine-grained empirical timing (preprocessing vs. model execution using torch.inference_mode and CUDA sync).
- Direct compatibility with FastAPI endpoints and CLI invocations.
"""

import sys
import os
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import joblib

# Ensure UTF-8 output encoding for Windows terminals
if sys.stdout.encoding != 'utf-8' and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Path configuration
INFERENCE_DIR = Path(__file__).resolve().parent
SRC_DIR = INFERENCE_DIR.parent
MODEL_TRAINING_DIR = SRC_DIR.parent
PROJECT_ROOT = MODEL_TRAINING_DIR.parent
CHECKPOINTS_ROOT = MODEL_TRAINING_DIR / "checkpoints"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Transformers and local model modules
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from models.multimodal_models import (
    GatedMultimodalClassifier,
    MultimodalConcatClassifier,
    MultiTaskGatedMultimodalClassifier,
    SIX_EMOTIONS,
    EMOTION_EMOJIS
)
from models.image_models import ResNet50Classifier, EfficientNetB0Classifier, MobileNetV3LargeClassifier
from preprocessing.clean_text import clean_ocr_text
from preprocessing.preprocess_images import get_image_transforms

LABEL_MAP = {0: "Negative", 1: "Neutral", 2: "Positive"}


def validate_text_input(text: Optional[str], mode_name: str) -> str:
    """
    Validates and cleans raw text input.
    Raises ValueError if text is missing, empty, or whitespace-only.
    """
    if text is None:
        raise ValueError(f"[{mode_name.upper()}] Text input cannot be None. Valid text is required.")
    
    if not isinstance(text, str):
        text = str(text)
        
    stripped = text.strip()
    if not stripped:
        raise ValueError(f"[{mode_name.upper()}] Text input cannot be empty or whitespace-only.")
        
    cleaned = clean_ocr_text(stripped)
    if cleaned == "[NO_TEXT]" and not any(c.isalnum() for c in stripped):
        raise ValueError(f"[{mode_name.upper()}] Text contains no alphanumeric characters or valid sentiment tokens: {repr(text)}")
        
    return cleaned


def validate_and_load_image(image_path: Optional[Union[str, Path]], mode_name: str) -> Image.Image:
    """
    Validates image existence and format integrity.
    Raises FileNotFoundError if missing, ValueError if corrupted or unreadable.
    """
    if image_path is None:
        raise ValueError(f"[{mode_name.upper()}] Image input cannot be None. Valid image path is required.")
        
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"[{mode_name.upper()}] Image file not found at: {path}")
        
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            return img.convert("RGB")
    except Exception as exc:
        raise ValueError(f"[{mode_name.upper()}] Image file is corrupted or invalid: {path}. Details: {exc}")


class IndicSentPredictor:
    """
    Production inference engine supporting classical, transformer, vision, and multimodal models.
    """
    def __init__(
        self,
        mode: str = "multimodal",
        model_name: Optional[str] = None,
        checkpoint_path: Optional[Union[str, Path]] = None,
        device: Optional[Union[str, torch.device]] = None
    ):
        mode_alias = mode.lower().strip()
        if mode_alias in ("classical", "ml", "baseline"):
            self.mode = "classical"
        elif mode_alias in ("transformer_text", "text", "transformer", "text_only"):
            self.mode = "transformer_text"
        elif mode_alias in ("image", "vision", "image_only"):
            self.mode = "image"
        elif mode_alias in ("multimodal", "fusion", "mm"):
            self.mode = "multimodal"
        else:
            raise ValueError(f"Unsupported prediction mode: '{mode}'. Must be one of ['classical', 'transformer_text', 'image', 'multimodal'].")

        # Configure computation device
        if self.mode == "classical":
            self.device = torch.device("cpu")
        else:
            if device is None:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            elif isinstance(device, str):
                self.device = torch.device(device)
            else:
                self.device = device

        self.model_name = model_name
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        
        # Load model and preprocessing components
        self._initialize_pipeline()

    def _initialize_pipeline(self):
        """Initializes model architecture and loads checkpoints without external downloads."""
        if self.mode == "classical":
            self._init_classical()
        elif self.mode == "transformer_text":
            self._init_transformer_text()
        elif self.mode == "image":
            self._init_image()
        elif self.mode == "multimodal":
            self._init_multimodal()

    def _init_classical(self):
        """Loads scikit-learn pipeline (.joblib)."""
        sub_model = (self.model_name or "logistic_regression").lower()
        if self.checkpoint_path and self.checkpoint_path.is_file():
            model_file = self.checkpoint_path
            self.display_name = "TF-IDF + Logistic Regression" 
        elif "svm" in sub_model:
            model_file = CHECKPOINTS_ROOT / "classical" / "tf_idf_plus_linear_svm.joblib"
            self.display_name = "TF-IDF + Linear SVM"
        elif "naive" in sub_model or "bayes" in sub_model:
            model_file = CHECKPOINTS_ROOT / "classical" / "tf_idf_plus_multinomial_naive_bayes.joblib"
            self.display_name = "TF-IDF + Multinomial Naive Bayes"
        elif "forest" in sub_model or "rf" in sub_model:
            model_file = CHECKPOINTS_ROOT / "classical" / "tf_idf_plus_random_forest.joblib"
            self.display_name = "TF-IDF + Random Forest"
        elif "majority" in sub_model:
            model_file = CHECKPOINTS_ROOT / "classical" / "majority_baseline.joblib"
            self.display_name = "Majority Class Baseline"
        elif "corrected" in sub_model:
            corr_file = CHECKPOINTS_ROOT / "corrected_experiments" / "classical" / "tf_idf_plus_logistic_regression.joblib"
            model_file = corr_file
            self.display_name = "TF-IDF + Logistic Regression (Corrected)"
        else:
            # Default: Logistic Regression (Better validated candidate kept as default per requirement 5)
            best_ckpt = CHECKPOINTS_ROOT / "best_model" / "model.joblib"
            if best_ckpt.is_file():
                model_file = best_ckpt
                self.display_name = "TF-IDF + Logistic Regression"
            else:
                model_file = CHECKPOINTS_ROOT / "classical" / "tf_idf_plus_logistic_regression.joblib"
                self.display_name = "TF-IDF + Logistic Regression"

        if not model_file.is_file():
            raise FileNotFoundError(f"Classical model checkpoint not found: {model_file}")

        self.model = joblib.load(model_file)
        self.model_file = model_file
        self.checkpoint_path = model_file

    def _init_transformer_text(self):
        """Loads Hugging Face sequence classification transformer model and tokenizer."""
        sub_model = (self.model_name or "xlm_roberta").lower()
        if self.checkpoint_path and self.checkpoint_path.is_dir():
            ckpt_dir = self.checkpoint_path
            self.display_name = "MuRIL (Fine-Tuned Text)" if "muril" in str(ckpt_dir).lower() else "XLM-RoBERTa (Fine-Tuned Text)" 
        elif "muril" in sub_model:
            ckpt_dir = CHECKPOINTS_ROOT / "muril"
            self.display_name = "MuRIL (Fine-Tuned Text)"
        else:
            ckpt_dir = CHECKPOINTS_ROOT / "xlm_roberta"
            self.display_name = "XLM-RoBERTa (Fine-Tuned Text)"

        if not ckpt_dir.is_dir():
            raise FileNotFoundError(f"Transformer model checkpoint directory not found: {ckpt_dir}")

        self.tokenizer = AutoTokenizer.from_pretrained(str(ckpt_dir), local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(str(ckpt_dir), local_files_only=True)
        self.model.to(self.device)
        self.model.eval()
        self.ckpt_dir = ckpt_dir
        self.checkpoint_path = ckpt_dir

    def _init_image(self):
        """Loads vision classifier with pretrained=False to avoid ImageNet downloads."""
        sub_model = (self.model_name or "resnet50").lower()
        if self.checkpoint_path and self.checkpoint_path.is_file():
            ckpt_path = self.checkpoint_path
            sub_model = ckpt_path.parent.name
        else:
            corr_img = CHECKPOINTS_ROOT / "corrected_experiments" / "resnet50" / "best_model.pt"
            if "efficient" in sub_model:
                ckpt_path = CHECKPOINTS_ROOT / "efficientnet_b0" / "best_model.pt"
            elif "mobile" in sub_model:
                ckpt_path = CHECKPOINTS_ROOT / "mobilenet_v3" / "best_model.pt"
            elif corr_img.is_file() and ("resnet" in sub_model or not sub_model):
                ckpt_path = corr_img
                self.display_name = "ResNet-50 (Corrected Vision Classifier)"
            else:
                ckpt_path = CHECKPOINTS_ROOT / "resnet50" / "best_model.pt"

        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Image model checkpoint not found: {ckpt_path}")

        if "efficient" in sub_model:
            self.model = EfficientNetB0Classifier(num_classes=3, pretrained=False)
            self.transform = get_image_transforms("efficientnet", is_training=False)
            self.display_name = "EfficientNet-B0 (Vision Classifier)"
        elif "mobile" in sub_model:
            self.model = MobileNetV3LargeClassifier(num_classes=3, pretrained=False)
            self.transform = get_image_transforms("resnet50", is_training=False)
            self.display_name = "MobileNet-V3 (Vision Classifier)"
        else:
            self.model = ResNet50Classifier(num_classes=3, pretrained=False)
            self.transform = get_image_transforms("resnet50", is_training=False)
            self.display_name = "ResNet-50 (Vision Classifier)"

        state = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(state["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
        self.ckpt_path = ckpt_path
        self.checkpoint_path = ckpt_path

    def _init_multimodal(self):
        """Loads multimodal fusion architecture with pretrained=False."""
        sub_model = (self.model_name or "gated_fusion").lower()
        if self.checkpoint_path and self.checkpoint_path.is_file():
            ckpt_path = self.checkpoint_path
            sub_model = ckpt_path.parent.name
        else:
            corr_mm = CHECKPOINTS_ROOT / "corrected_experiments" / "gated_fusion" / "best_model.pt"
            if "concat" in sub_model or "muril_resnet" in sub_model:
                ckpt_path = CHECKPOINTS_ROOT / "multimodal" / "muril_resnet50" / "best_model.pt"
            elif "muril_efficient" in sub_model:
                ckpt_path = CHECKPOINTS_ROOT / "multimodal" / "muril_efficientnet" / "best_model.pt"
            elif "multitask" in sub_model or "emotion" in sub_model:
                final_prod_ckpt = CHECKPOINTS_ROOT / "final_production" / "best_model.pt"
                if final_prod_ckpt.is_file():
                    ckpt_path = final_prod_ckpt
                else:
                    ckpt_path = CHECKPOINTS_ROOT / "multitask_emotion" / "best_model.pt"
                self.display_name = "MuRIL + ResNet-50 (Multi-Task Sentiment & Emotion Fusion)"
            elif "xlm" in sub_model:
                ckpt_path = CHECKPOINTS_ROOT / "multimodal" / "xlm_resnet50" / "best_model.pt"
            elif corr_mm.is_file() and ("gated" in sub_model or not sub_model):
                ckpt_path = corr_mm
                self.display_name = "MuRIL + ResNet-50 (Corrected Gated Fusion)"
            else:
                ckpt_path = CHECKPOINTS_ROOT / "multimodal" / "gated_fusion" / "best_model.pt"

        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Multimodal checkpoint not found: {ckpt_path}")

        # Local tokenizer directories to guarantee 100% offline operation
        muril_dir = str(CHECKPOINTS_ROOT / "muril")
        xlm_dir = str(CHECKPOINTS_ROOT / "xlm_roberta")

        # Instantiate appropriate architecture
        if "multitask" in sub_model or "multitask_emotion" in str(ckpt_path) or "final_production" in str(ckpt_path):
            self.tokenizer = AutoTokenizer.from_pretrained(muril_dir, local_files_only=True)
            self.transform = get_image_transforms("resnet50", is_training=False)
            self.model = MultiTaskGatedMultimodalClassifier(
                text_backbone=muril_dir,
                vision_backbone="resnet50",
                num_classes=3,
                num_emotions=6,
                pretrained_text=False,
                pretrained_vision=False
            )
            self.display_name = "MuRIL + ResNet-50 (Multi-Task Sentiment & Emotion Fusion)"
        elif "gated" in sub_model or "gated_fusion" in str(ckpt_path):
            self.tokenizer = AutoTokenizer.from_pretrained(muril_dir, local_files_only=True)
            self.transform = get_image_transforms("resnet50", is_training=False)
            self.model = GatedMultimodalClassifier(
                text_backbone=muril_dir,
                vision_backbone="resnet50",
                num_classes=3,
                pretrained_text=False,
                pretrained_vision=False
            )
            self.display_name = "MuRIL + ResNet-50 (Gated Fusion)"
        elif "muril_efficient" in str(ckpt_path):
            self.tokenizer = AutoTokenizer.from_pretrained(muril_dir, local_files_only=True)
            self.transform = get_image_transforms("efficientnet", is_training=False)
            self.model = MultimodalConcatClassifier(
                text_backbone=muril_dir,
                vision_backbone="efficientnet_b0",
                num_classes=3,
                pretrained_text=False,
                pretrained_vision=False
            )
            self.display_name = "MuRIL + EfficientNet-B0 (Concat Fusion)"
        elif "xlm" in str(ckpt_path):
            self.tokenizer = AutoTokenizer.from_pretrained(xlm_dir, local_files_only=True)
            self.transform = get_image_transforms("resnet50", is_training=False)
            self.model = MultimodalConcatClassifier(
                text_backbone=xlm_dir,
                vision_backbone="resnet50",
                num_classes=3,
                pretrained_text=False,
                pretrained_vision=False
            )
            self.display_name = "XLM-RoBERTa + ResNet-50 (Concat Fusion)"
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(muril_dir, local_files_only=True)
            self.transform = get_image_transforms("resnet50", is_training=False)
            self.model = MultimodalConcatClassifier(
                text_backbone=muril_dir,
                vision_backbone="resnet50",
                num_classes=3,
                pretrained_text=False,
                pretrained_vision=False
            )
            self.display_name = "MuRIL + ResNet-50 (Concat Fusion)"

        state = torch.load(ckpt_path, map_location=self.device)
        # If model checkpoint contains emotion classifier heads but model is not multi-task, upgrade architecture
        if any(k.startswith("emotion_classifier") for k in state["model_state_dict"].keys()) and not isinstance(self.model, MultiTaskGatedMultimodalClassifier):
            self.model = MultiTaskGatedMultimodalClassifier(
                text_backbone=muril_dir,
                vision_backbone="resnet50",
                num_classes=3,
                num_emotions=6,
                pretrained_text=False,
                pretrained_vision=False
            )
            self.display_name = "MuRIL + ResNet-50 (Multi-Task Sentiment & Emotion Fusion)"

        self.model.load_state_dict(state["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
        self.ckpt_path = ckpt_path
        self.checkpoint_path = ckpt_path

    def predict(
        self,
        text: Optional[str] = None,
        image_path: Optional[Union[str, Path]] = None
    ) -> Dict[str, Any]:
        """
        Executes end-to-end sentiment classification with validation and latency profiling.

        Returns:
            dict containing:
            - predicted_sentiment: str ("Negative", "Neutral", "Positive")
            - predicted_class: int (0, 1, 2)
            - confidence: float (0.0 to 1.0)
            - confidence_percent: float (0.0 to 100.0)
            - probabilities: dict of class probabilities
            - model_name: str display name
            - model_mode: str ("classical", "transformer_text", "image", "multimodal")
            - device: str ("cuda" or "cpu")
            - timings_ms: dict of preprocessing, model_execution, and total ms
        """
        if self.mode == "classical":
            return self._predict_classical(text)
        elif self.mode == "transformer_text":
            return self._predict_transformer(text)
        elif self.mode == "image":
            return self._predict_image(image_path)
        elif self.mode == "multimodal":
            return self._predict_multimodal(text, image_path)

    def _predict_classical(self, text: Optional[str]) -> Dict[str, Any]:
        # Validation
        cleaned_text = validate_text_input(text, "classical")

        t_prep_start = time.perf_counter()
        # Input ready for sklearn pipeline
        input_list = [cleaned_text]
        t_prep_end = time.perf_counter()
        prep_ms = (t_prep_end - t_prep_start) * 1000.0

        t_exec_start = time.perf_counter()
        probs = self.model.predict_proba(input_list)[0]
        pred_class = int(np.argmax(probs))
        t_exec_end = time.perf_counter()
        exec_ms = (t_exec_end - t_exec_start) * 1000.0

        return self._format_result(probs, pred_class, prep_ms, exec_ms)

    def _predict_transformer(self, text: Optional[str]) -> Dict[str, Any]:
        # Validation
        cleaned_text = validate_text_input(text, "transformer_text")

        t_prep_start = time.perf_counter()
        inputs = self.tokenizer(
            cleaned_text,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt"
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        t_prep_end = time.perf_counter()
        prep_ms = (t_prep_end - t_prep_start) * 1000.0

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        t_exec_start = time.perf_counter()

        with torch.inference_mode():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
            pred_class = int(np.argmax(probs))

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        t_exec_end = time.perf_counter()
        exec_ms = (t_exec_end - t_exec_start) * 1000.0

        return self._format_result(probs, pred_class, prep_ms, exec_ms)

    def _predict_image(self, image_path: Optional[Union[str, Path]]) -> Dict[str, Any]:
        # Validation
        img = validate_and_load_image(image_path, "image")

        t_prep_start = time.perf_counter()
        pixel_values = self.transform(img).unsqueeze(0).to(self.device)
        t_prep_end = time.perf_counter()
        prep_ms = (t_prep_end - t_prep_start) * 1000.0

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        t_exec_start = time.perf_counter()

        with torch.inference_mode():
            logits = self.model(pixel_values)
            probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
            pred_class = int(np.argmax(probs))

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        t_exec_end = time.perf_counter()
        exec_ms = (t_exec_end - t_exec_start) * 1000.0

        return self._format_result(probs, pred_class, prep_ms, exec_ms)

    def _predict_multimodal(self, text: Optional[str], image_path: Optional[Union[str, Path]]) -> Dict[str, Any]:
        # Validation - BOTH modalities strictly required
        cleaned_text = validate_text_input(text, "multimodal")
        img = validate_and_load_image(image_path, "multimodal")

        t_prep_start = time.perf_counter()
        inputs = self.tokenizer(
            cleaned_text,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt"
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        pixel_values = self.transform(img).unsqueeze(0).to(self.device)
        t_prep_end = time.perf_counter()
        prep_ms = (t_prep_end - t_prep_start) * 1000.0

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        t_exec_start = time.perf_counter()

        with torch.inference_mode():
            out = self.model(input_ids, attention_mask, pixel_values)
            if isinstance(out, tuple) and len(out) == 2:
                logits, emotion_logits = out
                em_probs = torch.sigmoid(emotion_logits).squeeze(0).cpu().numpy()
            elif isinstance(out, dict) and "sentiment_logits" in out:
                logits = out["sentiment_logits"]
                emotion_logits = out.get("emotion_logits")
                em_probs = torch.sigmoid(emotion_logits).squeeze(0).cpu().numpy() if emotion_logits is not None else None
            else:
                logits = out
                em_probs = None
            probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
            pred_class = int(np.argmax(probs))

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        t_exec_end = time.perf_counter()
        exec_ms = (t_exec_end - t_exec_start) * 1000.0

        return self._format_result(probs, pred_class, prep_ms, exec_ms, em_probs=em_probs)

    def _format_result(
        self,
        probs: np.ndarray,
        pred_class: int,
        prep_ms: float,
        exec_ms: float,
        em_probs: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Assembles standard structured response dictionary."""
        conf = float(probs[pred_class])
        total_ms = prep_ms + exec_ms

        result = {
            "predicted_sentiment": LABEL_MAP[pred_class],
            "predicted_class": pred_class,
            "confidence": round(conf, 4),
            "confidence_percent": round(conf * 100.0, 2),
            "probabilities": {
                "Negative": round(float(probs[0]), 4),
                "Neutral": round(float(probs[1]), 4),
                "Positive": round(float(probs[2]), 4)
            },
            "model_name": self.display_name,
            "model_mode": self.mode,
            "device": str(self.device),
            "timings_ms": {
                "preprocessing": round(prep_ms, 2),
                "model_execution": round(exec_ms, 2),
                "total": round(total_ms, 2)
            }
        }

        if em_probs is not None:
            # Multi-label Six-Emotion Detection
            em_dict = {}
            for i, name in enumerate(SIX_EMOTIONS):
                em_dict[name] = round(float(em_probs[i]) * 100.0, 2)

            dom_idx = int(np.argmax(em_probs))
            dom_name = SIX_EMOTIONS[dom_idx]
            dom_prob = round(float(em_probs[dom_idx]) * 100.0, 2)
            dom_emoji = EMOTION_EMOJIS.get(dom_name, "🎭")

            result["emotion"] = {
                "is_available": True,
                "dominant_emotion": dom_name,
                "dominant_probability": dom_prob,
                "dominant_emoji": dom_emoji,
                "probabilities": em_dict,
                "message": "Dominant emotion detected via multi-task multimodal model."
            }
        else:
            result["emotion"] = {
                "is_available": False,
                "dominant_emotion": None,
                "dominant_probability": None,
                "dominant_emoji": None,
                "probabilities": None,
                "message": "Emotion detection not loaded for this model mode."
            }

        return result


# =====================================================================
# Command Line Interface
# =====================================================================
def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser(
        description="IndicSent-M Unified Sentiment Inference Engine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="multimodal",
        choices=["multimodal", "classical", "transformer_text", "image"],
        help="Inference mode: 'multimodal', 'classical', 'transformer_text', or 'image'."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model architecture name (e.g. 'gated_fusion', 'muril_resnet50', 'xlm_roberta', 'muril', 'resnet50', 'efficientnet_b0', 'logistic_regression')."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional direct path to model checkpoint (.pt or .joblib file or directory)."
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Hinglish OCR meme text."
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Absolute or relative file path to the meme image."
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda", "cuda:0"],
        help="Target execution device (defaults to CUDA if available, else CPU)."
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON response for programmatic piping."
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    try:
        predictor = IndicSentPredictor(
            mode=args.mode,
            model_name=args.model,
            checkpoint_path=args.checkpoint,
            device=args.device
        )
        result = predictor.predict(text=args.text, image_path=args.image)

        if args.json:
            print(json.dumps(result, indent=2))
            return

        print("\n" + "=" * 60)
        print("INDICSENT-M SENTIMENT INFERENCE RESULT")
        print("=" * 60)
        print(f"Model       : {result['model_name']}")
        print(f"Mode        : {result['model_mode']}")
        print(f"Device      : {result['device']}")
        print("-" * 60)
        print(f"Prediction  : {result['predicted_sentiment'].upper()} (Class {result['predicted_class']})")
        print(f"Confidence  : {result['confidence_percent']}%")
        print("Class Probabilities:")
        for sentiment, prob in result["probabilities"].items():
            bar_len = int(prob * 30)
            bar = "#" * bar_len + "-" * (30 - bar_len)
            print(f"  [{bar}] {sentiment:8s}: {prob * 100.0:5.2f}%")
        print("-" * 60)
        print("Latency Breakdown:")
        print(f"  Preprocessing   : {result['timings_ms']['preprocessing']:6.2f} ms")
        print(f"  Model Execution : {result['timings_ms']['model_execution']:6.2f} ms")
        print(f"  Total Latency   : {result['timings_ms']['total']:6.2f} ms")
        print("=" * 60 + "\n")

    except Exception as exc:
        if args.json:
            print(json.dumps({"error": str(exc)}, indent=2))
        else:
            print(f"\n[ERROR] Inference failed: {exc}\n", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()