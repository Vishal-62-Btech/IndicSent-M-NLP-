"""
IndicSent-M: Predictor Service & Model Lifecycle Manager
=========================================================
Manages singleton instances of IndicSentPredictor across modes,
handles safe file upload verification, and provides uniform prediction formatting.
"""

import os
import sys
import shutil
import tempfile
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Set
from contextlib import contextmanager
from PIL import Image
from fastapi import UploadFile
import torch

from app.core.config import settings
from app.core.errors import (
    BadRequestException,
    FileTooLargeException,
    UnsupportedMediaTypeException,
    ModelNotFoundException,
    ModelUnavailableException,
    InferenceException
)
from app.schemas.prediction import (
    PredictionResponse,
    PredictionData,
    ConfidenceScores,
    EmotionScores,
    EmotionData,
    HealthResponse,
    ModelsResponse,
    ModeModels,
    ModelInfo
)

# Ensure model_training/src is in Python path
if str(settings.MODEL_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(settings.MODEL_SRC_DIR))

from inference.predictor import IndicSentPredictor

logger = logging.getLogger("indicsent.service")

# Model Registry Configuration
TEXT_MODELS_MAP = {
    "xlm_roberta": ("transformer_text", "xlm_roberta", "XLM-RoBERTa (Fine-Tuned)"),
    "muril": ("transformer_text", "muril", "MuRIL (Fine-Tuned)"),
    "logistic_regression": ("classical", "logistic_regression", "TF-IDF + Logistic Regression"),
    "logistic_regression_corrected": ("classical", "logistic_regression_corrected", "TF-IDF + Logistic Regression (Corrected)"),
    "linear_svm": ("classical", "linear_svm", "TF-IDF + Linear SVM"),
    "multinomial_naive_bayes": ("classical", "multinomial_naive_bayes", "TF-IDF + Multinomial Naive Bayes"),
    "random_forest": ("classical", "random_forest", "TF-IDF + Random Forest"),
}

IMAGE_MODELS_MAP = {
    "resnet50": ("image", "resnet50", "ResNet-50 (Corrected Vision Classifier)"),
    "efficientnet_b0": ("image", "efficientnet_b0", "EfficientNet-B0 (Vision Classifier)"),
    "mobilenet_v3": ("image", "mobilenet_v3", "MobileNet-V3 (Vision Classifier)"),
}

MULTIMODAL_MODELS_MAP = {
    "gated_fusion": ("multimodal", "gated_fusion", "MuRIL + ResNet-50 (Corrected Gated Fusion)"),
    "multitask_emotion": ("multimodal", "multitask_emotion", "MuRIL + ResNet-50 (Multi-Task Sentiment & Emotion Fusion)"),
    "muril_resnet50": ("multimodal", "muril_resnet50", "MuRIL + ResNet-50 (Concat Fusion)"),
    "muril_efficientnet": ("multimodal", "muril_efficientnet", "MuRIL + EfficientNet-B0 (Concat Fusion)"),
    "xlm_resnet50": ("multimodal", "xlm_resnet50", "XLM-RoBERTa + ResNet-50 (Concat Fusion)"),
}


class PredictorService:
    """
    Singleton service maintaining pre-initialized model instances
    and handling validated inference executions.
    """
    _instance: Optional["PredictorService"] = None

    def __init__(self):
        self._predictors: Dict[Tuple[str, str], IndicSentPredictor] = {}
        self.device = settings.DEVICE
        self.is_initialized = False

    @classmethod
    def get_instance(cls) -> "PredictorService":
        if cls._instance is None:
            cls._instance = PredictorService()
        return cls._instance

    def initialize(self):
        """
        Preloads default models during application startup lifespan.
        Loads default Multimodal, Transformer Text, Classical, and Vision models once.
        """
        logger.info(f"Initializing PredictorService on device: {self.device}")
        
        # 1. Preload Default Multimodal (Champion Model)
        try:
            self._get_or_create_predictor("multimodal", settings.DEFAULT_MULTIMODAL_MODEL)
            logger.info("--> Successfully preloaded Multimodal Gated Fusion model.")
        except Exception as e:
            logger.error(f"Failed to preload Multimodal model: {e}")

        # 2. Preload Default Text Model (XLM-RoBERTa)
        try:
            self._get_or_create_predictor("transformer_text", settings.DEFAULT_TEXT_MODEL)
            logger.info("--> Successfully preloaded XLM-RoBERTa text model.")
        except Exception as e:
            logger.error(f"Failed to preload Transformer Text model: {e}")

        # 3. Preload Default Classical Model (Logistic Regression)
        try:
            self._get_or_create_predictor("classical", settings.DEFAULT_CLASSICAL_MODEL)
            logger.info("--> Successfully preloaded Classical Logistic Regression model.")
        except Exception as e:
            logger.error(f"Failed to preload Classical model: {e}")

        # 4. Preload Default Vision Model (ResNet-50)
        try:
            self._get_or_create_predictor("image", settings.DEFAULT_IMAGE_MODEL)
            logger.info("--> Successfully preloaded ResNet-50 vision model.")
        except Exception as e:
            logger.error(f"Failed to preload Vision model: {e}")

        self.is_initialized = True
        logger.info("PredictorService initialization completed.")

    def cleanup(self):
        """Releases cached predictor resources upon application shutdown."""
        logger.info("Cleaning up PredictorService resources...")
        self._predictors.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.is_initialized = False

    def _get_or_create_predictor(self, mode: str, model_name: str) -> IndicSentPredictor:
        """Retrieves an existing predictor from memory or initializes it once."""
        cache_key = (mode, model_name)
        if cache_key in self._predictors:
            return self._predictors[cache_key]

        try:
            predictor = IndicSentPredictor(
                mode=mode,
                model_name=model_name,
                device=self.device
            )
            self._predictors[cache_key] = predictor
            return predictor
        except FileNotFoundError as e:
            raise ModelUnavailableException(f"Checkpoint for model '{model_name}' ({mode}) was not found: {e}")
        except Exception as e:
            raise ModelUnavailableException(f"Failed to initialize model '{model_name}' ({mode}): {e}")

    # =========================================================================
    # Prediction Handlers
    # =========================================================================
    def predict_text(self, text: str, model_id: Optional[str] = None) -> PredictionResponse:
        """Processes text sentiment prediction."""
        if not text or not text.strip():
            raise BadRequestException("Text input cannot be empty or whitespace-only.")

        target_model = (model_id or settings.DEFAULT_TEXT_MODEL).lower().strip()
        if target_model not in TEXT_MODELS_MAP:
            valid_models = ", ".join(list(TEXT_MODELS_MAP.keys()))
            raise ModelNotFoundException(f"Unknown text model '{model_id}'. Supported text models are: {valid_models}")

        mode, sub_model, _ = TEXT_MODELS_MAP[target_model]
        predictor = self._get_or_create_predictor(mode, sub_model)

        try:
            raw_res = predictor.predict(text=text)
            return self._format_prediction_response(raw_res)
        except ValueError as e:
            raise BadRequestException(str(e))
        except Exception as e:
            logger.error(f"Text prediction error: {e}", exc_info=True)
            raise InferenceException(f"Text prediction failed: {e}")

    def predict_image(self, image_path: Path, model_id: Optional[str] = None) -> PredictionResponse:
        """Processes vision sentiment prediction."""
        target_model = (model_id or settings.DEFAULT_IMAGE_MODEL).lower().strip()
        if target_model not in IMAGE_MODELS_MAP:
            valid_models = ", ".join(list(IMAGE_MODELS_MAP.keys()))
            raise ModelNotFoundException(f"Unknown image model '{model_id}'. Supported image models are: {valid_models}")

        mode, sub_model, _ = IMAGE_MODELS_MAP[target_model]
        predictor = self._get_or_create_predictor(mode, sub_model)

        try:
            raw_res = predictor.predict(image_path=image_path)
            return self._format_prediction_response(raw_res)
        except ValueError as e:
            raise BadRequestException(str(e))
        except Exception as e:
            logger.error(f"Image prediction error: {e}", exc_info=True)
            raise InferenceException(f"Image prediction failed: {e}")

    def predict_multimodal(
        self,
        text: str,
        image_path: Path,
        model_id: Optional[str] = None
    ) -> PredictionResponse:
        """Processes multimodal sentiment prediction combining OCR text and image."""
        if not text or not text.strip():
            raise BadRequestException("Multimodal prediction requires non-empty text input.")

        target_model = (model_id or settings.DEFAULT_MULTIMODAL_MODEL).lower().strip()
        if target_model not in MULTIMODAL_MODELS_MAP:
            valid_models = ", ".join(list(MULTIMODAL_MODELS_MAP.keys()))
            raise ModelNotFoundException(f"Unknown multimodal model '{model_id}'. Supported multimodal models are: {valid_models}")

        mode, sub_model, _ = MULTIMODAL_MODELS_MAP[target_model]
        predictor = self._get_or_create_predictor(mode, sub_model)

        try:
            raw_res = predictor.predict(text=text, image_path=image_path)
            return self._format_prediction_response(raw_res)
        except ValueError as e:
            raise BadRequestException(str(e))
        except Exception as e:
            logger.error(f"Multimodal prediction error: {e}", exc_info=True)
            raise InferenceException(f"Multimodal prediction failed: {e}")

    def _format_prediction_response(self, raw_res: Dict[str, Any]) -> PredictionResponse:
        """Maps internal predictor output to standardized API response format."""
        probs = raw_res["probabilities"]
        timings = raw_res["timings_ms"]

        confidence_scores = ConfidenceScores(
            Negative=round(probs["Negative"] * 100.0, 2),
            Neutral=round(probs["Neutral"] * 100.0, 2),
            Positive=round(probs["Positive"] * 100.0, 2)
        )

        # Multi-label Six-Emotion Detection Block
        raw_em = raw_res.get("emotion")
        if raw_em and raw_em.get("is_available") and raw_em.get("probabilities"):
            em_scores = EmotionScores(**raw_em["probabilities"])
            emotion_data = EmotionData(
                is_available=True,
                dominant_emotion=raw_em["dominant_emotion"],
                dominant_probability=raw_em["dominant_probability"],
                dominant_emoji=raw_em.get("dominant_emoji", "✨"),
                probabilities=em_scores,
                message=raw_em.get("message", "Dominant emotion detected via multi-task model.")
            )
        else:
            emotion_data = EmotionData(
                is_available=False,
                dominant_emotion=None,
                dominant_probability=None,
                dominant_emoji=None,
                probabilities=None,
                message=raw_em.get("message") if raw_em else "Emotion detection is not available for this model mode."
            )

        prediction_data = PredictionData(
            predicted_sentiment=raw_res["predicted_sentiment"],
            predicted_class_id=raw_res["predicted_class"],
            confidence_scores=confidence_scores,
            model_name=raw_res["model_name"],
            mode=raw_res["model_mode"],
            preprocessing_time_ms=timings["preprocessing"],
            model_execution_time_ms=timings["model_execution"],
            total_time_ms=timings["total"],
            device=raw_res.get("device"),
            emotion=emotion_data
        )

        return PredictionResponse(success=True, prediction=prediction_data)

    # =========================================================================
    # Metadata & Health
    # =========================================================================
    def get_health(self) -> HealthResponse:
        """Builds health report with hardware and loaded models status."""
        gpu_avail = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if gpu_avail else None

        models_loaded = {
            "multimodal": any(k[0] == "multimodal" for k in self._predictors),
            "transformer_text": any(k[0] == "transformer_text" for k in self._predictors),
            "classical": any(k[0] == "classical" for k in self._predictors),
            "image": any(k[0] == "image" for k in self._predictors),
        }

        return HealthResponse(
            status="healthy",
            version=settings.VERSION,
            device=self.device,
            gpu_available=gpu_avail,
            gpu_name=gpu_name,
            models_loaded=models_loaded,
            active_default_model="MuRIL + ResNet-50 (Corrected Gated Fusion)"
        )

    def get_models_catalog(self) -> ModelsResponse:
        """Builds metadata catalog of all supported models across all modes."""
        multimodal_list = [
            ModelInfo(
                id=k,
                name=info[2],
                status="available",
                description="Multimodal fusion combining Hinglish text and visual features.",
                type="multimodal"
            )
            for k, info in MULTIMODAL_MODELS_MAP.items()
        ]

        text_list = [
            ModelInfo(
                id=k,
                name=info[2],
                status="available",
                description="Fine-tuned transformer text classifier" if info[0] == "transformer_text" else "Classical scikit-learn TF-IDF pipeline",
                type=info[0]
            )
            for k, info in TEXT_MODELS_MAP.items()
        ]

        image_list = [
            ModelInfo(
                id=k,
                name=info[2],
                status="available",
                description="Convolutional transfer learning image classifier",
                type="vision"
            )
            for k, info in IMAGE_MODELS_MAP.items()
        ]

        return ModelsResponse(
            modes={
                "multimodal": ModeModels(default=settings.DEFAULT_MULTIMODAL_MODEL, models=multimodal_list),
                "text": ModeModels(default=settings.DEFAULT_TEXT_MODEL, models=text_list),
                "image": ModeModels(default=settings.DEFAULT_IMAGE_MODEL, models=image_list),
            }
        )

    # =========================================================================
    # Safe Upload Context Manager
    # =========================================================================
    @staticmethod
    @contextmanager
    def validate_and_save_upload(file: UploadFile):
        """
        Validates uploaded image and safely writes to a temporary file.
        Guarantees deletion of the temporary file upon context exit.
        
        Validations:
        - Filename and extension check (.jpg, .jpeg, .png, .webp)
        - MIME content-type check
        - File size limit (10MB)
        - PIL image integrity check (Image.open().verify())
        """
        if not file or not file.filename:
            raise BadRequestException("No image file was uploaded.")

        filename = file.filename
        ext = os.path.splitext(filename)[1].lower()

        # 1. Extension check
        if ext not in settings.ALLOWED_EXTENSIONS:
            allowed = ", ".join(sorted(settings.ALLOWED_EXTENSIONS))
            raise UnsupportedMediaTypeException(f"Unsupported file extension '{ext}'. Allowed extensions: {allowed}")

        # 2. MIME type check
        content_type = (file.content_type or "").lower()
        if content_type not in settings.ALLOWED_MIME_TYPES:
            allowed_mime = ", ".join(sorted(settings.ALLOWED_MIME_TYPES))
            raise UnsupportedMediaTypeException(f"Unsupported MIME type '{content_type}'. Allowed types: {allowed_mime}")

        # 3. Create temp file
        temp_file = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
        temp_path = Path(temp_file.name)
        total_size = 0

        try:
            # Stream in chunks to enforce size limit safely
            chunk_size = 1024 * 64  # 64 KB
            while True:
                chunk = file.file.read(chunk_size)
                if not chunk:
                    break
                total_size += len(chunk)
                if total_size > settings.MAX_FILE_SIZE_BYTES:
                    raise FileTooLargeException(f"Uploaded file exceeds maximum limit of {settings.MAX_FILE_SIZE_BYTES / (1024 * 1024):.0f} MB.")
                temp_file.write(chunk)
            temp_file.flush()
            temp_file.close()

            if total_size == 0:
                raise BadRequestException("Uploaded file is empty (0 bytes).")

            # 4. PIL integrity verification
            try:
                with Image.open(temp_path) as img:
                    img.verify()
                # Reopen to test format
                with Image.open(temp_path) as img:
                    img.load()
            except Exception as e:
                raise BadRequestException(f"Corrupted or invalid image file. Details: {e}")

            # Yield validated temp path
            yield temp_path

        finally:
            # Guaranteed cleanup
            try:
                if temp_path.exists():
                    temp_path.unlink()
            except Exception as e:
                logger.warning(f"Could not remove temporary file {temp_path}: {e}")


def get_predictor_service() -> PredictorService:
    """Dependency provider for FastAPI route injection."""
    return PredictorService.get_instance()
