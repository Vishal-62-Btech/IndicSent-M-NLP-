"""
IndicSent-M: Prediction Endpoints Router
"""

from typing import Optional
from pathlib import Path
from fastapi import APIRouter, Depends, UploadFile, File, Form, status
from app.schemas.prediction import (
    PredictionResponse,
    TextPredictionRequest
)
from app.services.predictor_service import PredictorService, get_predictor_service

router = APIRouter(prefix="/predict", tags=["Predictions"])


@router.post(
    "/text",
    response_model=PredictionResponse,
    summary="Text Sentiment Analysis",
    description="Analyzes sentiment of Hinglish meme OCR text using fine-tuned Transformers (XLM-RoBERTa, MuRIL) or Classical Baselines."
)
async def predict_text(
    payload: TextPredictionRequest,
    service: PredictorService = Depends(get_predictor_service)
) -> PredictionResponse:
    return service.predict_text(text=payload.text, model_id=payload.model)


@router.post(
    "/image",
    response_model=PredictionResponse,
    summary="Image Sentiment Analysis",
    description="Analyzes sentiment from meme visual content using transfer learning vision classifiers (ResNet-50, EfficientNet-B0, MobileNet-V3)."
)
async def predict_image(
    file: UploadFile = File(..., description="Meme image file (JPEG, PNG, WebP up to 10MB)"),
    model: Optional[str] = Form(None, description="Optional vision model ID ('resnet50', 'efficientnet_b0', 'mobilenet_v3')"),
    service: PredictorService = Depends(get_predictor_service)
) -> PredictionResponse:
    with service.validate_and_save_upload(file) as temp_image_path:
        return service.predict_image(image_path=temp_image_path, model_id=model)


@router.post(
    "/multimodal",
    response_model=PredictionResponse,
    summary="Multimodal Meme Sentiment Analysis",
    description="Analyzes meme sentiment combining both Hinglish text and visual features via MuRIL + ResNet-50 Gated Fusion or Concat Fusion."
)
async def predict_multimodal(
    text: str = Form(..., min_length=1, description="Hinglish/Hindi-English meme OCR text"),
    file: UploadFile = File(..., description="Meme image file (JPEG, PNG, WebP up to 10MB)"),
    model: Optional[str] = Form(None, description="Optional multimodal model ID ('gated_fusion', 'muril_resnet50', 'muril_efficientnet', 'xlm_resnet50')"),
    service: PredictorService = Depends(get_predictor_service)
) -> PredictionResponse:
    with service.validate_and_save_upload(file) as temp_image_path:
        return service.predict_multimodal(text=text, image_path=temp_image_path, model_id=model)
