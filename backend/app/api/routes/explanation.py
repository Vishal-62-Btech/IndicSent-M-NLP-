"""
IndicSent-M: Explainable AI API Router
======================================
Provides on-demand model interpretability endpoints (Word Attribution, Grad-CAM, Gate Attention).
Never triggered during normal sentiment prediction.
"""

from typing import Optional
from pathlib import Path
from fastapi import APIRouter, Depends, UploadFile, File, Form, status
from app.schemas.explanation import ExplanationResponse
from app.services.predictor_service import PredictorService, get_predictor_service
from app.services.explanation_service import ExplanationService, get_explanation_service

router = APIRouter(prefix="/explain", tags=["Explainable AI"])


@router.post(
    "",
    response_model=ExplanationResponse,
    summary="Generate Model Explanation On-Demand",
    description="Calculates mathematical model explanations: Grad-CAM attention heatmap, token gradient attributions, and dynamic gating weights."
)
def generate_explanation(
    mode: str = Form("multimodal", description="Prediction mode ('multimodal', 'text', 'image')"),
    model: Optional[str] = Form(None, description="Optional model ID"),
    text: Optional[str] = Form(None, description="Meme text or OCR text"),
    file: Optional[UploadFile] = File(None, description="Uploaded meme image"),
    predicted_sentiment: Optional[str] = Form(None, description="Predicted sentiment class to explain"),
    predictor_service: PredictorService = Depends(get_predictor_service),
    explanation_service: ExplanationService = Depends(get_explanation_service)
) -> ExplanationResponse:
    """
    Executes explainability analysis only when explicitly requested by user.
    """
    temp_image_path = None
    if file and file.filename:
        try:
            with predictor_service.validate_and_save_upload(file) as saved_path:
                return explanation_service.explain(
                    mode=mode,
                    model_id=model,
                    text=text,
                    image_path=saved_path,
                    target_sentiment=predicted_sentiment
                )
        except Exception as e:
            return ExplanationResponse(
                success=False,
                explanation=None,
                message=f"Could not read uploaded image for explanation: {str(e)}"
            )

    return explanation_service.explain(
        mode=mode,
        model_id=model,
        text=text,
        image_path=None,
        target_sentiment=predicted_sentiment
    )
