"""
IndicSent-M: Supported Models Catalog Router
"""

from fastapi import APIRouter, Depends
from app.schemas.prediction import ModelsResponse
from app.services.predictor_service import PredictorService, get_predictor_service

router = APIRouter(prefix="", tags=["Models"])


@router.get(
    "/models",
    response_model=ModelsResponse,
    summary="List Supported Models",
    description="Returns full catalog of supported sentiment classification models and architectures grouped by modality."
)
async def list_models(service: PredictorService = Depends(get_predictor_service)) -> ModelsResponse:
    return service.get_models_catalog()
