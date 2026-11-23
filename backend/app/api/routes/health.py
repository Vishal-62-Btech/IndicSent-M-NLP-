"""
IndicSent-M: Health & Status Router
"""

from fastapi import APIRouter, Depends
from app.schemas.prediction import HealthResponse
from app.services.predictor_service import PredictorService, get_predictor_service

router = APIRouter(prefix="", tags=["System"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Backend Health Check",
    description="Returns backend server health status, GPU availability, and model initialization state."
)
async def get_health(service: PredictorService = Depends(get_predictor_service)) -> HealthResponse:
    return service.get_health()
