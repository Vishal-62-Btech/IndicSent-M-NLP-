"""
IndicSent-M: Meme Text Suggestions Router
"""

from fastapi import APIRouter, Depends, UploadFile, File, status
from app.schemas.suggestion import SuggestionResponse
from app.services.suggestion_service import SuggestionService, get_suggestion_service
from app.services.predictor_service import PredictorService, get_predictor_service

router = APIRouter(prefix="/suggestions", tags=["Suggestions"])

@router.post(
    "/meme-text",
    response_model=SuggestionResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate Meme-Based Hinglish Text Suggestions",
    description="Analyzes uploaded meme image. If exact match found in Memotion 3.0, retrieves verified dataset caption. Otherwise, analyzes visual content and generates 2-3 relevant Hinglish meme phrases."
)
@router.post(
    "/suggest",
    response_model=SuggestionResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate Meme-Based Hinglish Text Suggestions (Alias)",
    description="Alias for /meme-text endpoint."
)
def suggest_meme_text(
    file: UploadFile = File(..., description="Uploaded meme image file"),
    suggestion_service: SuggestionService = Depends(get_suggestion_service),
    predictor_service: PredictorService = Depends(get_predictor_service)
) -> SuggestionResponse:
    with predictor_service.validate_and_save_upload(file) as temp_image_path:
        return suggestion_service.get_suggestions(temp_image_path)
