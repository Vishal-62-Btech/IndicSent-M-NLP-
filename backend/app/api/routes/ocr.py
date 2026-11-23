"""
IndicSent-M: OCR API Router
===========================
Provides automated OCR extraction for meme images.
"""

from pathlib import Path
from fastapi import APIRouter, Depends, UploadFile, File, status
from app.schemas.ocr import OcrResponse
from app.services.predictor_service import PredictorService, get_predictor_service
from app.services.ocr_service import OcrService, get_ocr_service

router = APIRouter(prefix="/ocr", tags=["OCR"])


@router.post(
    "/extract",
    response_model=OcrResponse,
    summary="Extract Text From Meme Image",
    description="Automatically extracts Hinglish/English text from an uploaded meme image using offline EasyOCR."
)
def extract_ocr_text(
    file: UploadFile = File(..., description="Uploaded meme image file"),
    predictor_service: PredictorService = Depends(get_predictor_service),
    ocr_service: OcrService = Depends(get_ocr_service)
) -> OcrResponse:
    """
    Validates uploaded image and extracts text.
    Always returns a structured OcrResponse, ensuring downstream sentiment prediction is never blocked.
    """
    try:
        with predictor_service.validate_and_save_upload(file) as temp_image_path:
            return ocr_service.extract_text(image_path=temp_image_path)
    except Exception as exc:
        return OcrResponse(
            success=False,
            text="",
            confidence=0.0,
            has_text=False,
            message=f"OCR processing failed: {str(exc)}"
        )
