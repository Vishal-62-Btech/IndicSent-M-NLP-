"""
IndicSent-M: OCR Response Schemas
"""

from typing import Optional
from pydantic import BaseModel, Field


class OcrResponse(BaseModel):
    success: bool = Field(..., description="Whether OCR processing executed successfully")
    text: str = Field(default="", description="Extracted text from meme image")
    confidence: float = Field(default=0.0, description="Confidence score of text extraction (0.0 to 1.0)")
    has_text: bool = Field(default=False, description="True if non-empty text was recognized")
    message: Optional[str] = Field(default=None, description="Informational message or empty state reason")
