"""
IndicSent-M: Explainable AI Schemas
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class WordAttribution(BaseModel):
    word: str = Field(..., description="Token or word from meme text")
    score: float = Field(..., description="Relative attribution score / importance (0.0 to 1.0 or normalized)")
    raw_attribution: float = Field(default=0.0, description="Raw mathematical contribution value")
    influence: str = Field(default="Supports", description="Whether this word supports or opposes the predicted sentiment")


class GateWeights(BaseModel):
    text_weight: float = Field(..., description="Percentage weight learned by the dynamic attention gate for text")
    image_weight: float = Field(..., description="Percentage weight learned by the dynamic attention gate for visual features")
    dominant_modality: str = Field(..., description="Modality with highest learned influence ('Text' or 'Image')")
    explanation: str = Field(..., description="Plain English description of the gate weights")


class ExplanationData(BaseModel):
    model_name: str = Field(..., description="Name of the model used to generate the explanation")
    mode: str = Field(..., description="Analysis mode ('multimodal', 'text', 'image')")
    predicted_sentiment: str = Field(..., description="Target sentiment class explained ('Negative', 'Neutral', 'Positive')")
    predicted_class_id: int = Field(..., description="Class integer ID (0, 1, 2)")
    word_attributions: Optional[List[WordAttribution]] = Field(default=None, description="Ranking of words by influence on prediction")
    heatmap_image: Optional[str] = Field(default=None, description="Base64 encoded Grad-CAM heatmap overlay PNG")
    gate_weights: Optional[GateWeights] = Field(default=None, description="Learned modality weights for Gated Fusion models")
    summary: str = Field(..., description="Plain-English summary of what influenced the prediction")
    execution_time_ms: float = Field(default=0.0, description="Time taken to calculate explanation in milliseconds")


class ExplanationResponse(BaseModel):
    success: bool = Field(..., description="Whether explanation was generated successfully")
    explanation: Optional[ExplanationData] = Field(default=None, description="Explanation payload if available")
    message: Optional[str] = Field(default=None, description="Informational message or error description")
