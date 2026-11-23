"""
IndicSent-M: Pydantic Request & Response Schemas
"""

from typing import Dict, Optional, Any, List
from pydantic import BaseModel, Field


class ConfidenceScores(BaseModel):
    Negative: float = Field(..., description="Probability of negative sentiment in percent (0-100)")
    Neutral: float = Field(..., description="Probability of neutral sentiment in percent (0-100)")
    Positive: float = Field(..., description="Probability of positive sentiment in percent (0-100)")


class EmotionScores(BaseModel):
    Happiness: float = Field(..., description="Probability of Happiness emotion in percent (0-100)")
    Sadness: float = Field(..., description="Probability of Sadness emotion in percent (0-100)")
    Anger: float = Field(..., description="Probability of Anger emotion in percent (0-100)")
    Fear: float = Field(..., description="Probability of Fear emotion in percent (0-100)")
    Disgust: float = Field(..., description="Probability of Disgust emotion in percent (0-100)")
    Surprise: float = Field(..., description="Probability of Surprise emotion in percent (0-100)")


class EmotionData(BaseModel):
    is_available: bool = Field(False, description="Whether emotion detection is available for this prediction")
    dominant_emotion: Optional[str] = Field(None, description="Dominant emotion with highest predicted probability")
    dominant_probability: Optional[float] = Field(None, description="Confidence percentage of the dominant emotion (0-100)")
    dominant_emoji: Optional[str] = Field(None, description="Emoji representing the dominant emotion")
    probabilities: Optional[EmotionScores] = Field(None, description="Multi-label probability scores for all six emotions")
    message: Optional[str] = Field(None, description="Status or information message regarding emotion detection")


class PredictionData(BaseModel):
    predicted_sentiment: str = Field(..., description="Sentiment classification ('Negative', 'Neutral', 'Positive')")
    predicted_class_id: int = Field(..., description="Class integer ID (0=Negative, 1=Neutral, 2=Positive)")
    confidence_scores: ConfidenceScores = Field(..., description="Calibrated sentiment probabilities")
    model_name: str = Field(..., description="Human-readable model name")
    mode: str = Field(..., description="Inference modality ('classical', 'transformer_text', 'image', 'multimodal')")
    preprocessing_time_ms: float = Field(..., description="Preprocessing latency in milliseconds")
    model_execution_time_ms: float = Field(..., description="Model forward execution latency in milliseconds")
    total_time_ms: float = Field(..., description="End-to-end inference latency in milliseconds")
    device: Optional[str] = Field(None, description="Computation device used ('cpu' or 'cuda')")
    emotion: Optional[EmotionData] = Field(
        default_factory=lambda: EmotionData(is_available=False, message="Emotion detection not loaded for this model mode."),
        description="Six-emotion multi-label detection analysis"
    )


class PredictionResponse(BaseModel):
    success: bool = Field(True, description="Request success flag")
    prediction: PredictionData = Field(..., description="Prediction payload")


class TextPredictionRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        description="Raw Hinglish/Hindi-English meme OCR text",
        json_schema_extra={"example": "Wah bhai kya mast meme hai maza aa gaya"}
    )
    model: Optional[str] = Field(
        None,
        description="Optional model ID (e.g. 'xlm_roberta', 'muril', 'logistic_regression', 'linear_svm', 'multinomial_naive_bayes', 'random_forest')",
        json_schema_extra={"example": "xlm_roberta"}
    )


class HealthResponse(BaseModel):
    status: str = Field(..., description="Backend operational status")
    version: str = Field(..., description="Application version")
    device: str = Field(..., description="Active compute device ('cuda' or 'cpu')")
    gpu_available: bool = Field(..., description="Whether CUDA GPU acceleration is active")
    gpu_name: Optional[str] = Field(None, description="Name of the GPU if available")
    models_loaded: Dict[str, bool] = Field(..., description="Dictionary of loaded model modes")
    active_default_model: str = Field(..., description="Default champion model loaded on startup")


class ModelInfo(BaseModel):
    id: str = Field(..., description="Model identifier for API requests")
    name: str = Field(..., description="Display title")
    status: str = Field(..., description="Availability ('available' or 'unavailable')")
    description: Optional[str] = Field(None, description="Model architectural details")
    type: Optional[str] = Field(None, description="Model family type")


class ModeModels(BaseModel):
    default: str = Field(..., description="Default model ID for this mode")
    models: List[ModelInfo] = Field(..., description="List of supported models")


class ModelsResponse(BaseModel):
    modes: Dict[str, ModeModels] = Field(..., description="Supported inference modes and their models")
