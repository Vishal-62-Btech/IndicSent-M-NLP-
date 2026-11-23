from typing import List, Optional
from pydantic import BaseModel, Field

class TextSuggestion(BaseModel):
    text: str = Field(..., description="Suggested Hinglish meme phrase")
    source: str = Field(..., description="'dataset' or 'ai_suggestion'")
    badge: str = Field(..., description="Badge label to display in UI")
    confidence: Optional[float] = Field(None, description="Confidence percentage if applicable")

class SuggestionResponse(BaseModel):
    success: bool
    is_dataset_match: bool = Field(False, description="Whether exact hash matched Memotion 3.0 dataset")
    dataset_name: Optional[str] = Field(None, description="Dataset name if matched")
    image_filename: Optional[str] = Field(None, description="Dataset image filename if matched")
    message: str
    suggestions: List[TextSuggestion] = Field(default_factory=list)
