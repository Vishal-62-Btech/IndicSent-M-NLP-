"""
IndicSent-M: Meme-Based Hinglish Text Suggestion Service
Integrates Memotion 3.0 exact dataset annotation retrieval,
EasyOCR on-image text extraction, and vision-guided contextual
AI suggestions for Hinglish memes.
"""

import json
import hashlib
import logging
from pathlib import Path
from typing import Optional, List
from PIL import Image

from app.schemas.suggestion import SuggestionResponse, TextSuggestion
from app.services.predictor_service import PredictorService
from app.services.ocr_service import OcrService

logger = logging.getLogger("indicsent.suggestions")

# Predefined contextual templates mapped to visual sentiment & context
VISUAL_SUGGESTION_BANK = {
    "Positive": [
        "Jab code bina error ke first try me run ho jaye! 🔥",
        "Finally assignment complete ho gaya! 😍",
        "Teacher ne bola well done beta! Shabash! 🎉",
        "Hard work ka result finally dekh kar khushi! ✨",
        "Mera confidence level right now: 100/100 😎"
    ],
    "Negative": [
        "Padhai chhod ke sab ho raha hai life me, bas syllabus complete nahi ho raha 😭",
        "Code phir se unexpected error de raha hai 😭",
        "Har baar mere sath hi aisa kyun hota hai? 🫠",
        "Monday morning struggle shuru ho gaya ⏰",
        "Mera result dekhne ke baad ghar ka mahaul 💀"
    ],
    "Neutral": [
        "Kal subah exam hai aur main yahan chill kar raha hoon 🥱",
        "Main aur meri normal daily routine 🚶‍♂️",
        "Dekhte hain ab aage kya naya twist aata hai 🤔",
        "Bas chal raha hai life me, jaise chalna chahiye 😐",
        "Aaj kuch khaas nahi hua, bas timepass ☕"
    ]
}

class SuggestionService:
    """
    Service for generating Hinglish meme text suggestions:
    1. Exact hash matching against Memotion 3.0 dataset.
    2. OCR detection of on-image meme caption text.
    3. Vision-guided contextual Hinglish suggestions.
    """
    _instance: Optional["SuggestionService"] = None

    def __init__(self):
        self.hash_index = {}
        self.index_path = Path(__file__).resolve().parent / "memotion_hash_index.json"
        self._load_index()

    @classmethod
    def get_instance(cls) -> "SuggestionService":
        if cls._instance is None:
            cls._instance = SuggestionService()
        return cls._instance

    def _load_index(self):
        if self.index_path.is_file():
            try:
                with open(self.index_path, "r", encoding="utf-8") as f:
                    self.hash_index = json.load(f)
                logger.info(f"Loaded {len(self.hash_index)} Memotion 3.0 dataset hashes.")
            except Exception as e:
                logger.error(f"Failed to load Memotion hash index: {e}")
                self.hash_index = {}
        else:
            logger.warning(f"Memotion hash index file not found at: {self.index_path}")
            self.hash_index = {}

    def get_suggestions(self, image_path: Path) -> SuggestionResponse:
        """
        Analyzes the uploaded image:
        - If matched in Memotion 3.0 -> returns verified dataset annotation.
        - If text is detected on image -> provides detected text + contextual Hinglish adaptations.
        - If no text detected -> uses ResNet-50 visual feature sentiment to offer contextual suggestions.
        - If image cannot be understood -> returns message asking user to enter text manually.
        """
        if not image_path or not image_path.is_file():
            return SuggestionResponse(
                success=False,
                is_dataset_match=False,
                message="Image file not found. Please upload a valid meme image.",
                suggestions=[]
            )

        # 1. Check for Exact Dataset Match via File MD5
        try:
            image_bytes = image_path.read_bytes()
            img_hash = hashlib.md5(image_bytes).hexdigest()
        except Exception as e:
            logger.error(f"Error hashing image file: {e}")
            return SuggestionResponse(
                success=False,
                is_dataset_match=False,
                message="Could not read uploaded image. Please enter text manually.",
                suggestions=[]
            )

        if img_hash in self.hash_index:
            entry = self.hash_index[img_hash]
            caption = entry.get("caption", "").strip()
            filename = entry.get("image_filename", "dataset_image.jpg")
            
            logger.info(f"Exact Memotion 3.0 dataset match: {filename} (hash: {img_hash})")
            return SuggestionResponse(
                success=True,
                is_dataset_match=True,
                dataset_name="Memotion 3.0",
                image_filename=filename,
                message=f"Verified match found in Memotion 3.0 dataset ({filename}).",
                suggestions=[
                    TextSuggestion(
                        text=caption,
                        source="dataset",
                        badge="Verified Dataset Annotation",
                        confidence=100.0
                    )
                ]
            )

        # Verify image format with PIL
        try:
            with Image.open(image_path) as img:
                img.verify()
        except Exception as e:
            logger.warning(f"Image verification failed: {e}")
            return SuggestionResponse(
                success=False,
                is_dataset_match=False,
                message="Could not read image content. Please enter text manually.",
                suggestions=[]
            )

        # 2. Visual Content Tone Analysis using ResNet-50
        predicted_sentiment = "Neutral"
        confidence_val = 50.0
        try:
            predictor_service = PredictorService.get_instance()
            vision_res = predictor_service.predict_image(image_path=image_path, model_id="resnet50")
            predicted_sentiment = vision_res.prediction.predicted_sentiment
            conf_scores = vision_res.prediction.confidence_scores
            confidence_val = getattr(conf_scores, predicted_sentiment, 50.0)
        except Exception as e:
            logger.warning(f"Vision model prediction warning during suggestion: {e}")

        # 3. Check for On-Image Text via OCR
        detected_text = ""
        ocr_conf = 0.0
        try:
            ocr_service = OcrService.get_instance()
            ocr_res = ocr_service.extract_text(image_path)
            if ocr_res.has_text and ocr_res.text.strip():
                detected_text = ocr_res.text.strip()
                ocr_conf = ocr_res.confidence
        except Exception as e:
            logger.warning(f"OCR extraction warning during suggestion: {e}")

        suggestions = []

        if detected_text:
            # Suggestion 1: Exact text detected on the meme image
            suggestions.append(
                TextSuggestion(
                    text=detected_text,
                    source="ocr_detected",
                    badge="Detected Meme Text",
                    confidence=round(ocr_conf * 100.0, 1) if ocr_conf > 0 else 88.0
                )
            )

            # Suggestion 2 & 3: Contextual Hinglish extensions combining text + visual tone
            if predicted_sentiment == "Positive":
                suggestions.append(
                    TextSuggestion(
                        text=f"{detected_text} 🔥 Shabash!",
                        source="ai_contextual",
                        badge="AI Contextual Suggestion",
                        confidence=round(max(confidence_val, 75.0), 1)
                    )
                )
                suggestions.append(
                    TextSuggestion(
                        text=f"Finally {detected_text.lower()}! 😍 Well done!",
                        source="ai_contextual",
                        badge="AI Contextual Suggestion",
                        confidence=round(max(confidence_val - 5.0, 70.0), 1)
                    )
                )
            elif predicted_sentiment == "Negative":
                suggestions.append(
                    TextSuggestion(
                        text=f"{detected_text} 😭 Ye kya ho gaya!",
                        source="ai_contextual",
                        badge="AI Contextual Suggestion",
                        confidence=round(max(confidence_val, 75.0), 1)
                    )
                )
                suggestions.append(
                    TextSuggestion(
                        text=f"Life me bas yahi baaki tha: {detected_text.lower()} 💀",
                        source="ai_contextual",
                        badge="AI Contextual Suggestion",
                        confidence=round(max(confidence_val - 5.0, 70.0), 1)
                    )
                )
            else: # Neutral
                suggestions.append(
                    TextSuggestion(
                        text=f"{detected_text} 🤔 Daily life moments",
                        source="ai_contextual",
                        badge="AI Contextual Suggestion",
                        confidence=round(max(confidence_val, 75.0), 1)
                    )
                )
                suggestions.append(
                    TextSuggestion(
                        text=f"Normal routine scene: {detected_text} 😐",
                        source="ai_contextual",
                        badge="AI Contextual Suggestion",
                        confidence=round(max(confidence_val - 5.0, 70.0), 1)
                    )
                )

            logger.info(f"Generated {len(suggestions)} suggestions combining OCR text and visual sentiment ({predicted_sentiment}).")
            return SuggestionResponse(
                success=True,
                is_dataset_match=False,
                message=f"Text detected on meme with {predicted_sentiment.lower()} visual tone. Generated {len(suggestions)} suggestions.",
                suggestions=suggestions
            )

        # 4. Fallback: Pure visual context templates (when no text is detected on image)
        template_pool = VISUAL_SUGGESTION_BANK.get(predicted_sentiment, VISUAL_SUGGESTION_BANK["Neutral"])
        selected_phrases = template_pool[:3]

        suggestions = [
            TextSuggestion(
                text=phrase,
                source="ai_suggestion",
                badge="Visual Tone Template",
                confidence=round(confidence_val, 1)
            )
            for phrase in selected_phrases
        ]

        logger.info(f"Generated {len(suggestions)} visual suggestions for non-dataset image (visual sentiment: {predicted_sentiment}).")
        return SuggestionResponse(
            success=True,
            is_dataset_match=False,
            message=f"Visual content analyzed ({predicted_sentiment} tone). Generated {len(suggestions)} relevant Hinglish suggestions.",
            suggestions=suggestions
        )

def get_suggestion_service() -> SuggestionService:
    return SuggestionService.get_instance()
