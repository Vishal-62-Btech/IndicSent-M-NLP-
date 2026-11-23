"""
IndicSent-M Backend: Six-Emotion Detection & Dominant Emotion Tests
==================================================================
Tests for the multi-task emotion detection head extension:
1. MultiTask model prediction returns valid emotion block (is_available=True).
2. Dominant emotion is correctly chosen as the argmax across the 6 emotions.
3. Multi-label independence: all 6 emotions have valid calibrated Sigmoid probabilities (0-100%).
4. Backward compatibility: text-only and classical models return is_available=False.
5. Baseline multimodal model (gated_fusion) returns is_available=False.
6. PDF report generation endpoint accepts emotion fields and successfully renders.
"""

import json
import pytest
from models.multimodal_models import SIX_EMOTIONS, EMOTION_EMOJIS


def test_multitask_emotion_prediction_structure(client, valid_image_bytes):
    """Test that multimodal multitask model returns valid emotion detection data."""
    response = client.post(
        "/api/predict/multimodal",
        data={
            "text": "Wah bhai kya mast joke mara maza aa gaya",
            "model": "multitask_emotion"
        },
        files={"file": ("test.jpg", valid_image_bytes, "image/jpeg")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    pred = data["prediction"]

    # Sentiment check (existing functionality preserved)
    assert pred["predicted_sentiment"] in ["Negative", "Neutral", "Positive"]
    assert pred["predicted_class_id"] in [0, 1, 2]

    # Emotion block check
    assert "emotion" in pred
    em = pred["emotion"]
    assert em["is_available"] is True
    assert em["dominant_emotion"] in SIX_EMOTIONS
    assert 0.0 <= em["dominant_probability"] <= 100.0
    assert em["dominant_emoji"] == EMOTION_EMOJIS[em["dominant_emotion"]]

    # Probabilities check
    probs = em["probabilities"]
    assert probs is not None
    for name in SIX_EMOTIONS:
        assert name in probs
        assert 0.0 <= probs[name] <= 100.0

    # Verify that dominant_emotion matches the maximum probability among the 6
    max_emotion = max(probs, key=probs.get)
    assert em["dominant_emotion"] == max_emotion
    assert abs(em["dominant_probability"] - probs[max_emotion]) < 1e-2


def test_backward_compatibility_text_only_emotion_unavailable(client):
    """Text-only prediction must return is_available=False for emotion block."""
    response = client.post(
        "/api/predict/text",
        json={"text": "Exam stress bohot zyada hai life me", "model": "logistic_regression"}
    )
    assert response.status_code == 200
    data = response.json()
    pred = data["prediction"]
    assert "emotion" in pred
    assert pred["emotion"]["is_available"] is False
    assert pred["emotion"]["dominant_emotion"] is None
    assert pred["emotion"]["probabilities"] is None


def test_backward_compatibility_baseline_gated_fusion_emotion_unavailable(client, valid_image_bytes):
    """Baseline gated_fusion without emotion head must return is_available=False."""
    response = client.post(
        "/api/predict/multimodal",
        data={
            "text": "Neutral test message",
            "model": "gated_fusion"
        },
        files={"file": ("test.jpg", valid_image_bytes, "image/jpeg")}
    )
    assert response.status_code == 200
    data = response.json()
    pred = data["prediction"]
    assert "emotion" in pred
    assert pred["emotion"]["is_available"] is False
    assert pred["emotion"]["dominant_emotion"] is None


def test_pdf_report_with_emotion_data(client):
    """Test generating PDF report containing dominant emotion and 6-emotion breakdown."""
    em_dict = {
        "Happiness": 84.5,
        "Sadness": 2.1,
        "Anger": 1.5,
        "Fear": 3.2,
        "Disgust": 1.0,
        "Surprise": 28.6
    }
    response = client.post(
        "/api/report/pdf",
        data={
            "model_name": "MuRIL + ResNet-50 (Multi-Task Sentiment & Emotion Fusion)",
            "mode": "multimodal",
            "predicted_sentiment": "Positive",
            "confidence_positive": 88.0,
            "confidence_neutral": 8.0,
            "confidence_negative": 4.0,
            "processing_time_ms": 32.5,
            "input_text": "Jab assignment submit ho gaya aur presentation badhiya tha!",
            "dominant_emotion": "Happiness",
            "dominant_emotion_probability": 84.5,
            "dominant_emotion_emoji": "😊",
            "emotion_probabilities": json.dumps(em_dict)
        }
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    content = response.content
    assert len(content) > 1000
    assert content.startswith(b"%PDF-")
