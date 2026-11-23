"""
IndicSent-M: Live Inference Audit Script
========================================
Comprehensive verification of all inference modes, backward compatibility,
emotion model handling, and checkpoint loading.
"""

import sys
import io
import json
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "model_training" / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from inference.predictor import IndicSentPredictor
from fastapi.testclient import TestClient
from app.main import app

img_path = str(PROJECT_ROOT / "dataset" / "Memotion 3" / "images" / "test" / "0.jpg")
sample_text = "Jab presentation khatam ho gaya aur teacher ne shabash bola!"

print("=" * 70)
print("LIVE INFERENCE AUDIT: PREDICTOR, API & BACKWARD COMPATIBILITY")
print("=" * 70)

# 1. Classical Text Only
print("\n1. Testing Classical Text Only (TF-IDF + Logistic Regression):")
p_ml = IndicSentPredictor(mode="classical", model_name="logistic_regression")
res_ml = p_ml.predict(text=sample_text)
print(f"  Model: {res_ml['model_name']}")
print(f"  Sentiment: {res_ml['predicted_sentiment']} ({res_ml['confidence_percent']}%)")
print(f"  Emotion Available: {res_ml['emotion']['is_available']}")
assert res_ml["emotion"]["is_available"] is False, "Expected emotion.is_available=False for text"

# 2. Transformer Text Only
print("\n2. Testing Transformer Text Only (XLM-RoBERTa):")
p_xlm = IndicSentPredictor(mode="transformer_text", model_name="xlm_roberta")
res_xlm = p_xlm.predict(text=sample_text)
print(f"  Model: {res_xlm['model_name']}")
print(f"  Sentiment: {res_xlm['predicted_sentiment']} ({res_xlm['confidence_percent']}%)")
print(f"  Emotion Available: {res_xlm['emotion']['is_available']}")
assert res_xlm["emotion"]["is_available"] is False, "Expected emotion.is_available=False for transformer"

# 3. Vision Only
print("\n3. Testing Vision Only (ResNet-50):")
p_img = IndicSentPredictor(mode="image", model_name="resnet50")
res_img = p_img.predict(image_path=img_path)
print(f"  Model: {res_img['model_name']}")
print(f"  Sentiment: {res_img['predicted_sentiment']} ({res_img['confidence_percent']}%)")
print(f"  Emotion Available: {res_img['emotion']['is_available']}")
assert res_img["emotion"]["is_available"] is False, "Expected emotion.is_available=False for image"

# 4. Multimodal Baseline (gated_fusion) Backward Compatibility
print("\n4. Testing Multimodal Baseline Backward Compatibility (gated_fusion):")
p_base = IndicSentPredictor(mode="multimodal", model_name="gated_fusion")
res_base = p_base.predict(text=sample_text, image_path=img_path)
print(f"  Model: {res_base['model_name']}")
print(f"  Sentiment: {res_base['predicted_sentiment']} ({res_base['confidence_percent']}%)")
print(f"  Emotion Available: {res_base['emotion']['is_available']}")
assert res_base["emotion"]["is_available"] is False, "Expected emotion.is_available=False for gated_fusion"

# 5. Multimodal Multi-Task (multitask_emotion)
print("\n5. Testing Multimodal Multi-Task Emotion Detection (multitask_emotion):")
p_mt = IndicSentPredictor(mode="multimodal", model_name="multitask_emotion")
res_mt = p_mt.predict(text=sample_text, image_path=img_path)
print(f"  Model: {res_mt['model_name']}")
print(f"  Sentiment: {res_mt['predicted_sentiment']} ({res_mt['confidence_percent']}%)")
print(f"  Emotion Available: {res_mt['emotion']['is_available']}")
print(f"  Dominant Emotion: {res_mt['emotion']['dominant_emoji']} {res_mt['emotion']['dominant_emotion']} ({res_mt['emotion']['dominant_probability']}%)")
print(f"  Emotion Breakdown: {res_mt['emotion']['probabilities']}")
assert res_mt["emotion"]["is_available"] is True, "Expected emotion.is_available=True for multitask_emotion"
assert res_mt["emotion"]["dominant_emotion"] is not None
assert len(res_mt["emotion"]["probabilities"]) == 6

# 6. HTTP API Endpoint Verification
print("\n6. Testing FastAPI Endpoints via TestClient:")
client = TestClient(app)

with open(img_path, "rb") as f:
    img_bytes = f.read()

# API Test Multimodal Baseline
resp_b = client.post("/api/predict/multimodal", data={"text": sample_text, "model": "gated_fusion"}, files={"file": ("0.jpg", img_bytes, "image/jpeg")})
assert resp_b.status_code == 200
pred_b = resp_b.json()["prediction"]
print(f"  API gated_fusion -> Status: 200, Sentiment: {pred_b['predicted_sentiment']}, Emotion Available: {pred_b['emotion']['is_available']}")

# API Test Multimodal Multi-Task
resp_m = client.post("/api/predict/multimodal", data={"text": sample_text, "model": "multitask_emotion"}, files={"file": ("0.jpg", img_bytes, "image/jpeg")})
assert resp_m.status_code == 200
pred_m = resp_m.json()["prediction"]
print(f"  API multitask_emotion -> Status: 200, Sentiment: {pred_m['predicted_sentiment']}, Dominant Emotion: {pred_m['emotion']['dominant_emoji']} {pred_m['emotion']['dominant_emotion']} ({pred_m['emotion']['dominant_probability']}%)")

print("\nALL 6 INFERENCE TESTS AND BACKWARD COMPATIBILITY CHECKS SUCCEEDED 100%!")
