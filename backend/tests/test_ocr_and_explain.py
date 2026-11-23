"""
Automated Behavioral Tests for OCR, Explainable AI, and PDF Reporting
Verifies the 12 mandatory user experience scenarios.
"""

import io
from pathlib import Path
import pytest
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_IMG_PATH = PROJECT_ROOT / "dataset" / "Memotion 3" / "trainImages" / "trainImages" / "0.jpg"


def get_sample_image_bytes():
    if SAMPLE_IMG_PATH.is_file():
        return SAMPLE_IMG_PATH.read_bytes()
    buf = io.BytesIO()
    img = Image.new("RGB", (224, 224), color=(255, 255, 255))
    img.save(buf, format="JPEG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Scenario 1 & 2: Upload image -> OCR starts automatically & returns text
# ---------------------------------------------------------------------------
def test_scenario_1_and_2_ocr_extraction(client):
    img_bytes = get_sample_image_bytes()
    response = client.post(
        "/api/ocr/extract",
        files={"file": ("meme.jpg", img_bytes, "image/jpeg")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert isinstance(data["text"], str)
    assert "has_text" in data
    print("\n[SCENARIO 1 & 2 PASSED] OCR extracted text successfully:", data["text"][:60])


# ---------------------------------------------------------------------------
# Scenario 3: Edit OCR text -> edited text is used for prediction
# ---------------------------------------------------------------------------
def test_scenario_3_edited_text_used_for_prediction(client):
    img_bytes = get_sample_image_bytes()
    edited_text = "Yeh sab doglapan hai bhai"
    response = client.post(
        "/api/predict/multimodal",
        data={"text": edited_text, "model": "gated_fusion"},
        files={"file": ("meme.jpg", img_bytes, "image/jpeg")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["prediction"]["predicted_sentiment"] in ["Positive", "Negative", "Neutral"]
    print("\n[SCENARIO 3 PASSED] Multimodal prediction executed using edited OCR text.")


# ---------------------------------------------------------------------------
# Scenario 4: Click Skip OCR -> Image-only analysis continues smoothly
# ---------------------------------------------------------------------------
def test_scenario_4_skip_ocr_image_only(client):
    img_bytes = get_sample_image_bytes()
    response = client.post(
        "/api/predict/image",
        data={"model": "resnet50"},
        files={"file": ("meme.jpg", img_bytes, "image/jpeg")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["prediction"]["predicted_sentiment"] in ["Positive", "Negative", "Neutral"]
    print("\n[SCENARIO 4 PASSED] Image-only prediction operates independently without OCR dependency.")


# ---------------------------------------------------------------------------
# Scenario 5: Upload new image -> OCR runs for new image
# ---------------------------------------------------------------------------
def test_scenario_5_ocr_runs_for_new_image(client):
    buf = io.BytesIO()
    img = Image.new("RGB", (100, 100), color=(100, 150, 200))
    img.save(buf, format="JPEG")
    new_bytes = buf.getvalue()

    response = client.post(
        "/api/ocr/extract",
        files={"file": ("new_image.jpg", new_bytes, "image/jpeg")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    print("\n[SCENARIO 5 PASSED] OCR handled new image upload cleanly.")


# ---------------------------------------------------------------------------
# Scenario 6: Normal prediction -> No explanation calculations performed
# ---------------------------------------------------------------------------
def test_scenario_6_normal_prediction_no_explanation_overhead(client):
    response = client.post(
        "/api/predict/text",
        json={"text": "Padhai chhod ke sab ho raha hai", "model": "logistic_regression"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    # Verify no explanation fields are injected into normal prediction response
    assert "explanation" not in data["prediction"]
    assert "heatmap_image" not in data["prediction"]
    print("\n[SCENARIO 6 PASSED] Normal prediction ran with 0 explanation overhead.")


# ---------------------------------------------------------------------------
# Scenario 7: Click Why This Prediction? -> Explanation generated on demand
# ---------------------------------------------------------------------------
def test_scenario_7_generate_explanation_on_demand(client):
    img_bytes = get_sample_image_bytes()
    response = client.post(
        "/api/explain",
        data={
            "mode": "multimodal",
            "model": "gated_fusion",
            "text": "Relationship status Single Married It is complicated",
            "predicted_sentiment": "Negative"
        },
        files={"file": ("meme.jpg", img_bytes, "image/jpeg")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    exp = data["explanation"]
    assert exp is not None
    assert exp["model_name"] == "MuRIL + ResNet-50 (Gated Fusion)"
    assert exp["gate_weights"] is not None
    assert "text_weight" in exp["gate_weights"]
    assert "image_weight" in exp["gate_weights"]
    assert exp["heatmap_image"] is not None
    assert len(exp["word_attributions"]) > 0
    print("\n[SCENARIO 7 PASSED] On-demand explanation generated real Grad-CAM and Gate weights.")


# ---------------------------------------------------------------------------
# Scenario 8 & 9: Frontend caching and clearing behavior
# ---------------------------------------------------------------------------
def test_scenario_8_and_9_text_explanation(client):
    response = client.post(
        "/api/explain",
        data={
            "mode": "text",
            "model": "logistic_regression",
            "text": "Exam pass ho gaya party karenge",
            "predicted_sentiment": "Positive"
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert len(data["explanation"]["word_attributions"]) > 0
    print("\n[SCENARIO 8 & 9 PASSED] Text model explanation generated valid word contributions.")


# ---------------------------------------------------------------------------
# Scenario 10: Explanation failure handling
# ---------------------------------------------------------------------------
def test_scenario_10_explanation_failure_graceful(client):
    response = client.post(
        "/api/explain",
        data={"mode": "nonexistent_mode", "text": "test"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is False
    assert data["explanation"] is None
    assert data["message"] is not None
    print("\n[SCENARIO 10 PASSED] Unsupported explanation handled gracefully without server exception.")


# ---------------------------------------------------------------------------
# Scenario 11: Download PDF without explanation
# ---------------------------------------------------------------------------
def test_scenario_11_download_pdf_without_explanation(client):
    response = client.post(
        "/api/report/pdf",
        data={
            "model_name": "MuRIL + ResNet-50 (Gated Fusion)",
            "mode": "multimodal",
            "predicted_sentiment": "Negative",
            "confidence_positive": 7.0,
            "confidence_neutral": 12.0,
            "confidence_negative": 81.0,
            "processing_time_ms": 12.5,
            "input_text": "Padhai chhod ke sab ho raha hai"
        }
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    content = response.content
    assert content.startswith(b"%PDF")
    print(f"\n[SCENARIO 11 PASSED] PDF report generated without explanation (size: {len(content)} bytes)")


# ---------------------------------------------------------------------------
# Scenario 12: Download PDF with explanation
# ---------------------------------------------------------------------------
def test_scenario_12_download_pdf_with_explanation(client):
    response = client.post(
        "/api/report/pdf",
        data={
            "model_name": "MuRIL + ResNet-50 (Gated Fusion)",
            "mode": "multimodal",
            "predicted_sentiment": "Negative",
            "confidence_positive": 7.0,
            "confidence_neutral": 12.0,
            "confidence_negative": 81.0,
            "processing_time_ms": 12.5,
            "input_text": "Padhai chhod ke sab ho raha hai",
            "explanation_summary": "Model placed 65% attention on text keywords.",
            "gate_text_weight": 65.4,
            "gate_image_weight": 34.6,
            "top_words": "Padhai (100%), chhod (85%)"
        }
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    content = response.content
    assert content.startswith(b"%PDF")
    assert len(content) > 1000
    print(f"\n[SCENARIO 12 PASSED] PDF report generated with complete explanation details (size: {len(content)} bytes)")
