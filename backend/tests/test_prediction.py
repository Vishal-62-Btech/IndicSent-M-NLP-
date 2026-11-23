"""
IndicSent-M Backend: Prediction Endpoints & Validation Tests
"""

import tempfile
from pathlib import Path


def _assert_prediction_structure(data: dict, expected_mode: str):
    """Helper asserting standard prediction response structure."""
    assert data["success"] is True
    assert "prediction" in data
    pred = data["prediction"]

    assert pred["predicted_sentiment"] in ["Negative", "Neutral", "Positive"]
    assert pred["predicted_class_id"] in [0, 1, 2]
    assert "confidence_scores" in pred
    scores = pred["confidence_scores"]
    assert "Negative" in scores
    assert "Neutral" in scores
    assert "Positive" in scores
    assert 0.0 <= scores["Negative"] <= 100.0
    assert 0.0 <= scores["Neutral"] <= 100.0
    assert 0.0 <= scores["Positive"] <= 100.0

    assert "model_name" in pred
    assert pred["mode"] == expected_mode
    assert pred["preprocessing_time_ms"] >= 0.0
    assert pred["model_execution_time_ms"] >= 0.0
    assert pred["total_time_ms"] >= 0.0


# -----------------------------------------------------------------------------
# 1. Text Prediction Tests
# -----------------------------------------------------------------------------
def test_text_prediction_default(client):
    """Test text prediction using XLM-RoBERTa model."""
    response = client.post(
        "/api/predict/text",
        json={"text": "Wah bhai kya mast meme hai maza aa gaya", "model": "xlm_roberta"}
    )
    assert response.status_code == 200
    data = response.json()
    _assert_prediction_structure(data, "transformer_text")
    assert "XLM-RoBERTa" in data["prediction"]["model_name"]


def test_text_prediction_classical(client):
    """Test text prediction using classical Logistic Regression model."""
    response = client.post(
        "/api/predict/text",
        json={
            "text": "Bohot bekar joke tha ye",
            "model": "logistic_regression"
        }
    )
    assert response.status_code == 200
    data = response.json()
    _assert_prediction_structure(data, "classical")
    assert "Logistic Regression" in data["prediction"]["model_name"]


def test_text_prediction_empty_fails(client):
    """Test that empty or whitespace-only text returns HTTP 400."""
    response = client.post(
        "/api/predict/text",
        json={"text": "    "}
    )
    assert response.status_code == 400
    data = response.json()
    assert data["success"] is False
    assert data["error"]["code"] == "BAD_REQUEST"


def test_text_prediction_invalid_model_fails(client):
    """Test that requesting an unknown text model returns HTTP 400."""
    response = client.post(
        "/api/predict/text",
        json={
            "text": "Valid Hinglish meme text",
            "model": "unknown_quantum_model_999"
        }
    )
    assert response.status_code == 400
    data = response.json()
    assert data["success"] is False
    assert data["error"]["code"] == "MODEL_NOT_FOUND"


# -----------------------------------------------------------------------------
# 2. Image Prediction Tests
# -----------------------------------------------------------------------------
def test_image_prediction_valid(client, valid_image_bytes):
    """Test image prediction with valid JPEG upload."""
    files = {"file": ("test_meme.jpg", valid_image_bytes, "image/jpeg")}
    response = client.post("/api/predict/image", files=files)
    assert response.status_code == 200
    data = response.json()
    _assert_prediction_structure(data, "image")
    assert "ResNet-50" in data["prediction"]["model_name"]


def test_image_prediction_corrupt_fails(client, corrupt_image_bytes):
    """Test that a corrupt image file returns HTTP 400."""
    files = {"file": ("corrupt.jpg", corrupt_image_bytes, "image/jpeg")}
    response = client.post("/api/predict/image", files=files)
    assert response.status_code == 400
    data = response.json()
    assert data["success"] is False
    assert data["error"]["code"] == "BAD_REQUEST"
    assert "Corrupted or invalid" in data["error"]["message"]


def test_image_prediction_unsupported_mime_fails(client):
    """Test that uploading a text file as image returns HTTP 415."""
    files = {"file": ("script.txt", b"print('hello world')", "text/plain")}
    response = client.post("/api/predict/image", files=files)
    assert response.status_code == 415
    data = response.json()
    assert data["success"] is False
    assert data["error"]["code"] == "UNSUPPORTED_MEDIA_TYPE"


def test_image_prediction_oversized_fails(client, oversized_image_bytes):
    """Test that uploading a file larger than 10MB returns HTTP 413."""
    files = {"file": ("huge_image.jpg", oversized_image_bytes, "image/jpeg")}
    response = client.post("/api/predict/image", files=files)
    assert response.status_code == 413
    data = response.json()
    assert data["success"] is False
    assert data["error"]["code"] == "FILE_TOO_LARGE"


# -----------------------------------------------------------------------------
# 3. Multimodal Prediction Tests
# -----------------------------------------------------------------------------
def test_multimodal_prediction_valid(client, valid_image_bytes):
    """Test multimodal prediction combining text and image."""
    files = {"file": ("meme.jpg", valid_image_bytes, "image/jpeg")}
    data = {"text": "Troll Bollywood Memes BRACE YOURSELVES MITRON"}
    response = client.post("/api/predict/multimodal", files=files, data=data)
    assert response.status_code == 200
    res_data = response.json()
    _assert_prediction_structure(res_data, "multimodal")
    assert "Gated Fusion" in res_data["prediction"]["model_name"]


def test_multimodal_prediction_missing_text_fails(client, valid_image_bytes):
    """Test that multimodal prediction with empty text returns HTTP 400 or 422."""
    files = {"file": ("meme.jpg", valid_image_bytes, "image/jpeg")}
    data = {"text": "   "}
    response = client.post("/api/predict/multimodal", files=files, data=data)
    assert response.status_code in [400, 422]
    res_data = response.json()
    assert res_data["success"] is False


def test_multimodal_prediction_missing_image_fails(client):
    """Test that multimodal prediction without image returns HTTP 422."""
    data = {"text": "Valid meme text without image"}
    response = client.post("/api/predict/multimodal", data=data)
    assert response.status_code == 422
    res_data = response.json()
    assert res_data["success"] is False


def test_temp_file_cleanup_after_request(client, valid_image_bytes):
    """Verify that temporary files are deleted after prediction completes."""
    temp_dir = Path(tempfile.gettempdir())
    files_before = set(temp_dir.glob("tmp*.jpg"))

    files = {"file": ("cleanup_test.jpg", valid_image_bytes, "image/jpeg")}
    data = {"text": "Testing temp file cleanup"}
    response = client.post("/api/predict/multimodal", files=files, data=data)
    assert response.status_code == 200

    files_after = set(temp_dir.glob("tmp*.jpg"))
    # No orphaned temporary files created by our upload handler
    orphans = files_after - files_before
    assert len(orphans) == 0, f"Temporary file leak detected: {orphans}"
