"""
IndicSent-M: Tests for Meme Text Suggestions Endpoint
"""

import io
from PIL import Image

def test_suggestions_endpoint_with_synthetic_image(client):
    """Verifies that an uploaded non-dataset image generates template-based visual suggestions."""
    img = Image.new("RGB", (200, 200), color=(120, 200, 150))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    response = client.post(
        "/api/suggestions/meme-text",
        files={"file": ("synthetic_meme.jpg", buf.getvalue(), "image/jpeg")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["is_dataset_match"] is False
    assert len(data["suggestions"]) >= 2
    for item in data["suggestions"]:
        assert "text" in item
        assert item["source"] == "ai_suggestion"
        assert item["badge"] in ("Visual Tone Template", "AI Visual Suggestion")

def test_suggestions_alias_endpoint(client):
    """Verifies that the /api/suggestions/suggest alias functions identically to /meme-text."""
    img = Image.new("RGB", (200, 200), color=(120, 200, 150))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    response = client.post(
        "/api/suggestions/suggest",
        files={"file": ("synthetic_meme.jpg", buf.getvalue(), "image/jpeg")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert len(data["suggestions"]) >= 2

def test_suggestions_requires_file(client):
    """Verifies 422 validation error when file is missing."""
    response = client.post("/api/suggestions/meme-text")
    assert response.status_code == 422
