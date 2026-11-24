"""
IndicSent-M Backend: Models Catalog Tests
"""

def test_models_catalog_endpoint(client):
    """Verify /api/models returns complete categorized models catalog."""
    response = client.get("/api/models")
    assert response.status_code == 200
    data = response.json()
    assert "modes" in data
    modes = data["modes"]

    # Verify multimodal models
    assert "multimodal" in modes
    mm_mode = modes["multimodal"]
    assert mm_mode["default"] in ["multitask_emotion", "gated_fusion"]
    mm_ids = [m["id"] for m in mm_mode["models"]]
    assert "gated_fusion" in mm_ids or "multitask_emotion" in mm_ids
    assert "muril_resnet50" in mm_ids

    # Verify text models
    assert "text" in modes
    text_mode = modes["text"]
    text_ids = [m["id"] for m in text_mode["models"]]
    assert "xlm_roberta" in text_ids
    assert "logistic_regression" in text_ids

    # Verify image models
    assert "image" in modes
    img_mode = modes["image"]
    img_ids = [m["id"] for m in img_mode["models"]]
    assert "resnet50" in img_ids
    assert "efficientnet_b0" in img_ids
