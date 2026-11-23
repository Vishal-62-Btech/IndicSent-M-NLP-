"""
IndicSent-M Backend: Health & Root Endpoint Tests
"""

def test_root_endpoint(client):
    """Verify root / endpoint returns API info and documentation link."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "name" in data
    assert "version" in data
    assert data["docs_url"] == "/docs"
    assert "endpoints" in data


def test_health_endpoint(client):
    """Verify /api/health endpoint returns hardware and model status."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "device" in data
    assert "gpu_available" in data
    assert "models_loaded" in data
    assert isinstance(data["models_loaded"], dict)
    assert "active_default_model" in data
