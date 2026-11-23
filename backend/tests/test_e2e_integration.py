"""
IndicSent-M: End-to-End Frontend-Backend Integration Test Harness
"""

import sys
import time
import subprocess
from pathlib import Path
import httpx

BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_DIR.parent
TEST_IMAGE_PATH = PROJECT_ROOT / "dataset" / "Memotion 3" / "images" / "train" / "2842.jpg"

print("--- PHASE 11 INTEGRATION VERIFICATION ---", flush=True)

# 1. Start FastAPI server
cmd_backend = [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000"]
print("Starting FastAPI Backend on http://127.0.0.1:8000...", flush=True)
backend_proc = subprocess.Popen(cmd_backend, cwd=str(BACKEND_DIR))

try:
    client = httpx.Client(base_url="http://127.0.0.1:8000", timeout=60.0)
    for i in range(25):
        try:
            r = client.get("/api/health")
            if r.status_code == 200:
                print(f"[PASS] 1. Backend connection verified: {r.json()['status']} on {r.json()['device']}", flush=True)
                break
        except Exception:
            time.sleep(1.0)
    else:
        raise RuntimeError("Failed to connect to FastAPI backend within 25 seconds")

    # 2. Model catalog retrieval
    r_models = client.get("/api/models")
    assert r_models.status_code == 200
    catalog = r_models.json()["modes"]
    assert "multimodal" in catalog and "text" in catalog and "image" in catalog
    print(f"[PASS] 2. Model catalog retrieved successfully. Modes: {list(catalog.keys())}", flush=True)

    # 3. Text prediction
    r_text = client.post("/api/predict/text", json={"text": "Wah bhai kya mast meme hai maza aa gaya"})
    assert r_text.status_code == 200
    res_text = r_text.json()
    assert res_text["success"] is True
    sentiment_txt = res_text["prediction"]["predicted_sentiment"]
    conf_txt = res_text["prediction"]["confidence_scores"][sentiment_txt]
    print(f"[PASS] 3. Text prediction verified: {sentiment_txt} ({conf_txt}%) in {res_text['prediction']['total_time_ms']} ms", flush=True)

    # 4. Image prediction
    with open(TEST_IMAGE_PATH, "rb") as f:
        img_bytes = f.read()
    r_img = client.post("/api/predict/image", files={"file": ("meme.jpg", img_bytes, "image/jpeg")})
    assert r_img.status_code == 200
    res_img = r_img.json()
    assert res_img["success"] is True
    sentiment_img = res_img["prediction"]["predicted_sentiment"]
    conf_img = res_img["prediction"]["confidence_scores"][sentiment_img]
    print(f"[PASS] 4. Image prediction verified: {sentiment_img} ({conf_img}%) in {res_img['prediction']['total_time_ms']} ms", flush=True)

    # 5. Multimodal prediction
    r_mm = client.post(
        "/api/predict/multimodal",
        data={"text": "Troll Bollywood Memes BRACE YOURSELVES MITRON"},
        files={"file": ("meme.jpg", img_bytes, "image/jpeg")}
    )
    assert r_mm.status_code == 200
    res_mm = r_mm.json()
    assert res_mm["success"] is True
    sentiment_mm = res_mm["prediction"]["predicted_sentiment"]
    conf_mm = res_mm["prediction"]["confidence_scores"][sentiment_mm]
    print(f"[PASS] 5. Multimodal prediction verified: {sentiment_mm} ({conf_mm}%) in {res_mm['prediction']['total_time_ms']} ms", flush=True)

    # 6. Unsupported image upload
    r_bad_ext = client.post("/api/predict/image", files={"file": ("script.txt", b"print('test')", "text/plain")})
    assert r_bad_ext.status_code == 415
    print("[PASS] 6. Unsupported image upload properly rejected with HTTP 415 (UNSUPPORTED_MEDIA_TYPE)", flush=True)

    # 7. Empty text validation
    r_empty_txt = client.post("/api/predict/text", json={"text": "   "})
    assert r_empty_txt.status_code == 400
    print("[PASS] 7. Empty text validation properly rejected with HTTP 400 (BAD_REQUEST)", flush=True)

    # 8. Frontend build verification
    dist_html = PROJECT_ROOT / "dist" / "index.html"
    assert dist_html.is_file() and dist_html.stat().st_size > 0
    print(f"[PASS] 8. React production bundle verified ({dist_html.stat().st_size} bytes index.html)", flush=True)

    print("\nALL INTEGRATION TESTS PASSED SUCCESSFULLY!", flush=True)

finally:
    print("Terminating background server...", flush=True)
    backend_proc.terminate()
    backend_proc.wait()
    print("Server terminated cleanly.", flush=True)
