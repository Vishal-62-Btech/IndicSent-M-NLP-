# IndicSent-M: FastAPI Backend Service

Production-ready backend API for **IndicSent-M: Multimodal Sentiment Analysis of Indian-Language Memes Using Text and Images**.

Powered by FastAPI, PyTorch, Hugging Face Transformers, torchvision, and scikit-learn.

---

## 1. Architecture Overview

- **Modality Support:**
  - **Text Sentiment Analysis:** Fine-tuned XLM-RoBERTa, MuRIL, and Classical ML baselines (Logistic Regression, Linear SVM, Naive Bayes, Random Forest).
  - **Visual Sentiment Analysis:** Deep Transfer Learning via ResNet-50, EfficientNet-B0, and MobileNet-V3.
  - **Multimodal Meme Sentiment Analysis:** MuRIL + ResNet-50 Gated Fusion and Feature Concatenation architectures.
- **Model Lifecycle:** Singleton initialization via FastAPI `lifespan` manager—models are preloaded once at application startup.
- **Safe File Uploads:** Maximum 10MB file limit, MIME type validation, PIL integrity verification, streaming chunk writes, and guaranteed temporary file cleanup.
- **Hardware Acceleration:** Native NVIDIA CUDA GPU acceleration when available, with deterministic fallback to CPU.

---

## 2. Directory Layout

```text
backend/
├── app/
│   ├── main.py                     # FastAPI entry point & lifespan configuration
│   ├── api/
│   │   └── routes/
│   │       ├── health.py           # GET /api/health endpoint
│   │       ├── models.py           # GET /api/models catalog endpoint
│   │       └── prediction.py       # POST /api/predict/* endpoints
│   ├── core/
│   │   ├── config.py               # Application settings & CORS
│   │   └── errors.py               # Custom exceptions & handlers
│   ├── schemas/
│   │   └── prediction.py           # Pydantic request & response models
│   └── services/
│       └── predictor_service.py    # Predictor singleton & file validation
├── tests/
│   ├── conftest.py                 # Pytest fixtures & FastAPI TestClient
│   ├── test_health.py              # Health check test cases
│   ├── test_models.py              # Model catalog test cases
│   └── test_prediction.py          # Prediction & validation error test cases
├── requirements.txt                # Python backend dependencies
└── README.md                       # This documentation
```

---

## 3. Installation & Setup

From the `backend` directory:

```bash
pip install -r requirements.txt
```

---

## 4. Running the Server

### Standard Startup
```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

### Development Mode with Auto-Reload
```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Interactive Swagger API Documentation:
- Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- ReDoc: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

## 5. API Reference

### A. Health Check
`GET /api/health`

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "device": "cuda",
  "gpu_available": true,
  "gpu_name": "NVIDIA GeForce RTX 5050 Laptop GPU",
  "models_loaded": {
    "multimodal": true,
    "transformer_text": true,
    "classical": true,
    "image": true
  },
  "active_default_model": "MuRIL + ResNet-50 (Gated Fusion)"
}
```

### B. Supported Models Catalog
`GET /api/models`

Returns all supported models across `multimodal`, `text`, and `image` modes.

### C. Text Sentiment Analysis
`POST /api/predict/text`

**Request:**
```json
{
  "text": "Wah bhai kya mast meme banaya hai maza aa gaya",
  "model": "xlm_roberta"
}
```

**Response:**
```json
{
  "success": true,
  "prediction": {
    "predicted_sentiment": "Positive",
    "predicted_class_id": 2,
    "confidence_scores": {
      "Negative": 12.3,
      "Neutral": 25.4,
      "Positive": 62.3
    },
    "model_name": "XLM-RoBERTa (Fine-Tuned Text)",
    "mode": "transformer_text",
    "preprocessing_time_ms": 1.25,
    "model_execution_time_ms": 18.42,
    "total_time_ms": 19.67,
    "device": "cuda"
  }
}
```

### D. Image Sentiment Analysis
`POST /api/predict/image`

Multipart Form:
- `file`: Image file (`.jpg`, `.png`, `.webp` up to 10MB)
- `model`: Optional (`resnet50`, `efficientnet_b0`, `mobilenet_v3`)

### E. Multimodal Meme Sentiment Analysis
`POST /api/predict/multimodal`

Multipart Form:
- `text`: Hinglish meme OCR text string
- `file`: Meme image file
- `model`: Optional (`gated_fusion`, `muril_resnet50`, `muril_efficientnet`, `xlm_resnet50`)

---

## 6. Running the Test Suite

```bash
python -m pytest tests/ -v
```
