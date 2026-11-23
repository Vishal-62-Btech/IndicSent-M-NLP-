# IndicSent-M: REST API Documentation

## 1. Overview & Base URL

The **IndicSent-M** backend is a high-performance REST API built with FastAPI, designed for real-time sentiment analysis of Hinglish memes using text, image, and multimodal deep learning pipelines.

- **Base URL:** `http://127.0.0.1:8000`
- **Interactive OpenAPI Documentation (Swagger UI):** `http://127.0.0.1:8000/docs`
- **Alternative ReDoc UI:** `http://127.0.0.1:8000/redoc`
- **OpenAPI Schema Specification:** `http://127.0.0.1:8000/openapi.json`

---

## 2. Global Request & Response Specifications

### Headers
- For JSON endpoints (`/api/predict/text`): `Content-Type: application/json`
- For file upload endpoints (`/api/predict/image`, `/api/predict/multimodal`): `Content-Type: multipart/form-data`

### Error Response Format
All error responses adhere to a consistent JSON envelope hiding internal tracebacks:
```json
{
  "success": false,
  "error": {
    "code": "BAD_REQUEST",
    "message": "Input text cannot be empty or purely whitespace."
  }
}
```

---

## 3. API Endpoints

### 3.1 System Health & Hardware Status
Retrieves current service health, active execution device, and PyTorch VRAM consumption.

- **Method:** `GET`
- **Path:** `/api/health`
- **Authentication:** None

#### Response (HTTP 200 OK):
```json
{
  "status": "healthy",
  "service": "IndicSent-M Backend",
  "version": "1.0.0",
  "device": "cuda",
  "cuda_available": true,
  "gpu_name": "NVIDIA GeForce RTX 5050 Laptop GPU",
  "vram_allocated_mb": 2196.07,
  "models_loaded": 4
}
```

#### cURL Example:
```bash
curl -X GET "http://127.0.0.1:8000/api/health"
```

---

### 3.2 Model Catalog
Lists all registered models, their active operational status, input modality, and defaults.

- **Method:** `GET`
- **Path:** `/api/models`
- **Authentication:** None

#### Response (HTTP 200 OK):
```json
{
  "modes": {
    "multimodal": {
      "default": "gated_fusion",
      "models": [
        {
          "id": "gated_fusion",
          "name": "MuRIL + ResNet-50 (Gated Fusion)",
          "modality": "multimodal",
          "is_default": true,
          "description": "Adaptive sigmoid attention gate balancing text and vision"
        },
        {
          "id": "concat_fusion",
          "name": "MuRIL + ResNet-50 (Concat Fusion)",
          "modality": "multimodal",
          "is_default": false,
          "description": "Direct feature concatenation baseline"
        }
      ]
    },
    "text": {
      "default": "xlm_roberta",
      "models": [
        {
          "id": "xlm_roberta",
          "name": "XLM-RoBERTa Base",
          "modality": "text",
          "is_default": true,
          "description": "Multilingual RoBERTa fine-tuned on Hinglish OCR"
        },
        {
          "id": "muril",
          "name": "MuRIL Base Cased",
          "modality": "text",
          "is_default": false,
          "description": "Google's Multilingual Representations for Indian Languages"
        },
        {
          "id": "logistic_regression",
          "name": "TF-IDF + Logistic Regression",
          "modality": "text",
          "is_default": false,
          "description": "Ultra-fast classical baseline (0.48 ms latency)"
        }
      ]
    },
    "image": {
      "default": "resnet50",
      "models": [
        {
          "id": "resnet50",
          "name": "ResNet-50",
          "modality": "image",
          "is_default": true,
          "description": "Deep Residual Network vision classifier"
        },
        {
          "id": "efficientnet_b0",
          "name": "EfficientNet-B0",
          "modality": "image",
          "is_default": false,
          "description": "Compound scaled vision network"
        },
        {
          "id": "mobilenet_v3",
          "name": "MobileNetV3-Large",
          "modality": "image",
          "is_default": false,
          "description": "High-efficiency mobile vision network"
        }
      ]
    }
  }
}
```

#### cURL Example:
```bash
curl -X GET "http://127.0.0.1:8000/api/models"
```

---

### 3.3 Text Sentiment Prediction
Analyzes sentiment polarity of a text sequence or OCR punchline.

- **Method:** `POST`
- **Path:** `/api/predict/text`
- **Content-Type:** `application/json`

#### Request Body:
```json
{
  "text": "Wah bhai kya mast meme banaya hai maza aa gaya!",
  "model_name": "xlm_roberta"
}
```
*Note: `model_name` is optional. Defaults to `xlm_roberta`.*

#### Response (HTTP 200 OK):
```json
{
  "success": true,
  "prediction": {
    "predicted_sentiment": "Positive",
    "confidence_scores": {
      "Negative": 4.12,
      "Neutral": 9.45,
      "Positive": 86.43
    },
    "model_used": "XLM-RoBERTa",
    "mode": "text",
    "inference_time_ms": 9.70,
    "total_time_ms": 11.25
  }
}
```

#### cURL Example:
```bash
curl -X POST "http://127.0.0.1:8000/api/predict/text" \
     -H "Content-Type: application/json" \
     -d '{"text": "Bhai kal exam hai aur abhi tak syllabus bhi nahi khola", "model_name": "xlm_roberta"}'
```

---

### 3.4 Image Sentiment Prediction
Analyzes visual facial expressions and scene tone from an uploaded meme graphic.

- **Method:** `POST`
- **Path:** `/api/predict/image`
- **Content-Type:** `multipart/form-data`

#### Parameters:
- `file` (*required*, File): Binary image file (`.jpg`, `.jpeg`, `.png`, `.webp`, max 10 MB).
- `model_name` (*optional*, Form field): Model identifier (`resnet50`, `efficientnet_b0`, `mobilenet_v3`). Defaults to `resnet50`.

#### Response (HTTP 200 OK):
```json
{
  "success": true,
  "prediction": {
    "predicted_sentiment": "Neutral",
    "confidence_scores": {
      "Negative": 18.35,
      "Neutral": 62.14,
      "Positive": 19.51
    },
    "model_used": "ResNet-50",
    "mode": "image",
    "inference_time_ms": 14.16,
    "total_time_ms": 18.82
  }
}
```

#### cURL Example:
```bash
curl -X POST "http://127.0.0.1:8000/api/predict/image" \
     -F "file=@/path/to/meme.jpg" \
     -F "model_name=resnet50"
```

---

### 3.5 Multimodal Sentiment Prediction
Performs simultaneous text and image feature extraction and passes the representations through adaptive Gated Multimodal Fusion.

- **Method:** `POST`
- **Path:** `/api/predict/multimodal`
- **Content-Type:** `multipart/form-data`

#### Parameters:
- `file` (*required*, File): Binary image file (`.jpg`, `.jpeg`, `.png`, `.webp`, max 10 MB).
- `text` (*required*, Form field): Hinglish text punchline or meme caption.
- `model_name` (*optional*, Form field): `gated_fusion` or `concat_fusion`. Defaults to `gated_fusion`.

#### Response (HTTP 200 OK):
```json
{
  "success": true,
  "prediction": {
    "predicted_sentiment": "Negative",
    "confidence_scores": {
      "Negative": 78.21,
      "Neutral": 14.35,
      "Positive": 7.44
    },
    "model_used": "MuRIL + ResNet-50 (Gated Fusion)",
    "mode": "multimodal",
    "inference_time_ms": 25.51,
    "total_time_ms": 32.18
  }
}
```

#### cURL Example:
```bash
curl -X POST "http://127.0.0.1:8000/api/predict/multimodal" \
     -F "file=@/path/to/meme.jpg" \
     -F "text=Smiling face when professor assigns 50 pages reading over weekend" \
     -F "model_name=gated_fusion"
```

---

## 4. HTTP Status Code Reference

| Status Code | Meaning | When Triggered |
|:---:|:---|:---|
| `200 OK` | Success | Valid inference request successfully processed. |
| `400 Bad Request` | Client Error | Empty text input, whitespace-only text, invalid image data, or unparseable payload. |
| `413 Content Too Large` | Entity Size Error | Uploaded file size exceeds the 10 MB limit. |
| `415 Unsupported Media Type` | Format Error | Uploaded file MIME type or extension is not in `[.jpg, .jpeg, .png, .webp]`. |
| `422 Unprocessable Content` | Validation Error | Missing mandatory parameters (`text` or `file`) in request. |
| `500 Internal Server Error` | Server Fault | Unexpected inference failure during model computation. |
| `503 Service Unavailable` | Model Offline | Requested model checkpoint failed to initialize into memory. |
