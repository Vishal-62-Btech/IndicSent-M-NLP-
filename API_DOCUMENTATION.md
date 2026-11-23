# IndicSent-M: REST API Specification
=======================================

> **FastAPI Backend Endpoints, Request/Response Schemas, and Error Handling**  
> *Base URL:* `http://127.0.0.1:8000/api/v1`  
> *Interactive Swagger Docs:* `http://127.0.0.1:8000/docs`  
> *ReDoc Documentation:* `http://127.0.0.1:8000/redoc`

---

## 1. System & Metadata Endpoints

### 1.1 Root / Service Status
- **Method:** `GET`
- **Path:** `/`
- **Response:**
  ```json
  {
    "service": "IndicSent-M Multimodal Sentiment Analysis API",
    "version": "1.0.0",
    "status": "online",
    "docs_url": "/docs"
  }
  ```

### 1.2 Health Check
- **Method:** `GET`
- **Path:** `/api/v1/health`
- **Response:**
  ```json
  {
    "status": "healthy",
    "device": "cuda:0",
    "cuda_available": true,
    "active_models": 4
  }
  ```

### 1.3 Model Catalog
- **Method:** `GET`
- **Path:** `/api/v1/models`
- **Description:** Returns the catalog of supported models across Text, Vision, and Multimodal modes.
- **Sample Response:**
  ```json
  {
    "success": true,
    "models": {
      "multimodal": [
        {"id": "multitask_emotion", "name": "MuRIL + ResNet-50 (Multi-Task Sentiment & Emotion Fusion)", "is_default": true},
        {"id": "gated_fusion", "name": "MuRIL + ResNet-50 (Corrected Gated Fusion)", "is_default": false},
        {"id": "muril_resnet50", "name": "MuRIL + ResNet-50 (Concat Fusion)", "is_default": false}
      ],
      "text": [
        {"id": "xlm_roberta", "name": "XLM-RoBERTa (Fine-Tuned)", "is_default": true},
        {"id": "logistic_regression", "name": "TF-IDF + Logistic Regression", "is_default": false}
      ],
      "image": [
        {"id": "resnet50", "name": "ResNet-50 (Corrected Vision Classifier)", "is_default": true}
      ]
    }
  }
  ```

---

## 2. Prediction Endpoints

### 2.1 Multimodal Sentiment & Emotion Prediction
- **Method:** `POST`
- **Path:** `/api/v1/predict/multimodal`
- **Content-Type:** `multipart/form-data`
- **Request Parameters:**
  - `text` (string, required): Cleaned or extracted meme caption in Hinglish.
  - `image` (file, required): Meme image file (`image/jpeg`, `image/png`, `image/webp`, max 10MB).
  - `model_id` (string, optional, default: `"multitask_emotion"`): Target multimodal model identifier.
- **Success Response (200 OK):**
  ```json
  {
    "success": true,
    "prediction": {
      "predicted_sentiment": "Positive",
      "predicted_class_id": 2,
      "confidence_scores": {
        "Negative": 12.45,
        "Neutral": 22.18,
        "Positive": 65.37
      },
      "model_name": "MuRIL + ResNet-50 (Multi-Task Sentiment & Emotion Fusion)",
      "mode": "multimodal",
      "preprocessing_time_ms": 14.2,
      "model_execution_time_ms": 28.5,
      "total_time_ms": 42.7,
      "device": "cuda:0",
      "emotion": {
        "is_available": true,
        "dominant_emotion": "Happiness",
        "dominant_probability": 84.12,
        "dominant_emoji": "😊",
        "probabilities": {
          "Happiness": 84.12,
          "Sadness": 8.35,
          "Anger": 3.12,
          "Fear": 1.45,
          "Disgust": 0.89,
          "Surprise": 15.60
        },
        "message": "Dominant emotion detected via multi-task model."
      }
    }
  }
  ```

### 2.2 Text-Only Prediction
- **Method:** `POST`
- **Path:** `/api/v1/predict/text`
- **Content-Type:** `application/json`
- **Request Body:**
  ```json
  {
    "text": "Bhai presentation bahut mast gayi teacher ne tareef ki! 🥳",
    "model_id": "xlm_roberta"
  }
  ```
- **Success Response (200 OK):**
  ```json
  {
    "success": true,
    "prediction": {
      "predicted_sentiment": "Positive",
      "predicted_class_id": 2,
      "confidence_scores": {
        "Negative": 9.12,
        "Neutral": 23.45,
        "Positive": 67.43
      },
      "model_name": "XLM-RoBERTa (Fine-Tuned)",
      "mode": "text_only",
      "preprocessing_time_ms": 2.1,
      "model_execution_time_ms": 9.8,
      "total_time_ms": 11.9,
      "device": "cuda:0",
      "emotion": {
        "is_available": false,
        "dominant_emotion": null,
        "dominant_probability": null,
        "dominant_emoji": null,
        "probabilities": null,
        "message": "Emotion detection is not available for this model mode."
      }
    }
  }
  ```

### 2.3 Image-Only Prediction
- **Method:** `POST`
- **Path:** `/api/v1/predict/image`
- **Content-Type:** `multipart/form-data`
- **Request Parameters:**
  - `image` (file, required): Meme image file.
  - `model_id` (string, optional, default: `"resnet50"`).

---

## 3. Utility & Explainability Endpoints

### 3.1 OCR Extraction
- **Method:** `POST`
- **Path:** `/api/v1/ocr`
- **Content-Type:** `multipart/form-data`
- **Request Parameters:** `image` (file)
- **Response:**
  ```json
  {
    "success": true,
    "extracted_text": "Jab assignment submit karne ke baad teacher bolti hai 'Well done beta!'",
    "confidence": 0.94,
    "detected_languages": ["hi", "en"]
  }
  ```

### 3.2 Visual Explainability (Grad-CAM)
- **Method:** `POST`
- **Path:** `/api/v1/explain/gradcam`
- **Content-Type:** `multipart/form-data`
- **Request Parameters:** `image` (file), `target_class` (int, optional)
- **Response:**
  ```json
  {
    "success": true,
    "target_layer": "layer4",
    "heatmap_base64": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD...",
    "overlay_base64": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD..."
  }
  ```

### 3.3 Text Explainability (Token Attribution)
- **Method:** `POST`
- **Path:** `/api/v1/explain/tokens`
- **Content-Type:** `application/json`
- **Request Body:**
  ```json
  {
    "text": "Jab assignment submit karne ke baad teacher bolti hai 'Well done beta!'"
  }
  ```
- **Response:** Returns list of token attributions with positive/negative saliency weights.

### 3.4 PDF Report Export
- **Method:** `POST`
- **Path:** `/api/v1/report/pdf`
- **Content-Type:** `application/x-www-form-urlencoded`
- **Parameters:** Includes `model_name`, `mode`, `predicted_sentiment`, `dominant_emotion`, `dominant_emotion_probability`, `dominant_emotion_emoji`, and optional Grad-CAM `heatmap_base64`.
- **Response:** Streams formatted `application/pdf` binary.

---

## 4. Standard Error Responses

- **400 Bad Request:** Empty input or invalid parameters.
  ```json
  {"success": false, "error": {"code": "BAD_REQUEST", "message": "Multimodal prediction requires non-empty text input."}}
  ```
- **404 Model Not Found:** Unregistered model ID requested.
  ```json
  {"success": false, "error": {"code": "MODEL_NOT_FOUND", "message": "Unknown multimodal model 'xyz'."}}
  ```
- **413 Content Too Large:** File exceeds 10 MB limit.
- **415 Unsupported Media Type:** Uploaded file is not JPEG, PNG, or WebP.
- **500 Internal Server Error:** Unhandled exception with logged traceback.
