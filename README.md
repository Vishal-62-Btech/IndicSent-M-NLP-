# IndicSent-M: Multimodal Sentiment Analysis for Hinglish Memes

> **Tagline:** *Decoding the Emotional Subtext of Indian Internet Memes*  
> **Subtitle:** *An End-to-End Multimodal Deep Learning & NLP Framework with Gated Attention Fusion*

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CUDA 12](https://img.shields.io/badge/CUDA-12.x-76B900.svg?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React 18](https://img.shields.io/badge/React-18.x-61DAFB.svg?logo=react&logoColor=white)](https://react.dev/)
[![Vite](https://img.shields.io/badge/Vite-5.4-646CFF.svg?logo=vite&logoColor=white)](https://vitejs.dev/)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind-3.4-38B2AC.svg?logo=tailwind-css&logoColor=white)](https://tailwindcss.com/)

---

## 📌 Project Overview

**IndicSent-M** is a full-stack, research-grade multimodal sentiment analysis system specifically built for Indian social media memes. Internet memes in India present unique linguistic and computer vision challenges:

1. **Hinglish Code-Mixing:** Unstandardized Romanized Hindi mixed with English text (e.g., *"Bhai kal exam hai aur abhi tak syllabus bhi nahi khola"*).
2. **Visual-Textual Incongruity:** The humor or sarcasm often relies on a contradiction between the visual image (e.g., a dancing actor) and the textual punchline (e.g., failing an exam). Unimodal text or vision models fail on these examples.
3. **Multimodal Gated Fusion:** IndicSent-M introduces an adaptive sigmoid gating mechanism that balances the 768-d text representation from **MuRIL** and the 2048-d visual representation from **ResNet-50**, selectively filtering noise and outperforming standard concatenation fusion by **+2.10% Macro-F1**.

The system is deployed as a production-ready application with a **FastAPI backend** (serving a unified PyTorch inference engine) and an interactive **React + Vite frontend**.

---

## 🏗️ Repository Architecture

```
NLP project (sentiment analysis)/
├── backend/                        # FastAPI Backend Application
│   ├── app/
│   │   ├── api/routes/             # Health, Models, and Prediction REST endpoints
│   │   ├── core/                   # Configuration, CORS, and unified error handling
│   │   ├── schemas/                # Pydantic v2 validation models
│   │   ├── services/               # Thread-safe PredictorService singleton
│   │   └── main.py                 # ASGI entry point with lifespan management
│   ├── tests/                      # 15 automated pytest cases + E2E test harness
│   └── requirements.txt            # Locked backend dependencies
├── model_training/                 # ML & Deep Learning Pipeline
│   ├── checkpoints/                # Trained model weights (MuRIL, XLM-R, ResNet-50, Gated Fusion)
│   ├── data_splits/                # Stratified 80/20 train/validation splits (Memotion 3.0)
│   ├── results/                    # Confusion matrices, classification reports, CSV comparison
│   └── src/
│       ├── models/                 # PyTorch architectures (Gated & Concat Fusion)
│       └── inference/predictor.py  # Unified multi-model inference engine
├── src/                            # React 18 Frontend Application
│   ├── components/                 # ModelSelector, ConfidenceBar, SentimentBadge
│   ├── pages/                      # Dashboard, MemeAnalyzer, ModelComparison, Architecture, About
│   ├── services/api.js             # Axios client connecting to FastAPI backend
│   └── data/mockData.js            # Offline demonstration fallback data
├── dataset/Memotion 3/             # Memotion 3.0 dataset & image repository
├── API_DOCUMENTATION.md            # Comprehensive REST API specifications
├── DEPLOYMENT_GUIDE.md             # Production setup & server management guide
├── MODEL_EVALUATION_REPORT.md      # Exhaustive benchmark report on all 14 models
├── SYSTEM_ARCHITECTURE.md          # In-depth architectural & mathematical specification
├── FINAL_SYSTEM_AUDIT_REPORT.md    # Master system audit & deployment readiness certification
└── package.json                    # Node.js dependencies & scripts
```

---

## 🏆 Model Benchmark & Evaluation Summary

All 14 models were trained and evaluated on the official **Memotion 3.0** stratified validation split (1,400 samples):

| Model Architecture | Modality | Accuracy (%) | Macro-F1 (%) | Weighted F1 (%) | Hardware Latency |
|:---|:---|:---:|:---:|:---:|:---:|
| **TF-IDF + Logistic Regression** | Text (Hinglish OCR) | **43.79%** | **43.14%** | **43.98%** | 0.48 ms (CPU) |
| **XLM-RoBERTa (base)** | Text (Transformer) | 42.36% | 41.66% | 42.43% | 9.70 ms (CUDA) |
| **TF-IDF + Random Forest** | Text (Hinglish OCR) | 44.00% | 40.75% | 42.52% | 0.85 ms (CPU) |
| **MuRIL (base-cased)** | Text (Transformer) | 40.93% | 40.75% | 41.04% | 11.20 ms (CUDA) |
| **MuRIL + ResNet-50 (Gated Fusion)** | **Multimodal (Text + Image)** | **37.64%** | **37.24%** | **37.59%** | **25.51 ms (CUDA)** |
| **ResNet-50** | Vision Only | 40.43% | 37.04% | 38.88% | 14.16 ms (CUDA) |
| **TF-IDF + Naive Bayes** | Text (Hinglish OCR) | 42.86% | 37.06% | 39.73% | 0.35 ms (CPU) |
| **EfficientNet-B0** | Vision Only | 37.50% | 36.54% | 37.64% | 12.80 ms (CUDA) |
| **MobileNetV3-Large** | Vision Only | 38.21% | 36.25% | 37.74% | 10.50 ms (CUDA) |
| **MuRIL + EfficientNet-B0 (Concat)** | Multimodal (Text + Image) | 36.00% | 35.43% | 36.25% | 22.40 ms (CUDA) |
| **MuRIL + ResNet-50 (Concat Fusion)** | Multimodal (Text + Image) | 37.57% | 35.14% | 35.75% | 24.80 ms (CUDA) |
| **XLM-RoBERTa + ResNet-50 (Concat)** | Multimodal (Text + Image) | 31.43% | 30.00% | 28.51% | 26.10 ms (CUDA) |
| **TF-IDF + Linear SVM** | Text (Hinglish OCR) | 43.57% | 27.97% | 32.58% | 0.50 ms (CPU) |
| **Majority Baseline** | Baseline | 42.43% | 19.86% | 25.28% | 0.05 ms (CPU) |

> **Key Discovery:** Gated Multimodal Fusion outperforms Concat Fusion by **+2.10% Macro-F1**, achieving balanced recall across all three classes (Negative: 49.57%, Neutral: 37.21%, Positive: 29.01%) by filtering conflicting visual cues when textual punchlines carry the primary sentiment.

---

## ⚡ Quick Start: Running the Project Locally

### 🚀 One-Click Launcher (Starts & Stops Everything)
You can start both the Backend and Frontend with a single command or double-click:
```powershell
# Double-click start.bat in Windows Explorer, or run:
python run.py
```
- Starts the FastAPI Backend on `http://127.0.0.1:8000`
- Starts the React Frontend on `http://localhost:3000` (and automatically opens your browser)
- **To Stop Everything:** Press `Ctrl + C` in that terminal (or close the window) — both servers are cleanly terminated with zero lingering processes.

---

### Manual Multi-Terminal Startup
If you prefer running services in separate terminals:

#### 1. Backend Startup (Terminal 1)
```powershell
cd backend
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```
- API Health Status: `http://127.0.0.1:8000/api/health`
- Swagger Documentation: `http://127.0.0.1:8000/docs`

#### 2. Frontend Startup (Terminal 2)
```powershell
npm.cmd run dev
```
- Open browser at: `http://localhost:3000` (or `http://localhost:5173`)

### 4. Running the Automated Test Suite
```powershell
# Run backend test suite (15 unit and integration tests)
python -m pytest backend/tests --ignore=backend/tests/test_e2e_integration.py -v

# Run full end-to-end integration test harness
python backend/tests/test_e2e_integration.py
```

---

## 📖 Detailed Project Documentation

- [System Architecture Specification](SYSTEM_ARCHITECTURE.md) — Neural network math, tensor dimensions, gating formulas, and backend lifecycle.
- [Model Evaluation Report](MODEL_EVALUATION_REPORT.md) — In-depth analysis of all 14 models, confusion matrices, and Hinglish NLP challenges.
- [REST API Documentation](API_DOCUMENTATION.md) — Endpoint schemas, request/response bodies, cURL examples, and error codes.
- [Deployment Guide](DEPLOYMENT_GUIDE.md) — Production setup with Uvicorn, Nginx reverse proxy, systemd, and troubleshooting.
- [Final System Audit Report](FINAL_SYSTEM_AUDIT_REPORT.md) — Empirical audit of dataset integrity, model metrics, concurrency stress tests, and hardware latency.

---

## 📄 License
This project is developed for academic research and educational purposes using the Memotion 3.0 dataset.
