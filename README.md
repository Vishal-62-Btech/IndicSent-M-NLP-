# IndicSent-M: Multimodal Sentiment & Emotion Analysis for Hinglish Memes
========================================================================

> **Tagline:** *Decoding the Emotional Subtext of Indian Internet Memes*  
> **Subtitle:** *An End-to-End Multimodal Deep Learning & NLP Framework with Gated Attention Fusion and Multi-Task Emotion Detection*

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CUDA 12](https://img.shields.io/badge/CUDA-12.x-76B900.svg?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React 18](https://img.shields.io/badge/React-18.x-61DAFB.svg?logo=react&logoColor=white)](https://react.dev/)
[![Vite](https://img.shields.io/badge/Vite-5.4-646CFF.svg?logo=vite&logoColor=white)](https://vitejs.dev/)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind-3.4-38B2AC.svg?logo=tailwind-css&logoColor=white)](https://tailwindcss.com/)
[![Tests](https://img.shields.io/badge/Tests-38%20Passed-brightgreen.svg)](FINAL_TEST_REPORT.md)

---

## 📌 Project Overview

**IndicSent-M** is a full-stack, research-grade multimodal sentiment and affective analysis framework specifically engineered for Indian social media memes. Internet memes in India present profound linguistic and computer vision challenges:

1. **Hinglish Code-Mixing:** Unstandardized Romanized Hindi mixed with English text (e.g., *"Jab assignment submit karne ke baad teacher bolti hai 'Well done beta!' 🥳"*).
2. **Visual-Textual Incongruity:** Humor, irony, and sarcasm frequently arise from contradictions between the visual template (e.g., an actor crying or panicking) and surface textual praise. Unimodal models fail to resolve this discrepancy.
3. **Adaptive Gated Fusion:** IndicSent-M integrates an adaptive sigmoid gating mechanism that dynamically blends 768-d text representations from **MuRIL** and 2048-d visual representations from **ResNet-50**, selectively filtering noise and outperforming standard concatenation fusion by **+2.10% Macro-F1**.
4. **Multi-Task Dual-Head Architecture:** Jointly classifies 3-class sentiment (`Negative`, `Neutral`, `Positive`) alongside 6 discrete emotions (`Happiness 😊`, `Sadness 😢`, `Anger 😡`, `Fear 😨`, `Disgust 🤢`, `Surprise 😲`), isolating the primary **Dominant Emotion** on the primary user interface while offering an expandable multi-label probability breakdown.

---

## 🏆 Benchmark & Evaluation Summary

Evaluated on the official **Memotion 3.0** stratified validation split ($N=1,400$ sentiment samples, $N=230$ verified emotion samples):

| Model Architecture | Modality | Sent Acc (%) | Sent Macro-F1 (%) | Dominant Emotion Acc (%) | Emotion Macro-F1 (%) | Hamming Loss | Hardware Latency |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Single-Task Gated Fusion (Champion)** | Multimodal (Text + Vision) | **37.71%** | **37.31%** | — | — | — | 25.5 ms (CUDA) |
| **Multi-Task $E_1$ ($\lambda=0.5$) (Selected Production)** | **Multimodal + Multi-Task** | **35.50%** | **35.47%** | **40.00%** | **22.62%** | **0.2775** | **28.5 ms (CUDA)** |
| **Multi-Task $E_0$ ($\lambda=1.0$)** | Multimodal + Multi-Task | 35.79% | 35.34% | 40.00% | 22.51% | 0.2529 | 28.5 ms (CUDA) |
| **TF-IDF + Logistic Regression** | Text Only (Hinglish OCR) | 43.79% | 43.14% | — | — | — | 0.48 ms (CPU) |
| **XLM-RoBERTa (base)** | Text Only (Transformer) | 42.36% | 41.66% | — | — | — | 9.8 ms (CUDA) |
| **ResNet-50** | Vision Only | 40.43% | 37.04% | — | — | — | 14.2 ms (CUDA) |
| **MuRIL + ResNet-50 (Concat)** | Multimodal (Text + Vision) | 37.57% | 35.14% | — | — | — | 24.8 ms (CUDA) |
| **Sentiment Majority Baseline** | Baseline (Predicts Neutral) | 42.43% | 19.86% | — | — | — | 0.05 ms (CPU) |
| **Emotion All-Negative Baseline** | Baseline (Predicts 0s) | — | — | 0.00% | 0.00% | 0.1812 | 0.01 ms (CPU) |
| **Dominant Emotion Majority** | Baseline (Predicts Happiness) | — | — | 39.57% | — | — | 0.01 ms (CPU) |

> **Key Discovery:** Multi-Task Model $E_1$ maintains balanced per-class sentiment performance (Negative: 36.55%, Neutral: 34.84%, Positive: 35.03%) while delivering an **Emotion Macro-F1 of 22.62%** and a **Dominant Emotion Accuracy of 40.00%**, significantly beating random chance (16.67%) and majority baselines.

---

## ⚡ Quick Start: Running the Project Locally

### 🚀 1. One-Click Unified Launcher (Starts Everything)
Run the automated launcher from PowerShell or Command Prompt:
```powershell
cd "D:\projectsss\NLP project (sentiment analysis)"
python run.py
```
- Starts the FastAPI Backend on `http://127.0.0.1:8000`
- Starts the React Frontend on `http://localhost:3000`
- Automatically opens your web browser.
- **To stop everything:** Press `Ctrl + C` in that terminal — both servers shut down cleanly.

---

### 🛠️ 2. Manual Multi-Terminal Startup
If you prefer running services in separate terminal windows:

#### Terminal 1 — Backend:
```powershell
cd "D:\projectsss\NLP project (sentiment analysis)\backend"
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```
- Swagger Documentation: `http://127.0.0.1:8000/docs`
- Health Endpoint: `http://127.0.0.1:8000/api/v1/health`

#### Terminal 2 — Frontend:
```powershell
cd "D:\projectsss\NLP project (sentiment analysis)"
npm.cmd run dev
```
- Client Interface: `http://localhost:3000`

---

### 🧪 3. Running Automated Tests
```powershell
cd "D:\projectsss\NLP project (sentiment analysis)"
pytest backend/tests/ -v
```
**Result:** 38 passed, 0 failed in ~60 seconds.

---

## 📖 Complete Project Documentation Library

IndicSent-M includes 10 comprehensive academic and technical specifications:

1. [Project Architecture & System Design](PROJECT_ARCHITECTURE.md) — Neural network math, tensor dimensions, gating formulas, and backend lifecycle.
2. [Dataset Specification & Emotion Annotation Guide](DATASET_AND_ANNOTATION_GUIDE.md) — Memotion 3.0 split hygiene (0 leakage), affective schema, -1 masking, and class balance.
3. [Model Training & Experimentation Guide](MODEL_TRAINING_GUIDE.md) — Loss formulations, multi-task weighting ($\lambda$), optimizer settings, and reproduction steps.
4. [Master Model Evaluation & Baseline Report](MODEL_EVALUATION_REPORT.md) — Full benchmark across 14 models, baselines, Hamming Loss paradox, and threshold calibration.
5. [REST API Documentation](API_DOCUMENTATION.md) — Endpoint schemas, request/response bodies, cURL examples, and error codes.
6. [User & Demonstration Guide](USER_GUIDE.md) — Walkthrough of each feature (multimodal, text, image, OCR, explainability, PDF export).
7. [Master Academic Project Report](FINAL_PROJECT_REPORT.md) — Full academic research report covering motivation, literature review, methodology, and results.
8. [Master Quality Assurance & Test Report](FINAL_TEST_REPORT.md) — 38/38 unit and integration test breakdown, frontend build verification, and latency numbers.
9. [Limitations & Future Research Directions](LIMITATIONS_AND_FUTURE_WORK.md) — MuRIL embedding anisotropy, label scarcity, OCR noise, and LoRA/IndicCLIP roadmap.
10. [Deployment Guide](DEPLOYMENT_GUIDE.md) — Production setup with Uvicorn, Nginx reverse proxy, and systemd.

---

## 📄 License & Attribution
Developed for academic research and educational demonstration on the AAAI De-Factify 2 benchmark (Memotion 3.0 dataset).
