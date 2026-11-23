# IndicSent-M: Comprehensive User & Demonstration Guide
=========================================================

> **Step-by-Step Walkthrough for Running, Demonstrating, and Evaluating IndicSent-M**  
> *Target Audience: Academic Evaluators, Researchers, and Developers*

---

## 1. Quick Start & Launch Instructions

### 1.1 One-Click Launch (Recommended)
IndicSent-M includes a unified master runner that launches both the FastAPI backend and React frontend concurrently, with automated browser launching:
```powershell
cd "D:\projectsss\NLP project (sentiment analysis)"
python run.py
```
- **Backend:** Spawns on `http://127.0.0.1:8000`
- **Frontend:** Spawns on `http://localhost:3000` (or next free port) and automatically opens your default browser.
- **Shutdown:** Press `Ctrl + C` in the terminal to cleanly terminate both processes with zero orphaned background tasks.

### 1.2 Manual Two-Terminal Launch
If you prefer running services independently:

**Terminal 1 — FastAPI Backend:**
```powershell
cd "D:\projectsss\NLP project (sentiment analysis)\backend"
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```
Verify backend is online by visiting: `http://127.0.0.1:8000/docs`

**Terminal 2 — React Frontend:**
```powershell
cd "D:\projectsss\NLP project (sentiment analysis)"
npm run dev
```
Open `http://localhost:3000` in your web browser.

---

## 2. Navigating the Application

The web interface features an intuitive top navigation bar with five specialized modules:
1. **Meme Analyzer (`/`):** The primary multimodal inference, OCR, emotion detection, and explainability workspace.
2. **Dashboard (`/dashboard`):** Real-time system telemetry, active device (CUDA/CPU), loaded checkpoint paths, and latency monitor.
3. **Model Comparison (`/comparison`):** Interactive leaderboard comparing all 14 benchmarked single-task and multi-task models.
4. **Architecture (`/architecture`):** Interactive visualization of the MuRIL + ResNet-50 Gated Fusion and multi-task dual heads.
5. **About (`/about`):** Academic research context, AAAI De-Factify 2 background, and author attributions.

---

## 3. Step-by-Step Feature Walkthrough

### 3.1 Multimodal Sentiment & Dominant Emotion Analysis (Core Workflow)
1. Navigate to the **Meme Analyzer** tab.
2. Select **Multimodal (Text + Image)** mode in the mode selector.
3. In the Model dropdown, select **`MuRIL + ResNet-50 (Multi-Task Sentiment & Emotion Fusion)`** (default).
4. **Upload a Meme Image:**
   - Drag and drop an image into the drop zone, or click to browse.
   - Example image: `assignment_meme.jpg` or any image from `dataset/Memotion 3/trainImages/`.
5. **Automated OCR Extraction:**
   - The embedded `EasyOCR` engine automatically extracts the meme caption.
   - The extracted text appears in the **Caption Text** input box.
   - You can freely edit, correct, or expand the caption text.
6. **Execute Prediction:**
   - Click the green **"Analyze Meme"** button.
7. **Inspect Output:**
   - **Sentiment Prediction Badge:** Displays `Negative`, `Neutral`, or `Positive` with color-coded confidence percentage bars.
   - **Dominant Emotion Card:**
     - Displays the single primary emotion (e.g., `😊 Happiness 84.12%` or `😢 Sadness 65.40%`).
     - Includes a brief explanation badge: *"Dominant emotion detected via multi-task model."*
   - **Expandable Detailed Emotion Accordion:**
     - Click **"Detailed Emotion Analysis (6 Emotions)"** button.
     - The accordion expands to reveal all six independent sigmoid probabilities (`Happiness`, `Sadness`, `Anger`, `Fear`, `Disgust`, `Surprise`) with individual progress meters.

### 3.2 Explainable AI (Grad-CAM & Token Attribution)
1. Below the prediction results, click the **"Explain Prediction"** button.
2. **Visual Explainability (Grad-CAM):**
   - Renders a high-resolution heatmap overlaid on the meme image, highlighting which visual regions (e.g., facial expressions, gestures) the ResNet-50 vision branch attended to.
3. **Textual Explainability (Token Attribution):**
   - Highlights individual Hinglish tokens in green (positive contribution) or red (negative contribution), showing which words triggered the sentiment decision.

### 3.3 Generating & Exporting PDF Reports
1. In the results panel, click **"Export PDF Report"**.
2. The FastAPI backend compiles prediction results, dominant emotion, confidence distribution, and base64 Grad-CAM heatmaps into an academic PDF document using ReportLab.
3. The browser automatically downloads `IndicSent-M_Report.pdf`.

### 3.4 Text-Only Prediction Mode
1. Switch mode selector to **Text Only**.
2. Enter any Hinglish sentence (e.g., *"Aaj ka paper bohot easy tha maze aa gaye"*).
3. Select model (e.g., `XLM-RoBERTa (Fine-Tuned)` or `TF-IDF + Logistic Regression`).
4. Click **"Analyze Sentiment"**.

### 3.5 Image-Only Prediction Mode
1. Switch mode selector to **Image Only**.
2. Upload an image template without entering any text.
3. Select model (`ResNet-50`).
4. Click **"Analyze Image"**.

---

## 4. Troubleshooting & FAQ

- **Q: What if port 8000 is occupied?**  
  A: Set a different port in `run.py` or execute `uvicorn app.main:app --port 8080` and update `VITE_API_URL` in `.env`.
- **Q: Can the application run on CPU without an NVIDIA GPU?**  
  A: Yes. The backend automatically detects CUDA availability. If CUDA is not detected, it gracefully falls back to CPU execution.
- **Q: Why does dominant emotion use argmax rather than fixed 0.5?**  
  A: Argmax guarantees that every meme receives a single primary emotion, preventing empty-set predictions or multi-label over-prediction conflicts in the primary user card. Detailed multi-label probabilities remain accessible via the expandable accordion.
