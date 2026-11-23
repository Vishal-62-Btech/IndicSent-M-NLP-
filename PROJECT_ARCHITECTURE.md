# IndicSent-M: Project Architecture & System Design
=====================================================

> **Academic Reference Specification for Multimodal Sentiment & Six-Emotion Analysis**  
> *Target Domain: Indian Internet Memes (Hinglish Text + Visual Incongruity)*  
> *Frameworks: PyTorch 2.x, Hugging Face Transformers, FastAPI, React 18, Vite*

---

## 1. High-Level System Architecture

IndicSent-M is engineered as a decoupled, production-grade multimodal system comprising three foundational tiers:
1. **Interactive Client Tier (React 18 + Tailwind CSS):** A responsive single-page web application featuring real-time image drag-and-drop, automated OCR text extraction, multi-modal model selection, dominant emotion badge presentation with an expandable 6-emotion multi-label accordion, visual Grad-CAM explainability, token attribution highlights, and PDF report generation.
2. **REST API & Orchestration Tier (FastAPI + Pydantic v2):** An asynchronous ASGI application implementing singleton model lifecycle management, safe file validation, background resource cleanup, and structured schema serialization.
3. **Deep Learning & Inference Tier (PyTorch + Torchvision + Transformers):** A unified model registry containing classical ML baselines (TF-IDF + Logistic Regression, SVM, Naive Bayes), unimodal transformers (MuRIL, XLM-RoBERTa), unimodal computer vision classifiers (ResNet-50, EfficientNet-B0, MobileNet-V3), and multimodal gated fusion networks.

```
+---------------------------------------------------------------------------------------+
|                                  CLIENT LAYER (React 18)                              |
|   +-----------------------+   +------------------------+   +----------------------+   |
|   | Text-Only Sentiment   |   | Vision-Only Sentiment  |   | Multimodal + Emotion |   |
|   | Hinglish Token Attrib |   | Grad-CAM Visual Heatmap|   | Dominant + 6-Emotion |   |
|   +-----------------------+   +------------------------+   +----------------------+   |
+-------------------------------------------|-------------------------------------------+
                                            | HTTP REST API (JSON / Multipart-Form)
                                            v
+---------------------------------------------------------------------------------------+
|                                FASTAPI APPLICATION BACKEND                            |
|   +-----------------------+   +------------------------+   +----------------------+   |
|   | /predict/text         |   | /predict/image         |   | /predict/multimodal  |   |
|   | /explain/tokens       |   | /explain/gradcam       |   | /report/pdf          |   |
|   +-----------------------+   +------------------------+   +----------------------+   |
|                                           |                                           |
|   +-------------------------------------------------------------------------------+   |
|   |           PredictorService (Singleton Lifecycle & Resource Manager)           |   |
|   +-------------------------------------------------------------------------------+   |
+-------------------------------------------|-------------------------------------------+
                                            |
                                            v
+---------------------------------------------------------------------------------------+
|                               INFERENCE & NEURAL PIPELINE                             |
|                                                                                       |
|   [Text Input (Hinglish)]                     [Image Input (RGB Meme)]                |
|              |                                            |                           |
|              v                                            v                           |
|      MuRIL Tokenizer                              Torchvision Transforms              |
|   (google/muril-base-cased)                        (ResNet-50 Normalize)              |
|              |                                            |                           |
|              v                                            v                           |
|      MuRIL Transformer                                 ResNet-50                      |
|      (768-d Text Feature)                        (2048-d Visual Feature)              |
|              |                                            |                           |
|              v                                            v                           |
|      Linear Proj (512-d)                         Linear Proj (512-d)                  |
|              |                                            |                           |
|              +--------------------+  +--------------------+                           |
|                                   |  |                                                |
|                                   v  v                                                |
|                          Sigmoid Gating Network                                       |
|                       alpha = sigmoid(W * [h_t, h_v])                                 |
|                                   |                                                   |
|                                   v                                                   |
|                      Gated Representation (h_fused)                                   |
|                     alpha * h_t + (1 - alpha) * h_v                                   |
|                                   |                                                   |
|                 +-----------------+-----------------+                                 |
|                 |                                   |                                 |
|                 v                                   v                                 |
|       Sentiment Head (3-class)             Emotion Head (6-class)                     |
|         Linear -> ReLU -> Linear             Linear -> ReLU -> Linear                 |
|            Softmax Probabilities                Sigmoid Probabilities                 |
|         [Negative, Neutral, Positive]        [Happy, Sad, Anger, Fear, Disgust, Surp] |
|                                                     |                                 |
|                                                     v                                 |
|                                            Dominant Emotion: argmax                   |
+---------------------------------------------------------------------------------------+
```

---

## 2. Multimodal Neural Network Design

### 2.1 Encoders
1. **Text Backbone (MuRIL — Multilingual Representations for Indian Languages):**
   - MuRIL is pre-trained by Google on monolingual and parallel corpora across 17 Indic languages including Hindi, transliterated Hinglish, and English.
   - For a meme text $T = (t_1, t_2, \dots, t_L)$, the text representation $h_{text} \in \mathbb{R}^{768}$ is extracted via pooler output or mean pooling across active token positions:
     $$h_{text} = \text{MuRIL}(T)$$
2. **Vision Backbone (ResNet-50):**
   - ResNet-50 captures high-level visual features (facial expressions, scene geometry, iconic meme templates) via deep residual blocks.
   - For an input image $I \in \mathbb{R}^{3 \times 224 \times 224}$, the classification head is truncated, yielding the penultimate average-pooled feature:
     $$h_{vis} = \text{ResNet50}(I) \in \mathbb{R}^{2048}$$

### 2.2 Common Subspace Projection
Text and vision features originate from disparate semantic manifolds and dimensions. They are projected into a common dimension $d = 512$:
$$\tilde{h}_t = \text{ReLU}(\text{LayerNorm}(W_{pt} h_{text} + b_{pt})) \in \mathbb{R}^{512}$$
$$\tilde{h}_v = \text{ReLU}(\text{LayerNorm}(W_{pv} h_{vis} + b_{pv})) \in \mathbb{R}^{512}$$

### 2.3 Gated Multimodal Fusion
Rather than naive vector concatenation ($\tilde{h}_t \oplus \tilde{h}_v$), which treats modalities with equal weight regardless of sample noise or conflict, IndicSent-M uses a learnable gating mechanism:
$$\alpha = \sigma(W_{gate} [\tilde{h}_t \,\|\, \tilde{h}_v] + b_{gate}) \in (0, 1)^{512}$$
$$h_{fused} = \alpha \odot \tilde{h}_t + (1 - \alpha) \odot \tilde{h}_v$$
$$h_{repr} = \text{LayerNorm}(h_{fused})$$
- $\alpha \to 1$: Model dynamically relies predominantly on textual nuance (critical when sarcastic captions invert pleasant images).
- $\alpha \to 0$: Model focuses primarily on visual facial expressions or meme templates.

### 2.4 Multi-Task Dual Head Architecture
From the shared fused representation $h_{repr} \in \mathbb{R}^{512}$, the network branches into two task-specific multi-layer perceptron (MLP) heads:

1. **3-Class Sentiment Head:**
   $$z_{sent} = W_{s2} \cdot \text{ReLU}(W_{s1} h_{repr} + b_{s1}) + b_{s2} \in \mathbb{R}^3$$
   $$P(y_{sent} = c \mid I, T) = \frac{\exp(z_{sent}^{(c)})}{\sum_{j=1}^3 \exp(z_{sent}^{(j)})}$$
   Objective: Multiclass Cross-Entropy Loss with inverse class weighting:
   $$\mathcal{L}_{sent} = - \sum_{c=1}^3 w_c \, y_c \log P(y_{sent} = c)$$

2. **6-Class Multi-Label Emotion Head:**
   $$z_{em} = W_{e2} \cdot \text{ReLU}(W_{e1} h_{repr} + b_{e1}) + b_{e2} \in \mathbb{R}^6$$
   $$\hat{p}_k = \sigma(z_{em}^{(k)}) = \frac{1}{1 + \exp(-z_{em}^{(k)})}, \quad k \in \{1, \dots, 6\}$$
   Emotions: `{Happiness, Sadness, Anger, Fear, Disgust, Surprise}`.
   Objective: Masked Binary Cross-Entropy with positive class weighting ($pos\_weight_k$):
   $$\mathcal{L}_{em} = - \sum_{k=1}^6 m \cdot \left[ pos\_weight_k \cdot y_k \log \hat{p}_k + (1 - y_k) \log (1 - \hat{p}_k) \right]$$
   Where $m \in \{0, 1\}$ is the verification mask ($m=1$ for verified annotated samples; $m=0$ for unannotated samples to prevent penalizing the shared backbone).

3. **Multi-Task Joint Loss:**
   $$\mathcal{L}_{total} = \mathcal{L}_{sent} + \lambda_{emotion} \cdot \mathcal{L}_{em}$$
   In the champion production model ($E_1$), $\lambda_{emotion} = 0.5$, which balances gradient updates between sentiment classification and multi-label emotion supervision without causing negative transfer.

4. **Dominant Emotion Formulation:**
   $$\text{Dominant Emotion} = \arg\max_{k \in \{1, \dots, 6\}} \hat{p}_k$$

---

## 3. Backend Lifecycle & Singleton Predictor Service

The FastAPI backend implements the `lifespan` context protocol:
- **Application Startup:**
  - `PredictorService.get_instance().initialize()` preloads the default models (Gated Fusion, XLM-RoBERTa, TF-IDF + Logistic Regression, and ResNet-50) into memory/GPU.
  - OCR engines (`EasyOCR`) and hashing indexes are initialized.
  - Zero disk downloads: all tokenizers, weights, and vocabularies are resolved from local checkpoint directories.
- **Application Teardown:**
  - `cleanup()` releases PyTorch tensors, flushes CUDA caches via `torch.cuda.empty_cache()`, and purges any temporary files.
- **Per-Request Resource Safety:**
  - Uploaded multipart files are streamed into temporary disk files, validated for MIME type (`image/jpeg`, `image/png`, `image/webp`), scanned for dimensions/corruption using PIL, processed, and deleted in a `finally` block.

---

## 4. Frontend Component Hierarchy

The React 18 client is organized modularly under `src/`:
```
src/
├── main.jsx                        # React root entry point
├── App.jsx                         # React Router v6 setup & layout shell
├── index.css                       # Tailwind CSS directives & custom themes
├── components/
│   ├── Navbar.jsx                  # Header navigation with status indicator
│   ├── Sidebar.jsx                 # Collapsible navigational drawer
│   ├── Footer.jsx                  # Academic attribution & copyright footer
│   └── PrototypeNotice.jsx         # Status banner
├── pages/
│   ├── Dashboard.jsx               # System health, model catalog, and live analytics
│   ├── MemeAnalyzer.jsx            # Core multimodal sentiment & emotion analysis suite
│   ├── ModelComparison.jsx         # Interactive benchmark comparisons
│   ├── ModelArchitecture.jsx       # Interactive architectural & mathematical visualizer
│   └── AboutProject.jsx            # Academic context, dataset background, literature review
└── services/
    └── api.js                      # Axios instance with centralized baseURL and interceptors
```

### 4.1 MemeAnalyzer Component Flow
1. **Mode Switcher:** Switches between Multimodal (Image + Text), Text Only, and Image Only.
2. **Model Dropdown:** Populated dynamically from `/api/v1/models`. Defaults to `multitask_emotion`.
3. **OCR Integration:** Extracts text automatically upon image drop; user can edit the text prior to analysis.
4. **Analysis Execution:** Calls `/api/v1/predict/multimodal`.
5. **Results Presentation:**
   - Sentiment badge (Negative / Neutral / Positive) with confidence meter.
   - **Dominant Emotion Card:** Displays dominant emoji, emotion name, and confidence percentage.
   - **Expandable 6-Emotion Accordion:** User clicks "Detailed Emotion Analysis" to reveal all 6 independent sigmoid probabilities with progress bars.
   - **Explainability Drawer:** Loads Grad-CAM heatmap overlay and Hinglish token attributions on demand.
   - **PDF Export:** Generates an official report containing sentiment, dominant emotion, confidence breakdown, and explainability heatmaps.
