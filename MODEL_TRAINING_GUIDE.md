# IndicSent-M: Model Training & Experimentation Guide
=====================================================

> **Complete Guide to Training, Fine-Tuning, and Multi-Task Optimization**
> *Target Architecture: MuRIL + ResNet-50 Gated Fusion Network*
> *Tasks: 3-Class Sentiment Classification & 6-Class Multi-Label Emotion Detection*

---

## 1. Preprocessing Pipelines

### 1.1 Text Preprocessing (`clean_ocr_text`)
Located in `model_training/src/preprocessing/clean_text.py`:
- **OCR Artifact Removal:** Strips stray punctuation, noise characters, watermark text, and garbled symbols.
- **Hinglish Normalization:** Normalizes character elongation (e.g., *"bahhhhuuuut"* $\to$ *"bahut"*), standardizes whitespace, and preserves emotion-carrying emoji symbols.
- **Tokenization:**
  ```python
  from transformers import AutoTokenizer
  tokenizer = AutoTokenizer.from_pretrained("google/muril-base-cased", local_files_only=True)
  encoded = tokenizer(
      clean_text,
      truncation=True,
      padding="max_length",
      max_length=128,
      return_tensors="pt"
  )
  ```

### 1.2 Image Preprocessing (`get_image_transforms`)
Located in `model_training/src/preprocessing/preprocess_images.py`:
- **Training Transformations:**
  - Resize image to $256 \times 256$ pixels.
  - Random crop to $224 \times 224$ pixels.
  - Random horizontal flip ($p=0.5$).
  - Conversion to PyTorch Tensor.
  - ImageNet normalization: $\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$.
- **Validation Transformations:**
  - Resize image to $256 \times 256$ pixels.
  - Center crop to $224 \times 224$ pixels.
  - ImageNet normalization: $\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$.

---

## 2. Multi-Task Training Objectives & Loss Formulation

The multi-task model jointly minimizes:
$$\mathcal{L}_{total} = \mathcal{L}_{sent} + \lambda_{emotion} \cdot \mathcal{L}_{em}$$

### 2.1 Sentiment Loss ($\mathcal{L}_{sent}$)
Multiclass Cross-Entropy Loss with inverse class weighting:
$$w_c = \frac{N_{total}}{3 \cdot N_c}$$
Where $N_{total} = 5,600$, $N_{neg} = 1,405$, $N_{neu} = 2,376$, $N_{pos} = 1,819$:
$$w_{neg} = 1.330, \quad w_{neu} = 0.786, \quad w_{pos} = 1.026$$

### 2.2 Emotion Loss ($\mathcal{L}_{em}$)
Masked Binary Cross-Entropy with positive class weighting ($pos\_weight_k$):
$$\mathcal{L}_{em} = \frac{1}{\sum_{i} m_i} \sum_{i=1}^B m_i \sum_{k=1}^6 \text{BCEWithLogitsLoss}(z_{em}^{(k)}(i), y_k(i); pos\_weight_k)$$
Positive weights derived from empirical training set imbalance:
- Happiness: $0.55$
- Sadness: $3.52$
- Anger: $10.76$
- Fear: $17.40$
- Disgust: $46.39$
- Surprise: $12.84$

---

## 3. Controlled Multi-Task Optimization Experiments

To systematically investigate the gradient interaction between sentiment and emotion tasks, three controlled experiments were conducted holding random seed ($42$), learning rate ($10^{-4}$), batch size ($16$), and training annotations ($1,564$ verified samples) constant:

| Parameter | Baseline Champion ($E_{single}$) | Experiment $E_0$ | Experiment $E_1$ (Champion) | Experiment $E_2$ |
|:---|:---:|:---:|:---:|:---:|
| **Loss Weight ($\lambda_{emotion}$)** | $0.0$ (Single-Task) | $1.0$ | **$0.5$** | $0.25$ |
| **Epochs** | 4 | 3 | **3** | 3 |
| **Batch Size** | 16 | 16 | **16** | 16 |
| **Learning Rate** | $1 \times 10^{-4}$ | $1 \times 10^{-4}$ | **$1 \times 10^{-4}$** | $1 \times 10^{-4}$ |
| **Optimizer** | AdamW | AdamW | **AdamW** | AdamW |
| **Verified Train Samples** | — | 1,564 | **1,564** | 1,564 |
| **Validation Sent Acc** | 37.71% | 35.79% | **35.50%** | 33.36% |
| **Validation Sent Macro-F1** | 37.31% | 35.34% | **35.47%** | 33.09% |
| **Validation Emotion Macro-F1** | — | 22.51% | **22.62%** | 21.56% |
| **Validation Emotion Micro-F1** | — | 45.21% | **44.09%** | 35.28% |
| **Validation Dominant Emotion Acc** | — | 40.00% | **40.00%** | 26.09% |

### 3.1 Analysis of Gradient Dynamics
- **$\lambda = 1.0$ ($E_0$):** Equal weighting causes slight competition between 3-class sentiment gradients and multi-label emotion gradients, yielding 35.34% sentiment Macro-F1.
- **$\lambda = 0.5$ ($E_1$):** Provides optimal gradient balance. Sentiment Macro-F1 recovers to 35.47% with remarkably balanced per-class F1 (Negative: 36.55%, Neutral: 34.84%, Positive: 35.03%), while achieving the highest emotion Macro-F1 (22.62%) and 40.00% dominant emotion accuracy.
- **$\lambda = 0.25$ ($E_2$):** Emotion loss signal is under-weighted. Dominant emotion accuracy degrades drastically to 26.09% and sentiment Macro-F1 falls to 33.09%.

---

## 4. How to Reproduce Training

### 4.1 Prerequisites
Ensure PyTorch with CUDA support is installed:
```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### 4.2 Run Multi-Task Training Script
```powershell
cd "D:\projectsss\NLP project (sentiment analysis)"

# Train Experiment E1 (Recommended Production Configuration)
python model_training/src/training/train_multitask_emotion.py `
  --exp-name e1_lambda0_5 `
  --lambda-emotion 0.5 `
  --epochs 3 `
  --batch-size 16 `
  --seed 42
```

Checkpoints will be saved automatically to:
`model_training/checkpoints/experiments/e1_lambda0_5/best_model.pt`

### 4.3 Evaluate Checkpoint on Validation Split
```powershell
python model_training/src/evaluation/evaluate_all_final.py --batch-size 16
```
Outputs complete JSON evaluation metrics to `model_training/evaluation_results/final_comprehensive_evaluation.json`.

---

## 5. Classical Text & Emotion Model Training

### 5.1 Preprocessing and Idiom Compounds
In `clean_text.py`, compound idioms and Hinglish patterns are bound before tokenization:
```python
POSITIVE_EMOTION_COMPOUNDS = [
    (r'\bgenuinely\s+enjoy\b', 'genuinely_enjoy'),
    (r'\bwell\s+done\b', 'well_done'),
    (r'\bbina\s+kisi\s+error\b', 'bina_kisi_error'),
    (r'\bpefli\s+baar\s+me\b', 'pehli_baar_me'),
]
```

### 5.2 Structural Stopword Filtering
To prevent structural pronouns and conjunctions from skewing negative in code-mixed meme corpora, non-sentiment words are explicitly filtered during TF-IDF vectorization:
```python
STRUCTURAL_STOPWORDS = [
    'about', 'her', 'she', 'him', 'he', 'his', 'it', 'its',
    'because', 'and', 'the', 'a', 'an'
]
```

### 5.3 Retraining Commands
```powershell
# Retrain sentiment TF-IDF + Logistic Regression
python -c "from training.train_classical import train_tfidf_lr; train_tfidf_lr(C=0.55, max_features=18000)"

# Retrain 6 multi-label emotion heads
python -c "from training.train_text_emotions import train_text_emotion_heads; train_text_emotion_heads()"
```
Checkpoints saved to:
- `model_training/checkpoints/best_model/model.joblib`
- `model_training/checkpoints/classical/tfidf_emotion_heads.joblib`
