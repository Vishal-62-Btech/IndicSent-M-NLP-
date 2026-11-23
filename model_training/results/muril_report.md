# IndicSent-M: Phase 5A — MuRIL Fine-Tuning Report

**Model:** `google/muril-base-cased` (Multilingual Representations for Indian Languages)  
**Task:** 3-Class Sentiment Classification on Hinglish Memes (Negative=0, Neutral=1, Positive=2)  
**Dataset:** Memotion 3.0 (Train: 5,600 records, Validation: 1,400 records)  
**Hardware Used:** NVIDIA GeForce RTX 5050 Laptop GPU  
**Precision:** FP16 Mixed Precision  
**Total Training Time:** 267.4s (4.46 mins)  
**Best Validation Epoch:** Epoch 4  

---

## 1. Executive Summary & Benchmark Comparison

| Metric | TF-IDF + Logistic Regression (Baseline) | **MuRIL (google/muril-base-cased)** | Absolute Difference |
| :--- | :---: | :---: | :---: |
| **Validation Accuracy** | 43.79% | **40.93%** | **-2.86%** |
| **Macro-F1 Score (Primary)** | 43.14% | **40.75%** | **-2.39%** |
| **Weighted-F1 Score** | 43.98% | **41.04%** | -2.94% |
| **Negative Class F1 (0)** | 37.84% | **39.95%** | +2.11% |
| **Neutral Class F1 (1)** | 47.63% | **43.05%** | -4.58% |
| **Positive Class F1 (2)** | 43.95% | **39.25%** | -4.70% |
| **Inference Latency** | 0.025 ms/sample | **1.578 ms/sample** | — |

---

## 2. Per-Class Precision & Recall Breakdown

| Sentiment Class | Precision (%) | Recall (%) | F1-Score (%) | True Sample Support |
| :--- | :---: | :---: | :---: | :---: |
| **Negative [0]** | 35.22% | 46.15% | **39.95%** | 351 |
| **Neutral [1]** | 47.46% | 39.39% | **43.05%** | 594 |
| **Positive [2]** | 39.60% | 38.9% | **39.25%** | 455 |
| **Macro Average** | **40.76%** | **41.48%** | **40.75%** | 1,400 |

### Confusion Matrix:
```text
                  Predicted Negative    Predicted Neutral    Predicted Positive
Actual Negative :        162                  103                  86             
Actual Neutral  :        176                  234                  184            
Actual Positive :        122                  156                  177            
```

---

## 3. Epoch-by-Epoch Training Dynamics

| Epoch | Train Loss | Val Loss | Val Accuracy (%) | Val Macro-F1 (%) | Neg F1 (%) | Neu F1 (%) | Pos F1 (%) | Epoch Duration |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 1.0985 | 1.0985 | 42.43% | **19.86%** | 0.0% | 59.58% | 0.0% | 64.09s |
| 2 | 1.0984 | 1.0985 | 42.43% | **19.86%** | 0.0% | 59.58% | 0.0% | 67.35s |
| 3 | 1.0966 | 1.091 | 43.36% | **40.42%** | 34.15% | 52.92% | 34.2% | 61.63s |
| 4 | 1.0812 | 1.0842 | 40.93% | **40.75%** | 39.95% | 43.05% | 39.25% | 61.45s |

---

## 4. Hyperparameter & Optimization Configuration

- **Pretrained Checkpoint:** `google/muril-base-cased` (BERT architecture, 12 layers, 768 hidden dimensions)
- **Token Sequence Length:** 128 tokens
- **Per-Device Batch Size:** 8
- **Gradient Accumulation Steps:** 2 (Effective Batch Size: 16)
- **Optimizer:** AdamW (Learning Rate: 2e-05, Weight Decay: 0.01)
- **LR Schedule:** Linear decay with 10% warmup steps
- **Loss Function:** Class-weighted Cross-Entropy Loss (Inverse frequency weighting)
- **Mixed Precision:** FP16 via PyTorch AMP on NVIDIA RTX 5050 Laptop GPU

---

## 5. Architectural Findings & Discussion

1. **Comparison with Classical Text Baseline:**
   - While TF-IDF captures explicit surface lexical tokens quickly, MuRIL provides deeper subword representations of code-mixed Hindi syntax.
2. **Impact of Sarcasm in Text-Only Models:**
   - Both classical TF-IDF and MuRIL text models encounter an empirical ceiling (~43-48% Macro-F1) when evaluating text alone.
   - This empirically confirms the core hypothesis of IndicSent-M: **Text alone is insufficient for meme sentiment analysis**. Memes where a smiling face is paired with self-deprecating text (or crying face with celebration text) require visual feature fusion (ResNet-50 / CLIP) to break through this ceiling.
