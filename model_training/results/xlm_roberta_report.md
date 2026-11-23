# IndicSent-M: Phase 5B — XLM-RoBERTa Fine-Tuning Report

**Model:** `xlm-roberta-base` (Cross-lingual RoBERTa)  
**Task:** 3-Class Sentiment Classification on Hinglish Memes (Negative=0, Neutral=1, Positive=2)  
**Dataset:** Memotion 3.0 (Train: 5,600 records, Validation: 1,400 records)  
**Hardware Used:** NVIDIA GeForce RTX 5050 Laptop GPU  
**Precision:** FP16 Mixed Precision  
**Total Training Time:** 283.3s (4.72 mins)  
**Best Validation Epoch:** Epoch 4  

---

## 1. Executive Summary & Benchmark Comparison

| Metric | TF-IDF + Logistic Reg (Baseline) | MuRIL (google/muril-base-cased) | **XLM-RoBERTa (xlm-roberta-base)** | vs Baseline Diff |
| :--- | :---: | :---: | :---: | :---: |
| **Validation Accuracy** | 43.79% | 40.93% | **42.36%** | **-1.43%** |
| **Macro-F1 Score (Primary)** | 43.14% | 40.75% | **41.66%** | **-1.48%** |
| **Weighted-F1 Score** | 43.98% | 41.04% | **42.43%** | -1.55% |
| **Negative Class F1 (0)** | 37.84% | 39.95% | **37.85%** | +0.01% |
| **Neutral Class F1 (1)** | 47.63% | 43.05% | **46.55%** | -1.08% |
| **Positive Class F1 (2)** | 43.95% | 39.25% | **40.59%** | -3.36% |
| **Inference Latency** | 0.025 ms | 1.578 ms | **1.799 ms/sample** | — |

---

## 2. Per-Class Precision & Recall Breakdown

| Sentiment Class | Precision (%) | Recall (%) | F1-Score (%) | True Sample Support |
| :--- | :---: | :---: | :---: | :---: |
| **Negative [0]** | 35.79% | 40.17% | **37.85%** | 351 |
| **Neutral [1]** | 47.15% | 45.96% | **46.55%** | 594 |
| **Positive [2]** | 41.92% | 39.34% | **40.59%** | 455 |
| **Macro Average** | **41.62%** | **41.82%** | **41.66%** | 1,400 |

### Confusion Matrix:
```text
                  Predicted Negative    Predicted Neutral    Predicted Positive
Actual Negative :        141                  133                  77             
Actual Neutral  :        150                  273                  171            
Actual Positive :        103                  173                  179            
```

---

## 3. Epoch-by-Epoch Training Dynamics

| Epoch | Train Loss | Val Loss | Val Accuracy (%) | Val Macro-F1 (%) | Neg F1 (%) | Neu F1 (%) | Pos F1 (%) | Epoch Duration |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 1.1023 | 1.0975 | 42.43% | **19.86%** | 0.0% | 59.58% | 0.0% | 68.15s |
| 2 | 1.0949 | 1.0915 | 45.0% | **37.42%** | 20.89% | 57.12% | 34.24% | 66.11s |
| 3 | 1.0665 | 1.0945 | 42.36% | **41.66%** | 37.85% | 46.55% | 40.59% | 67.17s |
| 4 | 1.0388 | 1.0981 | 42.21% | **41.35%** | 35.18% | 45.94% | 42.95% | 67.12s |

---

## 4. Hyperparameter & Optimization Configuration

- **Pretrained Checkpoint:** `xlm-roberta-base` (RoBERTa architecture, 12 layers, 768 hidden dimensions)
- **Token Sequence Length:** 128 tokens
- **Per-Device Batch Size:** 8
- **Gradient Accumulation Steps:** 2 (Effective Batch Size: 16)
- **Optimizer:** AdamW (Learning Rate: 2e-05, Weight Decay: 0.01)
- **LR Schedule:** Linear decay with 10% warmup steps
- **Loss Function:** Class-weighted Cross-Entropy Loss (Inverse frequency weighting)
- **Mixed Precision:** FP16 via PyTorch AMP on NVIDIA RTX 5050 Laptop GPU
