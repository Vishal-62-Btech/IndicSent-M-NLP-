# IndicSent-M: Phase 6 — EfficientNet-B0 Training Report

**Model:** EfficientNet-B0  
**Task:** 3-Class Sentiment Classification on Meme Images (Negative=0, Neutral=1, Positive=2)  
**Dataset:** Memotion 3.0 (Train: 5,600 images, Validation: 1,400 images)  
**Hardware Used:** NVIDIA GeForce RTX 5050 Laptop GPU  
**Precision:** FP16 Mixed Precision  
**Total Training Time:** 197.0s (3.28 mins)  
**Best Validation Epoch:** Epoch 3  

---

## 1. Performance Overview

| Metric | Score |
| :--- | :---: |
| **Validation Accuracy** | **37.5%** |
| **Macro-F1 Score (Primary)** | **36.54%** |
| **Weighted-F1 Score** | 37.64% |
| **Negative Class F1 (0)** | 32.24% |
| **Neutral Class F1 (1)** | 44.43% |
| **Positive Class F1 (2)** | 32.95% |
| **Inference Latency** | 3.582 ms/sample |

### Confusion Matrix:
```text
                  Predicted Negative    Predicted Neutral    Predicted Positive
Actual Negative :        123                  126                  102            
Actual Neutral  :        159                  257                  178            
Actual Positive :        130                  180                  145            
```
