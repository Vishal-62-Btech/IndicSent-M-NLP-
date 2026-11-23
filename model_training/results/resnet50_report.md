# IndicSent-M: Phase 6 — ResNet-50 Training Report

**Model:** ResNet-50  
**Task:** 3-Class Sentiment Classification on Meme Images (Negative=0, Neutral=1, Positive=2)  
**Dataset:** Memotion 3.0 (Train: 5,600 images, Validation: 1,400 images)  
**Hardware Used:** NVIDIA GeForce RTX 5050 Laptop GPU  
**Precision:** FP16 Mixed Precision  
**Total Training Time:** 225.3s (3.75 mins)  
**Best Validation Epoch:** Epoch 3  

---

## 1. Performance Overview

| Metric | Score |
| :--- | :---: |
| **Validation Accuracy** | **40.43%** |
| **Macro-F1 Score (Primary)** | **37.04%** |
| **Weighted-F1 Score** | 38.88% |
| **Negative Class F1 (0)** | 30.64% |
| **Neutral Class F1 (1)** | 50.75% |
| **Positive Class F1 (2)** | 29.73% |
| **Inference Latency** | 4.511 ms/sample |

### Confusion Matrix:
```text
                  Predicted Negative    Predicted Neutral    Predicted Positive
Actual Negative :        106                  186                  59             
Actual Neutral  :        147                  354                  93             
Actual Positive :        88                   261                  106            
```
