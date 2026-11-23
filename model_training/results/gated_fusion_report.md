# IndicSent-M: Phase 7 — MuRIL + ResNet-50 (Gated Fusion) Training Report

**Model Architecture:** MuRIL + ResNet-50 (Gated Fusion)  
**Text Modality Backbone:** google/muril-base-cased  
**Vision Modality Backbone:** resnet50  
**Fusion Mechanism:** Gated  
**Task:** 3-Class Sentiment Classification on Hinglish Memes (Negative=0, Neutral=1, Positive=2)  
**Dataset:** Memotion 3.0 (Train: 5,600 multimodal samples, Validation: 1,400 multimodal samples)  
**Hardware Used:** NVIDIA GeForce RTX 5050 Laptop GPU  
**Precision:** FP16 Mixed Precision  
**Total Training Time:** 198.7s (3.31 mins)  
**Best Validation Epoch:** Epoch 2  

---

## 1. Performance Overview

| Metric | Score |
| :--- | :---: |
| **Validation Accuracy** | **37.64%** |
| **Macro-F1 Score (Primary)** | **37.24%** |
| **Weighted-F1 Score** | 37.59% |
| **Negative Class F1 (0)** | 37.83% |
| **Neutral Class F1 (1)** | 41.15% |
| **Positive Class F1 (2)** | 32.75% |
| **Inference Latency** | 5.961 ms/sample |

### Confusion Matrix:
```text
                  Predicted Negative    Predicted Neutral    Predicted Positive
Actual Negative :        174                  104                  73             
Actual Neutral  :        227                  221                  146            
Actual Positive :        168                  155                  132            
```
