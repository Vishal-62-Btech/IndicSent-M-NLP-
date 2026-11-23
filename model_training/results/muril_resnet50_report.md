# IndicSent-M: Phase 7 — MuRIL + ResNet-50 (Concat Fusion) Training Report

**Model Architecture:** MuRIL + ResNet-50 (Concat Fusion)  
**Text Modality Backbone:** google/muril-base-cased  
**Vision Modality Backbone:** resnet50  
**Fusion Mechanism:** Concat  
**Task:** 3-Class Sentiment Classification on Hinglish Memes (Negative=0, Neutral=1, Positive=2)  
**Dataset:** Memotion 3.0 (Train: 5,600 multimodal samples, Validation: 1,400 multimodal samples)  
**Hardware Used:** NVIDIA GeForce RTX 5050 Laptop GPU  
**Precision:** FP16 Mixed Precision  
**Total Training Time:** 250.2s (4.17 mins)  
**Best Validation Epoch:** Epoch 4  

---

## 1. Performance Overview

| Metric | Score |
| :--- | :---: |
| **Validation Accuracy** | **37.57%** |
| **Macro-F1 Score (Primary)** | **35.14%** |
| **Weighted-F1 Score** | 35.75% |
| **Negative Class F1 (0)** | 39.18% |
| **Neutral Class F1 (1)** | 44.33% |
| **Positive Class F1 (2)** | 21.9% |
| **Inference Latency** | 6.384 ms/sample |

### Confusion Matrix:
```text
                  Predicted Negative    Predicted Neutral    Predicted Positive
Actual Negative :        190                  134                  27             
Actual Neutral  :        255                  268                  71             
Actual Positive :        174                  213                  68             
```
