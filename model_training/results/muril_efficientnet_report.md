# IndicSent-M: Phase 7 — MuRIL + EfficientNet-B0 (Concat Fusion) Training Report

**Model Architecture:** MuRIL + EfficientNet-B0 (Concat Fusion)  
**Text Modality Backbone:** google/muril-base-cased  
**Vision Modality Backbone:** efficientnet_b0  
**Fusion Mechanism:** Concat  
**Task:** 3-Class Sentiment Classification on Hinglish Memes (Negative=0, Neutral=1, Positive=2)  
**Dataset:** Memotion 3.0 (Train: 5,600 multimodal samples, Validation: 1,400 multimodal samples)  
**Hardware Used:** NVIDIA GeForce RTX 5050 Laptop GPU  
**Precision:** FP16 Mixed Precision  
**Total Training Time:** 187.5s (3.13 mins)  
**Best Validation Epoch:** Epoch 4  

---

## 1. Performance Overview

| Metric | Score |
| :--- | :---: |
| **Validation Accuracy** | **36.0%** |
| **Macro-F1 Score (Primary)** | **35.43%** |
| **Weighted-F1 Score** | 36.25% |
| **Negative Class F1 (0)** | 30.99% |
| **Neutral Class F1 (1)** | 40.44% |
| **Positive Class F1 (2)** | 34.85% |
| **Inference Latency** | 5.271 ms/sample |

### Confusion Matrix:
```text
                  Predicted Negative    Predicted Neutral    Predicted Positive
Actual Negative :        117                  122                  112            
Actual Neutral  :        157                  222                  215            
Actual Positive :        130                  160                  165            
```
