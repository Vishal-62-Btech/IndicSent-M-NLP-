# IndicSent-M: Phase 7 — XLM-RoBERTa + ResNet-50 (Concat Fusion) Training Report

**Model Architecture:** XLM-RoBERTa + ResNet-50 (Concat Fusion)  
**Text Modality Backbone:** xlm-roberta-base  
**Vision Modality Backbone:** resnet50  
**Fusion Mechanism:** Concat  
**Task:** 3-Class Sentiment Classification on Hinglish Memes (Negative=0, Neutral=1, Positive=2)  
**Dataset:** Memotion 3.0 (Train: 5,600 multimodal samples, Validation: 1,400 multimodal samples)  
**Hardware Used:** NVIDIA GeForce RTX 5050 Laptop GPU  
**Precision:** FP16 Mixed Precision  
**Total Training Time:** 211.1s (3.52 mins)  
**Best Validation Epoch:** Epoch 4  

---

## 1. Performance Overview

| Metric | Score |
| :--- | :---: |
| **Validation Accuracy** | **31.43%** |
| **Macro-F1 Score (Primary)** | **30.0%** |
| **Weighted-F1 Score** | 28.51% |
| **Negative Class F1 (0)** | 38.88% |
| **Neutral Class F1 (1)** | 21.61% |
| **Positive Class F1 (2)** | 29.51% |
| **Inference Latency** | 6.044 ms/sample |

### Confusion Matrix:
```text
                  Predicted Negative    Predicted Neutral    Predicted Positive
Actual Negative :        242                  48                   61             
Actual Neutral  :        377                  86                   131            
Actual Positive :        275                  68                   112            
```
