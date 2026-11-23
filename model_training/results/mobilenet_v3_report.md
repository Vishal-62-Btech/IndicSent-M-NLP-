# IndicSent-M: Phase 6 — MobileNetV3-Large Training Report

**Model:** MobileNetV3-Large  
**Task:** 3-Class Sentiment Classification on Meme Images (Negative=0, Neutral=1, Positive=2)  
**Dataset:** Memotion 3.0 (Train: 5,600 images, Validation: 1,400 images)  
**Hardware Used:** NVIDIA GeForce RTX 5050 Laptop GPU  
**Precision:** FP16 Mixed Precision  
**Total Training Time:** 168.9s (2.81 mins)  
**Best Validation Epoch:** Epoch 4  

---

## 1. Performance Overview

| Metric | Score |
| :--- | :---: |
| **Validation Accuracy** | **38.21%** |
| **Macro-F1 Score (Primary)** | **36.25%** |
| **Weighted-F1 Score** | 37.74% |
| **Negative Class F1 (0)** | 25.08% |
| **Neutral Class F1 (1)** | 42.89% |
| **Positive Class F1 (2)** | 40.78% |
| **Inference Latency** | 3.359 ms/sample |

### Confusion Matrix:
```text
                  Predicted Negative    Predicted Neutral    Predicted Positive
Actual Negative :        77                   143                  131            
Actual Neutral  :        115                  249                  230            
Actual Positive :        71                   175                  209            
```
