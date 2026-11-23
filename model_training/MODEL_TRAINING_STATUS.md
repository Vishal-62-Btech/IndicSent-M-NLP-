# IndicSent-M: Model Training Status Dashboard

Last Updated: 2026-10-31

## Status Summary

| Model Family | Model Name | Status | Epochs Completed | Best Macro-F1 | Best Accuracy | Checkpoint Path |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Classical ML** | Majority Baseline | COMPLETED | — | 19.86% | 42.43% | `model_training/checkpoints/classical/majority_baseline.joblib` |
| **Classical ML** | TF-IDF + Naive Bayes | COMPLETED | — | 37.06% | 42.86% | `model_training/checkpoints/classical/tf_idf_plus_multinomial_naive_bayes.joblib` |
| **Classical ML** | TF-IDF + Logistic Regression | COMPLETED | — | 43.14% | 43.79% | `model_training/checkpoints/classical/tf_idf_plus_logistic_regression.joblib` |
| **Classical ML** | TF-IDF + Linear SVM | COMPLETED | — | 27.97% | 43.57% | `model_training/checkpoints/classical/tf_idf_plus_linear_svm.joblib` |
| **Classical ML** | TF-IDF + Random Forest | COMPLETED | — | 40.75% | 44.00% | `model_training/checkpoints/classical/tf_idf_plus_random_forest.joblib` |
| **Deep Text** | MuRIL (`google/muril-base-cased`) | COMPLETED | 4/4 | 40.75% | 40.93% | `model_training/checkpoints/muril/` |
| **Deep Text** | XLM-RoBERTa (`xlm-roberta-base`) | COMPLETED | 4/4 | 41.66% | 42.36% | `model_training/checkpoints/xlm_roberta/` |
| **Vision Only** | ResNet-50 | COMPLETED | 4/4 | 37.04% | 40.43% | `model_training/checkpoints/resnet50/` |
| **Vision Only** | EfficientNet-B0 | COMPLETED | 4/4 | 36.54% | 37.50% | `model_training/checkpoints/efficientnet_b0/` |
| **Vision Only** | MobileNetV3-Large | COMPLETED | 4/4 | 36.25% | 38.21% | `model_training/checkpoints/mobilenet_v3/` |
| **Multimodal** | MuRIL + ResNet-50 (Concat) | COMPLETED | 4/4 | 35.14% | 37.57% | `model_training/checkpoints/multimodal/muril_resnet50/` |
| **Multimodal** | MuRIL + EfficientNet-B0 (Concat) | COMPLETED | 4/4 | 35.43% | 36.00% | `model_training/checkpoints/multimodal/muril_efficientnet/` |
| **Multimodal** | XLM-RoBERTa + ResNet-50 (Concat) | COMPLETED | 4/4 | 30.00% | 31.43% | `model_training/checkpoints/multimodal/xlm_resnet50/` |
| **Multimodal** | MuRIL + ResNet-50 (Gated Fusion) | COMPLETED | 4/4 | 37.24% | 37.64% | `model_training/checkpoints/multimodal/gated_fusion/` |

- **Total Experiments:** 14
- **Successfully Completed:** 14 (100%)
- **In Progress:** 0
- **Failed:** 0
- **Not Attempted:** 0
