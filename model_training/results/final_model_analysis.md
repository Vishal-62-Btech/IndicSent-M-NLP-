# IndicSent-M: Comprehensive Model Benchmark & Empirical Selection Analysis

**Project Name:** IndicSent-M (Multimodal Sentiment Analysis for Hinglish Memes)  
**Dataset:** Memotion 3.0 (7,000 Total Annotated Records; 5,600 Train, 1,400 Validation)  
**Evaluation Protocol:** Stratified 80:20 Split (Seed=42), Evaluated strictly on the held-out 20% validation split.  
**Primary Metric:** **Macro-F1 Score** (Unweighted mean across Negative, Neutral, and Positive classes to mitigate class imbalance).

---

## 1. Complete Model Comparison Leaderboard

The table below summarizes empirical validation performance across all evaluated models:

| Rank | Model Name | Modality | Accuracy (%) | **Macro F1 (%)** | Negative F1 (%) | Neutral F1 (%) | Positive F1 (%) | Latency (ms/sample) |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **TF-IDF + Logistic Regression** | Text Only (Hinglish OCR) | 43.79% | **43.14%** | 37.84% | 47.63% | 43.95% | 0.025 ms |
| 2 | **TF-IDF + Random Forest** | Text Only (Hinglish OCR) | 44.0% | **40.75%** | 33.65% | 53.26% | 35.36% | 0.063 ms |
| 3 | **TF-IDF + Multinomial Naive Bayes** | Text Only (Hinglish OCR) | 42.86% | **37.06%** | 24.24% | 54.4% | 32.53% | 0.028 ms |
| 4 | **TF-IDF + Linear SVM** | Text Only (Hinglish OCR) | 43.57% | **27.97%** | 6.97% | 58.63% | 18.31% | 0.029 ms |
| 5 | **Majority Baseline** | Text Only (Hinglish OCR) | 42.43% | **19.86%** | 0.0% | 59.58% | 0.0% | 0.03 ms |

---

## 2. Benchmark Observations & Analysis

1. **Majority Baseline Limitations:**
   - Always predicting the majority class (Neutral) achieves 42.43% accuracy, but a dismal **19.86% Macro-F1** (0.0% F1 on Negative and Positive classes). This confirms that raw accuracy is an uninformative metric on imbalanced meme sentiment data.
2. **Classical Text Baselines:**
   - **TF-IDF + Logistic Regression (Balanced)** achieved **43.14% Macro-F1**, providing balanced recall across all three classes (Negative: 41.03%, Neutral: 45.62%, Positive: 43.52%).
   - **Linear SVM** and **Naive Bayes** struggled with negative class recall when text cues were subtle or sarcastic.
3. **Deep Text & Multimodal Architectures:**
   - Subsequent deep learning models (MuRIL, XLM-RoBERTa, ResNet-50, and Multimodal Fusion) build upon these baselines.
   - Multimodal cross-modal attention is designed to resolve irony where smiling imagery contradicts stressful text.

---

## 3. Best Model Selection

- **Current Top-Ranked Model:** `TF-IDF + Logistic Regression`
- **Validation Macro-F1:** `43.14%`
- **Validation Accuracy:** `43.79%`
- **Inference Latency:** `0.025 ms / sample`

*Note: Final model selection and backend checkpoint export will be finalized once all planned deep learning training phases have been executed.*
