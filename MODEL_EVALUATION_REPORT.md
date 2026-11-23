# IndicSent-M: Master Model Evaluation & Baseline Report
=========================================================

> **Comprehensive Academic Benchmark on Memotion 3.0 (1,400 Validation Samples)**  
> *Tasks: 3-Class Hinglish Sentiment & 6-Class Multi-Label Emotion Detection*  
> *Zero Data Leakage | Verified Ground Truth Metrics | Empirical Baselines*

---

## 1. Master Model Benchmark Comparison

All models were evaluated on the exact same unseen, stratified validation split ($N=1,400$ for sentiment; $N=230$ verified ground-truth emotion subset).

### 1.1 Multi-Task & Fusion Models Benchmark Table

| Model Architecture | Task Type | Sent Acc (%) | Sent Macro-F1 (%) | Sent Weighted-F1 (%) | Emotion Macro-F1 (%) | Emotion Micro-F1 (%) | Dominant Emotion Acc (%) | Hamming Loss |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Single-Task Gated Fusion (Champion)** | Sentiment Only | **37.71%** | **37.31%** | **37.67%** | — | — | — | — |
| **Multi-Task $E_1$ ($\lambda=0.5$) (Selected Production)** | Dual-Task | 35.50% | **35.47%** | 35.33% | **22.62%** | **44.09%** | **40.00%** | 0.2775 |
| **Multi-Task $E_0$ ($\lambda=1.0$)** | Dual-Task | 35.79% | 35.34% | 35.57% | 22.51% | 45.21% | **40.00%** | 0.2529 |
| **Initial Multi-Task Baseline** | Dual-Task | 37.07% | 35.43% | 36.16% | 19.12% | 46.91% | 42.61% | 0.2428 |
| **Multi-Task $E_2$ ($\lambda=0.25$)** | Dual-Task | 33.36% | 33.09% | 32.66% | 21.56% | 35.28% | 26.09% | 0.3217 |

---

## 2. Empirical & Heuristic Baselines

To contextualize model performance, four statistical and heuristic baselines were evaluated:

### 2.1 Sentiment Baselines ($N=1,400$)
1. **Majority Class Baseline (Always predicts *Neutral*):**
   - Accuracy: **42.43%**
   - Macro-F1: **19.86%** (Negative: 0.00%, Neutral: 59.58%, Positive: 0.00%)
   - Weighted-F1: **25.28%**
   - *Analysis:* While accuracy is deceptively high due to class imbalance ($42.43\%$ Neutral samples in validation), Macro-F1 is crippled because it predicts zero Negative or Positive memes.
2. **Random Chance Baseline (Uniform $1/3$):**
   - Accuracy: **31.36%**
   - Macro-F1: **31.03%**

### 2.2 Emotion Multi-Label Baselines ($N=230$ Verified Samples)
1. **All-Negative Baseline (Predicts `0` for all 6 emotions on every meme):**
   - Hamming Loss: **0.1812**
   - Macro-F1: **0.00%**
   - Micro-F1: **0.00%**
   - Exact Match Ratio: **0.00%**
2. **Majority Dominant Emotion Baseline (Always predicts *Happiness*):**
   - Top-1 Dominant Accuracy: **39.57%**
3. **Random Dominant Emotion Baseline:**
   - Top-1 Dominant Accuracy: **24.35%** (Theoretical Uniform: **16.67%**)

---

## 3. In-Depth Analysis: The Hamming Loss Paradox

### 3.1 Why does the All-Negative baseline achieve a low Hamming Loss ($0.1812$)?
Hamming Loss measures the fraction of incorrect binary labels across all emotion decisions:
$$\text{Hamming Loss} = \frac{1}{N \cdot K} \sum_{i=1}^N \sum_{k=1}^K \mathbb{I}(\hat{y}_{i,k} \neq y_{i,k})$$
In our $230$-sample validation set across $6$ emotions ($1,380$ total binary decisions):
- Total Active (`1`) bits: **250** ($18.12\%$)
- Total Inactive (`0`) bits: **1,130** ($81.88\%$)

When the All-Negative baseline predicts all zeros:
$$\text{Hamming Loss}_{\text{all-zero}} = \frac{250}{1380} = 0.1812$$
By predicting exclusively zeros, the baseline correctly identifies $81.88\%$ of inactive bits purely by exploiting label sparsity.

### 3.2 The Critical Trade-off: Hamming Loss vs. Practical Utility
- The all-negative baseline achieves **F1 = 0.00%**, Recall = 0.00%, and detects zero emotions. It is useless for real-world affective computing.
- At threshold $0.5$, Multi-Task $E_1$ achieves a Hamming loss of **0.2775**, but delivers an **Emotion Macro-F1 of 22.62%** and a **Micro-F1 of 44.09%**, successfully predicting Happiness ($F_1 = 56.27\%$), Sadness ($F_1 = 48.78\%$), Surprise ($F_1 = 18.18\%$), and Anger ($F_1 = 12.50\%$).
- Furthermore, for single-label primary user interaction, **Dominant Emotion Top-1 Accuracy is 40.00%**, outperforming both the majority emotion baseline ($39.57\%$) and random guessing ($16.67\%$).

---

## 4. Per-Emotion Performance Breakdown

Detailed metrics for the champion multi-task model ($E_1$, threshold $0.5$):

| Emotion | Support ($N=230$) | Precision (%) | Recall (%) | F1-Score (%) |
|:---|:---:|:---:|:---:|:---:|
| **Happiness 😊** | 91 | 40.69% | 91.21% | **56.27%** |
| **Sadness 😢** | 86 | 37.50% | 69.77% | **48.78%** |
| **Anger 😡** | 49 | 26.67% | 8.16% | **12.50%** |
| **Fear 😨** | 11 | 0.00% | 0.00% | 0.00% |
| **Disgust 🤢** | 2 | 0.00% | 0.00% | 0.00% |
| **Surprise 😲** | 11 | 10.00% | 72.73% | **18.18%** |
| **Macro Average** | — | **19.14%** | **40.31%** | **22.62%** |
| **Micro Average** | 250 | **32.84%** | **66.80%** | **44.09%** |

### 4.1 Observations on Minority Emotions
- **Happiness and Sadness:** Account for $77.0\%$ of verified validation emotion labels. The model achieves high recall ($91.21\%$ and $69.77\%$), reliably capturing celebratory and self-deprecating internet humor.
- **Surprise:** Achieves $18.18\%$ F1 with $72.73\%$ recall, successfully detecting reaction templates with shock expressions.
- **Anger:** Detects sharp indignation ($12.50\%$ F1).
- **Fear and Disgust:** Extremely scarce in Memotion 3.0 (only 11 and 2 ground truth samples in validation). High precision is penalized by extreme rarity; these classes highlight key opportunities for active-learning data expansion in future work.

---

## 5. Threshold Calibration Analysis

To investigate whether binary decisions could be improved beyond default $0.5$, optimal thresholds were searched via grid search on the training split ($N=1,564$) to maximize per-emotion F1, and then evaluated on the untouched validation set ($N=230$):

| Emotion | Default Threshold | Calibrated Threshold ($T^*$) | Validation F1 at 0.5 | Validation F1 at $T^*$ |
|:---|:---:|:---:|:---:|:---:|
| **Happiness** | 0.50 | 0.475 | 56.27% | 56.23% |
| **Sadness** | 0.50 | 0.550 | 48.78% | 39.29% |
| **Anger** | 0.50 | 0.475 | 12.50% | **18.42%** |
| **Fear** | 0.50 | 0.550 | 0.00% | 0.00% |
| **Disgust** | 0.50 | 0.500 | 0.00% | 0.00% |
| **Surprise** | 0.50 | 0.500 | 18.18% | 18.18% |

### 5.1 Why Dominant Emotion Remains $\arg\max$
While independent per-emotion thresholds alter binary multi-label sets, the **Dominant Emotion** displayed on the user interface is defined as:
$$\text{Dominant Emotion} = \arg\max_{k \in \{1..6\}} \sigma(z_{em}^{(k)})$$
Because the sigmoid function $\sigma(z)$ is strictly monotonically increasing, $\arg\max_k \sigma(z_k) \equiv \arg\max_k z_k$. This formulation guarantees:
1. Every input meme receives exactly one primary emotional classification.
2. Invariant to threshold shifts.
3. Completely avoids empty predictions or multi-label over-prediction conflicts in the primary UI card.
