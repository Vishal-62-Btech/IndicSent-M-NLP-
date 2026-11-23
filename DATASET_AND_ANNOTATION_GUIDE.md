# IndicSent-M: Dataset Specification & Emotion Annotation Guide
================================================================

> **Dataset Provenance, Split Hygiene, and Multi-Task Annotation Methodology**  
> *Target Benchmark: Memotion 3.0 (AAAI De-Factify 2 Shared Task)*  
> *Language: Hinglish (Code-Mixed Hindi-English in Romanized Script)*

---

## 1. Dataset Provenance & Overview

IndicSent-M utilizes the **Memotion 3.0** benchmark dataset, originally introduced in the AAAI De-Factify 2 Workshop. The dataset comprises authentic Indian internet memes sourced from social media platforms (Twitter, Instagram, Reddit, Facebook).

### 1.1 Modality Breakdown
- **Visual Modality:** Natural meme images capturing Bollywood film scenes, television stills, political figures, sports moments, cartoon templates, and conversational reaction images.
- **Textual Modality:** Embedded meme captions rendered in Romanized Hindi-English ("Hinglish"), containing:
  - Code-mixing (inter-sentential and intra-sentential language switches).
  - Phonetic spelling variants (e.g., *achha*, *accha*, *acha*; *bahut*, *bht*, *bohot*).
  - Colloquial slang (*bhai*, *mitron*, *jugaad*, *kat gaya*).
  - Sarcastic praise (positive surface words paired with negative situational contexts).

### 1.2 Target Task Schema
1. **Primary Task (3-Class Sentiment Classification):**
   - `Negative (0)`: Disparagement, frustration, sadness, condemnation, or distress.
   - `Neutral (1)`: Informational, observational, everyday relatable, or ambiguous humor.
   - `Positive (2)`: Celebration, affection, congratulation, humor, or triumph.
2. **Auxiliary Task (6-Class Multi-Label Emotion Classification):**
   - Modeled after the classical Ekman / Plutchik affective taxonomies:
     - `Happiness 😊`: Joy, celebration, wholesome humor, affection, pride.
     - `Sadness 😢`: Disappointment, heartbreak, exam failure, melancholy, feeling ignored.
     - `Anger 😡`: Irritation, fury, indignation, workplace rage, political outrage.
     - `Fear 😨`: Dread, panic over deadlines, terrifying confrontations, anxiety.
     - `Disgust 🤢`: Revulsion, contempt, aversion to cringe behavior, rejection.
     - `Surprise 😲`: Shock, disbelief, unexpected exam results, plotting twists.

---

## 2. Train-Validation Split Hygiene & Zero Leakage

To preserve strict academic evaluation integrity, the raw dataset was partitioned into stratified splits before any model training or annotation:
- **Total Samples:** $7,000$ memes.
- **Training Split:** $5,600$ memes ($80.0\%$).
- **Validation Split:** $1,400$ memes ($20.0\%$).

### 2.1 Split Integrity Verification
- **Image Filename Disjointness:** Verified via cryptographic SHA-256 and string set intersection:
  $$\text{Train Images} \cap \text{Validation Images} = \emptyset$$
  $$\text{Count of Overlapping Samples} = 0$$
- **Stratification:** Maintained across 3-class sentiment distributions:
  - Training ($N=5,600$): Negative: $1,405$ ($25.09\%$), Neutral: $2,376$ ($42.43\%$), Positive: $1,819$ ($32.48\%$).
  - Validation ($N=1,400$): Negative: $351$ ($25.07\%$), Neutral: $594$ ($42.43\%$), Positive: $455$ ($32.50\%$).

---

## 3. Emotion Annotation Methodology & Masking

Because the original Memotion 3.0 annotations focused on coarse humor/sarcasm scales rather than discrete Ekman emotions, an affective annotation schema was implemented to curate verified emotion labels.

### 3.1 Verification & Active Label Representation
To prevent data contamination:
- Every meme in the emotion annotation CSV contains:
  - `image_name`: Unique image identifier matching the parent sentiment split.
  - `clean_text`: Preprocessed Hinglish OCR text.
  - `sentiment_3class`: Target sentiment label.
  - `happiness, sadness, anger, fear, disgust, surprise`: Binary integer flags (`1` = active, `0` = inactive).
  - `is_verified`: Binary flag (`1` = verified annotated sample, `0` = unannotated sample).
  - `annotator_notes`: Rationales, keywords, and lexicon triggers.

### 3.2 Masked Multi-Label Loss Formulation
Unannotated samples are not discarded from training; instead, they continue to supervise the sentiment head while being masked out of the emotion loss:
$$m_i = \begin{cases} 1, & \text{if } \text{is\_verified}_i = 1 \\ 0, & \text{if } \text{is\_verified}_i = 0 \end{cases}$$

$$\mathcal{L}_{em}(i) = m_i \cdot \sum_{k=1}^6 \text{BCEWithLogits}(z_{em}^{(k)}(i), y_k(i), pos\_weight_k)$$

If $m_i = 0$, $\mathcal{L}_{em}(i) = 0$, ensuring zero spurious gradient updates are backpropagated into the shared fusion network from unverified samples.

---

## 4. Empirical Emotion Label Distribution

### 4.1 Training Split ($N=5,600$ Total, $N=1,564$ Verified Annotated)
Through targeted rare-emotion lexicon curation (`--curate-rare`), representation was expanded substantially across minority classes:

| Emotion | Active Count ($y_k = 1$) | Inactive Count ($y_k = 0$) | Prevalence (%) | Imbalance Ratio | Pos Weight ($pos\_weight_k$) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Happiness** | 1,009 | 555 | 64.51% | 0.55 : 1 | 0.55 |
| **Sadness** | 346 | 1,218 | 22.12% | 3.52 : 1 | 3.52 |
| **Anger** | 110 | 1,454 | 7.03% | 13.22 : 1 | 10.76 |
| **Fear** | 80 | 1,484 | 5.12% | 18.55 : 1 | 17.40 |
| **Disgust** | 30 | 1,534 | 1.92% | 51.13 : 1 | 46.39 |
| **Surprise** | 110 | 1,454 | 7.03% | 13.22 : 1 | 12.84 |
| **Total Decisions** | **1,685** | **7,699** | **17.96% active** | — | — |

### 4.2 Validation Split ($N=1,400$ Total, $N=230$ Verified Annotated)
The validation split was kept completely untouched as a blind test harness:

| Emotion | Active Count ($y_k = 1$) | Inactive Count ($y_k = 0$) | Prevalence (%) |
|:---|:---:|:---:|:---:|
| **Happiness** | 91 | 139 | 39.57% |
| **Sadness** | 86 | 144 | 37.39% |
| **Anger** | 49 | 181 | 21.30% |
| **Fear** | 11 | 219 | 4.78% |
| **Disgust** | 2 | 228 | 0.87% |
| **Surprise** | 11 | 219 | 4.78% |
| **Total Decisions** | **250** | **1,130** | **18.12% active** |

### 4.3 Key Empirical Insight on Dataset Imbalance
Indian internet memes in Memotion 3.0 are inherently comedic (`happiness`), self-deprecating/struggle-themed (`sadness`), or indignant (`anger`). Genuine `disgust` ($0.87\%$ in validation) and `fear` ($4.78\%$ in validation) are naturally scarce phenomena in meme culture. Acknowledging this natural empirical distribution is vital for honest evaluation.

---

## 5. Curation Tool & Workflow

The emotion annotation pipeline is managed via `model_training/src/preprocessing/emotion_annotation_tool.py`:
- **Interactive Annotation:**
  ```powershell
  python model_training/src/preprocessing/emotion_annotation_tool.py --interactive --split train
  ```
- **Automated Lexicon Matching:**
  ```powershell
  python model_training/src/preprocessing/emotion_annotation_tool.py --auto-lexicon --split train
  ```
- **Targeted Rare Emotion Expansion:**
  ```powershell
  python model_training/src/preprocessing/emotion_annotation_tool.py --curate-rare --split train
  ```
- **Audit Split Integrity:**
  ```powershell
  python model_training/src/preprocessing/emotion_annotation_tool.py --audit
  ```
