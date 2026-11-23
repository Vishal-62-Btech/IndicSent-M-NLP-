# IndicSent-M: Exploratory Dataset Analysis (Memotion 3.0)

**Project Name:** IndicSent-M (Multimodal Sentiment Analysis of Hinglish Memes)  
**Corpus:** Memotion 3.0 Dataset (De-Factify 2, AAAI-22)  
**Date of Analysis:** 2026-10-31  

---

## 1. Dataset Dimensions & Schema

The dataset consists of two official subsets:
- **Training Subset:** 7,000 records, 8 columns
- **Test Subset:** 1,500 records, 3 columns

### Training Set Schema (`annotations/train.csv`):
| Column Name | Non-Null Count | Data Type | Description |
| :--- | :---: | :---: | :--- |
| `Unnamed: 0` | 7,000 | `int64` | Primary Integer Identifier (maps to `<id>.jpg`) |
| `image_url` | 7,000 | `object` | Source web URL of the meme image |
| `humour` | 7,000 | `object` | Humour level: `not_funny`, `funny`, `very_funny`, `hilarious` |
| `sarcastic` | 7,000 | `object` | Sarcasm level: `not_sarcastic`, `general`, `twisted_meaning`, `very_twisted` |
| `offensive` | 7,000 | `object` | Offensiveness: `not_offensive`, `slight`, `very_offensive`, `hateful_offensive` |
| `motivational` | 7,000 | `object` | Motivation flag: `not_motivational`, `motivational` |
| `overall` | 7,000 | `object` | **Primary 5-class Sentiment Label** |
| `ocr` | 7,000 | `object` | Extracted optical character recognition text |

### Test Set Schema (`annotations/test.csv`):
| Column Name | Non-Null Count | Data Type | Description |
| :--- | :---: | :---: | :--- |
| `Unnamed: 0` | 1,500 | `int64` | Primary Integer Identifier (maps to `<id>.jpg`) |
| `image_url` | 1,500 | `object` | Source web URL of the meme image |
| `ocr` | 1,500 | `object` | Extracted optical character recognition text |

---

## 2. Data Integrity & Missing Values Analysis

- **Missing Values in CSV:**
  - Training Set: **0 null / NaN values** across all 8 columns.
  - Test Set: **0 null / NaN values** across all 3 columns.
- **Empty OCR Text:**
  - Training Set: **39 records** contain empty strings or pure whitespace.
  - Test Set: **16 records** contain empty strings or pure whitespace.
  - *Handling Strategy:* In accordance with multimodal research methodology, these records are **NOT discarded**. Meme graphics contain rich visual sentiment even when OCR is blank. The cleaning module assigns a dedicated token `[NO_TEXT]` so language backbones and fusion heads can process them without crashing.
- **Duplicate Records:**
  - Exact row duplicates: **0** in train, **0** in test.
  - Duplicate `Unnamed: 0` image IDs: **0** (strictly unique 0 to 6999).
  - Duplicate `image_url`: **0** unique source links.
  - Duplicate OCR texts: **61** in train (popular meme templates with identical captions).

---

## 3. Sentiment Label Analysis & 3-Class Mapping

The ground-truth sentiment column is `overall`. The official Memotion 3 annotation guidelines and reference `format.py` map the original 5 classes into 3 target categories:

$$
\text{Sentiment} = 
\begin{cases} 
\text{Positive} & \text{if } \text{overall} \in \{\text{'positive'}, \text{'very\_positive'}\} \\
\text{Neutral} & \text{if } \text{overall} = \text{'neutral'} \\
\text{Negative} & \text{if } \text{overall} \in \{\text{'negative'}, \text{'very\_negative'}\}
\end{cases}
$$

### Label Distribution Table:

| Original Label | Count | Percentage | Mapped 3-Class Label | 3-Class Count | 3-Class Percentage | Class Index |
| :--- | :---: | :---: | :--- | :---: | :---: | :---: |
| `neutral` | 2,970 | 42.43% | **Neutral** | 2,970 | 42.43% | 1 |
| `positive` | 1,902 | 27.17% | **Positive** | 2,275 | 32.50% | 2 |
| `very_positive` | 373 | 5.33% | (Merged into Positive) | — | — | — |
| `negative` | 1,454 | 20.77% | **Negative** | 1,755 | 25.07% | 0 |
| `very_negative` | 301 | 4.30% | (Merged into Negative) | — | — | — |
| **Total** | **7,000** | **100.00%** | **Total Labeled** | **7,000** | **100.00%** | — |

### Key Observation on Class Balance:
The dataset exhibits moderate natural class imbalance:
- **Neutral** is the plurality class (42.43%).
- **Positive** forms 32.50% of the dataset.
- **Negative** forms 25.07% of the dataset.
This validates the prompt's instruction: **Macro-F1 score must be the primary evaluation metric**, as accuracy alone would be biased toward the majority Neutral class.

---

## 4. Linguistic Characteristics of Meme Text

An audit of the 7,000 training OCR texts reveals:
- **Latin / Hinglish / English Script:** 6,956 samples (**99.37%**) contain Romanized text.
- **Devanagari Hindi Script:** 103 samples (**1.47%**) contain Devanagari Unicode characters.
- **Watermarks & URLs:** 108 samples contain website URLs (e.g., `memecenter.com`, `imgflip.com`), which are stripped during preprocessing to prevent spurious feature learning.
- **Text Length Statistics:**
  - Mean word count: **18.88 words** ($\pm 13.03$)
  - Median word count: **16 words**
  - Min word count: **0 words** (empty OCR)
  - Max word count: **172 words**
  - 95th percentile word count: **43 words** (well within transformer sequence limit of 128 tokens).

---

## 5. Visual Modality & Image Integrity Audit

A comprehensive Python PIL and MD5 hash audit was executed across all 10,000 images in the repository:

| Image Split | Directory | Expected Count | Verified Readable | Color Mode | Format | Corrupted Files |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Train Images** | `dataset/Memotion 3/images/train/` | 7,000 | 7,000 | 100% RGB | 100% JPEG | **0** |
| **Validation Images** | `dataset/Memotion 3/images/validation/`| 1,500 | 1,500 | 100% RGB | 100% JPEG | **0** |
| **Test Images** | `dataset/Memotion 3/images/test/` | 1,500 | 1,500 | 100% RGB | 100% JPEG | **0** |

### Duplicate Image Hash Analysis:
- Intra-split duplicates: **0 duplicate MD5 hashes** in train, test, or validation images.
- Cross-split duplicate hashes:
  - Train $\cap$ Test: **0**
  - Train $\cap$ Validation: **0**
  - Validation $\cap$ Test: **0**
**Conclusion:** There is **zero visual data leakage** between splits.

---

## 6. Official 80:20 Stratified Split Execution

Because the external validation ground-truth labels were historically locked in an unavailable encrypted archive, an exact **80:20 stratified split** was created on the 7,000 labeled training records with a fixed random seed ($42$):

| Split | Sample Count | Neutral (Count / %) | Positive (Count / %) | Negative (Count / %) | Output CSV |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Train (80%)** | 5,600 | 2,376 (42.43%) | 1,820 (32.50%) | 1,404 (25.07%) | `model_training/data_splits/train.csv` |
| **Val (20%)** | 1,400 | 594 (42.43%) | 455 (32.50%) | 351 (25.07%) | `model_training/data_splits/validation.csv` |
| **Test (Unlabeled)** | 1,500 | — | — | — | `model_training/data_splits/test_unlabeled.csv` |

Every subsequent experiment (classical ML, deep text transformers, image CNNs/ViTs, and multimodal fusion) will strictly evaluate on this exact split.
