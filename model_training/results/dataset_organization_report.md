# IndicSent-M: Dataset Organization & Directory Restructuring Report

**Project Name:** IndicSent-M (Multimodal Sentiment Analysis of Hinglish Memes)  
**Dataset:** Memotion 3.0 (AAAI De-Factify 2 Workshop Benchmark)  
**Date of Execution:** 2026-10-31  
**Project Base Path:** `D:\projectsss\NLP project (sentiment analysis)`

---

## 1. Initial Directory Inspection

Prior to restructuring, the dataset directory `dataset/Memotion 3/` had an unstructured and deeply nested layout:

```text
dataset/Memotion 3/
├── format.py                      (Submission formatting script)
├── README.txt                     (AAAI-22 De-Factify 2 Task description)
├── memotion3/
│   └── memotion3/
│       └── train.csv              (Training annotations: 7,000 rows, 8 columns)
├── memotion3-test/
│   └── test.csv                   (Test annotations: 1,500 rows, 3 columns)
├── trainImages/
│   ├── trainImages/               (7,000 JPEG images)
│   └── __MACOSX/                  (7,001 macOS resource fork files)
├── testImages/
│   └── testImages/                (1,500 JPEG images)
└── valImages/
    └── valImages/                 (1,500 JPEG images)
```

### Observations & Issues Identified:
1. **Redundant Nesting:** Directory structures like `trainImages/trainImages/`, `memotion3/memotion3/`, `testImages/testImages/`, and `valImages/valImages/` contained needless folder duplication.
2. **macOS Metadata Pollution:** 7,001 `._*` resource fork files and `__MACOSX` directories were present in `trainImages/`.
3. **Missing Validation Ground Truth:** The official competition validation annotations were historically distributed in an encrypted zip archive (`memotion3-val.zip`), which was not provided. In accordance with guidelines, no attempts were made to guess passwords.

---

## 2. Reorganized Target Structure

In accordance with project guidelines, the dataset was reorganized into a clean, flat, modular structure:

```text
dataset/Memotion 3/
├── annotations/
│   ├── train.csv                  (7,000 rows, 8 columns, 1,939,860 bytes)
│   └── test.csv                   (1,500 rows, 3 columns, 319,675 bytes)
├── images/
│   ├── train/                     (7,000 verified RGB JPEG images: 0.jpg - 6999.jpg)
│   ├── validation/                (1,500 verified RGB JPEG images: 0.jpg - 1499.jpg)
│   └── test/                      (1,500 verified RGB JPEG images: 0.jpg - 1499.jpg)
├── raw/
│   └── README.txt                 (Original challenge documentation)
└── scripts/
    └── format.py                  (Original challenge label formatting script)
```

---

## 3. Operations Performed & Integrity Verification

1. **Non-Destructive Safe Copying:**
   - All destination folders were created first (`annotations/`, `images/train/`, `images/validation/`, `images/test/`, `raw/`, `scripts/`).
   - Files were copied without deleting any original source files.
   - macOS metadata entries (`__MACOSX`, `._*`) were completely filtered out during copying.

2. **File Count & Verification Audit:**

| Artifact Category | Source Path | Target Path | Verified File Count | Integrity Status |
| :--- | :--- | :--- | :---: | :---: |
| **Train Annotations** | `memotion3/memotion3/train.csv` | `annotations/train.csv` | 1 file (7,000 rows) | Verified (100% Match) |
| **Test Annotations** | `memotion3-test/test.csv` | `annotations/test.csv` | 1 file (1,500 rows) | Verified (100% Match) |
| **Train Images** | `trainImages/trainImages/` | `images/train/` | 7,000 `.jpg` files | Verified (100% Readable) |
| **Validation Images** | `valImages/valImages/` | `images/validation/` | 1,500 `.jpg` files | Verified (100% Readable) |
| **Test Images** | `testImages/testImages/` | `images/test/` | 1,500 `.jpg` files | Verified (100% Readable) |
| **Challenge Docs** | `README.txt` | `raw/README.txt` | 1 file | Preserved |
| **Formatting Script**| `format.py` | `scripts/format.py` | 1 file | Preserved |

3. **Data Loss Check:**
   - **Zero files lost or corrupted.**
   - All 10,000 images were opened, verified, converted to RGB, and checked for MD5 collisions.
