# IndicSent-M: Model Training, Benchmarking & Preprocessing Pipeline

This directory contains the complete modular Machine Learning and Deep Learning pipeline for **IndicSent-M** (*Multimodal Sentiment Analysis of Hinglish Memes* using Memotion 3.0).

---

## 📁 Directory Structure

```text
model_training/
│
├── data_splits/
│   ├── train.csv                # 80% Stratified Training Data (5,600 samples)
│   ├── validation.csv           # 20% Held-Out Validation Data (1,400 samples)
│   ├── test_unlabeled.csv       # Official Unlabeled Test Data (1,500 samples)
│   ├── split_indices.json       # Exact integer indices for reproducible evaluation
│   └── label_mapping.json       # 3-class mapping dictionary & integer IDs
│
├── notebooks/                   # Jupyter notebooks for visual experimentation
│
├── src/
│   ├── preprocessing/
│   │   ├── clean_text.py        # Linguistically sensitive Hinglish/Hindi text cleaner
│   │   ├── preprocess_images.py # PyTorch transforms & PIL verification utilities
│   │   └── prepare_dataset.py   # Dataset ingestion & stratified 80:20 splitting script
│   │
│   ├── models/
│   │   ├── classical_models.py  # TF-IDF + Logistic Regression, SVM, Naive Bayes, RF
│   │   ├── text_models.py       # Fine-tunable MuRIL and XLM-RoBERTa neural heads
│   │   ├── image_models.py      # ResNet-50, EfficientNet-B0, and CLIP ViT classifiers
│   │   └── multimodal_models.py # Feature Concatenation & Cross-Modal Gated Attention
│   │
│   ├── training/
│   │   ├── train_classical.py   # Execution script for classical ML baselines
│   │   ├── train_text.py        # Fine-tuning script for MuRIL & XLM-RoBERTa
│   │   ├── train_image.py       # Transfer learning script for ResNet-50 & ViTs
│   │   └── train_multimodal.py  # Multimodal fusion training pipeline
│   │
│   ├── evaluation/
│   │   ├── evaluate.py          # Unified metrics (Macro-F1, Confusion Matrix, latency)
│   │   └── compare_models.py    # Leaderboard generator & markdown report exporter
│   │
│   └── inference.py             # Standalone production inference engine for FastAPI
│
├── checkpoints/                 # Saved model weights & joblib pipelines
│   ├── classical/               # 5 trained baseline model checkpoints
│   ├── text/                    # Fine-tuned transformer weights
│   ├── image/                   # Vision backbone checkpoints
│   ├── multimodal/              # Multimodal fusion checkpoints
│   └── best_model/              # Selected champion model exported for production
│
├── results/
│   ├── dataset_organization_report.md  # Safe directory restructuring audit
│   ├── dataset_analysis.md             # Comprehensive EDA of Memotion 3.0
│   ├── model_comparison.csv            # Master numerical benchmarking results
│   └── final_model_analysis.md         # Full research comparison & selection rationale
│
└── requirements.txt             # Python dependencies
```

---

## 🚀 Execution Guide

### 1. Data Preprocessing & Splitting (Completed)
```bash
python src/preprocessing/prepare_dataset.py
```

### 2. Classical ML Baselines (Completed)
```bash
python src/training/train_classical.py
```

### 3. Deep Text Models (Phase 5)
```bash
python src/training/train_text.py --model muril --epochs 3 --batch_size 32
python src/training/train_text.py --model xlm_roberta --epochs 3 --batch_size 32
```

### 4. Vision Baselines (Phase 6)
```bash
python src/training/train_image.py --model resnet50 --epochs 5 --batch_size 32
python src/training/train_image.py --model clip --epochs 5 --batch_size 32
```

### 5. Multimodal Fusion Models (Phase 7)
```bash
python src/training/train_multimodal.py --exp 1 --epochs 4 --batch_size 16
python src/training/train_multimodal.py --exp 2 --epochs 4 --batch_size 16
python src/training/train_multimodal.py --exp 3 --epochs 4 --batch_size 16
```

### 6. Leaderboard & Report Generation
```bash
python src/evaluation/compare_models.py
```
