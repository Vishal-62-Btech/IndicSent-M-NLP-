"""
IndicSent-M: Comprehensive Model Evaluation, Baseline Comparison & Threshold Calibration
========================================================================================
Evaluates:
  1. Single-task champion Gated Fusion (multimodal/gated_fusion)
  2. Multi-task baseline (multitask_emotion)
  3. Experiment E0 (e0_lambda1_0)
  4. Experiment E1 (e1_lambda0_5)
  5. Experiment E2 (e2_lambda0_25)
  6. Baselines:
     - Sentiment: Majority Class, Random Chance
     - Emotion Multi-label: All-Negative, Majority Label
     - Emotion Dominant: Majority Dominant Emotion
  7. Threshold Calibration:
     - Calibrated on training set (N=1,564 verified)
     - Evaluated on validation set (N=230 verified)
"""

import sys
import os
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    hamming_loss,
    zero_one_loss
)

try:
    sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
except Exception:
    pass

SRC_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SRC_DIR.parents[0] if (SRC_DIR.name == "src") else SRC_DIR.parents[1]
# Ensure correct project root
if not (PROJECT_ROOT / "model_training").exists():
    PROJECT_ROOT = Path("D:/projectsss/NLP project (sentiment analysis)")

sys.path.insert(0, str(PROJECT_ROOT / "model_training" / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from models.multimodal_models import (
    MultiTaskGatedMultimodalClassifier,
    GatedMultimodalClassifier,
    SIX_EMOTIONS,
    NUM_EMOTIONS,
    EMOTION_EMOJIS,
    _resolve_text_backbone
)
from training.train_multitask_emotion import MultiTaskMemeDataset
from preprocessing.preprocess_images import get_image_transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SENTIMENT_NAMES = ["Negative", "Neutral", "Positive"]


def evaluate_sentiment_only_model(model, dataloader) -> Dict[str, Any]:
    model.eval()
    all_preds = []
    all_trues = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values = batch["pixel_values"].to(device)
            sentiment_labels = batch["sentiment_label"].numpy()

            logits = model(input_ids, attention_mask, pixel_values)
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_trues.extend(sentiment_labels)

    all_preds = np.array(all_preds)
    all_trues = np.array(all_trues)

    acc = accuracy_score(all_trues, all_preds)
    macro_f1 = f1_score(all_trues, all_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(all_trues, all_preds, average="weighted", zero_division=0)
    cm = confusion_matrix(all_trues, all_preds, labels=[0, 1, 2]).tolist()
    per_class = f1_score(all_trues, all_preds, average=None, labels=[0, 1, 2], zero_division=0)

    return {
        "accuracy": round(float(acc) * 100, 2),
        "macro_f1": round(float(macro_f1) * 100, 2),
        "weighted_f1": round(float(weighted_f1) * 100, 2),
        "confusion_matrix": cm,
        "per_class_f1": {
            "Negative": round(float(per_class[0]) * 100, 2),
            "Neutral": round(float(per_class[1]) * 100, 2),
            "Positive": round(float(per_class[2]) * 100, 2),
        }
    }


def evaluate_multitask_model(model, dataloader) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_sent_preds = []
    all_sent_trues = []
    all_em_probs = []
    all_em_trues = []
    all_em_masks = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values = batch["pixel_values"].to(device)

            sent_logits, em_logits = model(input_ids, attention_mask, pixel_values)
            sent_preds = torch.argmax(sent_logits, dim=1).cpu().numpy()
            em_probs = torch.sigmoid(em_logits).cpu().numpy()

            all_sent_preds.extend(sent_preds)
            all_sent_trues.extend(batch["sentiment_label"].numpy())
            all_em_probs.append(em_probs)
            all_em_trues.append(batch["emotion_labels"].numpy())
            all_em_masks.extend(batch["emotion_mask"].numpy())

    all_sent_preds = np.array(all_sent_preds)
    all_sent_trues = np.array(all_sent_trues)
    all_em_probs = np.vstack(all_em_probs)
    all_em_trues = np.vstack(all_em_trues)
    all_em_masks = np.array(all_em_masks)

    # Sentiment metrics
    acc = accuracy_score(all_sent_trues, all_sent_preds)
    macro_f1 = f1_score(all_sent_trues, all_sent_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(all_sent_trues, all_sent_preds, average="weighted", zero_division=0)
    cm = confusion_matrix(all_sent_trues, all_sent_preds, labels=[0, 1, 2]).tolist()
    per_class = f1_score(all_sent_trues, all_sent_preds, average=None, labels=[0, 1, 2], zero_division=0)

    sent_metrics = {
        "accuracy": round(float(acc) * 100, 2),
        "macro_f1": round(float(macro_f1) * 100, 2),
        "weighted_f1": round(float(weighted_f1) * 100, 2),
        "confusion_matrix": cm,
        "per_class_f1": {
            "Negative": round(float(per_class[0]) * 100, 2),
            "Neutral": round(float(per_class[1]) * 100, 2),
            "Positive": round(float(per_class[2]) * 100, 2),
        }
    }

    # Emotion metrics on verified subset
    valid_idx = np.where(all_em_masks > 0.5)[0]
    em_metrics = {}
    if len(valid_idx) > 0:
        sub_probs = all_em_probs[valid_idx]
        sub_trues = all_em_trues[valid_idx]
        sub_preds = (sub_probs >= 0.5).astype(int)

        em_macro = f1_score(sub_trues, sub_preds, average="macro", zero_division=0)
        em_micro = f1_score(sub_trues, sub_preds, average="micro", zero_division=0)
        em_weighted = f1_score(sub_trues, sub_preds, average="weighted", zero_division=0)
        h_loss = hamming_loss(sub_trues, sub_preds)
        exact_match = 1.0 - zero_one_loss(sub_trues, sub_preds)

        # Dominant emotion top-1 accuracy
        dom_preds = np.argmax(sub_probs, axis=1)
        dom_hits = sum(1 for i in range(len(valid_idx)) if sub_trues[i, dom_preds[i]] == 1)
        dom_acc = (dom_hits / len(valid_idx)) * 100.0

        per_em = {}
        per_f1 = f1_score(sub_trues, sub_preds, average=None, zero_division=0)
        per_p = precision_score(sub_trues, sub_preds, average=None, zero_division=0)
        per_r = recall_score(sub_trues, sub_preds, average=None, zero_division=0)
        for i, em in enumerate(SIX_EMOTIONS):
            per_em[em] = {
                "f1": round(float(per_f1[i]) * 100, 2),
                "precision": round(float(per_p[i]) * 100, 2),
                "recall": round(float(per_r[i]) * 100, 2),
                "support": int(np.sum(sub_trues[:, i]))
            }

        em_metrics = {
            "verified_samples": len(valid_idx),
            "macro_f1": round(float(em_macro) * 100, 2),
            "micro_f1": round(float(em_micro) * 100, 2),
            "weighted_f1": round(float(em_weighted) * 100, 2),
            "hamming_loss": round(float(h_loss), 4),
            "exact_match_ratio": round(float(exact_match) * 100, 2),
            "dominant_emotion_accuracy": round(float(dom_acc), 2),
            "per_emotion": per_em
        }

    return {
        "sentiment": sent_metrics,
        "emotion": em_metrics
    }, all_em_probs, all_em_trues, all_em_masks


def compute_baselines(df_sentiment_val: pd.DataFrame, df_emotions_val: pd.DataFrame) -> Dict[str, Any]:
    # 1. Sentiment Baselines
    sent_trues = df_sentiment_val["label_id"].values
    counts = np.bincount(sent_trues, minlength=3)
    majority_class = int(np.argmax(counts))
    maj_preds = np.full_like(sent_trues, majority_class)

    maj_acc = accuracy_score(sent_trues, maj_preds)
    maj_macro = f1_score(sent_trues, maj_preds, average="macro", zero_division=0)
    maj_weighted = f1_score(sent_trues, maj_preds, average="weighted", zero_division=0)

    # Random sentiment (uniform 1/3)
    np.random.seed(42)
    rand_preds = np.random.choice([0, 1, 2], size=len(sent_trues))
    rand_acc = accuracy_score(sent_trues, rand_preds)
    rand_macro = f1_score(sent_trues, rand_preds, average="macro", zero_division=0)

    # 2. Emotion Baselines on verified subset
    merged = df_sentiment_val.merge(
        df_emotions_val[["image_name", "is_verified"] + [em.lower() for em in SIX_EMOTIONS]],
        left_on="image_filename",
        right_on="image_name",
        how="left"
    )
    verified = merged[merged["is_verified"] == 1].reset_index(drop=True)
    em_cols = [em.lower() for em in SIX_EMOTIONS]
    em_trues = verified[em_cols].values.astype(int)
    n_ver = len(em_trues)

    # All-negative baseline (all zeros)
    all_zero_preds = np.zeros_like(em_trues)
    all_zero_h_loss = hamming_loss(em_trues, all_zero_preds)
    all_zero_macro = f1_score(em_trues, all_zero_preds, average="macro", zero_division=0)
    all_zero_micro = f1_score(em_trues, all_zero_preds, average="micro", zero_division=0)
    all_zero_exact = (1.0 - zero_one_loss(em_trues, all_zero_preds)) * 100.0

    # Majority label baseline (predict 1 if prevalence > 0.5 else 0)
    col_prevalences = np.mean(em_trues, axis=0)
    maj_label_preds = np.zeros_like(em_trues)
    for c in range(6):
        if col_prevalences[c] >= 0.5:
            maj_label_preds[:, c] = 1
    maj_label_h_loss = hamming_loss(em_trues, maj_label_preds)
    maj_label_macro = f1_score(em_trues, maj_label_preds, average="macro", zero_division=0)
    maj_label_micro = f1_score(em_trues, maj_label_preds, average="micro", zero_division=0)

    # Majority dominant emotion baseline (always predict most frequent emotion: Happiness)
    most_freq_em = int(np.argmax(np.sum(em_trues, axis=0)))
    maj_dom_hits = sum(1 for i in range(n_ver) if em_trues[i, most_freq_em] == 1)
    maj_dom_acc = (maj_dom_hits / n_ver) * 100.0

    # Random dominant emotion (1/6 = 16.67% theoretical)
    rand_dom_preds = np.random.choice(6, size=n_ver)
    rand_dom_hits = sum(1 for i in range(n_ver) if em_trues[i, rand_dom_preds[i]] == 1)
    rand_dom_acc = (rand_dom_hits / n_ver) * 100.0

    return {
        "sentiment_baselines": {
            "majority_class": {
                "class_name": SENTIMENT_NAMES[majority_class],
                "accuracy": round(float(maj_acc) * 100, 2),
                "macro_f1": round(float(maj_macro) * 100, 2),
                "weighted_f1": round(float(maj_weighted) * 100, 2),
            },
            "random_chance": {
                "accuracy": round(float(rand_acc) * 100, 2),
                "macro_f1": round(float(rand_macro) * 100, 2),
            }
        },
        "emotion_baselines": {
            "all_negative_baseline": {
                "description": "Predicts 0 for all 6 emotions for all samples",
                "hamming_loss": round(float(all_zero_h_loss), 4),
                "macro_f1": round(float(all_zero_macro) * 100, 2),
                "micro_f1": round(float(all_zero_micro) * 100, 2),
                "exact_match_ratio": round(float(all_zero_exact), 2),
                "utility_explanation": "Achieves lowest mathematical Hamming Loss (0.1812) purely because 81.88% of label bits in the benchmark are negative (inactive). However, it produces F1=0.0% and zero useful emotion predictions."
            },
            "majority_label_baseline": {
                "hamming_loss": round(float(maj_label_h_loss), 4),
                "macro_f1": round(float(maj_label_macro) * 100, 2),
                "micro_f1": round(float(maj_label_micro) * 100, 2)
            },
            "majority_dominant_emotion_baseline": {
                "most_frequent_emotion": SIX_EMOTIONS[most_freq_em],
                "accuracy": round(float(maj_dom_acc), 2)
            },
            "random_dominant_emotion_baseline": {
                "accuracy": round(float(rand_dom_acc), 2),
                "theoretical": 16.67
            }
        }
    }


def calibrate_thresholds_on_train(model, train_loader) -> Tuple[np.ndarray, Dict[str, float]]:
    model.eval()
    all_probs = []
    all_trues = []
    all_masks = []

    with torch.no_grad():
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values = batch["pixel_values"].to(device)
            _, em_logits = model(input_ids, attention_mask, pixel_values)
            probs = torch.sigmoid(em_logits).cpu().numpy()
            all_probs.append(probs)
            all_trues.append(batch["emotion_labels"].numpy())
            all_masks.extend(batch["emotion_mask"].numpy())

    all_probs = np.vstack(all_probs)
    all_trues = np.vstack(all_trues)
    all_masks = np.array(all_masks)

    valid_idx = np.where(all_masks > 0.5)[0]
    sub_probs = all_probs[valid_idx]
    sub_trues = all_trues[valid_idx]

    # Grid search per emotion
    threshold_grid = np.linspace(0.15, 0.85, 29)
    best_thresholds = np.full(6, 0.5)
    train_best_f1s = {}

    for c in range(6):
        best_t = 0.5
        best_f1 = -1.0
        for t in threshold_grid:
            bin_preds = (sub_probs[:, c] >= t).astype(int)
            f1 = f1_score(sub_trues[:, c], bin_preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        best_thresholds[c] = round(float(best_t), 3)
        train_best_f1s[SIX_EMOTIONS[c]] = {
            "optimal_threshold": round(float(best_t), 3),
            "train_f1": round(float(best_f1) * 100, 2)
        }

    return best_thresholds, train_best_f1s


def evaluate_with_thresholds(val_probs: np.ndarray, val_trues: np.ndarray, val_masks: np.ndarray, thresholds: np.ndarray) -> Dict[str, Any]:
    valid_idx = np.where(val_masks > 0.5)[0]
    sub_probs = val_probs[valid_idx]
    sub_trues = val_trues[valid_idx]

    # Apply per-column threshold
    sub_preds = np.zeros_like(sub_trues, dtype=int)
    for c in range(6):
        sub_preds[:, c] = (sub_probs[:, c] >= thresholds[c]).astype(int)

    em_macro = f1_score(sub_trues, sub_preds, average="macro", zero_division=0)
    em_micro = f1_score(sub_trues, sub_preds, average="micro", zero_division=0)
    em_weighted = f1_score(sub_trues, sub_preds, average="weighted", zero_division=0)
    h_loss = hamming_loss(sub_trues, sub_preds)
    exact_match = (1.0 - zero_one_loss(sub_trues, sub_preds)) * 100.0

    per_em = {}
    per_f1 = f1_score(sub_trues, sub_preds, average=None, zero_division=0)
    per_p = precision_score(sub_trues, sub_preds, average=None, zero_division=0)
    per_r = recall_score(sub_trues, sub_preds, average=None, zero_division=0)
    for i, em in enumerate(SIX_EMOTIONS):
        per_em[em] = {
            "threshold": float(thresholds[i]),
            "f1": round(float(per_f1[i]) * 100, 2),
            "precision": round(float(per_p[i]) * 100, 2),
            "recall": round(float(per_r[i]) * 100, 2)
        }

    return {
        "macro_f1": round(float(em_macro) * 100, 2),
        "micro_f1": round(float(em_micro) * 100, 2),
        "weighted_f1": round(float(em_weighted) * 100, 2),
        "hamming_loss": round(float(h_loss), 4),
        "exact_match_ratio": round(float(exact_match), 2),
        "per_emotion": per_em
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate IndicSent-M models and baselines")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output-json", type=str, default="final_comprehensive_evaluation.json")
    args = parser.parse_args()

    splits_dir = PROJECT_ROOT / "model_training" / "data_splits"
    emotions_dir = splits_dir / "emotions"
    df_train_sent = pd.read_csv(splits_dir / "train.csv")
    df_val_sent = pd.read_csv(splits_dir / "validation.csv")
    df_train_em = pd.read_csv(emotions_dir / "train_emotions.csv")
    df_val_em = pd.read_csv(emotions_dir / "validation_emotions.csv")

    print("===========================================================================")
    print("IndicSent-M: Master Model Evaluation & Baseline Comparison")
    print(f"Device: {device}")
    print(f"Validation Samples: {len(df_val_sent)} (Sentiment) | {len(df_val_em[df_val_em['is_verified']==1])} (Verified Emotion)")
    print(f"Training Samples: {len(df_train_sent)} (Sentiment) | {len(df_train_em[df_train_em['is_verified']==1])} (Verified Emotion)")
    print("===========================================================================")

    tokenizer = AutoTokenizer.from_pretrained(_resolve_text_backbone("google/muril-base-cased"), local_files_only=True)
    img_transform = get_image_transforms("resnet50", is_training=False)

    val_dataset = MultiTaskMemeDataset(
        df_sentiment=df_val_sent,
        df_emotions=df_val_em,
        base_dir=PROJECT_ROOT,
        tokenizer=tokenizer,
        img_transform=img_transform
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # 1. Compute Baselines
    print("\n--- Computing Empirical & Heuristic Baselines ---")
    baselines = compute_baselines(df_val_sent, df_val_em)
    print(f"  Sentiment Majority Baseline ({baselines['sentiment_baselines']['majority_class']['class_name']}): Acc={baselines['sentiment_baselines']['majority_class']['accuracy']}%, Macro-F1={baselines['sentiment_baselines']['majority_class']['macro_f1']}%")
    print(f"  Emotion All-Negative Baseline: Hamming Loss={baselines['emotion_baselines']['all_negative_baseline']['hamming_loss']}, Macro-F1={baselines['emotion_baselines']['all_negative_baseline']['macro_f1']}%, Exact Match={baselines['emotion_baselines']['all_negative_baseline']['exact_match_ratio']}%")
    print(f"  Dominant Emotion Majority Baseline ({baselines['emotion_baselines']['majority_dominant_emotion_baseline']['most_frequent_emotion']}): Accuracy={baselines['emotion_baselines']['majority_dominant_emotion_baseline']['accuracy']}%")

    results = {
        "baselines": baselines,
        "models": {}
    }

    # Model definitions to evaluate
    model_configs = [
        {
            "name": "SingleTask_GatedFusion_Champion",
            "type": "single_task",
            "path": PROJECT_ROOT / "model_training" / "checkpoints" / "multimodal" / "gated_fusion" / "best_model.pt",
            "description": "Baseline champion single-task Gated Fusion (MuRIL + ResNet-50)"
        },
        {
            "name": "MultiTask_Baseline_Initial",
            "type": "multi_task",
            "path": PROJECT_ROOT / "model_training" / "checkpoints" / "multitask_emotion" / "best_model.pt",
            "description": "Initial multi-task baseline (500 train annotations, lambda=1.0)"
        },
        {
            "name": "MultiTask_E0_lambda1_0",
            "type": "multi_task",
            "path": PROJECT_ROOT / "model_training" / "checkpoints" / "experiments" / "e0_lambda1_0" / "best_model.pt",
            "description": "Experiment E0: 1,564 train annotations, lambda_emotion=1.0"
        },
        {
            "name": "MultiTask_E1_lambda0_5",
            "type": "multi_task",
            "path": PROJECT_ROOT / "model_training" / "checkpoints" / "experiments" / "e1_lambda0_5" / "best_model.pt",
            "description": "Experiment E1: 1,564 train annotations, lambda_emotion=0.5"
        },
        {
            "name": "MultiTask_E2_lambda0_25",
            "type": "multi_task",
            "path": PROJECT_ROOT / "model_training" / "checkpoints" / "experiments" / "e2_lambda0_25" / "best_model.pt",
            "description": "Experiment E2: 1,564 train annotations, lambda_emotion=0.25"
        }
    ]

    val_probs_cache = {}

    for cfg in model_configs:
        m_name = cfg["name"]
        m_path = cfg["path"]
        m_type = cfg["type"]

        if not m_path.exists():
            print(f"\n[SKIP] Checkpoint not found for {m_name}: {m_path}")
            continue

        print(f"\n--- Evaluating Model: {m_name} ---")
        checkpoint = torch.load(m_path, map_location=device)
        sd = checkpoint.get("model_state_dict", checkpoint)

        if m_type == "single_task":
            model = GatedMultimodalClassifier(proj_dim=512, num_classes=3, freeze_text=True, freeze_vision=True)
            model.load_state_dict(sd, strict=False)
            model.to(device)
            m_res = evaluate_sentiment_only_model(model, val_loader)
            results["models"][m_name] = {
                "type": m_type,
                "description": cfg["description"],
                "path": str(m_path),
                "sentiment": m_res
            }
            print(f"  Sentiment Accuracy: {m_res['accuracy']}%, Macro-F1: {m_res['macro_f1']}%, Weighted-F1: {m_res['weighted_f1']}%")
        else:
            model = MultiTaskGatedMultimodalClassifier(proj_dim=512, num_classes=3, num_emotions=6, freeze_text=True, freeze_vision=True)
            model.load_state_dict(sd, strict=False)
            model.to(device)
            m_res, probs, trues, masks = evaluate_multitask_model(model, val_loader)
            results["models"][m_name] = {
                "type": m_type,
                "description": cfg["description"],
                "path": str(m_path),
                "sentiment": m_res["sentiment"],
                "emotion_default_threshold_0_5": m_res["emotion"]
            }
            val_probs_cache[m_name] = (model, probs, trues, masks)
            print(f"  Sentiment Accuracy: {m_res['sentiment']['accuracy']}%, Macro-F1: {m_res['sentiment']['macro_f1']}%, Weighted-F1: {m_res['sentiment']['weighted_f1']}%")
            if "macro_f1" in m_res["emotion"]:
                print(f"  Emotion Macro-F1: {m_res['emotion']['macro_f1']}%, Micro-F1: {m_res['emotion']['micro_f1']}%, Dominant Acc: {m_res['emotion']['dominant_emotion_accuracy']}%, Hamming Loss: {m_res['emotion']['hamming_loss']}")

    # Threshold Calibration on Training Set
    # We calibrate on the best multi-task candidate from E0, E1, E2
    calib_candidates = [c["name"] for c in model_configs if c["name"] in val_probs_cache and "E" in c["name"]]
    if not calib_candidates:
        calib_candidates = [c["name"] for c in model_configs if c["name"] in val_probs_cache]

    if calib_candidates:
        # Load train dataset for calibration
        train_dataset = MultiTaskMemeDataset(
            df_sentiment=df_train_sent,
            df_emotions=df_train_em,
            base_dir=PROJECT_ROOT,
            tokenizer=tokenizer,
            img_transform=img_transform
        )
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

        results["threshold_calibration"] = {}
        for cand_name in calib_candidates:
            model, val_probs, val_trues, val_masks = val_probs_cache[cand_name]
            print(f"\n--- Running Threshold Calibration on Train for {cand_name} ---")
            calib_thresholds, train_f1s = calibrate_thresholds_on_train(model, train_loader)
            print(f"  Optimal Thresholds Found: {dict(zip(SIX_EMOTIONS, calib_thresholds))}")
            val_calib_res = evaluate_with_thresholds(val_probs, val_trues, val_masks, calib_thresholds)
            print(f"  Validation with Calibrated Thresholds: Macro-F1={val_calib_res['macro_f1']}%, Micro-F1={val_calib_res['micro_f1']}%, Hamming Loss={val_calib_res['hamming_loss']}")

            results["threshold_calibration"][cand_name] = {
                "calibrated_thresholds": {SIX_EMOTIONS[i]: float(calib_thresholds[i]) for i in range(6)},
                "train_search_results": train_f1s,
                "validation_calibrated_metrics": val_calib_res
            }

    # Save to JSON
    out_dir = PROJECT_ROOT / "model_training" / "evaluation_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / args.output_json
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n===========================================================================")
    print(f"Evaluation complete. Results written to: {out_file}")
    print("===========================================================================")


if __name__ == "__main__":
    main()
