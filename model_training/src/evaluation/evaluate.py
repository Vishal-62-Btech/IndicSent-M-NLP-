"""
IndicSent-M: Unified Evaluation Metric Module
Computes Accuracy, Macro-F1 (primary metric), Weighted-F1,
Per-class Precision, Recall, and F1, and Confusion Matrices.
"""

from typing import Dict, Any, List, Optional
import time
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

TARGET_NAMES = ["Negative", "Neutral", "Positive"]
CLASS_INDICES = [0, 1, 2]

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    latency_ms: Optional[float] = None,
    train_time_sec: Optional[float] = None
) -> Dict[str, Any]:
    """
    Computes full comprehensive evaluation metrics for 3-class sentiment prediction.
    """
    acc = accuracy_score(y_true, y_pred)
    macro_p = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_r = recall_score(y_true, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    weighted_p = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    weighted_r = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    # Per-class metrics
    per_class_p = precision_score(y_true, y_pred, average=None, labels=CLASS_INDICES, zero_division=0)
    per_class_r = recall_score(y_true, y_pred, average=None, labels=CLASS_INDICES, zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, average=None, labels=CLASS_INDICES, zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=CLASS_INDICES)

    results = {
        "model_name": model_name,
        "accuracy": round(float(acc) * 100, 2),
        "macro_f1": round(float(macro_f1) * 100, 2),
        "macro_precision": round(float(macro_p) * 100, 2),
        "macro_recall": round(float(macro_r) * 100, 2),
        "weighted_f1": round(float(weighted_f1) * 100, 2),
        "negative_f1": round(float(per_class_f1[0]) * 100, 2),
        "negative_recall": round(float(per_class_r[0]) * 100, 2),
        "neutral_f1": round(float(per_class_f1[1]) * 100, 2),
        "neutral_recall": round(float(per_class_r[1]) * 100, 2),
        "positive_f1": round(float(per_class_f1[2]) * 100, 2),
        "positive_recall": round(float(per_class_r[2]) * 100, 2),
        "train_time_sec": round(float(train_time_sec), 2) if train_time_sec is not None else None,
        "latency_ms_per_sample": round(float(latency_ms), 3) if latency_ms is not None else None,
        "confusion_matrix": cm.tolist()
    }

    return results

def print_evaluation_summary(metrics: Dict[str, Any]):
    print("\n" + "=" * 55)
    print(f"EVALUATION RESULTS: {metrics['model_name']}")
    print("=" * 55)
    print(f"Accuracy:        {metrics['accuracy']}%")
    print(f"Macro-F1 (Key):  {metrics['macro_f1']}%")
    print(f"Macro-Precision: {metrics['macro_precision']}%")
    print(f"Macro-Recall:    {metrics['macro_recall']}%")
    print(f"Weighted-F1:     {metrics['weighted_f1']}%")
    print("-" * 55)
    print("Per-Class Performance:")
    print(f"  Negative [0] -> F1: {metrics['negative_f1']}%, Recall: {metrics['negative_recall']}%")
    print(f"  Neutral  [1] -> F1: {metrics['neutral_f1']}%, Recall: {metrics['neutral_recall']}%")
    print(f"  Positive [2] -> F1: {metrics['positive_f1']}%, Recall: {metrics['positive_recall']}%")
    print("-" * 55)
    print("Confusion Matrix (Rows: True [Neg, Neu, Pos], Cols: Pred [Neg, Neu, Pos]):")
    for row in metrics["confusion_matrix"]:
        print(f"  {row}")
    if metrics.get("train_time_sec") is not None:
        print(f"Training Time:   {metrics['train_time_sec']}s")
    if metrics.get("latency_ms_per_sample") is not None:
        print(f"Inference Speed: {metrics['latency_ms_per_sample']} ms / sample")
    print("=" * 55 + "\n")
