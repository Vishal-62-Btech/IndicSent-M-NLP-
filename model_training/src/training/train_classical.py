"""
IndicSent-M: Classical ML Training Pipeline
Trains all 5 classical text-based baselines strictly on the 80% training split,
evaluates on the 20% validation split, saves checkpoints, and logs metrics.
"""

import sys
import time
from pathlib import Path
import pandas as pd
import joblib

# Ensure local imports work
project_root = Path(__file__).resolve().parents[3]
src_dir = Path(__file__).resolve().parents[1]
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from models.classical_models import build_classical_pipelines
from evaluation.evaluate import compute_metrics, print_evaluation_summary

def run_classical_experiments():
    print("=" * 60)
    print("IndicSent-M: Training Classical ML Baselines")
    print("=" * 60)

    splits_dir = project_root / "model_training" / "data_splits"
    checkpoints_dir = project_root / "model_training" / "checkpoints" / "classical"
    results_dir = project_root / "model_training" / "results"
    
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    train_path = splits_dir / "train.csv"
    val_path = splits_dir / "validation.csv"

    assert train_path.exists(), f"Train split missing at {train_path}"
    assert val_path.exists(), f"Val split missing at {val_path}"

    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)

    print(f"Loaded Train: {len(df_train)} samples, Val: {len(df_val)} samples.")

    X_train = df_train["clean_text"].fillna("[NO_TEXT]").astype(str)
    y_train = df_train["label_id"].values

    X_val = df_val["clean_text"].fillna("[NO_TEXT]").astype(str)
    y_val = df_val["label_id"].values

    pipelines = build_classical_pipelines(random_state=42)
    comparison_records = []

    for name, pipe in pipelines.items():
        print(f"\nTraining [{name}]...")
        t0 = time.time()
        pipe.fit(X_train, y_train)
        train_duration = time.time() - t0

        # Inference timing on validation split
        t1 = time.time()
        y_pred = pipe.predict(X_val)
        val_duration = time.time() - t1
        latency_ms = (val_duration / len(X_val)) * 1000.0

        # Metrics
        metrics = compute_metrics(
            y_true=y_val,
            y_pred=y_pred,
            model_name=name,
            latency_ms=latency_ms,
            train_time_sec=train_duration
        )

        print_evaluation_summary(metrics)

        # Save checkpoint
        safe_name = name.lower().replace(" ", "_").replace("+", "plus").replace("-", "_")
        ckpt_path = checkpoints_dir / f"{safe_name}.joblib"
        joblib.dump(pipe, ckpt_path)
        print(f"Checkpoint saved: {ckpt_path.name}")

        # Record for comparison
        rec = {
            "Model": name,
            "Modality": "Text Only (Hinglish OCR)",
            "Accuracy (%)": metrics["accuracy"],
            "Macro F1 (%)": metrics["macro_f1"],
            "Macro Precision (%)": metrics["macro_precision"],
            "Macro Recall (%)": metrics["macro_recall"],
            "Weighted F1 (%)": metrics["weighted_f1"],
            "Negative F1 (%)": metrics["negative_f1"],
            "Neutral F1 (%)": metrics["neutral_f1"],
            "Positive F1 (%)": metrics["positive_f1"],
            "Negative Recall (%)": metrics["negative_recall"],
            "Neutral Recall (%)": metrics["neutral_recall"],
            "Positive Recall (%)": metrics["positive_recall"],
            "Train Time (s)": metrics["train_time_sec"],
            "Latency (ms/sample)": metrics["latency_ms_per_sample"]
        }
        comparison_records.append(rec)

    # Save comparison CSV
    comparison_df = pd.DataFrame(comparison_records)
    csv_out = results_dir / "model_comparison.csv"
    
    if csv_out.exists():
        # Append / update
        existing_df = pd.read_csv(csv_out)
        # remove rows with same model names if re-running
        existing_df = existing_df[~existing_df["Model"].isin(comparison_df["Model"])]
        combined_df = pd.concat([existing_df, comparison_df], ignore_index=True)
        combined_df.to_csv(csv_out, index=False)
    else:
        comparison_df.to_csv(csv_out, index=False)

    print(f"\nAll classical baselines evaluated and saved to: {csv_out}")
    print("\n" + "=" * 60)
    print("SUMMARY COMPARISON TABLE:")
    print("=" * 60)
    print(comparison_df[["Model", "Accuracy (%)", "Macro F1 (%)", "Negative F1 (%)", "Neutral F1 (%)", "Positive F1 (%)"]].to_string(index=False))

if __name__ == "__main__":
    run_classical_experiments()
