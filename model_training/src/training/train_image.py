"""
IndicSent-M: Deep Image Models Training Pipeline
Trains and evaluates:
1. ResNet-50 (Pretrained ImageNet-1k)
2. EfficientNet-B0 (Pretrained ImageNet-1k)
3. MobileNetV3-Large (Pretrained ImageNet-1k)

Rigorous experimental standards:
- Training split: Mild augmentation (horizontal flip, color jitter)
- Validation split: Strictly deterministic resize and normalization
- Class-weighted CrossEntropyLoss
- FP16 Automatic Mixed Precision (AMP)
- Evaluation after each epoch (Accuracy, Macro-F1, Per-class metrics, Confusion Matrix)
- Export of weights, config, label mapping, and reports
"""

import sys
import os
import json
import time
import argparse
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Fix Windows console encoding for Indic characters
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Ensure local imports work
project_root = Path(__file__).resolve().parents[3]
src_dir = Path(__file__).resolve().parents[1]
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from models.image_models import ResNet50Classifier, EfficientNetB0Classifier, MobileNetV3LargeClassifier
from preprocessing.preprocess_images import get_image_transforms, load_and_verify_image
from evaluation.evaluate import compute_metrics, print_evaluation_summary

LABEL_MAP = {0: "Negative", 1: "Neutral", 2: "Positive"}
ID_MAP = {"Negative": 0, "Neutral": 1, "Positive": 2}

class VisionMemeDataset(Dataset):
    def __init__(self, df: pd.DataFrame, base_dir: Path, transform):
        self.df = df.reset_index(drop=True)
        self.base_dir = base_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.base_dir / row["image_rel_path"]
        
        img = load_and_verify_image(img_path)
        if img is None:
            # Fallback neutral grey RGB canvas if read error
            img = Image.new("RGB", (224, 224), (128, 128, 128))
            
        tensor_img = self.transform(img)
        label = torch.tensor(row["label_id"], dtype=torch.long)
        return {
            "pixel_values": tensor_img,
            "label": label
        }

def train_image_model(
    model_key: str = "resnet50",
    epochs: int = 4,
    batch_size: int = 16,
    lr: float = 1e-4,
    patience: int = 2
):
    key = model_key.lower().replace("-", "_")
    name_map = {
        "resnet50": ("ResNet-50", "resnet50", ResNet50Classifier),
        "efficientnet_b0": ("EfficientNet-B0", "efficientnet_b0", EfficientNetB0Classifier),
        "efficientnet": ("EfficientNet-B0", "efficientnet_b0", EfficientNetB0Classifier),
        "mobilenet_v3": ("MobileNetV3-Large", "mobilenet_v3", MobileNetV3LargeClassifier),
        "mobilenet": ("MobileNetV3-Large", "mobilenet_v3", MobileNetV3LargeClassifier),
    }

    if key not in name_map:
        raise ValueError(f"Unknown image model key: {model_key}. Available: {list(name_map.keys())}")

    display_name, folder_name, model_class = name_map[key]

    print("=" * 65)
    print(f"IndicSent-M: Phase 6 - Training Vision Model [{display_name}]")
    print("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Compute Device: {device} ({gpu_name})")

    splits_dir = project_root / "model_training" / "data_splits"
    checkpoint_dir = project_root / "model_training" / "checkpoints" / folder_name
    results_dir = project_root / "model_training" / "results"
    
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    df_train = pd.read_csv(splits_dir / "train.csv")
    df_val = pd.read_csv(splits_dir / "validation.csv")
    print(f"Train samples: {len(df_train)}, Validation samples: {len(df_val)}")

    t_train = get_image_transforms(key, is_training=True)
    t_val = get_image_transforms(key, is_training=False)

    train_dataset = VisionMemeDataset(df_train, project_root, t_train)
    val_dataset = VisionMemeDataset(df_val, project_root, t_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=torch.cuda.is_available(), num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size * 2, shuffle=False, num_workers=0)

    # 2. Class weights for loss
    class_counts = df_train["label_id"].value_counts().sort_index().values
    weights = len(df_train) / (3.0 * class_counts)
    class_weights = torch.tensor(weights, dtype=torch.float).to(device)
    print(f"Class distribution [Neg, Neu, Pos]: {class_counts.tolist()}")
    print(f"Inverse class weights: {class_weights.cpu().numpy().round(3).tolist()}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # 3. Model setup
    model = model_class(num_classes=3).to(device)

    # Differential learning rate: lower for backbone, higher for classification head
    head_params = list(model.classifier.parameters())
    backbone_params = [p for n, p in model.named_parameters() if not n.startswith("classifier")]
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": lr * 0.1},
        {"params": head_params, "lr": lr}
    ], weight_decay=0.01)

    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

    training_config = {
        "model_name": display_name,
        "folder_name": folder_name,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate_head": lr,
        "learning_rate_backbone": lr * 0.1,
        "weight_decay": 0.01,
        "patience": patience,
        "fp16": torch.cuda.is_available(),
        "device": gpu_name
    }

    best_macro_f1 = -1.0
    best_metrics = None
    best_epoch = 0
    training_history = []
    epochs_no_improve = 0
    t_start = time.time()

    print(f"\nCommencing image training ({epochs} epochs, batch size={batch_size})...")

    for epoch in range(1, epochs + 1):
        epoch_t0 = time.time()
        model.train()
        train_loss = 0.0

        for step, batch in enumerate(train_loader):
            pixels = batch["pixel_values"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                logits = model(pixels)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

            if (step + 1) % 100 == 0 or (step + 1) == len(train_loader):
                cur_loss = train_loss / (step + 1)
                print(f"  Epoch [{epoch}/{epochs}] Step [{step + 1}/{len(train_loader)}] | Running Loss: {cur_loss:.4f}", flush=True)

        avg_train_loss = train_loss / len(train_loader)
        epoch_train_time = time.time() - epoch_t0

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_trues = []
        val_t0 = time.time()

        with torch.no_grad():
            for batch in val_loader:
                pixels = batch["pixel_values"].to(device)
                labels = batch["label"].to(device)

                with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                    logits = model(pixels)
                    loss = criterion(logits, labels)

                val_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_trues.extend(labels.cpu().numpy())

        val_duration = time.time() - val_t0
        avg_val_loss = val_loss / len(val_loader)
        latency_ms = (val_duration / len(val_dataset)) * 1000.0

        current_metrics = compute_metrics(
            y_true=np.array(val_trues),
            y_pred=np.array(val_preds),
            model_name=display_name,
            latency_ms=latency_ms,
            train_time_sec=time.time() - t_start
        )

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": round(avg_val_loss, 4),
            "val_accuracy": current_metrics["accuracy"],
            "val_macro_f1": current_metrics["macro_f1"],
            "val_macro_precision": current_metrics["macro_precision"],
            "val_macro_recall": current_metrics["macro_recall"],
            "val_weighted_f1": current_metrics["weighted_f1"],
            "val_negative_f1": current_metrics["negative_f1"],
            "val_neutral_f1": current_metrics["neutral_f1"],
            "val_positive_f1": current_metrics["positive_f1"],
            "epoch_train_time_sec": round(epoch_train_time, 2)
        }
        training_history.append(epoch_record)

        print(f"\n--- Epoch {epoch}/{epochs} Summary ---")
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"Val Accuracy: {current_metrics['accuracy']}% | Val Macro-F1: {current_metrics['macro_f1']}%")
        print(f"Per-Class F1 -> Neg: {current_metrics['negative_f1']}%, Neu: {current_metrics['neutral_f1']}%, Pos: {current_metrics['positive_f1']}%")
        print(f"Epoch Time: {epoch_train_time:.1f}s | Val Latency: {latency_ms:.2f} ms/sample")

        if current_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = current_metrics["macro_f1"]
            best_metrics = current_metrics
            best_epoch = epoch
            epochs_no_improve = 0

            print(f"--> NEW BEST MODEL ACHIEVED! Saving checkpoint to {checkpoint_dir}...")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "macro_f1": best_macro_f1,
                "model_key": folder_name,
                "display_name": display_name
            }, checkpoint_dir / "best_model.pt")

            with open(checkpoint_dir / "label_mapping.json", "w", encoding="utf-8") as f:
                json.dump({"id2label": LABEL_MAP, "label2id": ID_MAP}, f, indent=2)

            with open(checkpoint_dir / "training_config.json", "w", encoding="utf-8") as f:
                json.dump(training_config, f, indent=2)
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epoch(s). Best Macro-F1: {best_macro_f1}%")
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

    # Save training history
    with open(checkpoint_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(training_history, f, indent=2)

    total_training_time = time.time() - t_start

    print("\n" + "=" * 65)
    print(f"{display_name.upper()} TRAINING COMPLETE")
    print("=" * 65)
    print(f"Best Validation Epoch: {best_epoch} (Macro-F1: {best_macro_f1}%)")
    print(f"Total Training Duration: {total_training_time:.1f}s ({total_training_time/60:.2f} mins)")
    print_evaluation_summary(best_metrics)

    # 4. Save results to <folder_name>_results.json
    results_json_path = results_dir / f"{folder_name}_results.json"
    full_results = {
        "model_name": display_name,
        "folder_name": folder_name,
        "best_epoch": best_epoch,
        "total_training_time_sec": round(total_training_time, 2),
        "best_metrics": best_metrics,
        "training_history": training_history,
        "training_config": training_config
    }
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2)
    print(f"Saved evaluation results to {results_json_path}")

    # 5. Update model_comparison.csv
    csv_out = results_dir / "model_comparison.csv"
    rec = {
        "Model": display_name,
        "Modality": "Image Only (Vision)",
        "Accuracy (%)": best_metrics["accuracy"],
        "Macro F1 (%)": best_metrics["macro_f1"],
        "Macro Precision (%)": best_metrics["macro_precision"],
        "Macro Recall (%)": best_metrics["macro_recall"],
        "Weighted F1 (%)": best_metrics["weighted_f1"],
        "Negative F1 (%)": best_metrics["negative_f1"],
        "Neutral F1 (%)": best_metrics["neutral_f1"],
        "Positive F1 (%)": best_metrics["positive_f1"],
        "Negative Recall (%)": best_metrics["negative_recall"],
        "Neutral Recall (%)": best_metrics["neutral_recall"],
        "Positive Recall (%)": best_metrics["positive_recall"],
        "Train Time (s)": round(total_training_time, 2),
        "Latency (ms/sample)": best_metrics["latency_ms_per_sample"]
    }

    rec_df = pd.DataFrame([rec])
    if csv_out.exists():
        existing_df = pd.read_csv(csv_out)
        existing_df = existing_df[existing_df["Model"] != display_name]
        combined_df = pd.concat([existing_df, rec_df], ignore_index=True)
        combined_df.to_csv(csv_out, index=False)
    else:
        rec_df.to_csv(csv_out, index=False)

    # 6. Generate markdown report
    report_path = results_dir / f"{folder_name}_report.md"
    md = f"""# IndicSent-M: Phase 6 — {display_name} Training Report

**Model:** {display_name}  
**Task:** 3-Class Sentiment Classification on Meme Images (Negative=0, Neutral=1, Positive=2)  
**Dataset:** Memotion 3.0 (Train: 5,600 images, Validation: 1,400 images)  
**Hardware Used:** {gpu_name}  
**Precision:** FP16 Mixed Precision  
**Total Training Time:** {total_training_time:.1f}s ({total_training_time/60:.2f} mins)  
**Best Validation Epoch:** Epoch {best_epoch}  

---

## 1. Performance Overview

| Metric | Score |
| :--- | :---: |
| **Validation Accuracy** | **{best_metrics['accuracy']}%** |
| **Macro-F1 Score (Primary)** | **{best_metrics['macro_f1']}%** |
| **Weighted-F1 Score** | {best_metrics['weighted_f1']}% |
| **Negative Class F1 (0)** | {best_metrics['negative_f1']}% |
| **Neutral Class F1 (1)** | {best_metrics['neutral_f1']}% |
| **Positive Class F1 (2)** | {best_metrics['positive_f1']}% |
| **Inference Latency** | {best_metrics['latency_ms_per_sample']} ms/sample |

### Confusion Matrix:
```text
                  Predicted Negative    Predicted Neutral    Predicted Positive
Actual Negative :        {best_metrics['confusion_matrix'][0][0]:<15}      {best_metrics['confusion_matrix'][0][1]:<15}      {best_metrics['confusion_matrix'][0][2]:<15}
Actual Neutral  :        {best_metrics['confusion_matrix'][1][0]:<15}      {best_metrics['confusion_matrix'][1][1]:<15}      {best_metrics['confusion_matrix'][1][2]:<15}
Actual Positive :        {best_metrics['confusion_matrix'][2][0]:<15}      {best_metrics['confusion_matrix'][2][1]:<15}      {best_metrics['confusion_matrix'][2][2]:<15}
```
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"Generated report at {report_path}")

    return best_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="resnet50", choices=["resnet50", "efficientnet_b0", "efficientnet", "mobilenet_v3", "mobilenet"])
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    train_image_model(model_key=args.model, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
