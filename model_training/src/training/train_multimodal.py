"""
IndicSent-M: Phase 7 — Multimodal Fusion Training Pipeline
Implements and benchmarks:
1. muril_resnet50: MuRIL + ResNet-50 Feature Concatenation
2. muril_efficientnet: MuRIL + EfficientNet-B0 Feature Concatenation
3. xlm_resnet50: XLM-RoBERTa + ResNet-50 Feature Concatenation
4. gated_fusion: MuRIL + ResNet-50 Gated Fusion (Trainable Gating Mechanism)

Standards:
- Training split: Mild image augmentation (horizontal flip, color jitter)
- Validation split: Deterministic resize and normalization
- Class-weighted CrossEntropyLoss
- FP16 Automatic Mixed Precision (AMP)
- Validation after every epoch (Accuracy, Macro-F1, Per-class F1, Confusion Matrix)
- Export of weights, config, label mapping, and comprehensive reports
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
from transformers import AutoTokenizer

# Fix Windows console encoding for Indic characters
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Ensure local imports work
project_root = Path(__file__).resolve().parents[3]
src_dir = Path(__file__).resolve().parents[1]
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from models.multimodal_models import MultimodalConcatClassifier, GatedMultimodalClassifier
from preprocessing.preprocess_images import get_image_transforms, load_and_verify_image
from evaluation.evaluate import compute_metrics, print_evaluation_summary

LABEL_MAP = {0: "Negative", 1: "Neutral", 2: "Positive"}
ID_MAP = {"Negative": 0, "Neutral": 1, "Positive": 2}

class MultimodalMemeDataset(Dataset):
    def __init__(self, df: pd.DataFrame, base_dir: Path, tokenizer, img_transform, max_length: int = 128):
        self.df = df.reset_index(drop=True)
        self.base_dir = base_dir
        self.tokenizer = tokenizer
        self.img_transform = img_transform
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 1. Text tokenization
        text = str(row["clean_text"])
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # 2. Image loading and transformation
        img_path = self.base_dir / row["image_rel_path"]
        img = load_and_verify_image(img_path)
        if img is None:
            img = Image.new("RGB", (224, 224), (128, 128, 128))
            
        pixel_values = self.img_transform(img)
        
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "pixel_values": pixel_values,
            "label": torch.tensor(row["label_id"], dtype=torch.long)
        }

def train_multimodal_model(
    arch_key: str = "muril_resnet50",
    epochs: int = 4,
    batch_size: int = 16,
    lr: float = 1e-4,
    patience: int = 2
):
    print("=" * 65)
    print(f"IndicSent-M: Phase 7 — Training Multimodal Fusion [{arch_key}]")
    print("=" * 65)

    arch_configs = {
        "muril_resnet50": {
            "display_name": "MuRIL + ResNet-50 (Concat Fusion)",
            "text_backbone": "google/muril-base-cased",
            "vision_backbone": "resnet50",
            "arch_type": "concat",
            "folder_name": "muril_resnet50"
        },
        "muril_efficientnet": {
            "display_name": "MuRIL + EfficientNet-B0 (Concat Fusion)",
            "text_backbone": "google/muril-base-cased",
            "vision_backbone": "efficientnet_b0",
            "arch_type": "concat",
            "folder_name": "muril_efficientnet"
        },
        "xlm_resnet50": {
            "display_name": "XLM-RoBERTa + ResNet-50 (Concat Fusion)",
            "text_backbone": "xlm-roberta-base",
            "vision_backbone": "resnet50",
            "arch_type": "concat",
            "folder_name": "xlm_resnet50"
        },
        "gated_fusion": {
            "display_name": "MuRIL + ResNet-50 (Gated Fusion)",
            "text_backbone": "google/muril-base-cased",
            "vision_backbone": "resnet50",
            "arch_type": "gated",
            "folder_name": "gated_fusion"
        }
    }

    if arch_key not in arch_configs:
        raise ValueError(f"Unknown multimodal arch: {arch_key}. Available: {list(arch_configs.keys())}")

    cfg = arch_configs[arch_key]
    display_name = cfg["display_name"]
    folder_name = cfg["folder_name"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Compute Device: {device} ({gpu_name})")

    splits_dir = project_root / "model_training" / "data_splits"
    checkpoint_dir = project_root / "model_training" / "checkpoints" / "multimodal" / folder_name
    results_dir = project_root / "model_training" / "results"
    
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    df_train = pd.read_csv(splits_dir / "train.csv")
    df_val = pd.read_csv(splits_dir / "validation.csv")
    print(f"Train samples: {len(df_train)}, Validation samples: {len(df_val)}")

    # 2. Tokenizer & Transforms
    print(f"Loading tokenizer for {cfg['text_backbone']}...")
    tokenizer = AutoTokenizer.from_pretrained(cfg["text_backbone"], local_files_only=True)
    t_train = get_image_transforms(cfg["vision_backbone"], is_training=True)
    t_val = get_image_transforms(cfg["vision_backbone"], is_training=False)

    train_dataset = MultimodalMemeDataset(df_train, project_root, tokenizer, t_train)
    val_dataset = MultimodalMemeDataset(df_val, project_root, tokenizer, t_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=torch.cuda.is_available(), num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size * 2, shuffle=False, num_workers=0)

    # 3. Class-weighted CrossEntropyLoss
    class_counts = df_train["label_id"].value_counts().sort_index().values
    weights = len(df_train) / (3.0 * class_counts)
    class_weights = torch.tensor(weights, dtype=torch.float).to(device)
    print(f"Class distribution [Neg, Neu, Pos]: {class_counts.tolist()}")
    print(f"Inverse class weights: {class_weights.cpu().numpy().round(3).tolist()}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # 4. Model instantiation
    print(f"Building {display_name}...")
    if cfg["arch_type"] == "concat":
        model = MultimodalConcatClassifier(
            text_backbone=cfg["text_backbone"],
            vision_backbone=cfg["vision_backbone"],
            num_classes=3,
            freeze_text=True,
            freeze_vision=True
        ).to(device)
    else:
        model = GatedMultimodalClassifier(
            text_backbone=cfg["text_backbone"],
            vision_backbone=cfg["vision_backbone"],
            num_classes=3,
            freeze_text=True,
            freeze_vision=True
        ).to(device)

    # Trainable parameters are projection layers and fusion head
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

    training_config = {
        "model_name": display_name,
        "arch_key": arch_key,
        "text_backbone": cfg["text_backbone"],
        "vision_backbone": cfg["vision_backbone"],
        "fusion_type": cfg["arch_type"],
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
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

    print(f"\nCommencing multimodal training ({epochs} epochs, batch size={batch_size})...")

    for epoch in range(1, epochs + 1):
        epoch_t0 = time.time()
        model.train()
        train_loss = 0.0

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixels = batch["pixel_values"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                logits = model(input_ids, attention_mask, pixels)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
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
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                pixels = batch["pixel_values"].to(device)
                labels = batch["label"].to(device)

                with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                    logits = model(input_ids, attention_mask, pixels)
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
                "arch_key": arch_key,
                "display_name": display_name,
                "config": training_config
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
        "arch_key": arch_key,
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
        "Modality": "Multimodal (Text + Image)",
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
    md = f"""# IndicSent-M: Phase 7 — {display_name} Training Report

**Model Architecture:** {display_name}  
**Text Modality Backbone:** {cfg['text_backbone']}  
**Vision Modality Backbone:** {cfg['vision_backbone']}  
**Fusion Mechanism:** {cfg['arch_type'].capitalize()}  
**Task:** 3-Class Sentiment Classification on Hinglish Memes (Negative=0, Neutral=1, Positive=2)  
**Dataset:** Memotion 3.0 (Train: 5,600 multimodal samples, Validation: 1,400 multimodal samples)  
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
    parser.add_argument("--arch", type=str, default="muril_resnet50", choices=["muril_resnet50", "muril_efficientnet", "xlm_resnet50", "gated_fusion"])
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    train_multimodal_model(arch_key=args.arch, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
