"""
IndicSent-M: Phase 5B - XLM-RoBERTa Fine-Tuning Pipeline
Fine-tunes xlm-roberta-base for 3-class sentiment classification on Hinglish memes.
Academic and experimental rigor:
- Class-weighted CrossEntropyLoss
- FP16 Mixed Precision
- AdamW optimizer (lr=2e-5, weight_decay=0.01)
- Linear learning rate schedule with 10% warmup
- Gradient clipping (1.0)
- Early stopping based on validation Macro-F1
- Evaluation after each epoch (Accuracy, Macro-F1, Per-class Precision/Recall/F1, Confusion Matrix)
- Export of weights, tokenizer, config, and comprehensive reports
"""

import sys
import os
import json
import time
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
    get_linear_schedule_with_warmup
)

# Fix Windows console encoding for Indic characters
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Ensure local imports work
project_root = Path(__file__).resolve().parents[3]
src_dir = Path(__file__).resolve().parents[1]
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from evaluation.evaluate import compute_metrics, print_evaluation_summary

LABEL_MAP = {0: "Negative", 1: "Neutral", 2: "Positive"}
ID_MAP = {"Negative": 0, "Neutral": 1, "Positive": 2}

class XLMRobertaDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length: int = 128):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }

def run_xlm_roberta_finetuning(
    epochs: int = 4,
    batch_size: int = 8,
    grad_accum_steps: int = 2,
    lr: float = 2e-5,
    max_length: int = 128,
    patience: int = 2
):
    print("=" * 65)
    print("IndicSent-M: Phase 5B - XLM-RoBERTa Fine-Tuning Execution")
    print("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Compute Device: {device} ({gpu_name})")

    splits_dir = project_root / "model_training" / "data_splits"
    checkpoint_dir = project_root / "model_training" / "checkpoints" / "xlm_roberta"
    results_dir = project_root / "model_training" / "results"
    
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    train_path = splits_dir / "train.csv"
    val_path = splits_dir / "validation.csv"
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)

    print(f"Train samples: {len(df_train)}, Validation samples: {len(df_val)}")

    # 2. Tokenizer & Datasets
    model_name = "xlm-roberta-base"
    print(f"Loading tokenizer: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)

    train_dataset = XLMRobertaDataset(df_train["clean_text"], df_train["label_id"], tokenizer, max_length)
    val_dataset = XLMRobertaDataset(df_val["clean_text"], df_val["label_id"], tokenizer, max_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        shuffle=False
    )

    # 3. Class-weighted CrossEntropyLoss
    class_counts = df_train["label_id"].value_counts().sort_index().values
    weights = len(df_train) / (3.0 * class_counts)
    class_weights = torch.tensor(weights, dtype=torch.float).to(device)
    print(f"Class distribution [Neg, Neu, Pos]: {class_counts.tolist()}")
    print(f"Inverse class weights: {class_weights.cpu().numpy().round(3).tolist()}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # 4. Model Setup
    print(f"Instantiating {model_name} with SequenceClassification head (num_labels=3)...")
    config = AutoConfig.from_pretrained(
        model_name,
        num_labels=3,
        id2label=LABEL_MAP,
        label2id=ID_MAP,
        local_files_only=True
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        config=config,
        local_files_only=True
    ).to(device)

    # 5. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_update_steps = (len(train_loader) // grad_accum_steps) * epochs
    warmup_steps = int(total_update_steps * 0.1)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_update_steps)
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

    training_config = {
        "model_name": model_name,
        "epochs": epochs,
        "batch_size_per_step": batch_size,
        "grad_accum_steps": grad_accum_steps,
        "effective_batch_size": batch_size * grad_accum_steps,
        "learning_rate": lr,
        "max_length": max_length,
        "weight_decay": 0.01,
        "warmup_steps": warmup_steps,
        "total_update_steps": total_update_steps,
        "early_stopping_patience": patience,
        "fp16": torch.cuda.is_available(),
        "device": gpu_name
    }

    best_macro_f1 = -1.0
    best_metrics = None
    best_epoch = 0
    training_history = []
    epochs_no_improve = 0
    t_start = time.time()

    print(f"\nCommencing training ({epochs} epochs, effective batch size={batch_size * grad_accum_steps})...")

    for epoch in range(1, epochs + 1):
        epoch_t0 = time.time()
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs.logits, labels)
                loss = loss / grad_accum_steps

            scaler.scale(loss).backward()

            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            train_loss += loss.item() * grad_accum_steps

            if (step + 1) % 100 == 0 or (step + 1) == len(train_loader):
                cur_loss = train_loss / (step + 1)
                lr_cur = scheduler.get_last_lr()[0]
                print(f"  Epoch [{epoch}/{epochs}] Step [{step + 1}/{len(train_loader)}] | Running Loss: {cur_loss:.4f} | LR: {lr_cur:.2e}", flush=True)

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
                labels = batch["label"].to(device)

                with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = criterion(outputs.logits, labels)

                val_loss += loss.item()
                preds = torch.argmax(outputs.logits, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_trues.extend(labels.cpu().numpy())

        val_duration = time.time() - val_t0
        avg_val_loss = val_loss / len(val_loader)
        latency_ms = (val_duration / len(val_dataset)) * 1000.0

        current_metrics = compute_metrics(
            y_true=np.array(val_trues),
            y_pred=np.array(val_preds),
            model_name="XLM-RoBERTa (xlm-roberta-base)",
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
            model.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)
            config.save_pretrained(checkpoint_dir)

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
    print("XLM-ROBERTA FINE-TUNING COMPLETE")
    print("=" * 65)
    print(f"Best Validation Epoch: {best_epoch} (Macro-F1: {best_macro_f1}%)")
    print(f"Total Training Duration: {total_training_time:.1f}s ({total_training_time/60:.2f} mins)")
    print_evaluation_summary(best_metrics)

    # 6. Save results to xlm_roberta_results.json
    results_json_path = results_dir / "xlm_roberta_results.json"
    full_results = {
        "model_name": "xlm-roberta-base",
        "best_epoch": best_epoch,
        "total_training_time_sec": round(total_training_time, 2),
        "best_metrics": best_metrics,
        "training_history": training_history,
        "training_config": training_config
    }
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2)
    print(f"Saved evaluation results to {results_json_path}")

    # 7. Update model_comparison.csv
    csv_out = results_dir / "model_comparison.csv"
    rec = {
        "Model": "XLM-RoBERTa (xlm-roberta-base)",
        "Modality": "Text Only (Transformer)",
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
        existing_df = existing_df[existing_df["Model"] != "XLM-RoBERTa (xlm-roberta-base)"]
        combined_df = pd.concat([existing_df, rec_df], ignore_index=True)
        combined_df.to_csv(csv_out, index=False)
    else:
        rec_df.to_csv(csv_out, index=False)

    # 8. Generate xlm_roberta_report.md
    generate_xlm_roberta_report(best_metrics, training_history, training_config, total_training_time)

def generate_xlm_roberta_report(best_metrics, history, config, total_time):
    results_dir = project_root / "model_training" / "results"
    report_path = results_dir / "xlm_roberta_report.md"

    lr_acc = 43.79
    lr_f1 = 43.14
    muril_f1 = 40.75

    acc_diff = best_metrics["accuracy"] - lr_acc
    f1_diff = best_metrics["macro_f1"] - lr_f1

    acc_sign = "+" if acc_diff >= 0 else ""
    f1_sign = "+" if f1_diff >= 0 else ""

    md = f"""# IndicSent-M: Phase 5B — XLM-RoBERTa Fine-Tuning Report

**Model:** `xlm-roberta-base` (Cross-lingual RoBERTa)  
**Task:** 3-Class Sentiment Classification on Hinglish Memes (Negative=0, Neutral=1, Positive=2)  
**Dataset:** Memotion 3.0 (Train: 5,600 records, Validation: 1,400 records)  
**Hardware Used:** {config['device']}  
**Precision:** FP16 Mixed Precision  
**Total Training Time:** {total_time:.1f}s ({total_time/60:.2f} mins)  
**Best Validation Epoch:** Epoch {best_metrics.get('epoch', history[-1]['epoch'])}  

---

## 1. Executive Summary & Benchmark Comparison

| Metric | TF-IDF + Logistic Reg (Baseline) | MuRIL (google/muril-base-cased) | **XLM-RoBERTa (xlm-roberta-base)** | vs Baseline Diff |
| :--- | :---: | :---: | :---: | :---: |
| **Validation Accuracy** | {lr_acc}% | 40.93% | **{best_metrics['accuracy']}%** | **{acc_sign}{acc_diff:.2f}%** |
| **Macro-F1 Score (Primary)** | {lr_f1}% | {muril_f1}% | **{best_metrics['macro_f1']}%** | **{f1_sign}{f1_diff:.2f}%** |
| **Weighted-F1 Score** | 43.98% | 41.04% | **{best_metrics['weighted_f1']}%** | {best_metrics['weighted_f1'] - 43.98:+.2f}% |
| **Negative Class F1 (0)** | 37.84% | 39.95% | **{best_metrics['negative_f1']}%** | {best_metrics['negative_f1'] - 37.84:+.2f}% |
| **Neutral Class F1 (1)** | 47.63% | 43.05% | **{best_metrics['neutral_f1']}%** | {best_metrics['neutral_f1'] - 47.63:+.2f}% |
| **Positive Class F1 (2)** | 43.95% | 39.25% | **{best_metrics['positive_f1']}%** | {best_metrics['positive_f1'] - 43.95:+.2f}% |
| **Inference Latency** | 0.025 ms | 1.578 ms | **{best_metrics['latency_ms_per_sample']} ms/sample** | — |

---

## 2. Per-Class Precision & Recall Breakdown

| Sentiment Class | Precision (%) | Recall (%) | F1-Score (%) | True Sample Support |
| :--- | :---: | :---: | :---: | :---: |
| **Negative [0]** | {best_metrics['confusion_matrix'][0][0] / max(sum(best_metrics['confusion_matrix'][i][0] for i in range(3)), 1)*100:.2f}% | {best_metrics['negative_recall']}% | **{best_metrics['negative_f1']}%** | 351 |
| **Neutral [1]** | {best_metrics['confusion_matrix'][1][1] / max(sum(best_metrics['confusion_matrix'][i][1] for i in range(3)), 1)*100:.2f}% | {best_metrics['neutral_recall']}% | **{best_metrics['neutral_f1']}%** | 594 |
| **Positive [2]** | {best_metrics['confusion_matrix'][2][2] / max(sum(best_metrics['confusion_matrix'][i][2] for i in range(3)), 1)*100:.2f}% | {best_metrics['positive_recall']}% | **{best_metrics['positive_f1']}%** | 455 |
| **Macro Average** | **{best_metrics['macro_precision']}%** | **{best_metrics['macro_recall']}%** | **{best_metrics['macro_f1']}%** | 1,400 |

### Confusion Matrix:
```text
                  Predicted Negative    Predicted Neutral    Predicted Positive
Actual Negative :        {best_metrics['confusion_matrix'][0][0]:<15}      {best_metrics['confusion_matrix'][0][1]:<15}      {best_metrics['confusion_matrix'][0][2]:<15}
Actual Neutral  :        {best_metrics['confusion_matrix'][1][0]:<15}      {best_metrics['confusion_matrix'][1][1]:<15}      {best_metrics['confusion_matrix'][1][2]:<15}
Actual Positive :        {best_metrics['confusion_matrix'][2][0]:<15}      {best_metrics['confusion_matrix'][2][1]:<15}      {best_metrics['confusion_matrix'][2][2]:<15}
```

---

## 3. Epoch-by-Epoch Training Dynamics

| Epoch | Train Loss | Val Loss | Val Accuracy (%) | Val Macro-F1 (%) | Neg F1 (%) | Neu F1 (%) | Pos F1 (%) | Epoch Duration |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for h in history:
        md += f"| {h['epoch']} | {h['train_loss']} | {h['val_loss']} | {h['val_accuracy']}% | **{h['val_macro_f1']}%** | {h['val_negative_f1']}% | {h['val_neutral_f1']}% | {h['val_positive_f1']}% | {h['epoch_train_time_sec']}s |\n"

    md += f"""
---

## 4. Hyperparameter & Optimization Configuration

- **Pretrained Checkpoint:** `xlm-roberta-base` (RoBERTa architecture, 12 layers, 768 hidden dimensions)
- **Token Sequence Length:** 128 tokens
- **Per-Device Batch Size:** {config['batch_size_per_step']}
- **Gradient Accumulation Steps:** {config['grad_accum_steps']} (Effective Batch Size: {config['effective_batch_size']})
- **Optimizer:** AdamW (Learning Rate: {config['learning_rate']}, Weight Decay: 0.01)
- **LR Schedule:** Linear decay with 10% warmup steps
- **Loss Function:** Class-weighted Cross-Entropy Loss (Inverse frequency weighting)
- **Mixed Precision:** FP16 via PyTorch AMP on NVIDIA RTX 5050 Laptop GPU
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"Generated comprehensive report at {report_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()

    run_xlm_roberta_finetuning(
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        lr=args.lr
    )
