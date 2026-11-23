"""
IndicSent-M: Deep Text Model Fine-Tuning Pipeline
Fine-tunes MuRIL and XLM-RoBERTa using PyTorch with:
- AdamW optimizer & linear learning rate schedule
- Class-weighted CrossEntropyLoss for imbalance mitigation
- Mixed precision training (AMP) on NVIDIA CUDA GPUs
- Early stopping tracking validation Macro-F1
- Comprehensive evaluation logging
"""

import sys
import os
import time
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import get_linear_schedule_with_warmup

# Ensure local imports work
project_root = Path(__file__).resolve().parents[3]
src_dir = Path(__file__).resolve().parents[1]
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from models.text_models import TransformerSentimentClassifier, get_text_tokenizer
from evaluation.evaluate import compute_metrics, print_evaluation_summary

class TextMemeDataset(Dataset):
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

def train_text_model(
    model_key: str = "muril",
    epochs: int = 3,
    batch_size: int = 32,
    lr: float = 2e-5,
    max_length: int = 128,
    patience: int = 2
):
    print("=" * 60)
    print(f"IndicSent-M: Fine-Tuning Text Model [{model_key.upper()}]")
    print("=" * 60)

    model_name_map = {
        "muril": "google/muril-base-cased",
        "xlm_roberta": "xlm-roberta-base"
    }
    pretrained_name = model_name_map.get(model_key.lower(), model_key)
    display_name = f"MuRIL (Fine-Tuned)" if "muril" in model_key.lower() else f"XLM-RoBERTa (Fine-Tuned)"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    splits_dir = project_root / "model_training" / "data_splits"
    checkpoints_dir = project_root / "model_training" / "checkpoints" / "text"
    results_dir = project_root / "model_training" / "results"
    
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    df_train = pd.read_csv(splits_dir / "train.csv")
    df_val = pd.read_csv(splits_dir / "validation.csv")

    tokenizer = get_text_tokenizer(pretrained_name)
    train_dataset = TextMemeDataset(df_train["clean_text"], df_train["label_id"], tokenizer, max_length)
    val_dataset = TextMemeDataset(df_val["clean_text"], df_val["label_id"], tokenizer, max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_dataset, batch_size=batch_size * 2, shuffle=False)

    # Class weights for loss
    class_counts = df_train["label_id"].value_counts().sort_index().values
    weights = len(df_train) / (3.0 * class_counts)
    class_weights = torch.tensor(weights, dtype=torch.float).to(device)
    print(f"Class counts [Neg, Neu, Pos]: {class_counts.tolist()}")
    print(f"Computed loss weights: {class_weights.cpu().numpy().round(3).tolist()}")

    model = TransformerSentimentClassifier(pretrained_model_name=pretrained_name, num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    best_macro_f1 = -1.0
    best_metrics = None
    epochs_no_improve = 0
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        print(f"\nEpoch {epoch}/{epochs} Training...")
        
        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch} Train Loss: {avg_loss:.4f}")

        # Evaluation
        model.eval()
        all_preds = []
        all_trues = []
        t_val_start = time.time()

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    logits = model(input_ids=input_ids, attention_mask=attention_mask)
                    preds = torch.argmax(logits, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_trues.extend(labels.cpu().numpy())

        val_duration = time.time() - t_val_start
        latency_ms = (val_duration / len(val_dataset)) * 1000.0

        current_metrics = compute_metrics(
            y_true=np.array(all_trues),
            y_pred=np.array(all_preds),
            model_name=display_name,
            latency_ms=latency_ms,
            train_time_sec=time.time() - t_start
        )

        print(f"Epoch {epoch} Val Accuracy: {current_metrics['accuracy']}%, Val Macro-F1: {current_metrics['macro_f1']}%")

        if current_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = current_metrics["macro_f1"]
            best_metrics = current_metrics
            epochs_no_improve = 0
            # Save checkpoint
            safe_name = model_key.lower().replace("-", "_")
            ckpt_path = checkpoints_dir / f"{safe_name}_best.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "macro_f1": best_macro_f1,
                "model_name": pretrained_name
            }, ckpt_path)
            print(f"--> Saved new best checkpoint to {ckpt_path.name} (Macro-F1: {best_macro_f1}%)")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch} epochs.")
                break

    print("\n" + "=" * 60)
    print(f"FINAL BEST EVALUATION: {display_name}")
    print("=" * 60)
    print_evaluation_summary(best_metrics)

    # Save to comparison CSV
    csv_out = results_dir / "model_comparison.csv"
    rec = {
        "Model": display_name,
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
        "Train Time (s)": best_metrics["train_time_sec"],
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

    print(f"Logged {display_name} results to {csv_out}")
    return best_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="muril", choices=["muril", "xlm_roberta"])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()

    train_text_model(model_key=args.model, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
