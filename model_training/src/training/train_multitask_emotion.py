"""
IndicSent-M: Multi-Task Multimodal Sentiment & Six-Emotion Training Pipeline
===========================================================================
Trains shared MuRIL + ResNet-50 Gated Fusion representation on two tasks:
  1. 3-class Sentiment Classification (CrossEntropyLoss, Softmax)
  2. 6-class Multi-Label Emotion Classification (Masked BCEWithLogitsLoss, Sigmoid)

Features:
- Preserves existing checkpoints; saves strictly to checkpoints/multitask_emotion/
- Supports unannotated sample masking (missing emotion labels are not penalized)
- Class-imbalance handling using pos_weight per emotion
- Independent tracking of Sentiment and Emotion metrics
"""

import sys
import os
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Fix Windows console line buffering
try:
    sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
except Exception:
    pass
SRC_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SRC_DIR.parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"

for p in [str(SRC_DIR), str(BACKEND_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from models.multimodal_models import (
    MultiTaskGatedMultimodalClassifier,
    SIX_EMOTIONS,
    NUM_EMOTIONS,
    EMOTION_EMOJIS,
    _resolve_text_backbone
)
from preprocessing.clean_text import clean_ocr_text
from preprocessing.preprocess_images import get_image_transforms, load_and_verify_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SENTIMENT_LABEL_MAP = {0: "Negative", 1: "Neutral", 2: "Positive"}
SENTIMENT_ID_MAP = {"Negative": 0, "Neutral": 1, "Positive": 2}


class MultiTaskMemeDataset(Dataset):
    """
    Dataset returning text tokens, image pixel values, 3-class sentiment label,
    and 6-class multi-label emotion vector with a verification mask.
    """
    def __init__(
        self,
        df_sentiment: pd.DataFrame,
        df_emotions: pd.DataFrame,
        base_dir: Path,
        tokenizer,
        img_transform,
        max_length: int = 128
    ):
        self.base_dir = base_dir
        self.tokenizer = tokenizer
        self.img_transform = img_transform
        self.max_length = max_length

        # Standardize emotion columns in df_emotions to match SIX_EMOTIONS
        df_em_copy = df_emotions.copy()
        col_rename = {em.lower(): em for em in SIX_EMOTIONS}
        df_em_copy = df_em_copy.rename(columns=col_rename)

        # Merge on image_filename / image_name
        merged = df_sentiment.merge(
            df_em_copy[["image_name", "is_verified"] + SIX_EMOTIONS],
            left_on="image_filename",
            right_on="image_name",
            how="left"
        ).reset_index(drop=True)

        self.df = merged

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 1. Text tokenization
        clean_text = clean_ocr_text(str(row["clean_text"]))
        enc = self.tokenizer(
            clean_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        # 2. Image loading & transform
        img_path = self.base_dir / row["image_rel_path"]
        img = load_and_verify_image(img_path)
        if img is None:
            img = Image.new("RGB", (224, 224), (128, 128, 128))
        pixel_values = self.img_transform(img)

        # 3. Sentiment label
        sentiment_label = int(row["label_id"])

        # 4. Multi-label Emotion vector & mask
        is_verified = int(row.get("is_verified", 0)) == 1
        emotion_vals = []
        for em in SIX_EMOTIONS:
            val = row.get(em, -1)
            # 1 if active, 0 if inactive, 0 fallback if unannotated
            emotion_vals.append(1.0 if val == 1 else 0.0)

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "pixel_values": pixel_values,
            "sentiment_label": torch.tensor(sentiment_label, dtype=torch.long),
            "emotion_labels": torch.tensor(emotion_vals, dtype=torch.float),
            "emotion_mask": torch.tensor(1.0 if is_verified else 0.0, dtype=torch.float)
        }


def compute_multitask_metrics(
    sentiment_trues: np.ndarray,
    sentiment_preds: np.ndarray,
    emotion_trues: np.ndarray,
    emotion_probs: np.ndarray,
    emotion_mask: np.ndarray
) -> Dict[str, Any]:
    """
    Computes comprehensive sentiment and multi-label emotion metrics.
    """
    # 1. Sentiment metrics
    acc = accuracy_score(sentiment_trues, sentiment_preds)
    sent_macro_f1 = f1_score(sentiment_trues, sentiment_preds, average="macro", zero_division=0)
    sent_weighted_f1 = f1_score(sentiment_trues, sentiment_preds, average="weighted", zero_division=0)
    per_class_sent = f1_score(sentiment_trues, sentiment_preds, average=None, labels=[0, 1, 2], zero_division=0)

    # 2. Emotion metrics (calculated strictly on verified samples)
    valid_idx = np.where(emotion_mask > 0.5)[0]
    if len(valid_idx) == 0:
        return {
            "sentiment_accuracy": round(float(acc) * 100, 2),
            "sentiment_macro_f1": round(float(sent_macro_f1) * 100, 2),
            "sentiment_weighted_f1": round(float(sent_weighted_f1) * 100, 2),
            "emotion_metrics_available": False
        }

    em_true_sub = emotion_trues[valid_idx]
    em_prob_sub = emotion_probs[valid_idx]
    em_pred_sub = (em_prob_sub >= 0.5).astype(int)

    # Per-class emotion P, R, F1
    per_em_f1 = f1_score(em_true_sub, em_pred_sub, average=None, zero_division=0)
    per_em_p = precision_score(em_true_sub, em_pred_sub, average=None, zero_division=0)
    per_em_r = recall_score(em_true_sub, em_pred_sub, average=None, zero_division=0)

    em_macro_f1 = f1_score(em_true_sub, em_pred_sub, average="macro", zero_division=0)
    em_micro_f1 = f1_score(em_true_sub, em_pred_sub, average="micro", zero_division=0)

    # Dominant Emotion Accuracy: Is the highest predicted probability emotion an active true emotion?
    dominant_preds = np.argmax(em_prob_sub, axis=1)
    dominant_hits = 0
    for i in range(len(valid_idx)):
        d_pred = dominant_preds[i]
        if em_true_sub[i, d_pred] == 1:
            dominant_hits += 1
    dominant_accuracy = (dominant_hits / len(valid_idx)) * 100.0

    per_em_dict = {}
    for i, em in enumerate(SIX_EMOTIONS):
        per_em_dict[em] = {
            "f1": round(float(per_em_f1[i]) * 100, 2),
            "precision": round(float(per_em_p[i]) * 100, 2),
            "recall": round(float(per_em_r[i]) * 100, 2),
            "support": int(np.sum(em_true_sub[:, i]))
        }

    return {
        "sentiment_accuracy": round(float(acc) * 100, 2),
        "sentiment_macro_f1": round(float(sent_macro_f1) * 100, 2),
        "sentiment_weighted_f1": round(float(sent_weighted_f1) * 100, 2),
        "sentiment_per_class_f1": {
            "Negative": round(float(per_class_sent[0]) * 100, 2),
            "Neutral": round(float(per_class_sent[1]) * 100, 2),
            "Positive": round(float(per_class_sent[2]) * 100, 2)
        },
        "emotion_metrics_available": True,
        "emotion_verified_eval_samples": len(valid_idx),
        "emotion_macro_f1": round(float(em_macro_f1) * 100, 2),
        "emotion_micro_f1": round(float(em_micro_f1) * 100, 2),
        "dominant_emotion_accuracy": round(float(dominant_accuracy), 2),
        "per_emotion": per_em_dict
    }


def train_multitask(
    epochs: int = 4,
    batch_size: int = 16,
    lr: float = 1e-4,
    lambda_emotion: float = 1.0,
    patience: int = 2,
    exp_name: str = "multitask_emotion",
    seed: int = 42
):
    # Set seed for exact reproducibility across experiments
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print("=" * 75)
    print(f"IndicSent-M: Multi-Task Training Experiment [{exp_name}]")
    print(f"Device: {device} | Epochs: {epochs} | Batch Size: {batch_size} | Lambda Emotion: {lambda_emotion} | Seed: {seed}")
    print("=" * 75)

    splits_dir = PROJECT_ROOT / "model_training" / "data_splits"
    emotions_dir = splits_dir / "emotions"
    
    if exp_name == "multitask_emotion":
        checkpoint_dir = PROJECT_ROOT / "model_training" / "checkpoints" / "multitask_emotion"
    else:
        checkpoint_dir = PROJECT_ROOT / "model_training" / "checkpoints" / "experiments" / exp_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    df_train_sent = pd.read_csv(splits_dir / "train.csv")
    df_val_sent = pd.read_csv(splits_dir / "validation.csv")
    df_train_em = pd.read_csv(emotions_dir / "train_emotions.csv")
    df_val_em = pd.read_csv(emotions_dir / "validation_emotions.csv")

    text_backbone = "google/muril-base-cased"
    resolved_text = _resolve_text_backbone(text_backbone)
    tokenizer = AutoTokenizer.from_pretrained(resolved_text, local_files_only=True)

    t_train = get_image_transforms("resnet50", is_training=True)
    t_val = get_image_transforms("resnet50", is_training=False)

    train_dataset = MultiTaskMemeDataset(df_train_sent, df_train_em, PROJECT_ROOT, tokenizer, t_train)
    val_dataset = MultiTaskMemeDataset(df_val_sent, df_val_em, PROJECT_ROOT, tokenizer, t_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=torch.cuda.is_available(), num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size * 2, shuffle=False, num_workers=0)

    # 2. Losses & Imbalance Weights
    # Sentiment inverse class weights
    class_counts = df_train_sent["label_id"].value_counts().sort_index().values
    sent_weights = torch.tensor(len(df_train_sent) / (3.0 * class_counts), dtype=torch.float).to(device)
    sent_criterion = nn.CrossEntropyLoss(weight=sent_weights)

    # Emotion pos_weights for BCEWithLogitsLoss
    verified_train_em = df_train_em[df_train_em["is_verified"] == 1].rename(columns={em.lower(): em for em in SIX_EMOTIONS})
    em_pos_weights = []
    for em in SIX_EMOTIONS:
        pos_cnt = max(1, int((verified_train_em[em] == 1).sum()))
        neg_cnt = max(1, len(verified_train_em) - pos_cnt)
        em_pos_weights.append(float(neg_cnt / pos_cnt))
    em_pos_weight_tensor = torch.tensor(em_pos_weights, dtype=torch.float).to(device)
    emotion_criterion = nn.BCEWithLogitsLoss(pos_weight=em_pos_weight_tensor, reduction="none")

    print(f"Sentiment class weights: {sent_weights.cpu().numpy().round(3).tolist()}")
    print(f"Emotion pos_weights: {em_pos_weight_tensor.cpu().numpy().round(3).tolist()}")

    # 3. Model Instantiation & Weight Initialization
    model = MultiTaskGatedMultimodalClassifier(
        text_backbone=text_backbone,
        vision_backbone="resnet50",
        num_classes=3,
        num_emotions=6,
        freeze_text=True,
        freeze_vision=True
    ).to(device)

    # Preload trained base weights from existing gated fusion checkpoint if present
    base_ckpt_path = PROJECT_ROOT / "model_training" / "checkpoints" / "multimodal" / "gated_fusion" / "best_model.pt"
    if base_ckpt_path.exists():
        print(f"Preloading existing shared weights from {base_ckpt_path} (strict=False)...")
        base_state = torch.load(base_ckpt_path, map_location=device)
        state_dict = base_state.get("model_state_dict", base_state)
        # Load all shared keys (encoders, projections, gate, norm, sentiment classifier)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        print(f"  Initialized shared layers. Newly added keys for training: {missing_keys}")

    # Optimize downstream heads and gating layers
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

    training_config = {
        "model_name": "MuRIL + ResNet-50 (Multi-Task Sentiment & Emotion Gated Fusion)",
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "lambda_emotion": lambda_emotion,
        "emotions": SIX_EMOTIONS,
        "num_emotions": 6,
        "device": str(device)
    }

    best_combined_score = -1.0
    best_metrics = None
    best_epoch = 0
    training_history = []
    epochs_no_improve = 0

    print(f"\nCommencing Multi-Task Training ({epochs} epochs)...")
    for epoch in range(1, epochs + 1):
        epoch_t0 = time.time()
        model.train()
        train_loss = 0.0
        train_sent_loss = 0.0
        train_em_loss = 0.0

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixels = batch["pixel_values"].to(device)
            sent_labels = batch["sentiment_label"].to(device)
            em_labels = batch["emotion_labels"].to(device)
            em_mask = batch["emotion_mask"].to(device)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                sent_logits, em_logits = model(input_ids, attention_mask, pixels)
                l_sent = sent_criterion(sent_logits, sent_labels)

                # Masked BCEWithLogitsLoss for emotions
                l_em_raw = emotion_criterion(em_logits, em_labels) # (B, 6)
                l_em_per_sample = l_em_raw.mean(dim=1) # (B,)
                
                # Apply mask: only penalize verified samples
                denom = torch.clamp(em_mask.sum(), min=1.0)
                l_em = (l_em_per_sample * em_mask).sum() / denom

                total_loss = l_sent + (lambda_emotion * l_em)

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += total_loss.item()
            train_sent_loss += l_sent.item()
            train_em_loss += l_em.item()

            if (step + 1) % 50 == 0 or (step + 1) == len(train_loader):
                print(f"  Epoch [{epoch}/{epochs}] Step [{step + 1}/{len(train_loader)}] | Running Loss: {train_loss / (step + 1):.4f}", flush=True)

        avg_train_loss = train_loss / len(train_loader)
        avg_sent_loss = train_sent_loss / len(train_loader)
        avg_em_loss = train_em_loss / len(train_loader)

        # Validation
        model.eval()
        val_sent_trues = []
        val_sent_preds = []
        val_em_trues = []
        val_em_probs = []
        val_em_masks = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                pixels = batch["pixel_values"].to(device)
                sent_labels = batch["sentiment_label"].to(device)
                em_labels = batch["emotion_labels"].to(device)
                em_mask = batch["emotion_mask"].to(device)

                with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                    sent_logits, em_logits = model(input_ids, attention_mask, pixels)

                sent_preds = torch.argmax(sent_logits, dim=1)
                em_probs = torch.sigmoid(em_logits)

                val_sent_trues.extend(sent_labels.cpu().numpy())
                val_sent_preds.extend(sent_preds.cpu().numpy())
                val_em_trues.extend(em_labels.cpu().numpy())
                val_em_probs.extend(em_probs.cpu().numpy())
                val_em_masks.extend(em_mask.cpu().numpy())

        current_metrics = compute_multitask_metrics(
            sentiment_trues=np.array(val_sent_trues),
            sentiment_preds=np.array(val_sent_preds),
            emotion_trues=np.array(val_em_trues),
            emotion_probs=np.array(val_em_probs),
            emotion_mask=np.array(val_em_masks)
        )

        sent_f1 = current_metrics["sentiment_macro_f1"]
        em_f1 = current_metrics.get("emotion_macro_f1", 0.0)
        dom_acc = current_metrics.get("dominant_emotion_accuracy", 0.0)
        # Combined score balancing sentiment and emotion macro-F1
        combined_score = sent_f1 + em_f1

        epoch_time = time.time() - epoch_t0
        print(f"Epoch [{epoch}/{epochs}] | Train Loss: {avg_train_loss:.4f} (Sent: {avg_sent_loss:.4f}, Em: {avg_em_loss:.4f})")
        print(f"  Sentiment -> Acc: {current_metrics['sentiment_accuracy']}%, Macro-F1: {sent_f1}%")
        print(f"  Emotion   -> Macro-F1: {em_f1}%, Micro-F1: {current_metrics.get('emotion_micro_f1', 0.0)}%, Dominant Emotion Acc: {dom_acc}% ({epoch_time:.1f}s)")

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "sentiment_loss": round(avg_sent_loss, 4),
            "emotion_loss": round(avg_em_loss, 4),
            "metrics": current_metrics
        }
        training_history.append(epoch_record)

        if combined_score > best_combined_score:
            best_combined_score = combined_score
            best_metrics = current_metrics
            best_epoch = epoch
            epochs_no_improve = 0

            print(f"  --> NEW BEST MULTI-TASK MODEL ACHIEVED! Saving to {checkpoint_dir}...")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "sentiment_macro_f1": sent_f1,
                "emotion_macro_f1": em_f1,
                "dominant_emotion_accuracy": dom_acc,
                "config": training_config,
                "emotions": SIX_EMOTIONS,
                "emotion_emojis": EMOTION_EMOJIS
            }, checkpoint_dir / "best_model.pt")

            with open(checkpoint_dir / "training_config.json", "w", encoding="utf-8") as f:
                json.dump(training_config, f, indent=2)

            with open(checkpoint_dir / "multitask_metrics.json", "w", encoding="utf-8") as f:
                json.dump(best_metrics, f, indent=2)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch}!")
                break

    print(f"\nTraining Complete. Best Epoch: {best_epoch} | Sentiment Macro-F1: {best_metrics['sentiment_macro_f1']}% | Emotion Macro-F1: {best_metrics.get('emotion_macro_f1', 0.0)}% | Dominant Emotion Accuracy: {best_metrics.get('dominant_emotion_accuracy', 0.0)}%")
    return best_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Task Sentiment and Six-Emotion Training")
    parser.add_argument("--epochs", type=int, default=4, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--lambda-emotion", type=float, default=1.0, help="Weight for emotion loss")
    parser.add_argument("--exp-name", type=str, default="multitask_emotion", help="Experiment name for checkpoint directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--patience", type=int, default=2, help="Early stopping patience")
    args = parser.parse_args()

    train_multitask(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lambda_emotion=args.lambda_emotion,
        patience=args.patience,
        exp_name=args.exp_name,
        seed=args.seed
    )
