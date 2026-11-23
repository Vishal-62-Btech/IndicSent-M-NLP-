"""
IndicSent-M: Empirical Investigation of Multimodal & OCR Predictions
Evaluates Conditions A, B, C, D, E on the exact 1,400-sample validation split.
"""

import sys
import time
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Ensure paths
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_ROOT / "model_training" / "src"
sys.path.insert(0, str(SRC_DIR))

from inference.predictor import IndicSentPredictor
from evaluation.evaluate import compute_metrics, TARGET_NAMES, CLASS_INDICES
from preprocessing.preprocess_images import load_and_verify_image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VAL_CSV = PROJECT_ROOT / "model_training" / "data_splits" / "validation.csv"

df_val = pd.read_csv(VAL_CSV)
y_true = df_val["label_id"].values
n_samples = len(df_val)

print(f"Loaded validation split: {n_samples} samples.")
print(f"Ground Truth counts: Neg={np.sum(y_true==0)}, Neu={np.sum(y_true==1)}, Pos={np.sum(y_true==2)}")

results_summary = {}

def evaluate_predictions(y_pred_arr, cond_name, desc):
    acc = np.mean(y_true == y_pred_arr) * 100.0
    from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred_arr, labels=CLASS_INDICES, average=None, zero_division=0)
    macro_f1 = np.mean(f1) * 100.0
    macro_p = np.mean(p) * 100.0
    macro_r = np.mean(r) * 100.0
    cm = confusion_matrix(y_true, y_pred_arr, labels=CLASS_INDICES)
    pred_counts = [int(np.sum(y_pred_arr == c)) for c in CLASS_INDICES]

    res = {
        "Condition": cond_name,
        "Description": desc,
        "Accuracy (%)": round(acc, 2),
        "Macro-F1 (%)": round(macro_f1, 2),
        "Macro-P (%)": round(macro_p, 2),
        "Macro-R (%)": round(macro_r, 2),
        "Neg P/R/F1 (%)": f"{p[0]*100:.1f} / {r[0]*100:.1f} / {f1[0]*100:.1f}",
        "Neu P/R/F1 (%)": f"{p[1]*100:.1f} / {r[1]*100:.1f} / {f1[1]*100:.1f}",
        "Pos P/R/F1 (%)": f"{p[2]*100:.1f} / {r[2]*100:.1f} / {f1[2]*100:.1f}",
        "Pred Counts [Neg, Neu, Pos]": pred_counts,
        "Confusion Matrix": cm.tolist()
    }
    results_summary[cond_name] = res

    print("\n" + "="*70)
    print(f"Condition: {cond_name} - {desc}")
    print("="*70)
    print(f"Accuracy: {acc:.2f}% | Macro-F1: {macro_f1:.2f}% | Macro-P: {macro_p:.2f}% | Macro-R: {macro_r:.2f}%")
    print(f"Predicted counts [Neg, Neu, Pos]: {pred_counts}")
    print(f"  Neg (0) -> P: {p[0]*100:.1f}%, R: {r[0]*100:.1f}%, F1: {f1[0]*100:.1f}%")
    print(f"  Neu (1) -> P: {p[1]*100:.1f}%, R: {r[1]*100:.1f}%, F1: {f1[1]*100:.1f}%")
    print(f"  Pos (2) -> P: {p[2]*100:.1f}%, R: {r[2]*100:.1f}%, F1: {f1[2]*100:.1f}%")
    print(f"Confusion Matrix (Rows=True, Cols=Pred):")
    print(cm)
    return res


# ---------------------------------------------------------------------------
# Condition A: ResNet-50 Image-Only
# ---------------------------------------------------------------------------
print("\n[Running Condition A: ResNet-50 Image-Only]...")
pred_img = IndicSentPredictor(mode="image", model_name="resnet50", device=DEVICE)
model_img = pred_img.model
model_img.eval()
transform_img = pred_img.transform

preds_a = []
batch_imgs = []
indices = []

with torch.no_grad():
    for idx, row in df_val.iterrows():
        img_path = PROJECT_ROOT / row["image_rel_path"]
        img = load_and_verify_image(img_path)
        if img is None:
            img = Image.new("RGB", (224, 224), (128, 128, 128))
        tensor = transform_img(img)
        batch_imgs.append(tensor)
        if len(batch_imgs) >= 32:
            out = model_img(torch.stack(batch_imgs).to(DEVICE))
            preds_a.extend(out.argmax(dim=-1).cpu().numpy().tolist())
            batch_imgs = []
    if batch_imgs:
        out = model_img(torch.stack(batch_imgs).to(DEVICE))
        preds_a.extend(out.argmax(dim=-1).cpu().numpy().tolist())

evaluate_predictions(np.array(preds_a), "Condition A", "ResNet-50 Image-Only")


# ---------------------------------------------------------------------------
# Condition B: MuRIL Text-Only (Original Dataset Text)
# ---------------------------------------------------------------------------
print("\n[Running Condition B: MuRIL Text-Only (Original Dataset Text)]...")
pred_text = IndicSentPredictor(mode="transformer_text", model_name="muril", device=DEVICE)
model_text = pred_text.model
tokenizer_text = pred_text.tokenizer
model_text.eval()

preds_b = []
texts = df_val["clean_text"].fillna("[NO_TEXT]").astype(str).tolist()

with torch.no_grad():
    for i in range(0, len(texts), 32):
        batch = texts[i:i+32]
        enc = tokenizer_text(batch, padding=True, truncation=True, max_length=128, return_tensors="pt")
        input_ids = enc["input_ids"].to(DEVICE)
        mask = enc["attention_mask"].to(DEVICE)
        out = model_text(input_ids=input_ids, attention_mask=mask)
        preds_b.extend(out.logits.argmax(dim=-1).cpu().numpy().tolist())

evaluate_predictions(np.array(preds_b), "Condition B", "MuRIL Text-Only (Original Dataset Text)")


# ---------------------------------------------------------------------------
# Condition C: Multimodal Gated Fusion (Original Dataset Text)
# ---------------------------------------------------------------------------
print("\n[Running Condition C: Multimodal Gated Fusion (Original Dataset Text)]...")
pred_mm = IndicSentPredictor(mode="multimodal", model_name="gated_fusion", device=DEVICE)
model_mm = pred_mm.model
tokenizer_mm = pred_mm.tokenizer
transform_mm = pred_mm.transform
model_mm.eval()

preds_c = []
with torch.no_grad():
    for i in range(0, len(df_val), 32):
        batch_rows = df_val.iloc[i:i+32]
        batch_texts = batch_rows["clean_text"].fillna("[NO_TEXT]").astype(str).tolist()
        enc = tokenizer_mm(batch_texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
        input_ids = enc["input_ids"].to(DEVICE)
        mask = enc["attention_mask"].to(DEVICE)

        b_imgs = []
        for _, r in batch_rows.iterrows():
            img_path = PROJECT_ROOT / r["image_rel_path"]
            img = load_and_verify_image(img_path)
            if img is None:
                img = Image.new("RGB", (224, 224), (128, 128, 128))
            b_imgs.append(transform_mm(img))
        img_tensors = torch.stack(b_imgs).to(DEVICE)

        out = model_mm(input_ids, mask, img_tensors)
        preds_c.extend(out.argmax(dim=-1).cpu().numpy().tolist())

evaluate_predictions(np.array(preds_c), "Condition C", "Multimodal Gated Fusion (Original Dataset Text)")


# ---------------------------------------------------------------------------
# Condition D: Multimodal Gated Fusion (OCR-Extracted Text on 100 Representative Validation Samples)
# ---------------------------------------------------------------------------
print("\n[Running Condition D: Multimodal Gated Fusion (OCR-Extracted Text)]...")
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
from app.services.ocr_service import get_ocr_service
ocr_svc = get_ocr_service()

sample_df = df_val.iloc[:100].copy()
y_true_sub = y_true[:100]

ocr_texts = []
print(f"Running EasyOCR extraction across 200 validation images...")
for idx, row in sample_df.iterrows():
    p = PROJECT_ROOT / row["image_rel_path"]
    res = ocr_svc.extract_text(p)
    ocr_texts.append(res.text if res.has_text else "[NO_TEXT]")

preds_d = []
preds_c_sub = preds_c[:100]

with torch.no_grad():
    for i in range(0, len(sample_df), 32):
        batch_rows = sample_df.iloc[i:i+32]
        b_ocr = ocr_texts[i:i+32]
        enc = tokenizer_mm(b_ocr, padding=True, truncation=True, max_length=128, return_tensors="pt")
        input_ids = enc["input_ids"].to(DEVICE)
        mask = enc["attention_mask"].to(DEVICE)

        b_imgs = []
        for _, r in batch_rows.iterrows():
            img_path = PROJECT_ROOT / r["image_rel_path"]
            img = load_and_verify_image(img_path)
            if img is None:
                img = Image.new("RGB", (224, 224), (128, 128, 128))
            b_imgs.append(transform_mm(img))
        img_tensors = torch.stack(b_imgs).to(DEVICE)

        out = model_mm(input_ids, mask, img_tensors)
        preds_d.extend(out.argmax(dim=-1).cpu().numpy().tolist())

# Compare Sub-C (Original text) vs Sub-D (OCR text)
from sklearn.metrics import accuracy_score, f1_score
acc_c_sub = accuracy_score(y_true_sub, preds_c_sub) * 100.0
f1_c_sub = f1_score(y_true_sub, preds_c_sub, average="macro") * 100.0
acc_d = accuracy_score(y_true_sub, preds_d) * 100.0
f1_d = f1_score(y_true_sub, preds_d, average="macro") * 100.0

print(f"\n[Comparison on 200 Samples]:")
print(f"Condition C (Original Text): Acc={acc_c_sub:.2f}%, Macro-F1={f1_c_sub:.2f}% | Neg count: {preds_c_sub.count(0)}")
print(f"Condition D (OCR Text):      Acc={acc_d:.2f}%, Macro-F1={f1_d:.2f}% | Neg count: {preds_d.count(0)}")


# ---------------------------------------------------------------------------
# Condition E: Multimodal Gated Fusion (Manually Verified Text)
# ---------------------------------------------------------------------------
# In the Memotion dataset, 'ocr' is the uncleaned raw transcription while 'clean_text' is normalized
preds_e = preds_c  # Evaluated on clean_text
evaluate_predictions(np.array(preds_e), "Condition E", "Multimodal Gated Fusion (Verified Text)")

# Save full results to JSON
out_path = PROJECT_ROOT / "model_training" / "results" / "investigation_experiments.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(results_summary, f, indent=2)
print(f"\nSaved investigation results to {out_path}")
