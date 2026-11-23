"""
IndicSent-M: Improved Text Classification Pipeline
Trains improved TF-IDF + Logistic Regression with:
- Sentiment emoji translation tokens (emoji_fire, emoji_party, etc.)
- Hinglish negation compound binding (bina_error, nahi_hua, etc.)
- FeatureUnion of word n-grams (1, 2) and character n-grams (3, 5)
- Tuned regularization and adjusted class weights
- Evaluates on validation split and 10 Hinglish benchmark cases
"""

import sys
import json
import time
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score, precision_recall_fscore_support

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

project_root = Path(__file__).resolve().parents[2]
src_dir = project_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from preprocessing.clean_text import clean_ocr_text

def run_training():
    splits_dir = project_root / "data_splits"
    checkpoints_dir = project_root / "checkpoints" / "classical"
    best_dir = project_root / "checkpoints" / "best_model"
    results_dir = project_root / "results"

    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    best_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    train_path = splits_dir / "train.csv"
    val_path = splits_dir / "validation.csv"

    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)

    print(f"Loaded Train: {len(df_train)} samples, Validation: {len(df_val)} samples.")

    X_train = [clean_ocr_text(t) for t in df_train["ocr"]]
    y_train = df_train["label_id"].values

    X_val = [clean_ocr_text(t) for t in df_val["ocr"]]
    y_val = df_val["label_id"].values

    # FeatureUnion of Word-level TF-IDF and Character-level n-grams
    union = FeatureUnion([
        ('word', TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.85,
            max_features=12000,
            sublinear_tf=True,
            token_pattern=r'(?u)\b\w+\b|!\b|\?\b'
        )),
        ('char', TfidfVectorizer(
            analyzer='char_wb',
            ngram_range=(3, 5),
            min_df=3,
            max_features=18000,
            sublinear_tf=True
        ))
    ])

    clf = LogisticRegression(
        C=0.4,
        class_weight={0: 1.15, 1: 0.85, 2: 1.15},
        max_iter=1000,
        random_state=42,
        solver='lbfgs'
    )

    pipeline = Pipeline([
        ('feats', union),
        ('clf', clf)
    ])

    print("\nTraining Improved TF-IDF + Logistic Regression...")
    t0 = time.time()
    pipeline.fit(X_train, y_train)
    train_time = time.time() - t0
    print(f"Training completed in {train_time:.2f}s.")

    # Validation evaluation
    t1 = time.time()
    val_preds = pipeline.predict(X_val)
    val_probs = pipeline.predict_proba(X_val)
    val_time = time.time() - t1
    latency_ms = (val_time / len(X_val)) * 1000.0

    acc = float(accuracy_score(y_val, val_preds) * 100.0)
    macro_f1 = float(f1_score(y_val, val_preds, average='macro') * 100.0)
    p_class, r_class, f1_class, support = precision_recall_fscore_support(y_val, val_preds, average=None)
    cm = confusion_matrix(y_val, val_preds).tolist()

    print("\n" + "=" * 65)
    print("VALIDATION EVALUATION METRICS (Improved Classical Model)")
    print("=" * 65)
    print(f"Accuracy:  {acc:.2f}% (Historical Baseline: 43.79%)")
    print(f"Macro-F1:  {macro_f1:.2f}% (Historical Baseline: 43.14%)")
    print(f"Negative (0) -> Precision: {p_class[0]*100:.1f}%, Recall: {r_class[0]*100:.1f}%, F1: {f1_class[0]*100:.1f}%")
    print(f"Neutral  (1) -> Precision: {p_class[1]*100:.1f}%, Recall: {r_class[1]*100:.1f}%, F1: {f1_class[1]*100:.1f}%")
    print(f"Positive (2) -> Precision: {p_class[2]*100:.1f}%, Recall: {r_class[2]*100:.1f}%, F1: {f1_class[2]*100:.1f}%")
    print(f"Confusion Matrix (Rows=True, Cols=Pred):\n{np.array(cm)}")

    # 10 Hinglish benchmark examples
    hinglish_cases = [
        ("Code bina error ke run ho gaya! 🔥", "Positive", 2),
        ("Finally assignment complete ho gaya! 😍", "Positive", 2),
        ("Aaj exam bahut achha gaya!", "Positive", 2),
        ("Teacher ne bola well done beta! 🎉", "Positive", 2),
        ("Kal exam hai.", "Neutral", 1),
        ("Code phir se error de raha hai 😭", "Negative", 0),
        ("Project successfully complete kar diya!", "Positive", 2),
        ("Mera result kharab aa gaya.", "Negative", 0),
        ("Aaj kuch khaas nahi hua.", "Context-dependent", 1),
        ("Assignment abhi tak complete nahi hua.", "Negative", 0)
    ]

    labels_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    case_results = []
    print("\n" + "=" * 80)
    print("HINGLISH EVALUATION BENCHMARK EXAMPLES")
    print("=" * 80)
    for text, exp_str, exp_id in hinglish_cases:
        cleaned = clean_ocr_text(text)
        pred_id = int(pipeline.predict([cleaned])[0])
        probs = pipeline.predict_proba([cleaned])[0]
        status = "MATCH" if (pred_id == exp_id or (exp_str == "Context-dependent" and pred_id in [1, 2])) else "DIFF"
        res_item = {
            "text": text,
            "expected": exp_str,
            "predicted": labels_map[pred_id],
            "predicted_id": pred_id,
            "probabilities": {
                "Negative": round(float(probs[0]), 3),
                "Neutral": round(float(probs[1]), 3),
                "Positive": round(float(probs[2]), 3)
            },
            "status": status
        }
        case_results.append(res_item)
        print(f"{text:<42} | Exp: {exp_str:<18} | Pred: {labels_map[pred_id]:<9} | Probs: [{probs[0]:.2f}, {probs[1]:.2f}, {probs[2]:.2f}] [{status}]")

    # Save checkpoints
    ckpt_path = checkpoints_dir / "tf_idf_plus_logistic_regression.joblib"
    best_path = best_dir / "model.joblib"
    joblib.dump(pipeline, ckpt_path)
    joblib.dump(pipeline, best_path)
    print(f"\nSaved updated checkpoint: {ckpt_path}")
    print(f"Saved updated best model: {best_path}")

    # Save results to json
    exp_log = {
        "model_name": "TF-IDF + Logistic Regression (Improved FeatureUnion)",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "training_time_sec": round(train_time, 2),
        "inference_latency_ms": round(latency_ms, 2),
        "validation_metrics": {
            "accuracy": acc,
            "macro_f1": macro_f1,
            "negative_f1": round(float(f1_class[0] * 100.0), 2),
            "neutral_f1": round(float(f1_class[1] * 100.0), 2),
            "positive_f1": round(float(f1_class[2] * 100.0), 2),
            "confusion_matrix": cm
        },
        "hinglish_benchmark_results": case_results
    }

    with open(results_dir / "text_model_experiments.json", "w", encoding="utf-8") as f:
        json.dump(exp_log, f, indent=2)
    print(f"Saved experiment report to: {results_dir / 'text_model_experiments.json'}")

if __name__ == "__main__":
    run_training()
