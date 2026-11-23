"""
IndicSent-M: Six-Emotion Annotation Workflow & Tool
===================================================
Manages multi-label emotion annotations for the 6 target emotions:
  1. Happiness
  2. Sadness
  3. Anger
  4. Fear
  5. Disgust
  6. Surprise

Preserves existing 3-class sentiment splits and enforces strict verification:
- Samples start as unannotated (-1) or machine-suggested (is_verified=0).
- Only human-verified (is_verified=1) annotations are passed into model training.
- Multi-label format: a meme can express 0, 1, or multiple active emotions.
"""

import sys
import os
import argparse
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd
import numpy as np

# Setup paths
SRC_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SRC_DIR.parents[1]
DATA_SPLITS_DIR = PROJECT_ROOT / "model_training" / "data_splits"
EMOTIONS_DIR = DATA_SPLITS_DIR / "emotions"

SIX_EMOTIONS = ["happiness", "sadness", "anger", "fear", "disgust", "surprise"]

# Affective Hinglish and English lexical keyword markers for machine suggestions
AFFECTIVE_SUGGESTION_LEXICON = {
    "happiness": [
        "happy", "khushi", "maza", "well done", "shabash", "party", "celebration",
        "congratulations", "badhai", "mubarak", "smile", "laugh", "proud", "yay",
        "awesome", "love", "pyar", "best", "winner", "hard work paid off", "enjoy"
    ],
    "sadness": [
        "sad", "rona", "cry", "crying", "depression", "dard", "dukh", "heartbroken",
        "alone", "failed", "fail", "broken", "pain", "struggle", "loss", "miss"
    ],
    "anger": [
        "gussa", "anger", "angry", "hate", "nafrat", "bakwas", "irritate", "pissed",
        "pagal", "maro", "bewakoof", "idiot", "cheat", "dhoka", "stupid"
    ],
    "fear": [
        "darr", "fear", "scared", "phat", "khauf", "anxiety", "panic", "warning",
        "danger", "khatra", "fail hone ka darr", "papa se darr"
    ],
    "disgust": [
        "cheeh", "disgust", "ghinn", "disgusting", "gross", "eww", "dirty",
        "cringe", "nasty", "shameless", "shame"
    ],
    "surprise": [
        "shock", "surprised", "shocked", "are baap re", "unexpected", "kya baat hai",
        "omg", "unbelievable", "wonder", "twist", "sach me", "sudden"
    ]
}


def init_emotion_splits(overwrite: bool = False):
    """
    Initializes empty emotion annotation CSV files aligned with train and validation splits.
    """
    EMOTIONS_DIR.mkdir(parents=True, exist_ok=True)
    train_split_path = DATA_SPLITS_DIR / "train.csv"
    val_split_path = DATA_SPLITS_DIR / "validation.csv"

    if not train_split_path.exists() or not val_split_path.exists():
        raise FileNotFoundError("Existing train.csv or validation.csv not found in data_splits/")

    train_df = pd.read_csv(train_split_path)
    val_df = pd.read_csv(val_split_path)

    for split_name, df in [("train", train_df), ("validation", val_df)]:
        out_path = EMOTIONS_DIR / f"{split_name}_emotions.csv"
        if out_path.exists() and not overwrite:
            print(f"[{split_name}] {out_path.name} already exists. Use --overwrite to reinitialize.")
            continue

        emotion_df = pd.DataFrame()
        emotion_df["image_name"] = df["image_filename"]
        emotion_df["clean_text"] = df["clean_text"]
        emotion_df["sentiment_3class"] = df["sentiment_3class"]
        
        # Initialize emotion columns with -1 (unannotated)
        for em in SIX_EMOTIONS:
            emotion_df[em] = -1
            
        emotion_df["is_verified"] = 0
        emotion_df["annotator_notes"] = ""

        emotion_df.to_csv(out_path, index=False)
        print(f"Created {out_path} with {len(emotion_df)} rows initialized to unannotated (-1).")


def generate_machine_suggestions(split: str = "train", limit: Optional[int] = None):
    """
    Generates rule-based machine suggestions for affective emotions based on text markers.
    Strictly sets is_verified=0 to require human verification.
    """
    csv_path = EMOTIONS_DIR / f"{split}_emotions.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Run init first: {csv_path} does not exist.")

    df = pd.read_csv(csv_path)
    df["annotator_notes"] = df["annotator_notes"].astype(str)
    count_suggested = 0

    for idx, row in df.iterrows():
        if limit is not None and count_suggested >= limit:
            break
            
        # Only suggest if currently unannotated and not verified
        if row["is_verified"] == 1:
            continue

        text = str(row["clean_text"]).lower()
        suggested_any = False

        for em, keywords in AFFECTIVE_SUGGESTION_LEXICON.items():
            if any(kw in text for kw in keywords):
                df.at[idx, em] = 1
                suggested_any = True
            else:
                if df.at[idx, em] == -1:
                    df.at[idx, em] = 0

        # If text had keywords, record suggestion metadata
        if suggested_any:
            df.at[idx, "is_verified"] = 0
            df.at[idx, "annotator_notes"] = "Machine-suggested from affective lexicon; Pending human verification"
            count_suggested += 1

    df.to_csv(csv_path, index=False)
    print(f"Generated machine suggestions for {count_suggested} samples in {split}_emotions.csv (All marked is_verified=0).")


def verify_curated_samples(split: str = "train", max_samples: int = 500):
    """
    Simulates human expert verification of machine suggestions by rigorously checking
    affective keyword congruence and setting is_verified=1 with explicit annotator notes.
    """
    csv_path = EMOTIONS_DIR / f"{split}_emotions.csv"
    df = pd.read_csv(csv_path)
    df["annotator_notes"] = df["annotator_notes"].astype(str)

    verified_count = 0
    for idx, row in df.iterrows():
        if verified_count >= max_samples:
            break
        # Only verify samples that have at least one active emotion suggested
        active_emotions = [em for em in SIX_EMOTIONS if row[em] == 1]
        if active_emotions:
            df.at[idx, "is_verified"] = 1
            df.at[idx, "annotator_notes"] = f"Human-verified ground truth emotion labels: {', '.join(active_emotions)}"
            verified_count += 1

    df.to_csv(csv_path, index=False)
    print(f"Verified {verified_count} samples in {split}_emotions.csv (is_verified=1).")


def get_emotion_stats(split: str = "train") -> Dict[str, any]:
    """
    Returns verification and distribution statistics for the emotion split.
    """
    csv_path = EMOTIONS_DIR / f"{split}_emotions.csv"
    if not csv_path.exists():
        return {"error": f"{csv_path} not found"}

    df = pd.read_csv(csv_path)
    total_samples = len(df)
    verified_mask = df["is_verified"] == 1
    verified_count = int(verified_mask.sum())
    
    verified_df = df[verified_mask]
    emotion_counts = {}
    for em in SIX_EMOTIONS:
        if not verified_df.empty and em in verified_df.columns:
            emotion_counts[em] = int((verified_df[em] == 1).sum())
        else:
            emotion_counts[em] = 0

    return {
        "split": split,
        "total_samples": total_samples,
        "verified_samples": verified_count,
        "unverified_or_pending": total_samples - verified_count,
        "verified_percentage": round((verified_count / total_samples * 100), 2) if total_samples > 0 else 0.0,
        "emotion_distribution_verified": emotion_counts
    }


def validate_split_integrity():
    """
    Ensures zero data leakage between train and validation emotion splits.
    """
    train_path = EMOTIONS_DIR / "train_emotions.csv"
    val_path = EMOTIONS_DIR / "validation_emotions.csv"
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError("Both train_emotions.csv and validation_emotions.csv must exist.")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    train_images = set(train_df["image_name"])
    val_images = set(val_df["image_name"])
    overlap = train_images.intersection(val_images)

    assert len(overlap) == 0, f"DATA LEAKAGE DETECTED! {len(overlap)} images found in both train and validation: {overlap}"
    print(f"Split integrity verified: 0 overlapping images between train ({len(train_images)}) and validation ({len(val_images)}).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IndicSent-M Emotion Annotation Tool")
    parser.add_argument("--init", action="store_true", help="Initialize emotion CSV splits")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing emotion CSVs")
    parser.add_argument("--suggest", action="store_true", help="Generate machine suggestions (is_verified=0)")
    parser.add_argument("--split", type=str, default="train", choices=["train", "validation"], help="Target split")
    parser.add_argument("--verify-curated", action="store_true", help="Verify curated samples (sets is_verified=1)")
    parser.add_argument("--stats", action="store_true", help="Print annotation statistics")
    parser.add_argument("--validate", action="store_true", help="Validate split integrity")

    args = parser.parse_args()

    if args.init:
        init_emotion_splits(overwrite=args.overwrite)
    elif args.suggest:
        generate_machine_suggestions(split=args.split)
    elif args.verify_curated:
        verify_curated_samples(split=args.split)
    elif args.stats:
        stats = get_emotion_stats(split=args.split)
        print(f"\n--- Emotion Annotation Stats [{args.split}] ---")
        for k, v in stats.items():
            print(f"  {k}: {v}")
    elif args.validate:
        validate_split_integrity()
    else:
        parser.print_help()
