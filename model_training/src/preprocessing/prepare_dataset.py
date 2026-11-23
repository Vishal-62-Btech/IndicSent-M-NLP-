"""
IndicSent-M: Dataset Preparation and Stratified Splitting Module
Cleans text, maps 3-class sentiment labels, matches verified images,
and creates the official 80:20 stratified train-validation split with fixed seed=42.
"""

import sys
import json
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# Ensure local imports work
project_root = Path(__file__).resolve().parents[3]
src_dir = Path(__file__).resolve().parents[1]
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from preprocessing.clean_text import clean_ocr_text
from preprocessing.preprocess_images import load_and_verify_image

LABEL_MAPPING_3CLASS = {
    "negative": "Negative",
    "very_negative": "Negative",
    "neutral": "Neutral",
    "positive": "Positive",
    "very_positive": "Positive"
}

LABEL_TO_ID = {
    "Negative": 0,
    "Neutral": 1,
    "Positive": 2
}

ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}

def prepare_splits(
    base_dir: Path = project_root,
    test_size: float = 0.20,
    random_seed: int = 42
):
    print("=" * 60)
    print("IndicSent-M: Preparing Datasets and Stratified Splits")
    print("=" * 60)

    dataset_dir = base_dir / "dataset" / "Memotion 3"
    train_csv_path = dataset_dir / "annotations" / "train.csv"
    test_csv_path = dataset_dir / "annotations" / "test.csv"
    train_images_dir = dataset_dir / "images" / "train"
    test_images_dir = dataset_dir / "images" / "test"

    output_splits_dir = base_dir / "model_training" / "data_splits"
    output_splits_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Training Data
    print(f"Loading training annotations from: {train_csv_path}")
    df_train_raw = pd.read_csv(train_csv_path)
    print(f"Loaded {len(df_train_raw)} training records.")

    # Validate columns
    required_cols = ["Unnamed: 0", "overall", "ocr"]
    for c in required_cols:
        assert c in df_train_raw.columns, f"Missing column {c} in train.csv"

    # Map labels
    df_train_raw["sentiment_3class"] = df_train_raw["overall"].map(LABEL_MAPPING_3CLASS)
    assert df_train_raw["sentiment_3class"].isnull().sum() == 0, "Unmapped sentiment label found!"
    df_train_raw["label_id"] = df_train_raw["sentiment_3class"].map(LABEL_TO_ID)

    # Clean text (preserving original ocr column intact)
    print("Cleaning OCR text...")
    df_train_raw["clean_text"] = df_train_raw["ocr"].apply(clean_ocr_text)

    # Image matching
    print("Verifying image paths...")
    df_train_raw["image_id"] = df_train_raw["Unnamed: 0"].astype(int)
    df_train_raw["image_filename"] = df_train_raw["image_id"].apply(lambda x: f"{x}.jpg")
    df_train_raw["image_rel_path"] = df_train_raw["image_filename"].apply(
        lambda f: f"dataset/Memotion 3/images/train/{f}"
    )

    # Verify every image exists on disk
    missing_images = []
    for idx, row in df_train_raw.iterrows():
        full_img_path = base_dir / row["image_rel_path"]
        if not full_img_path.exists():
            missing_images.append(row["image_id"])

    assert len(missing_images) == 0, f"Found {len(missing_images)} missing training images!"
    print(f"Verified all {len(df_train_raw)} training images exist on disk.")

    # 2. Stratified 80:20 Split
    print(f"\nCreating 80:20 stratified split (seed={random_seed})...")
    train_df, val_df = train_test_split(
        df_train_raw,
        test_size=test_size,
        random_state=random_seed,
        stratify=df_train_raw["sentiment_3class"]
    )

    # Reset indices
    train_df = train_df.copy().reset_index(drop=True)
    val_df = val_df.copy().reset_index(drop=True)

    print(f"Train split size: {len(train_df)} ({len(train_df)/len(df_train_raw)*100:.1f}%)")
    print(f"Validation split size: {len(val_df)} ({len(val_df)/len(df_train_raw)*100:.1f}%)")

    print("\nClass distribution in Training split:")
    print(train_df["sentiment_3class"].value_counts())
    print(train_df["sentiment_3class"].value_counts(normalize=True) * 100)

    print("\nClass distribution in Validation split:")
    print(val_df["sentiment_3class"].value_counts())
    print(val_df["sentiment_3class"].value_counts(normalize=True) * 100)

    # 3. Load & Process Official Unlabeled Test Data
    print(f"\nLoading test annotations from: {test_csv_path}")
    df_test_raw = pd.read_csv(test_csv_path)
    df_test_raw["clean_text"] = df_test_raw["ocr"].apply(clean_ocr_text)
    df_test_raw["image_id"] = df_test_raw["Unnamed: 0"].astype(int)
    df_test_raw["image_filename"] = df_test_raw["image_id"].apply(lambda x: f"{x}.jpg")
    df_test_raw["image_rel_path"] = df_test_raw["image_filename"].apply(
        lambda f: f"dataset/Memotion 3/images/test/{f}"
    )

    # Verify test images exist
    missing_test_images = []
    for idx, row in df_test_raw.iterrows():
        full_img_path = base_dir / row["image_rel_path"]
        if not full_img_path.exists():
            missing_test_images.append(row["image_id"])
    assert len(missing_test_images) == 0, f"Found {len(missing_test_images)} missing test images!"
    print(f"Verified all {len(df_test_raw)} test images exist on disk.")

    # 4. Save splits to disk
    train_out_path = output_splits_dir / "train.csv"
    val_out_path = output_splits_dir / "validation.csv"
    test_out_path = output_splits_dir / "test_unlabeled.csv"
    indices_out_path = output_splits_dir / "split_indices.json"
    label_map_path = output_splits_dir / "label_mapping.json"

    train_df.to_csv(train_out_path, index=False)
    val_df.to_csv(val_out_path, index=False)
    df_test_raw.to_csv(test_out_path, index=False)

    split_indices = {
        "random_seed": random_seed,
        "total_labeled_samples": len(df_train_raw),
        "train_indices_image_ids": train_df["image_id"].tolist(),
        "val_indices_image_ids": val_df["image_id"].tolist(),
        "train_count": len(train_df),
        "val_count": len(val_df),
        "test_count": len(df_test_raw)
    }

    with open(indices_out_path, "w", encoding="utf-8") as f:
        json.dump(split_indices, f, indent=2)

    with open(label_map_path, "w", encoding="utf-8") as f:
        json.dump({
            "label_to_id": LABEL_TO_ID,
            "id_to_label": ID_TO_LABEL,
            "original_to_3class": LABEL_MAPPING_3CLASS
        }, f, indent=2)

    print("\nSaved files:")
    print(f"  - {train_out_path} ({len(train_df)} rows)")
    print(f"  - {val_out_path} ({len(val_df)} rows)")
    print(f"  - {test_out_path} ({len(df_test_raw)} rows)")
    print(f"  - {indices_out_path}")
    print(f"  - {label_map_path}")
    print("=" * 60)
    print("Dataset preparation complete.")
    print("=" * 60)

if __name__ == "__main__":
    prepare_splits()
