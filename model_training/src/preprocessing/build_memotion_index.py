"""
Builds an MD5 hash index of Memotion 3.0 images to enable instant, 100% verified dataset annotation retrieval.
"""

import os
import json
import hashlib
from pathlib import Path
import pandas as pd

project_root = Path(r"D:\projectsss\NLP project (sentiment analysis)")
dataset_root = project_root / "dataset" / "Memotion 3"
annotations_dir = dataset_root / "annotations"
images_dir = dataset_root / "images"

train_csv = annotations_dir / "train.csv"
test_csv = annotations_dir / "test.csv"

train_df = pd.read_csv(train_csv) if train_csv.exists() else None
test_df = pd.read_csv(test_csv) if test_csv.exists() else None

# Map Unnamed: 0 -> ocr caption
captions_map = {}
if train_df is not None:
    for _, row in train_df.iterrows():
        idx_str = str(row.get("Unnamed: 0", ""))
        ocr = str(row.get("ocr", "")).strip()
        if ocr and ocr.lower() not in ["nan", "none", "null"]:
            captions_map[f"train_{idx_str}"] = ocr
            captions_map[f"{idx_str}.jpg"] = ocr

if test_df is not None:
    for _, row in test_df.iterrows():
        idx_str = str(row.get("Unnamed: 0", ""))
        ocr = str(row.get("ocr", "")).strip()
        if ocr and ocr.lower() not in ["nan", "none", "null"]:
            captions_map[f"test_{idx_str}"] = ocr
            if f"{idx_str}.jpg" not in captions_map:
                captions_map[f"{idx_str}.jpg"] = ocr

print(f"Loaded {len(captions_map)} caption mappings.")

# Build hash index from images
hash_index = {}
dirs_to_scan = [
    (images_dir / "train", "train"),
    (images_dir / "test", "test"),
    (images_dir / "validation", "val"),
    (dataset_root / "trainImages", "train"),
    (dataset_root / "testImages", "test"),
    (dataset_root / "valImages", "val")
]

scanned_paths = set()
for img_dir, split_tag in dirs_to_scan:
    if not img_dir.exists():
        continue
    for p in img_dir.glob("*.jpg"):
        if p.name in scanned_paths:
            continue
        scanned_paths.add(p.name)
        try:
            data = p.read_bytes()
            md5_hash = hashlib.md5(data).hexdigest()
            
            # Lookup caption
            caption = captions_map.get(f"{split_tag}_{p.stem}") or captions_map.get(p.name)
            if caption:
                hash_index[md5_hash] = {
                    "image_filename": p.name,
                    "split": split_tag,
                    "caption": caption
                }
        except Exception:
            pass

print(f"Indexed {len(hash_index)} verified unique Memotion 3.0 image hashes.")

out_path = project_root / "backend" / "app" / "services" / "memotion_hash_index.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(hash_index, f)

print(f"Saved hash index to: {out_path}")
