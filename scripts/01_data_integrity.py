import os
import sys
import zipfile
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from config.config import (
    TRAIN_CSV,
    TEST_CSV,
    TRAIN_IMAGE_ZIP,
    TRAIN_IMAGE_DIR,
    DATA_DIR,
    WORK_DIR,
)
from src.utils import ensure_dirs, check_file_exists

REQUIRED_TRAIN_COLS = [
    "isic_id",
    "target",
    "patient_id",
]

OPTIONAL_META_COLS = [
    "age_approx",
    "sex",
    "anatom_site_general",
    "clin_size_long_diam_mm",
]

def unzip_training_images_if_needed():
    if os.path.exists(TRAIN_IMAGE_DIR) and len(os.listdir(TRAIN_IMAGE_DIR)) > 0:
        print(f"[OK] Training image directory exists: {TRAIN_IMAGE_DIR}")
        return

    if not os.path.exists(TRAIN_IMAGE_ZIP):
        raise FileNotFoundError(
            f"Training image folder not found and zip file is missing:\n"
            f"- image dir: {TRAIN_IMAGE_DIR}\n"
            f"- zip file : {TRAIN_IMAGE_ZIP}"
        )

    print("[INFO] Training image directory not found or empty.")
    print(f"[INFO] Unzipping: {TRAIN_IMAGE_ZIP}")
    print(f"[INFO] To      : {WORK_DIR}")

    ensure_dirs(WORK_DIR)

    with zipfile.ZipFile(TRAIN_IMAGE_ZIP, "r") as zip_ref:
        zip_ref.extractall(WORK_DIR)

    if not os.path.exists(TRAIN_IMAGE_DIR):
        raise FileNotFoundError(
            f"Unzip finished but expected folder not found: {TRAIN_IMAGE_DIR}"
        )

    print("[OK] Unzip completed.")

def check_metadata_file():
    check_file_exists(TRAIN_CSV, "train metadata CSV")

    print("\n=== TRAIN METADATA ===")
    print(f"Path: {TRAIN_CSV}")

    df = pd.read_csv(TRAIN_CSV, low_memory=False)

    print(f"Shape: {df.shape}")
    print("\nColumns:")
    print(list(df.columns))

    missing_cols = [c for c in REQUIRED_TRAIN_COLS if c not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    print("\n[OK] Required columns exist:")
    print(REQUIRED_TRAIN_COLS)

    available_meta_cols = [c for c in OPTIONAL_META_COLS if c in df.columns]
    missing_meta_cols = [c for c in OPTIONAL_META_COLS if c not in df.columns]

    print("\nAvailable metadata columns:")
    print(available_meta_cols)

    if missing_meta_cols:
        print("\nMissing optional metadata columns:")
        print(missing_meta_cols)

    print("\nFirst 5 rows:")
    print(df[["isic_id", "target", "patient_id"]].head())

    print("\nTarget distribution:")
    print(df["target"].value_counts(dropna=False))

    print("\nTarget ratio:")
    print(df["target"].value_counts(normalize=True, dropna=False))

    print("\nPatient count:")
    print(df["patient_id"].nunique())

    print("\nSample isic_id:")
    print(df["isic_id"].head(10).tolist())

    bad_ids = df[~df["isic_id"].astype(str).str.startswith("ISIC_")]

    if len(bad_ids) > 0:
        print("\n[WARNING] Some isic_id values do not start with 'ISIC_'.")
        print("Sample bad ids:")
        print(bad_ids["isic_id"].head(10).tolist())
    else:
        print("\n[OK] isic_id format looks correct. Example: ISIC_xxxxxxx")

    return df

def check_training_images(df):
    print("\n=== TRAINING IMAGES ===")

    unzip_training_images_if_needed()

    image_files = [
        f for f in os.listdir(TRAIN_IMAGE_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    print(f"Image directory: {TRAIN_IMAGE_DIR}")
    print(f"Number of image files: {len(image_files)}")

    if len(image_files) == 0:
        raise ValueError("No image files found in training image directory.")

    print("\nSample image files:")
    print(image_files[:10])

    print("\nChecking image paths from metadata...")

    sample_df = df.sample(
        min(1000, len(df)),
        random_state=42
    )

    missing = []

    for isic_id in sample_df["isic_id"]:
        image_path = os.path.join(TRAIN_IMAGE_DIR, str(isic_id) + ".jpg")

        if not os.path.exists(image_path):
            missing.append(image_path)

    if missing:
        print("\n[WARNING] Missing images in sample check.")
        print(f"Missing count in sample: {len(missing)}")
        print("Sample missing paths:")
        print(missing[:10])
    else:
        print("[OK] Sample image path check passed.")

    first_id = df.iloc[0]["isic_id"]
    first_path = os.path.join(TRAIN_IMAGE_DIR, str(first_id) + ".jpg")

    print("\nFirst metadata image path:")
    print(first_path)

    if os.path.exists(first_path):
        print("[OK] First metadata image exists.")
    else:
        print("[WARNING] First metadata image does not exist.")


def check_test_metadata():
    print("\n=== TEST METADATA ===")

    if not os.path.exists(TEST_CSV):
        print(f"[WARNING] Test CSV not found: {TEST_CSV}")
        print("Skip test metadata check.")
        return

    test_df = pd.read_csv(TEST_CSV, low_memory=False)

    print(f"Path: {TEST_CSV}")
    print(f"Shape: {test_df.shape}")

    print("\nColumns:")
    print(list(test_df.columns))

    if "isic_id" not in test_df.columns:
        raise ValueError("Missing required column in test metadata: isic_id")

    print("\nFirst 5 test rows:")
    print(test_df.head())

    print("\nSample test isic_id:")
    print(test_df["isic_id"].head(10).tolist())

def main():
    print("=== ISIC 2024 DATA INTEGRITY CHECK ===")
    print(f"DATA_DIR : {DATA_DIR}")
    print(f"WORK_DIR : {WORK_DIR}")

    df = check_metadata_file()
    check_training_images(df)
    check_test_metadata()

    print("\n=== RESULT ===")
    print("[DONE] Data integrity check completed.")

if __name__ == "__main__":
    main()