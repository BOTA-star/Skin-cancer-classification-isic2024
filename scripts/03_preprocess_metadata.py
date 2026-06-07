import os
import sys
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

import config.config as cfg

from src.metadata import (
    resolve_existing_columns,
    fit_transform_metadata,
    transform_metadata,
    get_metadata_feature_names,
    build_processed_dataframe,
    save_preprocessor,
    save_metadata_info,
)

from src.utils import (
    ensure_dirs,
    check_file_exists,
    seed_everything,
)


METADATA_INFO_PATH = os.path.join(
    cfg.ARTIFACT_DIR,
    "metadata_features.json",
)


def load_split(path: str, name: str) -> pd.DataFrame:
    check_file_exists(path, f"{name} split CSV")

    df = pd.read_csv(path, low_memory=False)

    print(f"\n========== LOAD {name.upper()} ==========")
    print(f"Path: {path}")
    print(f"Shape: {df.shape}")

    required_cols = [
        "isic_id",
        "patient_id",
        "target",
    ]

    missing_cols = [c for c in required_cols if c not in df.columns]

    if missing_cols:
        raise ValueError(f"{name} split missing columns: {missing_cols}")

    print("Target count:")
    print(df["target"].value_counts())

    return df


def save_processed(df: pd.DataFrame, path: str, name: str):
    df.to_csv(path, index=False)

    print(f"\n[OK] Saved {name} processed CSV:")
    print(path)
    print(f"Shape: {df.shape}")


def main():
    print("========== ISIC 2024 METADATA PREPROCESSING ==========")

    seed_everything(cfg.SEED)
    ensure_dirs(
        cfg.PROCESSED_DIR,
        cfg.ARTIFACT_DIR,
    )

    train_df = load_split(
        cfg.TRAIN_SPLIT_CSV,
        "train",
    )

    val_df = load_split(
        cfg.VAL_SPLIT_CSV,
        "val",
    )

    holdout_df = load_split(
        cfg.HOLDOUT_SPLIT_CSV,
        "holdout",
    )

    num_cols, cat_cols = resolve_existing_columns(
        train_df,
        cfg.NUM_COLS,
        cfg.CAT_COLS,
    )

    print("\n========== SELECTED METADATA COLUMNS ==========")
    print(f"Numeric columns ({len(num_cols)}):")
    print(num_cols)

    print(f"\nCategorical columns ({len(cat_cols)}):")
    print(cat_cols)

    if len(num_cols) == 0 and len(cat_cols) == 0:
        raise ValueError("No metadata columns found for preprocessing.")

    print("\n========== FIT PREPROCESSOR ON TRAIN ONLY ==========")

    preprocessor, train_features = fit_transform_metadata(
        train_df,
        num_cols,
        cat_cols,
    )

    val_features = transform_metadata(
        preprocessor,
        val_df,
    )

    holdout_features = transform_metadata(
        preprocessor,
        holdout_df,
    )

    feature_names = get_metadata_feature_names(
        preprocessor,
        num_cols,
        cat_cols,
    )

    print(f"Metadata feature dim: {len(feature_names)}")
    print("First 20 feature names:")
    print(feature_names[:20])

    train_processed = build_processed_dataframe(
        train_df,
        train_features,
        feature_names,
    )

    val_processed = build_processed_dataframe(
        val_df,
        val_features,
        feature_names,
    )

    holdout_processed = build_processed_dataframe(
        holdout_df,
        holdout_features,
        feature_names,
    )

    save_processed(
        train_processed,
        cfg.TRAIN_PROCESSED_CSV,
        "train",
    )

    save_processed(
        val_processed,
        cfg.VAL_PROCESSED_CSV,
        "val",
    )

    save_processed(
        holdout_processed,
        cfg.HOLDOUT_PROCESSED_CSV,
        "holdout",
    )

    save_preprocessor(
        preprocessor,
        cfg.METADATA_PREPROCESSOR_PATH,
    )

    print("\n[OK] Saved metadata preprocessor:")
    print(cfg.METADATA_PREPROCESSOR_PATH)

    save_metadata_info(
        METADATA_INFO_PATH,
        num_cols,
        cat_cols,
        feature_names,
    )

    print("\n[OK] Saved metadata feature info:")
    print(METADATA_INFO_PATH)

    print("\n========== FINAL CHECK ==========")
    print("Train processed shape:", train_processed.shape)
    print("Val processed shape:", val_processed.shape)
    print("Holdout processed shape:", holdout_processed.shape)

    print("\nTrain processed columns sample:")
    print(train_processed.columns[:20].tolist())

    print("\n========== RESULT ==========")
    print("[DONE] Metadata preprocessing completed.")


if __name__ == "__main__":
    main()