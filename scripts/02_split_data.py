import os
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

import config.config as cfg
from src.utils import ensure_dirs, check_file_exists, seed_everything


REQUIRED_COLS = [
    "isic_id",
    "target",
    "patient_id",
]


def check_required_columns(df: pd.DataFrame):
    missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")


def make_patient_level_df(df: pd.DataFrame) -> pd.DataFrame:
    patient_df = (
        df.groupby("patient_id")["target"]
        .max()
        .reset_index()
        .rename(columns={"target": "patient_target"})
    )

    return patient_df


def split_patients(patient_df: pd.DataFrame):
    holdout_size = getattr(cfg, "HOLDOUT_SIZE", 0.1)
    val_size = getattr(cfg, "VAL_SIZE", 0.2)

    train_val_patients, holdout_patients = train_test_split(
        patient_df,
        test_size=holdout_size,
        random_state=cfg.SEED,
        stratify=patient_df["patient_target"],
    )

    # Điều chỉnh để val chiếm khoảng VAL_SIZE trên toàn bộ dataset
    adjusted_val_size = val_size / (1 - holdout_size)

    train_patients, val_patients = train_test_split(
        train_val_patients,
        test_size=adjusted_val_size,
        random_state=cfg.SEED,
        stratify=train_val_patients["patient_target"],
    )

    return train_patients, val_patients, holdout_patients


def build_row_split(
    df: pd.DataFrame,
    patient_split_df: pd.DataFrame,
) -> pd.DataFrame:
    patient_ids = set(patient_split_df["patient_id"])
    split_df = df[df["patient_id"].isin(patient_ids)].copy()
    return split_df.reset_index(drop=True)


def sample_negatives(
    df: pd.DataFrame,
    negative_sample_size: int,
    seed: int,
) -> pd.DataFrame:
    positive_df = df[df["target"] == 1].copy()
    negative_df = df[df["target"] == 0].copy()

    n_neg = min(negative_sample_size, len(negative_df))

    sampled_negative_df = negative_df.sample(
        n=n_neg,
        random_state=seed,
    )

    sampled_df = pd.concat(
        [positive_df, sampled_negative_df],
        axis=0,
    )

    sampled_df = sampled_df.sample(
        frac=1,
        random_state=seed,
    ).reset_index(drop=True)

    return sampled_df


def check_patient_overlap(train_df, val_df, holdout_df):
    train_patients = set(train_df["patient_id"])
    val_patients = set(val_df["patient_id"])
    holdout_patients = set(holdout_df["patient_id"])

    train_val_overlap = train_patients.intersection(val_patients)
    train_holdout_overlap = train_patients.intersection(holdout_patients)
    val_holdout_overlap = val_patients.intersection(holdout_patients)

    if train_val_overlap:
        raise ValueError(f"Train/Val patient overlap: {len(train_val_overlap)}")

    if train_holdout_overlap:
        raise ValueError(f"Train/Holdout patient overlap: {len(train_holdout_overlap)}")

    if val_holdout_overlap:
        raise ValueError(f"Val/Holdout patient overlap: {len(val_holdout_overlap)}")

    print("[OK] No patient overlap between train/val/holdout.")


def print_split_summary(name: str, df: pd.DataFrame):
    print(f"\n========== {name.upper()} ==========")
    print(f"Rows: {len(df)}")
    print(f"Patients: {df['patient_id'].nunique()}")

    print("\nTarget count:")
    print(df["target"].value_counts())

    print("\nTarget ratio:")
    print(df["target"].value_counts(normalize=True))

    print("\nFirst 5 rows:")
    print(df[["isic_id", "target", "patient_id"]].head())


def save_split(df: pd.DataFrame, path: str, name: str):
    df.to_csv(path, index=False)
    print(f"[OK] Saved {name}: {path}")


def main():
    print("========== ISIC 2024 SPLIT DATA ==========")

    seed_everything(cfg.SEED)
    ensure_dirs(cfg.SPLIT_DIR)

    check_file_exists(cfg.TRAIN_CSV, "train metadata CSV")

    df = pd.read_csv(cfg.TRAIN_CSV, low_memory=False)
    check_required_columns(df)

    print(f"Loaded train metadata: {cfg.TRAIN_CSV}")
    print(f"Shape: {df.shape}")

    print("\nOriginal target distribution:")
    print(df["target"].value_counts())

    print("\nOriginal patient count:")
    print(df["patient_id"].nunique())

    patient_df = make_patient_level_df(df)

    print("\nPatient-level target distribution:")
    print(patient_df["patient_target"].value_counts())

    train_patients, val_patients, holdout_patients = split_patients(patient_df)

    full_train_df = build_row_split(df, train_patients)
    full_val_df = build_row_split(df, val_patients)
    full_holdout_df = build_row_split(df, holdout_patients)

    check_patient_overlap(
        full_train_df,
        full_val_df,
        full_holdout_df,
    )

    train_df = sample_negatives(
        full_train_df,
        negative_sample_size=cfg.NEGATIVE_SAMPLE_SIZE,
        seed=cfg.SEED,
    )

    val_df = sample_negatives(
        full_val_df,
        negative_sample_size=cfg.EVAL_NEGATIVE_SAMPLE_SIZE,
        seed=cfg.SEED,
    )

    holdout_df = sample_negatives(
        full_holdout_df,
        negative_sample_size=cfg.EVAL_NEGATIVE_SAMPLE_SIZE,
        seed=cfg.SEED,
    )

    print_split_summary("train", train_df)
    print_split_summary("val", val_df)
    print_split_summary("holdout", holdout_df)

    check_patient_overlap(
        train_df,
        val_df,
        holdout_df,
    )

    save_split(
        train_df,
        cfg.TRAIN_SPLIT_CSV,
        "train split",
    )

    save_split(
        val_df,
        cfg.VAL_SPLIT_CSV,
        "val split",
    )

    save_split(
        holdout_df,
        cfg.HOLDOUT_SPLIT_CSV,
        "holdout split",
    )

    print("\n========== RESULT ==========")
    print("[DONE] Split data completed.")


if __name__ == "__main__":
    main()