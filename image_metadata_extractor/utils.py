import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from config import NEVER_TARGET_COLS, DEFAULT_TARGET_PREFIX, WEAK_IMAGE_TARGETS


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def select_image_target_cols(
    df: pd.DataFrame,
    target_prefix: str = DEFAULT_TARGET_PREFIX,
    include_weak_targets: bool = True,
):
    cols = []
    for col in df.columns:
        if col in NEVER_TARGET_COLS:
            continue
        if not col.startswith(target_prefix):
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        if (not include_weak_targets) and (col in WEAK_IMAGE_TARGETS):
            continue
        cols.append(col)

    if not cols:
        raise ValueError("No image-derived target columns found.")

    return cols


def compute_target_stats(df: pd.DataFrame, target_cols):
    stats = {
        "mean": {},
        "std": {},
        "missing_rate": {},
    }

    for col in target_cols:
        values = pd.to_numeric(df[col], errors="coerce")
        mean = float(values.mean())
        std = float(values.std())
        if not np.isfinite(mean):
            mean = 0.0
        if (not np.isfinite(std)) or std == 0:
            std = 1.0

        stats["mean"][col] = mean
        stats["std"][col] = std
        stats["missing_rate"][col] = float(values.isna().mean())

    return stats


def inverse_scale(pred_scaled: np.ndarray, target_cols, target_stats):
    mean = np.array([target_stats["mean"][c] for c in target_cols], dtype=np.float32)
    std = np.array([target_stats["std"][c] for c in target_cols], dtype=np.float32)
    return pred_scaled * std + mean


def sample_dataframe(df: pd.DataFrame, sample_size: int | None, seed: int = 42):
    if sample_size is None or sample_size <= 0 or sample_size >= len(df):
        return df.reset_index(drop=True)

    return df.sample(n=sample_size, random_state=seed).reset_index(drop=True)
