from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

class ImageMetadataDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        image_dir: str | Path,
        target_cols: List[str],
        target_mean: Dict[str, float],
        target_std: Dict[str, float],
        transform=None,
    ):
        self.df = df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.target_cols = list(target_cols)
        self.target_mean = target_mean
        self.target_std = target_std
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def _image_path(self, isic_id: str) -> Path:
        path = self.image_dir / f"{isic_id}.jpg"
        if not path.exists():
            path = self.image_dir / f"{isic_id}.jpeg"
        if not path.exists():
            path = self.image_dir / f"{isic_id}.png"
        return path

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        isic_id = str(row["isic_id"])

        image_path = self._image_path(isic_id)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found for {isic_id}: {image_path}")

        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        y_values = []
        mask_values = []

        for col in self.target_cols:
            value = row[col]
            if pd.isna(value):
                y_values.append(0.0)
                mask_values.append(0.0)
                continue

            mean = float(self.target_mean[col])
            std = float(self.target_std[col])
            if std == 0:
                std = 1.0

            y_values.append((float(value) - mean) / std)
            mask_values.append(1.0)

        y = torch.tensor(y_values, dtype=torch.float32)
        mask = torch.tensor(mask_values, dtype=torch.float32)

        return {
            "image": image,
            "target": y,
            "mask": mask,
            "isic_id": isic_id,
        }
