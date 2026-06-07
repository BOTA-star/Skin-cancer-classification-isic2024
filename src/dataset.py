import os
from typing import List, Optional

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

class ISICMultimodalDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        image_dir: str,
        meta_cols: List[str],
        transform=None,
        has_label: bool = True,
    ):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.meta_cols = meta_cols
        self.transform = transform
        self.has_label = has_label

        required_cols = ["isic_id"] + meta_cols

        if has_label:
            required_cols.append("target")

        missing_cols = [c for c in required_cols if c not in self.df.columns]

        if missing_cols:
            raise ValueError(f"Dataset missing columns: {missing_cols}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        isic_id = str(row["isic_id"])
        img_path = os.path.join(self.image_dir, isic_id + ".jpg")

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        meta_values = row[self.meta_cols].values.astype("float32")
        meta = torch.tensor(meta_values, dtype=torch.float32)

        if self.has_label:
            label = torch.tensor(row["target"], dtype=torch.float32)
            return image, meta, label

        return image, meta, isic_id