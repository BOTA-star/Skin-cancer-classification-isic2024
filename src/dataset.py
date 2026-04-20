import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import torch

class ISICDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

        self.image_ids = self.df["isic_id"].values
        self.labels = self.df["target"].values

        self.features = self.df.drop(columns=["isic_id", "target"]).values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # ---- IMAGE ----
        img_id = self.image_ids[idx]
        img_path = os.path.join(self.image_dir, f"{img_id}.jpg")

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # ---- TABULAR ----
        tabular = torch.tensor(self.features[idx], dtype=torch.float32)

        # ---- LABEL ----
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        return image, tabular, label