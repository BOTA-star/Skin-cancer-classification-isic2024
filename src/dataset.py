import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image


class ISICDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform = None):
        self.df = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_id = row["isic_id"]
        label = row["malignant"]

        img_path = os.path.join(self.image_dir, f"{img_id}.jpg")
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label