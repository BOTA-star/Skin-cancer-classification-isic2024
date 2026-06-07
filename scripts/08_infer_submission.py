import io
import os
import sys
from pathlib import Path

import h5py
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

import config.config as cfg

from src.metadata import (
    load_preprocessor,
    transform_metadata,
    get_metadata_feature_names,
)

from src.model import ISICMultimodalModel

from src.transforms import get_val_transforms

from src.utils import (
    ensure_dirs,
    check_file_exists,
    get_device,
    seed_everything,
)

class ISICTestHDF5Dataset(Dataset):
    def __init__(
        self,
        df,
        meta_features,
        hdf5_path,
        transform=None,
    ):
        self.df = df.reset_index(drop=True)
        self.meta_features = meta_features
        self.hdf5_path = hdf5_path
        self.transform = transform
        self.hdf5_file = h5py.File(hdf5_path, "r")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        isic_id = str(row["isic_id"])

        image_bytes = self.hdf5_file[isic_id][()]
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        if self.transform:
            image = self.transform(image)

        meta = torch.tensor(
            self.meta_features[idx],
            dtype=torch.float32,
        )

        return image, meta, isic_id

    def __del__(self):
        try:
            self.hdf5_file.close()
        except Exception:
            pass

def main():
    print("=== ISIC 2024 INFER SUBMISSION ===")

    seed_everything(cfg.SEED)

    ensure_dirs(cfg.OUTPUT_DIR)

    check_file_exists(
        cfg.TEST_CSV,
        "test metadata CSV",
    )

    check_file_exists(
        cfg.TEST_IMAGE_HDF5,
        "test image HDF5",
    )

    check_file_exists(
        cfg.BEST_MODEL_PATH,
        "best model checkpoint",
    )

    check_file_exists(
        cfg.METADATA_PREPROCESSOR_PATH,
        "metadata preprocessor",
    )

    device = get_device()

    print("Device:", device)

    test_df = pd.read_csv(cfg.TEST_CSV, low_memory=False)

    print("Test shape:", test_df.shape)

    preprocessor = load_preprocessor(
        cfg.METADATA_PREPROCESSOR_PATH,
    )

    # Lấy đúng danh sách cột metadata đã fit trong preprocessor
    num_cols = []
    cat_cols = []

    for name, transformer, cols in preprocessor.transformers_:
        if name == "num":
            num_cols = list(cols)
        elif name == "cat":
            cat_cols = list(cols)

    meta_features = transform_metadata(
        preprocessor,
        test_df,
    )

    feature_names = get_metadata_feature_names(
        preprocessor,
        num_cols,
        cat_cols,
    )

    meta_dim = len(feature_names)

    print("Metadata dim:", meta_dim)

    checkpoint = torch.load(
        cfg.BEST_MODEL_PATH,
        map_location=device,
        weights_only=False,
    )

    backbone = checkpoint.get("backbone", cfg.BACKBONE)
    checkpoint_meta_dim = checkpoint.get("meta_dim", meta_dim)

    print("Checkpoint backbone:", backbone)
    print("Checkpoint meta dim:", checkpoint_meta_dim)

    if checkpoint_meta_dim != meta_dim:
        raise ValueError(
            f"Metadata dim mismatch. "
            f"Checkpoint: {checkpoint_meta_dim}, test: {meta_dim}"
        )

    model = ISICMultimodalModel(
        meta_dim=meta_dim,
        backbone_name=backbone,
        pretrained=False,
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"],
    )

    model.eval()

    test_ds = ISICTestHDF5Dataset(
        df=test_df,
        meta_features=meta_features,
        hdf5_path=cfg.TEST_IMAGE_HDF5,
        transform=get_val_transforms(cfg.IMAGE_SIZE),
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    ids = []
    preds = []

    with torch.no_grad():
        for images, metas, isic_ids in test_loader:
            images = images.to(device)
            metas = metas.to(device)

            logits = model(images, metas)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            ids.extend(list(isic_ids))
            preds.extend(probs.tolist())

    submission = pd.DataFrame({
        "isic_id": ids,
        "target": preds,
    })

    submission.to_csv(
        cfg.SUBMISSION_PATH,
        index=False,
    )

    print("\nSubmission preview:")
    print(submission.head())

    print("\n=== RESULT ===")
    print(f"[OK] Submission saved: {cfg.SUBMISSION_PATH}")
    print("[DONE] Inference completed.")

if __name__ == "__main__":
    main()