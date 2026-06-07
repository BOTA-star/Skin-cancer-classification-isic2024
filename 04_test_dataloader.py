import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent
sys.path.append(str(ROOT_DIR))

import config as cfg

from dataloader import (
    create_train_loader,
    create_eval_loader,
    get_meta_columns,
)

from utils import (
    check_file_exists,
    check_dir_exists,
    seed_everything,
)

def main():
    print("=== ISIC 2024 TEST MULTIMODAL DATALOADER ===")

    seed_everything(cfg.SEED)

    check_file_exists(
        cfg.TRAIN_PROCESSED_CSV,
        "train processed CSV",
    )

    check_file_exists(
        cfg.VAL_PROCESSED_CSV,
        "val processed CSV",
    )

    check_dir_exists(
        cfg.TRAIN_IMAGE_DIR,
        "training image directory",
    )

    train_df = pd.read_csv(cfg.TRAIN_PROCESSED_CSV)
    val_df = pd.read_csv(cfg.VAL_PROCESSED_CSV)

    print("\nTrain processed shape:", train_df.shape)
    print("Val processed shape:", val_df.shape)

    meta_cols = get_meta_columns(train_df)

    print("\nMetadata columns:", len(meta_cols))
    print("First 10 metadata columns:")
    print(meta_cols[:10])

    train_loader, train_meta_cols = create_train_loader(
        train_df=train_df,
        image_dir=cfg.TRAIN_IMAGE_DIR,
        batch_size=cfg.BATCH_SIZE,
        image_size=cfg.IMAGE_SIZE,
        num_workers=cfg.NUM_WORKERS,
        use_weighted_sampler=True,
    )

    val_loader, val_meta_cols = create_eval_loader(
        eval_df=val_df,
        image_dir=cfg.TRAIN_IMAGE_DIR,
        batch_size=cfg.BATCH_SIZE,
        image_size=cfg.IMAGE_SIZE,
        num_workers=cfg.NUM_WORKERS,
    )

    if train_meta_cols != val_meta_cols:
        raise ValueError("Train and val metadata columns do not match.")

    print("\n[OK] Train/Val metadata columns match.")

    print("\n=== CHECK ONE TRAIN BATCH ===")

    images, metas, labels = next(iter(train_loader))

    print("Image batch shape:", images.shape)
    print("Meta batch shape :", metas.shape)
    print("Label batch shape:", labels.shape)

    print("Image dtype:", images.dtype)
    print("Meta dtype :", metas.dtype)
    print("Label dtype:", labels.dtype)

    print("Label sample:")
    print(labels[:20])

    print("\n=== CHECK ONE VAL BATCH ===")

    val_images, val_metas, val_labels = next(iter(val_loader))

    print("Val image batch shape:", val_images.shape)
    print("Val meta batch shape :", val_metas.shape)
    print("Val label batch shape:", val_labels.shape)

    print("\n=== EXPECTED ===")
    print(f"Image shape should be: [batch_size, 3, {cfg.IMAGE_SIZE}, {cfg.IMAGE_SIZE}]")
    print(f"Meta dim should be: {len(meta_cols)}")
    print("Label shape should be: [batch_size]")

    print("\n=== RESULT ===")
    print("[DONE] Multimodal dataloader test completed.")

if __name__ == "__main__":
    main()