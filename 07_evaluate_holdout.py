import json
import os
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

ROOT_DIR = Path(__file__).resolve().parent
sys.path.append(str(ROOT_DIR))

import config as cfg

from dataloader import (
    create_eval_loader,
    get_meta_columns,
)

from model import ISICMultimodalModel
from train import evaluate

from utils import (
    ensure_dirs,
    check_file_exists,
    check_dir_exists,
    get_device,
    seed_everything,
)

def main():
    print("=== ISIC 2024 EVALUATE HOLDOUT ===")

    seed_everything(cfg.SEED)

    ensure_dirs(cfg.OUTPUT_DIR)

    check_file_exists(
        cfg.HOLDOUT_PROCESSED_CSV,
        "holdout processed CSV",
    )

    check_file_exists(
        cfg.BEST_MODEL_PATH,
        "best model checkpoint",
    )

    check_dir_exists(
        cfg.TRAIN_IMAGE_DIR,
        "training image directory",
    )

    device = get_device()

    print("Device:", device)

    holdout_df = pd.read_csv(cfg.HOLDOUT_PROCESSED_CSV)

    meta_cols = get_meta_columns(holdout_df)
    meta_dim = len(meta_cols)

    print("Holdout shape:", holdout_df.shape)
    print("Metadata dim:", meta_dim)

    holdout_loader, _ = create_eval_loader(
        eval_df=holdout_df,
        image_dir=cfg.TRAIN_IMAGE_DIR,
        batch_size=cfg.BATCH_SIZE,
        image_size=cfg.IMAGE_SIZE,
        num_workers=cfg.NUM_WORKERS,
    )

    checkpoint = torch.load(
        cfg.BEST_MODEL_PATH,
        map_location=device,
        weights_only=False,
    )

    backbone = checkpoint.get("backbone", cfg.BACKBONE)
    image_size = checkpoint.get("image_size", cfg.IMAGE_SIZE)
    checkpoint_meta_dim = checkpoint.get("meta_dim", meta_dim)

    print("Checkpoint backbone:", backbone)
    print("Checkpoint image size:", image_size)
    print("Checkpoint meta dim:", checkpoint_meta_dim)

    if checkpoint_meta_dim != meta_dim:
        raise ValueError(
            f"Metadata dim mismatch. "
            f"Checkpoint: {checkpoint_meta_dim}, current holdout: {meta_dim}"
        )

    model = ISICMultimodalModel(
        meta_dim=meta_dim,
        backbone_name=backbone,
        pretrained=False,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])

    criterion = nn.BCEWithLogitsLoss()

    metrics = evaluate(
        model=model,
        loader=holdout_loader,
        criterion=criterion,
        device=device,
        max_batches=None,
    )

    print("\n=== HOLDOUT METRICS ===")
    print(json.dumps(metrics, indent=2))

    output_path = os.path.join(
        cfg.OUTPUT_DIR,
        "holdout_metrics.json",
    )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            metrics,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\n=== RESULT ===")
    print(f"[OK] Holdout metrics saved: {output_path}")
    print("[DONE] Holdout evaluation completed.")

if __name__ == "__main__":
    main()