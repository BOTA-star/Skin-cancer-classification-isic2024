import sys
from pathlib import Path

import pandas as pd
import torch


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

import config.config as cfg

from src.dataloader import (
    create_train_loader,
    get_meta_columns,
)

from src.model import ISICMultimodalModel

from src.utils import (
    check_file_exists,
    check_dir_exists,
    get_device,
    seed_everything,
)


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return total, trainable


def main():
    print("========== ISIC 2024 TEST MULTIMODAL MODEL ==========")

    seed_everything(cfg.SEED)

    check_file_exists(
        cfg.TRAIN_PROCESSED_CSV,
        "train processed CSV",
    )

    check_dir_exists(
        cfg.TRAIN_IMAGE_DIR,
        "training image directory",
    )

    device = get_device()
    print("Device:", device)
    print("Backbone:", cfg.BACKBONE)
    print("Use pretrained:", cfg.USE_PRETRAINED)

    train_df = pd.read_csv(cfg.TRAIN_PROCESSED_CSV)

    meta_cols = get_meta_columns(train_df)
    meta_dim = len(meta_cols)

    print("Train shape:", train_df.shape)
    print("Metadata dim:", meta_dim)

    train_loader, _ = create_train_loader(
        train_df=train_df,
        image_dir=cfg.TRAIN_IMAGE_DIR,
        batch_size=cfg.BATCH_SIZE,
        image_size=cfg.IMAGE_SIZE,
        num_workers=cfg.NUM_WORKERS,
        use_weighted_sampler=True,
    )

    # Test local thì để pretrained=False để tránh phải tải weights.
    # Khi train thật trên Colab có thể dùng pretrained=True.
    model = ISICMultimodalModel(
        meta_dim=meta_dim,
        backbone_name=cfg.BACKBONE,
        pretrained=cfg.USE_PRETRAINED,
    ).to(device)

    total_params, trainable_params = count_parameters(model)

    print("\n========== MODEL PARAMS ==========")
    print(f"Total params    : {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")

    print("\n========== FREEZE IMAGE BACKBONE TEST ==========")

    model.freeze_image_backbone()

    total_params, trainable_params = count_parameters(model)

    print(f"Total params    : {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")

    print("\n========== FORWARD PASS TEST ==========")

    images, metas, labels = next(iter(train_loader))

    images = images.to(device)
    metas = metas.to(device)
    labels = labels.to(device).unsqueeze(1)

    print("Images shape:", images.shape)
    print("Metas shape :", metas.shape)
    print("Labels shape:", labels.shape)

    model.eval()

    with torch.no_grad():
        logits = model(images, metas)
        probs = torch.sigmoid(logits)

    print("Logits shape:", logits.shape)
    print("Probs shape :", probs.shape)

    print("Probs min/max:")
    print(float(probs.min()), float(probs.max()))

    if logits.shape != labels.shape:
        raise ValueError(
            f"Output shape {logits.shape} does not match label shape {labels.shape}"
        )

    print("\n========== EXPECTED ==========")
    print("Logits shape should be: [batch_size, 1]")
    print("Labels shape should be: [batch_size, 1]")

    print("\n========== RESULT ==========")
    print("[DONE] Multimodal model test completed.")


if __name__ == "__main__":
    main()