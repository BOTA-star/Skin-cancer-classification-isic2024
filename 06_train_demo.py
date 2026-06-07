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
    create_train_loader,
    create_eval_loader,
    get_meta_columns,
)

from model import ISICMultimodalModel

from train import (
    train_one_epoch,
    evaluate,
)

from utils import (
    ensure_dirs,
    check_file_exists,
    check_dir_exists,
    get_device,
    seed_everything,
)


def save_checkpoint(
    model,
    optimizer,
    epoch,
    best_auc,
    path,
):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_auc": best_auc,
            "backbone": cfg.BACKBONE,
            "image_size": cfg.IMAGE_SIZE,
            "meta_dim": None,
        },
        path,
    )

def main():
    print("=== ISIC 2024 TRAIN DEMO ===")

    seed_everything(cfg.SEED)

    ensure_dirs(
        cfg.CHECKPOINT_DIR,
        cfg.OUTPUT_DIR,
    )

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

    device = get_device()

    print("Device:", device)
    print("Backbone:", cfg.BACKBONE)
    print("Use pretrained:", cfg.USE_PRETRAINED)
    print("Image size:", cfg.IMAGE_SIZE)
    print("Batch size:", cfg.BATCH_SIZE)
    print("Epochs:", cfg.EPOCHS)
    print("Max train batches:", cfg.MAX_TRAIN_BATCHES)
    print("Max val batches:", cfg.MAX_VAL_BATCHES)

    train_df = pd.read_csv(cfg.TRAIN_PROCESSED_CSV)
    val_df = pd.read_csv(cfg.VAL_PROCESSED_CSV)

    meta_cols = get_meta_columns(train_df)
    meta_dim = len(meta_cols)

    print("\nTrain shape:", train_df.shape)
    print("Val shape:", val_df.shape)
    print("Metadata dim:", meta_dim)

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

    model = ISICMultimodalModel(
        meta_dim=meta_dim,
        backbone_name=cfg.BACKBONE,
        pretrained=cfg.USE_PRETRAINED,
        backbone_weights_path=getattr(cfg, "BACKBONE_WEIGHTS_PATH", None),
    ).to(device)

    # Demo CPU: vẫn train toàn bộ MobileNet nhỏ.
    # Nếu máy quá chậm, có thể bật dòng dưới để chỉ train metadata + classifier.
    # model.freeze_image_backbone()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.LR,
    )

    criterion = nn.BCEWithLogitsLoss()

    best_auc = 0.0
    history = []

    for epoch in range(1, cfg.EPOCHS + 1):
        print(f"\n=== EPOCH {epoch}/{cfg.EPOCHS} ===")

        if epoch == cfg.UNFREEZE_EPOCH:
            print("[INFO] Unfreezing image backbone...")
            model.unfreeze_image_backbone()

        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            max_batches=cfg.MAX_TRAIN_BATCHES,
        )

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            max_batches=cfg.MAX_VAL_BATCHES,
        )

        epoch_result = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            **val_metrics,
        }

        history.append(epoch_result)

        print("\nEpoch result:")
        print(json.dumps(epoch_result, indent=2))

        current_auc = val_metrics["auc"]

        if current_auc > best_auc:
            best_auc = current_auc

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "backbone": cfg.BACKBONE,
                    "image_size": cfg.IMAGE_SIZE,
                    "meta_dim": meta_dim,
                    "meta_cols": meta_cols,
                    "best_auc": best_auc,
                },
                cfg.BEST_MODEL_PATH,
            )

            print(f"[OK] Best model saved: {cfg.BEST_MODEL_PATH}")

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "best_auc": best_auc,
                "backbone": cfg.BACKBONE,
                "image_size": cfg.IMAGE_SIZE,
                "meta_dim": meta_dim,
                "meta_cols": meta_cols,
            },
            cfg.LAST_MODEL_PATH,
        )

        print(f"[OK] Last checkpoint saved: {cfg.LAST_MODEL_PATH}")

    history_path = os.path.join(
        cfg.OUTPUT_DIR,
        "train_demo_history.csv",
    )

    pd.DataFrame(history).to_csv(
        history_path,
        index=False,
    )

    metrics_path = os.path.join(
        cfg.OUTPUT_DIR,
        "train_demo_metrics.json",
    )

    final_result = {
        "best_auc": best_auc,
        "history": history,
        "config": {
            "backbone": cfg.BACKBONE,
            "image_size": cfg.IMAGE_SIZE,
            "batch_size": cfg.BATCH_SIZE,
            "epochs": cfg.EPOCHS,
            "max_train_batches": cfg.MAX_TRAIN_BATCHES,
            "max_val_batches": cfg.MAX_VAL_BATCHES,
            "use_pretrained": cfg.USE_PRETRAINED,
        },
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            final_result,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\n=== RESULT ===")
    print(f"Best AUC: {best_auc:.4f}")
    print(f"History saved: {history_path}")
    print(f"Metrics saved: {metrics_path}")
    print("[DONE] Train demo completed.")

if __name__ == "__main__":
    main()