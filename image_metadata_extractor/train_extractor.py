import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import ImageMetadataDataset
from model import ImageMetadataRegressor
from transforms import get_train_transform, get_eval_transform
from utils import (
    set_seed,
    save_json,
    select_image_target_cols,
    compute_target_stats,
    inverse_scale,
    sample_dataframe,
)
from config import WEAK_IMAGE_TARGETS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-csv", type=str, required=True)
    parser.add_argument("--image-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="outputs/image_metadata_extractor")

    parser.add_argument("--sample-size", type=int, default=50000)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--backbone", type=str, default="mobilenet_v3_small",
                        choices=["mobilenet_v3_small", "efficientnet_b0", "efficientnet_b3"])
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)

    parser.add_argument("--exclude-weak-targets", action="store_true",
                        help="Exclude tbp_lv_x/y/z. Default keeps all numeric tbp_lv_* columns.")
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    return parser.parse_args()


def masked_smooth_l1_loss(pred, target, mask):
    loss = torch.nn.functional.smooth_l1_loss(pred, target, reduction="none")
    loss = loss * mask
    denom = mask.sum().clamp(min=1.0)
    return loss.sum() / denom


@torch.no_grad()
def evaluate(model, loader, device, target_cols, target_stats, max_batches=0):
    model.eval()
    total_loss = 0.0
    n_batches = 0

    preds_scaled = []
    targets_scaled = []
    masks = []

    for batch_idx, batch in enumerate(tqdm(loader, desc="Val", leave=False)):
        if max_batches and batch_idx >= max_batches:
            break

        images = batch["image"].to(device)
        y = batch["target"].to(device)
        mask = batch["mask"].to(device)

        pred = model(images)
        loss = masked_smooth_l1_loss(pred, y, mask)

        total_loss += float(loss.item())
        n_batches += 1

        preds_scaled.append(pred.cpu().numpy())
        targets_scaled.append(y.cpu().numpy())
        masks.append(mask.cpu().numpy())

    if n_batches == 0:
        return {"loss": None}, pd.DataFrame()

    preds_scaled = np.concatenate(preds_scaled, axis=0)
    targets_scaled = np.concatenate(targets_scaled, axis=0)
    masks = np.concatenate(masks, axis=0)

    preds = inverse_scale(preds_scaled, target_cols, target_stats)
    targets = inverse_scale(targets_scaled, target_cols, target_stats)

    rows = []
    for j, col in enumerate(target_cols):
        valid = masks[:, j] > 0
        if valid.sum() < 2:
            rows.append({"feature": col, "mae": None, "r2": None, "valid_count": int(valid.sum())})
            continue

        y_true = targets[valid, j]
        y_pred = preds[valid, j]
        mae = mean_absolute_error(y_true, y_pred)
        try:
            r2 = r2_score(y_true, y_pred)
        except Exception:
            r2 = None

        rows.append({
            "feature": col,
            "mae": float(mae),
            "r2": None if r2 is None else float(r2),
            "valid_count": int(valid.sum()),
        })

    metrics_df = pd.DataFrame(rows)
    macro_mae = float(metrics_df["mae"].dropna().mean()) if "mae" in metrics_df else None
    macro_r2 = float(metrics_df["r2"].dropna().mean()) if "r2" in metrics_df else None

    metrics = {
        "loss": total_loss / n_batches,
        "macro_mae": macro_mae,
        "macro_r2": macro_r2,
    }
    return metrics, metrics_df


def main():
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.metadata_csv)
    if "isic_id" not in df.columns:
        raise ValueError("metadata CSV must contain isic_id.")

    target_cols = select_image_target_cols(
        df,
        include_weak_targets=not args.exclude_weak_targets,
    )

    weak_selected = [c for c in target_cols if c in WEAK_IMAGE_TARGETS]
    if weak_selected:
        print("Warning: weak image targets selected:", weak_selected)
        print("These may depend on 3D-TBP device/location and may not generalize to normal uploaded images.")

    df = sample_dataframe(df, args.sample_size, args.seed)

    # Fit target stats on the selected training pool before split.
    target_stats = compute_target_stats(df, target_cols)

    if "patient_id" in df.columns and df["patient_id"].notna().any():
        splitter = GroupShuffleSplit(n_splits=1, test_size=args.val_size, random_state=args.seed)
        train_idx, val_idx = next(splitter.split(df, groups=df["patient_id"].fillna("unknown")))
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)
    else:
        train_df, val_df = train_test_split(df, test_size=args.val_size, random_state=args.seed)

    print(f"Train size: {len(train_df):,}")
    print(f"Val size  : {len(val_df):,}")
    print(f"Targets   : {len(target_cols)}")
    print(target_cols)

    save_json(target_cols, output_dir / "target_cols.json")
    save_json(target_stats, output_dir / "target_stats.json")

    train_ds = ImageMetadataDataset(
        train_df,
        args.image_dir,
        target_cols,
        target_stats["mean"],
        target_stats["std"],
        transform=get_train_transform(args.image_size),
    )
    val_ds = ImageMetadataDataset(
        val_df,
        args.image_dir,
        target_cols,
        target_stats["mean"],
        target_stats["std"],
        transform=get_eval_transform(args.image_size),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model = ImageMetadataRegressor(
        num_outputs=len(target_cols),
        backbone=args.backbone,
        pretrained=args.pretrained,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_loss = float("inf")
    log_rows = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch_idx, batch in enumerate(pbar):
            if args.max_train_batches and batch_idx >= args.max_train_batches:
                break

            images = batch["image"].to(device)
            y = batch["target"].to(device)
            mask = batch["mask"].to(device)

            optimizer.zero_grad()
            pred = model(images)
            loss = masked_smooth_l1_loss(pred, y, mask)
            loss.backward()
            optimizer.step()

            train_loss += float(loss.item())
            n_batches += 1
            pbar.set_postfix(loss=train_loss / max(n_batches, 1))

        avg_train_loss = train_loss / max(n_batches, 1)
        val_metrics, val_metrics_df = evaluate(
            model,
            val_loader,
            device,
            target_cols,
            target_stats,
            max_batches=args.max_val_batches,
        )

        row = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": val_metrics["loss"],
            "val_macro_mae": val_metrics["macro_mae"],
            "val_macro_r2": val_metrics["macro_r2"],
        }
        log_rows.append(row)

        print(row)

        pd.DataFrame(log_rows).to_csv(output_dir / "training_log.csv", index=False)
        val_metrics_df.to_csv(output_dir / "val_metrics.csv", index=False)

        ckpt = {
            "model_state": model.state_dict(),
            "backbone": args.backbone,
            "pretrained": args.pretrained,
            "image_size": args.image_size,
            "target_cols": target_cols,
            "target_stats": target_stats,
        }

        torch.save(ckpt, output_dir / "last_extractor.pth")

        val_loss = val_metrics["loss"]
        if val_loss is not None and val_loss < best_loss:
            best_loss = val_loss
            torch.save(ckpt, output_dir / "best_extractor.pth")
            print(f"Saved best checkpoint: {best_loss:.6f}")

    print("Done.")
    print(f"Output dir: {output_dir}")


if __name__ == "__main__":
    main()
