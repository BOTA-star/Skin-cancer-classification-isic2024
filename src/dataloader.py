import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.dataset import ISICMultimodalDataset
from src.transforms import get_train_transforms, get_val_transforms


BASE_COLS = [
    "isic_id",
    "patient_id",
    "target",
]


def get_meta_columns(df):
    meta_cols = [c for c in df.columns if c not in BASE_COLS]

    if len(meta_cols) == 0:
        raise ValueError("No metadata feature columns found.")

    return meta_cols


def create_weighted_sampler(df):
    if "target" not in df.columns:
        raise ValueError("Cannot create sampler because 'target' column is missing.")

    class_counts = df["target"].value_counts().to_dict()

    if 0 not in class_counts or 1 not in class_counts:
        raise ValueError(
            f"Both classes are required for weighted sampler. Found: {class_counts}"
        )

    weights = df["target"].map({
        0: 1.0 / class_counts[0],
        1: 1.0 / class_counts[1],
    }).values.copy()

    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(weights),
        num_samples=len(weights),
        replacement=True,
    )

    return sampler


def create_train_loader(
    train_df,
    image_dir,
    batch_size,
    image_size,
    num_workers,
    use_weighted_sampler=True,
):
    meta_cols = get_meta_columns(train_df)

    train_ds = ISICMultimodalDataset(
        df=train_df,
        image_dir=image_dir,
        meta_cols=meta_cols,
        transform=get_train_transforms(image_size),
        has_label=True,
    )

    sampler = None
    shuffle = True

    if use_weighted_sampler:
        sampler = create_weighted_sampler(train_df)
        shuffle = False

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, meta_cols


def create_eval_loader(
    eval_df,
    image_dir,
    batch_size,
    image_size,
    num_workers,
):
    meta_cols = get_meta_columns(eval_df)

    eval_ds = ISICMultimodalDataset(
        df=eval_df,
        image_dir=image_dir,
        meta_cols=meta_cols,
        transform=get_val_transforms(image_size),
        has_label=True,
    )

    eval_loader = DataLoader(
        eval_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return eval_loader, meta_cols