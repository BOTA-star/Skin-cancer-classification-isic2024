from torch.utils.data import DataLoader
from .dataset import ISICDataset
from .transforms import get_train_transforms, get_val_transforms

def get_dataloaders(
    train_csv,
    val_csv,
    image_dir,
    batch_size=32,
    num_workers=4
):
    train_dataset = ISICDataset(
        csv_file=train_csv,
        image_dir=image_dir,
        transform=get_train_transforms()
    )

    val_dataset = ISICDataset(
        csv_file=val_csv,
        image_dir=image_dir,
        transform=get_val_transforms()
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader