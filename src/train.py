import torch
from tqdm import tqdm

def train_one_epoch(model, loader, optimizer, loss_fn, device, max_batches=None):
    model.train()
    total_loss = 0
    total_samples = 0

    for i, (images, labels) in enumerate(tqdm(loader)):
        if max_batches and i >= max_batches:
            break

        images = images.to(device, non_blocking=True)
        labels = labels.float().unsqueeze(1).to(device, non_blocking=True)

        outputs = model(images)
        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples

def validate(model, loader, loss_fn, device, max_batches=None):
    model.eval()
    total_loss = 0
    total_samples = 0

    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            if max_batches and i >= max_batches:
                break

            images = images.to(device, non_blocking=True)
            labels = labels.float().unsqueeze(1).to(device, non_blocking=True)

            outputs = model(images)
            loss = loss_fn(outputs, labels)

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    return total_loss / total_samples