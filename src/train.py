import torch
import torch.nn as nn
from tqdm import tqdm

def train_one_epoch(model, loader, optimizer, loss_fn, device, max_batches=None):
    model.train()
    total_loss = 0

    for i, (images, labels) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break

        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        outputs = model(images)
        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / (i + 1)

def validate(model, loader, loss_fn, device, max_batches=None):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break

            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            outputs = model(images)
            loss = loss_fn(outputs, labels)

            total_loss += loss.item()

    return total_loss / (i + 1)