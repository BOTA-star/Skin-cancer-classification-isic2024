import numpy as np
import torch

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from tqdm import tqdm

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    max_batches=None,
):
    model.train()

    total_loss = 0.0
    total_batches = 0

    progress_bar = tqdm(loader, desc="Training", leave=False)

    for batch_idx, (images, metas, labels) in enumerate(progress_bar):
        if max_batches is not None and batch_idx >= max_batches:
            break

        images = images.to(device)
        metas = metas.to(device)
        labels = labels.to(device).unsqueeze(1)

        optimizer.zero_grad()

        logits = model(images, metas)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_batches += 1

        progress_bar.set_postfix({
            "loss": f"{loss.item():.4f}"
        })

    avg_loss = total_loss / max(total_batches, 1)

    return avg_loss

def evaluate(
    model,
    loader,
    criterion,
    device,
    max_batches=None,
    threshold=0.5,
):
    model.eval()

    total_loss = 0.0
    total_batches = 0

    all_probs = []
    all_targets = []

    progress_bar = tqdm(loader, desc="Evaluating", leave=False)

    with torch.no_grad():
        for batch_idx, (images, metas, labels) in enumerate(progress_bar):
            if max_batches is not None and batch_idx >= max_batches:
                break

            images = images.to(device)
            metas = metas.to(device)
            labels_device = labels.to(device).unsqueeze(1)

            logits = model(images, metas)
            loss = criterion(logits, labels_device)

            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            targets = labels.numpy().flatten()

            all_probs.extend(probs)
            all_targets.extend(targets)

            total_loss += loss.item()
            total_batches += 1

    avg_loss = total_loss / max(total_batches, 1)

    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)

    pred_labels = (all_probs >= threshold).astype(int)

    metrics = {
        "loss": float(avg_loss),
        "accuracy": float(accuracy_score(all_targets, pred_labels)),
        "precision": float(
            precision_score(all_targets, pred_labels, zero_division=0)
        ),
        "recall": float(
            recall_score(all_targets, pred_labels, zero_division=0)
        ),
        "f1": float(
            f1_score(all_targets, pred_labels, zero_division=0)
        ),
    }

    try:
        metrics["auc"] = float(roc_auc_score(all_targets, all_probs))
    except Exception:
        metrics["auc"] = 0.5

    try:
        metrics["pr_auc"] = float(average_precision_score(all_targets, all_probs))
    except Exception:
        metrics["pr_auc"] = 0.0

    return metrics