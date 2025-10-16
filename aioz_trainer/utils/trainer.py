import torch
from tqdm import tqdm

from .metrics import compute_metrics


def move_to_device(imgs, targets, device, task_type):
    if task_type == "classification":
        return imgs.to(device), targets.to(device)
    elif task_type == "detection":
        imgs = [img.to(device) for img in imgs]
        for t in targets:
            t["boxes"] = t["boxes"].to(device)
            t["labels"] = t["labels"].to(device)
            t["image_id"] = t["image_id"].to(device)
        return imgs, targets
    else:
        raise ValueError(f"Unsupported task_type: {task_type}")


def train_one_epoch(model, dataloader, optimizer, criterion, device, task_type="classification"):
    model.train()
    total_loss = 0.0

    for imgs, targets in tqdm(dataloader, desc="Training"):
        imgs, targets = move_to_device(imgs, targets, device, task_type)

        optimizer.zero_grad()
        
        if task_type == "classification":
            outputs = model(imgs)
            loss = criterion(outputs, targets)
        elif task_type == "detection":
            outputs = model(imgs, targets)
            loss = sum(loss for loss in outputs.values())

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss


def validate(model, dataloader, device, task_type="classification"):
    model.eval()
    all_outputs, all_targets = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            if task_type == "classification":
                imgs, targets = batch
                imgs = imgs.to(device)
                targets = targets.to(device)

                outputs = model(imgs)
                all_outputs.append(outputs)
                all_targets.append(targets)

            elif task_type == "detection":
                imgs, targets = batch
                imgs = [img.to(device) for img in imgs]

                for t in targets:
                    t["boxes"] = t["boxes"].to(device)
                    t["labels"] = t["labels"].to(device)
                    t["image_id"] = t["image_id"].to(device)

                outputs = model(imgs)
                all_outputs.extend(outputs)
                all_targets.extend(targets)
            else:
                raise ValueError(f"Unsupported task_type: {task_type}")

    metrics = compute_metrics(all_outputs, all_targets, task_type)
    return metrics
