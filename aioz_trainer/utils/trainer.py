from typing import Dict, List, Union

import torch
from torch.utils.data import DataLoader

from ..enums import Tasks
from .metrics import compute_metrics

TensorOrList = Union[torch.Tensor, List[torch.Tensor]]
TargetType = Union[torch.Tensor, List[Dict[str, torch.Tensor]]]


def move_to_device(imgs: TensorOrList, targets: TargetType, device: str, task_type: str):
    if task_type == Tasks.CLASSIFICATION.value:
        return imgs.to(device), targets.to(device)
    elif task_type == Tasks.DETECTION.value:
        imgs = [img.to(device) for img in imgs]
        for t in targets:
            t["boxes"] = t["boxes"].to(device)
            t["labels"] = t["labels"].to(device)
            t["image_id"] = t["image_id"].to(device)
        return imgs, targets
    else:
        raise ValueError(f"Unsupported task_type: {task_type}")


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: Union[torch.nn.Module, callable],
    device: str,
    task_type: str,
) -> float:
    model.train()
    total_loss = 0.0

    for imgs, targets, _ in dataloader:
        imgs, targets = move_to_device(imgs, targets, device, task_type)

        optimizer.zero_grad()

        if task_type == Tasks.CLASSIFICATION.value:
            outputs = model(imgs)
            loss = criterion(outputs, targets)
        elif task_type == Tasks.DETECTION.value:
            outputs = model(imgs, targets)
            loss = sum(loss for loss in outputs.values())

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss


def validate(model: torch.nn.Module, dataloader: DataLoader, device: str, task_type: str) -> dict:
    model.eval()
    all_outputs, all_targets = [], []

    with torch.no_grad():
        for batch in dataloader:
            imgs, targets, _ = batch
            if task_type == Tasks.CLASSIFICATION.value:
                imgs = imgs.to(device)
                targets = targets.to(device)

                outputs = model(imgs)
                all_outputs.append(outputs)
                all_targets.append(targets)

            elif task_type == Tasks.DETECTION.value:
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
