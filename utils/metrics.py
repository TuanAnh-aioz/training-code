import logging

import torch
import torch.optim as optim
from sklearn.metrics import accuracy_score
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision import transforms

logger = logging.getLogger(__name__)


def compute_metrics(outputs, targets, task_type):
    if task_type == "classification":
        preds = torch.cat(outputs).argmax(dim=1).cpu().numpy()
        labels = torch.cat(targets).cpu().numpy()
        acc = accuracy_score(labels, preds)
        return {"accuracy": acc}

    elif task_type == "detection":
        metric = MeanAveragePrecision(iou_type="bbox")
        metric.update(outputs, targets)
        result = metric.compute()
        return {"mAP": result["map"].item()}

    else:
        raise ValueError("Unsupported task")


def get_optimizer(model, config):
    """
    Create optimizer from config JSON.

    Expected config format:
    {
      "optimizer": {
        "name": "Adam" | "AdamW" | "SGD" | "RMSprop" | ...,
        "lr": 0.001,
        "momentum": 0.9,
        "weight_decay": 1e-4,
        "betas": [0.9, 0.999]
      }
    }
    """
    opt_cfg = config.get("optimizer", {})
    name = opt_cfg.get("type", "Adam").lower()
    lr = opt_cfg.get("lr", 1e-3)
    weight_decay = opt_cfg.get("weight_decay", 0.0)
    momentum = opt_cfg.get("momentum", 0.9)
    betas = tuple(opt_cfg.get("betas", (0.9, 0.999)))

    params = model.parameters()

    if name == "adam":
        optimizer = optim.Adam(params, lr=lr, betas=betas, weight_decay=weight_decay)

    elif name == "adamw":
        optimizer = optim.AdamW(params, lr=lr, betas=betas, weight_decay=weight_decay)

    elif name == "sgd":
        optimizer = optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)

    elif name == "rmsprop":
        optimizer = optim.RMSprop(params, lr=lr, momentum=momentum, weight_decay=weight_decay)

    elif name == "adagrad":
        optimizer = optim.Adagrad(params, lr=lr, weight_decay=weight_decay)

    else:
        raise ValueError(f"Unsupported optimizer type: {name}")

    print(f"Using optimizer: {name.upper()}")
    return optimizer


def get_scheduler(optimizer, config, steps_per_epoch=None):
    """
    Build learning rate scheduler from JSON config.

    Supported schedulers:
      - StepLR
      - MultiStepLR
      - CosineAnnealingLR
      - ReduceLROnPlateau
      - OneCycleLR

    Example JSON:
      "scheduler": {
        "name": "CosineAnnealingLR",
        "T_max": 10,
        "eta_min": 1e-6
      }
    """
    sched_cfg = config.get("scheduler", {})
    if not sched_cfg or "type" not in sched_cfg:
        print("No scheduler specified.")
        return None

    name = sched_cfg["type"].lower()
    print(f"Using scheduler: {name}")

    if name == "steplr":
        return optim.lr_scheduler.StepLR(optimizer, step_size=sched_cfg.get("step_size", 10), gamma=sched_cfg.get("gamma", 0.1))

    elif name == "multisteplr":
        return optim.lr_scheduler.MultiStepLR(optimizer, milestones=sched_cfg.get("milestones", [30, 60, 90]), gamma=sched_cfg.get("gamma", 0.1))

    elif name == "cosineannealinglr":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=sched_cfg.get("T_max", 10), eta_min=sched_cfg.get("eta_min", 1e-6))

    elif name == "reducelronplateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=sched_cfg.get("mode", "max"),
            factor=sched_cfg.get("factor", 0.1),
            patience=sched_cfg.get("patience", 5),
            min_lr=sched_cfg.get("min_lr", 1e-7),
        )

    elif name == "onecyclelr":
        # cần steps_per_epoch và total_epochs trong config
        total_epochs = config.get("epochs", 10)
        if steps_per_epoch is None:
            raise ValueError("steps_per_epoch required for OneCycleLR.")
        max_lr = sched_cfg.get("max_lr", 0.001)
        return optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, steps_per_epoch=steps_per_epoch, epochs=total_epochs)

    else:
        raise ValueError(f"Unsupported scheduler type: {name}")


def get_transforms(config, task_type="classification", mode="train"):
    t_cfg = config["transforms"].get(mode, {})
    t_list = []

    # Resize ảnh
    if "resize" in t_cfg:
        t_list.append(transforms.Resize((t_cfg["resize"], t_cfg["resize"])))

    # Chỉ train mới augment
    if mode == "train":
        if t_cfg.get("horizontal_flip", False):
            t_list.append(transforms.RandomHorizontalFlip())
        if "color_jitter" in t_cfg and task_type == "classification":
            cj = t_cfg["color_jitter"]
            t_list.append(
                transforms.ColorJitter(
                    brightness=cj.get("brightness", 0), contrast=cj.get("contrast", 0), saturation=cj.get("saturation", 0), hue=cj.get("hue", 0)
                )
            )
        if "random_crop" in t_cfg and task_type == "classification":
            t_list.append(transforms.RandomResizedCrop(t_cfg["random_crop"]))

    t_list.append(transforms.ToTensor())

    # Normalize
    if "normalize" in t_cfg:
        mean = t_cfg["normalize"].get("mean", [0.485, 0.456, 0.406])
        std = t_cfg["normalize"].get("std", [0.229, 0.224, 0.225])
        t_list.append(transforms.Normalize(mean=mean, std=std))

    return transforms.Compose(t_list)
