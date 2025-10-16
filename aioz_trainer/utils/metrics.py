import logging

import torch
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


def get_optimizer(model, config, param_groups=None):
    """
    Create optimizer from config JSON.

    Expected config format:
        {
            "optimizer": {
                "name": "Adam" | "AdamW" | "SGD" | "RMSprop" | ...,
                "lr": 0.001,
                params: {
                    "momentum": 0.9,
                    "weight_decay": 1e-4,
                    "betas": [0.9, 0.999]
                }
            }
        }
    """
    opt_cfg = config.get("optimizer", {})
    optimizer_name = opt_cfg.get("type", "Adam").lower()
    lr_initial = opt_cfg.get("lr", 1e-3)
    optimizer_params = opt_cfg.get("params", {})

    if param_groups is not None:
        params = param_groups
    else:
        params = [p for p in model.parameters() if p.requires_grad]

    # Optimizer mapping
    optimizers = {
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
        "sgd": torch.optim.SGD,
        "rmsprop": torch.optim.RMSprop,
        "adagrad": torch.optim.Adagrad,
        "adadelta": torch.optim.Adadelta,
        "adamax": torch.optim.Adamax,
        "lion": torch.optim.Lion if hasattr(torch.optim, "Lion") else None,  # PyTorch ≥ 2.1
    }

    if optimizer_name not in optimizers or optimizers[optimizer_name] is None:
        raise NotImplementedError(f"Optimizer '{optimizer_name}' is not implemented or not available in this PyTorch version.")

    # Merge lr into optimizer_params
    if "lr" not in optimizer_params:
        optimizer_params["lr"] = lr_initial

    optimizer_class = optimizers[optimizer_name]
    optimizer = optimizer_class(params, **optimizer_params)

    logger.info(f"Using optimizer: {optimizer_name.upper()} with params: {optimizer_params}")
    return optimizer


def get_scheduler(optimizer, config, num_epochs=None, steps_per_epoch=None):
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
        "params": {
            "T_max": 10,
            "eta_min": 1e-6
        }
      }
    """
    sched_cfg = config.get("scheduler", {})
    if not sched_cfg or "type" not in sched_cfg:
        logger.warning("No scheduler specified.")
        return None

    scheduler_name = sched_cfg["type"].lower()
    scheduler_params = sched_cfg.get("params", {})
    schedulers = {
        "steplr": torch.optim.lr_scheduler.StepLR,
        "multi_steplr": torch.optim.lr_scheduler.MultiStepLR,
        "exponential": torch.optim.lr_scheduler.ExponentialLR,
        "cosine": torch.optim.lr_scheduler.CosineAnnealingLR,
        "cosine_annealing": torch.optim.lr_scheduler.CosineAnnealingLR,
        "cosine_restarts": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
        "plateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
        "onecycle": torch.optim.lr_scheduler.OneCycleLR,
        "linear": torch.optim.lr_scheduler.LinearLR,
        "polynomial": torch.optim.lr_scheduler.PolynomialLR if hasattr(torch.optim.lr_scheduler, "PolynomialLR") else None,
    }

    if scheduler_name not in schedulers or schedulers[scheduler_name] is None:
        raise NotImplementedError(f"Scheduler '{scheduler_name}' is not implemented or not available in this PyTorch version.")

    # Special handling for some schedulers
    if scheduler_name in ["onecycle"]:
        # For OneCycleLR, you must provide total_steps or epochs * steps_per_epoch
        if num_epochs is None or steps_per_epoch is None:
            raise ValueError("For OneCycleLR, you must provide num_epochs and num_steps_per_epoch.")

        total_steps = num_epochs * steps_per_epoch
        scheduler_params.setdefault("total_steps", total_steps)
        scheduler_params.setdefault("max_lr", [pg["lr"] for pg in optimizer.param_groups])
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, **scheduler_params)
    else:
        scheduler_class = schedulers[scheduler_name]
        scheduler = scheduler_class(optimizer, **scheduler_params)

    logger.info(f"Using LR scheduler: {scheduler_name.upper()} with params: {scheduler_params}")
    return scheduler


def get_transforms(config, task_type="classification", mode="train"):
    t_cfg = config["transforms"].get(mode, {})
    t_list = []

    if "resize" in t_cfg:
        t_list.append(transforms.Resize((t_cfg["resize"], t_cfg["resize"])))

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
    if "normalize" in t_cfg:
        mean = t_cfg["normalize"].get("mean", [0.485, 0.456, 0.406])
        std = t_cfg["normalize"].get("std", [0.229, 0.224, 0.225])
        t_list.append(transforms.Normalize(mean=mean, std=std))

    return transforms.Compose(t_list)
