import json
import logging
import os
import traceback
from typing import Union

import torch
from aioz_ainode_base.trainer.exception import AINodeTrainerException
from aioz_ainode_base.trainer.schemas import TrainerInput, TrainerOutput

from .models.builder import build_model
from .utils.dataset_loader import get_dataloader
from .utils.metrics import get_criterion, get_optimizer, get_scheduler
from .utils.trainer import train_one_epoch, validate

logger = logging.getLogger(__name__)


def load_config(fp: str) -> dict:
    with open(fp, "r") as file:
        d = json.load(file)
    return d


def run(input_obj: Union[dict, TrainerInput] = None) -> TrainerOutput:
    try:
        # Write code here
        if isinstance(input_obj, dict):
            input_obj = TrainerInput.model_validate(input_obj)

        checkpoint_dir = input_obj.checkpoint_directory
        # output_dir = input_obj.output_directory

        config = load_config(input_obj.config)
        task_type = config["task_type"]

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = build_model(task_type, config["model"], device)

        train_loader, val_loader = get_dataloader(task_type, config["dataset"])

        criterion = get_criterion(task_type, config)
        optimizer = get_optimizer(model, config)
        scheduler = get_scheduler(optimizer, config, num_epochs=config["epochs"], steps_per_epoch=len(train_loader))

        best_weight_path = os.path.join(checkpoint_dir, "best.pt")
        os.makedirs(checkpoint_dir, exist_ok=True)

        start_epoch = 0
        best_score = 0.0
        resume = config["model"]["resume"]
        if resume:
            checkpoint = torch.load(resume, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            if "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            if "scheduler_state_dict" in checkpoint and scheduler is not None:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

            start_epoch = checkpoint.get("epoch", 0) + 1
            best_score = checkpoint.get("best_score", 0.0)

            logger.info(f"Resumed from epoch {start_epoch}, best_score={best_score:.4f}")

        for epoch in range(start_epoch, config["epochs"]):
            logger.info(f"Epoch [{epoch+1}/{config['epochs']}]")
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, config["task_type"])
            metrics = validate(model, val_loader, device, config["task_type"])
            score = list(metrics.values())[0]

            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(score)
                else:
                    scheduler.step()

            logger.info(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Validate Metrics={metrics}")

            if score > best_score:
                best_score = score
                torch.save(model.state_dict(), best_weight_path)

            checkpoint_path = os.path.join(checkpoint_dir, "last.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                    "best_score": best_score,
                },
                checkpoint_path,
            )

        output_obj = TrainerOutput(weights=best_weight_path, metric=best_score, examples=[])
        return output_obj

    except Exception:
        logger.warning(f"Occur an error {traceback.format_exc()}")
        raise AINodeTrainerException()
