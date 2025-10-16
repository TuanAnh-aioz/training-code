import json
import logging
import os
import traceback
from typing import Union

import torch
from aioz_ainode_base.trainer.exception import AINodeTrainerException
from aioz_ainode_base.trainer.schemas import TrainerInput, TrainerOutput
from torch import nn

from .models.builder import build_model
from .utils.dataset_loader import get_dataloader
from .utils.metrics import get_optimizer, get_scheduler
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

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = build_model(config).to(device)

        train_loader, val_loader = get_dataloader(config)

        criterion = nn.CrossEntropyLoss() if config["task_type"] == "classification" else None
        optimizer = get_optimizer(model, config)
        scheduler = get_scheduler(optimizer, config, num_epochs=config["epochs"], steps_per_epoch=len(train_loader))

        best_score = 0
        best_weight_path = f"{checkpoint_dir}/best.pt"
        os.makedirs(checkpoint_dir, exist_ok=True)

        for epoch in range(config["epochs"]):
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

        output_obj = TrainerOutput(weights=best_weight_path, metrix=best_score, examples=[])
        return output_obj

    except Exception:
        logger.warning(f"Occur an error {traceback.format_exc()}")
        raise AINodeTrainerException()
