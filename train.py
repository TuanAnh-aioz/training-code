import json
import os

import torch
from torch import nn

from models.builder import build_model
from utils.dataset_loader import get_dataloader
from utils.metrics import get_optimizer, get_scheduler
from utils.trainer import train_one_epoch, validate

# logger = logging.getLogger(__name__)


def main(config_path):
    with open(config_path) as f:
        config = json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(config).to(device)

    train_loader, val_loader = get_dataloader(config)

    criterion = nn.CrossEntropyLoss() if config["task_type"] == "classification" else None
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config, steps_per_epoch=len(train_loader))

    best_score = 0
    best_weight_path = f"{config['save_dir']}/best.pt"
    os.makedirs(config["save_dir"], exist_ok=True)

    for epoch in range(config["epochs"]):
        print(f"\n=== Epoch [{epoch+1}/{config['epochs']}] ===")
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, config["task_type"])
        metrics = validate(model, val_loader, device, config["task_type"])
        score = list(metrics.values())[0]
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(score)
            else:
                scheduler.step()

        print(f"Epoch {epoch+1}: loss={train_loss:.4f}, metrics={metrics}")

        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), best_weight_path)

    print(f"Best model saved to {best_weight_path} with {best_score}")
    return best_weight_path


if __name__ == "__main__":
    main("configs/train_config_det.json")
