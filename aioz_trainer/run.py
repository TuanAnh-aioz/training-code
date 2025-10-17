import json
import logging
import os
import traceback
from typing import Union

import cv2
import torch
from aioz_ainode_base.trainer.exception import AINodeTrainerException
from aioz_ainode_base.trainer.schemas import IOExample, IOMetadata, TrainerInput, TrainerOutput

from .draw import draw_label, draw_rounded_rectangle, draw_transparent_box, get_label_colors
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
        output_dir = input_obj.output_directory

        config = load_config(input_obj.config)
        task_type = config["task_type"]

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = build_model(task_type, config["model"], device)

        train_loader, val_loader, _ = get_dataloader(task_type, config["dataset"])

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

        results = inference(task_type, config, output_dir, best_weight_path, device)
        example = [IOExample(input=IOMetadata(data=path, type=str), output=IOMetadata(data=label, type=str)) for path, label in results]
        output_obj = TrainerOutput(weights=best_weight_path, metric=best_score, examples=example)

        return output_obj

    except Exception:
        logger.warning(f"Occur an error {traceback.format_exc()}")
        raise AINodeTrainerException()


def inference(task_type, config, output_dir, weight_path, device):
    os.makedirs(output_dir, exist_ok=True)
    model = build_model(task_type, config["model"], device)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()

    _, val_loader, dataset = get_dataloader(task_type, config["dataset"])
    idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
    results = []
    if task_type == "detection":
        colors = get_label_colors(dataset.class_to_idx)
        with torch.no_grad():
            for idx, (imgs, _, img_paths) in enumerate(val_loader):
                imgs = [img.to(device) for img in imgs]
                outputs = model(imgs)

                for i, (output, img_path_orig) in enumerate(zip(outputs, img_paths)):
                    img_cv = cv2.imread(img_path_orig)

                    boxes = output["boxes"].cpu().numpy()
                    labels = output["labels"].cpu().numpy()
                    scores = output["scores"].cpu().numpy()

                    for box, label, score in zip(boxes, labels, scores):
                        if score < config["threshold"]:
                            continue

                        color = colors.get(label, (255, 0, 0))
                        xmin, ymin, xmax, ymax = map(int, box)
                        class_name = idx_to_class.get(label, str(label))

                        draw_transparent_box(img_cv, (xmin, ymin), (xmax, ymax), color, alpha=0.2)
                        draw_rounded_rectangle(img_cv, (xmin, ymin), (xmax, ymax), color, 2, r=8)
                        draw_label(img_cv, f"{class_name}: {score:.2f}", (xmin, ymin), color)

                    save_path = os.path.join(output_dir, f"{idx}_{i}.jpg")
                    cv2.imwrite(save_path, img_cv)

                results.append((save_path, class_name))

    elif task_type == "classification":
        with torch.no_grad():
            for _, (imgs, _, img_paths) in enumerate(val_loader):
                input = imgs.to(device)
                output = model(input)
                prob = torch.softmax(output, dim=1)
                _, predicted_classes = torch.max(prob, 1)

                for pred, path in zip(predicted_classes, img_paths):
                    class_name = idx_to_class.get(pred.item(), str(pred.item()))
                    score = prob[0, pred].item()

                results.append((os.path.abspath(path), class_name))
    else:
        raise ValueError(f"Unsupported task_type: {task_type}")

    return results
