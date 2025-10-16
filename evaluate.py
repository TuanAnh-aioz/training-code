import json
import logging
import os

import torch
from PIL import ImageDraw
from torchvision import transforms

from models.builder import build_model
from utils.dataset_loader import get_dataloader
from utils.trainer import validate

logger = logging.getLogger(__name__)


def evaluate(config_path, weight_path):
    with open(config_path) as f:
        config = json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    model = build_model(config).to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()

    # get_dataloader trả về train_loader, val_loader
    _, val_loader = get_dataloader(config, seed=config.get("seed", 42))

    # Tính metrics
    metrics = validate(model, val_loader, device, config["task_type"])
    print(f"Evaluation: {metrics}")

    task_type = config["task_type"].lower()
    if task_type == "detection":
        for imgs, targets in val_loader:
            imgs = [img.to(device) for img in imgs]
            outputs = model(imgs)

            for i, img_tensor in enumerate(imgs):
                img = transforms.ToPILImage()(img_tensor.cpu())
                draw = ImageDraw.Draw(img)
                output = outputs[i]

                boxes = output["boxes"].cpu().numpy()
                labels = output["labels"].cpu().numpy()
                scores = output["scores"].cpu().numpy()

                for box, label, score in zip(boxes, labels, scores):
                    x1, y1, x2, y2 = box
                    draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
                    draw.text((x1, y1), f"{label}:{score:.2f}", fill="yellow")

                # Lưu ảnh
                img_name = f"{i}.jpg"
                img.save(os.path.join(config["save_dir"], img_name))


if __name__ == "__main__":
    evaluate("configs/train_config.json", "checkpoints/best.pt")
