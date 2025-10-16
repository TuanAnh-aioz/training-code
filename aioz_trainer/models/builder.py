import logging

import torch

from .classifier import build_classifier
from .detector import build_detector

logger = logging.getLogger(__name__)


def build_model(task_type, config, device):
    """
    Build a model (classification or detection) based on the given config.

    Supported:
    - Classification: uses torchvision.models.*
    - Detection: uses torchvision.models.detection.*

    Expected config fields:
        - task_type: "classification" | "detection"
        - model_name: e.g. "resnet18", "fasterrcnn_resnet50_fpn"
        - num_classes: int
        - pretrained: bool (optional, default True)
        - device: "cpu" | "cuda" (optional)
        - weights_path: optional path to load custom weights
    """
    model_name = config["name"]
    num_classes = config.get("num_classes", 1000)
    pretrained = config.get("pretrained", True)
    weights_path = config.get("weights_path", None)

    # ---- Build model ----
    if task_type == "classification":
        model = build_classifier(model_name, num_classes, pretrained=pretrained)
    elif task_type == "detection":
        model = build_detector(model_name, num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"Unsupported task type: {task_type}")

    # ---- Load custom weights if specified ----
    if weights_path is not None:
        logger.info(f"Loading custom weights from: {weights_path}")
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)

    # ---- Move to device ----
    model = model.to(device)

    return model
