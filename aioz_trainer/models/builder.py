import torch
from .classifier import build_classifier
from .detector import build_detector


def build_model(config):
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
    task = config["task_type"].lower()
    model_name = config["model_name"]
    num_classes = config.get("num_classes", 1000)
    pretrained = config.get("pretrained", True)
    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    weights_path = config.get("weights_path", None)

    # ---- Build model ----
    if task == "classification":

        model = build_classifier(model_name, num_classes, pretrained=pretrained)

    elif task == "detection":

        model = build_detector(model_name, num_classes, pretrained=pretrained)

    else:
        raise ValueError(f"Unsupported task type: {task}")

    # ---- Load custom weights if specified ----
    if weights_path is not None:
        print(f"🔹 Loading custom weights from: {weights_path}")
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)

    # ---- Move to device ----
    model = model.to(device)

    return model
