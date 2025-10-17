import logging

import torch
import torchvision
from torch import nn

logger = logging.getLogger(__name__)


def build_detector(model_name: str, num_classes: int, pretrained: str):
    """
    Build any object detection model from torchvision with automatic head replacement.

    Args:
        model_name (str): Name of the model (e.g. 'fasterrcnn_resnet50_fpn', 'retinanet_resnet50_fpn', 'maskrcnn_resnet50_fpn')
        num_classes (int): Number of classes (including background if needed).
        pretrained (bool): Whether to load pretrained weights.

    Returns:
        torch.nn.Module: Detection model with updated head for num_classes.
    """

    # Get the list of detection models that torchvision supports
    available_models = list(torchvision.models.detection.__dict__.keys())
    available_models = [m for m in available_models if not m.startswith("_") and callable(getattr(torchvision.models.detection, m))]

    if model_name not in available_models:
        raise ValueError(
            f"Model '{model_name}' not found in torchvision.models.detection.\n" f"Available detection models: {', '.join(sorted(available_models))}"
        )

    # Get function constructor
    model_fn = getattr(torchvision.models.detection, model_name)

    if pretrained:
        model = model_fn(weights=None, weights_backbone=None)

        # Load pretrained state dict
        state_dict = torch.load(pretrained, map_location="cpu")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning(f"Missing keys: {missing}")
        if unexpected:
            logger.warning(f"Unexpected keys: {unexpected}")
        logger.warning(f"Using default torchvision pretrained weights for {model_name}")
    else:
        logger.warning(f"Initializing '{model_name}' from scratch")
        model = model_fn(weights=None, weights_backbone=None)

    # Automatic head replacement
    model = _replace_detection_head(model, num_classes)
    return model


def _replace_detection_head(model: nn.Module, num_classes: int):
    """
    Replace detection head depending on model type.
    """

    # Faster R-CNN & Mask R-CNN
    if hasattr(model, "roi_heads"):
        if hasattr(model.roi_heads, "box_predictor"):
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            predictor_class = type(model.roi_heads.box_predictor)
            model.roi_heads.box_predictor = predictor_class(in_features, num_classes)

        if hasattr(model.roi_heads, "mask_predictor") and model.roi_heads.mask_predictor is not None:
            in_channels = model.roi_heads.mask_predictor.conv5_mask.in_channels
            mask_predictor_class = type(model.roi_heads.mask_predictor)
            model.roi_heads.mask_predictor = mask_predictor_class(in_channels, 256, num_classes)

    # RetinaNet
    elif hasattr(model, "head") and hasattr(model.head, "classification_head"):
        in_channels = model.head.classification_head.cls_logits.in_channels
        num_anchors = model.head.classification_head.num_anchors
        model.head.classification_head.cls_logits = nn.Conv2d(in_channels, num_anchors * num_classes, kernel_size=3, stride=1, padding=1)

    else:
        raise RuntimeError(f"Don't know how to replace detection head for model type: {type(model).__name__}")

    return model


if __name__ == "__main__":
    test_models = [
        "fasterrcnn_resnet50_fpn",
        "retinanet_resnet50_fpn",
        "maskrcnn_resnet50_fpn",
    ]

    for name in test_models:
        try:
            print(f"\nBuilding detector: {name}")
            model = build_detector(name, num_classes=5, pretrained=False)
            model.eval()
            dummy = [torch.randn(3, 3, 224, 224)]
            out = model(dummy)
            print(f"{name}: OK, output length = {len(out)}")
        except Exception as e:
            print(f"{name}: {e}")
