import torch
import torch.nn as nn
import torchvision.models as models


def build_classifier(model_name: str, num_classes: int, pretrained: bool = True):
    """
    Build any classification model from torchvision with dynamic head replacement.

    Args:
        model_name (str): name of the model in torchvision.models (e.g. "resnet18", "efficientnet_b0", ...)
        num_classes (int): number of output classes.
        pretrained (bool): whether to load pretrained ImageNet weights.

    Returns:
        torch.nn.Module: classification model with adjusted output layer.
    """
    # Check if the model exists in torchvision.models
    if not hasattr(models, model_name):
        raise ValueError(f"Model '{model_name}' not found in torchvision.models.\n" f"Available models: {', '.join(sorted(models.list_models()))}")

    # Get constructor
    model_fn = getattr(models, model_name)

    # Get default weights if pretrained=True
    weights = "DEFAULT" if pretrained else None

    # Creat model
    try:
        model = model_fn(weights=weights)
    except TypeError:
        # Some older models still use 'pretrained=True' instead of 'weights'
        model = model_fn(pretrained=pretrained)

    # ---- Find and replace the last classification layer ----
    model = _replace_classification_head(model, num_classes)
    return model


def _replace_classification_head(model: nn.Module, num_classes: int) -> nn.Module:
    """
    Replace the classification head automatically depending on model architecture.
    """

    # ResNet / EfficientNet / MobileNetVx / ViT / ConvNeXt
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    elif hasattr(model, "classifier"):
        # The classifier can be either Sequential or Linear.
        if isinstance(model.classifier, nn.Linear):
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features, num_classes)
        elif isinstance(model.classifier, nn.Sequential):
            last_layer = list(model.classifier.children())[-1]
            if isinstance(last_layer, nn.Linear):
                in_features = last_layer.in_features
                model.classifier[-1] = nn.Linear(in_features, num_classes)
            else:
                model.classifier.append(nn.Linear(last_layer.out_features, num_classes))

    elif hasattr(model, "heads") and isinstance(model.heads, nn.Module):
        # Ex: VisionTransformer
        for name, module in model.heads.named_children():
            if isinstance(module, nn.Linear):
                in_features = module.in_features
                setattr(model.heads, name, nn.Linear(in_features, num_classes))

    else:
        raise RuntimeError(f"Don't know how to replace classification head for model type: {type(model).__name__}")

    return model


if __name__ == "__main__":
    for name in ["resnet18", "mobilenet_v3_small", "efficientnet_b0", "vit_b_16", "convnext_tiny"]:
        try:
            print(f"Building {name.upper()} ...")
            model = build_classifier(name, num_classes=5, pretrained=False)
            x = torch.randn(2, 3, 224, 224)
            y = model(x)
            print(f"{name.upper()}: output shape = {y.shape}")
        except Exception as e:
            print(f"{name.upper()}: {e}")
