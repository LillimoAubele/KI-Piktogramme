"""
Model definitions: ImageNet-pretrained CNN backbones with a new classification head.
"""
import torch.nn as nn
from torchvision import models


def build_model(name="efficientnet_b0", num_classes=5, pretrained=True):
    """
    Build a pretrained backbone and replace its final layer with a
    classifier for `num_classes` classes.

    Supported: efficientnet_b0 (default), resnet50, convnext_tiny
    """
    if name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    elif name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif name == "convnext_tiny":
        weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.convnext_tiny(weights=weights)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)

    else:
        raise ValueError(f"Unknown model name: {name}")

    return model