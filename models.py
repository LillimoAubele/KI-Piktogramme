import torch.nn as nn
from torchvision.models import resnet18, resnet50, ResNet50_Weights, densenet121, efficientnet_b0, efficientnet_b3, EfficientNet_B3_Weights
from monai.networks.nets import DenseNet121


# --------------------------------------------------
# ResNet18
# --------------------------------------------------
def build_resnet(num_classes, pretrained=True, fine_tune=True):
    model = resnet18(weights="IMAGENET1K_V1" if pretrained else None)

    
    # Freeze backbone if not fine-tuning
    if not fine_tune:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


# --------------------------------------------------
# ResNet50
# --------------------------------------------------


def build_resnet50(num_classes, pretrained=True, fine_tune=True):
    
    if pretrained:
        weights = ResNet50_Weights.DEFAULT
        model = resnet50(weights=weights)
    else:
        model = resnet50(weights=None)

    # Klassifikationskopf ersetzen
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Fine-tuning
    if not fine_tune:
        for p in model.parameters():
            p.requires_grad = False
        for p in model.fc.parameters():
            p.requires_grad = True

    return model

# --------------------------------------------------
# DenseNet121
# --------------------------------------------------
def build_densenet(num_classes, pretrained=True, fine_tune=True):
    model = densenet121(weights="IMAGENET1K_V1" if pretrained else None)

    # Freeze backbone if not fine-tuning
    if not fine_tune:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)

    return model


# --------------------------------------------------
#  MONAI DenseNet
# --------------------------------------------------

def build_monai_densenet(num_classes):

    model = DenseNet121(
        spatial_dims=2,
        in_channels=3,      # bleibt 3 wegen ImageFolder (RGB)
        out_channels=num_classes,
        pretrained=True
    )

    return model

# --------------------------------------------------
# EfficientNet B0
# --------------------------------------------------
def build_efficientnet_b0(num_classes, pretrained=True, fine_tune=True):
    model = efficientnet_b0(weights="IMAGENET1K_V1" if pretrained else None)

    # Freeze backbone if not fine-tuning
    if not fine_tune:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model


# --------------------------------------------------
# EfficientNet B3
# --------------------------------------------------

def build_efficientnet_b3(num_classes, pretrained=True, fine_tune=True):

   
    weights = EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
    model = efficientnet_b3(weights=weights)

    # Freeze backbone
    for param in model.parameters():
        param.requires_grad = not fine_tune

    # Replace classifier
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model



# --------------------------------------------------
# ConvNeXt Tiny
# --------------------------------------------------

def build_convnext_tiny(num_classes, pretrained=True, fine_tune=True):
    from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

    weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
    model = convnext_tiny(weights=weights)

    # Freeze backbone if not fine-tuning
    if not fine_tune:
        for param in model.parameters():
            param.requires_grad = False


    # Replace classifier
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, num_classes)

    return model
