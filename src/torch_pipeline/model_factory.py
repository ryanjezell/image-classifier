from torch import nn
from torchvision.models import (
    ResNet34_Weights,
    ResNet50_Weights,
    EfficientNet_B0_Weights,
    EfficientNet_B3_Weights,
    resnet34,
    resnet50,
    efficientnet_b0,
    efficientnet_b3,
)


MODEL_REGISTRY = {
    "resnet34": (resnet34, ResNet34_Weights.DEFAULT),
    "resnet50": (resnet50, ResNet50_Weights.DEFAULT),
    "efficientnet_b0": (efficientnet_b0, EfficientNet_B0_Weights.DEFAULT),
    "efficientnet_b3": (efficientnet_b3, EfficientNet_B3_Weights.DEFAULT),
}


def get_supported_architectures():
    return sorted(MODEL_REGISTRY.keys())


def create_model(architecture: str, num_classes: int, pretrained: bool = True, dropout: float = 0.2):
    if architecture not in MODEL_REGISTRY:
        raise ValueError(f"Unknown architecture '{architecture}'. Available: {get_supported_architectures()}")

    ctor, default_weights = MODEL_REGISTRY[architecture]
    model = ctor(weights=default_weights if pretrained else None)

    if architecture.startswith("resnet"):
        in_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(in_features, num_classes))
    elif architecture.startswith("efficientnet"):
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)

    return model
