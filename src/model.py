import torch
import torch.nn as nn
from torchvision import models

def build_model(num_classes=10, freeze_backbone=True):
    """ResNet18 pretrained on ImageNet, fine-tuned for GTZAN."""
    model = models.resnet18(weights="IMAGENET1K_V1")

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Linear(256, num_classes)
    )
    return model
