import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import os
import sys

# Config
CHECKPOINT_PATH = "checkpoints/best.pt"
SPECTRO_TEST_PATH = "./gtzan/spectrograms/test"
BATCH_SIZE = 32
NUM_CLASSES = 10
SEED = 42

torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {device}")

# Transforms
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Dataset
test_dataset = datasets.ImageFolder(
    root=SPECTRO_TEST_PATH,
    transform=val_transforms
)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2
)

# Modèle
def build_model(num_classes=NUM_CLASSES):
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Linear(256, num_classes)
    )
    return model

model = build_model().to(device)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
model.eval()
print(f"Checkpoint chargé : {CHECKPOINT_PATH}")

# Évaluation
correct, total = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        correct += (outputs.argmax(1) == labels).sum().item()
        total   += labels.size(0)

acc = correct / total
print(f"Test Accuracy : {acc:.4f} ({acc*100:.1f}%)")
