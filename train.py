import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import yaml
import json

from src.model import build_model
from src.data import get_dataloaders
from src.utils import train_epoch, eval_epoch

# Load config
with open("configs/default.yaml", "r") as f:
    cfg = yaml.safe_load(f)

# Seed
SEED = cfg["seed"]
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {device} | Seed : {SEED}")

# Data
train_loader, val_loader, _ = get_dataloaders(
    cfg["data"]["spectro_path"],
    batch_size=cfg["data"]["batch_size"],
    num_workers=cfg["data"]["num_workers"]
)

# Model
model     = build_model(num_classes=cfg["model"]["num_classes"],
                        freeze_backbone=True).to(device)
criterion = nn.CrossEntropyLoss()
os.makedirs("checkpoints", exist_ok=True)
best_val_acc = 0

# Phase 1
optimizer1 = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=cfg["training"]["phase1_lr"]
)
scheduler1 = optim.lr_scheduler.StepLR(
    optimizer1,
    step_size=cfg["training"]["scheduler_step"],
    gamma=cfg["training"]["scheduler_gamma"]
)

print("=" * 50)
print("PHASE 1 — Feature Extraction")
print("=" * 50)
for epoch in range(1, cfg["training"]["phase1_epochs"] + 1):
    tl, ta = train_epoch(model, train_loader, optimizer1, criterion, device)
    vl, va = eval_epoch(model, val_loader, criterion, device)
    scheduler1.step()
    if va > best_val_acc:
        best_val_acc = va
        torch.save(model.state_dict(), cfg["paths"]["checkpoint"])
    print(f"Epoch {epoch:02d} | Train Acc: {ta:.3f} | Val Acc: {va:.3f}")

# Phase 2
for name, param in model.named_parameters():
    if "layer3" in name or "layer4" in name or "fc" in name:
        param.requires_grad = True

optimizer2 = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=cfg["training"]["phase2_lr"]
)
scheduler2 = optim.lr_scheduler.StepLR(
    optimizer2,
    step_size=cfg["training"]["scheduler_step"],
    gamma=cfg["training"]["scheduler_gamma"]
)

print("=" * 50)
print("PHASE 2 — Fine-tuning")
print("=" * 50)
for epoch in range(1, cfg["training"]["phase2_epochs"] + 1):
    tl, ta = train_epoch(model, train_loader, optimizer2, criterion, device)
    vl, va = eval_epoch(model, val_loader, criterion, device)
    scheduler2.step()
    if va > best_val_acc:
        best_val_acc = va
        torch.save(model.state_dict(), cfg["paths"]["checkpoint"])
        print(f"  💾 Checkpoint saved (val_acc={va:.3f})")
    print(f"Epoch {epoch:02d} | Train Acc: {ta:.3f} | Val Acc: {va:.3f}")

print(f"\nBest Val Accuracy : {best_val_acc:.4f}")
print(f"Checkpoint saved  : {cfg['paths']['checkpoint']}")
