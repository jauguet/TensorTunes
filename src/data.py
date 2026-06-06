import os
import librosa
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

GENRES = ["blues","classical","country","disco","hiphop",
          "jazz","metal","pop","reggae","rock"]

def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def get_dataloaders(spectro_path, batch_size=32, num_workers=2):
    train_dataset = datasets.ImageFolder(
        root=os.path.join(spectro_path, "train"),
        transform=get_transforms(train=True)
    )
    val_dataset = datasets.ImageFolder(
        root=os.path.join(spectro_path, "val"),
        transform=get_transforms(train=False)
    )
    test_dataset = datasets.ImageFolder(
        root=os.path.join(spectro_path, "test"),
        transform=get_transforms(train=False)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size,
                              shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader

def audio_to_melspectrogram(file_path, sr=22050, duration=3,
                             n_mels=128, hop_length=512):
    audio, _ = librosa.load(file_path, sr=sr, duration=duration)
    target_len = sr * duration
    if len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)))
    mel    = librosa.feature.melspectrogram(y=audio, sr=sr,
                                            n_mels=n_mels, hop_length=hop_length)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db

def save_spectrogram_image(mel_db, output_path):
    fig, ax = plt.subplots(figsize=(224/96, 224/96), dpi=96)
    ax.imshow(mel_db, aspect="auto", origin="lower", cmap="viridis")
    ax.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
