import os
import random
import subprocess
from pathlib import Path
import shutil
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


def download_rice_dataset(data_path: Path):
    if not data_path.exists():
        print("Downloading Rice Images dataset…")
        subprocess.run(
            f"curl -L -o rice-images-dataset.zip https://www.kaggle.com/api/v1/datasets/download/mbsoroush/rice-images-dataset",
            shell=True,
            check=True,
        )
        subprocess.run(
            f"unzip -q rice-images-dataset.zip -d rice_images", shell=True, check=True)
        os.remove("rice-images-dataset.zip")
        print("Dataset ready at", data_path.resolve())
    else:
        print("Dataset already present →", data_path.resolve())


def make_train_test_split(data_path):
    if not (data_path / "train").exists():
        print("[info] Creating train/test split...")
        all_classes = [d for d in (data_path).iterdir() if d.is_dir()]
        train_dir = data_path / "train"
        test_dir = data_path / "test"
        train_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)
        for class_dir in all_classes:
            images = list(class_dir.glob("*.jpg"))
            random.shuffle(images)
            split_idx = int(0.7 * len(images))
            train_imgs = images[:split_idx]
            test_imgs = images[split_idx:]
            cls_name = class_dir.name
            (train_dir / cls_name).mkdir(parents=True, exist_ok=True)
            (test_dir / cls_name).mkdir(parents=True, exist_ok=True)
            for img in train_imgs:
                shutil.copy(img, train_dir / cls_name / img.name)
            for img in test_imgs:
                shutil.copy(img, test_dir / cls_name / img.name)
        print("[info] Train/Test split created.")


def get_dataloaders(data_dir: Path, batch_size: int, img_size: int) -> tuple[DataLoader, DataLoader]:
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    train_data = ImageFolder(
        os.path.join(data_dir, "train"),
        transform=transform
    )
    test_data = ImageFolder(
        os.path.join(data_dir, "test"),
        transform=transform
    )
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=os.cpu_count(),
        persistent_workers=True
    )
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=os.cpu_count(),
        persistent_workers=True
    )

    return train_loader, test_loader
