import argparse
import json
from pathlib import Path

import torch
from callbacks import MemoryLogger
from dataset import download_rice_dataset, get_dataloaders, make_train_test_split
from lightning_model import LitClassifier

from model import SmallCNN
import torch.optim as optim
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl

SEED = 52
IMG_SIZE = 64
BATCH_SIZE = 64
EPOCHS = 2
DATA_ROOT = Path("./rice_images")
RESULTS_DIR = Path("./results")

optimizer_classes = [
    ('SGD', optim.SGD, {"momentum": 0.0}),
    ('SGD momentum ', optim.SGD, {"momentum": 0.9}),
    ('Adam', optim.Adam, dict()),
    ('AdamW', optim.AdamW, dict()),
    ('Adagrad', optim.Adagrad, dict())
]


def train_model(
    optimizer_class,
    opt_conf,
    train_loader,
    test_loader,
    epochs,
    results_dir,
):
    if torch.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    num_classes = len(train_loader.dataset.classes)
    model = SmallCNN(num_classes=num_classes)
    classifier = LitClassifier(
        model,
        optimizer_class=optimizer_class,
        opt_conf=opt_conf,
        num_classes=num_classes
    )

    logger = TensorBoardLogger(
        results_dir,
        name=f"opt_{optimizer_class.__name__}"
    )

    trainer = pl.Trainer(
        max_epochs=epochs,
        logger=logger,
        enable_progress_bar=True,
        deterministic=True,
        callbacks=[MemoryLogger()],
        val_check_interval=0.5,
    )
    trainer.fit(classifier, train_loader, test_loader)

    result = {}

    weight_norm = trainer.logged_metrics.get("train_weight_norm", None)
    result["train_weight_norm"] = weight_norm.item() if weight_norm is not None else None

    mem_usage_MB = trainer.logged_metrics.get("mem_usage_MB", None)
    result["mem_usage_MB"] = mem_usage_MB.item() if mem_usage_MB is not None else None

    val_mertics = trainer.validate(
        classifier,
        dataloaders=test_loader,
        verbose=True
    )[0]
    result.update(val_mertics)

    return result


def run_experiment():
    pl.seed_everything(SEED, workers=True)

    download_rice_dataset(DATA_ROOT)
    make_train_test_split(DATA_ROOT)
    train_loader, test_loader = get_dataloaders(
        DATA_ROOT,
        BATCH_SIZE,
        IMG_SIZE
    )
    results = {}

    for name, opt_cls, opt_conf in optimizer_classes:
        print(f"\nTraining with optimizer: {name}")

        result = train_model(
            opt_cls, opt_conf,
            train_loader, test_loader,
            EPOCHS, RESULTS_DIR
        )
        results[name] = {
            "val_f1": result["val_f1"],
            "val_loss": result["val_loss"],
            "val_acc": result["val_acc"],
            "val_precision": result["val_precision"],
            "val_recall": result["val_recall"],
            "train_weight_norm": result["train_weight_norm"],
        }

        print(
            f"Test F1: {result['val_f1']:.4f}, Test Loss: {result['val_loss']:.4f}"
        )

    print("\nResults:")
    for name, (f1, loss) in results.items():
        print(f"{name}: F1: {f1:.4f}, Loss: {loss:.4f}")

    with open(RESULTS_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=4)
        print(f"Results saved to {RESULTS_DIR / 'results.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a CNN on rice images")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=DATA_ROOT,
        help="Path to the dataset directory"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=RESULTS_DIR,
        help="Path to the results directory"
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=IMG_SIZE,
        help="Image size for training"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=BATCH_SIZE,
        help="Batch size for training"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help="Number of epochs for training"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Random seed for reproducibility"
    )
    args = parser.parse_args()
    DATA_ROOT = Path(args.data_dir)
    RESULTS_DIR = Path(args.results_dir)
    IMG_SIZE = args.img_size
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs

    print(f"Using seed: {SEED}")
    print(f"Using data directory: {DATA_ROOT}")
    print(f"Using results directory: {RESULTS_DIR}")
    print(f"Using image size: {IMG_SIZE}")
    print(f"Using batch size: {BATCH_SIZE}")
    print(f"Using epochs: {EPOCHS}")
    print(f"Using optimizers: {optimizer_classes}")
    print(f"Using results directory: {RESULTS_DIR}")

    run_experiment()
