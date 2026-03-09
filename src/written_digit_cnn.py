import click
from pathlib import Path

import lightning as L
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchmetrics.classification import MulticlassConfusionMatrix

from cnn_model import SmallCNN

import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2

x_step = np.linspace(0, 7, 7)
x_epoch = np.linspace(0,10,10)
y_loss = []
y_validation = []


#   _____                            _              _
#  |_   _|                          | |            | |
#    | |  _ __ ___  _ __   ___  _ __| |_ __ _ _ __ | |_
#    | | | '_ ` _ \| '_ \ / _ \| '__| __/ _` | '_ \| __|
#   _| |_| | | | | | |_) | (_) | |  | || (_| | | | | |_
#  |_____|_| |_| |_| .__/ \___/|_|   \__\__,_|_| |_|\__|
#                  | |
#                  |_|
# YOUR NAME: Colin Truong
# YOUR WPI ID: 997075740



# ---------------------------
# Dataset
# ---------------------------
class DigitImageDataset(Dataset):
    """
    Loads images and labels from .npy files.
    """
    def __init__(self, images_path, labels_path):
        self.images = np.load(images_path)  # (N, 1, 28, 28)
        self.labels = np.load(labels_path).astype(np.int64)  # (N,)

    def __len__(self):
        #  __   __                  ____          _         _   _
        #  \ \ / /__  _   _ _ __   / ___|___   __| | ___   | | | | ___ _ __ ___
        #   \ V / _ \| | | | '__| | |   / _ \ / _` |/ _ \  | |_| |/ _ \ '__/ _ \
        #    | | (_) | |_| | |    | |__| (_) | (_| |  __/  |  _  |  __/ | |  __/
        #    |_|\___/ \__,_|_|     \____\___/ \__,_|\___|  |_| |_|\___|_|  \___|
        return len(self.labels)

    """
    Return a tuple of (image_tensor, label)
        image_tensor: (1, 28, 28) float32
        label: int in [0..9]
    
    If the pixel values are not in [0, 1], normalize them accordingly.
    """
    def __getitem__(self, idx):
        #  __   __                  ____          _         _   _
        #  \ \ / /__  _   _ _ __   / ___|___   __| | ___   | | | | ___ _ __ ___
        #   \ V / _ \| | | | '__| | |   / _ \ / _` |/ _ \  | |_| |/ _ \ '__/ _ \
        #    | | (_) | |_| | |    | |__| (_) | (_| |  __/  |  _  |  __/ | |  __/
        #    |_|\___/ \__,_|_|     \____\___/ \__,_|\___|  |_| |_|\___|_|  \___|
        img = torch.from_numpy(self.images[idx]).float()
        label = torch.tensor(self.labels[idx])
        return img, label


# ---------------------------
# Data prep
# ---------------------------
def prepare_dataloaders(data_dir, batch_size, fabric):
    train_images = data_dir / "train_images.npy"
    train_labels = data_dir / "train_labels.npy"
    val_images = data_dir / "val_images.npy"
    val_labels = data_dir / "val_labels.npy"

    train_ds = DigitImageDataset(train_images, train_labels)
    val_ds = DigitImageDataset(val_images, val_labels)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)

    # Fabric places/parallelizes DataLoaders as needed
    train_loader, val_loader = fabric.setup_dataloaders(train_loader, val_loader)
    return train_loader, val_loader


# ---------------------------
# Train & Eval for one epoch
# ---------------------------
def train_one_epoch(fabric, model, loader, criterion, optimizer):
    #   _____            _
    #  |  __ \          (_)
    #  | |__) |_____   ___  _____      __
    #  |  _  // _ \ \ / / |/ _ \ \ /\ / /
    #  | | \ \  __/\ V /| |  __/\ V  V /
    #  |_|  \_\___| \_/ |_|\___| \_/\_/

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for step, (images, labels) in enumerate(loader, start=1):
        optimizer.zero_grad()

        # move to device managed by Fabric
        images, labels = fabric.to_device((images, labels))
        # images: (B,1,28,28) -> (B,28,28)
        images = images.squeeze(1)

        logits = model(images)
        loss = criterion(logits, labels)

        fabric.backward(loss)
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        if step % 10 == 0:
            avg_loss = running_loss / max(total, 1)
            acc = correct / max(total, 1)
            fabric.print(f"Step {step:04d} - train_loss={avg_loss:.4f} - train_acc={acc:.4f}")
            y_loss.append(avg_loss)
        


    epoch_loss = running_loss / max(total, 1)
    epoch_acc = correct / max(total, 1)
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(fabric, model, loader, criterion):
    #   _____            _
    #  |  __ \          (_)
    #  | |__) |_____   ___  _____      __
    #  |  _  // _ \ \ / / |/ _ \ \ /\ / /
    #  | | \ \  __/\ V /| |  __/\ V  V /
    #  |_|  \_\___| \_/ |_|\___| \_/\_/

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    confmat_metric = MulticlassConfusionMatrix(num_classes=10).to(fabric.device)

    for images, labels in loader:
        images, labels = fabric.to_device((images, labels))
        images = images.squeeze(1)  # (B,28,28)

        logits = model(images)
        loss = criterion(logits, labels)

        running_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        confmat_metric.update(preds, labels)

    val_loss = running_loss / max(total, 1)
    val_acc = correct / max(total, 1)
    confmat = confmat_metric.compute().cpu()

    return val_loss, val_acc, confmat


# ⚠️ THE AG REQUIRES --data-dir --epochs --ckpt.
# You need to provide default values for any other arguments.
# The AG will run training for 10 epochs only, make sure you archieve a decent training accuracy by then.
@click.command()
@click.option("--data-dir", type=click.Path(exists=True, file_okay=False), default="../data/img_data/", show_default=True, help="Path to data directory.")
@click.option("--batch-size", type=int, default=64, show_default=True, help="Batch size for training.")
@click.option("--epochs", type=int, default=10, show_default=True, help="Number of training epochs.")
@click.option("--lr", type=float, default=1e-3, show_default=True, help="Learning rate.")
@click.option("--weight-decay", type=float, default=1e-5, show_default=True, help="Weight decay (L2 penalty).")
@click.option("--ckpt", type=click.Path(file_okay=True), default="./cnn.ckpt", show_default=True, help="Path to save checkpoint.")
def main(data_dir, batch_size, epochs, lr, weight_decay, ckpt):
    data_dir = Path(data_dir)
    ckpt_path = Path(ckpt) # ⚠️ This is where the best model will be saved, DO NOT change the checkpoint name.
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    L.seed_everything(42) # for reproducibility only, you CAN choose your own seed.

    # Initialize Fabric
    fabric = L.Fabric(accelerator="cpu") # ⚠️ AG only has CPU.
    fabric.launch()

    # Data
    train_loader, val_loader = prepare_dataloaders(data_dir, batch_size, fabric)

    # Model/optim/criterion
    model = SmallCNN(w=28, h=28, num_classes=10)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    # Let Fabric set up model & optimizer
    model, optimizer = fabric.setup(model, optimizer)

    #   _____            _
    #  |  __ \          (_)
    #  | |__) |_____   ___  _____      __
    #  |  _  // _ \ \ / / |/ _ \ \ /\ / /
    #  | | \ \  __/\ V /| |  __/\ V  V /
    #  |_|  \_\___| \_/ |_|\___| \_/\_/
    # Training loop (do your training, validation, logging and plotting, checkpoint saving here)

    best_val_acc = 0.0
    best_confmat = None

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(fabric, model, train_loader, criterion, optimizer)
        plt.plot(x_step, y_loss)
        plt.title("Loss vs. Training Step (" + str(epoch) + ")")
        plt.xlabel("Training Step")
        plt.ylabel("Train Loss")
        # plt.show()
        y_loss.clear()
        val_loss, val_acc, confmat = evaluate(fabric, model, val_loader, criterion)
        y_validation.append(val_acc)
        fabric.print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_confmat = confmat
            fabric.print(f"New best val_acc={best_val_acc:.4f}, saving checkpoint to {ckpt_path}")
            fabric.save(ckpt_path, {"model": model.state_dict()})
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "model_state_dict": model.state_dict(),
                    "model": model.state_dict(),
                },
                ckpt_path,
            )
        

    plt2.plot(x_epoch, y_validation)
    plt2.title("Validation Accuracy vs. #Epoch")
    plt2.xlabel("Epoch #")
    plt2.ylabel("Validation Accuracy")
    # plt2.show()

    fabric.print(f"Best validation accuracy: {best_val_acc:.4f}")
    fabric.print("Confusion matrix for best checkpoint:")
    fabric.print(best_confmat) # Optionally, you may generate a heatmap plot for the confusion matrix using
                               #    matplotlib or seaborn for your report.

if __name__ == "__main__":
    main()
