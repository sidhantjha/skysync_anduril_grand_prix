from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.tiny_cnn import TinyCNN


# ============================================================
# CONFIG
# ============================================================

DATASET_DIR = Path("phase1_cnn/crop_dataset")
OUTPUT_DIR = Path("phase1_cnn/outputs")
MODEL_PATH = OUTPUT_DIR / "tiny_cnn_best.pt"

IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 1e-3
NUM_WORKERS = 0

# ============================================================
# END CONFIG
# ============================================================


def get_dataloaders():
    transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
        ]
    )

    train_dataset = datasets.ImageFolder(DATASET_DIR / "train", transform=transform)
    val_dataset = datasets.ImageFolder(DATASET_DIR / "val", transform=transform)
    test_dataset = datasets.ImageFolder(DATASET_DIR / "test", transform=transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    return train_loader, val_loader, test_loader, train_dataset.classes


def run_one_epoch(model, loader, criterion, optimizer, device, train_mode: bool):
    if train_mode:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        if train_mode:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train_mode):
            logits = model(images)
            loss = criterion(logits, labels)

            if train_mode:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * images.size(0)

        preds = torch.argmax(logits, dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

        all_preds.extend(preds.detach().cpu().numpy().tolist())
        all_labels.extend(labels.detach().cpu().numpy().tolist())

    avg_loss = total_loss / total
    accuracy = correct / total

    return avg_loss, accuracy, np.array(all_preds), np.array(all_labels)


def plot_training_curves(train_losses, val_losses, train_accs, val_accs):
    plt.figure()
    plt.plot(train_losses, label="train loss")
    plt.plot(val_losses, label="val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("TinyCNN Loss Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "block_1_3_loss_curve.png")
    plt.close()

    plt.figure()
    plt.plot(train_accs, label="train accuracy")
    plt.plot(val_accs, label="val accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("TinyCNN Accuracy Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "block_1_3_accuracy_curve.png")
    plt.close()


def plot_confusion_matrix(preds, labels, class_names):
    num_classes = len(class_names)
    matrix = np.zeros((num_classes, num_classes), dtype=int)

    for true_label, pred_label in zip(labels, preds):
        matrix[true_label, pred_label] += 1

    plt.figure()
    plt.imshow(matrix)
    plt.title("TinyCNN Test Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")

    plt.xticks(range(num_classes), class_names, rotation=45)
    plt.yticks(range(num_classes), class_names)

    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(j, i, str(matrix[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "block_1_3_confusion_matrix.png")
    plt.close()

    return matrix


def save_prediction_grid(model, test_loader, class_names, device):
    model.eval()

    images, labels = next(iter(test_loader))
    images = images.to(device)

    with torch.no_grad():
        logits = model(images)
        preds = torch.argmax(logits, dim=1).cpu()

    images = images.cpu()

    num_images = min(16, images.size(0))

    plt.figure(figsize=(10, 10))

    for i in range(num_images):
        image = images[i].permute(1, 2, 0).numpy()

        true_name = class_names[labels[i].item()]
        pred_name = class_names[preds[i].item()]

        plt.subplot(4, 4, i + 1)
        plt.imshow(image)
        plt.axis("off")
        plt.title(f"T: {true_name}\nP: {pred_name}", fontsize=8)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "block_1_3_prediction_grid.png")
    plt.close()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Using device:")
    print(device)
    print()

    train_loader, val_loader, test_loader, class_names = get_dataloaders()

    print("Class names:")
    print(class_names)
    print()

    model = TinyCNN(num_classes=len(class_names)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_acc = 0.0

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(EPOCHS):
        train_loss, train_acc, _, _ = run_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            train_mode=True,
        )

        val_loss, val_acc, _, _ = run_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            train_mode=False,
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} | "
            f"train loss: {train_loss:.4f} | "
            f"train acc: {train_acc:.4f} | "
            f"val loss: {val_loss:.4f} | "
            f"val acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)

    print()
    print("Best validation accuracy:")
    print(f"{best_val_acc:.4f}")
    print()

    print("Loading best model for test evaluation...")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    test_loss, test_acc, test_preds, test_labels = run_one_epoch(
        model=model,
        loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        train_mode=False,
    )

    print()
    print("Test results:")
    print(f"test loss: {test_loss:.4f}")
    print(f"test acc:  {test_acc:.4f}")
    print()

    plot_training_curves(train_losses, val_losses, train_accs, val_accs)

    confusion_matrix = plot_confusion_matrix(
        preds=test_preds,
        labels=test_labels,
        class_names=class_names,
    )

    print("Confusion matrix:")
    print(confusion_matrix)
    print()

    save_prediction_grid(model, test_loader, class_names, device)

    print("Saved model:")
    print(MODEL_PATH)
    print()

    print("Saved outputs:")
    print(OUTPUT_DIR / "block_1_3_loss_curve.png")
    print(OUTPUT_DIR / "block_1_3_accuracy_curve.png")
    print(OUTPUT_DIR / "block_1_3_confusion_matrix.png")
    print(OUTPUT_DIR / "block_1_3_prediction_grid.png")


if __name__ == "__main__":
    main()