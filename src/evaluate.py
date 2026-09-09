import argparse
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from dataset import DriveDataset
from model import UNet
from transforms import get_val_transform


def update_confusion_counts(preds, targets, fov, counts, threshold):
    valid = fov > 0.5
    preds = (preds > threshold).float()

    preds = preds[valid]
    targets = targets[valid]

    counts["tp"] += ((preds == 1) & (targets == 1)).sum().item()
    counts["tn"] += ((preds == 0) & (targets == 0)).sum().item()
    counts["fp"] += ((preds == 1) & (targets == 0)).sum().item()
    counts["fn"] += ((preds == 0) & (targets == 1)).sum().item()


def compute_metrics(counts):
    tp = counts["tp"]
    tn = counts["tn"]
    fp = counts["fp"]
    fn = counts["fn"]

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "dice": dice,
        "iou": iou,
    }


def evaluate(model, loader, device, threshold):
    model.eval()
    counts = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            fov = batch["fov"].to(device)

            logits = model(images)
            preds = torch.sigmoid(logits)

            update_confusion_counts(preds, masks, fov, counts, threshold)

    return compute_metrics(counts)


def get_validation_dataset(data_root, val_split, seed):
    full_dataset = DriveDataset(root_dir=data_root, split="training", transform=get_val_transform())
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(full_dataset), generator=generator).tolist()
    val_indices = indices[train_size:]
    return Subset(full_dataset, val_indices)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="data/raw/DRIVE")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/best_model.pth")
    parser.add_argument("--output", type=str, default="outputs/metrics.csv")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = get_validation_dataset(args.data_root, args.val_split, args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    model = UNet(in_channels=3, out_channels=1).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    metrics = evaluate(model, loader, device, args.threshold)

    print(f"Evaluation on training-set validation split ({len(dataset)} images):")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df = pd.DataFrame(list(metrics.items()), columns=["metric", "value"])
    df.to_csv(args.output, index=False)
    print(f"Saved metrics to {args.output}")


if __name__ == "__main__":
    main()
