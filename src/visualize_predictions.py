import argparse
import os

import matplotlib.pyplot as plt
import torch
from torch.utils.data import Subset

from dataset import DriveDataset
from model import UNet
from transforms import get_val_transform


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
    parser.add_argument("--output-dir", type=str, default="outputs/predictions")
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--num-images", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = get_validation_dataset(args.data_root, args.val_split, args.seed)
    num_images = min(args.num_images, len(dataset))

    model = UNet(in_channels=3, out_channels=1).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)

    for idx in range(num_images):
        sample = dataset[idx]
        image = sample["image"].unsqueeze(0).to(device)
        mask = sample["mask"]
        fov = sample["fov"]

        with torch.no_grad():
            logits = model(image)
            pred = torch.sigmoid(logits)
            pred = (pred > args.threshold).float()

        image_np = sample["image"].permute(1, 2, 0).cpu().numpy()
        mask_np = mask.squeeze(0).cpu().numpy()
        fov_np = fov.squeeze(0).cpu().numpy()
        pred_np = pred.squeeze(0).squeeze(0).cpu().numpy()
        pred_np = pred_np * fov_np

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(image_np)
        axes[0].set_title("Retinal Image")
        axes[0].axis("off")

        axes[1].imshow(mask_np, cmap="gray")
        axes[1].set_title("Ground Truth")
        axes[1].axis("off")

        axes[2].imshow(pred_np, cmap="gray")
        axes[2].set_title("Prediction")
        axes[2].axis("off")

        output_path = os.path.join(args.output_dir, f"prediction_{idx}.png")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
