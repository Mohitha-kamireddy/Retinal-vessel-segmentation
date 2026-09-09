import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch


def _list_files(folder):
    return sorted(
        os.path.join(folder, fname)
        for fname in os.listdir(folder)
        if not fname.startswith(".")
    )


class DriveDataset(Dataset):
    def __init__(
        self,
        root_dir,
        split="training",
        images_dir="images",
        masks_dir="1st_manual",
        fov_dir="mask",
        transform=None,
    ):
        self.split = split
        self.transform = transform

        split_dir = os.path.join(root_dir, split)
        self.images_dir = os.path.join(split_dir, images_dir)
        self.masks_dir = os.path.join(split_dir, masks_dir)
        self.fov_dir = os.path.join(split_dir, fov_dir)

        self.image_paths = _list_files(self.images_dir)
        self.mask_paths = _list_files(self.masks_dir)
        self.fov_paths = _list_files(self.fov_dir)

        assert len(self.image_paths) == len(self.mask_paths) == len(self.fov_paths), (
            f"Mismatched number of files in '{split}' split: "
            f"{len(self.image_paths)} images, {len(self.mask_paths)} masks, "
            f"{len(self.fov_paths)} fov masks."
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("L")
        fov = Image.open(self.fov_paths[idx]).convert("L")

        image = np.array(image, dtype=np.uint8)
        mask = np.array(mask, dtype=np.uint8)
        fov = np.array(fov, dtype=np.uint8)

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask, fov=fov)
            image, mask, fov = augmented["image"], augmented["mask"], augmented["fov"]

        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy((mask > 0).astype(np.float32)).unsqueeze(0)
        fov = torch.from_numpy((fov > 0).astype(np.float32)).unsqueeze(0)

        return {"image": image, "mask": mask, "fov": fov}


if __name__ == "__main__":
    dataset = DriveDataset(root_dir="data/raw/DRIVE", split="training")
    print(f"Dataset size: {len(dataset)}")
    sample = dataset[0]
    print("image:", sample["image"].shape, sample["image"].dtype)
    print("mask :", sample["mask"].shape, sample["mask"].dtype)
    print("fov  :", sample["fov"].shape, sample["fov"].dtype)
