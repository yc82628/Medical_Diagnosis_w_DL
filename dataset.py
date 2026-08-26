from __future__ import annotations

from pathlib import Path

import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

from config import Config, CONFIG, PreprocessConfig, NUM_GRADES
from splits import load_labels, split_by_patient, ordinal_targets


# Augmentation
def build_transforms(split: str, pc: PreprocessConfig = CONFIG.preprocess):
    norm = A.Normalize(mean=pc.norm_mean, std=pc.norm_std)

    if split == "train":
        return A.Compose([
            A.RandomResizedCrop(size=(pc.image_size, pc.image_size),
                                scale=(0.85, 1.0), ratio=(0.95, 1.05), p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Affine(translate_percent=0.05, scale=(0.9, 1.1), rotate=(-180, 180),
                     border_mode=cv2.BORDER_CONSTANT, fill=0, p=0.7),
            A.RandomBrightnessContrast(brightness_limit=0.10, contrast_limit=0.10, p=0.5),
            norm, ToTensorV2(),
        ])
    return A.Compose([A.Resize(pc.image_size, pc.image_size), norm, ToTensorV2()])


# Dataset
class FundusDataset(Dataset):
    def __init__(self, df: pd.DataFrame, cache_dir: str, transform=None,
                 pc: PreprocessConfig = CONFIG.preprocess, verify_cache: bool = True):
        self.df = df.reset_index(drop=True)
        self.cache_dir = Path(cache_dir)
        self.transform = transform
        self.pc = pc

        if not self.cache_dir.exists():
            raise FileNotFoundError(
                f"Cache not found: {self.cache_dir}\n"
                f"Run:  python preprocessing.py --cache {self.cache_dir}"
            )
        if verify_cache:
            fp = self.cache_dir / ".fingerprint"
            if fp.exists() and fp.read_text().strip() != pc.fingerprint():
                raise RuntimeError(
                    f"Cache fingerprint mismatch.\n"
                    f"  cache built with: {fp.read_text().strip()}\n"
                    f"  current config:   {pc.fingerprint()}\n"
                    f"Preprocessing changed since the cache was built. Rebuild it. "
                    f"Training on stale preprocessing is a train/serve skew bug that "
                    f"produces no error message and a quietly worse model."
                )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = self.cache_dir / (Path(row["filename"]).stem + ".png")
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Cached image unreadable: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            img = self.transform(image=img)["image"]

        grade = int(row["grade"])
        return {
            "image": img,
            "ordinal": torch.from_numpy(ordinal_targets(grade)),
            "grade": torch.tensor(grade, dtype=torch.long),
            "referable": torch.tensor(int(row["referable"]), dtype=torch.long),
            "index": torch.tensor(idx, dtype=torch.long),
        }

# Entry points
def get_dataloaders(cfg: Config = CONFIG):
    df = load_labels(cfg.data)
    splits = split_by_patient(df, cfg.data)

    datasets = {
        s: FundusDataset(splits[s], cfg.data.cache_dir,
                         build_transforms(s, cfg.preprocess), cfg.preprocess)
        for s in ("train", "val", "test")
    }
    loaders = {
        s: DataLoader(
            datasets[s],
            batch_size=cfg.train.batch_size,
            shuffle=(s == "train"),
            num_workers=cfg.train.num_workers,
            pin_memory=True,
            drop_last=(s == "train"),
            persistent_workers=cfg.train.num_workers > 0,
        )
        for s in datasets
    }
    return loaders, splits, df


def pos_weights(train_df: pd.DataFrame) -> torch.Tensor:
    """Per-threshold positive weighting for the ordinal BCE loss."""
    w = []
    for k in range(NUM_GRADES - 1):
        pos = int((train_df["grade"] > k).sum())
        neg = len(train_df) - pos
        w.append(neg / max(pos, 1))
    return torch.tensor(w, dtype=torch.float32)
