from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

from config import Config, CONFIG, PreprocessConfig, NUM_GRADES
from preprocessing import cache_name
from splits import load_labels, split_by_patient, ordinal_targets


# Augmentation
class FieldOfViewBandCrop(A.ImageOnlyTransform):
    """
    Black out a band at the top and bottom of the image.

    Wide-sensor fundus cameras (IDRiD's Kowa VX-10, 4288x2848) cut the circular
    field of view off at the top and bottom; DeepDRiD's are mostly full circles.
    Without this a model trained on DeepDRiD has never seen a flat-edged field
    of view, and the bright rim the normalisation leaves along that edge.
    Applied before rotation, so the band can end up at any angle.
    """

    def __init__(self, max_frac: float = 0.18, rim: bool = True, p: float = 0.5):
        super().__init__(p=p)
        self.max_frac = max_frac
        self.rim = rim

    def apply(self, img, top=0, bottom=0, rim_value=0, **params):
        out = img.copy()
        h = out.shape[0]
        if top:
            out[:top] = 0
        if bottom:
            out[h - bottom:] = 0
        if self.rim and rim_value:
            # A thin light edge, like the halo Ben Graham leaves at a hard cut.
            # Only inside the field of view: the background stays black.
            for edge in ([top] if top else []) + ([h - bottom - 3] if bottom else []):
                band = out[edge:edge + 3]
                inside = band.max(axis=-1, keepdims=True) > 0
                lit = np.clip(band.astype(np.int16) + rim_value, 0, 255).astype(out.dtype)
                out[edge:edge + 3] = np.where(inside, lit, band)
        return out

    def get_params_dependent_on_data(self, params, data):
        # self.py_random is albumentations' seeded generator (set_random_seed).
        h = data["image"].shape[0]
        top = self.py_random.uniform(0.02, self.max_frac)
        bottom = min(top * self.py_random.uniform(0.7, 1.3), self.max_frac)
        return {"top": int(top * h), "bottom": int(bottom * h),
                "rim_value": self.py_random.randint(0, 60)}

    def get_transform_init_args_names(self):
        return ("max_frac", "rim")


class KeepBackground(A.ImageOnlyTransform):
    """
    Run photometric transforms on the field of view only.

    Brightness, noise and colour shifts would otherwise turn the black
    background grey -- something no real preprocessed image has, and so a cue
    that separates training images from everything the model meets later.
    """

    def __init__(self, transforms, threshold: int = 4, p: float = 1.0):
        super().__init__(p=p)
        self.inner = A.Compose(transforms)
        self.threshold = threshold

    def apply(self, img, **params):
        background = img.max(axis=-1) <= self.threshold
        out = self.inner(image=img)["image"]
        out[background] = 0
        return out

    def get_transform_init_args_names(self):
        return ("threshold",)


AUGMENTATIONS = ("mild", "strong")


def build_transforms(split: str, pc: PreprocessConfig = CONFIG.preprocess,
                     augment: str = "mild"):
    """
    `augment` picks the training profile.

    "mild" (the original): geometry plus +-10% brightness/contrast.
    "strong": targets the camera shift measured on IDRiD (see MODEL_CARD.md,
    "Domain-shift experiments") -- a truncated field of view, sharpness in
    both directions, small colour-balance shifts and compression artefacts.
    Hue moves at most +-4 degrees: haemorrhage red vs exudate yellow is signal.
    """
    norm = A.Normalize(mean=pc.norm_mean, std=pc.norm_std)
    if augment not in AUGMENTATIONS:
        raise ValueError(f"Unknown augmentation {augment!r}. Options: {AUGMENTATIONS}")

    if split == "train":
        geometry = [
            A.RandomResizedCrop(size=(pc.image_size, pc.image_size),
                                scale=(0.85, 1.0), ratio=(0.95, 1.05), p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Affine(translate_percent=0.05, scale=(0.9, 1.1), rotate=(-180, 180),
                     border_mode=cv2.BORDER_CONSTANT, fill=0, p=0.7),
        ]
        if augment == "mild":
            photometric = [
                A.RandomBrightnessContrast(brightness_limit=0.10, contrast_limit=0.10, p=0.5),
            ]
        else:
            geometry.insert(0, FieldOfViewBandCrop(max_frac=0.18, p=0.5))
            photometric = [
                A.RandomBrightnessContrast(brightness_limit=0.20, contrast_limit=0.20, p=0.7),
                A.RandomGamma(gamma_limit=(80, 120), p=0.3),
                A.HueSaturationValue(hue_shift_limit=4, sat_shift_limit=20,
                                     val_shift_limit=10, p=0.5),
                A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.3),
                A.OneOf([
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                    A.Defocus(radius=(1, 3), p=1.0),
                    A.Sharpen(alpha=(0.1, 0.4), lightness=(0.9, 1.1), p=1.0),
                    A.Downscale(scale_range=(0.5, 0.85), p=1.0),
                ], p=0.5),
                A.GaussNoise(std_range=(0.01, 0.04), p=0.2),
                A.ImageCompression(quality_range=(50, 95), p=0.3),
            ]
            photometric = [KeepBackground(photometric)]
        return A.Compose([*geometry, *photometric, norm, ToTensorV2()])
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
        path = self.cache_dir / cache_name(row["filename"])
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
                         build_transforms(s, cfg.preprocess, cfg.train.augment),
                         cfg.preprocess)
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
