"""
PyTorch dataset for the BraTS 2018/2019 2D slices.

Emits (sample_id, image, mask) so that every tensor stays traceable back to
the exact MRI slice it came from. The corruption ledger keys off sample_id,
so it has to survive the whole pipeline: file -> tensor -> batch -> model.

    from src.dataset import MRIDataset, build_loader

    train_ds = MRIDataset(split="train", augment=True)
    loader   = build_loader("train", batch_size=16, augment=True)

Shapes (per sample):
    image  float32  [1, 240, 240]
    mask   float32  [1, 240, 240]   values in {0.0, 1.0}

Batched, that is [B, C, H, W] as PyTorch expects.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

MANIFEST = "data/processed/manifest.csv"
SPLIT_FILE = "splits/split_v1.json"

# Masks arrive as 8-bit PNGs holding {0, 255}. Anything above this threshold
# is tumour. A threshold rather than "== 255" keeps us safe if a mask ever
# picks up interpolation artefacts upstream.
MASK_THRESHOLD = 127


class MRIDataset(Dataset):
    """
    One item = one axial slice and its binary tumour mask.

    Args:
        split:      "train" | "val" | "test"
        normalize:  "zscore" -> per-slice zero mean, unit variance (default;
                    standard for MRI, whose intensities have no absolute
                    scale and vary between scanners)
                    "unit"   -> scale to [0, 1]; use this if you later need
                    a pixel-space perturbation budget
        augment:    apply paired flips/rotations. Train split only.
        min_tumor_pixels: drop slices with fewer tumour pixels than this.
                    0 keeps everything.
    """

    def __init__(self, split, normalize="zscore", augment=False,
                 min_tumor_pixels=0, manifest=MANIFEST, split_file=SPLIT_FILE,
                 root="."):
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unknown split: {split}")
        if normalize not in {"zscore", "unit"}:
            raise ValueError(f"Unknown normalize mode: {normalize}")
        if augment and split != "train":
            raise ValueError("Augmentation must not be applied to val or test")

        self.root = Path(root)
        self.split = split
        self.normalize = normalize
        self.augment = augment

        splits = json.loads((self.root / split_file).read_text())
        wanted = set(splits["samples"][split])

        df = pd.read_csv(self.root / manifest)
        df = df[df["sample_id"].isin(wanted)]
        if min_tumor_pixels > 0:
            df = df[df["tumor_pixels"] >= min_tumor_pixels]

        self.records = df.reset_index(drop=True)
        self.patients = sorted(self.records["patient_id"].unique())

    def __len__(self):
        return len(self.records)

    # -- loading ---------------------------------------------------------

    def _load_pair(self, row):
        image = np.array(Image.open(self.root / row["image_path"]), dtype=np.float32)
        mask = np.array(Image.open(self.root / row["mask_path"]))
        if mask.ndim == 3:
            mask = mask[..., 0]
        return image, (mask > MASK_THRESHOLD).astype(np.float32)

    def _normalize(self, image):
        if self.normalize == "unit":
            lo, hi = image.min(), image.max()
            return (image - lo) / (hi - lo) if hi > lo else np.zeros_like(image)

        std = image.std()
        # A constant slice has no information; return zeros rather than NaNs.
        return (image - image.mean()) / std if std > 0 else np.zeros_like(image)

    @staticmethod
    def _augment(image, mask, rng):
        """
        Flips and 90-degree rotations only.

        These are exact pixel permutations, so the mask stays perfectly binary.
        Arbitrary-angle rotation would need interpolation, which invents
        non-binary values in the mask and quietly corrupts the labels.
        """
        if rng.random() < 0.5:
            image, mask = np.fliplr(image), np.fliplr(mask)
        if rng.random() < 0.5:
            image, mask = np.flipud(image), np.flipud(mask)
        k = rng.integers(0, 4)
        if k:
            image, mask = np.rot90(image, k), np.rot90(mask, k)
        return np.ascontiguousarray(image), np.ascontiguousarray(mask)

    def __getitem__(self, idx):
        row = self.records.iloc[idx]
        image, mask = self._load_pair(row)
        image = self._normalize(image)

        if self.augment:
            image, mask = self._augment(image, mask, np.random.default_rng())

        # add the channel dimension -> [1, H, W]
        return (
            row["sample_id"],
            torch.from_numpy(image).unsqueeze(0).float(),
            torch.from_numpy(mask).unsqueeze(0).float(),
        )


def build_loader(split, batch_size=16, num_workers=0, augment=False, **kwargs):
    """DataLoader with train-appropriate shuffling and drop_last."""
    dataset = MRIDataset(split=split, augment=augment, **kwargs)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        drop_last=(split == "train"),
        pin_memory=False,      # MPS does not benefit from pinned memory
    )


# ------------------------------------------------------------------------

if __name__ == "__main__":
    print("Verifying dataset and loaders\n" + "=" * 58)

    for split in ("train", "val", "test"):
        ds = MRIDataset(split=split)
        print(f"{split:<6} {len(ds):>6} slices   {len(ds.patients):>4} patients")

    # No patient may appear in two splits.
    sets = {s: set(MRIDataset(split=s).patients) for s in ("train", "val", "test")}
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        assert not (sets[a] & sets[b]), f"Patient leakage: {a} vs {b}"
    print("\nNo patient leakage between splits.")

    loader = build_loader("train", batch_size=8, augment=True)
    sample_ids, images, masks = next(iter(loader))

    print("\nOne training batch")
    print("-" * 58)
    print(f"images      {tuple(images.shape)}  {images.dtype}")
    print(f"masks       {tuple(masks.shape)}  {masks.dtype}")
    print(f"image range [{images.min():.3f}, {images.max():.3f}]  "
          f"mean {images.mean():.3f}  std {images.std():.3f}")
    print(f"mask values {sorted(torch.unique(masks).tolist())}")
    print(f"tumour px   {masks.sum().item():.0f} of {masks.numel()} "
          f"({100 * masks.mean().item():.3f}%)")
    print(f"sample ids  {list(sample_ids[:3])}")

    assert images.shape == (8, 1, 240, 240), "Unexpected image shape"
    assert masks.shape == (8, 1, 240, 240), "Unexpected mask shape"
    assert set(torch.unique(masks).tolist()) <= {0.0, 1.0}, "Mask is not binary"
    print("\nAll checks passed.")
