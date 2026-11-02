"""
Training-data poisoning for brain-tumour segmentation.

Implements the A1 attack described in docs/threat_model.md. The attacker owns
the annotation pipeline only: masks may be corrupted, images may not be
touched, and the validation and test splits are off limits.

The objective is STEALTH, not destruction. A poisoned mask should still look
like a competent segmentation - one smooth connected region of plausible size
sitting on a real lesion - while its boundary is wrong.

Usage:
    python src/poison.py --method shift --rate 0.10
    python src/poison.py --method grow_shrink --rate 0.05 --seed 7

Writes:
    data/processed/poison/<run_id>/masks/*.png     corrupted masks
    data/processed/poison/<run_id>/manifest.csv    drop-in replacement manifest
    data/processed/poison/<run_id>/ledger.csv      ground truth: who was hit
    data/processed/poison/<run_id>/config.json     run parameters

Train on a poisoned set by pointing the loader at the new manifest:

    MRIDataset(split="train", manifest=".../poison/<run_id>/manifest.csv")

The ledger is ground truth for evaluating detection and localisation. It must
never be visible to the defender.
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
from tqdm import tqdm

try:                                    # python -m src.poison
    from src.plausibility import brain_mask
except ImportError:                     # python src/poison.py
    from plausibility import brain_mask

MANIFEST = "data/processed/manifest.csv"
SPLIT_FILE = "splits/split_v1.json"
POISON_ROOT = "data/processed/poison"

METHODS = ("shift", "grow_shrink", "blank")


# --------------------------------------------------------------------------
# corruptions
# --------------------------------------------------------------------------

def _disk(radius):
    """Circular structuring element, so dilation stays isotropic."""
    size = 2 * radius + 1
    yy, xx = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    return (yy ** 2 + xx ** 2) <= radius ** 2 * 1.0 + 1e-9 if size else np.ones((1, 1), bool)


def shift_mask(mask, rng, brain, min_px=3, max_px=10, max_tries=20):
    """
    Translate the mask by a small random offset, kept inside the brain.

    Size, shape and smoothness are preserved exactly - only the position moves.
    Offsets that would push part of the mask into the black background are
    rejected and redrawn: spilling outside the brain is the one thing that
    makes a shifted mask visibly fake, so the attack must avoid it.
    """
    binary = mask.astype(bool)
    outside = ~brain.astype(bool)
    best = best_params = None
    best_spill = None

    for _ in range(max_tries):
        magnitude = int(rng.integers(min_px, max_px + 1))
        angle = rng.uniform(0, 2 * np.pi)
        dy = int(round(magnitude * np.sin(angle)))
        dx = int(round(magnitude * np.cos(angle)))

        shifted = ndimage.shift(binary.astype(np.float32), shift=(dy, dx),
                                order=0, mode="constant", cval=0.0) > 0.5
        spill = int((shifted & outside).sum())
        params = {"dy": dy, "dx": dx, "magnitude": magnitude,
                  "tries": _ + 1, "clipped": False}

        if spill == 0:                       # fully inside the brain, done
            return shifted.astype(np.uint8), params
        if best_spill is None or spill < best_spill:
            best, best_params, best_spill = shifted, params, spill

    # No fully-contained offset exists - usually a tumour nearly filling a
    # small brain cross-section. Clip the least-bad attempt to the brain.
    # This no longer preserves shape exactly, so it is recorded.
    clipped = best & brain.astype(bool)
    if not clipped.any():                    # clipping removed everything
        return mask.astype(np.uint8), {"dy": 0, "dx": 0, "magnitude": 0,
                                       "tries": max_tries, "clipped": True,
                                       "failed": True}
    best_params["clipped"] = True
    return clipped.astype(np.uint8), best_params

def grow_shrink_mask(mask, rng, brain, min_r=2, max_r=6):
    """
    Dilate or erode the mask by a morphological radius.

    Dilation is clipped to the brain, so a grown mask never bleeds into the
    black background. Erosion only shrinks and so can never leave the brain.

    Erosion that would empty the mask is retried at smaller radii and finally
    falls back to dilation: an empty mask is a different (obvious) attack, and
    would break the stealth constraint.
    """
    radius = int(rng.integers(min_r, max_r + 1))
    grow = bool(rng.integers(0, 2))
    binary = mask.astype(bool)
    inside = brain.astype(bool)

    if grow:
        out = ndimage.binary_dilation(binary, structure=_disk(radius)) & inside
    else:
        out = ndimage.binary_erosion(binary, structure=_disk(radius))
        while not out.any() and radius > 1:          # too aggressive, back off
            radius -= 1
            out = ndimage.binary_erosion(binary, structure=_disk(radius))
        if not out.any():                            # tiny lesion, grow instead
            grow, radius = True, min_r
            out = ndimage.binary_dilation(binary, structure=_disk(radius)) & inside

    return out.astype(np.uint8), {"radius": radius, "op": "dilate" if grow else "erode"}

def blank_mask(mask, rng, brain):
    """
    Control condition only. Crude and trivially detectable - see §5.3.

    Takes `brain` to match the other corruptions' signature (they are all
    called through the CORRUPTIONS dict) but has no use for it.
    """
    return np.zeros_like(mask, dtype=np.uint8), {}


CORRUPTIONS = {"shift": shift_mask, "grow_shrink": grow_shrink_mask, "blank": blank_mask}


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def dice(a, b):
    """Dice overlap between two binary masks. 1.0 = identical, 0.0 = disjoint."""
    a, b = a.astype(bool), b.astype(bool)
    total = a.sum() + b.sum()
    return 1.0 if total == 0 else float(2.0 * (a & b).sum() / total)


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def select_victims(df, rate, rng):
    """Choose which TRAIN samples to poison. Val and test are never eligible."""
    n = int(round(len(df) * rate))
    return set(rng.choice(df["sample_id"].values, size=n, replace=False).tolist())


def poison_dataset(method, rate, seed, root=".", manifest=MANIFEST,
                   split_file=SPLIT_FILE, run_id=None):
    root = Path(root)
    rng = np.random.default_rng(seed)

    full = pd.read_csv(root / manifest)
    splits = json.loads((root / split_file).read_text())
    train_ids = set(splits["samples"]["train"])

    train = full[full["sample_id"].isin(train_ids)]
    victims = select_victims(train, rate, rng)

    # The attacker cannot reach evaluation data. Assert it rather than trust it.
    for guarded in ("val", "test"):
        assert not (victims & set(splits["samples"][guarded])), \
            f"Poison leaked into {guarded} - threat model violated"

    run_id = run_id or f"{method}_rate{rate:g}_seed{seed}"
    out_dir = root / POISON_ROOT / run_id
    mask_dir = out_dir / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    corrupt = CORRUPTIONS[method]
    rows, ledger = [], []

    for row in tqdm(full.itertuples(index=False), total=len(full),
                    desc=f"poisoning {method} @ {rate:.0%}", unit="mask"):
        record = row._asdict()

        if record["sample_id"] not in victims:
            rows.append(record)                       # untouched, original path
            continue

        original = np.array(Image.open(root / record["mask_path"]))
        if original.ndim == 3:
            original = original[..., 0]
        original = (original > 127).astype(np.uint8)

        # The brain region lives in the IMAGE, not the mask: a mask only says
        # "tumour / not tumour" and has no idea what is underneath it.
        image = np.array(Image.open(root / record["image_path"]), dtype=np.float32)
        brain = brain_mask(image)

        corrupted, params = corrupt(original, rng, brain)

        new_path = mask_dir / f"{record['sample_id']}.png"
        Image.fromarray(corrupted * 255).save(new_path)

        record["mask_path"] = str(new_path.relative_to(root))
        record["tumor_pixels"] = int(corrupted.sum())
        record["tumor_fraction"] = float(corrupted.sum() / corrupted.size)
        record["is_empty"] = bool(corrupted.sum() == 0)
        rows.append(record)

        ledger.append({
            "sample_id": record["sample_id"],
            "patient_id": record["patient_id"],
            "method": method,
            "params": json.dumps(params),
            "clean_pixels": int(original.sum()),
            "poisoned_pixels": int(corrupted.sum()),
            "dice_clean_vs_poisoned": dice(original, corrupted),
        })

    poisoned_manifest = pd.DataFrame(rows)
    ledger_df = pd.DataFrame(ledger)

    poisoned_manifest.to_csv(out_dir / "manifest.csv", index=False)
    ledger_df.to_csv(out_dir / "ledger.csv", index=False)
    (out_dir / "config.json").write_text(json.dumps({
        "run_id": run_id, "method": method, "rate": rate, "seed": seed,
        "source_manifest": manifest, "split_file": split_file,
        "n_train": len(train), "n_poisoned": len(victims),
    }, indent=2))

    return out_dir, poisoned_manifest, ledger_df


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--method", choices=METHODS, required=True)
    ap.add_argument("--rate", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()

    out_dir, _, ledger = poison_dataset(args.method, args.rate, args.seed,
                                        run_id=args.run_id)

    print(f"\nRun written to {out_dir}")
    print(f"Samples poisoned: {len(ledger)}")
    if len(ledger):
        d = ledger["dice_clean_vs_poisoned"]
        print(f"Dice(clean, poisoned) - mean {d.mean():.3f}  "
              f"median {d.median():.3f}  min {d.min():.3f}  max {d.max():.3f}")
        print(f"Tumour pixels - clean {ledger['clean_pixels'].sum()} "
              f"-> poisoned {ledger['poisoned_pixels'].sum()}")
        print("\nA high mean Dice here means the corruption is subtle: the "
              "poisoned mask still largely overlaps the truth, which is "
              "exactly the stealth the threat model asks for.")


if __name__ == "__main__":
    main()
