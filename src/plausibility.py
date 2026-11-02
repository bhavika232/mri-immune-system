"""
Plausibility metric for tumour segmentation masks.

Dice answers "is this mask CORRECT?". This module answers a different and,
for our paper, more important question: "does this mask LOOK correct?"

A mask can score terribly on Dice while still being a smooth, single, properly
sized blob sitting on a visibly abnormal part of the brain - in other words,
something a clinician would accept without a second glance. That combination
is the silent failure the paper is about, and this module is how we measure it.

Six features, each a thing a radiologist would implicitly notice:

    n_components      real tumours are one mass, not scattered specks
    largest_fraction  how much of the mask is in its biggest piece
    circularity       smooth compact boundary vs jagged noise
    solidity          filled-in shape vs ring or lace
    brain_containment is the mask inside the brain, or floating in black space
    lesion_contrast   does the masked region actually look abnormal on the MRI

Scores are CALIBRATED, not invented. We measure each feature across the real
expert annotations, then score a new mask by how typical it is within that
empirical distribution. This matters: it means we never hand-pick a threshold,
and "plausible" is defined as "indistinguishable from a real annotation".

Usage:
    python src/plausibility.py --fit                      # build the reference
    python src/plausibility.py --run shift_rate0.1_seed42 # score a poison run

    from src.plausibility import mask_features, PlausibilityScorer
    scorer = PlausibilityScorer.load()
    result = scorer.score(mask, image)     # -> per-feature + overall in [0,1]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
from skimage import measure
from tqdm import tqdm

REFERENCE_PATH = "data/processed/plausibility_reference.json"

# Features where only a LOW value is suspicious (more is always fine).
# For these we score one-sided: being above the clean median is never penalised.
ONE_SIDED_LOW = ("largest_fraction", "circularity", "solidity",
                 "brain_containment", "lesion_contrast")

FEATURES = ("n_components", "largest_fraction", "circularity", "solidity",
            "brain_containment", "lesion_contrast")


# --------------------------------------------------------------------------
# feature extraction
# --------------------------------------------------------------------------

def brain_mask(image, threshold=0.05):
    """
    Rough brain-tissue mask: anything meaningfully brighter than the black
    background. BraTS slices are skull-stripped, so the background is a true
    zero and a low threshold separates tissue cleanly.
    """
    img = image.astype(np.float32)
    if img.max() > img.min():
        img = (img - img.min()) / (img.max() - img.min())
    return img > threshold


def mask_features(mask, image=None):
    """
    Geometric and radiological descriptors of a binary mask.

    `image` is optional. Without it, brain_containment and lesion_contrast
    cannot be computed and are returned as NaN - the purely geometric
    features still work.
    """
    binary = mask.astype(bool)
    area = int(binary.sum())

    if area == 0:
        # An empty mask is maximally implausible - a scan was segmented as
        # having no tumour at all. Report the worst value for everything.
        return {"area": 0, "n_components": 0, "largest_fraction": 0.0,
                "circularity": 0.0, "solidity": 0.0,
                "brain_containment": 0.0, "lesion_contrast": 0.0}

    labelled, n_components = ndimage.label(binary)
    sizes = ndimage.sum(binary, labelled, range(1, n_components + 1))
    largest_fraction = float(sizes.max() / area)

    # regionprops on the largest component: shape quality should describe the
    # tumour itself, not be diluted by stray specks elsewhere in the image.
    largest = labelled == (int(np.argmax(sizes)) + 1)
    props = measure.regionprops(largest.astype(int))[0]

    perimeter = props.perimeter if props.perimeter > 0 else 1.0
    # Isoperimetric quotient: 1.0 for a perfect circle, falls towards 0 as the
    # boundary becomes jagged or the shape elongates.
    circularity = float(min(1.0, 4 * np.pi * props.area / (perimeter ** 2)))
    solidity = float(props.solidity)       # area / convex hull area

    containment, contrast = np.nan, np.nan
    if image is not None:
        brain = brain_mask(image)
        containment = float((binary & brain).sum() / area)

        # Does the masked region look different from the tissue just outside it?
        # A plausible mask sits on something visibly abnormal; a mask shifted
        # onto normal tissue does not.
        img = image.astype(np.float32)
        if img.max() > img.min():
            img = (img - img.min()) / (img.max() - img.min())
        ring = ndimage.binary_dilation(binary, iterations=6) & ~binary & brain
        if ring.any():
            inside, outside = img[binary].mean(), img[ring].mean()
            # Normalised so it is comparable across slices with different
            # overall brightness. Clipped: we care about "stands out at all",
            # not how spectacularly.
            contrast = float(np.clip(abs(inside - outside) / (img.std() + 1e-6), 0, 3))

    return {"area": area, "n_components": int(n_components),
            "largest_fraction": largest_fraction, "circularity": circularity,
            "solidity": solidity, "brain_containment": containment,
            "lesion_contrast": contrast}


# --------------------------------------------------------------------------
# calibration
# --------------------------------------------------------------------------

class PlausibilityScorer:
    """
    Turns raw features into a 0-1 plausibility score by comparing them to the
    distribution of real expert annotations.

    A feature value sitting inside the clean 5th-95th percentile band scores
    1.0. Outside it, the score falls off linearly with distance from the band,
    measured in units of the clean interquartile range so each feature is on a
    comparable scale.
    """

    def __init__(self, reference):
        self.reference = reference

    # -- construction ----------------------------------------------------

    @classmethod
    def fit(cls, feature_rows):
        df = pd.DataFrame(feature_rows)
        reference = {}
        for name in FEATURES:
            col = df[name].dropna()
            if col.empty:
                continue
            q1, q3 = float(col.quantile(0.25)), float(col.quantile(0.75))
            reference[name] = {
                "lo": float(col.quantile(0.05)),
                "hi": float(col.quantile(0.95)),
                "median": float(col.median()),
                # Guard against a zero-width IQR making the score binary.
                "scale": max(q3 - q1, 1e-6),
                "one_sided_low": name in ONE_SIDED_LOW,
            }
        return cls(reference)

    @classmethod
    def load(cls, path=REFERENCE_PATH, root="."):
        return cls(json.loads((Path(root) / path).read_text())["features"])

    def save(self, path=REFERENCE_PATH, root=".", meta=None):
        out = Path(root) / path
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"features": self.reference,
                                   "meta": meta or {}}, indent=2))
        return out

    # -- scoring ---------------------------------------------------------

    def _score_feature(self, name, value):
        ref = self.reference.get(name)
        if ref is None or value is None or (isinstance(value, float) and np.isnan(value)):
            return np.nan
        if ref["one_sided_low"]:
            # Being unusually high is not suspicious for these.
            deficit = max(0.0, ref["lo"] - value)
        else:
            deficit = max(0.0, ref["lo"] - value, value - ref["hi"])
        return float(np.clip(1.0 - deficit / ref["scale"], 0.0, 1.0))

    def score(self, mask, image=None):
        """Per-feature scores plus the overall plausibility (their mean)."""
        feats = mask_features(mask, image)
        per_feature = {n: self._score_feature(n, feats.get(n)) for n in FEATURES}
        valid = [v for v in per_feature.values() if not np.isnan(v)]
        overall = float(np.mean(valid)) if valid else np.nan
        return {"features": feats, "scores": per_feature, "plausibility": overall}


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------

def _iter_manifest(df, root, limit=None, desc="scanning"):
    rows = df if limit is None else df.sample(min(limit, len(df)), random_state=0)
    for row in tqdm(rows.itertuples(index=False), total=len(rows), desc=desc, unit="mask"):
        rec = row._asdict()
        image = np.array(Image.open(Path(root) / rec["image_path"]), dtype=np.float32)
        mask = np.array(Image.open(Path(root) / rec["mask_path"]))
        if mask.ndim == 3:
            mask = mask[..., 0]
        yield rec, image, (mask > 127)


def cmd_fit(root, manifest, limit):
    df = pd.read_csv(Path(root) / manifest)
    rows = []
    for rec, image, mask in _iter_manifest(df, root, limit, "calibrating on clean masks"):
        rows.append({"sample_id": rec["sample_id"], **mask_features(mask, image)})

    scorer = PlausibilityScorer.fit(rows)
    path = scorer.save(root=root, meta={"n_masks": len(rows), "source": manifest})

    print(f"\nReference built from {len(rows)} expert annotations -> {path}\n")
    print(f"{'feature':<20}{'5th pct':>10}{'median':>10}{'95th pct':>10}")
    print("-" * 50)
    for name, ref in scorer.reference.items():
        print(f"{name:<20}{ref['lo']:>10.3f}{ref['median']:>10.3f}{ref['hi']:>10.3f}")
    return scorer


def cmd_run(root, run_id, limit):
    scorer = PlausibilityScorer.load(root=root)
    run_dir = Path(root) / "data/processed/poison" / run_id
    ledger = pd.read_csv(run_dir / "ledger.csv")
    poisoned_ids = set(ledger["sample_id"])

    man = pd.read_csv(run_dir / "manifest.csv")
    man = man[man["sample_id"].isin(poisoned_ids)]

    clean_man = pd.read_csv(Path(root) / "data/processed/manifest.csv")
    clean_man = clean_man[clean_man["sample_id"].isin(poisoned_ids)]

    out = []
    for label, m in (("clean", clean_man), ("poisoned", man)):
        for rec, image, mask in _iter_manifest(m, root, limit, f"scoring {label}"):
            r = scorer.score(mask, image)
            out.append({"sample_id": rec["sample_id"], "variant": label,
                        "plausibility": r["plausibility"],
                        **{f"s_{k}": v for k, v in r["scores"].items()}})

    df = pd.DataFrame(out)
    dest = run_dir / "plausibility.csv"
    df.to_csv(dest, index=False)

    piv = df.groupby("variant")[[c for c in df.columns if c.startswith("s_")]
                                + ["plausibility"]].mean()
    print(f"\nRun: {run_id}   (n={len(poisoned_ids)} poisoned samples)\n")
    print(piv.T.to_string(float_format=lambda v: f"{v:.3f}"))

    med = df.groupby("variant")["plausibility"].median()
    print(f"\nmedian plausibility   clean {med.get('clean', float('nan')):.3f}"
          f"   poisoned {med.get('poisoned', float('nan')):.3f}")
    print(f"\nSaved to {dest}")
    print("\nA small gap here is the result we want: it means the corrupted "
          "masks are hard to tell apart from real annotations on appearance "
          "alone, even though Dice says they are wrong.")
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fit", action="store_true",
                    help="calibrate the reference distribution on clean masks")
    ap.add_argument("--run", default=None, help="score a poison run by id")
    ap.add_argument("--manifest", default="data/processed/manifest.csv")
    ap.add_argument("--limit", type=int, default=None,
                    help="sample only N masks (for a quick check)")
    ap.add_argument("--root", default=".")
    args = ap.parse_args()

    if args.fit:
        cmd_fit(args.root, args.manifest, args.limit)
    if args.run:
        cmd_run(args.root, args.run, args.limit)
    if not args.fit and not args.run:
        ap.error("pass --fit, --run <run_id>, or both")


if __name__ == "__main__":
    main()
