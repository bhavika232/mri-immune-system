"""
Dataset analysis for the BraTS 2018/2019 2D slice dataset.

Produces a reproducible dataset report and a per-sample manifest.

Usage:
    python src/analyze_dataset.py
    python src/analyze_dataset.py --root data/raw/brats --out docs/dataset_analysis.md

Outputs:
    docs/dataset_analysis.md        human-readable report
    data/processed/manifest.csv     one row per image/mask pair
"""

import argparse
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


# --------------------------------------------------------------------------
# filename parsing
# --------------------------------------------------------------------------

def parse_sample_id(stem):
    """
    'Brats18_2013_0_1_19' -> ('Brats18_2013_0_1', 19)

    The slice index is the final underscore-separated token; everything
    before it identifies the patient. Works for both the Brats18_2013_*
    and Brats18_TCIA13_* naming families.
    """
    patient, _, slice_token = stem.rpartition("_")
    if not patient or not slice_token.isdigit():
        return stem, None
    return patient, int(slice_token)


def file_md5(path, chunk_size=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------
# checks
# --------------------------------------------------------------------------

def check_pairing(image_dir, mask_dir):
    """Verify every image has exactly one mask and vice versa."""
    images = {p.stem for p in image_dir.glob("*.png")}
    masks = {p.stem for p in mask_dir.glob("*.png")}
    return {
        "n_images": len(images),
        "n_masks": len(masks),
        "paired": sorted(images & masks),
        "images_without_mask": sorted(images - masks),
        "masks_without_image": sorted(masks - images),
    }


def scan_samples(paired_ids, image_dir, mask_dir):
    """
    Single pass over every pair, collecting per-sample stats.

    Returns (rows, failures, mask_value_set).
    """
    rows, failures = [], []
    mask_values = set()

    for stem in tqdm(paired_ids, desc="scanning", unit="pair"):
        img_path = image_dir / f"{stem}.png"
        mask_path = mask_dir / f"{stem}.png"
        try:
            with Image.open(img_path) as im:
                img_mode, img_size = im.mode, im.size
                im.load()
            with Image.open(mask_path) as mk:
                mask_mode, mask_size = mk.mode, mk.size
                mask_arr = np.array(mk)
        except Exception as exc:                      # noqa: BLE001
            failures.append((stem, repr(exc)))
            continue

        # Multi-channel masks: collapse so pixel counts mean one thing.
        flat = mask_arr[..., 0] if mask_arr.ndim == 3 else mask_arr
        mask_values.update(np.unique(flat).tolist())

        tumor_pixels = int((flat > 0).sum())
        patient, slice_idx = parse_sample_id(stem)

        rows.append({
            "sample_id": stem,
            "patient_id": patient,
            "slice_idx": slice_idx,
            "image_path": str(img_path),
            "mask_path": str(mask_path),
            "img_mode": img_mode,
            "img_w": img_size[0],
            "img_h": img_size[1],
            "mask_mode": mask_mode,
            "mask_w": mask_size[0],
            "mask_h": mask_size[1],
            "dims_match": img_size == mask_size,
            "tumor_pixels": tumor_pixels,
            "tumor_fraction": tumor_pixels / flat.size,
            "is_empty": tumor_pixels == 0,
            "img_md5": file_md5(img_path),
        })

    return rows, failures, mask_values


def find_duplicates(df):
    """Group sample_ids that share an identical image hash."""
    dupes = defaultdict(list)
    for h, group in df.groupby("img_md5"):
        if len(group) > 1:
            dupes[h] = group["sample_id"].tolist()
    return dupes


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

def build_report(pairing, df, failures, mask_values, dupes, root):
    L = []
    add = L.append

    add("# Dataset Analysis — BraTS 2018/2019 2D\n")
    add(f"Source: `{root}`\n")

    add("## 1. Pairing integrity\n")
    add(f"- Images: **{pairing['n_images']}**")
    add(f"- Masks: **{pairing['n_masks']}**")
    add(f"- Matched pairs: **{len(pairing['paired'])}**")
    add(f"- Images without a mask: **{len(pairing['images_without_mask'])}**")
    add(f"- Masks without an image: **{len(pairing['masks_without_image'])}**")
    for label, key in [("Orphan images", "images_without_mask"),
                       ("Orphan masks", "masks_without_image")]:
        if pairing[key]:
            add(f"  - {label} (first 10): {pairing[key][:10]}")
    add("")

    add("## 2. Patients and slices\n")
    per_patient = df.groupby("patient_id").size()
    add(f"- Unique patients: **{df['patient_id'].nunique()}**")
    add(f"- Slices per patient — min **{per_patient.min()}**, "
        f"median **{per_patient.median():.0f}**, "
        f"mean **{per_patient.mean():.1f}**, "
        f"max **{per_patient.max()}**")
    add(f"- Slice index range: {df['slice_idx'].min()} to {df['slice_idx'].max()}")
    unparsed = int(df["slice_idx"].isna().sum())
    add(f"- Filenames that failed slice parsing: **{unparsed}**")
    add("\nPatients with the fewest slices:\n")
    add("```")
    for pid, n in per_patient.nsmallest(5).items():
        add(f"  {pid}: {n}")
    add("```\n")

    add("## 3. Dimensions\n")
    dims = Counter(zip(df["img_w"], df["img_h"]))
    add("Image dimensions:\n")
    add("```")
    for (w, h), n in dims.most_common(10):
        add(f"  {w}x{h}: {n}")
    add("```")
    add(f"- All images 240x240: **{dims.get((240, 240), 0) == len(df)}**")
    add(f"- Image/mask dimension mismatches: **{int((~df['dims_match']).sum())}**\n")

    add("## 4. Colour modes\n")
    add(f"- Image modes: `{dict(Counter(df['img_mode']))}`")
    add(f"- Mask modes: `{dict(Counter(df['mask_mode']))}`\n")

    add("## 5. Mask pixel values\n")
    vals = sorted(mask_values)
    add(f"- Unique values across all masks: `{vals}`")
    if vals == [0, 255] or vals == [0, 1]:
        add("- **Binary masks** — classes already collapsed to background/tumor.")
    elif set(vals) <= {0, 1, 2, 4}:
        add("- **Original BraTS labels detected** "
            "(1 = necrotic core, 2 = edema, 4 = enhancing tumor). "
            "Collapsing to binary discards sub-region information.")
    else:
        add("- Non-standard value set — inspect before assuming binary.")
    add("")

    add("## 6. Tumor content\n")
    empty = int(df["is_empty"].sum())
    add(f"- Completely empty masks: **{empty}** ({100 * empty / len(df):.2f}%)")
    add(f"- Slices containing tumor: **{len(df) - empty}**")
    nz = df.loc[~df["is_empty"], "tumor_fraction"]
    if len(nz):
        add(f"- Tumor fraction (non-empty) — min **{nz.min():.5f}**, "
            f"median **{nz.median():.5f}**, "
            f"mean **{nz.mean():.5f}**, "
            f"max **{nz.max():.5f}**")
        add(f"- Slices with <0.1% tumor pixels: **{int((nz < 0.001).sum())}** "
            "(candidates for a minimum-size filter)")
    add("")

    add("## 7. Duplicates\n")
    add(f"- Duplicate image hashes: **{len(dupes)}**")
    if dupes:
        add(f"- Samples involved: **{sum(len(v) for v in dupes.values())}**")
        add("\nFirst few duplicate groups:\n")
        add("```")
        for group in list(dupes.values())[:5]:
            add(f"  {group}")
        add("```")
    add("")

    add("## 8. Load failures\n")
    add(f"- Files that failed to open: **{len(failures)}**")
    if failures:
        add("```")
        for stem, err in failures[:10]:
            add(f"  {stem}: {err}")
        add("```")
    add("")

    return "\n".join(L)


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="data/raw/brats")
    ap.add_argument("--out", default="docs/dataset_analysis.md")
    ap.add_argument("--manifest", default="data/processed/manifest.csv")
    args = ap.parse_args()

    root = Path(args.root)
    image_dir, mask_dir = root / "images", root / "masks"
    for d in (image_dir, mask_dir):
        if not d.is_dir():
            raise SystemExit(f"Missing directory: {d}")

    pairing = check_pairing(image_dir, mask_dir)
    rows, failures, mask_values = scan_samples(pairing["paired"], image_dir, mask_dir)

    df = pd.DataFrame(rows)
    dupes = find_duplicates(df)
    report = build_report(pairing, df, failures, mask_values, dupes, root)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report)

    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_cols = ["sample_id", "patient_id", "slice_idx", "image_path", "mask_path",
                     "img_w", "img_h", "tumor_pixels", "tumor_fraction", "is_empty"]
    df[manifest_cols].to_csv(manifest_path, index=False)

    print(report)
    print(f"\nReport written to   {out_path}")
    print(f"Manifest written to {manifest_path}  ({len(df)} rows)")


if __name__ == "__main__":
    main()
