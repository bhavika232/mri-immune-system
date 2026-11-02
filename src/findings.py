"""
Findings extractor: binary tumour mask in, structured clinical features out.

This is the bridge between Phase 1 and Phase 2. The U-Net (or, for now, the
ground-truth annotation) produces a mask; this module turns it into the
structured findings that the knowledge graph, Graph-of-Thoughts rules and RAG
layer reason over.

Geometry (area, components, circularity, solidity) is NOT re-implemented here:
it comes from src.plausibility.mask_features so that Phase 1 and Phase 2 can
never disagree about what a mask looks like. This module adds the anatomical
features on top:

    midline_x              estimated brain midline (column), via mirror symmetry
    hemisphere             image_left | image_right | bilateral
    crosses_midline        does the mask span the midline
    midline_distance_px    closest mask pixel to the midline (0 if crossing)
    centroid_offset_px     signed centroid offset from midline (+ = image right)
    lateral_position       0 = at midline, 1 = at the brain's lateral edge
    ap_position            0 = top of brain in image, 1 = bottom
    region                 coarse 3x2 grid cell, e.g. "top_left"
    lobe_estimate          HEURISTIC lobe guess, with confidence "low"

------------------------------------------------------------------------------
Known limitations - read before trusting any anatomical label
------------------------------------------------------------------------------
1. ORIENTATION IS UNVERIFIED. The PNGs carry no header, so we do not know
   whether image-left is patient-left (neurological) or patient-right
   (radiological), nor for certain that anterior is at the top. Hemisphere is
   therefore reported in IMAGE coordinates. Flip IMAGE_LEFT_IS_PATIENT and
   ANTERIOR_AT_TOP once someone checks a slice against the original NIfTI.
2. LOBES NEED AN ATLAS. A 2D, unregistered slice has no reliable lobe map.
   lobe_estimate is a positional heuristic (front/back third, lateral depth,
   relative slice height) and exists so the knowledge graph has a slot to
   fill. Treat it as a placeholder until we have 3D registered volumes.
3. VOLUME IS AREA. Pixel spacing is assumed to be PIXEL_MM (BraTS is
   resampled to 1 mm isotropic, but the 2D repackage does not confirm it),
   and slice spacing between the PNGs is unknown. Per-patient "volume" is the
   sum of slice areas - a relative measure, not millilitres.

------------------------------------------------------------------------------
Usage
------------------------------------------------------------------------------
    python -m src.findings --limit 20                 # quick check
    python -m src.findings --split test               # ground truth, test set
    python -m src.findings                            # all 12,100 slices

Swapping in model predictions later: point --manifest at any CSV with columns
sample_id, patient_id, slice_idx, image_path, mask_path (where mask_path is the
PREDICTED mask) and set --source, e.g.

    python -m src.findings --manifest outputs/unet_v1/manifest.csv --source unet_v1

Poisoned runs work the same way:

    python -m src.findings --manifest data/processed/poison/<run_id>/manifest.csv \\
                           --source poison_<run_id>

Writes (under data/processed/findings/<source>/):
    findings.jsonl     one JSON object per slice (what the KG builder reads)
    findings.csv       the same, flattened, for pandas
    patients.csv       per-patient aggregates

In code:
    from src.findings import extract_findings
    f = extract_findings(mask, image)            # mask: HxW bool/0-1/0-255
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from src.plausibility import brain_mask, mask_features

SCHEMA_VERSION = "findings-v1"

# Assumptions to verify (see module docstring).
PIXEL_MM = 1.0
IMAGE_LEFT_IS_PATIENT = None     # None = unknown -> report image coordinates
ANTERIOR_AT_TOP = True           # unverified; affects lobe_estimate only

# A side must hold at least this share of the tumour to count toward
# "bilateral". Stops a couple of boundary pixels flipping the label.
BILATERAL_MIN_FRACTION = 0.10

# How far either side of the image centre to search for the symmetry axis.
MIDLINE_SEARCH_PX = 25

MASK_THRESHOLD = 127


# --------------------------------------------------------------------------
# midline
# --------------------------------------------------------------------------

def estimate_midline(image, search_px=MIDLINE_SEARCH_PX):
    """
    Column of the brain's left-right symmetry axis.

    Mirrors the brain mask about each candidate column and keeps the one with
    the best overlap (IoU). Uses the brain silhouette rather than intensities,
    so the tumour itself - which is asymmetric by nature - barely affects it.

    Returns (midline_x, symmetry_iou, brain). With no image, falls back to the
    geometric image centre with symmetry_iou = NaN.
    """
    if image is None:
        return None, float("nan"), None

    brain = brain_mask(image)
    h, w = brain.shape
    if brain.sum() == 0:
        return (w - 1) / 2.0, float("nan"), brain

    # Brain bounding-box centre is the starting guess.
    cols = np.where(brain.any(axis=0))[0]
    guess = (cols.min() + cols.max()) / 2.0

    best_x, best_iou = guess, -1.0
    # Half-pixel steps: the axis may fall between two columns.
    for c2 in range(int(2 * (guess - search_px)), int(2 * (guess + search_px)) + 1):
        c = c2 / 2.0
        # Reflect column j -> 2c - j.
        src = np.arange(w)
        dst = np.rint(2 * c - src).astype(int)
        ok = (dst >= 0) & (dst < w)
        mirrored = np.zeros_like(brain)
        mirrored[:, dst[ok]] = brain[:, src[ok]]
        inter = np.logical_and(brain, mirrored).sum()
        union = np.logical_or(brain, mirrored).sum()
        iou = inter / union if union else 0.0
        if iou > best_iou:
            best_x, best_iou = c, iou

    return float(best_x), float(best_iou), brain


# --------------------------------------------------------------------------
# anatomy
# --------------------------------------------------------------------------

def _side_label(side):
    """'left'/'right' in image space -> label honouring the orientation flag."""
    if IMAGE_LEFT_IS_PATIENT is None:
        return f"image_{side}"
    if IMAGE_LEFT_IS_PATIENT:
        return side
    return {"left": "right", "right": "left"}[side]


def _band(v, names):
    """Map v in [0,1] to one of len(names) equal bands."""
    i = min(int(v * len(names)), len(names) - 1)
    return names[max(i, 0)]


def estimate_lobe(ap_position, lateral_position, axial_level):
    """
    Positional lobe heuristic for an axial slice. LOW CONFIDENCE by design.

    ap_position:      0 = anterior-most brain row, 1 = posterior-most
    lateral_position: 0 = midline, 1 = lateral edge of the brain
    axial_level:      0 = lowest slice for this patient, 1 = highest (or None)

    Rules of thumb (axial anatomy):
      - anterior third                         -> frontal
      - posterior third, lower/middle levels   -> occipital
      - posterior third, upper levels          -> parietal
      - middle third, lateral, lower levels    -> temporal
      - middle third, upper levels             -> parietal
      - middle third, otherwise                -> frontal/parietal boundary
      - deep and central (lateral < 0.25)      -> "deep/central" (basal
        ganglia, thalamus, ventricles - not a lobe)
    """
    if ap_position is None or np.isnan(ap_position):
        return None
    if not ANTERIOR_AT_TOP:
        ap_position = 1.0 - ap_position

    level = axial_level if axial_level is not None else 0.5
    if lateral_position < 0.25 and 0.3 < ap_position < 0.7:
        return "deep_central"
    if ap_position < 1 / 3:
        return "frontal"
    if ap_position > 2 / 3:
        return "parietal" if level > 0.66 else "occipital"
    if level > 0.66:
        return "parietal"
    if lateral_position > 0.5 and level < 0.5:
        return "temporal"
    return "frontal_parietal"


# --------------------------------------------------------------------------
# main entry point
# --------------------------------------------------------------------------

def _to_bool_mask(mask):
    m = np.asarray(mask)
    if m.ndim == 3:
        m = m[..., 0]
    if m.dtype == bool:
        return m
    return m > (MASK_THRESHOLD if m.max() > 1 else 0.5)


def extract_findings(mask, image=None, axial_level=None):
    """
    Structured findings for one slice.

    mask:        HxW array, bool / {0,1} / {0,255}
    image:       HxW MRI slice (optional, but needed for every anatomical
                 feature - without it only geometry is returned)
    axial_level: this slice's relative height within the patient, in [0,1]
                 (optional; improves lobe_estimate)

    Missing values are None, never NaN, so the output is valid JSON.
    """
    binary = _to_bool_mask(mask)
    h, w = binary.shape

    geom = mask_features(binary, image)
    area = int(geom["area"])

    out = {
        "schema": SCHEMA_VERSION,
        "has_tumour": area > 0,
        "area_px": area,
        "area_mm2": area * PIXEL_MM ** 2,
        "n_components": int(geom["n_components"]),
        "largest_fraction": geom["largest_fraction"],
        "circularity": geom["circularity"],
        "solidity": geom["solidity"],
        "brain_containment": geom["brain_containment"],
        "lesion_contrast": geom["lesion_contrast"],
        "irregularity": None,
        "centroid_x": None, "centroid_y": None,
        "bbox": None,
        "midline_x": None, "midline_symmetry_iou": None,
        "hemisphere": None,
        "crosses_midline": None,
        "fraction_image_left": None, "fraction_image_right": None,
        "midline_distance_px": None, "midline_distance_mm": None,
        "centroid_offset_px": None,
        "lateral_position": None,
        "ap_position": None,
        "region": None,
        "axial_level": axial_level,
        "lobe_estimate": None,
        "lobe_confidence": "low",
        "orientation_verified": IMAGE_LEFT_IS_PATIENT is not None,
    }

    if area == 0:
        return _clean(out)

    # Single irregularity score: 0 = perfectly round and convex, 1 = maximally
    # irregular. Mean of the two shape deficits.
    out["irregularity"] = float(1 - (geom["circularity"] + geom["solidity"]) / 2)

    ys, xs = np.nonzero(binary)
    cx, cy = float(xs.mean()), float(ys.mean())
    out["centroid_x"], out["centroid_y"] = cx, cy
    out["bbox"] = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]

    midline_x, sym_iou, brain = estimate_midline(image)
    if midline_x is None:
        return _clean(out)        # no image -> geometry only

    out["midline_x"] = midline_x
    out["midline_symmetry_iou"] = sym_iou

    left = float((xs < midline_x).sum() / area)
    right = 1.0 - left
    out["fraction_image_left"], out["fraction_image_right"] = left, right

    crosses = bool(xs.min() < midline_x < xs.max())
    out["crosses_midline"] = crosses
    dist = 0.0 if crosses else float(np.abs(xs - midline_x).min())
    out["midline_distance_px"] = dist
    out["midline_distance_mm"] = dist * PIXEL_MM
    out["centroid_offset_px"] = cx - midline_x

    if min(left, right) >= BILATERAL_MIN_FRACTION:
        out["hemisphere"] = "bilateral"
    else:
        out["hemisphere"] = _side_label("left" if left > right else "right")

    # Position within the brain's own extent, so it is comparable across
    # slices where the brain is larger or smaller.
    if brain is not None and brain.any():
        bys, bxs = np.nonzero(brain)
        top, bottom = bys.min(), bys.max()
        half_width = max(midline_x - bxs.min(), bxs.max() - midline_x, 1.0)
        ap = float(np.clip((cy - top) / max(bottom - top, 1), 0, 1))
        lat = float(np.clip(abs(cx - midline_x) / half_width, 0, 1))
        out["ap_position"] = ap
        out["lateral_position"] = lat

        vert = _band(ap, ["top", "middle", "bottom"])
        horiz = "midline" if out["hemisphere"] == "bilateral" else (
            "left" if cx < midline_x else "right")
        out["region"] = f"{vert}_{horiz}"
        out["lobe_estimate"] = estimate_lobe(ap, lat, axial_level)

    return _clean(out)


def _clean(d):
    """NaN -> None and numpy scalars -> Python, so json.dumps never chokes."""
    res = {}
    for k, v in d.items():
        if isinstance(v, (np.floating, float)):
            v = None if np.isnan(v) else float(v)
        elif isinstance(v, np.integer):
            v = int(v)
        elif isinstance(v, np.bool_):
            v = bool(v)
        res[k] = v
    return res


# --------------------------------------------------------------------------
# batch run over a manifest
# --------------------------------------------------------------------------

def aggregate_patients(df):
    """Per-patient summary of the slice-level findings."""
    t = df[df["has_tumour"]]
    g = t.groupby("patient_id")
    agg = pd.DataFrame({
        "n_slices_with_tumour": g.size(),
        "total_area_px": g["area_px"].sum(),          # relative "volume"
        "max_area_px": g["area_px"].max(),
        "mean_irregularity": g["irregularity"].mean(),
        "max_components": g["n_components"].max(),
        "any_crosses_midline": g["crosses_midline"].any(),
        "min_midline_distance_px": g["midline_distance_px"].min(),
        "dominant_hemisphere": g["hemisphere"].agg(
            lambda s: s.value_counts().idxmax() if s.notna().any() else None),
        "dominant_lobe_estimate": g["lobe_estimate"].agg(
            lambda s: s.value_counts().idxmax() if s.notna().any() else None),
    })
    return agg.reset_index()


def run(manifest, split, split_file, limit, source, out_root, root):
    root = Path(root)
    df = pd.read_csv(root / manifest)

    if split:
        splits = json.loads((root / split_file).read_text())
        df = df[df["sample_id"].isin(set(splits["samples"][split]))]
    if limit:
        df = df.sample(min(limit, len(df)), random_state=0)

    # Relative slice height per patient, for the lobe heuristic. Computed on
    # the FULL manifest so --limit does not distort it.
    full = pd.read_csv(root / manifest)
    span = full.groupby("patient_id")["slice_idx"].agg(["min", "max"])

    rows = []
    for rec in tqdm(df.to_dict("records"), desc=f"findings [{source}]", unit="slice"):
        image = np.array(Image.open(root / rec["image_path"]), dtype=np.float32)
        mask = np.array(Image.open(root / rec["mask_path"]))

        lo, hi = span.loc[rec["patient_id"]]
        level = (rec["slice_idx"] - lo) / (hi - lo) if hi > lo else 0.5

        f = extract_findings(mask, image, axial_level=float(level))
        rows.append({"sample_id": rec["sample_id"], "patient_id": rec["patient_id"],
                     "slice_idx": int(rec["slice_idx"]), "source": source, **f})

    out_dir = root / out_root / source
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "findings.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    flat = pd.DataFrame(rows)
    flat.to_csv(out_dir / "findings.csv", index=False)
    patients = aggregate_patients(flat)
    patients.to_csv(out_dir / "patients.csv", index=False)

    print(f"\n{len(flat)} slices, {len(patients)} patients -> {out_dir}\n")
    t = flat[flat["has_tumour"]]
    print("hemisphere      ", t["hemisphere"].value_counts().to_dict())
    print("lobe_estimate   ", t["lobe_estimate"].value_counts().to_dict())
    print(f"crosses midline  {t['crosses_midline'].mean():.1%}")
    print(f"multi-component  {(t['n_components'] > 1).mean():.1%}")
    print(f"median area      {t['area_px'].median():.0f} px")
    print(f"median midline symmetry IoU  {t['midline_symmetry_iou'].median():.3f}"
          "   (low values = midline estimate unreliable on that slice)")
    return flat


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="data/processed/manifest.csv")
    ap.add_argument("--split", choices=["train", "val", "test"], default=None)
    ap.add_argument("--split-file", default="splits/split_v1.json")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--source", default="ground_truth",
                    help="label for where the masks came from (ground_truth, unet_v1, ...)")
    ap.add_argument("--out", default="data/processed/findings")
    ap.add_argument("--root", default=".")
    args = ap.parse_args()
    run(args.manifest, args.split, args.split_file, args.limit,
        args.source, args.out, args.root)


if __name__ == "__main__":
    main()
