"""
Build reproducible patient-level train/validation/test splits.

Splitting happens at the PATIENT level, never the slice level. A patient
contributes many near-identical adjacent slices, so splitting by slice would
put almost-copies of the same scan in both train and test and inflate every
metric we report.

Usage:
    python src/make_splits.py
    python src/make_splits.py --seed 42 --train 0.7 --val 0.15

Outputs:
    splits/split_v1.json    patient ids and sample ids per split (commit this)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def split_patients(patients, train_frac, val_frac, seed):
    """Shuffle patients with a fixed seed and cut into three disjoint groups."""
    rng = np.random.default_rng(seed)
    shuffled = np.array(sorted(patients))       # sort first so the input order
    rng.shuffle(shuffled)                       # can never affect the result

    n = len(shuffled)
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))

    return {
        "train": sorted(shuffled[:n_train].tolist()),
        "val": sorted(shuffled[n_train:n_train + n_val].tolist()),
        "test": sorted(shuffled[n_train + n_val:].tolist()),
    }


def summarise(df, patient_groups):
    """Per-split counts and tumour statistics, to check the split is fair."""
    out = {}
    for split, patients in patient_groups.items():
        sub = df[df["patient_id"].isin(patients)]
        out[split] = {
            "n_patients": len(patients),
            "n_slices": len(sub),
            "tumor_fraction_mean": float(sub["tumor_fraction"].mean()),
            "tumor_fraction_median": float(sub["tumor_fraction"].median()),
            "slices_per_patient_min": int(sub.groupby("patient_id").size().min()),
            "slices_per_patient_max": int(sub.groupby("patient_id").size().max()),
        }
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default="data/processed/manifest.csv")
    ap.add_argument("--out", default="splits/split_v1.json")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train", type=float, default=0.70)
    ap.add_argument("--val", type=float, default=0.15)
    args = ap.parse_args()

    df = pd.read_csv(args.manifest)
    patients = df["patient_id"].unique()
    groups = split_patients(patients, args.train, args.val, args.seed)

    # No patient may appear in more than one split. This is the whole point.
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        overlap = set(groups[a]) & set(groups[b])
        assert not overlap, f"Patient leakage between {a} and {b}: {sorted(overlap)[:5]}"

    assert sum(len(v) for v in groups.values()) == len(patients), "Lost patients"

    samples = {
        split: sorted(df.loc[df["patient_id"].isin(patients_), "sample_id"].tolist())
        for split, patients_ in groups.items()
    }
    stats = summarise(df, groups)

    payload = {
        "version": "v1",
        "seed": args.seed,
        "fractions": {"train": args.train, "val": args.val,
                      "test": round(1 - args.train - args.val, 4)},
        "split_level": "patient",
        "n_patients_total": len(patients),
        "n_samples_total": len(df),
        "stats": stats,
        "patients": groups,
        "samples": samples,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))

    print(f"Split written to {out_path}  (seed={args.seed})\n")
    print(f"{'split':<8}{'patients':>10}{'slices':>10}{'tumor frac (mean)':>20}")
    print("-" * 48)
    for split, s in stats.items():
        print(f"{split:<8}{s['n_patients']:>10}{s['n_slices']:>10}"
              f"{s['tumor_fraction_mean']:>20.5f}")
    print("\nNo patient appears in more than one split.")


if __name__ == "__main__":
    main()
