"""
Scoring for poisoned-sample detection.

A detector sees the poisoned training set and the model trained on it, and must
guess which samples were corrupted. This module marks that guess against the
corruption ledger.

THE LEDGER LIVES ON THIS SIDE OF THE BOUNDARY ON PURPOSE. A detector that can
read `ledger.csv` is not a detector, it is a lookup. Detection code must expose

    detect(model, train_manifest, ...) -> predicted sample_ids   (or scores)

and nothing else. This module is the only place the answer key is opened, and
it runs after the detector has committed its answer.

Two detector shapes are supported:

  binary   a set of sample_ids the detector flags as poisoned
  scored   one anomaly score per training sample, higher = more suspicious.
           Scored detectors are preferred: they give a threshold-free ROC AUC
           and average precision, so the result does not depend on an
           arbitrary cut-off.

Usage:
    python src/detection_eval.py --run shift_rate0.1_seed42 --predictions preds.csv

    preds.csv with a `sample_id` column only      -> binary evaluation
    preds.csv with `sample_id,score` columns      -> scored evaluation

    from src.detection_eval import evaluate_binary, evaluate_scores
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

POISON_ROOT = "data/processed/poison"
SPLIT_FILE = "splits/split_v1.json"

# Severity bands over dice(clean mask, poisoned mask). Low Dice means the mask
# was mangled; high Dice means the corruption was slight. Reporting recall per
# band answers the question the paper actually cares about: does the detector
# only catch the obvious ones?
SEVERITY_BINS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
SEVERITY_LABELS = ["0.0-0.2 (severe)", "0.2-0.4", "0.4-0.6", "0.6-0.8",
                   "0.8-1.0 (subtle)"]


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def load_truth(run_id, root=".", split_file=SPLIT_FILE):
    """
    Returns (train_ids, poisoned_ids, ledger).

    `train_ids` is the detector's full search space - every sample it could
    have flagged. Without it, true negatives and therefore precision at the
    population level are undefined.
    """
    root = Path(root)
    run_dir = root / POISON_ROOT / run_id

    ledger = pd.read_csv(run_dir / "ledger.csv")
    splits = json.loads((root / split_file).read_text())
    train_ids = set(splits["samples"]["train"])
    poisoned_ids = set(ledger["sample_id"])

    stray = poisoned_ids - train_ids
    assert not stray, f"Ledger contains non-train samples: {sorted(stray)[:5]}"

    return train_ids, poisoned_ids, ledger


def load_predictions(path):
    """
    Read a detector's output. Returns (mode, payload).

    mode "scored" -> DataFrame with sample_id and score
    mode "binary" -> set of flagged sample_ids
    """
    df = pd.read_csv(path)
    if "sample_id" not in df.columns:
        raise SystemExit(f"{path} must have a 'sample_id' column")

    score_col = next((c for c in ("score", "anomaly_score", "suspicion")
                      if c in df.columns), None)
    if score_col:
        return "scored", df[["sample_id", score_col]].rename(
            columns={score_col: "score"})
    return "binary", set(df["sample_id"])


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------

def _confusion(flagged, poisoned, train_ids):
    tp = len(flagged & poisoned)
    fp = len(flagged - poisoned)
    fn = len(poisoned - flagged)
    tn = len(train_ids - poisoned - flagged)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "f1": f1}


def recall_by_severity(flagged, ledger):
    """
    Recall split by how badly each sample was corrupted.

    A detector that scores well overall but has near-zero recall in the
    "subtle" band is not solving the problem this paper poses - the subtle
    corruptions are the dangerous ones.
    """
    df = ledger.copy()
    df["caught"] = df["sample_id"].isin(flagged)
    df["band"] = pd.cut(df["dice_clean_vs_poisoned"], bins=SEVERITY_BINS,
                        labels=SEVERITY_LABELS, include_lowest=True)
    grouped = df.groupby("band", observed=False)["caught"]
    return pd.DataFrame({"n": grouped.size(), "caught": grouped.sum(),
                         "recall": grouped.mean()})


def evaluate_binary(predicted_ids, run_id, root="."):
    """Score a detector that returns a hard set of flagged sample_ids."""
    train_ids, poisoned_ids, ledger = load_truth(run_id, root)

    flagged = set(predicted_ids)
    outside = flagged - train_ids
    if outside:
        # Flagging val/test samples means the detector looked somewhere it
        # should not have. Report it rather than silently ignoring.
        raise SystemExit(f"Detector flagged {len(outside)} samples outside the "
                         f"train split, e.g. {sorted(outside)[:3]}")

    base_rate = len(poisoned_ids) / len(train_ids)
    result = {
        "run_id": run_id, "mode": "binary",
        "n_train": len(train_ids), "n_poisoned": len(poisoned_ids),
        "n_flagged": len(flagged), "base_rate": base_rate,
        **_confusion(flagged, poisoned_ids, train_ids),
    }
    # A detector is only useful if it beats flagging samples at random, which
    # would achieve precision equal to the poison rate.
    result["lift_over_random"] = (result["precision"] / base_rate
                                  if base_rate else float("nan"))
    return result, recall_by_severity(flagged, ledger)


def evaluate_scores(scores, run_id, root=".", top_k=None):
    """
    Score a detector that returns one anomaly score per training sample.

    Threshold-free metrics (ROC AUC, average precision) are the headline;
    precision and recall are also reported at an operating point of
    "flag the top N samples", where N defaults to the true number poisoned.
    """
    train_ids, poisoned_ids, ledger = load_truth(run_id, root)

    df = scores[scores["sample_id"].isin(train_ids)].copy()
    missing = len(train_ids) - len(df)
    if missing:
        # Unscored samples are treated as least suspicious; otherwise a
        # detector could inflate its AUC by scoring only easy cases.
        filler = pd.DataFrame({"sample_id": sorted(train_ids - set(df["sample_id"])),
                               "score": df["score"].min() - 1.0 if len(df) else 0.0})
        df = pd.concat([df, filler], ignore_index=True)

    df["truth"] = df["sample_id"].isin(poisoned_ids).astype(int)
    k = top_k or len(poisoned_ids)
    flagged = set(df.nlargest(k, "score")["sample_id"])

    base_rate = len(poisoned_ids) / len(train_ids)
    result = {
        "run_id": run_id, "mode": "scored",
        "n_train": len(train_ids), "n_poisoned": len(poisoned_ids),
        "n_scored": len(df) - missing, "n_unscored_filled": missing,
        "base_rate": base_rate,
        "roc_auc": float(roc_auc_score(df["truth"], df["score"])),
        "average_precision": float(average_precision_score(df["truth"], df["score"])),
        "operating_point_k": k,
        **_confusion(flagged, poisoned_ids, train_ids),
    }
    result["lift_over_random"] = (result["precision"] / base_rate
                                  if base_rate else float("nan"))
    return result, recall_by_severity(flagged, ledger)


# --------------------------------------------------------------------------

def report(result, severity):
    print(f"\nRun          {result['run_id']}   ({result['mode']} detector)")
    print(f"Train pool   {result['n_train']}   poisoned {result['n_poisoned']} "
          f"({result['base_rate']:.1%})")
    print("-" * 58)

    if result["mode"] == "scored":
        print(f"{'ROC AUC':<22}{result['roc_auc']:.3f}")
        print(f"{'Average precision':<22}{result['average_precision']:.3f}")
        print(f"{'Operating point':<22}top {result['operating_point_k']}")

    print(f"{'Precision':<22}{result['precision']:.3f}")
    print(f"{'Recall':<22}{result['recall']:.3f}")
    print(f"{'F1':<22}{result['f1']:.3f}")
    print(f"{'Lift over random':<22}{result['lift_over_random']:.2f}x")
    print(f"\n{'TP':<6}{result['tp']:<8}{'FP':<6}{result['fp']:<8}"
          f"{'FN':<6}{result['fn']:<8}{'TN':<6}{result['tn']}")

    print("\nRecall by corruption severity")
    print("-" * 58)
    print(severity.to_string(float_format=lambda v: f"{v:.3f}"))
    print("\nThe bottom band is the one that matters: subtle corruptions are "
          "the ones a human reviewer would also miss.")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, help="poison run id")
    ap.add_argument("--predictions", required=True,
                    help="CSV with sample_id, optionally a score column")
    ap.add_argument("--top-k", type=int, default=None,
                    help="scored detectors: flag this many (default: n poisoned)")
    ap.add_argument("--out", default=None, help="write results as JSON")
    ap.add_argument("--root", default=".")
    args = ap.parse_args()

    mode, payload = load_predictions(args.predictions)
    if mode == "scored":
        result, severity = evaluate_scores(payload, args.run, args.root, args.top_k)
    else:
        result, severity = evaluate_binary(payload, args.run, args.root)

    report(result, severity)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({**result,
                                   "recall_by_severity": severity.to_dict()}, indent=2))
        print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
