"""
Train the U-Net on clean or poisoned BraTS 2D slices.

Covers the Phase 1 deliverables: a clean baseline Dice, and a model trained
on each poison run so degradation can be measured against it.

Usage:
    # clean baseline
    python src/train.py --run-name baseline

    # poisoned run (produced by src/poison.py)
    python src/train.py --poison-run shift_rate0.1_seed42

    # quick end-to-end check on a handful of batches
    python src/train.py --run-name smoke --epochs 1 --limit 64

    # pick up an interrupted run
    python src/train.py --run-name baseline --resume

Threat-model guarantee: --poison-run only changes the TRAIN manifest. Val and
test are always read from the clean manifest, so evaluation never sees a
poisoned mask (docs/threat_model.md §3).

Writes to checkpoints/<run_name>/:
    config.json              every argument, plus parameter count and device
    best.pt                  weights with the highest val Dice
    last.pt                  latest weights + optimiser state, for --resume
    history.csv              one row per epoch: loss, Dice, IoU, lr, time
    train_sample_stats.csv   per-sample loss and Dice for every training
                             slice, every epoch (input for detection)
    test_per_sample.csv      per-slice Dice / IoU of best.pt on the test set
    test_summary.json        test Dice / IoU summary for best.pt
"""

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# Allow `python src/train.py` from the repo root as well as `python -m src.train`.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dataset import MANIFEST, SPLIT_FILE, MRIDataset  # noqa: E402
from src.losses import BCEDiceLoss, dice_iou_per_image  # noqa: E402
from src.model import UNet, count_parameters  # noqa: E402

POISON_ROOT = "data/processed/poison"
CHECKPOINT_ROOT = "checkpoints"


# --------------------------------------------------------------------------
# setup
# --------------------------------------------------------------------------

def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed):
    """
    Seeds weight init, shuffling and dropout.

    Note: MRIDataset._augment draws from an unseeded np.random.default_rng(),
    so augmentation itself is not reproducible run to run.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_train_manifest(args):
    if args.poison_run and args.train_manifest != MANIFEST:
        raise SystemExit("Pass either --poison-run or --train-manifest, not both")
    if args.poison_run:
        path = Path(POISON_ROOT) / args.poison_run / "manifest.csv"
        if not (Path(args.root) / path).exists():
            raise SystemExit(f"No poisoned manifest at {path}. Run src/poison.py first.")
        return str(path)
    return args.train_manifest


def build_loaders(args, train_manifest, generator):
    common = dict(normalize=args.normalize, min_tumor_pixels=args.min_tumor_pixels,
                  split_file=args.split_file, root=args.root)

    train_ds = MRIDataset("train", augment=not args.no_augment,
                          manifest=train_manifest, **common)
    # Evaluation data is always clean, whatever the training manifest says.
    val_ds = MRIDataset("val", manifest=args.manifest, **common)
    test_ds = MRIDataset("test", manifest=args.manifest, **common)

    if args.limit:
        train_ds = Subset(train_ds, range(min(args.limit, len(train_ds))))
        val_ds = Subset(val_ds, range(min(args.limit, len(val_ds))))
        test_ds = Subset(test_ds, range(min(args.limit, len(test_ds))))

    def loader(ds, train):
        return DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=train,
            drop_last=train and len(ds) > args.batch_size,
            num_workers=args.num_workers,
            generator=generator if train else None,
            persistent_workers=args.num_workers > 0,
        )

    return loader(train_ds, True), loader(val_ds, False), loader(test_ds, False)


# --------------------------------------------------------------------------
# per-sample loss (for the corruption ledger / detection work)
# --------------------------------------------------------------------------

@torch.no_grad()
def per_sample_loss(logits, target, bce_weight, dice_weight, eps=1.0):
    """Same formula as BCEDiceLoss, but one value per image instead of a mean."""
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none").flatten(1).mean(1)
    probs = torch.sigmoid(logits).flatten(1)
    t = target.flatten(1)
    dice = (2 * (probs * t).sum(1) + eps) / (probs.sum(1) + t.sum(1) + eps)
    return bce_weight * bce + dice_weight * (1.0 - dice)


# --------------------------------------------------------------------------
# train / evaluate
# --------------------------------------------------------------------------

def autocast(device, enabled):
    return torch.autocast(device_type=device.type, dtype=torch.float16,
                          enabled=enabled and device.type == "cuda")


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, args,
                    epoch, sample_writer):
    model.train()
    total, n = 0.0, 0
    dices = []

    for sample_ids, images, masks in tqdm(loader, desc=f"epoch {epoch:>3} train",
                                          leave=False, unit="batch"):
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad(set_to_none=True)
        with autocast(device, args.amp):
            logits = model(images)
            loss = criterion(logits.float(), masks)

        scaler.scale(loss).backward()
        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        logits = logits.detach().float()
        dice, _, _, _, _ = dice_iou_per_image(logits, masks)
        dices.append(dice.cpu())
        total += loss.item() * images.size(0)
        n += images.size(0)

        if sample_writer is not None:
            losses = per_sample_loss(logits, masks, criterion.bce_weight,
                                     criterion.dice_weight).cpu().tolist()
            for sid, l, d in zip(sample_ids, losses, dice.cpu().tolist()):
                sample_writer.writerow([epoch, sid, f"{l:.6f}", f"{d:.6f}"])

    return total / max(n, 1), torch.cat(dices).mean().item() if dices else float("nan")


@torch.no_grad()
def evaluate(model, loader, criterion, device, args, desc="val"):
    """Returns (summary dict, list of per-sample rows)."""
    model.eval()
    total, n = 0.0, 0
    rows = []
    g_inter = g_psum = g_tsum = 0.0

    for sample_ids, images, masks in tqdm(loader, desc=desc, leave=False, unit="batch"):
        images, masks = images.to(device), masks.to(device)
        with autocast(device, args.amp):
            logits = model(images)
        logits = logits.float()
        total += criterion(logits, masks).item() * images.size(0)
        n += images.size(0)

        dice, iou, inter, psum, tsum = dice_iou_per_image(logits, masks, args.threshold)
        g_inter += inter.sum().item()
        g_psum += psum.sum().item()
        g_tsum += tsum.sum().item()
        for row in zip(sample_ids, dice.tolist(), iou.tolist(),
                       psum.tolist(), tsum.tolist()):
            rows.append(row)

    dices = np.array([r[1] for r in rows]) if rows else np.array([np.nan])
    ious = np.array([r[2] for r in rows]) if rows else np.array([np.nan])
    denom = g_psum + g_tsum
    summary = {
        "loss": total / max(n, 1),
        "dice": float(dices.mean()),           # mean per-slice Dice (headline)
        "dice_median": float(np.median(dices)),
        "iou": float(ious.mean()),
        # Pooled over all pixels in the split; less sensitive to tiny lesions.
        "dice_global": float(2 * g_inter / denom) if denom else 1.0,
        "n": len(rows),
    }
    return summary, rows


# --------------------------------------------------------------------------
# checkpoints
# --------------------------------------------------------------------------

def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, best_dice,
                    epochs_no_improve, config):
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "best_dice": best_dice,
        "epochs_no_improve": epochs_no_improve,
        "config": config,
    }, path)


def load_model(path, device):
    """Rebuild a trained U-Net from a checkpoint written by this script."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = UNet(base_channels=ckpt["config"]["base_channels"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    data = ap.add_argument_group("data")
    data.add_argument("--root", default=".", help="repo root that manifest paths are relative to")
    data.add_argument("--manifest", default=MANIFEST,
                      help="clean manifest; always used for val and test")
    data.add_argument("--train-manifest", default=MANIFEST,
                      help="manifest for the train split (default: clean)")
    data.add_argument("--poison-run", default=None,
                      help="train on data/processed/poison/<id>/manifest.csv")
    data.add_argument("--split-file", default=SPLIT_FILE)
    data.add_argument("--normalize", choices=("zscore", "unit"), default="zscore")
    data.add_argument("--min-tumor-pixels", type=int, default=0)
    data.add_argument("--no-augment", action="store_true")
    data.add_argument("--limit", type=int, default=0,
                      help="cap samples per split (smoke tests)")
    data.add_argument("--num-workers", type=int, default=0)

    opt = ap.add_argument_group("optimisation")
    opt.add_argument("--epochs", type=int, default=40)
    opt.add_argument("--batch-size", type=int, default=16)
    opt.add_argument("--lr", type=float, default=1e-3)
    opt.add_argument("--weight-decay", type=float, default=1e-4)
    opt.add_argument("--bce-weight", type=float, default=0.5)
    opt.add_argument("--dice-weight", type=float, default=0.5)
    opt.add_argument("--grad-clip", type=float, default=1.0, help="0 disables")
    opt.add_argument("--patience", type=int, default=8,
                     help="stop after this many epochs without val Dice gain; 0 disables")
    opt.add_argument("--amp", action="store_true", help="mixed precision (CUDA only)")

    mdl = ap.add_argument_group("model")
    mdl.add_argument("--base-channels", type=int, default=16)
    mdl.add_argument("--threshold", type=float, default=0.5)

    run = ap.add_argument_group("run")
    run.add_argument("--run-name", default=None,
                     help="defaults to the poison run id, or 'clean'")
    run.add_argument("--out-dir", default=CHECKPOINT_ROOT)
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    run.add_argument("--resume", action="store_true", help="continue from last.pt")
    run.add_argument("--no-sample-stats", action="store_true",
                     help="skip writing train_sample_stats.csv")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    seed_everything(args.seed)
    device = pick_device(args.device)

    train_manifest = resolve_train_manifest(args)
    run_name = args.run_name or args.poison_run or "clean"
    out_dir = Path(args.out_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    generator = torch.Generator().manual_seed(args.seed)
    train_loader, val_loader, test_loader = build_loaders(args, train_manifest, generator)

    model = UNet(base_channels=args.base_channels).to(device)
    criterion = BCEDiceLoss(args.bce_weight, args.dice_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1), eta_min=args.lr * 0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    config = {**vars(args), "run_name": run_name, "train_manifest_resolved": train_manifest,
              "device": str(device), "n_params": count_parameters(model),
              "n_train": len(train_loader.dataset), "n_val": len(val_loader.dataset),
              "n_test": len(test_loader.dataset), "torch": torch.__version__}

    start_epoch, best_dice, epochs_no_improve = 1, -1.0, 0
    last_path, best_path = out_dir / "last.pt", out_dir / "best.pt"

    if args.resume:
        if not last_path.exists():
            raise SystemExit(f"--resume given but {last_path} does not exist")
        ckpt = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        best_dice = ckpt["best_dice"]
        epochs_no_improve = ckpt.get("epochs_no_improve", 0)
        print(f"Resumed from epoch {ckpt['epoch']} (best val Dice {best_dice:.4f})")
    else:
        (out_dir / "config.json").write_text(json.dumps(config, indent=2))

    print(f"Run        {run_name}  ->  {out_dir}")
    print(f"Device     {device}")
    print(f"Train on   {train_manifest}")
    print(f"Samples    train {config['n_train']}  val {config['n_val']}  test {config['n_test']}")
    print(f"Params     {config['n_params']:,}")

    history_path = out_dir / "history.csv"
    history_new = not (args.resume and history_path.exists())
    history_f = open(history_path, "a" if not history_new else "w", newline="")
    history = csv.writer(history_f)
    if history_new:
        history.writerow(["epoch", "lr", "train_loss", "train_dice", "val_loss",
                          "val_dice", "val_dice_global", "val_iou", "seconds"])

    sample_f = sample_writer = None
    if not args.no_sample_stats:
        sample_path = out_dir / "train_sample_stats.csv"
        sample_new = not (args.resume and sample_path.exists())
        sample_f = open(sample_path, "a" if not sample_new else "w", newline="")
        sample_writer = csv.writer(sample_f)
        if sample_new:
            sample_writer.writerow(["epoch", "sample_id", "loss", "dice"])

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            t0 = time.time()
            lr = optimizer.param_groups[0]["lr"]
            train_loss, train_dice = train_one_epoch(
                model, train_loader, criterion, optimizer, scaler, device, args,
                epoch, sample_writer)
            val, _ = evaluate(model, val_loader, criterion, device, args)
            scheduler.step()
            dt = time.time() - t0

            history.writerow([epoch, f"{lr:.6g}", f"{train_loss:.5f}", f"{train_dice:.5f}",
                              f"{val['loss']:.5f}", f"{val['dice']:.5f}",
                              f"{val['dice_global']:.5f}", f"{val['iou']:.5f}", f"{dt:.1f}"])
            history_f.flush()
            if sample_f:
                sample_f.flush()

            improved = val["dice"] > best_dice
            if improved:
                best_dice, epochs_no_improve = val["dice"], 0
            else:
                epochs_no_improve += 1

            print(f"epoch {epoch:>3}  lr {lr:.2e}  train loss {train_loss:.4f} "
                  f"dice {train_dice:.4f} | val loss {val['loss']:.4f} "
                  f"dice {val['dice']:.4f} iou {val['iou']:.4f}  "
                  f"{dt:.0f}s{'  *best*' if improved else ''}")

            if improved:
                save_checkpoint(best_path, model, optimizer, scheduler, scaler,
                                epoch, best_dice, epochs_no_improve, config)
            save_checkpoint(last_path, model, optimizer, scheduler, scaler,
                            epoch, best_dice, epochs_no_improve, config)

            if args.patience and epochs_no_improve >= args.patience:
                print(f"Early stop: no val Dice gain for {args.patience} epochs")
                break
    finally:
        history_f.close()
        if sample_f:
            sample_f.close()

    # ---- final test evaluation on the best checkpoint --------------------
    if not best_path.exists():
        print("No best checkpoint written; skipping test evaluation.")
        return

    model, ckpt = load_model(best_path, device)
    test, rows = evaluate(model, test_loader, criterion, device, args, desc="test")

    with open(out_dir / "test_per_sample.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "dice", "iou", "pred_pixels", "true_pixels"])
        for sid, d, i, p, t in rows:
            w.writerow([sid, f"{d:.6f}", f"{i:.6f}", int(p), int(t)])

    summary = {"run_name": run_name, "best_epoch": ckpt["epoch"],
               "val_dice": ckpt["best_dice"], **{f"test_{k}": v for k, v in test.items()}}
    (out_dir / "test_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\nBest epoch {ckpt['epoch']}  val Dice {ckpt['best_dice']:.4f}")
    print(f"Test       Dice {test['dice']:.4f} (median {test['dice_median']:.4f}, "
          f"global {test['dice_global']:.4f})  IoU {test['iou']:.4f}  n={test['n']}")


if __name__ == "__main__":
    main()
