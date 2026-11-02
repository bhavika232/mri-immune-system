"""
Loss and metrics for binary segmentation.

Tumour is ~1% of pixels. Plain BCE is minimised by predicting all-background
(99% pixel accuracy, Dice 0). Dice loss measures overlap with the tumour
directly, so it cannot be gamed that way. BCE is kept alongside it for
stable per-pixel gradients early in training.

Report Dice and IoU. Never pixel accuracy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-6


def soft_dice_loss(logits, target, eps=1.0):
    """1 - soft Dice, computed per image then averaged. Smoothed by eps."""
    probs = torch.sigmoid(logits).flatten(1)
    target = target.flatten(1)
    inter = (probs * target).sum(1)
    denom = probs.sum(1) + target.sum(1)
    return 1.0 - ((2 * inter + eps) / (denom + eps)).mean()


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, target):
        bce = F.binary_cross_entropy_with_logits(logits, target)
        return self.bce_weight * bce + self.dice_weight * soft_dice_loss(logits, target)


@torch.no_grad()
def dice_iou_per_image(logits, target, threshold=0.5):
    """
    Hard Dice and IoU for each image in the batch, shape [B].

    An image where both prediction and truth are empty scores 1.0 (correctly
    found nothing). Every BraTS slice in our manifest has tumour, so this
    only matters for poisoned 'blank' masks.
    """
    pred = (torch.sigmoid(logits) > threshold).flatten(1).float()
    target = target.flatten(1)
    inter = (pred * target).sum(1)
    psum, tsum = pred.sum(1), target.sum(1)
    union = psum + tsum - inter
    empty = (psum + tsum) == 0
    dice = torch.where(empty, torch.ones_like(inter), 2 * inter / (psum + tsum + EPS))
    iou = torch.where(empty, torch.ones_like(inter), inter / (union + EPS))
    return dice, iou, inter, psum, tsum
