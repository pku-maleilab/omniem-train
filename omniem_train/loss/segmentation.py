"""Segmentation loss (``image2label``) — DiceCE + optional BoundaryLoss.

Operates on the omniem-train layout (``[B, C, Y, X, Z]`` logits,
``[B, 1, Y, X, Z]`` integer target):

* **DiceCELoss** — ``to_onehot_y=True, softmax=True, squared_pred=True,
  smooth_nr, smooth_dr`` — always on; class weights when
  ``label_weights_alpha > 0`` (``(1 / freq)^alpha`` computed from the train manifest
  label histogram).
* **BoundaryLoss** — operates on the pre-softmax logits, applies softmax
  internally, with a 2D/3D distinction (``z_index`` set for 3D so the slice-wise
  loop runs).

The DiceCELoss path is shape-tolerant: MONAI's spatial-channel DiceCE accepts 5D
logits + ``[B, 1, ...]`` integer target without further reshape.
"""

from __future__ import annotations

import numpy as np
import torch
from monai.losses import DiceCELoss
from torch import nn
from torch.nn import functional as F

from ..config import LossSegCfg
from .base import CombinationLoss


class BoundaryLoss(nn.Module):
    """Boundary loss — operates on logits.

    A 2D variant plus an explicit 3D slice loop:
      * if ``z_index is None`` and logits are 5D with ``Z == 1``, treat as 2D
        (squeeze Z) — the 2D-model path.
      * if ``z_index is not None``, loop slice-by-slice along that axis (3D path).

    Internal softmax over the channel axis; the gt is converted to one-hot.
    """

    def __init__(self, *, z_index: int | None = None, theta0: int = 3, theta: int = 5) -> None:
        super().__init__()
        self.z_index = z_index
        self.theta0 = theta0
        self.theta = theta

    @staticmethod
    def _one_hot(label: torch.Tensor, n_classes: int) -> torch.Tensor:
        """``[B, H, W]`` long → ``[B, C, H, W]`` float one-hot."""
        oh = F.one_hot(label.long(), num_classes=n_classes)  # [B, H, W, C]
        return oh.permute(0, 3, 1, 2).contiguous().float()

    def _forward_2d(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """2D path — ``pred``=``[N, C, H, W]``, ``gt``=``[N, H, W]`` (or [N,1,H,W])."""
        if gt.dim() == 4:
            assert gt.shape[1] == 1, "boundary 2D path expects [N, 1, H, W] integer gt"
            gt = gt[:, 0, :, :]
        n, c, _h, _w = pred.shape

        pred = torch.softmax(pred, dim=1)
        one_hot_gt = self._one_hot(gt, c)

        gt_b = F.max_pool2d(
            1 - one_hot_gt, kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2
        )
        gt_b = gt_b - (1 - one_hot_gt)

        pred_b = F.max_pool2d(
            1 - pred, kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2
        )
        pred_b = pred_b - (1 - pred)

        gt_b_ext = F.max_pool2d(
            gt_b, kernel_size=self.theta, stride=1, padding=(self.theta - 1) // 2
        )
        pred_b_ext = F.max_pool2d(
            pred_b, kernel_size=self.theta, stride=1, padding=(self.theta - 1) // 2
        )

        gt_b = gt_b.view(n, c, -1)
        pred_b = pred_b.view(n, c, -1)
        gt_b_ext = gt_b_ext.view(n, c, -1)
        pred_b_ext = pred_b_ext.view(n, c, -1)

        P = torch.sum(pred_b * gt_b_ext, dim=2) / (torch.sum(pred_b, dim=2) + 1e-7)
        R = torch.sum(pred_b_ext * gt_b, dim=2) / (torch.sum(gt_b, dim=2) + 1e-7)
        BF1 = 2 * P * R / (P + R + 1e-7)
        return torch.mean(1 - BF1)

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        # 5D logits / target → squeeze Z if it's 1 (2D model path) or loop over z_index.
        if pred.dim() == 5:
            if self.z_index is None:
                # 2D-model 5D layout: [B, C, Y, X, Z=1] → squeeze.
                if pred.shape[-1] != 1:
                    raise ValueError(
                        "BoundaryLoss(z_index=None) requires Z==1 5D logits; "
                        f"got Z={pred.shape[-1]} — pass z_index for 3D"
                    )
                p2d = pred[..., 0]  # [B, C, Y, X]
                g2d = gt[..., 0] if gt.dim() == 5 else gt
                return self._forward_2d(p2d, g2d)
            # 3D path — loop slice-by-slice along z_index.
            preds = pred.unbind(self.z_index + 2)  # +2 because spatial dims start after B,C
            gts = gt.unbind(self.z_index + 2)
            return torch.stack(
                [self._forward_2d(p, g) for p, g in zip(preds, gts, strict=True)]
            ).mean()
        # Already 4D (direct 2D input).
        return self._forward_2d(pred, gt)


def _class_weights(
    items: list[dict] | None,
    *,
    out_channels: int,
    alpha: float,
    axes_order: str,
) -> torch.Tensor | None:
    """``(1 / freq)^alpha`` class weights from the train items' label histogram."""
    if alpha <= 0 or items is None:
        return None
    from ..data.loader import label_histogram

    counts = label_histogram(items, out_channels=out_channels, axes_order=axes_order)
    total = counts.sum()
    if total == 0:
        return None
    freq = counts.astype(np.float64) / float(total)
    # Avoid div-by-zero for absent classes (give them a flat weight of 0 — they
    # never contribute to the dice).
    freq = np.where(freq > 0, freq, 1.0)
    weights = (1.0 / freq) ** float(alpha)
    return torch.tensor(weights, dtype=torch.float32)


def build_segmentation_loss(
    *,
    seg_cfg: LossSegCfg,
    out_channels: int,
    img_z: int,
    train_items: list[dict] | None = None,
    axes_order: str = "zxy",
) -> CombinationLoss:
    """Assemble the seg loss per ``seg_cfg``.

    Args:
        seg_cfg: validated typed config.
        out_channels: model out_channels (≥ 2 for image2label).
        img_z: model img_z (1 → 2D; >1 → 3D — controls BoundaryLoss z_index).
        train_items: the train manifest items (for class-weight histogram).
        axes_order: on-disk axis convention (the data spine's reader).
    """
    loss = CombinationLoss()
    weight = _class_weights(
        train_items,
        out_channels=out_channels,
        alpha=seg_cfg.label_weights_alpha,
        axes_order=axes_order,
    )
    # DiceCE always on; logits-in, softmax internal.
    dice_ce = DiceCELoss(
        to_onehot_y=True,
        softmax=True,
        squared_pred=True,
        weight=weight,
        smooth_nr=seg_cfg.smooth_nr,
        smooth_dr=seg_cfg.smooth_dr,
    )
    loss.add_loss(dice_ce, weight=1.0)
    if seg_cfg.boundary_loss:
        # 2D vs 3D: img_z==1 → 2D (z_index=None); else 3D → z_index=2.
        boundary = BoundaryLoss(z_index=None if img_z == 1 else 2)
        loss.add_loss(boundary, weight=1.0)
    return loss
