"""Task-gated metric factory — exact MONAI params + correct domain.

The metric set is DERIVED per task; there are no user-facing knobs.
"""

from __future__ import annotations

import torch
from monai.metrics import (
    ConfusionMatrixMetric,
    DiceMetric,
    MeanIoU,
    PSNRMetric,
    SSIMMetric,
)
from monai.utils.enums import MetricReduction

from ..config import RunConfig
from .aggregator import MetricAggregator


def _seg_postproc_pred(out_channels: int):
    """``logits[B,C,...]`` → one-hot ``[B,C,...]`` via argmax."""

    def postproc(logits: torch.Tensor) -> torch.Tensor:
        if logits.dim() < 2:
            raise ValueError("seg metric expects [B, C, ...] logits")
        argmax = torch.argmax(logits, dim=1, keepdim=True)  # [B,1,...]
        # to one-hot along channel.
        oh = torch.zeros_like(logits)
        oh.scatter_(1, argmax, 1)
        return oh

    return postproc


def _seg_postproc_target(out_channels: int):
    """``target[B,1,...]`` integer → one-hot ``[B,C,...]``."""

    def postproc(t: torch.Tensor) -> torch.Tensor:
        if t.dim() < 2:
            raise ValueError("seg metric expects [B, 1, ...] target")
        long = t.long()
        oh = torch.zeros(
            (long.shape[0], out_channels, *long.shape[2:]),
            device=t.device,
            dtype=torch.float32,
        )
        oh.scatter_(1, long, 1)
        return oh

    return postproc


def _make_restore_postproc(img_z: int):
    """Restoration postproc with the right spatial squeeze for SSIM.

    SSIM ``spatial_dims=2`` requires ``[B, C, H, W]`` (no Z); for the 2D case
    (img_z == 1) we squeeze the trailing Z=1; the 3D case keeps 5D.
    """

    def _pred(logits: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logits).clamp(0.0, 1.0)
        if img_z == 1 and p.dim() == 5 and p.shape[-1] == 1:
            p = p[..., 0]
        return p

    def _target(t: torch.Tensor) -> torch.Tensor:
        t = t.clamp(0.0, 1.0)
        if img_z == 1 and t.dim() == 5 and t.shape[-1] == 1:
            t = t[..., 0]
        return t

    return _pred, _target


def build_metrics(cfg: RunConfig) -> MetricAggregator:
    """Build the task's metric aggregator.

    image2label → DiceMetric / MeanIoU / ConfusionMatrixMetric(precision, recall, f1).
    image2image → PSNRMetric / SSIMMetric.
    """
    task_type = cfg.model["task_type"]
    if task_type == "image2label":
        out_channels = int(cfg.model["out_channels"])
        # MONAI params: include_background=False, reduction=mean, get_not_nans.
        # The confusion-matrix metric returns a 3-element list (precision/recall/f1);
        # the aggregator splits it into three named scalars on aggregate() so the
        # metric set is reported as 5 distinct metrics, not a single confusion mean.
        confusion = ConfusionMatrixMetric(
            metric_name=("precision", "recall", "f1 score"),
            include_background=False,
            reduction=MetricReduction.MEAN,
            get_not_nans=True,
        )
        metrics = {
            "dice": DiceMetric(
                include_background=False,
                reduction=MetricReduction.MEAN,
                get_not_nans=True,
            ),
            "iou": MeanIoU(
                include_background=False,
                reduction=MetricReduction.MEAN,
                get_not_nans=True,
            ),
            "_confusion": confusion,  # internal — aggregator splits into 3 keys
        }
        return MetricAggregator(
            metrics=metrics,
            postproc_pred=_seg_postproc_pred(out_channels),
            postproc_target=_seg_postproc_target(out_channels),
            list_expand={"_confusion": ("precision", "recall", "f1")},
        )
    if task_type == "image2image":
        img_z = int(cfg.model.get("img_z", 1))
        spatial_dims = 2 if img_z == 1 else 3
        metrics = {
            "psnr": PSNRMetric(
                max_val=1.0,
                reduction=MetricReduction.MEAN,
                get_not_nans=True,
            ),
            "ssim": SSIMMetric(
                spatial_dims=spatial_dims,
                data_range=1.0,
                reduction=MetricReduction.MEAN,
                get_not_nans=True,
            ),
        }
        pred_fn, target_fn = _make_restore_postproc(img_z)
        return MetricAggregator(
            metrics=metrics,
            postproc_pred=pred_fn,
            postproc_target=target_fn,
        )
    raise ValueError(f"unknown task_type {task_type!r}")  # pragma: no cover
