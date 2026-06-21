"""``build_loss`` — task-gated loss factory.

The trainer calls :func:`build_loss` once per run; this function picks the right
family based on ``cfg.model.task_type`` and threads the per-task knobs through.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..config import LossRestoreCfg, LossSegCfg, RunConfig
from .base import CombinationLoss

if TYPE_CHECKING:
    from pathlib import Path


def build_loss(
    cfg: RunConfig,
    *,
    train_items: list[dict] | None = None,
    feature_weights_path: str | Path | None = None,
) -> CombinationLoss:
    """Return the task's loss as a :class:`CombinationLoss` (callable on logits).

    Args:
        cfg: validated run config.
        train_items: optional train manifest items (used for the seg label
            histogram when ``label_weights_alpha > 0``).
        feature_weights_path: resolved path for ``loss.feature.weights``
            (restoration; required when the feature term is active).
    """
    task_type = cfg.model["task_type"]
    if task_type == "image2label":
        from .segmentation import build_segmentation_loss

        seg = cfg.typed_loss()
        assert isinstance(seg, LossSegCfg)
        return build_segmentation_loss(
            seg_cfg=seg,
            out_channels=int(cfg.model["out_channels"]),
            img_z=int(cfg.model.get("img_z", 1)),
            train_items=train_items,
            axes_order=cfg.data.axes_order,
        )
    if task_type == "image2image":
        from .restoration import build_restoration_loss

        rest = cfg.typed_loss()
        assert isinstance(rest, LossRestoreCfg)
        # data_mean/data_std come from the opaque model block (model.mean / model.std),
        # in [0,1] image space — the same convention apply_input uses. The feature loss
        # normalizes the [0,1] restoration target with them directly (no rescaling).
        return build_restoration_loss(
            restore_cfg=rest,
            data_mean=float(cfg.model.get("mean", 0.0)),
            data_std=float(cfg.model.get("std", 1.0)),
            feature_weights_path=str(feature_weights_path) if feature_weights_path else None,
            encoder_arch=str(cfg.model.get("encoder", "emdinov1")),
        )
    raise ValueError(f"unknown task_type {task_type!r}")  # pragma: no cover
