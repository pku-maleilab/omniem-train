"""Restoration loss (``image2image``) — L1 + L2 + FeatureLoss.

All restoration terms apply ``sigmoid`` internally: ``predict`` returns pure
logits and the loss owns the activation, so a single forward serves both train
(sigmoid-in-loss) and infer (``apply_output`` sigmoids for the saver). Targets are
pre-normalised ``[0, 1]`` floats by the data pipeline.

FeatureLoss is the emdino-perceptual term: it formats inputs into
``[B*Z, 3ch, H, W]``, mean/std-normalises, and computes the ``image`` + ``patch``
cosine-embedding losses. It uses omniem's public ``EMEncoder`` (the canonical
backbone) so the restoration loss does not duplicate a backbone build.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import CosineEmbeddingLoss

from ..config import LossRestoreCfg
from .base import CombinationLoss


def _sigmoid(t: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(t)


class L1ImageLoss(nn.Module):
    """``L1(sigmoid(logits), target)`` — restoration L1 in image domain."""

    def __init__(self) -> None:
        super().__init__()
        self.l1 = nn.L1Loss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.l1(_sigmoid(logits), target)


class L2ImageLoss(nn.Module):
    """``MSE(sigmoid(logits), target)`` — restoration L2 in image domain."""

    def __init__(self) -> None:
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.mse(_sigmoid(logits), target)


class FeatureLoss(nn.Module):
    """emdino-perceptual feature loss.

    Computes the cosine-embedding loss between emdino-encoded ``sigmoid(logits)``
    and ``target`` (image + patch tokens). The encoder is built via omniem's
    public ``EMEncoder``; the weights file is **required**, no fallback.

    Input normalisation uses ``data_mean`` / ``data_std`` directly — the dataset stats in
    ``[0,1]`` image space (the same convention as ``model.mean`` / ``model.std``). Since
    restoration targets are ``[0,1]``, the affine maps them to roughly zero-mean/unit-std
    for the emdino encoder.
    """

    def __init__(
        self,
        *,
        weights: str,
        data_mean: float,
        data_std: float,
        image_weight: float = 1.0,
        patch_weight: float = 1.0,
        encoder_arch: str = "emdinov1",
    ) -> None:
        super().__init__()
        from .._omniem import require_omniem

        omniem = require_omniem()
        # Public API: load the emdino backbone (encoder only; head not needed).
        # We use the same encoder ID as omniem's omniemv1 (emdinov1).
        self.encoder = omniem.EMEncoder.load(encoder_arch, weights)
        # Freeze: perceptual loss is fixed during omniem-train.
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self.encoder.eval()
        self.data_mean = float(data_mean)
        self.data_std = float(data_std)
        self.image_weight = float(image_weight)
        self.patch_weight = float(patch_weight)
        self.image_loss = CosineEmbeddingLoss(reduction="mean")
        self.patch_loss = CosineEmbeddingLoss(reduction="mean")
        # Cached scalar "y=1" tensor for CosineEmbeddingLoss (one direction).
        self.register_buffer("_y", torch.tensor([1]), persistent=False)

    def _to_grayscale_byx(self, x: torch.Tensor) -> torch.Tensor:
        """``[B, 1, Y, X, Z]`` → ``[B*Z, Y, X]`` (grayscale + drop channel axis).

        Public ``EMEncoder.forward`` requires grayscale (``axes='byx'`` — no
        ``c`` axis) and applies its own gray→3ch + normalization in
        ``apply_input``, so the encoder owns that pre-processing pipeline.
        """
        if x.dim() == 5:
            b, c, y, x_, z = x.shape
            if c != 1:
                raise ValueError(f"FeatureLoss: expected single-channel input, got C={c}")
            # [B, 1, Y, X, Z] → [B, Z, Y, X] → [B*Z, Y, X]
            return x[:, 0].permute(0, 3, 1, 2).reshape(b * z, y, x_).float()
        if x.dim() == 4:
            if x.shape[1] != 1:
                raise ValueError(f"FeatureLoss: expected single-channel input, got C={x.shape[1]}")
            return x[:, 0].float()
        if x.dim() == 3:
            return x.float()
        raise ValueError(f"FeatureLoss expects 3D/4D/5D tensors, got {tuple(x.shape)}")

    def _encode(self, x: torch.Tensor) -> dict:
        """Run the public ``EMEncoder.forward`` with the dataset mean/std.

        ``norm={'mean','std'}`` overrides the encoder's default (pretraining)
        stats so the affine uses the restoration dataset stats (``data_mean`` /
        ``data_std``, in ``[0,1]`` image space).
        """
        return self.encoder.forward(
            self._to_grayscale_byx(x),
            axes="byx",
            return_cls=True,
            return_patch=True,
            norm={"mean": self.data_mean, "std": self.data_std},
        )

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        feat_p = self._encode(_sigmoid(logits))
        feat_r = self._encode(target)
        cls_p = feat_p["cls"]  # [N, D]
        cls_r = feat_r["cls"]
        d = cls_p.shape[-1]
        # patch is [N, P, D]; collapse to [N*P, D] for the per-patch cosine loss.
        patch_p = feat_p["patch"].reshape(-1, d)
        patch_r = feat_r["patch"].reshape(-1, d)
        y_img = self._y.to(cls_p.device).expand(cls_p.shape[0])
        y_patch = self._y.to(patch_p.device).expand(patch_p.shape[0])
        return self.image_weight * self.image_loss(
            cls_p, cls_r, y_img
        ) + self.patch_weight * self.patch_loss(patch_p, patch_r, y_patch)


def build_restoration_loss(
    *,
    restore_cfg: LossRestoreCfg,
    data_mean: float,
    data_std: float,
    feature_weights_path: str | None = None,
    encoder_arch: str = "emdinov1",
) -> CombinationLoss:
    """Assemble the restoration loss per ``restore_cfg``.

    Args:
        restore_cfg: validated typed config (at least one term required).
        data_mean: dataset mean in ``[0,1]`` image space (same convention as
            ``model.mean``; used directly to normalize the ``[0,1]`` restoration target).
        data_std: dataset std in ``[0,1]`` image space (same convention as ``model.std``).
        feature_weights_path: optional override of the path inside ``feature.weights``
            (after resolution).
    """
    loss = CombinationLoss()
    if restore_cfg.l1 is not None:
        loss.add_loss(L1ImageLoss(), weight=restore_cfg.l1.weight)
    if restore_cfg.l2 is not None:
        loss.add_loss(L2ImageLoss(), weight=restore_cfg.l2.weight)
    if restore_cfg.feature is not None:
        weights_path = feature_weights_path or restore_cfg.feature.weights
        feature = FeatureLoss(
            weights=weights_path,
            data_mean=float(data_mean),
            data_std=float(data_std),
            image_weight=restore_cfg.feature.image_weight,
            patch_weight=restore_cfg.feature.patch_weight,
            encoder_arch=encoder_arch,
        )
        loss.add_loss(feature, weight=restore_cfg.feature.weight)
    if len(loss) == 0:  # pragma: no cover — schema-time guard
        raise ValueError("restoration loss has no active terms")
    return loss
