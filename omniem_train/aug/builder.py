"""Aug builder — read ``aug.<process>`` config and return a MONAI ``Compose``.

Per-transform shapes:

```
flip:           { prob: 0.5, axes: [0, 1, 2] }
rotate:         { prob: 0.5, range_x: [-pi, pi] }
gaussian_noise: { prob: 0.2, mean: 0.0, std: 0.1 }
gaussian_smooth:{ prob: 0.0, sigma: [0.5, 1.5] }
contrast:       { prob: 0.0, gamma: [0.65, 1.5] }
intensity:      { prob: 0.0, factors: 0.1 }
```

The image key is always ``"image"``; the target key (``"target"``) participates
in spatial transforms but NOT in intensity-only transforms (flip/rotate/elastic
move pixels in both image and label; gaussian noise/smooth/intensity change pixel
values in the image only). The seg vs restore interp differs (seg uses nearest for label;
restore uses trilinear) — wired via the ``target_mode`` argument.

Returns a MONAI ``Compose`` over a dict input; the trainer wraps this around the
data spine's :class:`~omniem_train.data.loader.LoadImageAndTarget` (which loads
the raw tensors into the dict).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from monai import transforms as mtf
from monai.transforms import Compose, MapTransform

from ..config import AugProcessCfg

# Spatial keys (transforms that move pixels apply to both image and target).
_SPATIAL_KEYS = ("image", "target")
# Intensity keys (transforms that change pixel values apply to image only).
_INTENSITY_KEYS = ("image",)


def _maybe_spatial_keys(has_target: bool) -> tuple[str, ...]:
    return _SPATIAL_KEYS if has_target else ("image",)


def build_aug_transforms(
    aug_cfg: AugProcessCfg,
    *,
    task_type: str,
    has_target: bool,
) -> MapTransform | Compose | None:
    """Compose the augmentation transforms per ``aug_cfg``.

    Returns ``None`` when the process is disabled (no-op transform — the data
    spine emits the raw tensor as-is).
    """
    if not aug_cfg.enabled:
        return None
    if not aug_cfg.transforms:
        return None

    transforms: list[Any] = []
    spatial_keys = _maybe_spatial_keys(has_target)
    spatial_modes = _spatial_modes(task_type, has_target)
    cfg_t = aug_cfg.transforms

    # Random flip on spatial axes.
    if "flip" in cfg_t:
        f = cfg_t["flip"]
        transforms.append(
            mtf.RandFlipd(
                keys=spatial_keys,
                prob=float(f.get("prob", 0.5)),
                spatial_axis=tuple(int(a) for a in f.get("axes", [0, 1, 2])),
            )
        )

    # Random in-plane rotation (XY).
    if "rotate" in cfg_t:
        r = cfg_t["rotate"]
        rx = tuple(float(x) for x in r.get("range_x", (-math.pi, math.pi)))
        transforms.append(
            mtf.RandRotated(
                keys=spatial_keys,
                prob=float(r.get("prob", 0.5)),
                range_x=rx,
                mode=spatial_modes,
                padding_mode="zeros",
            )
        )

    # Gaussian noise (image only).
    if "gaussian_noise" in cfg_t:
        n = cfg_t["gaussian_noise"]
        transforms.append(
            mtf.RandGaussianNoised(
                keys=_INTENSITY_KEYS,
                prob=float(n.get("prob", 0.2)),
                mean=float(n.get("mean", 0.0)),
                std=float(n.get("std", 0.1)),
            )
        )

    # Gaussian smoothing (image only).
    if "gaussian_smooth" in cfg_t:
        s = cfg_t["gaussian_smooth"]
        sigma = s.get("sigma", (0.5, 1.5))
        sx = tuple(float(x) for x in sigma)
        transforms.append(
            mtf.RandGaussianSmoothd(
                keys=_INTENSITY_KEYS,
                prob=float(s.get("prob", 0.0)),
                sigma_x=sx,
                sigma_y=sx,
                sigma_z=sx,
            )
        )

    # Contrast (gamma) — image only.
    if "contrast" in cfg_t:
        c = cfg_t["contrast"]
        gamma = c.get("gamma", (0.65, 1.5))
        transforms.append(
            mtf.RandAdjustContrastd(
                keys=_INTENSITY_KEYS,
                prob=float(c.get("prob", 0.0)),
                gamma=tuple(float(x) for x in gamma),
            )
        )

    # Intensity scaling — image only.
    if "intensity" in cfg_t:
        i = cfg_t["intensity"]
        factors = i.get("factors", 0.1)
        if isinstance(factors, (list, tuple)):
            factors_arg = tuple(float(x) for x in factors)
        else:
            factors_arg = float(factors)
        transforms.append(
            mtf.RandScaleIntensityd(
                keys=_INTENSITY_KEYS,
                prob=float(i.get("prob", 0.0)),
                factors=factors_arg,
            )
        )

    if not transforms:
        return None
    return Compose(transforms)


def _spatial_modes(task_type: str, has_target: bool) -> list[str]:
    """Per-key interp modes for spatial transforms.

    Seg: image trilinear, label nearest. Restore: image trilinear, target trilinear.
    """
    if not has_target:
        return ["trilinear"]
    if task_type == "image2label":
        return ["trilinear", "nearest"]
    return ["trilinear", "trilinear"]


__all__ = ["build_aug_transforms"]
# Silence the unused-import linter — re-exporting np for downstream tests.
_ = np
