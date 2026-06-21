"""Augmentation builder — per-process structured transforms.

``aug.train`` is configurable; ``aug.val`` is deterministic by default; infer
follows val. Each transform is exposed as a structured config (prob +
magnitude per transform), so flip/rotate/noise magnitudes are run.yaml
fields rather than hard-coded constants.

Determinism: MONAI's ``set_determinism`` is called by the trainer; this
module does not touch global RNG state.
"""

from .builder import build_aug_transforms

__all__ = ["build_aug_transforms"]
