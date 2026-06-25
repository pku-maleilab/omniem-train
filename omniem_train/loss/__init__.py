"""Loss factory + task-gated families.

Public surface (called from the trainer):
  * :func:`build_loss` — given a validated :class:`~omniem_train.config.RunConfig`,
    return the task's :class:`Loss` (a callable ``logits, target -> Tensor``).
  * :class:`Loss` — the contract every term/combination implements.

Internals:
  * :mod:`omniem_train.loss.base` — :class:`Loss` + :class:`CombinationLoss`.
  * :mod:`omniem_train.loss.segmentation` — DiceCE / BoundaryLoss.
  * :mod:`omniem_train.loss.restoration` — L1 / L2 / FeatureLoss.

contract: the model returns **pure logits** (``run(..., return_logits=True)``); loss
is task-gated and computed on logits. ``image2image`` applies ``sigmoid``
**internally** (the model stays pure-logits — the locked delta).
"""

from .base import CombinationLoss, Loss
from .factory import build_loss
from .restoration import FeatureLoss
from .segmentation import BoundaryLoss

__all__ = ["Loss", "CombinationLoss", "build_loss", "BoundaryLoss", "FeatureLoss"]
