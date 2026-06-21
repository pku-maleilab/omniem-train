"""Optimizer + LR schedule factory.

Schedule families:
  * ``warmup_cosine`` — :class:`LinearWarmupCosineAnnealingLR`.
  * ``cosine_anneal`` — :class:`torch.optim.lr_scheduler.CosineAnnealingLR`.
  * ``none`` — no scheduler (``None``).

All schedulers step **per epoch**.

Optimizers: ``adam`` / ``adamw`` / ``sgd``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, _LRScheduler

from .config import OptimCfg


class LinearWarmupCosineAnnealingLR(_LRScheduler):
    """Linear warmup → cosine annealing.

    The formula is written out explicitly so the train layer is self-contained
    and the per-step LR is reproducible for resume tests.

    For ``warmup_epochs == 0``: pure cosine annealing from epoch 0 (no warmup).
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        warmup_start_lr: float = 0.0,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        self.warmup_epochs = int(warmup_epochs)
        self.max_epochs = int(max_epochs)
        self.warmup_start_lr = float(warmup_start_lr)
        self.eta_min = float(eta_min)
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        # Closed-form LR for the current step (``last_epoch``).
        if self.warmup_epochs > 0 and self.last_epoch < self.warmup_epochs:
            # Linear ramp from warmup_start_lr → base_lr over warmup_epochs.
            return [
                self.warmup_start_lr
                + self.last_epoch
                * (base_lr - self.warmup_start_lr)
                / max(1, self.warmup_epochs - 1)
                for base_lr in self.base_lrs
            ]
        # Cosine annealing from base_lr → eta_min over (max_epochs - warmup_epochs).
        progress_denom = max(1, self.max_epochs - self.warmup_epochs)
        progress = (self.last_epoch - self.warmup_epochs) / progress_denom
        # Clamp progress so the last epoch's LR floors at eta_min cleanly.
        progress = min(1.0, max(0.0, progress))
        return [
            self.eta_min + 0.5 * (base_lr - self.eta_min) * (1.0 + math.cos(math.pi * progress))
            for base_lr in self.base_lrs
        ]


# ---- builders -------------------------------------------------------------


def build_optimizer(
    params: Iterable[torch.nn.Parameter], optim_cfg: OptimCfg
) -> torch.optim.Optimizer:
    """Build the optimizer named in ``optim_cfg`` from a parameter iterator."""
    name = optim_cfg.optimizer
    if name == "adam":
        return torch.optim.Adam(params, lr=optim_cfg.lr, weight_decay=optim_cfg.weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=optim_cfg.lr, weight_decay=optim_cfg.weight_decay)
    if name == "sgd":
        return torch.optim.SGD(
            params,
            lr=optim_cfg.lr,
            momentum=optim_cfg.momentum,
            weight_decay=optim_cfg.weight_decay,
        )
    raise ValueError(f"unknown optimizer {name!r}")  # pragma: no cover (schema-checked)


def build_scheduler(optimizer: torch.optim.Optimizer, optim_cfg: OptimCfg) -> Any:
    """Build the LR scheduler named in ``optim_cfg`` (or None for ``"none"``).

    All schedulers step **per epoch**.
    """
    name = optim_cfg.lr_schedule
    if name == "none":
        return None
    if name == "warmup_cosine":
        return LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=optim_cfg.warmup_epochs,
            max_epochs=optim_cfg.max_epochs,
        )
    if name == "cosine_anneal":
        return CosineAnnealingLR(optimizer, T_max=optim_cfg.max_epochs)
    raise ValueError(f"unknown lr_schedule {name!r}")  # pragma: no cover
