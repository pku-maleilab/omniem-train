"""Loss base + combination shape.

Every term in this package follows the :class:`Loss` contract: ``__call__(logits,
target) -> Tensor`` where ``logits`` is the model's **pure** logits (the
``run(..., return_logits=True)`` output — no activation). Restoration terms
(L1/L2/Feature) apply ``sigmoid`` internally, since the activation lives in the loss
rather than the model.

:class:`CombinationLoss` is a weighted sum: ``sum(w_i * term_i(logits, target))``.
Terms are added with :meth:`add_loss`; the trainer never reaches into the list (the
optimizer sees only the scalar tensor output).
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn

# Loss = anything callable as (logits, target) → scalar Tensor. Implementations
# include CombinationLoss + the term nn.Modules below; the trainer only needs the
# callable contract.
Loss = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class CombinationLoss(nn.Module):
    """Weighted sum of sub-losses."""

    def __init__(self) -> None:
        super().__init__()
        self.losses: nn.ModuleList = nn.ModuleList()
        self.weights: list[float] = []

    def add_loss(self, loss: nn.Module, weight: float = 1.0) -> None:
        """Append a term + its scalar weight."""
        self.losses.append(loss)
        self.weights.append(float(weight))

    def __len__(self) -> int:
        return len(self.losses)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """sum_i w_i * term_i(logits, target). ``.sum()`` on each term reduces
        any non-scalar reduction down to a scalar before weighting."""
        out = torch.zeros((), device=logits.device, dtype=logits.dtype)
        for term, w in zip(self.losses, self.weights, strict=True):
            out = out + term(logits, target).sum() * w
        return out
