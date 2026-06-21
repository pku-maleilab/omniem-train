"""Metric accumulation — keeps a tiny shim around the MONAI metric classes so
the trainer and validate / infer code path share one interface.

MONAI metrics already accumulate state internally (``__call__(pred, target)``
appends a batch and ``.aggregate()`` returns the mean); this aggregator just
groups several metrics by name + their per-task postproc.
"""

from __future__ import annotations

from collections.abc import Callable

import torch


class MetricAggregator:
    """Wrap several MONAI metric objects + the per-task postproc functions.

    Usage:
        agg = MetricAggregator(metrics={"dice": DiceMetric(...), ...},
                                postproc_pred=…, postproc_target=…)
        for logits, target in loader:
            agg.update(logits, target)
        result = agg.aggregate()  # → {"dice": float, ...}
        agg.reset()
    """

    def __init__(
        self,
        *,
        metrics: dict[str, object],
        postproc_pred: Callable[[torch.Tensor], torch.Tensor],
        postproc_target: Callable[[torch.Tensor], torch.Tensor],
        list_expand: dict[str, tuple[str, ...]] | None = None,
    ) -> None:
        """Args:
        list_expand: optional ``{internal_name: (key1, key2, …)}``. When
            an internal metric (e.g. ``ConfusionMatrixMetric`` configured
            with several ``metric_name`` values) aggregates to a list, we
            emit one float per ``keyN`` instead of averaging them into one
            scalar — required for metric-set parity.
        """
        self.metrics = metrics
        self.postproc_pred = postproc_pred
        self.postproc_target = postproc_target
        self.list_expand = list_expand or {}

    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        """Apply the postproc, then push the batch into each metric."""
        p = self.postproc_pred(logits)
        t = self.postproc_target(target)
        for m in self.metrics.values():
            m(y_pred=p, y=t)

    def aggregate(self) -> dict[str, float]:
        """Compute each metric's mean (handling MONAI's ``get_not_nans``
        + ConfusionMatrixMetric list return shapes).

        For an entry in :attr:`list_expand`, a list result is emitted as one
        float per expanded key (not averaged together) — required for.
        """
        out: dict[str, float] = {}
        for name, m in self.metrics.items():
            val = m.aggregate()
            if isinstance(val, list):
                floats = [_tensor_to_float(v) for v in val]
                if name in self.list_expand:
                    keys = self.list_expand[name]
                    for k, v in zip(keys, floats, strict=False):
                        out[k] = float(v)
                else:
                    out[name] = float(sum(floats) / max(1, len(floats)))
            else:
                out[name] = float(_tensor_to_float(val))
        return out

    def reset(self) -> None:
        for m in self.metrics.values():
            if hasattr(m, "reset"):
                m.reset()


def _tensor_to_float(val: object) -> float:
    """Normalize a MONAI metric result (scalar / 1-elem / multi-elem / tuple) to a float."""
    if isinstance(val, tuple):
        val = val[0]
    if isinstance(val, torch.Tensor):
        if val.ndim == 0:
            return float(val.item())
        if val.numel() == 1:
            return float(val.flatten()[0].item())
        return float(val.mean().item())
    return float(val)
