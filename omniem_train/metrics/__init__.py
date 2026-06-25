"""Metric factory + task-gated metric sets.

Public surface (called from validate/infer):
  * :func:`build_metrics` — task-gated factory; metric set is DERIVED, not a
    user field. Returns a :class:`MetricAggregator` that accumulates per item +
    reports a mean dict on :meth:`MetricAggregator.aggregate`.
  * :class:`MetricAggregator` — the contract.

Metric **domain** is the metric-domain postproc, NOT the task-output uint
(``run(..., dtype=…)``):
  * ``image2label`` → ``argmax`` over channel axis → one-hot vs target one-hot.
  * ``image2image`` → ``sigmoid(logits)`` in ``[0,1]`` vs the ``[0,1]`` target
    (pre-quantization, not the uint file).

exact MONAI params:
  * ``DiceMetric(include_background=False, reduction="mean", get_not_nans=True)``
  * ``MeanIoU(include_background=False)``
  * ``ConfusionMatrixMetric(["precision","recall","f1 score"], include_background=False)``
  * ``PSNRMetric(max_val=1.0)``
  * ``SSIMMetric(spatial_dims = 2 if img_z==1 else 3, data_range=1.0)``
"""

from .aggregator import MetricAggregator
from .factory import build_metrics

__all__ = ["MetricAggregator", "build_metrics"]
