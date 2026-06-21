"""omniem-train data spine.

Emits raw single-channel float ``[B, 1, Y, X, Z]`` batches; normalization +
channel synthesis are the model's job (``apply_input``). Importing this
subpackage pulls in torch + MONAI, so it is a submodule import (not
re-exported from ``omniem_train``).
"""

from .loader import get_items, get_loaders
from .reader import read_image

__all__ = ["get_loaders", "get_items", "read_image"]
