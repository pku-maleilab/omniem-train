"""Image reader — tif/png → canonical ``[Y, X, Z]`` float32.

Load the file, squeeze, reorder the on-disk axes to the canonical
``(Y, X, Z)`` spatial order, and return float32. Normalization is **not**
done here — the mean/std affine and the gray → 3-channel synthesis are the
model's job (``apply_input``). Augmentation and spatial conform live in the
augmentation builder; this reader assumes on-disk tiles are already at the
configured spatial size (or that the aug pipeline will resize them).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from tifffile import imread as tifread

# Spatial axes the reader canonicalises to: rows (Y), cols (X), depth (Z).
_CANON = ("y", "x", "z")


def _read_array(path: str | Path) -> np.ndarray:
    """Read a tif/png/jpg file into a numpy array (no dtype coercion yet)."""
    path = Path(path)
    ext = path.suffix.lower().lstrip(".")
    if ext in ("tif", "tiff"):
        return np.asarray(tifread(path))
    if ext in ("png", "jpg", "jpeg"):
        # Lazy import: Pillow is only needed for PNG/JPEG. A TIFF-only run
        # (the common case) never touches it, so importing this module / running a
        # TIFF `check` does not require Pillow even though it is a declared dep.
        from PIL import Image

        with Image.open(path) as im:
            return np.asarray(im)
    raise ValueError(f"unsupported image extension: {path.suffix!r} ({path})")


def read_image(path: str | Path, axes_order: str = "zxy") -> np.ndarray:
    """Read ``path`` → a ``[Y, X, Z]`` float32 array.

    Args:
        path: tif/png/jpg file.
        axes_order: the on-disk spatial axis convention, a permutation of ``"xyz"``
            (e.g. ``"zxy"`` means the array dims are ordered depth, col, row). Used
            to transpose a 3D array into the canonical ``(Y, X, Z)`` order. A 2D
            array is read as a single ``(Y, X)`` plane and given ``Z = 1``.

    Returns:
        ``np.ndarray`` of shape ``[Y, X, Z]``, dtype float32 (raw intensities — the
        caller/model owns int→float scaling and the norm affine).
    """
    arr = np.asarray(_read_array(path))
    # Only drop spurious singleton dims when there are MORE than 3 (e.g. an extra
    # leading/trailing channel axis). A genuine 3D volume with a singleton Z must
    # keep its 3 dims so `axes_order` still applies — squeezing it down to 2D would
    # silently mislabel the spatial axes (a (Z=1, X, Y) volume → (X, Y), read as
    # (Y, X), bypassing the reorder).
    if arr.ndim > 3:
        arr = arr.squeeze()

    if arr.ndim == 2:
        # Single plane → (Y, X, 1).
        arr = arr[:, :, np.newaxis]
    elif arr.ndim == 3:
        axes = axes_order.lower()
        if sorted(axes) != ["x", "y", "z"]:
            raise ValueError(
                f"axes_order must be a permutation of 'xyz' for a 3D image (got {axes_order!r})"
            )
        # Permutation that sends the on-disk order → canonical (Y, X, Z).
        perm = tuple(axes.index(a) for a in _CANON)
        arr = np.transpose(arr, perm)
    else:
        raise ValueError(f"expected a 2D or 3D image, got shape {arr.shape}")

    return arr.astype(np.float32, copy=False)
