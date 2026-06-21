"""Data spine — JSON manifest → MONAI ``DataLoader`` of raw ``image``/``target`` tuples.

* Each manifest item carries ``image`` + (optional) ``label``. ``require_label``
  selects the contract (train/validate need labels; infer/check tolerate items
  that omit it — gt-gated metrics still work).
* Path resolution — manifest item ``image``/``label`` paths resolve **relative
  to that manifest file's directory**; the ``cfg.data.*_jsons`` paths are
  pre-resolved by the CLI handler against the run.yaml's directory.
* Label auto-normalize — for ``image2label``, an exact binary ``{0, 255}``
  ndarray remaps to ``{0, 1}`` (when ``out_channels == 2``); any other case
  must already be class-indexed in ``[0, C-1]`` (out-of-range → error).
* Target dtype/interp per task — seg integer (nearest if resized), restore
  ``[0, 1]`` float (trilinear).
* Deterministic seeding — ``get_loaders`` takes a ``torch.Generator`` plus a
  ``worker_init_fn`` so a seeded run is reproducible (the trainer wires these
  in; the bare ``get_loaders`` default is None → MONAI's default seed).

The dataloader emits **raw single-channel float** images (no normalization,
no gray→3ch — those live inside ``model.apply_input``).
"""

from __future__ import annotations

import json
import logging
import random
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from monai.data import CacheDataset, DataLoader
from monai.transforms import Compose, Transform

from .reader import read_image

if TYPE_CHECKING:
    from ..config import LoadedConfig, RunConfig

# Single in-package logger for data-spine events.
_LOG = logging.getLogger("omniem_train.data")
# We log the auto-normalize message at most once per process to avoid flooding
# (a large manifest has many labels but the rule is invariant).
_LOGGED_BINARY_NORMALIZE = False


# ---- manifest reading + per-item path resolution ---------------------


def get_items(jsons: list[str | Path], *, resolve_paths: bool = True) -> list[dict[str, Any]]:
    """Concatenate the item lists of several JSON manifests.

    Each manifest must be a JSON **list** of item mappings; a non-list (e.g. a dict)
    raises a clear error instead of silently extending by its keys.

    When ``resolve_paths`` (default), each item's ``image`` / ``label`` path is
    rewritten **relative to that manifest's directory**. Absolute paths pass
    through. Older callers that pass ``resolve_paths=False`` get the raw strings
    (back-compat for the existing manifest-rejects-non-list test).
    """
    items: list[dict[str, Any]] = []
    for file in jsons:
        file_path = Path(file).resolve()
        with open(file_path) as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(
                f"manifest {file!r} must be a JSON list of items (got {type(data).__name__})"
            )
        if resolve_paths:
            manifest_dir = file_path.parent
            for item in data:
                if "image" in item:
                    item["image"] = str(_resolve_item_path(item["image"], manifest_dir))
                if "label" in item:
                    item["label"] = str(_resolve_item_path(item["label"], manifest_dir))
        items.extend(data)
    return items


def _resolve_item_path(p: str, manifest_dir: Path) -> Path:
    """Resolve a manifest-item path: absolute pass-through, else under the manifest dir."""
    pp = Path(p)
    return pp if pp.is_absolute() else (manifest_dir / pp).resolve()


# ---- label auto-normalize --------------------------------------------


def normalize_segmentation_label(
    arr: np.ndarray, *, out_channels: int, source_path: str | None = None
) -> np.ndarray:
    """Convert a raw label array to a class-indexed map in ``[0, out_channels-1]``.

    Auto-normalize rule:
      * Exact binary ``{0, 255}`` array and ``out_channels == 2`` → remap to
        ``{0, 1}`` (scale 0/255 → 0/1 for the binary two-class case).
      * Any other case: the array must already be class-indexed in
        ``[0, out_channels - 1]``. Anything out-of-range is a **hard error** —
        real class IDs are never heuristically remapped.

    The applied mapping is logged via the returned ``mapping`` (caller may log it);
    here we keep the function pure and just return the normalised array.
    """
    a = arr.astype(np.int64, copy=False)
    unique = np.unique(a)
    # The {0, 255} binary case. The condition is "every value
    # is in {0, 255}" — an all-foreground tile (unique=={255}) or all-background
    # tile (unique=={0}) still triggers the remap, since patch-based training
    # can legitimately produce either. The 255-only case is what
    # distinguishes the binary form from the class-indexed form; an all-zero
    # tile is also valid as already-class-indexed, but the remap is a no-op there.
    if (
        out_channels == 2
        and unique.size > 0
        and set(int(v) for v in unique).issubset({0, 255})
        and 255 in unique
    ):
        global _LOGGED_BINARY_NORMALIZE
        if not _LOGGED_BINARY_NORMALIZE:
            _LOG.info(
                "label auto-normalize: binary {0, 255} → {0, 1} "
                "(out_channels=2) — applied to %s",
                source_path or "<unknown>",
            )
            _LOGGED_BINARY_NORMALIZE = True
        return (a == 255).astype(np.int64)
    # Otherwise: must already be class-indexed in [0, C-1].
    if unique.size and (unique.min() < 0 or unique.max() >= out_channels):
        where = f" ({source_path})" if source_path else ""
        raise ValueError(
            f"label{where}: values must be class indices in [0, {out_channels - 1}]; "
            f"got min={int(unique.min())}, max={int(unique.max())}. Only the exact "
            "binary {0, 255} case is auto-normalized."
        )
    return a


def normalize_restoration_target(arr: np.ndarray) -> np.ndarray:
    """Convert a restoration target to a ``[0, 1]`` float array.

    Scale ``[0, 255]`` linearly to ``[0, 1]``: cast to float, divide by 255 for
    an integer source, then clip to ``[0, 1]``.
    """
    if np.issubdtype(arr.dtype, np.integer):
        return (arr.astype(np.float32) / 255.0).clip(0.0, 1.0)
    a = arr.astype(np.float32, copy=False)
    # Float source: clip to [0, 1] (clipping is always applied).
    return a.clip(0.0, 1.0)


# ---- the MONAI Transform that loads an item -------------------------------


class LoadImageAndTarget(Transform):
    """Load one manifest item → ``{"image": tensor, ["target": tensor]}``.

    The single channel is the on-disk grayscale; the gray→3ch synthesis the encoder
    needs happens later inside ``model.apply_input`` (the package owns it), not here.

    Args:
        axes_order: on-disk axis convention (e.g. ``"zxy"``).
        task_type: ``image2label`` or ``image2image`` — picks the target transform.
        out_channels: model out_channels.
        require_label: when True, an item without ``label`` raises (train/validate
            contract); when False, missing label → no ``target`` entry (infer/check).
    """

    def __init__(
        self,
        *,
        axes_order: str,
        task_type: str,
        out_channels: int,
        require_label: bool,
    ) -> None:
        self.axes_order = axes_order
        self.task_type = task_type
        self.out_channels = out_channels
        self.require_label = require_label

    def __call__(self, item: dict[str, Any]) -> dict[str, torch.Tensor]:
        image_path = item["image"]
        arr = read_image(image_path, axes_order=self.axes_order)  # [Y, X, Z] float32
        out: dict[str, torch.Tensor] = {"image": torch.from_numpy(arr).unsqueeze(0)}

        label_path = item.get("label")
        if label_path is None:
            if self.require_label:
                raise ValueError(
                    f"manifest item missing 'label' (require_label=True): image={image_path}"
                )
            return out

        # Load and reorder the label like the image.
        label_arr = read_image(label_path, axes_order=self.axes_order)  # [Y, X, Z]
        if self.task_type == "image2label":
            label_arr = normalize_segmentation_label(
                label_arr, out_channels=self.out_channels, source_path=str(label_path)
            )
            tensor = torch.from_numpy(label_arr).unsqueeze(0).long()  # [1, Y, X, Z]
        elif self.task_type == "image2image":
            label_arr = normalize_restoration_target(label_arr)
            tensor = torch.from_numpy(label_arr).unsqueeze(0).float()  # [1, Y, X, Z]
        else:  # pragma: no cover (parse-time guard)
            raise ValueError(f"unknown task_type {self.task_type!r}")
        out["target"] = tensor
        return out


# ---- worker_init_fn --------------------------------------------------


def _worker_init_fn_factory(base_seed: int) -> Callable[[int], None]:
    """Create a worker_init_fn that seeds python+numpy+torch per-worker.

    Each DataLoader worker derives ``base_seed + worker_id`` so each item is
    deterministic given the master seed.
    """

    def _init(worker_id: int) -> None:
        seed = base_seed + worker_id
        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.manual_seed(seed)

    return _init


# ---- loader factory -------------------------------------------------------


def _build_loader(
    items: list[dict[str, Any]],
    *,
    transform: Transform,
    batch_size: int,
    cache_num: int,
    workers: int,
    shuffle: bool,
    generator: torch.Generator | None,
    worker_init_fn: Callable[[int], None] | None,
) -> DataLoader:
    """Wrap items in a ``CacheDataset`` + ``DataLoader`` (default collate stacks
    the per-item tensors into ``[B, ...]`` for both ``image`` and ``target``)."""
    ds = CacheDataset(
        data=items,
        transform=transform,
        cache_num=min(cache_num, len(items)) if items else 0,
        cache_rate=1.0,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        # persistent_workers=False even when workers>0: persistent
        # workers keep their RNG state across epochs in the uninterrupted run,
        # but a RESUMED run respawns them from the seeded ``worker_init_fn``
        # — those two streams would diverge. Re-spawning workers each epoch
        # makes BOTH paths derive worker RNG from ``base_seed + worker_id``
        # deterministically, restoring the reproducibility guarantee.
        persistent_workers=False,
        generator=generator,
        worker_init_fn=worker_init_fn,
    )


def get_loaders(
    cfg: RunConfig | LoadedConfig,
    *,
    splits: tuple[str, ...] = ("train", "val"),
    require_label: bool = False,
    generator: torch.Generator | None = None,
    base_worker_seed: int | None = None,
    config_dir: Path | None = None,
    apply_aug: bool = False,
) -> dict[str, DataLoader]:
    """Build the requested split loaders.

    Accepts either the typed :class:`~omniem_train.config.RunConfig` (the original
    callers) **or** a :class:`~omniem_train.config.LoadedConfig` (trainer,
    so path resolution is automatic). When passed a raw ``RunConfig``,
    ``config_dir`` may be supplied for resolution; otherwise the run.yaml
    paths are taken verbatim (the committed sample's paths are repo-root-relative,
    which the test runs from).

    Args:
        cfg: parsed run config (or a LoadedConfig).
        splits: which of ``"train"``/``"val"``/``"infer"`` to build (default the two
            training splits). A split whose manifest list is empty is skipped.
        require_label: when True, every loaded item MUST carry a ``label`` (the
            train/validate contract); when False, items may omit it (infer/check).
        generator: optional seeded ``torch.Generator`` for the DataLoader.
        base_worker_seed: optional master seed for ``worker_init_fn``.
        config_dir: directory of the source run.yaml; if None and ``cfg`` is a
            LoadedConfig, taken from ``cfg.source_dir``.

    Returns:
        ``{split: DataLoader}`` for the splits that have manifests.
    """
    from ..config import LoadedConfig as _LoadedConfig  # local to avoid a cycle

    if isinstance(cfg, _LoadedConfig):
        bundle = cfg
        run_cfg = bundle.cfg
        cfg_dir = bundle.source_dir
    else:
        run_cfg = cfg
        cfg_dir = config_dir

    task_type = run_cfg.model["task_type"]
    out_channels = int(run_cfg.model.get("out_channels", 2))

    load_transform = LoadImageAndTarget(
        axes_order=run_cfg.data.axes_order,
        task_type=task_type,
        out_channels=out_channels,
        require_label=require_label,
    )

    # per-process aug (train configurable / val deterministic).
    aug_train = None
    aug_val = None
    if apply_aug:
        from ..aug import build_aug_transforms

        aug_train = build_aug_transforms(
            run_cfg.aug.train, task_type=task_type, has_target=require_label
        )
        aug_val = build_aug_transforms(
            run_cfg.aug.val, task_type=task_type, has_target=require_label
        )

    def _split_transform(split: str) -> Transform:
        if not apply_aug:
            return load_transform
        aug = aug_train if split == "train" else aug_val
        if aug is None:
            return load_transform
        return Compose([load_transform, aug])

    jsons_map = {
        "train": run_cfg.data.train_jsons,
        "val": run_cfg.data.val_jsons,
        "infer": run_cfg.data.infer_jsons,
    }

    worker_init_fn = (
        _worker_init_fn_factory(int(base_worker_seed)) if base_worker_seed is not None else None
    )

    loaders: dict[str, DataLoader] = {}
    for split in splits:
        files = jsons_map.get(split, [])
        if not files:
            continue
        # Apply resolution to manifest file paths (run.yaml-relative).
        resolved = [_resolve_run_path(f, cfg_dir) for f in files]
        items = get_items(resolved)
        if not items:
            continue
        loaders[split] = _build_loader(
            items,
            transform=_split_transform(split),
            batch_size=run_cfg.optim.batch_size,
            cache_num=run_cfg.data.cache_num,
            workers=run_cfg.data.workers,
            # train shuffles; val/infer are deterministic order.
            shuffle=(split == "train"),
            generator=generator,
            worker_init_fn=worker_init_fn,
        )
    return loaders


def _resolve_run_path(p: str | Path, cfg_dir: Path | None) -> str:
    """Resolve a run.yaml-relative path; pass through when ``cfg_dir`` is None."""
    pp = Path(p)
    if pp.is_absolute() or cfg_dir is None:
        return str(pp)
    return str((cfg_dir / pp).resolve())


def manifest_path(item: dict[str, Any]) -> Path:
    """Helper: the image path of a manifest item (used by tests / logging)."""
    return Path(item["image"])


# ---- seg label histogram ----------------


def label_histogram(items: list[dict[str, Any]], out_channels: int, axes_order: str) -> np.ndarray:
    """Count label-class pixels across a manifest (for ``(1/freq)^alpha`` weights).

    Works on the already-resolved item paths and applies the label
    auto-normalize so the {0, 255} binary case is counted as {0, 1}.
    """
    counts = np.zeros(out_channels, dtype=np.int64)
    for item in items:
        label_path = item.get("label")
        if label_path is None:
            continue
        arr = read_image(label_path, axes_order=axes_order)
        arr = normalize_segmentation_label(
            arr, out_channels=out_channels, source_path=str(label_path)
        )
        for c in range(out_channels):
            counts[c] += int((arr == c).sum())
    return counts
