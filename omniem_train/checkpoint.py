"""Checkpoint save/load — the two artifacts + atomic paired write.

Two artifacts per save:
  * **Inference weights** under ``weights/``. Filename per ``save_format``:
    * ``split`` (default) → ``head_e<NNN>.pt`` (+ best ``head_best.pt``) paired with a
      backbone. When the encoder is fine-tuned (``train.train_encoder: true``) the
      backbone is tagged per save — ``backbone_e<NNN>.pt`` / ``backbone_best.pt``. When
      the encoder is frozen (the default) the backbone is invariant, so a single shared
      ``backbone.pt`` is written and overwritten each save.
    * ``merged`` → ``merged_e<NNN>.pt`` (+ ``merged_best.pt``).
    Always also written: ``model.yaml`` (the clean omniem config) at the run root.
  * **Trainer state** under ``state/``: ``trainer_e<NNN>.pt`` (numeric epoch) +
    ``trainer_latest.pt`` (the newest pointer; the resume reads this).
    No ``trainer_best`` — resume is from *latest*, not best.

**Atomicity:** every save writes to a temp file in the same directory and
``os.replace``-s into place — so a crash mid-write never leaves a torn file. The
``state/`` artifact is written **last**, so a half-write never advertises a
missing weight set.

**Resume:** scan ``state/`` for ``trainer_latest.pt``, restore in order
model → optimizer → scheduler → scaler → RNG → dataloader seed; continue at
``epoch_completed + 1``.
"""

from __future__ import annotations

import os
import random
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    from .config import RunConfig


# ---- atomic write helper --------------------------------------------


def _atomic_torch_save(obj: Any, path: Path) -> None:
    """Atomic save: tmp file in the same dir, then ``os.replace``.

    Same-dir is required for ``os.replace`` to be atomic on POSIX. We do NOT
    re-open + fsync the file: torch.save's zip writer already closes the file
    properly, and re-opening it for fsync was racing with subsequent torch.save
    calls (the "unexpected pos vs vs" error mode).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


# ---- inference-weight save -----------------------------------------------


def save_inference_weights(
    model: Any,
    *,
    output_dir: Path,
    save_format: str,
    tag: str,
    copy_from: dict[str, Path] | None = None,
    frozen_backbone: bool = False,
) -> dict[str, Path]:
    """Write the inference artifact (split or merged) and return the paths.

    Args:
        model: an :class:`omniem.OmniEM` (has ``.save_weights``).
        output_dir: the run output dir; this writes under ``output_dir/weights/``.
        save_format: ``split`` or ``merged``.
        tag: a filename stem qualifier — e.g. ``e042``, ``best``, ``latest``.
        copy_from: when not None, **copy** the already-serialized files from this
            dict (``{"backbone": Path, "head": Path}`` or ``{"merged": Path}``)
            into the new ``tag``-d filenames instead of running ``torch.save``
            again. This is used by the trainer when the periodic and best
            checkpoints are saved in the SAME epoch — the model state is
            identical, so we avoid the (slow + race-prone) double serialization.
        frozen_backbone: split mode only. When True (the encoder is frozen, so the
            backbone is invariant across epochs), the backbone is written to a single
            shared ``weights/backbone.pt`` — atomically OVERWRITTEN on every save rather
            than tagged per epoch/best, so the run keeps exactly one backbone file that
            is always current (a stale ``backbone.pt`` from a prior run into the same
            output dir cannot survive). The head is still tagged ``head_<tag>.pt``.
            omniem's ``save_weights`` has no head-only mode, so the backbone is still
            serialized each periodic save — only the on-disk file count drops to one.

    Returns:
        ``{"backbone": …, "head": …}`` for split mode, ``{"merged": …}`` for merged.

    Atomicity: every file is written to ``<final>.pt.tmp`` first, then
    ``os.replace`` makes it the final path. We do **not** explicitly fsync (an
    earlier version did, but that interacted badly with torch.save's zip writer
    when two big saves ran back-to-back — the "unexpected pos vs vs" race). The
    state/ artifact is still written LAST by the trainer, so a crash between
    the weight + state writes leaves the resume scanner with a consistent
    older pair.
    """
    weights_dir = output_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    if save_format == "split":
        # Frozen encoder → one shared, always-overwritten backbone.pt; else tag per save.
        backbone = weights_dir / ("backbone.pt" if frozen_backbone else f"backbone_{tag}.pt")
        head = weights_dir / f"head_{tag}.pt"
        h_tmp = head.with_suffix(".pt.tmp")
        # Commit order is backbone-FIRST, head-LAST in every branch: the head is the
        # resolver's discovery marker (find_latest globs head_e*), so a head must never
        # become visible before the backbone it pairs with is already in place.
        if copy_from is not None:
            # Same-epoch best copy: only the head needs a new tag. The frozen backbone is
            # already the shared file (written by the periodic save), so it is not re-copied.
            if not frozen_backbone:
                b_tmp = backbone.with_suffix(".pt.tmp")
                shutil.copy2(copy_from["backbone"], b_tmp)
                os.replace(b_tmp, backbone)
            shutil.copy2(copy_from["head"], h_tmp)
            os.replace(h_tmp, head)
            return {"backbone": backbone, "head": head}
        b_tmp = backbone.with_suffix(".pt.tmp")
        model.save_weights(backbone=b_tmp, head=h_tmp)
        os.replace(b_tmp, backbone)
        os.replace(h_tmp, head)
        return {"backbone": backbone, "head": head}
    if save_format == "merged":
        merged = weights_dir / f"merged_{tag}.pt"
        m_tmp = merged.with_suffix(".pt.tmp")
        if copy_from is not None:
            shutil.copy2(copy_from["merged"], m_tmp)
        else:
            model.save_weights(path=m_tmp)
        os.replace(m_tmp, merged)
        return {"merged": merged}
    raise ValueError(f"unknown save_format {save_format!r}")


def clear_inference_weights(output_dir: Path) -> None:
    """Remove split/merged inference weight files from ``output_dir/weights/``.

    Called at the start of a FRESH (non-resume) train so a rerun into an existing
    output dir cannot leave stale artifacts that the resolvers would later pair with
    THIS run's weights. The dangerous case is a shorter rerun: an old higher-epoch
    ``head_e<NNN>.pt`` (or an unrefreshed ``head_best.pt``) survives and resolves
    against the newly overwritten shared ``backbone.pt`` — a mixed-generation pair.
    Clearing the inference files up front means the resolvers only ever see this run's
    artifacts. ``state/`` (the resume checkpoints) is intentionally left untouched, and
    a ``resume`` never calls this (it continues the same run).

    This wipes the whole inference namespace; the caller MUST ensure no configured input
    weight lives inside ``output_dir/weights/`` (the trainer validates and rejects that
    for fresh runs — the run owns this directory, so inputs belong elsewhere).
    """
    weights_dir = output_dir / "weights"
    if not weights_dir.exists():
        return
    for pattern in ("backbone*.pt", "head_*.pt", "merged_*.pt", "*.pt.tmp"):
        for p in weights_dir.glob(pattern):
            p.unlink()


# ---- trainer-state save / load -------------------------------------------


def _rng_state(generator: torch.Generator | None = None) -> dict[str, Any]:
    """Snapshot every RNG state omniem-train touches.

    Includes:
      * python ``random`` / ``numpy`` / ``torch`` (CPU + CUDA when present)
      * the seeded ``DataLoader`` generator (when supplied) — without this the
        sampler reorders to seed-0 ordering after a resume.
      * MONAI determinism seed (best-effort: we re-apply via
        ``set_determinism`` on restore)
    """
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    if generator is not None:
        state["dataloader_generator"] = generator.get_state()
    return state


def _restore_rng(state: dict[str, Any], *, generator: torch.Generator | None = None) -> None:
    """Restore RNG state captured by :func:`_rng_state`. Also re-seeds MONAI
    if a determinism seed was recorded (best-effort)."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    if generator is not None and "dataloader_generator" in state:
        generator.set_state(state["dataloader_generator"])


def save_trainer_state(
    *,
    output_dir: Path,
    epoch_completed: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None,
    scaler: Any | None,
    best_metric: float | None,
    best_epoch: int | None,
    dataloader_seed: int,
    config_hash_value: str,
    write_latest_pointer: bool = True,
    dataloader_generator: torch.Generator | None = None,
) -> dict[str, Path]:
    """Write the trainer-state checkpoint(s) atomically.

    Writes ``state/trainer_e<NNN>.pt`` + (when ``write_latest_pointer``) updates
    ``state/trainer_latest.pt`` — the file resume reads.
    """
    state_dir = output_dir / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch_completed": int(epoch_completed),
        "model_state": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "best_metric": (float(best_metric) if best_metric is not None else None),
        "best_epoch": (int(best_epoch) if best_epoch is not None else None),
        "dataloader_seed": int(dataloader_seed),
        "config_hash": config_hash_value,
        "rng": _rng_state(generator=dataloader_generator),
    }
    epoch_path = state_dir / f"trainer_e{int(epoch_completed):03d}.pt"
    _atomic_torch_save(payload, epoch_path)
    written = {"epoch": epoch_path}
    if write_latest_pointer:
        latest_path = state_dir / "trainer_latest.pt"
        _atomic_torch_save(payload, latest_path)
        written["latest"] = latest_path
    return written


def load_trainer_state(path: Path) -> dict[str, Any]:
    """Read a trainer-state checkpoint (``trainer_latest.pt`` or ``_e<NNN>.pt``)."""
    return torch.load(path, map_location="cpu", weights_only=False)


def restore_trainer_state(
    state: dict[str, Any],
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None,
    scaler: Any | None,
    dataloader_generator: torch.Generator | None = None,
) -> None:
    """Restore the trainer state in the canonical order: model → optim →
    sched → scaler → RNG (incl. the seeded DataLoader generator)."""
    model.load_state_dict(state["model_state"])
    optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and state.get("scheduler") is not None:
        scheduler.load_state_dict(state["scheduler"])
    if scaler is not None and state.get("scaler") is not None:
        scaler.load_state_dict(state["scaler"])
    if "rng" in state:
        _restore_rng(state["rng"], generator=dataloader_generator)


# ---- best-checkpoint resolver --------------------------------------------


def _resolve_backbone_path(weights_dir: Path, tag: str) -> Path | None:
    """Resolve a split backbone for ``tag``, tolerating the frozen-encoder layout.

    Prefers the tagged ``backbone_<tag>.pt`` (an unfrozen / fine-tuned run, where the
    backbone changes per save); falls back to the single shared ``backbone.pt`` a
    frozen-encoder run writes; returns None if neither exists. Discovery of WHICH heads
    exist is the caller's job — this only pairs a known head's tag with a backbone.
    """
    tagged = weights_dir / f"backbone_{tag}.pt"
    if tagged.exists():
        return tagged
    shared = weights_dir / "backbone.pt"
    if shared.exists():
        return shared
    return None


def find_best_inference_weights(output_dir: Path, save_format: str) -> dict[str, Path] | None:
    """Return the best-inference weight paths if they exist, else None."""
    weights_dir = output_dir / "weights"
    if save_format == "split":
        head = weights_dir / "head_best.pt"
        backbone = _resolve_backbone_path(weights_dir, "best")
        if head.exists() and backbone is not None:
            return {"backbone": backbone, "head": head}
        return None
    if save_format == "merged":
        merged = weights_dir / "merged_best.pt"
        return {"merged": merged} if merged.exists() else None
    raise ValueError(f"unknown save_format {save_format!r}")


def _parse_epoch_tag(path: Path, prefix: str) -> int | None:
    """Extract the numeric epoch from ``<prefix>_e<NNN>.pt`` filenames.

    Lexicographic sort treats ``e1000`` < ``e999``; we sort by the parsed
    integer instead so runs that reach epoch 1000+ pick the truly newest.
    """
    stem = path.stem  # backbone_e042 / head_e042 / merged_e042
    name = stem.removeprefix(f"{prefix}_e")
    try:
        return int(name)
    except ValueError:
        return None


def find_latest_inference_weights(output_dir: Path, save_format: str) -> dict[str, Path] | None:
    """Return the highest-epoch periodic inference weight paths.

    For split mode: scan ``head_e*.pt`` from highest epoch (NUMERIC sort) to lowest and
    return the first one whose backbone resolves (tagged ``backbone_<tag>.pt`` else the
    shared frozen ``backbone.pt``). Discovery keys off the HEAD because the head is the
    per-epoch artifact in BOTH layouts — a frozen run has no ``backbone_e*.pt`` to glob.
    An orphan newest head (no resolvable backbone) is skipped; the previous complete pair
    is returned.
    """
    weights_dir = output_dir / "weights"
    if not weights_dir.exists():
        return None
    if save_format == "split":
        candidates = [
            (epoch, p)
            for p in weights_dir.glob("head_e*.pt")
            if (epoch := _parse_epoch_tag(p, "head")) is not None
        ]
        candidates.sort(key=lambda t: t[0], reverse=True)
        for _epoch, head in candidates:
            tag = head.stem.replace("head_", "")
            backbone = _resolve_backbone_path(weights_dir, tag)
            if backbone is not None:
                return {"backbone": backbone, "head": head}
        return None
    if save_format == "merged":
        candidates = [
            (epoch, p)
            for p in weights_dir.glob("merged_e*.pt")
            if (epoch := _parse_epoch_tag(p, "merged")) is not None
        ]
        if not candidates:
            return None
        candidates.sort(key=lambda t: t[0], reverse=True)
        return {"merged": candidates[0][1]}
    raise ValueError(f"unknown save_format {save_format!r}")


def find_epoch_inference_weights(
    output_dir: Path, save_format: str, epoch: int
) -> dict[str, Path] | None:
    """Resolve the weight set for a given epoch (``--epoch <n>``)."""
    weights_dir = output_dir / "weights"
    tag = f"e{int(epoch):03d}"
    if save_format == "split":
        head = weights_dir / f"head_{tag}.pt"
        backbone = _resolve_backbone_path(weights_dir, tag)
        if head.exists() and backbone is not None:
            return {"backbone": backbone, "head": head}
        return None
    if save_format == "merged":
        merged = weights_dir / f"merged_{tag}.pt"
        return {"merged": merged} if merged.exists() else None
    raise ValueError(f"unknown save_format {save_format!r}")


# ---- best-tracking primitives ---------------------------------------


def _primary_metric_name(task_type: str) -> str:
    """The metric used to pick "best": dice for seg, psnr for restore."""
    if task_type == "image2label":
        return "dice"
    if task_type == "image2image":
        return "psnr"
    raise ValueError(f"unknown task_type {task_type!r}")


def is_new_best(
    *,
    cfg: RunConfig,
    metrics: dict[str, float],
    current_best: float | None,
) -> bool:
    """Return True iff ``metrics`` should become the new best.

    Rule: tracked metric = primary per task (dice/psnr), **maximize**; NaN never
    becomes best; ties → keep the earlier (return False on equal).
    """
    name = _primary_metric_name(cfg.model["task_type"])
    val = metrics.get(name)
    if val is None:
        return False
    if val != val:  # NaN check (float NaN != itself)
        return False
    if current_best is None:
        return True
    return val > current_best


def detect_save_format(output_dir: Path) -> str | None:
    """Detect whether a run dir holds split or merged weights.

    Looks at filenames present in ``output_dir/weights/`` and returns
    ``"split"`` (when any ``backbone*.pt`` — tagged or the shared frozen
    ``backbone.pt`` — or a ``head_*.pt`` is found), ``"merged"`` (when
    ``merged_*.pt`` is found), or ``None`` (neither). When both somehow coexist
    the caller's preference wins (the CLI's ``cfg.checkpoint.save_format``).
    """
    weights_dir = Path(output_dir) / "weights"
    if not weights_dir.exists():
        return None
    if any(weights_dir.glob("backbone*.pt")) or any(weights_dir.glob("head_*.pt")):
        return "split"
    if any(weights_dir.glob("merged_*.pt")):
        return "merged"
    return None


__all__ = [
    "save_inference_weights",
    "save_trainer_state",
    "load_trainer_state",
    "restore_trainer_state",
    "find_best_inference_weights",
    "find_latest_inference_weights",
    "find_epoch_inference_weights",
    "detect_save_format",
    "is_new_best",
]
