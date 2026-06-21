"""The trainer — train / validate / infer / resume / run loops.

Design points:

* **Build via public API only** (``OmniEM.from_config`` / ``prepare_train``); the
  optimizer is taken over ``requires_grad`` params (no internal name match).
* **Device** — auto unless ``run.device`` overrides; **AMP cuda-only**.
* **Pure-logits ``predict``** → loss is on logits; restoration loss applies its
  own sigmoid internally.
* **In-loop validation** every ``checkpoint.val_every`` epochs — loss + metrics
  (metric-domain postproc).
* **Two checkpoint artifacts** — paired weights + ``trainer_e<NNN>`` state.
* **Best-tracking** — dice for seg, psnr for restore; maximize; NaN never
  best; ties → earlier; empty val → no ``_best``.
* **Resume** — read ``trainer_latest``; config-hash mismatch → hard
  error; restore model → optim → sched → scaler → RNG; continue ``epoch + 1``.
* **No TB** — text log + ``metrics.jsonl``.
* **Aug** wired via the data spine's ``apply_aug=True`` flag.
* **Deterministic seeding** via ``seed_everything``.

Public entry: :func:`train`, :func:`validate`, :func:`infer`, :func:`resume_run`.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from .checkpoint import (
    clear_inference_weights,
    detect_save_format,
    find_best_inference_weights,
    find_latest_inference_weights,
    is_new_best,
    load_trainer_state,
    restore_trainer_state,
    save_inference_weights,
    save_trainer_state,
)
from .config import (
    RESUME_OVERRIDE_WHITELIST,
    LoadedConfig,
    RunConfig,
    config_hash,
    export_inference_config,
    load_config_bundle,
)
from .data import get_loaders
from .data.loader import _resolve_run_path, get_items
from .logging import RunLogger
from .loss import build_loss
from .metrics import build_metrics
from .model import trainable_parameters
from .schedule import build_optimizer, build_scheduler
from .seed import seed_everything

if TYPE_CHECKING:
    pass


# ---- helpers --------------------------------------------------------------


def _resolve_device(cfg: RunConfig) -> str:
    """Select cuda/cpu per ``run.device``."""
    d = cfg.run.device
    if d == "cpu":
        return "cpu"
    if d == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("run.device=cuda but torch.cuda.is_available() is False")
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _amp_enabled(cfg: RunConfig, device: str) -> bool:
    """AMP on iff user opted in AND we're on cuda."""
    return bool(cfg.run.amp) and device == "cuda"


def _resolve_train_weights_for_build(bundle: LoadedConfig) -> dict[str, Any]:
    """Build the ``from_config`` kwargs from ``train.weights`` with paths."""
    w = bundle.cfg.train.weights
    out: dict[str, Any] = {}
    if w.merged is not None:
        out["weights"] = str(bundle.resolve_run_path(w.merged))
        return out
    if w.encoder is not None:
        out["encoder_weights"] = str(bundle.resolve_run_path(w.encoder))
    if w.head is not None:
        out["head_weights"] = str(bundle.resolve_run_path(w.head))
    return out


def _build_model_with_resolved_weights(bundle: LoadedConfig) -> Any:
    """Build the trainable model, applying to weight paths.

    Bypasses ``build_trainable`` only because that helper reads paths directly
    from the cfg; here we resolve them via the bundle.
    """
    from ._omniem import require_omniem
    from .config import model_yaml_text

    OmniEM = require_omniem().OmniEM
    kwargs = _resolve_train_weights_for_build(bundle)
    model = OmniEM.from_config(model_yaml_text(bundle.cfg), **kwargs)
    model.prepare_train(fix_encoder=not bundle.cfg.train.train_encoder)
    return model


def _load_inference_model(
    *,
    model_yaml: str | Path,
    weights_paths: dict[str, Path],
) -> Any:
    """Build an OmniEM in inference mode from a model.yaml + a resolved weights set.

    ``model_yaml`` accepts either a path (to a written ``model.yaml``) or an
    already-rendered inline YAML string — omniem's ``from_config`` handles both,
    which lets validate/infer skip writing a scratch file when the user passes
    ``--merged`` / ``--encoder``+``--head``.
    """
    from ._omniem import require_omniem

    OmniEM = require_omniem().OmniEM
    if isinstance(model_yaml, Path):
        model_text = model_yaml.read_text()
    else:
        model_text = model_yaml
    if "merged" in weights_paths:
        return OmniEM.from_config(model_text, weights=str(weights_paths["merged"]))
    return OmniEM.from_config(
        model_text,
        encoder_weights=str(weights_paths["backbone"]),
        head_weights=str(weights_paths["head"]),
    )


def _seed_epoch(base_seed: int, epoch: int) -> None:
    """Reseed every RNG to a deterministic function of ``(base_seed, epoch)``.

    Ensures the aug stream on epoch ``N`` is identical between an uninterrupted
    run and a resumed run that just restarted at ``N`` — without serializing
    each MONAI Random* transform's internal state.
    """
    import random as _random

    import numpy as _np

    # 64-bit-derived epoch seed; xor mixes the inputs so successive epochs
    # don't have correlated prefixes.
    epoch_seed = (int(base_seed) * 2_654_435_761 ^ int(epoch) * 2_246_822_519) & 0xFFFFFFFF
    _random.seed(epoch_seed)
    _np.random.seed(epoch_seed)
    torch.manual_seed(epoch_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(epoch_seed)
    try:
        from monai.utils import set_determinism as _set_det

        _set_det(epoch_seed)
    except Exception:  # pragma: no cover
        pass


def _gather_train_items_for_loss(bundle: LoadedConfig) -> list[dict] | None:
    """Return the train manifest items iff the seg loss needs them.

    The seg ``DiceCELoss`` runs with class weights when
    ``loss.label_weights_alpha > 0``. Used in BOTH ``train`` and
    ``validate`` so a weighted run validates with the SAME weighted loss
   .

    Raises:
        ValueError: if ``label_weights_alpha > 0`` but no train manifest is
            configured — a "weighted validate-only" config would silently
            report an unweighted loss without this guard.
    """
    cfg = bundle.cfg
    if cfg.model["task_type"] != "image2label":
        return None
    seg_cfg = cfg.typed_loss()
    if getattr(seg_cfg, "label_weights_alpha", 0.0) <= 0:
        return None
    resolved = [_resolve_run_path(p, bundle.source_dir) for p in cfg.data.train_jsons]
    if not resolved:
        raise ValueError(
            "loss.label_weights_alpha > 0 requires data.train_jsons (for the "
            "label histogram); a validate-only or infer-only config cannot "
            "compute class weights. Either set alpha=0 or include the train "
            "manifest used to derive the weights."
        )
    return get_items(resolved)


def _save_run_manifest(
    bundle: LoadedConfig,
    *,
    output_dir: Path,
    cfg_hash: str,
    weights: dict[str, Path] | None = None,
    state_paths: dict[str, Path] | None = None,
    selected_checkpoint_kind: str | None = None,
    model_config_source: str | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write ``run_manifest.json`` next to ``model.yaml``."""
    payload: dict[str, Any] = {
        "config_hash": cfg_hash,
        "model_config_source": model_config_source or "run.yaml",
        "selected_checkpoint_kind": selected_checkpoint_kind,
        "weights": {k: str(v) for k, v in (weights or {}).items()},
        "state": {k: str(v) for k, v in (state_paths or {}).items()},
        "resolved_config": bundle.cfg.model_dump(mode="json"),
        "source_run_yaml": str(bundle.source_path),
    }
    if extra:
        payload.update(extra)
    out = output_dir / "run_manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=False))
    return out


# ---- train loop ------------------------------------------------------------


def _run_train_epoch(
    *,
    model: Any,
    loader: Any,
    optimizer: torch.optim.Optimizer,
    loss_fn: Any,
    device: str,
    scaler: torch.amp.GradScaler | None,
    use_amp: bool,
    max_batches: int | None = None,
) -> float:
    """One training epoch — return the mean loss."""
    model.train()
    total = 0.0
    n = 0
    for idx, batch in enumerate(loader):
        if max_batches is not None and idx >= max_batches:
            break
        image = batch["image"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            with torch.amp.autocast(device_type=device, dtype=torch.float16):
                # Input-transform (norm + channel synthesis) then forward. Splitting
                # apply_input out of predict is byte-identical per the omniem v0.1.0
                # contract; it keeps a single normalization site across all passes.
                prepared = model.apply_input(image, axes="bcyxz")
                logits = model.predict(prepared)
                loss = loss_fn(logits, target)
            assert scaler is not None
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            prepared = model.apply_input(image, axes="bcyxz")
            logits = model.predict(prepared)
            loss = loss_fn(logits, target)
            loss.backward()
            optimizer.step()
        total += float(loss.detach().item()) * image.shape[0]
        n += int(image.shape[0])
    return total / max(1, n)


def _run_validate(
    *,
    model: Any,
    loader: Any,
    loss_fn: Any,
    agg: Any,
    device: str,
    max_batches: int | None = None,
) -> tuple[float, dict[str, float]]:
    """One eval pass — return (mean loss, metric dict)."""
    was_training = model.training
    model.eval()
    agg.reset()
    total = 0.0
    n = 0
    with torch.no_grad():
        for idx, batch in enumerate(loader):
            if max_batches is not None and idx >= max_batches:
                break
            image = batch["image"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            prepared = model.apply_input(image, axes="bcyxz")
            logits = model.predict(prepared)
            loss = loss_fn(logits, target)
            agg.update(logits, target)
            total += float(loss.detach().item()) * image.shape[0]
            n += int(image.shape[0])
    metrics = agg.aggregate() if n > 0 else {}
    if was_training:
        model.train()
    return total / max(1, n), metrics


def _save_artifacts(
    *,
    bundle: LoadedConfig,
    model: Any,
    output_dir: Path,
    epoch_completed: int,
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None,
    scaler: Any | None,
    best_metric: float | None,
    best_epoch: int | None,
    dataloader_seed: int,
    cfg_hash: str,
    is_best: bool,
    dataloader_generator: torch.Generator | None = None,
) -> dict[str, Path]:
    """Write inference + trainer-state artifacts."""
    save_format = bundle.cfg.checkpoint.save_format
    # Frozen encoder (split only) → one shared, always-current weights/backbone.pt instead
    # of a tagged backbone per save. The encoder is invariant, so per-epoch backbone copies
    # would be identical; the head still varies and stays tagged.
    frozen_backbone = save_format == "split" and not bundle.cfg.train.train_encoder
    tag = f"e{epoch_completed:03d}"
    written: dict[str, Path] = {}
    weights = save_inference_weights(
        model,
        output_dir=output_dir,
        save_format=save_format,
        tag=tag,
        frozen_backbone=frozen_backbone,
    )
    written.update({f"weights_{k}": v for k, v in weights.items()})
    if is_best:
        # The model state at this instant is identical to what we just wrote, so
        # copy the periodic files into the best filenames rather than running a
        # second torch.save (avoids a serialization race + halves the IO).
        best_weights = save_inference_weights(
            model,
            output_dir=output_dir,
            save_format=save_format,
            tag="best",
            copy_from=weights,
            frozen_backbone=frozen_backbone,
        )
        written.update({f"best_{k}": v for k, v in best_weights.items()})
    state_paths = save_trainer_state(
        output_dir=output_dir,
        epoch_completed=epoch_completed,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        best_metric=best_metric,
        best_epoch=best_epoch,
        dataloader_seed=dataloader_seed,
        config_hash_value=cfg_hash,
        dataloader_generator=dataloader_generator,
    )
    written.update({f"state_{k}": v for k, v in state_paths.items()})
    return written


def train(
    bundle: LoadedConfig,
    *,
    output_dir: Path | None = None,
    max_train_batches: int | None = None,
    max_val_batches: int | None = None,
    resume_state: dict[str, Any] | None = None,
    model_override: Any | None = None,
    loss_override: Any | None = None,
) -> dict[str, Any]:
    """Run the full training loop.

    Returns a summary dict with the best metric + the final epoch.

    Args:
        bundle: parsed config bundle.
        output_dir: override of ``cfg.run.output_dir`` (after resolution).
        max_train_batches: cap on batches/epoch (tests use this).
        max_val_batches: cap on val batches/epoch (tests).
        resume_state: pre-loaded trainer state when invoked via ``resume_run``.
    """
    cfg = bundle.cfg
    out_dir = (
        Path(output_dir) if output_dir is not None else bundle.resolve_run_path(cfg.run.output_dir)
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    # Fresh run (not a resume): clear any stale inference weights left in an existing
    # output dir, so the resolvers never mix this run's artifacts with a prior run's
    # (e.g. an old higher-epoch head surviving a shorter rerun paired with the new shared
    # backbone). A resume (resume_state set) keeps them — it continues the same run.
    #
    # The run OWNS output_dir/weights/, so configured input weights must live elsewhere:
    # an input inside that dir would either be wiped by the cleanup or, if it used a
    # generated artifact name (e.g. head_e009.pt), become resolver-visible and get
    # mis-selected. VALIDATE that contract here (raises, deletes nothing) so a bad config
    # fails fast — but DEFER the actual stale-weight deletion until after setup succeeds
    # (device + model build + dataloaders), so a missing input checkpoint or an
    # unavailable device never destroys a prior run's usable weights.
    if resume_state is None:
        weights_dir = (out_dir / "weights").resolve()
        for role, path in _resolve_train_weights_for_build(bundle).items():
            resolved = Path(path).resolve()
            if resolved == weights_dir or weights_dir in resolved.parents:
                raise ValueError(
                    f"train.weights ({role}) resolves inside the run's own output weights "
                    f"dir ({weights_dir}): {path}. Point input weights at a path OUTSIDE "
                    f"run.output_dir so a fresh run can clear stale artifacts without "
                    f"touching or mis-selecting your inputs."
                )
    logger = RunLogger(out_dir)
    cfg_hash = config_hash(cfg)

    # Determinism + seeded generator.
    dataloader_seed = int(cfg.run.seed)
    generator = seed_everything(dataloader_seed)

    device = _resolve_device(cfg)
    use_amp = _amp_enabled(cfg, device)

    logger.log(
        f"train: device={device} amp={use_amp} task={cfg.model['task_type']} "
        f"arch={cfg.model['arch']} config_hash={cfg_hash[:12]}"
    )

    # Build model + optimizer + schedule.
    # The fast trainer tests inject a tiny ``model_override`` so the loop runs
    # without building the real ViT-L.
    if model_override is not None:
        model = model_override
        if hasattr(model, "prepare_train"):
            model.prepare_train(fix_encoder=not cfg.train.train_encoder)
    else:
        model = _build_model_with_resolved_weights(bundle)
    model.to(device)
    params = (
        trainable_parameters(model)
        if model_override is None
        else [p for p in model.parameters() if p.requires_grad]
    )
    if not params:
        raise RuntimeError("no trainable parameters — prepare_train left an empty head")
    optimizer = build_optimizer(params, cfg.optim)
    scheduler = build_scheduler(optimizer, cfg.optim)
    scaler = torch.amp.GradScaler(device) if use_amp else None

    # Loss + metrics + dataloaders.
    # Train items only loaded when needed for class weights.
    train_items_for_loss = _gather_train_items_for_loss(bundle)
    feature_weights_path = None
    if cfg.model["task_type"] == "image2image":
        rest = cfg.typed_loss()
        if getattr(rest, "feature", None) is not None:
            feature_weights_path = bundle.resolve_run_path(rest.feature.weights)
    if loss_override is not None:
        loss_fn = loss_override.to(device) if hasattr(loss_override, "to") else loss_override
    else:
        loss_fn = build_loss(
            cfg,
            train_items=train_items_for_loss,
            feature_weights_path=feature_weights_path,
        ).to(device)
    metrics_agg = build_metrics(cfg)

    # Resume state restoration runs BEFORE building loaders so the RNG /
    # generator state is in the right position when the DataLoader's
    # ``CacheDataset`` first samples. For a fresh run resume_state
    # is None and we keep the seeded-from-cfg.run.seed initial position.
    start_epoch = 0
    best_metric: float | None = None
    best_epoch: int | None = None
    if resume_state is not None:
        restore_trainer_state(
            resume_state,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            dataloader_generator=generator,
        )
        start_epoch = int(resume_state["epoch_completed"]) + 1
        best_metric = resume_state.get("best_metric")
        best_epoch = resume_state.get("best_epoch")
        logger.log(f"resume: continuing at epoch {start_epoch} (best={best_metric})")

    loaders = get_loaders(
        bundle,
        splits=("train", "val"),
        require_label=True,
        generator=generator,
        base_worker_seed=dataloader_seed,
        apply_aug=True,
    )
    if "train" not in loaders:
        raise ValueError("train: no train manifest configured (data.train_jsons is empty)")
    train_loader = loaders["train"]
    val_loader = loaders.get("val")

    # Setup succeeded (device resolved, model + optimizer + loss + dataloaders all built),
    # so it is now safe to clear stale inference weights from a reused output dir. Deferring
    # the deletion to here means a failure above (missing input checkpoint, unavailable
    # device, bad manifest) leaves the previous run's weights intact. Resume never clears.
    # The input-location contract was already validated up front.
    if resume_state is None:
        clear_inference_weights(out_dir)

    # Always write model.yaml + run_manifest.json upfront.
    export_inference_config(cfg, out_dir / "model.yaml")
    _save_run_manifest(bundle, output_dir=out_dir, cfg_hash=cfg_hash)

    max_epochs = cfg.optim.max_epochs
    val_every = cfg.checkpoint.val_every
    save_every = cfg.checkpoint.save_every

    for epoch in range(start_epoch, max_epochs):
        # Deterministic per-epoch seeding: the MONAI
        # transform-internal RandomState and (re-created) worker RNG can't be
        # captured cheaply, but reseeding at the start of every epoch with a
        # function of (run.seed, epoch) makes the aug stream a pure function of
        # the epoch index — so resume reproduces the un-interrupted run's aug
        # for the resumed epoch's batches exactly. This is in addition to the
        # process-level RNG restore performed on resume above.
        _seed_epoch(dataloader_seed, epoch)

        t0 = time.time()
        train_loss = _run_train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            scaler=scaler,
            use_amp=use_amp,
            max_batches=max_train_batches,
        )
        train_dt = time.time() - t0

        # In-loop validation. Runs every val_every AND on the final epoch — so
        # a 3-epoch run with val_every=5 still gets a final pass + a _best
        # checkpoint.: "validate every val_every + a final pass".
        val_loss: float | None = None
        val_metrics: dict[str, float] = {}
        new_best = False
        is_last_epoch = (epoch + 1) == max_epochs
        if val_loader is not None and (((epoch + 1) % val_every == 0) or is_last_epoch):
            t1 = time.time()
            val_loss, val_metrics = _run_validate(
                model=model,
                loader=val_loader,
                loss_fn=loss_fn,
                agg=metrics_agg,
                device=device,
                max_batches=max_val_batches,
            )
            new_best = is_new_best(cfg=cfg, metrics=val_metrics, current_best=best_metric)
            if new_best:
                primary = "dice" if cfg.model["task_type"] == "image2label" else "psnr"
                best_metric = float(val_metrics[primary])
                best_epoch = epoch
            val_dt = time.time() - t1
            logger.log(
                f"epoch {epoch:03d} train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f} {val_metrics} "
                f"train_dt={train_dt:.1f}s val_dt={val_dt:.1f}s " + ("BEST" if new_best else "")
            )
        else:
            logger.log(f"epoch {epoch:03d} train_loss={train_loss:.4f} train_dt={train_dt:.1f}s")

        # Per-epoch JSONL line.
        logger.log_epoch(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                **{f"val_{k}": v for k, v in val_metrics.items()},
                "best_metric": best_metric,
                "best_epoch": best_epoch,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )

        # Per-epoch scheduler step — applied BEFORE the checkpoint write so
        # the persisted scheduler / optimizer LR state is the post-step state for
        # epoch+1. A resume from this checkpoint then continues at epoch+1 with
        # the correct LR, matching the uninterrupted run.
        if scheduler is not None:
            scheduler.step()

        # Checkpoint saves: every save_every epochs and at the end + whenever
        # we got a new best (so a finite run always leaves at least one set).
        save_this_epoch = ((epoch + 1) % save_every == 0) or is_last_epoch or new_best
        if save_this_epoch:
            _save_artifacts(
                bundle=bundle,
                model=model,
                output_dir=out_dir,
                epoch_completed=epoch,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                best_metric=best_metric,
                best_epoch=best_epoch,
                dataloader_seed=dataloader_seed,
                cfg_hash=cfg_hash,
                is_best=new_best,
                dataloader_generator=generator,
            )

    return {
        "best_metric": best_metric,
        "best_epoch": best_epoch,
        "epochs_run": max_epochs - start_epoch,
        "output_dir": str(out_dir),
    }


# ---- validate (standalone) -------------------------------------------------


def validate(
    bundle: LoadedConfig,
    *,
    checkpoint_dir: Path | None,
    weights: dict[str, Path] | None,
    max_val_batches: int | None = None,
) -> dict[str, Any]:
    """Standalone ML validation — load weights, run val_jsons, report loss + metrics.

    Args:
        bundle: parsed config bundle (model block may be omitted if
            ``checkpoint_dir`` is set —).
        checkpoint_dir: a run output dir; if set, weights are resolved per
            ``--best`` / latest / ``--epoch`` and model.yaml is the checkpoint's.
        weights: alternative — pre-resolved weights set (``--merged`` /
            ``--encoder``+``--head``).
        max_val_batches: cap (tests).
    """
    cfg = bundle.cfg
    device = _resolve_device(cfg)
    seed_everything(int(cfg.run.seed))

    if checkpoint_dir is not None:
        model_yaml = checkpoint_dir / "model.yaml"
        # Auto-detect the save format on disk.
        fmt = detect_save_format(checkpoint_dir) or cfg.checkpoint.save_format
        if weights is None:
            # Best else latest.
            weights = find_best_inference_weights(checkpoint_dir, fmt)
            if weights is None:
                weights = find_latest_inference_weights(checkpoint_dir, fmt)
            if weights is None:
                raise FileNotFoundError(
                    f"no weights found in {checkpoint_dir}/weights/ — train first."
                )
    else:
        # --merged or --encoder+--head: pass the model block in-memory as inline
        # YAML — no scratch file dirties the user's config tree.
        from .config import model_yaml_text

        model_yaml = model_yaml_text(cfg)
    if weights is None:  # pragma: no cover (validated above)
        raise ValueError("validate: weights resolution failed")

    model = _load_inference_model(model_yaml=model_yaml, weights_paths=weights)
    model.to(device)

    feature_weights_path = None
    if cfg.model["task_type"] == "image2image":
        rest = cfg.typed_loss()
        if getattr(rest, "feature", None) is not None:
            feature_weights_path = bundle.resolve_run_path(rest.feature.weights)
    # — gather train items when label_weights_alpha > 0 so a
    # weighted seg run validates with the SAME weighted loss as during train.
    train_items_for_loss = _gather_train_items_for_loss(bundle)
    loss_fn = build_loss(
        cfg,
        train_items=train_items_for_loss,
        feature_weights_path=feature_weights_path,
    ).to(device)
    agg = build_metrics(cfg)

    loaders = get_loaders(
        bundle,
        splits=("val",),
        require_label=True,
        apply_aug=True,
    )
    if "val" not in loaders:
        raise ValueError("validate: no val manifest configured")
    val_loss, val_metrics = _run_validate(
        model=model,
        loader=loaders["val"],
        loss_fn=loss_fn,
        agg=agg,
        device=device,
        max_batches=max_val_batches,
    )

    out_dir = bundle.resolve_run_path(cfg.run.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "validate").mkdir(parents=True, exist_ok=True)
    summary = {"val_loss": val_loss, **{f"val_{k}": v for k, v in val_metrics.items()}}
    (out_dir / "validate" / "metrics.json").write_text(json.dumps(summary, indent=2))
    return summary


# ---- infer ----------------------------------------------------------------


def _write_pred_tiff(path: Path, arr: torch.Tensor) -> None:
    """Save a uint prediction tensor as a TIFF (compresses Z=1 squeeze)."""
    from tifffile import imwrite

    path.parent.mkdir(parents=True, exist_ok=True)
    np_arr = arr.detach().cpu().numpy()
    if np_arr.ndim > 0 and np_arr.shape[-1] == 1:
        np_arr = np_arr[..., 0]
    imwrite(path, np_arr)


def _output_relative_path(item: dict, manifest_path: Path) -> Path:
    """Mirror an item's path relative to its manifest's directory."""
    image = Path(item["image"]).resolve()
    try:
        return image.relative_to(manifest_path.resolve().parent)
    except ValueError:
        # The path is outside its manifest dir (post-resolve): fall back to stem.
        return Path(image.name)


def infer(
    bundle: LoadedConfig,
    *,
    checkpoint_dir: Path | None,
    weights: dict[str, Path] | None,
    weight_selector: str = "best_else_latest",
) -> dict[str, Any]:
    """Run inference over ``data.infer_jsons``: predictions + gt-gated metrics.

    Output layout:
      ``infer/pred/<manifest-stem>/<rel>.tif``
      ``infer/logits/<manifest-stem>/<rel>.npy``       (opt-in via save_logits)
      ``infer/metrics/<manifest-stem>/<rel>.json``     (gt-gated)

    ``weight_selector`` resolves the inference weights when ``checkpoint_dir`` is
    set and ``weights`` is not: ``"best_else_latest"``, ``"latest"``, or
    ``"best"``.
    """
    from tifffile import imwrite

    cfg = bundle.cfg
    device = _resolve_device(cfg)
    seed_everything(int(cfg.run.seed))

    if checkpoint_dir is not None:
        model_yaml = checkpoint_dir / "model.yaml"
        # Auto-detect the save format on disk.
        fmt = detect_save_format(checkpoint_dir) or cfg.checkpoint.save_format
        if weights is None:
            if weight_selector == "best":
                weights = find_best_inference_weights(checkpoint_dir, fmt)
            elif weight_selector == "latest":
                weights = find_latest_inference_weights(checkpoint_dir, fmt)
            else:  # best_else_latest
                weights = find_best_inference_weights(
                    checkpoint_dir, fmt
                ) or find_latest_inference_weights(checkpoint_dir, fmt)
            if weights is None:
                raise FileNotFoundError(f"infer: no weights in {checkpoint_dir}/weights/")
    else:
        # Loose weights: pass model block as inline YAML — no scratch file
        #.
        from .config import model_yaml_text

        model_yaml = model_yaml_text(cfg)
    if weights is None:
        raise ValueError("infer: weights resolution failed")

    model = _load_inference_model(model_yaml=model_yaml, weights_paths=weights)
    model.to(device)

    # Gt-gated metrics — build the aggregator anyway; gate per item below.
    agg = build_metrics(cfg)

    out_dir = bundle.resolve_run_path(cfg.run.output_dir)
    pred_root = out_dir / "infer" / "pred"
    logits_root = out_dir / "infer" / "logits"
    metrics_root = out_dir / "infer" / "metrics"
    pred_root.mkdir(parents=True, exist_ok=True)
    if cfg.checkpoint.save_logits:
        logits_root.mkdir(parents=True, exist_ok=True)
    metrics_root.mkdir(parents=True, exist_ok=True)

    counts = {"pred": 0, "metrics": 0}
    model.eval()
    with torch.no_grad():
        for manifest_str in cfg.data.infer_jsons:
            manifest_path = Path(_resolve_run_path(manifest_str, bundle.source_dir))
            items = get_items([str(manifest_path)])
            stem = manifest_path.stem
            for item in items:
                # Run one item.
                image_arr = _read_for_infer(item["image"], axes_order=cfg.data.axes_order)
                image_t = torch.from_numpy(image_arr).unsqueeze(0).unsqueeze(0).to(device)
                prepared = model.apply_input(image_t, axes="bcyxz")
                logits = model.predict(prepared)
                pred = model.apply_output(logits, axes="bcyxz", dtype="uint8")

                rel = _output_relative_path(item, manifest_path).with_suffix(".tif")
                _write_pred_tiff(pred_root / stem / rel, pred[0])
                counts["pred"] += 1
                if cfg.checkpoint.save_logits:
                    np_logits = logits.detach().cpu().numpy()
                    npy = (logits_root / stem / rel).with_suffix(".npy")
                    npy.parent.mkdir(parents=True, exist_ok=True)
                    np.save(npy, np_logits)

                # Gt-gated metric.
                if "label" in item:
                    target_arr = _read_for_infer(item["label"], axes_order=cfg.data.axes_order)
                    target_t = (
                        _to_metric_target(
                            target_arr,
                            task_type=cfg.model["task_type"],
                            out_channels=int(cfg.model["out_channels"]),
                        )
                        .unsqueeze(0)
                        .unsqueeze(0)
                        .to(device)
                    )
                    agg.reset()
                    agg.update(logits, target_t)
                    metrics = agg.aggregate()
                    metrics_path = (metrics_root / stem / rel).with_suffix(".json")
                    metrics_path.parent.mkdir(parents=True, exist_ok=True)
                    metrics_path.write_text(json.dumps(metrics, indent=2))
                    counts["metrics"] += 1

    _ = imwrite  # silence linter (we used it inside _write_pred_tiff)
    return {"counts": counts, "output_dir": str(out_dir)}


def _read_for_infer(path: str | Path, *, axes_order: str) -> np.ndarray:
    """Read a tif/png to ``[Y, X, Z]`` float32 (mirrors LoadImageAndTarget)."""
    from .data.reader import read_image

    return read_image(path, axes_order=axes_order)


def _to_metric_target(arr: np.ndarray, *, task_type: str, out_channels: int) -> torch.Tensor:
    """Apply the per-task target normalize/dtype for gt-gated infer metrics."""
    from .data.loader import normalize_restoration_target, normalize_segmentation_label

    if task_type == "image2label":
        normed = normalize_segmentation_label(arr, out_channels=out_channels)
        return torch.from_numpy(normed).long()
    return torch.from_numpy(normalize_restoration_target(arr)).float()


# ---- resume ----------------------------------------------------------------


def resume_run(
    output_dir: Path,
    *,
    overrides: list[str] | None = None,
) -> dict[str, Any]:
    """Resume a training run from ``output_dir/state/trainer_latest.pt``.

    Reads the trainer state, reads the run's persisted ``run_manifest.json`` for
    the bundle's source path, verifies the **config-hash** (full config minus
    whitelist) — mismatch = hard error. ``--set`` only allows the whitelist.
    """
    state_dir = Path(output_dir) / "state"
    latest = state_dir / "trainer_latest.pt"
    if not latest.exists():
        raise FileNotFoundError(
            f"resume: no trainer_latest.pt at {latest} — was a `train` run completed?"
        )
    state = load_trainer_state(latest)
    expected_hash = state["config_hash"]

    # `--set` on resume is whitelisted-only.
    if overrides:
        for entry in overrides:
            path = entry.split("=", 1)[0]
            if path not in RESUME_OVERRIDE_WHITELIST:
                raise ValueError(
                    f"resume: --set {entry!r} is not in the resume whitelist "
                    f"({sorted(RESUME_OVERRIDE_WHITELIST)})"
                )

    # Find the run.yaml from the manifest.
    manifest_path = Path(output_dir) / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    source_path = Path(manifest.get("source_run_yaml") or "")
    if not source_path.exists():
        raise FileNotFoundError(f"resume: source run.yaml at {source_path} no longer exists")

    bundle = load_config_bundle(source_path, overrides=overrides)
    new_hash = config_hash(bundle.cfg)
    if new_hash != expected_hash:
        raise ValueError(
            "resume: config-hash mismatch — the resolved config (minus the "
            "resume whitelist) changed since run "
            f"(expected {expected_hash[:12]}, got {new_hash[:12]}). "
            "Allowed --set keys: " + ", ".join(sorted(RESUME_OVERRIDE_WHITELIST))
        )

    return train(
        bundle,
        output_dir=Path(output_dir),
        resume_state=state,
    )


# ---- run orchestrator ------------------------------------------------------


def run_orchestrator(
    bundle: LoadedConfig,
    *,
    max_train_batches: int | None = None,
    max_val_batches: int | None = None,
) -> dict[str, Any]:
    """``omniem-train run`` — train then infer per jsons present."""
    cfg = bundle.cfg
    summary: dict[str, Any] = {}
    if cfg.data.train_jsons:
        summary["train"] = train(
            bundle,
            max_train_batches=max_train_batches,
            max_val_batches=max_val_batches,
        )
    if cfg.data.infer_jsons:
        out_dir = bundle.resolve_run_path(cfg.run.output_dir)
        summary["infer"] = infer(
            bundle,
            checkpoint_dir=out_dir,
            weights=None,
            weight_selector="best_else_latest",
        )
    return summary


# Helper for finding latest epoch state (used by tests).


def _find_latest_epoch_state(output_dir: Path) -> Path | None:
    state_dir = Path(output_dir) / "state"
    if not state_dir.exists():
        return None
    candidates = sorted(state_dir.glob("trainer_e*.pt"))
    return candidates[-1] if candidates else None
