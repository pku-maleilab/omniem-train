"""omniem-train run-config schema (Pydantic v2) — *parse + validate only*.

The guiding rule is **compose, not subclass**: the ``model:`` block is held
**opaquely** (a plain mapping = exactly omniem's ``ModelConfig`` YAML) and is
never modelled here — it is handed to ``OmniEM.from_config`` (which validates
omniem's own fields) and dumped verbatim as ``model.yaml``. The train-only
fields that have no place in ``model:`` live in a separate ``train:`` section.

Anything fully determined by ``task_type`` / architecture (``target_scale``,
the metric set, the label transform, input normalization, output activation)
is **derived internally** and is deliberately *not* a user field here.

Highlights:
  * Typed per-task loss schema (``LossSegCfg`` / ``LossRestoreCfg``);
    cross-task keys are rejected at parse time.
  * ``LossFeatureCfg.weights`` is required when the ``feature`` term is
    active (no implicit fallback to ``train.weights.encoder``).
  * ``CheckpointCfg`` carries only the cadence + format + opt-in flags;
    weights / state / ``model.yaml`` are always written.
  * ``AugCfg`` typed per-process (``train`` / ``val``); transforms held
    loosely inside.
  * ``RunCfg.device`` selects ``auto`` / ``cpu`` / ``cuda``.
  * ``config_hash`` computes a SHA-256 over the resolved config **minus**
    the resume whitelist, so a changed loss / weights / optimizer / aug /
    model block trips a hard error on resume.

torch stays unimported; the lazy ``model_arch_info`` lookup is the only
cross-call into omniem (a cheap catalog read).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Task families recognised by omniem-train (mirrors omniem's TaskType).
TaskType = Literal["image2label", "image2image"]

# Loss keys allowed per task_type. image2label = the seg loss knobs;
# image2image = the restoration loss terms (FACL / FFL are NOT supported).
_IMAGE2LABEL_LOSS_KEYS = {"boundary_loss", "label_weights_alpha", "smooth_nr", "smooth_dr"}
_IMAGE2IMAGE_LOSS_KEYS = {"l1", "l2", "feature"}


class _StrictModel(BaseModel):
    """Base for every typed sub-config.

    ``extra="forbid"`` is what makes ``--set <typed.path>=…`` on an *unknown* field
    an error: a stray key in a typed
    section fails validation, while the opaque ``model:`` mapping (a plain dict, not
    a model) accepts any key. ``protected_namespaces=()`` lets us name a field
    ``model`` without a Pydantic ``model_``-namespace warning.
    """

    model_config = ConfigDict(extra="forbid", protected_namespaces=())


class RunCfg(_StrictModel):
    """Top-level run knobs."""

    output_dir: str
    seed: int = 0
    amp: bool = True  # mixed precision (effective only on cuda)
    # Device selection. ``auto`` picks cuda when available else cpu; the user
    # can force one. AMP is off on cpu (omniem rejects fp16-on-CPU).
    device: Literal["auto", "cpu", "cuda"] = "auto"


class WeightsCfg(_StrictModel):
    """Weight inputs for ``from_config``.

    ``merged`` is mutually exclusive with ``encoder``/``head`` — enforced on the
    parent ``TrainCfg`` so the error names the ambiguity rather than silently
    letting ``from_config`` ignore the split kwargs.
    """

    encoder: str | None = None  # → from_config(encoder_weights=…): pretrained backbone
    head: str | None = None  # → from_config(head_weights=…): pretrained head
    merged: str | None = None  # → from_config(weights=…): single merged checkpoint


class TrainCfg(_StrictModel):
    """Train-only fields, kept OUT of the pure ``model:`` block."""

    weights: WeightsCfg = Field(default_factory=WeightsCfg)
    # false → prepare_train(fix_encoder=True): frozen emdino backbone (canonical);
    # true → also train the encoder. The head (decoder + out + STAdapter) always trains.
    train_encoder: bool = False

    @model_validator(mode="after")
    def _reject_merged_with_split(self) -> TrainCfg:
        """``merged`` together with ``encoder``/``head`` is ambiguous → reject."""
        w = self.weights
        if w.merged is not None and (w.encoder is not None or w.head is not None):
            raise ValueError(
                "train.weights: `merged` is mutually exclusive with `encoder`/`head` "
                "(pass either a single merged checkpoint OR the split encoder/head "
                "files, not both)."
            )
        return self


class DataCfg(_StrictModel):
    """Data manifests + loader knobs.

    ``img_size_xy`` is the square XY side; it must be a multiple of the model stride
    (validated on ``RunConfig`` against ``model_arch_info(arch).stride``).
    ``target_scale`` / ``normalize`` / metrics are DERIVED, not fields.
    """

    train_jsons: list[str] = Field(default_factory=list)
    val_jsons: list[str] = Field(default_factory=list)
    infer_jsons: list[str] = Field(default_factory=list)
    axes_order: str = "zxy"  # on-disk axis convention for the reader
    img_size_xy: int = Field(gt=0)
    img_size_z: int = Field(default=1, gt=0)
    cache_num: int = Field(default=24, ge=0)
    workers: int = Field(default=0, ge=0)


class OptimCfg(_StrictModel):
    """Optimizer + schedule."""

    optimizer: Literal["adam", "adamw", "sgd"] = "adamw"
    lr: float = 1.0e-4
    weight_decay: float = 0.0
    momentum: float = 0.9  # sgd only
    max_epochs: int = Field(default=100, gt=0)
    batch_size: int = Field(default=1, gt=0)
    lr_schedule: Literal["warmup_cosine", "cosine_anneal", "none"] = "warmup_cosine"
    warmup_epochs: int = Field(default=0, ge=0)


class CheckpointCfg(_StrictModel):
    """Checkpoint cadence.

    The trainer ALWAYS writes weights / trainer-state / ``model.yaml`` (the toggles
    were removed: validate/infer/resume all depend on them being present). The
    only knobs left are the cadence and the format/logits-opt-in.
    """

    save_format: Literal["split", "merged"] = "split"
    save_every: int = Field(default=50, gt=0)  # periodic snapshot cadence
    val_every: int = Field(default=1, gt=0)  # in-loop validation cadence
    save_logits: bool = False  # infer-only: also write raw logits/<rel>.npy


# ---- Loss typed schema ----------------------------------------------------


class LossSegCfg(_StrictModel):
    """``image2label`` loss recipe."""

    boundary_loss: bool = False
    # (1/freq)^alpha class weights computed from the train manifest label histogram
    # when alpha > 0. Alpha = 0 → unweighted (the canonical recipe).
    label_weights_alpha: float = Field(default=0.0, ge=0.0)
    smooth_nr: float = Field(default=0.0, ge=0.0)
    smooth_dr: float = Field(default=1e-5, gt=0.0)


class LossTermCfg(_StrictModel):
    """A scalar image-domain restoration term (L1 / L2): ``{weight: …}``. A bare
    ``l1: true`` is invalid."""

    weight: float = Field(default=1.0, ge=0.0)


class LossFeatureCfg(_StrictModel):
    """The emdino-perceptual restoration term.

    No implicit fallback to ``train.weights.encoder`` (a merged checkpoint may
    expose none); missing → config error.
    """

    weight: float = Field(default=1.0, ge=0.0)
    image_weight: float = Field(default=1.0, ge=0.0)
    patch_weight: float = Field(default=1.0, ge=0.0)
    weights: str  # REQUIRED: emdino backbone path for FeatureLoss


class LossRestoreCfg(_StrictModel):
    """``image2image`` loss recipe."""

    l1: LossTermCfg | None = None
    l2: LossTermCfg | None = None
    feature: LossFeatureCfg | None = None

    @model_validator(mode="after")
    def _at_least_one_term(self) -> LossRestoreCfg:
        if not any([self.l1, self.l2, self.feature]):
            raise ValueError(
                "loss (image2image) requires at least one of `l1`, `l2`, or `feature` "
                "(got empty); see."
            )
        return self


# ---- Augmentation ----------------------------------------------------


class AugProcessCfg(_StrictModel):
    """Per-process aug recipe (``train`` configurable / ``val`` deterministic).

    The ``transforms`` mapping is held loosely here (each transform has its own
    little schema in the aug builder); ``extra="forbid"`` still rejects unknown
    top-level keys (``enabled`` and ``transforms`` are the only two).
    """

    enabled: bool = True
    transforms: dict[str, Any] = Field(default_factory=dict)


class AugCfg(_StrictModel):
    """Per-process aug. ``val`` defaults to disabled
    (deterministic eval for validation); infer follows
    val (no separate infer aug block)."""

    train: AugProcessCfg = Field(default_factory=AugProcessCfg)
    val: AugProcessCfg = Field(default_factory=lambda: AugProcessCfg(enabled=False))


# ---- Top-level RunConfig ----------------------------------------------------


class RunConfig(_StrictModel):
    """The whole ``run.yaml`` — composes the typed sections + the opaque blocks.

    ``model`` and ``loss`` are held as **plain mappings**:
      * ``model`` is omniem's config (opaque passthrough → ``from_config`` /
        ``model.yaml``); omniem-train reads only ``task_type`` and ``arch`` from it
        (for cross-validation) and never models omniem's fields.
      * ``loss`` is task-gated (keys checked against ``model.task_type``); the
        per-task typed shape is validated below — call :meth:`typed_loss` to
        get the typed object back at use site.
    """

    run: RunCfg
    model: dict[str, Any]  # OPAQUE — exactly omniem's ModelConfig YAML
    train: TrainCfg = Field(default_factory=TrainCfg)
    data: DataCfg
    optim: OptimCfg = Field(default_factory=OptimCfg)
    loss: dict[str, Any] = Field(default_factory=dict)  # task-gated (see validator)
    aug: AugCfg = Field(default_factory=AugCfg)
    checkpoint: CheckpointCfg = Field(default_factory=CheckpointCfg)

    # ---- model-block sanity (omniem-train's slice; omniem validates the rest) -----

    @field_validator("model")
    @classmethod
    def _model_has_required_keys(cls, v: dict[str, Any]) -> dict[str, Any]:
        """The opaque block must carry the two keys omniem-train reads downstream.

        ``arch`` (→ stride lookup + ``from_config``) and ``task_type`` (→ loss
        gating + the derived output side). All *other* model fields are omniem's to
        validate inside ``from_config`` — we do not re-check them here.
        """
        for key in ("arch", "task_type"):
            if key not in v:
                raise ValueError(f"model.{key} is required in the model block")
        task_type = v["task_type"]
        if task_type not in ("image2label", "image2image"):
            raise ValueError(
                f"model.task_type must be 'image2label' or 'image2image' (got {task_type!r})"
            )
        return v

    @model_validator(mode="after")
    def _cross_validate(self) -> RunConfig:
        """Cross-field rules that span sections."""
        task_type = self.model["task_type"]
        arch = self.model["arch"]

        # img_size_xy must be a multiple of the model stride (square side; the
        # reader yields a square XY tile). Stride comes from omniem's arch catalog
        # — imported lazily (via the install guard) so the schema import stays
        # torch-free and a shadowed/absent omniem fails with a clear message.
        from .._omniem import require_omniem

        stride = require_omniem().model_arch_info(arch).stride
        if self.data.img_size_xy % stride != 0:
            raise ValueError(
                f"data.img_size_xy ({self.data.img_size_xy}) must be a multiple of the "
                f"model stride ({stride}) for arch {arch!r}"
            )

        # loss keys are task-gated: reject keys that belong to the other task FIRST
        # (gives the historical "not valid for task_type" error before the typed
        # validation below trips with the less-specific "extra forbidden").
        allowed = _IMAGE2LABEL_LOSS_KEYS if task_type == "image2label" else _IMAGE2IMAGE_LOSS_KEYS
        foreign = set(self.loss) - allowed
        if foreign:
            other = "image2image" if task_type == "image2label" else "image2label"
            raise ValueError(
                f"loss keys {sorted(foreign)} are not valid for task_type={task_type!r} "
                f"(they belong to {other}); allowed keys: {sorted(allowed)}"
            )

        # typed per-task validation of the (now task-aligned) loss block.
        # extra="forbid" on the sub-cfg rejects mistyped knobs; LossRestoreCfg
        # also enforces "at least one term".
        if task_type == "image2label":
            LossSegCfg.model_validate(self.loss)
        else:
            LossRestoreCfg.model_validate(self.loss)
        return self

    # ---- typed accessors ------------------------------------------------------

    def typed_loss(self) -> LossSegCfg | LossRestoreCfg:
        """Return the validated per-task loss config."""
        if self.model["task_type"] == "image2label":
            return LossSegCfg.model_validate(self.loss)
        return LossRestoreCfg.model_validate(self.loss)


# ---- resume whitelist + config-hash --------------------------------------

# Fields that may change freely on `resume` (--set whitelist + excluded from the
# config-hash). Everything else trips a hard error on resume. Tuple paths
# point into ``cfg.model_dump()``.
_RESUME_WHITELIST: tuple[tuple[str, ...], ...] = (
    ("optim", "max_epochs"),
    ("checkpoint", "save_every"),
    ("checkpoint", "val_every"),
)

# Same whitelist, as dotted strings (for the `--set` rule check on resume).
RESUME_OVERRIDE_WHITELIST: frozenset[str] = frozenset({".".join(p) for p in _RESUME_WHITELIST})


def config_hash(cfg: RunConfig) -> str:
    """SHA-256 of the resolved config MINUS the resume whitelist.

    The hash mismatches on resume if anything load-bearing changed (loss,
    weights, optimizer/lr/schedule, batch_size, aug, model, data) — so a silent
    different-experiment resume is impossible. Only ``max_epochs``,
    ``save_every``, and ``val_every`` are deliberately excluded (you may resume a
    20-epoch run for 10 more epochs without tripping the hash).
    """
    raw = cfg.model_dump(mode="json")
    for path in _RESUME_WHITELIST:
        node = raw
        for key in path[:-1]:
            if not isinstance(node, dict):
                break
            node = node.get(key, {})
        if isinstance(node, dict):
            node.pop(path[-1], None)
    blob = json.dumps(raw, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()
