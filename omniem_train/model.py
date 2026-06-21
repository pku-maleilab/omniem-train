"""Model build glue — the public-API bridge to ``omniem``.

``build_trainable`` is the *one* place omniem-train turns a :class:`RunConfig` into
a trainable :class:`omniem.OmniEM`. It uses omniem's public surface only:

    OmniEM.from_config(<model block as inline YAML>, **weight_kwargs)
        .prepare_train(fix_encoder=not train.train_encoder)

No ``_net`` reach-in, no name-based parameter-group selection — the only freeze
control is the encoder (``prepare_train`` derives the backbone prefix itself). torch
is imported lazily inside the helpers so importing this module stays cheap.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .config import RunConfig, model_yaml_text

if TYPE_CHECKING:  # avoid importing omniem/torch at module import time
    from omniem import OmniEM


def _weight_kwargs(cfg: RunConfig) -> dict[str, Any]:
    """Map ``train.weights`` to ``from_config``'s keyword arguments.

    The schema already rejected ``merged`` together with ``encoder``/``head``
    (ambiguous), so the two branches here are exclusive:
      * ``merged`` set → ``weights=<merged>`` (whole-model load);
      * else → ``encoder_weights=`` / ``head_weights=`` (separable; an unset group
        keeps its random init — ``encoder_weights=<emdino>`` alone is the canonical
        pretrained-backbone + random-head recipe).
    Passing ``None`` for an unset path is a no-op in ``from_config`` (random init).
    """
    w = cfg.train.weights
    if w.merged is not None:
        return {"weights": w.merged}
    return {"encoder_weights": w.encoder, "head_weights": w.head}


def build_trainable(cfg: RunConfig) -> OmniEM:
    """Build an omniem model from ``cfg`` and hand it over for training.

    Steps (public API only):
      1. ``OmniEM.from_config(model_yaml_text(cfg), **weight_kwargs)`` — build from
         the opaque ``model:`` block (inline YAML, the exact text exported as
         ``model.yaml``); omniem validates the model fields and loads any weights.
      2. ``.prepare_train(fix_encoder=not cfg.train.train_encoder)`` — flip the
         inference-frozen model into a trainable state; freeze the encoder backbone
         unless ``train_encoder`` is set. The head (decoder + out + STAdapter)
         always trains.

    The returned model is in ``train()`` mode with ``requires_grad`` set per the
    freeze policy; build the optimizer from its ``requires_grad`` parameters (see
    :func:`trainable_parameters`).
    """
    from ._omniem import require_omniem

    OmniEM = require_omniem().OmniEM
    model = OmniEM.from_config(model_yaml_text(cfg), **_weight_kwargs(cfg))
    model.prepare_train(fix_encoder=not cfg.train.train_encoder)
    return model


def trainable_parameters(model: OmniEM) -> list[Any]:
    """The optimizer's parameter list = every ``requires_grad`` parameter.

    This is the *only* selection rule — no ``_net`` reach-in, no name matching. After
    ``prepare_train`` the frozen-encoder/trainable-head partition is already encoded
    in each parameter's ``requires_grad`` flag.

    A future per-group LR (encoder vs head) would key off
    ``omniem.EMEncoder.name_parameter_group()`` (the public, live-derived encoder
    dotted name) rather than re-deriving the split here.
    """
    return [p for p in model.parameters() if p.requires_grad]
