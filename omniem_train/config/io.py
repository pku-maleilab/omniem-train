"""Config I/O: YAML load, ``--set`` dotted-path overrides, inference-config export,
and the **path-resolution rules** + the **checkpoint-aware load**.

Path resolution rules pinned by the plan:
  * Paths inside ``run.yaml`` — ``data.*_jsons``, ``train.weights.*``,
    ``loss.feature.weights``, ``run.output_dir`` — resolve **relative to the
    run.yaml file's directory** (so a committed sample with relative paths works
    no matter the cwd). Absolute paths pass through unchanged.
  * **CLI**-passed paths (``--merged`` / ``--encoder`` / ``--head`` /
    ``--output-dir``) resolve **relative to cwd** — handled by the CLI handlers,
    not here.
  * Each manifest item's ``image`` / ``label`` path resolves **relative to that
    manifest file** — handled in the data loader.

To preserve the originals (so ``model.yaml`` and ``run_manifest.json`` keep the
user's strings) we do **not** mutate ``cfg``; we expose a small ``LoadedConfig``
helper that pairs ``cfg`` with the source-YAML directory and a ``resolve(p)``
method. Handlers receive the ``LoadedConfig``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .schema import RunConfig

# ---- LoadedConfig + path resolution -----------------------------------


@dataclass(frozen=True)
class LoadedConfig:
    """A parsed ``run.yaml`` paired with its source directory (for).

    Handlers that need a real filesystem path call :meth:`resolve_run_path` on
    each run.yaml-relative path (``data.*_jsons``, ``train.weights.*``,
    ``loss.feature.weights``, ``run.output_dir``). The originals on ``cfg`` stay
    intact so ``model.yaml`` / ``run_manifest.json`` keep the user's strings.
    """

    cfg: RunConfig
    source_path: Path  # the run.yaml file (absolute)

    @property
    def source_dir(self) -> Path:
        return self.source_path.parent

    def resolve_run_path(self, p: str | Path) -> Path:
        """Resolve a path written *inside* ``run.yaml``."""
        return resolve_relative(p, self.source_dir)


def resolve_relative(p: str | Path, base: Path) -> Path:
    """Make ``p`` absolute. Absolute paths pass through; relative paths join ``base``."""
    pp = Path(p)
    if pp.is_absolute():
        return pp
    return (base / pp).resolve()


# ---- YAML load + --set -----------------------------------------------------


def _apply_set_override(raw: dict[str, Any], assignment: str) -> None:
    """Apply one ``--set a.b.c=value`` assignment to the raw config dict in place.

    Semantics:
      * ``value`` is parsed as a **YAML scalar** (so ``true``/``false``/``null``/
        ints/floats/``[lists]`` coerce to their YAML types; bare text stays a str).
      * The dotted path navigates nested mappings, creating intermediate dicts as
        needed. **Typed sections** reject an unknown leaf later, at ``RunConfig``
        validation (``extra="forbid"``); the **opaque ``model:`` block** (a plain
        dict) accepts any added/overridden key. So we do not special-case ``model``
        here — Pydantic enforces the typed/opaque distinction.
    """
    if "=" not in assignment:
        raise ValueError(f"--set expects 'path=value' (got {assignment!r})")
    path, _, value_str = assignment.partition("=")
    keys = path.split(".")
    if not all(keys):
        raise ValueError(f"--set path has an empty segment: {path!r}")
    # Parse the RHS as a YAML scalar (safe_load of a scalar → the typed value).
    value = yaml.safe_load(value_str)

    node = raw
    for key in keys[:-1]:
        nxt = node.get(key)
        if not isinstance(nxt, dict):
            # Overwrite a non-mapping (or absent) intermediate with a fresh dict so
            # the path can be navigated; an unknown TOP-level section still trips
            # RunConfig's extra="forbid".
            nxt = {}
            node[key] = nxt
        node = nxt
    node[keys[-1]] = value


def load_run_config(
    source: str | Path,
    *,
    overrides: list[str] | None = None,
) -> RunConfig:
    """Load + validate a ``run.yaml``, applying any ``--set`` overrides first.

    Most callers want :func:`load_config_bundle` (the ``LoadedConfig`` form, which
    also remembers the source directory for path resolution); this raw form
    stays for back-compat with the existing tests.
    """
    return load_config_bundle(source, overrides=overrides).cfg


def load_config_bundle(
    source: str | Path,
    *,
    overrides: list[str] | None = None,
    checkpoint_model_block: dict[str, Any] | None = None,
) -> LoadedConfig:
    """Load + validate a ``run.yaml`` → :class:`LoadedConfig` (cfg + source dir).

    When ``checkpoint_model_block`` is set:
      * If the user config OMITS ``model:``, we splice the checkpoint's block in
        (the checkpoint's ``model.yaml`` is authoritative —).
      * If the user config provides ``model:``, it MUST match the checkpoint's
        block exactly (semantic compare of the parsed dicts); disagreement is a
        **hard error** —.
    """
    source_path = Path(source).resolve()
    text = source_path.read_text()
    raw = yaml.safe_load(text)
    if not isinstance(raw, dict):
        raise ValueError(f"run config must be a YAML mapping (got {type(raw).__name__})")
    for assignment in overrides or []:
        _apply_set_override(raw, assignment)

    # checkpoint splicing / consistency check.
    if checkpoint_model_block is not None:
        if "model" not in raw or raw.get("model") in (None, {}):
            raw["model"] = checkpoint_model_block
        else:
            user_block = raw["model"]
            if not _model_blocks_equal(user_block, checkpoint_model_block):
                raise ValueError(
                    "validate/infer: --config `model` block disagrees with the "
                    "checkpoint's model.yaml. Either omit `model:` from --config (use "
                    "the checkpoint's) or make them match exactly."
                )

    cfg = RunConfig.model_validate(raw)
    return LoadedConfig(cfg=cfg, source_path=source_path)


def _model_blocks_equal(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """Semantic equality between two opaque model blocks (round-tripped via YAML).

    YAML round-trip normalises floats/ints/bools/None so a user-typed ``mean: 136``
    matches a checkpoint's ``mean: 136.0`` if and only if omniem treats them as
    equivalent (which it does — both are ``float``).
    """
    return yaml.safe_load(yaml.safe_dump(a, sort_keys=True)) == yaml.safe_load(
        yaml.safe_dump(b, sort_keys=True)
    )


def model_yaml_text(run_cfg: RunConfig) -> str:
    """Serialize the opaque ``model:`` block to its ``model.yaml`` text.

    This is the **single source** of the omniem-config dump: ``build_trainable``
    feeds this exact string to ``OmniEM.from_config`` and
    :func:`export_inference_config` writes this exact string to ``model.yaml`` — so
    "what we build" and "what we export" are byte-identical by construction.
    ``sort_keys=False`` preserves the author's key order (semantic pass-through, not
    byte-for-byte of the source file).
    """
    return yaml.safe_dump(run_cfg.model, sort_keys=False)


def export_inference_config(run_cfg: RunConfig, dest: str | Path) -> Path:
    """Write the clean omniem model-config (``model.yaml``) to ``dest``.

    Shared by ``check`` and (in later phases) ``train``/``validate``; given the same
    ``run_cfg`` it always writes identical bytes. Creates the parent dir if missing.

    Returns:
        The path written.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(model_yaml_text(run_cfg))
    return dest


def read_model_yaml(path: str | Path) -> dict[str, Any]:
    """Read a written ``model.yaml`` back into a plain dict (the opaque block)."""
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError(f"model.yaml at {path} must be a YAML mapping")
    return data
