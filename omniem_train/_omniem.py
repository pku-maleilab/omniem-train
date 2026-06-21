"""Single point of access to the ``omniem`` dependency, with a clear install guard.

omniem-train consumes omniem's **public API only**. Routing every first use through
:func:`require_omniem` turns the two confusing failure modes into one actionable
error:

* omniem not installed at all → a plain ``ModuleNotFoundError`` becomes a message
  pointing at the editable install of omniem in the active environment;
* omniem resolving to an **empty namespace package** (``__file__ is None``) because a
  stray ``omniem/`` directory shadows the real install — the silent failure that made
  ``model_arch_info`` raise a cryptic ``ImportError`` — becomes an explicit
  "shadowed install" message.

This module imports omniem lazily (inside the function), so importing it stays cheap.
"""

from __future__ import annotations

import re
from types import ModuleType

# Minimum omniem release omniem-train builds against — the v0.1.0 public contract.
# Kept in sync with the ``omniem>=0.1.0,<1.0`` pin in pyproject.toml.
_MIN_OMNIEM = (0, 1, 0)

# Module-level public symbols omniem-train relies on; their presence is the
# install-identity check (a namespace shadow has none of them). ``EMEncoder`` is
# used by the restoration FeatureLoss. NB: the per-class methods below are checked
# on the class, not listed here (they are not module attributes).
_REQUIRED_PUBLIC = ("OmniEM", "EMEncoder", "model_arch_info", "list_models", "list_encoders")

# Methods omniem-train hard-uses, checked on their owning class so a too-old /
# incomplete omniem fails at import rather than deep in a training step.
_REQUIRED_METHODS = {
    "OmniEM": (
        "from_config",
        "prepare_train",
        "apply_input",
        "predict",
        "apply_output",
        "save_weights",
    ),
    "EMEncoder": ("load", "forward"),
}


def _parse_release(version: str) -> tuple[int, ...]:
    """Parse the leading numeric ``N.N.N`` release of a version string into a tuple.

    Stops at the first non-numeric component, so a pre-release / local suffix
    (``0.1.0rc1``, ``0.1.0+local``) parses to its final release ``(0, 1, 0)``.
    A raw lexicographic string compare is avoided on purpose — it mishandles
    ``0.10.0`` vs ``0.2.0`` and any non-digit suffix. Returns ``()`` if nothing
    numeric leads the string.
    """
    parts: list[int] = []
    for token in str(version).split("."):
        # Take the leading digit run of each dotted component (handles "0rc1").
        digits = ""
        for ch in token:
            if ch.isdigit():
                digits += ch
            else:
                break
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


# PEP 440 pre-release / dev markers (NOT post-releases or local versions).
_PRERELEASE_RE = re.compile(r"^[._-]?(a|b|c|rc|alpha|beta|pre|preview|dev)\d*", re.IGNORECASE)


def _is_prerelease(version: str) -> bool:
    """True if ``version`` is a pre-release / dev of its numeric release.

    ``0.1.0rc1`` / ``0.1.0.dev0`` → True (they sort *before* the final ``0.1.0``);
    ``0.1.0`` / ``0.1.0+local`` (local) / ``0.1.0.post1`` (post) → False. Used to
    reject a pre-release *at* the version floor, which ``_parse_release`` alone
    would treat as equal to the final release.
    """
    main = str(version).lower().split("+", 1)[0]  # drop the local (+…) segment
    # Strip the leading numeric release (optional `v`, digits + dots) to isolate
    # the suffix that carries any pre-release / post marker.
    suffix = re.sub(r"^\s*v?[0-9]+(?:\.[0-9]+)*", "", main)
    if suffix.startswith((".post", "post", "-post")):
        return False  # post-release is *after* the final release
    return bool(_PRERELEASE_RE.match(suffix))


def require_omniem() -> ModuleType:
    """Import and return the ``omniem`` module, or raise a clear error.

    Raises:
        RuntimeError: omniem is missing, resolved to an empty namespace (shadowed),
            or is missing an expected public symbol.
    """
    try:
        import omniem
    except ModuleNotFoundError as exc:  # not installed
        raise RuntimeError(
            "omniem is not importable. omniem-train needs the omniem package "
            "installed in the active environment "
            "(`pip install -e <omniem checkout>`)."
        ) from exc

    if getattr(omniem, "__file__", None) is None:
        # A directory named `omniem/` without `__init__.py` on sys.path makes
        # `import omniem` resolve to an EMPTY namespace package — every attribute
        # access then fails cryptically. Name the real cause. This guard runs
        # BEFORE the version check below, since a namespace shadow has no
        # `__version__` to parse.
        raise RuntimeError(
            "omniem resolved to an empty namespace package (omniem.__file__ is None) "
            "— a stray `omniem/` directory is shadowing the real editable install. "
            f"omniem.__path__={list(getattr(omniem, '__path__', []))}. Remove the "
            "stray directory or fix the editable install."
        )

    # Version floor — refuse a pre-0.1.0 install (e.g. the old `0.0.0.dev0`),
    # which predates the public contract omniem-train builds against. A
    # pre-release *at* the floor (`0.1.0rc1`, `0.1.0.dev0`) is also refused: it
    # sorts before the final `0.1.0` and may not yet honor the contract.
    version = getattr(omniem, "__version__", None)
    release = _parse_release(version) if version is not None else ()
    too_old = release < _MIN_OMNIEM or (
        release == _MIN_OMNIEM and version is not None and _is_prerelease(version)
    )
    if too_old:
        want = ".".join(str(p) for p in _MIN_OMNIEM)
        raise RuntimeError(
            f"omniem {version!r} is too old for omniem-train (need >= {want} final). "
            f"Installed at {omniem.__file__}. Re-install the released omniem "
            f"(`pip install 'omniem>={want},<1.0'`) or fix the editable install."
        )

    missing = [name for name in _REQUIRED_PUBLIC if not hasattr(omniem, name)]
    # Hard-used methods live on their owning class, not the module.
    for cls_name, methods in _REQUIRED_METHODS.items():
        cls = getattr(omniem, cls_name, None)
        if cls is not None:
            missing += [f"{cls_name}.{m}" for m in methods if not hasattr(cls, m)]
    if missing:
        raise RuntimeError(
            f"omniem is importable ({omniem.__file__}, version {version!r}) but "
            f"missing expected public API: {missing}. The installed omniem is too "
            "old / incomplete for omniem-train."
        )
    return omniem
