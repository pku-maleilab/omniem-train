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

# Supported omniem version window: ``_MIN_OMNIEM`` is the inclusive floor,
# ``_MAX_OMNIEM`` the EXCLUSIVE ceiling. omniem-train builds against omniem's
# channel-less ``run`` / ``predict`` public surface, available in this window and
# kept in sync with the ``omniem>=0.1.1,<0.2`` pin in pyproject.toml. The ceiling is
# enforced at runtime (not only by pip) because an editable install ignores the pip
# constraint — without it a 0.2.x with a changed surface would load silently.
_MIN_OMNIEM = (0, 1, 1)
_MAX_OMNIEM = (0, 2, 0)

# Module-level public symbols omniem-train relies on; their presence is the
# install-identity check (a namespace shadow has none of them). ``EMEncoder`` is
# used by the restoration FeatureLoss; ``InputContractError`` is the error type
# omniem raises for an invalid input contract. NB: the per-class methods below are
# checked on the class, not listed here (they are not module attributes).
_REQUIRED_PUBLIC = (
    "OmniEM",
    "EMEncoder",
    "InputContractError",
    "model_arch_info",
    "list_models",
    "list_encoders",
)

# Methods omniem-train hard-uses, checked on their owning class so a too-old /
# incomplete omniem fails at import rather than deep in a training step. ``run`` is
# the one-shot raw-image -> output path; ``predict`` is the channel-less canonical
# (pre-built tensor) path.
_REQUIRED_METHODS = {
    "OmniEM": (
        "from_config",
        "prepare_train",
        "run",
        "predict",
        "save_weights",
    ),
    "EMEncoder": ("load", "run"),
}


def _parse_release(version: str) -> tuple[int, ...]:
    """Parse the leading numeric ``N.N.N`` release of a version string into a tuple.

    Stops at the first non-numeric component, so a pre-release / local suffix
    (``0.1.1rc1``, ``0.1.1+local``) parses to its final release ``(0, 1, 1)``.
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

    ``0.1.1rc1`` / ``0.1.1.dev0`` → True (they sort *before* the final ``0.1.1``);
    ``0.1.1`` / ``0.1.1+local`` (local) / ``0.1.1.post1`` (post) → False. Used to
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

    version = getattr(omniem, "__version__", None)
    release = _parse_release(version) if version is not None else ()
    # Compare on a 3-component tuple so a short version like "0.2" is treated as
    # "0.2.0": a shorter tuple sorts below its zero-padded form, so "0.2" -> (0, 2)
    # would otherwise slip under the (0, 2, 0) ceiling.
    cmp = release + (0,) * (3 - len(release)) if len(release) < 3 else release
    floor = ".".join(str(p) for p in _MIN_OMNIEM)
    ceil = ".".join(str(p) for p in _MAX_OMNIEM)

    # Version floor — refuse anything below 0.1.1, which lacks the channel-less
    # `run`/`predict` surface omniem-train builds against. A pre-release *at* the
    # floor (`0.1.1rc1`, `0.1.1.dev0`) is also refused: it sorts before the final
    # `0.1.1` and may not yet honor the contract.
    too_old = cmp < _MIN_OMNIEM or (
        cmp == _MIN_OMNIEM and version is not None and _is_prerelease(version)
    )
    if too_old:
        raise RuntimeError(
            f"omniem {version!r} is too old for omniem-train (need >= {floor} final). "
            f"Installed at {omniem.__file__}. Re-install a supported omniem "
            f"(`pip install 'omniem>={floor},<{ceil}'`) or fix the editable install."
        )

    # Version ceiling — refuse 0.2.0+ at RUNTIME (an editable install bypasses the
    # pip `<0.2` pin). A pre-1.0 minor bump may break the public contract, so an
    # unverified API is refused rather than loaded. Anything that parses to >= 0.2.0
    # is rejected, including a 0.2.0 pre-release (`0.2.0rc1` parses to (0, 2, 0)).
    if cmp >= _MAX_OMNIEM:
        raise RuntimeError(
            f"omniem {version!r} is too new for omniem-train (need < {ceil}; this "
            f"omniem-train builds against the {floor} contract and has not been "
            f"verified against {ceil}+). Installed at {omniem.__file__}. Install a "
            f"supported omniem (`pip install 'omniem>={floor},<{ceil}'`)."
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
