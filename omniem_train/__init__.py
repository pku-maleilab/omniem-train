"""omniem-train — training pipeline for OmniEM models.

A **public-API consumer** of the frozen ``omniem`` inference package: it builds /
trains models through ``omniem``'s public surface only (``OmniEM.from_config`` /
``prepare_train`` / ``predict`` / ``apply_output`` / ``save_weights`` + the arch
catalog) and never reaches into omniem internals.

Import-lightness contract: the package top level exports **only**
``__version__``. The config schema, CLI, model glue, and data stub are submodule
imports — so a bare ``import omniem_train`` does not eagerly pull
``config`` → ``omniem`` (which would drag in torch). Import what you need:
``from omniem_train.config import RunConfig`` / ``from omniem_train.cli import main``.
"""

# Path-based version (hatchling regex-extracts this; the build never imports the
# package). Mirrors omniem's own 0.0.0.dev0 pre-release versioning.
__version__ = "0.0.0.dev0"

__all__ = ["__version__"]
