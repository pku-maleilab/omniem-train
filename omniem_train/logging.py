"""Tiny in-package logger — text log + per-epoch ``metrics.jsonl``.

No TensorBoard dependency in v1; the per-epoch loss + metrics are written line by
line to ``logs/metrics.jsonl`` (easy to read back into matplotlib / pandas).

Also wires every ``omniem_train.*`` Python logger (e.g. the data spine's
auto-normalize event) into ``train.log`` so package-level INFO messages are always
recorded to the file during a run. Whether those INFO messages also print to the
console is a separate decision owned by :func:`configure_console_logging` (the CLI
sets it from ``--verbose``; default is quiet — WARNING and above only).
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

_PKG_LOGGER_NAME = "omniem_train"


def configure_console_logging(verbose: bool) -> None:
    """Set the console verbosity for package ``[INFO]`` logs.

    Owns the single stdout handler on the ``omniem_train`` package logger so the CLI
    decides verbosity once per invocation. ``verbose=False`` (default) sets the handler
    to ``WARNING`` — package ``_LOG.info`` chatter is suppressed on the console, leaving
    only the CLI's own result lines (``RunLogger.log`` writes those to stdout directly,
    bypassing handlers) and genuine warnings. ``verbose=True`` sets it to ``INFO`` so
    every ``omniem_train.*`` info message is echoed.

    Idempotent: a second call reuses the tagged stdout handler and only updates its
    level (no duplicate handlers). The package logger itself stays at ``INFO`` so a
    ``RunLogger`` file handler still records the full detail to ``train.log``.
    """
    pkg_logger = logging.getLogger(_PKG_LOGGER_NAME)
    pkg_logger.setLevel(logging.INFO)
    pkg_logger.propagate = False  # avoid double-print if the root has basicConfig
    level = logging.INFO if verbose else logging.WARNING
    for handler in pkg_logger.handlers:
        if getattr(handler, "_omniem_train_path", None) == "<stdout>":
            handler.setLevel(level)
            return
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
    stream_handler._omniem_train_path = "<stdout>"  # type: ignore[attr-defined]
    pkg_logger.addHandler(stream_handler)


class RunLogger:
    """Append-only text log + per-epoch JSONL metrics writer.

    On construction, attaches the per-run ``train.log`` FileHandler to the package
    logger (``omniem_train``) at INFO so any ``_LOG.info()`` call inside the package
    (e.g. the data spine's normalize event) is recorded to the file, without the caller
    having to ``logging.basicConfig``. Whether those messages also print to the console
    is owned separately by :func:`configure_console_logging` (set from ``--verbose``);
    this class no longer attaches a stdout handler. Result lines go to stdout directly
    via :meth:`log` (bypassing handlers), so they print regardless of console verbosity.
    """

    def __init__(self, output_dir: Path) -> None:
        self.logs_dir = output_dir / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.text_path = self.logs_dir / "train.log"
        self.jsonl_path = self.logs_dir / "metrics.jsonl"
        self._attach_package_logger()

    def _attach_package_logger(self) -> None:
        """Idempotently install the ``train.log`` file handler on the omniem_train logger.

        The stdout handler is owned by :func:`configure_console_logging` (the CLI sets its
        level from ``--verbose``); here we only add the per-run FileHandler at ``INFO`` so
        ``train.log`` always records the full package detail regardless of console verbosity.
        """
        pkg_logger = logging.getLogger(_PKG_LOGGER_NAME)
        pkg_logger.setLevel(logging.INFO)
        # Mark our handler with a tag attr so re-construction reuses rather than duplicates it.
        existing_paths = {getattr(h, "_omniem_train_path", None) for h in pkg_logger.handlers}
        if str(self.text_path) not in existing_paths:
            file_handler = logging.FileHandler(self.text_path)
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
            file_handler._omniem_train_path = str(self.text_path)  # type: ignore[attr-defined]
            pkg_logger.addHandler(file_handler)
        # Don't propagate to root (avoids double-print when basicConfig was set).
        pkg_logger.propagate = False

    def log(self, msg: str) -> None:
        """Append a line to ``train.log`` and echo to stdout."""
        with open(self.text_path, "a") as f:
            f.write(msg.rstrip("\n") + "\n")
        sys.stdout.write(msg.rstrip("\n") + "\n")
        sys.stdout.flush()

    def log_epoch(self, payload: dict) -> None:
        """Append a JSON object as one line of ``metrics.jsonl``."""
        with open(self.jsonl_path, "a") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
