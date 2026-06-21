"""omniem-train command-line entry point.

Subcommands:
  * ``check``     — dry-run preflight: build model + dataloaders + one forward;
                    write ``model.yaml``. No loss / metrics / optimizer.
  * ``list``      — pass-through to omniem's catalog (models / encoders / tasks).
  * ``train``     — full loop with in-loop validation; writes ``weights/``,
                    ``state/``, ``model.yaml``, ``logs/``.
  * ``validate``  — standalone ML validation: loss + metrics on ``val_jsons``.
  * ``infer``     — predictions + gt-gated metrics over ``infer_jsons``.
  * ``resume``    — continue training from ``state/trainer_latest.pt``.
  * ``run``       — orchestrator: ``train`` (+ in-loop validate) then ``infer``.
  * ``convert-legacy`` — deferred stub (omniem already saves/loads bare-key
    weights, so pretrained omniem-format checkpoints load directly; nothing
    to convert).

Heavy imports (torch + omniem via the trainer) are deferred into the handlers
so ``import omniem_train.cli`` stays light.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Preference order for which loader supplies ``check``'s single forward batch.
_CHECK_BATCH_ORDER = ("train", "val", "infer")


def _add_config_arg(parser: argparse.ArgumentParser) -> None:
    """Shared ``--config`` + ``--set`` options for config-driven subcommands."""
    parser.add_argument("--config", required=True, help="path to run.yaml")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="path=value",
        dest="overrides",
        help="dotted-path override (value parsed as a YAML scalar); repeatable",
    )


def _add_verbose_arg(parser: argparse.ArgumentParser) -> None:
    """Shared ``--verbose`` flag for the commands that emit package ``[INFO]``.

    Attached to the work commands (train / validate / infer / run / resume); NOT to the
    one-line ``check`` preflight or the ``list`` / ``convert-legacy`` helpers. Without it,
    only the command's result lines print; with it, package ``[INFO]`` is echoed too.
    """
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="echo package [INFO] logs to stdout (default: result lines only)",
    )


def _add_weight_args(parser: argparse.ArgumentParser) -> None:
    """Shared weight-form options for validate/infer.

    Three mutually exclusive forms:
      * ``--run-checkpoint <dir>`` — a run output dir (resolves ``model.yaml`` +
        ``weights/``); ``--best`` / ``--epoch <n>`` pick which weights.
      * ``--merged <pt>`` — a single merged checkpoint (no ``model.yaml``; the
        ``--config`` ``model:`` block must be present).
      * ``--encoder <pt> --head <pt>`` — split form (ditto).
    """
    parser.add_argument(
        "--run-checkpoint",
        dest="run_checkpoint",
        default=None,
        help="run output dir (resolves model.yaml + weights/)",
    )
    parser.add_argument(
        "--best",
        action="store_true",
        help="when --run-checkpoint set, use the best weights (default: best else latest)",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=None,
        help="when --run-checkpoint set, use the e<NNN> weights for this epoch",
    )
    parser.add_argument("--merged", default=None, help="single merged .pt")
    parser.add_argument("--encoder", default=None, help="split: encoder backbone .pt")
    parser.add_argument("--head", default=None, help="split: head .pt")


def _resolve_weights_arg(args, save_format: str) -> tuple[Path | None, dict | None, str]:
    """Resolve the (checkpoint_dir, weights_dict, selector) tuple from CLI args.

    When ``--run-checkpoint <dir>`` is used, the save format is **auto-detected**
    from the files on disk. The caller's
    ``cfg.checkpoint.save_format`` is only used as a fallback when nothing is
    present (e.g. a fresh directory).
    """
    from .checkpoint import (
        detect_save_format,
        find_best_inference_weights,
        find_epoch_inference_weights,
    )

    if args.run_checkpoint is not None:
        ckpt_dir = Path(args.run_checkpoint).resolve()
        fmt = detect_save_format(ckpt_dir) or save_format
        if args.epoch is not None:
            weights = find_epoch_inference_weights(ckpt_dir, fmt, args.epoch)
            if weights is None:
                raise FileNotFoundError(
                    f"--epoch {args.epoch}: no matching weight set in {ckpt_dir}/weights/"
                )
            return ckpt_dir, weights, "epoch"
        if args.best:
            weights = find_best_inference_weights(ckpt_dir, fmt)
            if weights is None:
                raise FileNotFoundError(f"--best: no best weights in {ckpt_dir}/weights/")
            return ckpt_dir, weights, "best"
        return ckpt_dir, None, "best_else_latest"
    if args.merged is not None:
        return None, {"merged": Path(args.merged).resolve()}, "merged"
    if args.encoder is not None and args.head is not None:
        return (
            None,
            {
                "backbone": Path(args.encoder).resolve(),
                "head": Path(args.head).resolve(),
            },
            "split",
        )
    raise SystemExit(
        "weights required: pass --run-checkpoint <dir>, --merged <pt>, "
        "or --encoder <pt> --head <pt>"
    )


# ---- check ---------------------------------------------------------------


def _cmd_check(args: argparse.Namespace) -> int:
    """Preflight: parse → build model + dataloaders → one batch + one forward →
    write ``output_dir/model.yaml``. No loss/metrics/optimiser."""
    import torch

    from .config import export_inference_config, load_config_bundle
    from .data import get_loaders
    from .trainer import _build_model_with_resolved_weights

    bundle = load_config_bundle(args.config, overrides=args.overrides)
    cfg = bundle.cfg

    # Build through the SAME bundle-resolving path `train` uses, so `check`
    # resolves train.weights.* relative to the run.yaml directory. This makes the
    # preflight catch a missing / misplaced weights file here instead of at the
    # first training step. (build_trainable reads the cfg paths verbatim, which
    # resolves them against cwd -- a different file than train would load.)
    model = _build_model_with_resolved_weights(bundle)

    loaders = get_loaders(bundle, splits=_CHECK_BATCH_ORDER, require_label=False)
    if not loaders:
        print(
            "check: no manifests configured (data.train_jsons / val_jsons / infer_jsons "
            "are all empty) — cannot pull a batch.",
            file=sys.stderr,
        )
        return 1

    batches = {s: next(iter(loaders[s])) for s in _CHECK_BATCH_ORDER if s in loaders}
    split = next(s for s in _CHECK_BATCH_ORDER if s in batches)
    image = batches[split]["image"]

    model.eval()
    with torch.no_grad():
        prepared = model.apply_input(image, axes="bcyxz")
        logits = model.predict(prepared)

    out_dir = bundle.resolve_run_path(cfg.run.output_dir)
    model_yaml = export_inference_config(cfg, out_dir / "model.yaml")

    print(
        f"check OK: built {cfg.model['arch']} ({cfg.model['task_type']}); "
        f"loaded one batch from splits {sorted(batches)}; "
        f"forward on '{split}' batch {tuple(image.shape)} -> logits {tuple(logits.shape)}; "
        f"wrote {model_yaml}"
    )
    return 0


# ---- list ----------------------------------------------------------------


def _cmd_list(args: argparse.Namespace) -> int:
    """Pass-through to omniem's catalog helpers."""
    from ._omniem import require_omniem

    omniem = require_omniem()
    what = args.what or "all"
    if what in ("models", "all"):
        print("models:   " + ", ".join(omniem.list_models()))
    if what in ("encoders", "all"):
        print("encoders: " + ", ".join(omniem.list_encoders()))
    if what == "tasks":
        print("tasks:    image2label, image2image")
    return 0


# ---- train ---------------------------------------------------------------


def _cmd_train(args: argparse.Namespace) -> int:
    from .config import load_config_bundle
    from .trainer import train

    bundle = load_config_bundle(args.config, overrides=args.overrides)
    summary = train(bundle)
    print(json.dumps({"train": summary}, indent=2))
    return 0


# ---- validate ------------------------------------------------------------


def _cmd_validate(args: argparse.Namespace) -> int:
    from .config import load_config_bundle, read_model_yaml
    from .trainer import validate

    # when --run-checkpoint, splice the checkpoint's model.yaml so the user
    # can omit `model:` in their --config. Mismatch is a hard error.
    checkpoint_model_block = None
    if args.run_checkpoint is not None:
        model_yaml = Path(args.run_checkpoint) / "model.yaml"
        if not model_yaml.exists():
            print(
                f"validate: --run-checkpoint {args.run_checkpoint}: no model.yaml found",
                file=sys.stderr,
            )
            return 2
        checkpoint_model_block = read_model_yaml(model_yaml)
    bundle = load_config_bundle(
        args.config,
        overrides=args.overrides,
        checkpoint_model_block=checkpoint_model_block,
    )

    ckpt_dir, weights, _selector = _resolve_weights_arg(args, bundle.cfg.checkpoint.save_format)
    summary = validate(bundle, checkpoint_dir=ckpt_dir, weights=weights)
    print(json.dumps({"validate": summary}, indent=2))
    return 0


# ---- infer ---------------------------------------------------------------


def _cmd_infer(args: argparse.Namespace) -> int:
    from .config import load_config_bundle, read_model_yaml
    from .trainer import infer

    checkpoint_model_block = None
    if args.run_checkpoint is not None:
        model_yaml = Path(args.run_checkpoint) / "model.yaml"
        if not model_yaml.exists():
            print(
                f"infer: --run-checkpoint {args.run_checkpoint}: no model.yaml found",
                file=sys.stderr,
            )
            return 2
        checkpoint_model_block = read_model_yaml(model_yaml)
    bundle = load_config_bundle(
        args.config,
        overrides=args.overrides,
        checkpoint_model_block=checkpoint_model_block,
    )

    # Allow --infer-jsons to override data.infer_jsons. CLI paths are
    # resolved relative to **cwd**, so we make them absolute here
    # before assigning to the bundle (which then treats them as already-absolute).
    if args.infer_jsons:
        bundle.cfg.data.infer_jsons.clear()
        bundle.cfg.data.infer_jsons.extend(str(Path(p).resolve()) for p in args.infer_jsons)

    ckpt_dir, weights, selector = _resolve_weights_arg(args, bundle.cfg.checkpoint.save_format)
    summary = infer(bundle, checkpoint_dir=ckpt_dir, weights=weights, weight_selector=selector)
    print(json.dumps({"infer": summary}, indent=2))
    return 0


# ---- resume --------------------------------------------------------------


def _cmd_resume(args: argparse.Namespace) -> int:
    from .trainer import resume_run

    summary = resume_run(Path(args.output_dir), overrides=args.overrides)
    print(json.dumps({"resume": summary}, indent=2))
    return 0


# ---- run -----------------------------------------------------------------


def _cmd_run(args: argparse.Namespace) -> int:
    from .config import load_config_bundle
    from .trainer import run_orchestrator

    bundle = load_config_bundle(args.config, overrides=args.overrides)
    summary = run_orchestrator(bundle)
    print(json.dumps({"run": summary}, indent=2))
    return 0


# ---- convert-legacy (deferred stub) --------------------------------------


def _cmd_convert_legacy(_args: argparse.Namespace) -> int:
    print(
        "omniem-train convert-legacy: DEFERRED in v1.\n"
        "omniem already loads merged/split/partial bare-key weights directly, so "
        "pretrained omniem-format checkpoints need no conversion. Nothing to convert "
        "today; revisit when a specific old checkpoint trips a direct "
        "`from_config(weights=…)` load.",
        file=sys.stderr,
    )
    return 2


# ---- parser --------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="omniem-train", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_check = sub.add_parser(
        "check", help="dry-run preflight: build + one forward + write model.yaml"
    )
    _add_config_arg(p_check)
    p_check.set_defaults(func=_cmd_check)

    p_list = sub.add_parser("list", help="list omniem encoders/models/tasks")
    p_list.add_argument(
        "what",
        nargs="?",
        choices=["encoders", "models", "tasks", "all"],
        default="all",
    )
    p_list.set_defaults(func=_cmd_list)

    p_train = sub.add_parser("train", help="run a training loop")
    _add_config_arg(p_train)
    _add_verbose_arg(p_train)
    p_train.set_defaults(func=_cmd_train)

    p_validate = sub.add_parser("validate", help="standalone ML validation")
    _add_config_arg(p_validate)
    _add_weight_args(p_validate)
    _add_verbose_arg(p_validate)
    p_validate.set_defaults(func=_cmd_validate)

    p_infer = sub.add_parser("infer", help="run inference + gt-gated metrics")
    _add_config_arg(p_infer)
    _add_weight_args(p_infer)
    _add_verbose_arg(p_infer)
    p_infer.add_argument(
        "--infer-jsons",
        action="append",
        default=[],
        help="override data.infer_jsons; repeatable",
    )
    p_infer.set_defaults(func=_cmd_infer)

    p_resume = sub.add_parser("resume", help="resume a training run")
    p_resume.add_argument("--output-dir", required=True, help="run output dir")
    p_resume.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="path=value",
        dest="overrides",
        help="whitelisted resume overrides (max_epochs / save_every / val_every)",
    )
    _add_verbose_arg(p_resume)
    p_resume.set_defaults(func=_cmd_resume)

    p_run = sub.add_parser("run", help="orchestrator: train → infer")
    _add_config_arg(p_run)
    _add_verbose_arg(p_run)
    p_run.set_defaults(func=_cmd_run)

    p_convert = sub.add_parser(
        "convert-legacy",
        help="DEFERRED — omniem already loads bare-key weights; nothing to convert",
    )
    p_convert.set_defaults(func=_cmd_convert_legacy)

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    argv = sys.argv[1:] if argv is None else argv
    parser = build_parser()
    args = parser.parse_args(argv)
    # Single console-verbosity decision per invocation. Commands without a --verbose
    # flag (check / list / convert-legacy) default to quiet (getattr → False).
    from .logging import configure_console_logging

    configure_console_logging(getattr(args, "verbose", False))
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
