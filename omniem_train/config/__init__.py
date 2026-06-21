"""omniem-train config package — the ``run.yaml`` schema + I/O helpers.

Importing this subpackage pulls in Pydantic and (lazily, on cross-validation) the
omniem arch catalog — heavier than the bare ``import omniem_train``, which is why
the schema lives here and not at the top-level package.
"""

from .io import (
    LoadedConfig,
    export_inference_config,
    load_config_bundle,
    load_run_config,
    model_yaml_text,
    read_model_yaml,
    resolve_relative,
)
from .schema import (
    RESUME_OVERRIDE_WHITELIST,
    AugCfg,
    AugProcessCfg,
    CheckpointCfg,
    DataCfg,
    LossFeatureCfg,
    LossRestoreCfg,
    LossSegCfg,
    LossTermCfg,
    OptimCfg,
    RunCfg,
    RunConfig,
    TrainCfg,
    WeightsCfg,
    config_hash,
)

__all__ = [
    "RunConfig",
    "RunCfg",
    "DataCfg",
    "OptimCfg",
    "TrainCfg",
    "WeightsCfg",
    "CheckpointCfg",
    "AugCfg",
    "AugProcessCfg",
    "LossSegCfg",
    "LossRestoreCfg",
    "LossFeatureCfg",
    "LossTermCfg",
    "LoadedConfig",
    "load_run_config",
    "load_config_bundle",
    "export_inference_config",
    "model_yaml_text",
    "read_model_yaml",
    "resolve_relative",
    "config_hash",
    "RESUME_OVERRIDE_WHITELIST",
]
