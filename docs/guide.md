# omniem-train usage guide

This guide is the in-depth reference for driving `omniem-train`. For install and a
minimal walkthrough, start with the [README](../README.md); for the omniem model
config and the inference API, see the
[omniem README](https://github.com/pku-maleilab/omniem-package).

## Contents

- [How a training run is specified](#how-a-training-run-is-specified)
- [The `run.yaml`](#the-runyaml)
- [`--set` overrides](#--set-overrides)
- [Commands](#commands)
- [Weight forms](#weight-forms)
- [Data manifests](#data-manifests)
- [Data formats](#data-formats)
  - [Image format](#image-format)
  - [Label format](#label-format)
  - [Auto-matching input ↔ target shape](#auto-matching-input--target-shape)
  - [Examples](#examples)
- [Loss options per task](#loss-options-per-task)
- [Metrics](#metrics)
- [Augmentation](#augmentation)
- [Output layout](#output-layout)
- [Resume](#resume)
- [Extending the pipeline](#extending-the-pipeline)
- [CLI reference](#cli-reference)

## How a training run is specified

Everything about a run lives in one `run.yaml`. It carries the omniem `model:`
block (passed verbatim to `omniem.OmniEM.from_config`) plus the training-only
sections omniem does not own: data manifests, optimizer, loss, augmentation, and
checkpointing. Paths inside `run.yaml` are resolved relative to the `run.yaml`
file's own directory, so a run folder is self-contained and portable.

A model is built from the `model:` block plus weights named under `train.weights`.
The canonical fresh-training recipe is a pretrained EM-DINO **encoder** (backbone)
with a **random head**: set `train.weights.encoder` to the backbone file and leave
the head unset. With `train.train_encoder: false` (the default) the backbone is
frozen and only the head trains; set `train.train_encoder: true` to also fine-tune
the backbone (needs more data and memory).

## The `run.yaml`

```yaml
run:
  output_dir: outputs/mito_seg_001
  seed: 777
  amp: true                  # mixed precision (effective only on cuda)
  device: auto               # auto | cpu | cuda

model:                       # OPAQUE block, passed verbatim to omniem.OmniEM.from_config
  arch: omniemv1
  encoder: emdinov1
  img_z: 1                   # 1 for a 2D head, >1 for a 3D head
  out_channels: 2            # classes for image2label; 1 for image2image
  task_type: image2label     # image2label | image2image
  resize4emdino: false
  mean: 0.5333               # fixed training normalization, in [0, 1] image space
  std: 0.2314                #   (applied inside model.run)

train:
  weights:
    encoder: weights/backbone_emdino_v1.pt   # pretrained backbone + random head, OR…
    head: weights/head_only.pt               # …also load a head (split form), OR…
    merged: weights/merged.pt                # …continue from one merged whole-model file
  train_encoder: false       # false freezes the backbone (default), true also fine-tunes it

data:
  train_jsons: [manifests/train.json]
  val_jsons:   [manifests/val.json]
  infer_jsons: []
  axes_order: zxy            # on-disk axis order; the reader permutes to canonical (Y,X,Z)
  img_size_xy: 112           # square, a multiple of the model stride (omniemv1 is 112)
  img_size_z: 1
  cache_num: 24              # MONAI CacheDataset cache size
  workers: 0                 # dataloader workers (0 = main process)
  match_target_shape: false  # opt-in: XY-resize each image to its target/label
  shape_mismatch: stop       # on a shape-incompatible item: stop (error) | skip (drop + warn)
  resize_interp: cubic       # image resize interpolation: cubic (bicubic) | linear (bilinear)

optim:
  optimizer: adamw           # adam | adamw | sgd
  lr: 1.0e-4
  weight_decay: 0.0
  max_epochs: 100
  batch_size: 1
  lr_schedule: warmup_cosine # warmup_cosine | cosine_anneal | none
  warmup_epochs: 5

loss:                        # TASK-GATED (cross-task keys are rejected at parse time)
  # image2label:
  boundary_loss: false
  label_weights_alpha: 0.0   # (1/freq)^alpha class weights (set >0 to enable)
  smooth_nr: 0.0
  smooth_dr: 1.0e-5
  # image2image (rejected under image2label):
  # l1:      { weight: 1.0 }
  # l2:      { weight: 1.0 }
  # feature: { image_weight: 0.25, patch_weight: 0.25, weights: path/to/emdino.pt }
  #   `weights` is REQUIRED when the feature term is active.

aug:                         # per-process: train configurable / val deterministic / infer follows val
  train:
    enabled: true
    transforms:
      flip:   { prob: 0.5, axes: [0, 1, 2] }
      rotate: { prob: 0.5, range_x: [-3.1416, 3.1416] }
      gaussian_noise: { prob: 0.2, mean: 0.0, std: 0.1 }
  val:
    enabled: false

checkpoint:
  save_format: split         # split (backbone + head) | merged
  save_every: 50             # periodic snapshot cadence (epochs)
  val_every: 1               # in-loop validation cadence (epochs)
  save_logits: false         # infer: also write raw logits/<rel>.npy (opt-in)
```

### Derived, not configured

Anything fixed by `task_type` or architecture is derived internally and is not a
user field: the metric set, the label transform, the input normalization affine,
and the output activation. The only task-conditional user choice is the `loss:`
block.

## `--set` overrides

Dotted-path overrides layer on top of the YAML (value parsed as a YAML scalar):

```bash
omniem-train train --config run.yaml \
    --set optim.lr=2e-4 \
    --set run.output_dir=outputs/lr2e4 \
    --set model.img_z=1
```

`resume` only accepts the **resume whitelist**: `optim.max_epochs`,
`checkpoint.save_every`, `checkpoint.val_every`. Overriding anything else on a
resume changes the config hash and is rejected, so a resumed run cannot silently
drift from the run it continues.

## Commands

| command | what it does |
|---------|--------------|
| `check` | Dry-run preflight: build the model + every configured dataloader, pull one batch, run one forward, write `model.yaml`. No loss / metrics / optimizer. Use it to validate a `run.yaml` before committing to a full run. |
| `train` | Full loop with in-loop validation every `checkpoint.val_every` epochs. Writes inference weights, trainer state, `model.yaml`, and logs under `run.output_dir`. |
| `validate` | Standalone validation: reports loss + metrics on `val_jsons`. Does not write `model.yaml`. |
| `infer` | Writes predictions for every item; writes metrics for items that carry a `label` (others write the prediction only). |
| `resume` | Continues from the highest-epoch trainer-state checkpoint in the run dir. |
| `run` | Orchestrator: runs `train` (with in-loop validate) then `infer` for whichever manifests are present. After training, `infer` uses the best weights when present, else the latest. |
| `list` | Lists available models and encoders; `list tasks` prints the task types. |

## Weight forms

`validate` and `infer` accept three mutually exclusive weight forms:

```bash
--run-checkpoint <dir>            # a run output dir: resolves model.yaml + weights/
--run-checkpoint <dir> --best     # explicitly use the best weights
--run-checkpoint <dir> --epoch 42 # explicitly use the e042 set
--merged <pt>                     # a single merged checkpoint (cwd-relative)
--encoder <pt> --head <pt>        # a split checkpoint (cwd-relative)
```

With `--run-checkpoint`, the save format (split vs merged) is auto-detected from
the files on disk; the default selection is the best checkpoint, else the latest.

For **training** weights, `train.weights` in `run.yaml` accepts the same ideas:
`encoder` alone (pretrained backbone + random head, the fresh recipe),
`encoder` + `head` (split), or `merged` (continue from a whole-model file). The
split and merged forms are mutually exclusive.

## Data manifests

Each entry in `data.train_jsons` / `val_jsons` / `infer_jsons` is a path to a JSON
file, and each JSON file is a **list of item mappings**:

```json
[
  {"image": "tile_0.tif", "label": "tile_0_seg.tif"},
  {"image": "tile_1.tif", "label": "tile_1_seg.tif"}
]
```

A manifest that is not a JSON list (e.g. a top-level dict) is a hard error. Extra
keys in an item are ignored. Only `image` and `label` are consumed.

### Required keys per command

| command | `image` | `label` |
|---------|---------|---------|
| `train`    | required | required (loss / metrics need it) |
| `validate` | required | required |
| `infer`    | required | optional per item; items with a label get metrics, items without one write the prediction only |
| `check`    | required | optional (one batch is pulled; labels are not used) |

### Path resolution

- Paths inside an item (`image`, `label`) resolve **relative to the manifest
  file's directory**. Absolute paths pass through.
- The list of manifest files (`data.*_jsons`) inside `run.yaml` resolves
  **relative to the `run.yaml` file's directory**. Absolute paths pass through.
- CLI overrides like `--infer-jsons …`, `--merged`, `--encoder`, `--head`, and
  `--output-dir` resolve **relative to the current working directory**.

A self-contained layout:

```
runs/
  mito_seg.yaml              # data.train_jsons: [manifests/train.json]
  manifests/
    train.json               # {"image": "tile_0.tif", "label": "tile_0_seg.tif"}
    val.json
    tile_0.tif               # the actual image, next to the manifest
    tile_0_seg.tif
    ...
```

## Data formats

How the loader reads the `image` / `label` files a manifest points at, and how it can
auto-match their shapes.

### Image format

- **Extensions:** `.tif` / `.tiff` or `.png` / `.jpg` / `.jpeg`.
- **Channels:** single-channel grayscale only. The grayscale to 3-channel synthesis
  the encoder needs happens inside `model.run`; the loader emits raw
  `[1, Y, X, Z]` float32 tensors.
- **Dtype:** any numeric dtype the reader can load (uint8 / uint16 / float). It is
  cast to float32 unchanged. The loader does no normalization, since
  `model.run` applies the configured `model.mean` / `std`.

  > **Match `model.mean` / `model.std` to your image scale.** The reader casts
  > intensities to float **without rescaling**: a uint8 image reaches the model as
  > `[0, 255]`, a `[0, 1]` image as `[0, 1]`. `model.mean` / `model.std` must be in
  > the **same** scale as the loaded images, since `model.run` applies them directly.
  > The example above uses `[0, 1]`-domain stats (`0.5333` / `0.2314`), which assume
  > `[0, 1]`-scaled inputs; for raw uint8 inputs either pre-scale your images to
  > `[0, 1]` first, or use `[0, 255]`-domain stats (e.g. `136` / `59`). Mismatched
  > scales train and infer on a wrong normalization. (omniem warns when stats or
  > inputs fall outside `[0, 1]`.)
- **Shape:** 2D `(Y, X)` or 3D in any axis order. `data.axes_order` (default `zxy`)
  tells the reader which on-disk axis is which, and the reader permutes to
  canonical `(Y, X, Z)`. A 2D image becomes `(Y, X, 1)`; a 3D volume with
  `img_size_z == 1` keeps `Z = 1` (no silent squeeze).
- **Spatial size:** after the reader, the tile must match
  `img_size_xy × img_size_xy × img_size_z`. By default the loader does **not** rescale:
  preprocess tiles to size, or enable `data.match_target_shape` (below) to have the loader
  XY-resize each image to its target. `img_size_xy` must be a multiple of the model stride
  (`omniemv1` uses 112).

### Label format

| task | label on disk | what the loader does | constraint |
|------|---------------|----------------------|------------|
| `image2label` | integer class map (uint8 / uint16 / int) | Cast to `int64`. If the array is exactly `{0, 255}` AND `out_channels == 2`, it is remapped to `{0, 1}` (logged once per run). Otherwise the array must already be class-indexed in `[0, C-1]`. | Out-of-range values (e.g. `255` with `out_channels > 2`, or any negative value) are a hard error; real class IDs are never silently remapped. |
| `image2image` | image-like target (uint8 / uint16 / float) | Cast to float32. Integer dtypes are divided by `255.0`, then the result is clipped to `[0, 1]`; float dtypes are clipped only. | A target outside `[0, 1]` after conversion is clipped. |

In both tasks the label must share the same spatial shape as the image (after the
reader's axis reorder), **unless** `data.match_target_shape` is on, in which case the
image's XY may differ and the loader resizes the image to the label's XY (the label is
never resized; Z must still match). See *Auto-matching input ↔ target shape* below.

### Auto-matching input ↔ target shape

omniem models are **shape-preserving** (output XY == input XY). For tasks where the input
and target differ in XY (e.g. **super-resolution**: a low-res input, a high-res target),
you have two options:

1. **Pre-match the shapes yourself** (default, `match_target_shape: false`). Build manifests
   whose `image` and `label` already share spatial shape.
2. **Let omniem-train resize** (`match_target_shape: true`). The loader XY-resizes each
   **image** to its paired target/label, so the model sees the input at the target
   resolution and produces output there. The label is the untouched reference.

**Shape contract** (enforced per item when the flag is on, errors name the file paths):
- the target/label must be `img_size_xy × img_size_xy` in XY (square; already
  stride-validated) and `img_size_z` in Z;
- the image must be `img_size_z` in Z (its XY is free; it is resized to `img_size_xy`).

This keeps every tile uniform, so batching works at any `batch_size`. **Z is never
resized.** A per-item violation is governed by `shape_mismatch`: `stop` (default) raises a
clear error; `skip` drops the item and warns (a split emptied entirely by `skip` is an
error). `resize_interp` picks the image interpolation: `cubic` (bicubic, default) or
`linear` (bilinear); it applies to the image only and the result is clamped back to the
image's original value range.

**The resize is gated on a target/label being present.** Train and val always carry one
(already required), so every train/val image is resized. Infer is per-item: an item *with*
a label is resized to the label's size; an item *without* a label is predicted at its
native size (no-op), so supply a label for any infer item you want resized. Under `skip`,
shape-incompatible infer items are dropped and counted in the run summary's `skipped_shape`;
label-less native-size items are ordinary predictions, counted under `pred`.

### Examples

Segmentation (binary masks auto-normalized from `{0, 255}` to `{0, 1}`):

```json
[
  {"image": "vol_000.tif", "label": "vol_000_mito.tif"},
  {"image": "vol_001.tif", "label": "vol_001_mito.tif"}
]
```

with `model.task_type: image2label`, `model.out_channels: 2`.

Restoration (clean target scaled into `[0, 1]`):

```json
[
  {"image": "noisy_000.tif", "label": "clean_000.tif"},
  {"image": "noisy_001.tif", "label": "clean_001.tif"}
]
```

with `model.task_type: image2image`, `model.out_channels: 1`.

Infer-only (no labels; predictions only, no metrics):

```json
[
  {"image": "unlabelled_000.tif"},
  {"image": "unlabelled_001.tif"}
]
```

## Loss options per task

The `loss:` block is task-gated: keys for the wrong task are rejected at parse
time.

**`image2label`.** A Dice + cross-entropy term is always on (it softmaxes the
logits internally). Optional additions:

- `boundary_loss: true`: adds a boundary term on the predicted boundary (a 2D
  variant for `img_z == 1`, an explicit per-slice loop for 3D).
- `label_weights_alpha > 0`: `(1 / class_frequency) ^ alpha` class weights, so
  rare classes are up-weighted. With `alpha = 0` (default) all classes are equal.
- `smooth_nr` / `smooth_dr`: Dice numerator/denominator smoothing.

**`image2image`.** The image-domain terms apply a `sigmoid` to the logits
internally (the model itself returns raw logits). Weighted terms:

- `l1: { weight: ... }`: L1 between the activated prediction and the target.
- `l2: { weight: ... }`: L2.
- `feature: { image_weight, patch_weight, weights }`: an EM-DINO perceptual term
  comparing encoder features of the prediction and target. `weights` (the encoder
  backbone file) is required when this term is active. It normalizes the `[0, 1]`
  target with the configured `model.mean` / `model.std` directly, the same
  `[0, 1]`-space stats `model.run`'s affine uses (no separate rescaling).

## Metrics

Metrics are fixed per task (no `metrics:` field). They are computed on a
**metric-domain** postproc of the logits (`image2label` → argmax→one-hot;
`image2image` → `sigmoid(logits)` in `[0, 1]`), **not** the uint task-output that
`model.run(..., dtype=…)` saves, versus the reference:

- `image2label`: dice, iou, precision, recall, f1.
- `image2image`: psnr, ssim.

## Augmentation

Augmentation is per process: `aug.train` is configurable, `aug.val` is
deterministic and off by default, and inference follows the validation setting.
Each transform is a structured config (a probability plus its magnitude), so the
flip / rotate / noise magnitudes are `run.yaml` fields rather than fixed
constants. Spatial transforms (flip, rotate, elastic) move pixels in both the
image and the label; intensity transforms (noise, smooth, contrast) change pixel
values in the image only.

## Output layout

```
<output_dir>/
  model.yaml                 # clean omniem model-config (written by check + train)
  run_manifest.json          # reproducibility: config-hash, weight/state paths, versions
  logs/
    train.log                # text log (incl. data INFO messages)
    metrics.jsonl            # per-epoch loss + metrics
  weights/                   # omniem split (default) or merged inference weights
    backbone.pt              # frozen encoder (default): one shared backbone, overwritten each save
    head_best.pt   head_e<NNN>.pt
    # train_encoder: true tags the backbone instead → backbone_best.pt  backbone_e<NNN>.pt
  state/                     # trainer-state checkpoints (for resume)
    trainer_e<NNN>.pt  trainer_latest.pt
  validate/
    metrics.json
  infer/
    pred/<manifest-stem>/<rel>.tif     # uint output via model.run(..., dtype=…)
    logits/<manifest-stem>/<rel>.npy   # opt-in via checkpoint.save_logits
    metrics/<manifest-stem>/<rel>.json # written for items that carry a label
```

A trained run is reproducible from `model.yaml` + the weight file(s) +
`run_manifest.json`: load it with `OmniEM.load(...)` (split or merged form) and
continue.

A **fresh** `train` (not `resume`) into an existing `output_dir` first clears stale
inference weights from `weights/` (any `backbone*.pt` / `head_*.pt` / `merged_*.pt`),
so a shorter rerun cannot leave a prior run's higher-epoch head to be paired with this
run's weights. The `state/` checkpoints are left untouched; `resume` never clears.

Because the run owns `output_dir/weights/`, your **configured input weights**
(`train.weights.encoder` / `head` / `merged`) must live **outside** that directory. A
fresh `train` errors if an input path resolves inside `output_dir/weights/` (it would
otherwise be wiped, or (with a checkpoint-like name) mis-selected by the resolvers).

## Resume

`resume --output-dir <run dir>` continues from the highest-epoch
`state/trainer_latest.pt`, restoring the model, optimizer, scheduler, AMP scaler,
and RNG state so the run continues deterministically. Only the resume whitelist
(`optim.max_epochs`, `checkpoint.save_every`, `checkpoint.val_every`) may be
overridden with `--set`; any other override trips the config-hash guard.

## Extending the pipeline

omniem-train has no plugin system. You add a new augmentation, optimizer, LR
schedule, or loss term by editing one small factory function, plus (for
everything except augmentation) one entry in the config schema. Each addition is
local and does not touch the training loop.

### Add an augmentation transform

`aug.train.transforms` is a free-form mapping, so a new transform needs no schema
change. In `build_aug_transforms` (`omniem_train/aug/builder.py`), add a block that
reads its parameters from the config mapping and appends a MONAI dictionary
transform:

```python
# alongside the existing "flip" / "rotate" blocks in build_aug_transforms
if "my_transform" in cfg_t:
    t = cfg_t["my_transform"]
    transforms.append(
        mtf.RandSomethingd(
            keys=spatial_keys,             # see note below
            prob=float(t.get("prob", 0.5)),
            # read other parameters from t, e.g. t.get("strength", 1.0)
        )
    )
```

Use `keys=spatial_keys` for geometric transforms (they move both the image and its
label) and `keys=("image",)` for intensity transforms (image only). Then configure
it under `aug.train.transforms.my_transform` in your `run.yaml`.

### Add an optimizer

Two edits:

1. In `OptimCfg` (`omniem_train/config/schema.py`), add the name to the `optimizer`
   `Literal`, and add any new hyperparameter field it needs (e.g. `momentum`).
2. In `build_optimizer` (`omniem_train/schedule.py`), add a branch:

```python
if name == "my_optim":
    return MyOptimizer(params, lr=optim_cfg.lr, weight_decay=optim_cfg.weight_decay)
```

### Add an LR schedule

Mirror the optimizer steps:

1. Add the name to the `lr_schedule` `Literal` in `OptimCfg`.
2. Add a branch in `build_scheduler` (`omniem_train/schedule.py`) that returns a
   scheduler stepped once per epoch (or `None` for a no-op).

### Add a loss term

Loss is task-gated, so edit the side that matches the task.

**`image2label`** (`omniem_train/loss/segmentation.py` + the schema):

1. Add a field to `LossSegCfg` (`omniem_train/config/schema.py`).
2. Add that field name to `_IMAGE2LABEL_LOSS_KEYS` in the schema (the set that
   rejects cross-task keys at parse time).
3. Consume it in `build_segmentation_loss`, adding the term to the combined loss.

**`image2image`** (`omniem_train/loss/restoration.py` + the schema):

1. Add a field to `LossRestoreCfg`.
2. Add the field name to `_IMAGE2IMAGE_LOSS_KEYS`.
3. Consume it in `build_restoration_loss`.

The top-level `build_loss` (`omniem_train/loss/factory.py`) routes by `task_type`
and needs no change. Metrics are fixed per task and are not user-configurable;
changing a metric set means editing `build_metrics`
(`omniem_train/metrics/factory.py`) directly.

## CLI reference

Invocation is `omniem-train <command> [options]`. Every command returns exit code
`0` on success.

### Shared options

| option | commands | meaning |
|--------|----------|---------|
| `--config <run.yaml>` | `check`, `train`, `validate`, `infer`, `run` | Required. Path to the `run.yaml`. |
| `--set path=value` | `check`, `train`, `validate`, `infer`, `run`, `resume` | Repeatable dotted-path override; the value is parsed as a YAML scalar. On `resume`, only the whitelist (`optim.max_epochs`, `checkpoint.save_every`, `checkpoint.val_every`) is allowed. |
| `--verbose` | `train`, `validate`, `infer`, `run`, `resume` | Echo package `[INFO]` logs to stdout. Without it, only the command's result lines print. (Training writes the full `[INFO]` to `logs/train.log` as well; that file is produced by `train` / `resume` / the train phase of `run`; standalone `validate` / `infer` do not write it.) Not available on `check` / `list` / `convert-legacy`. |

### Weight selection (`validate`, `infer`)

Three mutually exclusive forms. Paths given here resolve relative to the current
working directory.

| option | meaning |
|--------|---------|
| `--run-checkpoint <dir>` | A run output dir. Auto-resolves `model.yaml` + `weights/` and the save format. Default selection: best, else latest. |
| `--best` | With `--run-checkpoint`, use the best weights. |
| `--epoch <n>` | With `--run-checkpoint`, use the `e<NNN>` weights for epoch `n`. |
| `--merged <pt>` | A single merged checkpoint. |
| `--encoder <pt> --head <pt>` | A split checkpoint (both required together). |

### Commands

| command | arguments | summary |
|---------|-----------|---------|
| `check` | `--config`, `--set` | Preflight: build model + dataloaders, one batch, one forward, write `model.yaml`. No loss / metrics / optimizer. |
| `train` | `--config`, `--set` | Full training loop with in-loop validation. Writes weights, trainer state, `model.yaml`, logs. |
| `validate` | `--config`, `--set`, weight form | Standalone validation: loss + metrics on `val_jsons`. Writes `validate/metrics.json`. |
| `infer` | `--config`, `--set`, weight form, `--infer-jsons <json>` (repeatable) | Predictions for every item; metrics for items that carry a `label`. `--infer-jsons` overrides `data.infer_jsons`. |
| `resume` | `--output-dir <dir>` (required), `--set` (whitelist only) | Continue from the highest-epoch trainer-state checkpoint in `<dir>`. |
| `run` | `--config`, `--set` | Orchestrator: `train` (with in-loop validate) then `infer` for whichever manifests are present. |
| `list` | `[encoders\|models\|tasks\|all]` (positional, default `all`) | `all` prints models and encoders; `encoders` / `models` print just one; `tasks` prints the task types. |
| `convert-legacy` | (none) | Deferred / not implemented (omniem already loads bare-key weights, so nothing needs converting). |
