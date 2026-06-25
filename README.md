# omniem-train

`omniem-train` is the training pipeline for **OmniEM** models, from the
[EM-SSL project](https://github.com/pku-maleilab/EM-SSL-project). It is a consumer
of the [`omniem`](https://github.com/pku-maleilab/omniem-package) inference
package: it uses omniem's public API to build a model, trains a head on your data,
and writes weights back in omniem's format so the result loads straight into
omniem for inference.

One CLI trains both task families behind a single `task_type`:

- **`image2label`**: segmentation.
- **`image2image`**: restoration / denoising.

A model you finetune here imports directly into the
[napari-omniem](https://github.com/pku-maleilab/napari-omniem) plugin for GUI
inference. It supports interactive in-memory inference and large-scale on-disk
volume inference. See [Model Usage](#model-usage).

## Contents

- [Install](#install)
- [Concepts](#concepts)
- [Quick Start](#quick-start)
- [Commands](#commands)
- [Model Usage](#model-usage)

## Install

`omniem-train` requires **Python >= 3.10** and **`omniem >= 0.1.1, < 0.2`**.
It is installed from a local clone (it is not published on PyPI). The recommended
order is a fresh conda env, then omniem, then this package:

```bash
# 1. Create and activate a dedicated environment.
conda create -n omniem-train python=3.11
conda activate omniem-train

# 2. Install omniem (the inference package this builds on).
#    See the omniem README for the PyTorch + omniem install details:
#    https://github.com/pku-maleilab/omniem-package
pip install omniem            # or clone omniem-package and `pip install .`

# 3. Clone this repository and install it locally.
git clone https://github.com/pku-maleilab/omniem-train.git
cd omniem-train
pip install .                 # use `pip install -e .` for a dev/editable install

# 4. Smoke-test the install.
omniem-train list             # prints available models and encoders
```

`import omniem_train` is guarded: if `omniem` is missing, too old, or shadowed,
the first command fails with an actionable message instead of a cryptic error.

## Concepts

An OmniEM model is **a config plus weights**. This is the same idea omniem uses for
inference (see the [omniem README](https://github.com/pku-maleilab/omniem-package)
for the model-config fields and the split/merged weight forms). `omniem-train`
adds the training layer omniem leaves out:

- **Config-driven, no coding.** You describe the model entirely in the `model:`
  block of your `run.yaml`. omniem-train passes that block to omniem to build the
  model and never edits omniem's model code.

- **Train the head, optionally fine-tune the backbone.** omniem's encoder (the
  EM-DINO backbone) is pretrained on a large amount of EM data. You start from that
  backbone plus a fresh, randomly-initialized **head**. By default the backbone is
  frozen and only the head trains. This is fast and needs far less data than
  training from scratch. To also fine-tune the backbone, set
  `train.train_encoder: true`. The trained model is saved as two files, a
  `backbone` and a `head`, in omniem's format. It loads back into omniem with no
  conversion.

- **Two kinds of files are saved per run.** First, the **model weights** in
  omniem's format. These are the files you deploy and run inference with. Second, a
  **trainer-state checkpoint**. It records where training was: the epoch, the
  optimizer and learning-rate-schedule state, and the random-number state. You do
  not need it for inference. It exists only to `resume` an interrupted run from
  where it stopped.

**Getting the backbone.** The pretrained encoder backbone and example model
configs are distributed with omniem on Google Drive. Download the backbone weights
file `backbone_emdino_v1.pt` from the
[omniem weights folder](https://drive.google.com/drive/folders/1vpzVk6vDui8Aj34FdTMfJpXbt5wlMsx_?usp=drive_link),
and example model-config YAMLs from the
[omniem configs folder](https://drive.google.com/drive/folders/1cFPBmozY5VAh8ZgSe16U7ydX9RMmvbzu?usp=drive_link).

## Quick Start

Train a segmentation head, starting from the pretrained EM-DINO backbone.

**1. Get the encoder backbone.** Download `backbone_emdino_v1.pt` from the
[omniem weights folder](https://drive.google.com/drive/folders/1vpzVk6vDui8Aj34FdTMfJpXbt5wlMsx_?usp=drive_link)
and put it in a local `weights/` folder.

**2. Write a `run.yaml`.** Point it at your image/label manifests and the backbone:

```yaml
run:
  output_dir: outputs/mito_seg_001
  seed: 777
  device: auto                       # auto | cpu | cuda

model:                               # opaque block, passed verbatim to omniem.OmniEM.from_config
  arch: omniemv1
  encoder: emdinov1
  img_z: 1
  out_channels: 2
  task_type: image2label
  mean: 0.5333                       # fixed training normalization, in [0, 1] image space
  std: 0.2314

train:
  weights:
    encoder: weights/backbone_emdino_v1.pt   # pretrained backbone + random head
  train_encoder: false               # freeze the backbone; train the head

data:
  train_jsons: [manifests/train.json]
  val_jsons:   [manifests/val.json]
  axes_order: zxy
  img_size_xy: 112                   # square, a multiple of the model stride (112)
  img_size_z: 1
  match_target_shape: false          # opt-in: XY-resize each image to its target/label
  shape_mismatch: stop               # incompatible item: stop (error) | skip (drop + warn)
  resize_interp: cubic               # image resize: cubic (bicubic) | linear (bilinear)

optim:
  optimizer: adamw
  lr: 1.0e-4
  max_epochs: 100
  batch_size: 1
  lr_schedule: warmup_cosine
  warmup_epochs: 5
```

Each manifest is a JSON list of `{"image": ..., "label": ...}` items (paths
resolved relative to the manifest file). See the [full guide](docs/guide.md) for
the manifest and image/label format rules.

**3. Train.**

```bash
# Preflight: build the model + one batch + one forward, write model.yaml. No training.
omniem-train check --config run.yaml

# Train: writes weights/, state/, model.yaml, logs/ under run.output_dir.
omniem-train train --config run.yaml

# Validate against val_jsons (loss + metrics).
omniem-train validate --config run.yaml --run-checkpoint outputs/mito_seg_001

# Inference (predictions; metrics for items that carry a label).
omniem-train infer --config run.yaml --run-checkpoint outputs/mito_seg_001 \
    --infer-jsons manifests/test.json
```

The run writes `model.yaml` plus weights under `outputs/mito_seg_001/`. To use the
trained model in Python or through a GUI, see [Model Usage](#model-usage).

## Commands

| command | what it does |
|---------|--------------|
| `check` | preflight: build model + every configured dataloader, pull one batch, one forward, write `model.yaml`. No loss / metrics / optimizer. |
| `train` | full loop with in-loop validation; writes weights + trainer state + `model.yaml` + logs. |
| `validate` | standalone validation: loss + metrics on `val_jsons`. |
| `infer` | predictions; metrics for items that carry a `label`. |
| `resume` | continue from the latest trainer-state checkpoint. |
| `run` | orchestrator: `train` (with in-loop validate) then `infer` per manifests present. |
| `list` | lists available models and encoders; `list tasks` prints the task types. |

Weight forms for `validate` / `infer`: `--run-checkpoint <dir>` (auto-resolves
`model.yaml` + weights, best-else-latest), `--merged <pt>`, or
`--encoder <pt> --head <pt>`.

> **📖 Full guide: [`docs/guide.md`](docs/guide.md).** Everything in depth: the
> complete `run.yaml` schema, `--set` overrides, data-manifest and image/label
> format rules, path resolution, augmentation, the loss/metric options per task,
> the output layout, the resume contract, and how to run the tests.

## Model Usage

Your trained model is standard omniem split-format weights (a backbone + a
`head_best.pt`) plus a `model.yaml`, under `<output_dir>/weights/` and
`<output_dir>/`. With the default frozen encoder the backbone is a single shared
`backbone.pt`; if you fine-tuned it (`train.train_encoder: true`) it is the tagged
`backbone_best.pt`. Load it programmatically with omniem:

```python
from omniem import OmniEM
model = OmniEM.load(
    "outputs/mito_seg_001/model.yaml",
    backbone="outputs/mito_seg_001/weights/backbone.pt",  # backbone_best.pt if train_encoder: true
    head="outputs/mito_seg_001/weights/head_best.pt",
)
```

Or run it through a GUI in the
[napari-omniem](https://github.com/pku-maleilab/napari-omniem) plugin. It does both
interactive in-memory inference and large-scale on-disk volume inference
(sliding-window tiling and stitching, multi-GPU). To import a model you trained
here:

1. **Gather the model files** from your run's `<output_dir>`: `model.yaml` and the
   trained weights under `weights/`. Use the split pair: the backbone (`backbone.pt`
   for the default frozen encoder, or `backbone_best.pt` if you fine-tuned it) and
   `head_best.pt` (or a merged `.pt`).

2. **Create a task** in the plugin's model settings, and set its type. For an
   `image2label` (segmentation) task, define the **labels**, the semantic meaning
   of each output class (e.g. `Background`, `Mitochondria`). The number of labels
   must match the model's `out_channels`.

3. **Import the model as a solution** under that task, pointing at the `model.yaml`
   and the weights from step 1. The plugin then loads it like any built-in model.

For the exact dialogs, supported weight formats, and inference options, see the
[napari-omniem](https://github.com/pku-maleilab/napari-omniem) repository and its
documentation.
