# YOLOv5n Grayscale SiLU Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a tested YOLOv5n v7.0 training path whose images, model input, pretrained initialization, validation, and FPGA-facing contract are all native `256×256×1` grayscale.

**Architecture:** Keep `third_party/yolov5-v7.0-original` immutable and apply a small, explicit channel parameter to the `third_party/yolov5` working copy. Add project-owned model/run YAML files and a preflight launcher that refuses to start training unless the dataloader, first convolution, pretrained RGB-to-gray conversion, class count, and three detection heads match the ZYNQ-7010 contract.

**Tech Stack:** Python 3.12, PyTorch 2.11, torchvision, OpenCV, PyYAML, YOLOv5 v7.0, `unittest`, PowerShell.

## Global Constraints

- Never modify `third_party/yolov5-v7.0-original`.
- Dataset is `data/GC10-DET-YOLO-ROI256`; do not regenerate or overwrite it in this plan.
- Input is native `256×256×1` grayscale, not replicated RGB.
- Classes remain `0..9`; each detection scale has `3 × (5 + 10) = 45` output channels.
- Baseline activation remains upstream SiLU.
- Initialize from official `yolov5n.pt`; convert only the first RGB convolution by arithmetic mean over its input-channel dimension.
- Training defaults are 300 epochs, batch size 32, seed 20260927, device 0, no image cache.
- Generated weights and run outputs stay under `runs/` and are ignored by Git.
- Immutable backup baseline: 144 files, tree SHA-256 `4ce25aa77ae56fae91dcf61dcdc72a2c9ec052a32d82fc245156132eb6014698`.

---

### Task 0: Preserve and version the upstream baseline

**Files:**

- Modify: `.gitignore`
- Add unchanged upstream tree: `third_party/yolov5/`
- Preserve locally: `third_party/yolov5-v7.0-original/`

**Interfaces:**

- Produces: a tracked pristine YOLOv5 v7.0 working baseline before adaptation.
- Protects: the local immutable backup from accidental Git staging.

- [ ] **Step 1: Verify the backup before any vendor edit.**

Compute a deterministic tree hash by sorting all relative file paths and feeding each UTF-8 path, a null byte, and its file bytes into SHA-256. Expected: 144 files and digest `4ce25aa77ae56fae91dcf61dcdc72a2c9ec052a32d82fc245156132eb6014698`.

- [ ] **Step 2: Ignore the immutable backup.**

Add this exact rule to `.gitignore`:

```gitignore
third_party/yolov5-v7.0-original/
```

- [ ] **Step 3: Confirm the working copy is still identical to the backup.**

Use `git diff --no-index -- third_party/yolov5-v7.0-original third_party/yolov5`. Expected: exit code 0 and no output.

- [ ] **Step 4: Commit the pristine upstream working copy.**

```powershell
git add .gitignore third_party/yolov5
git commit -m "build: vendor YOLOv5 v7.0"
```

This commit must precede all adaptation commits so later diffs show only project changes.

---

### Task 1: Native grayscale detection dataloader

**Files:**

- Modify: `third_party/yolov5/utils/dataloaders.py`
- Create: `models/tests/test_yolov5_gray_dataloader.py`

**Interfaces:**

- Produces: `read_image(path: str | Path, channels: int) -> numpy.ndarray | None`.
- Extends: `create_dataloader(..., channels: int = 3)`.
- Extends: `LoadImagesAndLabels(..., channels: int = 3)`.
- Guarantees: a `channels=1` sample is read internally as `H×W×1` and returned as `1×H×W`.

- [ ] **Step 1: Write a grayscale fixture and failing read test.**

Add a fixture that creates four `256×256` grayscale PNG files and paired YOLO labels beneath temporary `images/train` and `labels/train` directories. Add this test before production changes:

```python
def test_read_image_preserves_singleton_channel_dimension(self) -> None:
    with fixture_yolo_dataset() as image_dir:
        image = read_image(str(next(image_dir.glob("*.png"))), channels=1)
        self.assertEqual(image.shape, (256, 256, 1))
```

- [ ] **Step 2: Run the focused test and verify RED.**

Run:

```powershell
& 'D:\Conda\conda_envs\zynq-defect\python.exe' -m unittest models.tests.test_yolov5_gray_dataloader -v
```

Expected: import failure for `read_image` or an unexpected three-channel shape.

- [ ] **Step 3: Implement channel-aware image reading.**

In `utils/dataloaders.py`, add:

```python
def read_image(path, channels=3):
    if channels not in (1, 3):
        raise ValueError(f"channels must be 1 or 3, got {channels}")
    flag = cv2.IMREAD_GRAYSCALE if channels == 1 else cv2.IMREAD_COLOR
    image = cv2.imread(str(path), flag)
    if image is not None and channels == 1 and image.ndim == 2:
        image = image[..., None]
    return image


def ensure_hwc(image, channels):
    if channels == 1 and image.ndim == 2:
        return image[..., None]
    return image
```

Use `read_image` in `load_image`, `check_cache_ram`, and `cache_images_to_disk`. Call `ensure_hwc` after OpenCV resize/warp/letterbox operations that may drop a singleton channel.

- [ ] **Step 4: Run the read test and verify GREEN.**

Run the command from Step 2. Expected: the read test passes.

- [ ] **Step 5: Write failing dataset and Mosaic tests.**

Add tests that instantiate `LoadImagesAndLabels` twice, once with `augment=False` and once with `augment=True`, using a complete YOLO hyperparameter dictionary. Assert:

```python
self.assertEqual(dataset[0][0].shape, (1, 256, 256))
self.assertEqual(augmented_dataset.load_mosaic(0)[0].shape, (256, 256, 1))
```

Also call `create_dataloader(..., channels=1, workers=0)` and assert the first batch has shape `(2, 1, 256, 256)`.

- [ ] **Step 6: Run the focused tests and verify RED.**

Expected: `channels` is not accepted yet, or HSV/Albumentations/Mosaic fails on one channel.

- [ ] **Step 7: Propagate `channels` and make augmentations channel-safe.**

Make these exact behavior changes:

```python
self.channels = channels
self.albumentations = Albumentations(size=img_size) if augment and channels == 3 else None
```

Pass `channels` from `create_dataloader` into `LoadImagesAndLabels`. Guard the Albumentations call with `if self.albumentations`, run `augment_hsv` only when `self.channels == 3`, preserve geometric augmentation for both modes, and call `ensure_hwc` before `img.transpose((2, 0, 1))`. Existing Mosaic allocation already uses `img.shape[2]` and must remain dynamic.

- [ ] **Step 8: Run the focused tests and all existing conversion tests.**

Run:

```powershell
& 'D:\Conda\conda_envs\zynq-defect\python.exe' -m unittest models.tests.test_yolov5_gray_dataloader -v
& 'D:\Conda\conda_envs\zynq-defect\python.exe' -m unittest discover -s models/tests -v
```

Expected: all tests pass.

- [ ] **Step 9: Commit the dataloader change.**

```powershell
git add third_party/yolov5/utils/dataloaders.py models/tests/test_yolov5_gray_dataloader.py
git commit -m "feat: add native grayscale YOLOv5 dataloader"
```

---

### Task 2: Single-channel model construction and pretrained conversion

**Files:**

- Modify: `third_party/yolov5/train.py`
- Modify: `third_party/yolov5/val.py`
- Modify: `third_party/yolov5/utils/general.py`
- Modify: `third_party/yolov5/utils/torch_utils.py`
- Create: `models/tests/test_yolov5_gray_model.py`

**Interfaces:**

- Produces: `adapt_input_conv(state_dict: dict[str, Tensor], target_state_dict: dict[str, Tensor], input_channels: int) -> tuple[dict[str, Tensor], str | None]`.
- Extends training CLI: `--channels {1,3}`, default 3.
- Extends validation API/CLI: `channels: int = 3` and `--channels {1,3}`.
- Extends AMP check: `check_amp(model, image_channels: int = 3) -> bool`.

- [ ] **Step 1: Write failing unit tests for RGB-to-gray conversion.**

Use synthetic tensors so the test is deterministic:

```python
def test_adapt_input_conv_averages_rgb_channels(self) -> None:
    rgb = torch.arange(2 * 3 * 2 * 2, dtype=torch.float32).reshape(2, 3, 2, 2)
    source = {"model.0.conv.weight": rgb.clone()}
    target = {"model.0.conv.weight": torch.empty(2, 1, 2, 2)}
    adapted, key = adapt_input_conv(source, target, input_channels=1)
    self.assertEqual(key, "model.0.conv.weight")
    torch.testing.assert_close(adapted[key], rgb.mean(dim=1, keepdim=True))
```

Add a second test proving `input_channels=3` leaves the state dict unchanged and returns `None`.

- [ ] **Step 2: Run the focused model test and verify RED.**

Run:

```powershell
& 'D:\Conda\conda_envs\zynq-defect\python.exe' -m unittest models.tests.test_yolov5_gray_model -v
```

Expected: import failure for `adapt_input_conv`.

- [ ] **Step 3: Implement the minimal pretrained conversion helper.**

In `utils/torch_utils.py`, find the first target four-dimensional convolution weight whose target input dimension equals `input_channels`. If the matching source tensor has three input channels and all other dimensions match a one-channel target, replace only that state-dict value with `source.mean(dim=1, keepdim=True)`. Return the copied state dict and converted key. Raise `ValueError` when `input_channels == 1` but no compatible first convolution exists; do not silently train with a random first layer.

- [ ] **Step 4: Run the conversion tests and verify GREEN.**

Run the command from Step 2. Expected: conversion tests pass.

- [ ] **Step 5: Write failing model-contract and CLI tests.**

Load the custom model YAML path that Task 3 will create via a temporary equivalent YAML fixture, instantiate `Model(..., ch=1, nc=10)`, and assert:

```python
self.assertEqual(model.model[0].conv.in_channels, 1)
self.assertEqual([layer.out_channels for layer in model.model[-1].m], [45, 45, 45])
```

Patch `sys.argv` and assert `train.parse_opt().channels == 1` when `--channels 1` is supplied. Add an AMP-input test that patches the inference image and proves `check_amp(..., image_channels=1)` creates a one-channel sample.

- [ ] **Step 6: Run the model tests and verify RED.**

Expected: CLI and AMP functions do not yet accept the channel value.

- [ ] **Step 7: Thread `channels` through training and validation.**

In `train.py`:

- add `channels = opt.channels` in `train()`;
- construct both pretrained and scratch models with `Model(..., ch=channels, ...)`;
- call `adapt_input_conv` before `intersect_dicts` when pretrained;
- log the converted key;
- call `check_amp(model, image_channels=channels)`;
- pass `channels=channels` into both train and validation dataloaders;
- add `parser.add_argument('--channels', type=int, choices=(1, 3), default=3)`.

In `val.py`, add `channels=3` to `run`, use it in warmup shape and standalone `create_dataloader`, expose `--channels`, and pass it from `main`. Training-time validation continues using the already-created grayscale dataloader.

In `utils/general.py`, make `check_amp` use a local `numpy.ones((640, 640, image_channels), dtype=uint8)` input for one-channel models rather than the RGB bus image.

- [ ] **Step 8: Run model tests and the complete test suite.**

Run:

```powershell
& 'D:\Conda\conda_envs\zynq-defect\python.exe' -m unittest models.tests.test_yolov5_gray_model -v
& 'D:\Conda\conda_envs\zynq-defect\python.exe' -m unittest discover -s models/tests -v
```

Expected: all tests pass.

- [ ] **Step 9: Commit the model/CLI change.**

```powershell
git add third_party/yolov5/train.py third_party/yolov5/val.py third_party/yolov5/utils/general.py third_party/yolov5/utils/torch_utils.py models/tests/test_yolov5_gray_model.py
git commit -m "feat: train YOLOv5 with grayscale pretrained weights"
```

---

### Task 3: Project-owned baseline configuration and preflight launcher

**Files:**

- Create: `configs/yolov5n_gray_256.yaml`
- Create: `configs/yolov5n_gray_silu_baseline.yaml`
- Create: `tools/train_yolov5n_silu.py`
- Create: `models/tests/test_yolov5_gray_preflight.py`
- Modify: `.gitignore`
- Modify: `PROJECT_STRUCTURE.md`

**Interfaces:**

- Produces: `load_run_config(path: Path) -> dict`.
- Produces: `run_preflight(project_root: Path, run_config: dict) -> dict`.
- CLI: `python tools/train_yolov5n_silu.py [--config PATH] [--preflight-only]`.

- [ ] **Step 1: Write failing run-config tests.**

The test loads `configs/yolov5n_gray_silu_baseline.yaml` and asserts these exact values:

```python
self.assertEqual(config["data"], "gc10_yolo.yaml")
self.assertEqual(config["cfg"], "configs/yolov5n_gray_256.yaml")
self.assertEqual(config["weights"], "yolov5n.pt")
self.assertEqual(config["imgsz"], 256)
self.assertEqual(config["channels"], 1)
self.assertEqual(config["epochs"], 300)
self.assertEqual(config["batch_size"], 32)
self.assertEqual(config["seed"], 20260927)
self.assertIsNone(config["cache"])
```

- [ ] **Step 2: Run the preflight tests and verify RED.**

Run:

```powershell
& 'D:\Conda\conda_envs\zynq-defect\python.exe' -m unittest models.tests.test_yolov5_gray_preflight -v
```

Expected: config or launcher module does not exist.

- [ ] **Step 3: Add the model and run YAML files.**

Copy the YOLOv5n v7.0 graph from `third_party/yolov5/models/yolov5n.yaml` into `configs/yolov5n_gray_256.yaml`, keep all anchors/layers unchanged, and set `nc: 10` plus a comment that input channels are supplied by `--channels 1`.

Create the run YAML with exactly:

```yaml
data: gc10_yolo.yaml
cfg: configs/yolov5n_gray_256.yaml
weights: yolov5n.pt
hyp: third_party/yolov5/data/hyps/hyp.scratch-low.yaml
imgsz: 256
channels: 1
epochs: 300
batch_size: 32
seed: 20260927
device: "0"
workers: 4
cache: null
project: runs
name: yolov5n_gray256_silu_baseline
exist_ok: false
```

- [ ] **Step 4: Implement strict config loading.**

`load_run_config` must reject missing keys, values other than `channels: 1`, values other than `imgsz: 256`, non-null cache, or a dataset YAML whose `nc` is not 10. Resolve file paths relative to project root, not the current shell directory.

- [ ] **Step 5: Run config tests and verify GREEN.**

Run the command from Step 2. Expected: run-config tests pass.

- [ ] **Step 6: Write failing preflight-contract tests.**

Use the four-image fixture from Task 1 and a temporary data YAML. Assert the returned report contains:

```python
self.assertEqual(report["batch_shape"][1:], [1, 256, 256])
self.assertEqual(report["first_conv_in_channels"], 1)
self.assertEqual(report["head_out_channels"], [45, 45, 45])
self.assertTrue(report["pretrained_first_conv_verified"])
self.assertEqual(report["status"], "ready")
```

Use a synthetic checkpoint in the unit test so it does not require network access.

- [ ] **Step 7: Run the preflight test and verify RED.**

Expected: `run_preflight` is absent or does not produce the contract report.

- [ ] **Step 8: Implement preflight and the training handoff.**

`run_preflight` must:

1. insert `third_party/yolov5` into `sys.path`;
2. load data/model/run YAMLs;
3. build `Model(model_yaml, ch=1, nc=10)`;
4. read one batch using `create_dataloader(..., channels=1, workers=0)`;
5. perform one no-gradient model forward pass;
6. inspect the three detection-head convolution outputs;
7. load or receive the checkpoint state dict and verify the averaged first layer numerically;
8. return the report above or raise a descriptive exception.

The CLI runs preflight first. With `--preflight-only`, print JSON and exit. Otherwise call `train.run(**run_config)` only after report status is `ready`.

- [ ] **Step 9: Run preflight tests and verify GREEN.**

Run the command from Step 2. Expected: all preflight tests pass.

- [ ] **Step 10: Ignore outputs and document the new structure.**

Add `runs/` and `yolov5n.pt` to `.gitignore`. Add `configs/`, the launcher, both YOLOv5 copies, and the run-output directory to `PROJECT_STRUCTURE.md`; label `third_party/yolov5-v7.0-original` as immutable backup.

- [ ] **Step 11: Run the complete unit-test suite.**

```powershell
& 'D:\Conda\conda_envs\zynq-defect\python.exe' -m unittest discover -s models/tests -v
```

Expected: all tests pass.

- [ ] **Step 12: Commit configuration and preflight.**

```powershell
git add configs tools/train_yolov5n_silu.py models/tests/test_yolov5_gray_preflight.py .gitignore PROJECT_STRUCTURE.md
git commit -m "feat: configure grayscale YOLOv5n SiLU baseline"
```

---

### Task 4: Real-data preflight and final verification

**Files:**

- Modify: `models/README.md`
- Modify: `docs/plans/YOLOv5n_ZYNQ7010_训练迁移计划.md`

**Interfaces:**

- Consumes: `tools/train_yolov5n_silu.py --preflight-only`.
- Produces: documented PowerShell commands for preflight, training, resume, validation-set evaluation, and test-set evaluation.

- [ ] **Step 1: Validate the relocated dataset.**

```powershell
& 'D:\Conda\conda_envs\zynq-defect\python.exe' models/convert_coco_to_yolo_roi256.py --validate data/GC10-DET-YOLO-ROI256
```

Expected: `valid: true`, 3109 images, 3109 labels, 4251 annotations.

- [ ] **Step 2: Run the real preflight.**

```powershell
& 'D:\Conda\conda_envs\zynq-defect\python.exe' tools/train_yolov5n_silu.py --preflight-only
```

Expected JSON includes batch channel 1, first-convolution channel 1, head channels `[45, 45, 45]`, verified pretrained conversion, CUDA device, and `status: ready`.

- [ ] **Step 3: Verify the original backup remains byte-identical to its pre-change state.**

Recompute the deterministic tree hash from Task 0. Confirm the backup still contains 144 files and digest `4ce25aa77ae56fae91dcf61dcdc72a2c9ec052a32d82fc245156132eb6014698`. Confirm all source modifications are under `third_party/yolov5`, never the backup.

- [ ] **Step 4: Run all tests fresh.**

```powershell
& 'D:\Conda\conda_envs\zynq-defect\python.exe' -m unittest discover -s models/tests -v
```

Expected: zero failures and zero errors.

- [ ] **Step 5: Document operator commands.**

Add these commands to `models/README.md` and check off only the completed stage-0 items in the migration plan:

```powershell
conda activate zynq-defect
python tools/train_yolov5n_silu.py --preflight-only
python tools/train_yolov5n_silu.py
python third_party/yolov5/train.py --resume runs/yolov5n_gray256_silu_baseline/weights/last.pt
python third_party/yolov5/val.py --weights runs/yolov5n_gray256_silu_baseline/weights/best.pt --data gc10_yolo.yaml --imgsz 256 --channels 1 --task val
python third_party/yolov5/val.py --weights runs/yolov5n_gray256_silu_baseline/weights/best.pt --data gc10_yolo.yaml --imgsz 256 --channels 1 --task test
```

Do not claim stage 1 is complete until the 300-epoch run and val/test evaluation have actually finished.

- [ ] **Step 6: Run documentation/path checks.**

```powershell
rg -n "models/data|TinyDefect|model/Data" . --glob '!third_party/**' --glob '!data/**' --glob '!docs/superpowers/plans/2026-09-27-yolov5n-gray-training.md'
git diff --check
```

Expected: no legacy-path/old-model matches and no whitespace errors.

- [ ] **Step 7: Commit documentation and verification updates.**

```powershell
git add models/README.md docs/plans/YOLOv5n_ZYNQ7010_训练迁移计划.md
git commit -m "docs: add grayscale baseline training workflow"
```
