# AC550 YOLOv5n-ReLU QAT Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fine-tune the existing four-class, grayscale YOLOv5n-ReLU checkpoint with AC550-equivalent fake quantization, then export a model package whose integer graph meets the G2 tensor and detection acceptance gates.

**Architecture:** Insert custom symmetric fake-quant modules into the ReLU YOLO model before optimizer construction. The modules emulate AC550's signed `[-127, 127]` INT8 activation/weight storage, per-output-channel weight scales, INT32 convolution bias/accumulation, Round-Half-To-Even requantization, and fixed per-edge activation scales. Use a staged schedule: collect ranges, train with fake quant, freeze observer/BN statistics, then fine-tune with straight-through-estimator gradients. Export must consume the frozen QAT parameters rather than recalibrating them as unrelated PTQ values.

**Tech Stack:** Python 3, PyTorch, existing `third_party/yolov5-relu`, YOLOv5 `ComputeLoss`, Pillow, unittest.

## Global Constraints

- Preserve the AC550 deployment contract: `1×256×256` grayscale input, four classes in the existing order, ReLU normal Conv, and linear 27-channel Detect convolutions.
- Activation and weight fake quantization must be symmetric signed INT8, zero point `0`, saturation `[-127,127]`, and Round-Half-To-Even.
- Normal Conv uses per-output-channel weight scale; Detect Conv remains linear and is never followed by ReLU.
- MaxPool preserves scale; Add must requantize both inputs to its declared output scale; Concat must requantize each branch to a declared shared output scale; Upsample preserves scale.
- QAT uses the existing frozen ReLU `best.pt` as its starting point and never uses the SiLU checkpoint as its deployment candidate.
- QAT validation uses `val`, final evidence uses `test`; do not select a package only from training-image G2 results.
- Keep AMP disabled during QAT because fake-quant scale observers and the target arithmetic are FP32/INT8, not FP16.
- Do not lower G2 thresholds after seeing results. If a threshold changes, document the product reason and update the manifest schema before rerunning all evidence.

---

### Task 1: Implement AC550-equivalent fake quantization primitives

**Files:**
- Create: `models/qat/ac550_fake_quant.py`
- Create: `models/tests/test_ac550_fake_quant.py`

**Interfaces:**
- `Ac550Observer(per_channel: bool, channel_axis: int | None, momentum: float)`: tracks EMA absolute maximum in FP32.
- `Ac550FakeQuant(observer, enabled: bool, observer_enabled: bool)`: returns `x + (dequantize(quantize(x)) - x).detach()`.
- `requantize_i8(x, source_scale, target_scale) -> Tensor`: uses the exact signed range and rounding rule.

- [ ] **Step 1: Write failing tests** for `-127/127` saturation, Round-Half-To-Even ties, zero-point `0`, per-output-channel scales, disabled fake quant identity, and STE gradient equal to one in range.

```python
def test_requantize_clamps_to_signed_ac550_range():
    actual = requantize_i8(torch.tensor([-100.0, 100.0]), 0.5, 0.5)
    assert torch.equal(actual, torch.tensor([-63.5, 63.5]))
```

- [ ] **Step 2: Run the focused test** and confirm it fails before the primitive exists.

- [ ] **Step 3: Implement the primitive** using `torch.round`, `clamp(-127, 127)`, FP32 EMA range updates, and the STE expression above. Reject zero, negative, NaN, and infinite scales.

- [ ] **Step 4: Run the focused tests** and confirm all pass.

### Task 2: Insert fake quant at every AC550 graph boundary

**Files:**
- Create: `models/qat/ac550_qat_graph.py`
- Modify: `third_party/yolov5-relu/models/common.py`
- Modify: `third_party/yolov5-relu/models/yolo.py`
- Create: `models/tests/test_ac550_qat_graph.py`

**Interfaces:**
- `prepare_ac550_qat(model, observer_momentum: float) -> nn.Module`
- `set_qat_phase(model, phase: Literal['collect', 'train', 'freeze']) -> None`
- `qat_metadata(model) -> dict[str, Any]`

- [ ] **Step 1: Write failing contract tests** that prepare the ReLU model and assert: every normal Conv has per-channel weight and per-tensor activation fake quant; Detect has no ReLU; each C3 residual Add has an explicit output fake quant; each Concat has branch requantization to one output scale; all three SPPF pools and both Upsamples have an explicit boundary scale.

- [ ] **Step 2: Run the tests** and confirm they fail because no QAT graph preparation exists.

- [ ] **Step 3: Implement QAT wrappers.** Normal Conv must execute `fake_quant_activation(ReLU(BN(FakeQuantWeight(Conv))))` during training, while Detect executes `fake_quant_activation(FakeQuantWeight(Conv))` with no ReLU. Use fused-QAT behavior for Conv+BN+ReLU so the learned model matches inference-time Conv+BN fusion.

- [ ] **Step 4: Implement non-Conv operators.** In `Bottleneck`, requantize residual and main paths to the Add output scale before INT32-like add/saturation. In `C3`, SPPF, and `Concat`, requantize branch tensors to the declared concat scale before concatenation. In Upsample and MaxPool, retain the producer scale and still fake quantize their output boundary.

- [ ] **Step 5: Run graph tests and a one-batch forward/backward smoke test** with `1×1×256×256` input and YOLO `ComputeLoss`; verify all parameter gradients are finite.

### Task 3: Add a QAT training entry point and deterministic schedule

**Files:**
- Create: `models/train/train_yolov5n_relu_ac550_qat.py`
- Create: `configs/hyp.ac550_qat.yaml`
- Create: `models/tests/test_ac550_qat_schedule.py`

**Interfaces:**
- CLI accepts `--weights`, `--data`, `--imgsz`, `--channels`, `--epochs`, `--batch-size`, `--device`, `--qat-observer-epochs`, `--qat-freeze-observer-epoch`, `--qat-freeze-bn-epoch`, `--project`, and `--name`.
- The entry point stores `qat_metadata.json` with phase boundaries, observer algorithm, qrange, source checkpoint SHA-256, data YAML SHA-256, and seed.

- [ ] **Step 1: Write failing schedule tests** for phase transitions: epochs `0–4` collect observer statistics, `5–14` train fake quant with moving observers, `15–39` freeze observer scales, and epochs `20–39` freeze BN running statistics.

- [ ] **Step 2: Run the tests** and confirm the schedule/CLI is missing.

- [ ] **Step 3: Implement the entry point** by loading `runs/yolov5n_gray_relu_256/weights/best.pt`, applying `prepare_ac550_qat` before optimizer construction, setting `amp=False`, and calling the vendor training loop with callbacks that invoke `set_qat_phase` at each epoch start.

- [ ] **Step 4: Use this initial GPU command** after implementation:

```powershell
$py = 'D:\Conda\conda_envs\zynq-defect\python.exe'
& $py .\models\train\train_yolov5n_relu_ac550_qat.py `
  --weights .\runs\yolov5n_gray_relu_256\weights\best.pt `
  --data .\lv_yolo.yaml `
  --imgsz 256 --channels 1 --epochs 40 --batch-size 16 --device 0 `
  --qat-observer-epochs 5 --qat-freeze-observer-epoch 15 `
  --qat-freeze-bn-epoch 20 --seed 20260930 `
  --project .\runs --name yolov5n_gray_relu_256_qat_v1
```

- [ ] **Step 5: Run an eight-image CPU smoke test** for one epoch, then run the schedule and graph tests.

### Task 4: Export QAT parameters without losing the trained quantization contract

**Files:**
- Modify: `models/export/export_yolov5n_relu_ac550.py`
- Modify: `models/export/int8_inference_ac550.py`
- Modify: `models/tests/test_yolov5n_relu_export.py`
- Create: `models/tests/test_ac550_qat_export.py`

**Interfaces:**
- `export_package(..., qat_metadata: Path | None = None)` detects a QAT checkpoint and serializes frozen observer scales plus quantized weights.
- `quant_params.json` schema becomes QAT-aware: each Conv and non-Conv node has source, output scale, qrange, rounding mode, and any required Add/Concat multiplier/right-shift.

- [ ] **Step 1: Write failing tests** proving QAT scale values are preserved by export and that exporter rejects a QAT checkpoint with missing/floating observer scales.

- [ ] **Step 2: Run the tests** and confirm the existing PTQ exporter does not meet this behavior.

- [ ] **Step 3: Implement QAT export.** Use frozen QAT scales for weight packing, bias INT32 conversion, Conv requantization, Add branch requantization, Concat branch requantization, and node Golden tensors. Never overwrite QAT scales by recalculating per-image PTQ scales during this path.

- [ ] **Step 4: Run exporter and package-hash tests**. Verify the package manifest records `quantization_method: QAT_AC550_SYMMETRIC_INT8` and source QAT checkpoint SHA-256.

### Task 5: Perform objective validation and G2 acceptance

**Files:**
- Modify: `models/export/ac550_g2_acceptance.py`
- Create: `docs/plans/ac550-yolov5n-relu-qat-验收记录.md`
- Create: `models/tests/test_ac550_qat_acceptance.py`

**Interfaces:**
- Acceptance report records FP32 baseline, QAT float-emulation, Python INT8 graph, validation/test image lists, per-node errors, P3/P4/P5, classes, boxes, confidences, and manifest SHA.

- [ ] **Step 1: Write failing tests** that reject a report when it uses training images as final evidence, omits QAT metadata, has class mismatch, unmatched detections, or has incomplete graph-node coverage.

- [ ] **Step 2: Run the tests** and confirm they fail under the current PTQ-only package.

- [ ] **Step 3: Validate the QAT checkpoint on the independent validation split.** Compare FP32 baseline and QAT float-emulation with the same labels. Suggested precommitted initial quality gates are: mAP@0.5 loss no more than `1.0` absolute percentage point; overall recall loss no more than `1.0` point; any class recall loss no more than `2.0` points.

- [ ] **Step 4: Export a fresh immutable package** such as `runs/ac550_yolov5n_relu_qat_int8_v1`; run the Python INT8 graph across the validation set, then once across the test set. Keep initial hardware gates: every graph node covered; no class mismatch or unmatched detection; P3/P4/P5 mean absolute error `≤0.25`; RMSE `≤0.35`; matched-box IoU `≥0.90`; confidence difference `≤0.05`.

- [ ] **Step 5: Freeze only a passing package.** Write package SHA-256, QAT checkpoint SHA-256, dataset/YAML SHA-256, commands, all gate results, and known limitations into the acceptance record. A failure triggers either threshold-independent QAT tuning or a data/label review; it never triggers silent threshold relaxation.

## Self-Review

- The plan covers target arithmetic, model graph boundaries, training schedule, export, and independent acceptance.
- No code task assumes PyTorch's default unsigned quantizer; the AC550 signed contract is explicit.
- QAT packaging is tied to the model's frozen observer scales, preventing the current PTQ exporter from discarding learned quantization behavior.
- FPGA timing/resource validation remains outside this QAT plan and starts only after G2 passes.

