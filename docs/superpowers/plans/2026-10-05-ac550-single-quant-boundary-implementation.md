# AC550 Single Quantization Boundary Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate redundant activation fake-quant operations from the AC550 YOLOv5n-ReLU QAT graph while preserving required INT8 storage boundaries.

**Architecture:** Normal Conv and Detect emit a single quantized output edge. Sequential consumers, MaxPool, and nearest Upsample consume that edge without a second quantization. Add and Concat consume already quantized branches and quantize only their result. Training checkpoints retain `best_qat.pt` only after observers and BN statistics are frozen.

**Tech Stack:** Python 3, PyTorch, vendored YOLOv5-ReLU, unittest.

## Global Constraints

- Keep 1×256×256 grayscale input, four classes, ReLU normal Conv, and linear Detect heads.
- Use symmetric signed INT8 range `[-127, 127]`, zero point 0, and Round-Half-To-Even.
- Retain Conv+BN folded-weight QAT and disable AMP.
- Do not export a QAT package until validation mAP@0.5 loss is at most 1.0 percentage point from the FP32 baseline.
- Work in the existing user-authorized workspace; do not commit unrelated dirty files.

---

### Task 1: Make activation quantization edge-owned

**Files:**
- Modify: `models/qat/ac550_qat_graph.py`
- Modify: `models/tests/test_ac550_qat_graph.py`

**Interfaces:**
- `Ac550QatConv.quantize_input: bool` is true only for the first model Conv.
- `Ac550QatDetectConv`, `Ac550QatBottleneck`, `Ac550QatC3`, `Ac550QatSPPF`, `Ac550QatConcat`, and `Ac550QatUpsample` consume predecessor edges without a redundant fake-quant.

- [ ] **Step 1: Write failing graph tests.** Assert the first Conv observes its input, a later Conv input observer remains untouched after a forward, and a two-branch Concat only updates its output observer.
- [ ] **Step 2: Run focused tests.**

```powershell
& $py -m unittest models.tests.test_ac550_qat_graph.Ac550QatGraphTests.test_only_first_conv_quantizes_model_input -v
```

- [ ] **Step 3: Implement edge ownership.** Gate normal Conv input fake-quant on `quantize_input`; set it only for `model.0` in `prepare_ac550_qat`. Remove repeated branch, pool, upsample, sequential input, and Detect-input fake-quant calls from the relevant forwards.
- [ ] **Step 4: Run graph and backward tests.**

```powershell
& $py -m unittest models.tests.test_ac550_qat_graph -v
```

### Task 2: Preserve explicit merge semantics and frozen candidate selection

**Files:**
- Modify: `models/qat/ac550_qat_graph.py`
- Modify: `models/train/train_yolov5n_relu_ac550_qat.py`
- Modify: `models/tests/test_ac550_qat_graph.py`
- Modify: `models/tests/test_ac550_qat_schedule.py`

**Interfaces:**
- `Ac550QatBottleneck.add_output_fake_quant` and `Ac550QatConcat.concat_output_fake_quant` are the sole result quantizers.
- `is_deployment_candidate_epoch(epoch, freeze_observer_epoch, freeze_bn_epoch) -> bool` controls `weights/best_qat.pt` creation.

- [ ] **Step 1: Write failing tests** proving residual Add and Concat update only their result observer and epochs before 20 never qualify.
- [ ] **Step 2: Run focused tests** and verify they fail on redundant observer updates.
- [ ] **Step 3: Implement the minimal forwards** that call only the output fake-quant after add/cat. Retain initial metadata creation and frozen-stage `best_qat.pt` selection.
- [ ] **Step 4: Run schedule and graph tests.**

### Task 3: Verify integration and publish re-training command

**Files:**
- Modify: `models/tests/test_ac550_qat_graph.py`
- Modify: `docs/superpowers/specs/2026-10-05-ac550-single-quant-boundary-design.md`

- [ ] **Step 1: Run all QAT unit tests and Python compilation.**

```powershell
& $py -m unittest models.tests.test_ac550_fake_quant models.tests.test_ac550_qat_graph models.tests.test_ac550_qat_schedule -v
& $py -m py_compile models\qat\ac550_qat_graph.py models\train\train_yolov5n_relu_ac550_qat.py
```

- [ ] **Step 2: Record the re-training candidate and gate.** Use a new run name, review `weights/best_qat.pt`, compare validation mAP with `0.61957`, and do not export when mAP@0.5 loss exceeds 0.01.
