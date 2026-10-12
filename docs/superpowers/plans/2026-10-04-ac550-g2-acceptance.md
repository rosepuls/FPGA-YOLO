# AC550 G2 INT8 Acceptance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the AC550 Python comparison into a reproducible multi-image G2 acceptance report covering package identity, per-node execution coverage, P3/P4/P5 tensor error, detections, and configurable pass/fail thresholds.

**Architecture:** Keep the existing single-image `int8_inference_ac550.py` API and add an acceptance layer that selects a deterministic image list, runs the package-backed INT8 simulator against the frozen FP32 ReLU model, aggregates worst-case metrics, and writes a machine-readable report. The report will explicitly distinguish Conv nodes measured with package INT8 parameters from non-Conv topology nodes that are not yet integer-executed.

**Tech Stack:** Python 3, PyTorch, NumPy, Pillow, unittest, existing YOLOv5-ReLU vendor tree.

## Global Constraints

- Require package status `PRE_G2_STATIC_QUANTIZATION` until a later manifest-freeze step.
- Reuse package `weights_i8.bin`, `bias_i32.bin`, `quant_params.json`, `graph.json`, and `golden_input_u8.bin`; do not silently re-quantize from FP32.
- Keep the existing one-image comparison CLI compatible.
- A report may pass only when all requested metrics satisfy explicit thresholds and no unmatched detections or class mismatches occur.
- Do not claim FPGA RTL, timing, resource, QAT, or full ARM integer acceptance from Python-only evidence.

---

### Task 1: Define acceptance aggregation behavior

**Files:**
- Modify: `models/tests/test_ac550_int8_inference.py`
- Create: `models/export/ac550_g2_acceptance.py`

**Interfaces:**
- `aggregate_reports(reports, thresholds) -> dict[str, Any]`
- `select_images(image_dir, limit) -> list[Path]`
- CLI accepts `--image-dir`, `--max-images`, `--min-images`, `--max-head-mean-abs-diff`, `--max-head-rmse`, `--min-box-iou`, `--max-confidence-diff`, `--output`.

- [ ] **Step 1: Write failing tests** for deterministic image selection, aggregation of worst-case P3/P4/P5 metrics, and failure when an image has unmatched detections or package coverage is incomplete.
- [ ] **Step 2: Run the focused tests** and confirm they fail because the acceptance module does not exist.
- [ ] **Step 3: Implement the smallest aggregation module** with explicit `passed`, `failure_reasons`, `image_count`, `worst_heads`, `detection_summary`, and `node_coverage` fields.
- [ ] **Step 4: Run focused tests** and confirm they pass.

### Task 2: Connect acceptance runner to the existing package-backed inference

**Files:**
- Modify: `models/export/int8_inference_ac550.py`
- Modify: `models/export/ac550_g2_acceptance.py`
- Modify: `models/tests/test_ac550_int8_inference.py`

**Interfaces:**
- Reuse `compare_package(...)` for each selected image so single-image behavior remains the source of truth.
- Add package manifest SHA, graph node count, executor coverage, and selected input paths to the aggregate report.

- [ ] **Step 1: Add a failing CLI integration test** that runs on the existing package with one real calibration image and checks report schema and threshold failure/pass fields.
- [ ] **Step 2: Run the test** and confirm the new CLI is missing or incomplete.
- [ ] **Step 3: Implement deterministic image discovery and per-image execution** without changing existing single-image output.
- [ ] **Step 4: Implement threshold evaluation** for head errors, IoU, confidence difference, unmatched detections, class mismatches, and minimum image count.
- [ ] **Step 5: Run focused and existing inference tests** until all pass.

### Task 3: Run the real G2 acceptance and record evidence

**Files:**
- Create: `runs/ac550_yolov5n_relu_int8/g2_acceptance.json`

- [ ] **Step 1: Run the acceptance CLI** against a deterministic subset of `data/LV-YOLO-ROI256/images/train` using the existing package and frozen ReLU checkpoint.
- [ ] **Step 2: Inspect the report** for image count, package SHA, node coverage, worst-case head metrics, detection mismatches, and threshold decisions.
- [ ] **Step 3: Run Python compilation, full model/export/inference tests, and `git diff --check`.
- [ ] **Step 4: Report whether G2 passed; if not, identify the exact metric or missing integer node coverage preventing manifest freeze.

