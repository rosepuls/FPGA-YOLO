# YOLOv5n ReLU Variant Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce an AC550-friendly, isolated YOLOv5n-ReLU source variant and a runnable model contract test.

**Architecture:** The existing grayscale-capable YOLOv5 tree is copied without caches. A dedicated YAML changes the default `Conv` activation to ReLU while preserving the standard YOLOv5n topology; Detect heads remain raw linear convolutions. A subprocess-based test builds the real copied model so global activation state cannot leak from other tests.

**Tech Stack:** Python, PyTorch, the project's vendored YOLOv5 v7.0 tree, `unittest`.

## Global Constraints

- Preserve `third_party/yolov5` as the SiLU baseline.
- Create `third_party/yolov5-relu` from the current modified `third_party/yolov5` source, excluding `__pycache__` directories.
- AC550 target: keep YOLOv5n `depth_multiple: 0.33` and `width_multiple: 0.25`; do not add layers or channels.
- Deployment model contract: `ch: 1`, `nc: 4`, ordinary Conv activation `nn.ReLU()`, Detect heads linear with 27 output channels.
- Use `C:\\Study\\IC_Study\\Project\\FPGA\\Contest\\CGEDC_1\\FPGA-YOLO` as the project root and do not alter the user's existing plan-document edit.

---

### Task 1: Add the ReLU model contract test

**Files:**
- Create: `models/tests/test_yolov5_relu_variant.py`

**Interfaces:**
- Consumes: `third_party/yolov5-relu/models/yolov5n_relu_4cls_gray.yaml`
- Produces: a `unittest` test that runs the real copied model in a clean Python subprocess.

- [x] **Step 1: Write the failing test**

```python
def test_relu_variant_has_ac550_deployment_contract(self) -> None:
    result = subprocess.run([sys.executable, "-c", MODEL_PROBE], text=True, capture_output=True, check=False)
    self.assertEqual(result.returncode, 0, result.stderr)
    self.assertEqual(contract["first_conv_in_channels"], 1)
    self.assertEqual(contract["detect_output_channels"], [27, 27, 27])
```

- [x] **Step 2: Run the test to verify it fails**

Run: `D:\\Conda\\conda_envs\\zynq-defect\\python.exe -m unittest models.tests.test_yolov5_relu_variant -v`

Expected: FAIL because the ReLU source tree and model configuration do not yet exist.

### Task 2: Copy the source tree and add the fixed ReLU configuration

**Files:**
- Create: `third_party/yolov5-relu/` (copy of `third_party/yolov5/`, without `__pycache__/`)
- Create: `third_party/yolov5-relu/models/yolov5n_relu_4cls_gray.yaml`

**Interfaces:**
- Consumes: current project-local YOLOv5 source.
- Produces: a self-contained YOLOv5n model definition for 1-channel, 4-class ReLU training and deployment.

- [x] **Step 1: Copy source excluding runtime caches**

Run: `robocopy third_party\\yolov5 third_party\\yolov5-relu /E /XD __pycache__ .pytest_cache`

- [x] **Step 2: Add the model configuration**

```yaml
nc: 4
ch: 1
depth_multiple: 0.33
width_multiple: 0.25
activation: nn.ReLU()
```

Keep the source `anchors`, `backbone`, and `head` lists unchanged, including the final `Detect` node.

### Task 3: Verify the isolated model contract

**Files:**
- Test: `models/tests/test_yolov5_relu_variant.py`

- [x] **Step 1: Run the focused contract test**

Run: `D:\\Conda\\conda_envs\\zynq-defect\\python.exe -m unittest models.tests.test_yolov5_relu_variant -v`

Expected: PASS; the test observes one input channel, four classes, ReLU Conv activation, three 27-channel linear detect heads, and 32/16/8 feature-map sizes.

- [x] **Step 2: Run existing grayscale model regression tests**

Run: `D:\\Conda\\conda_envs\\zynq-defect\\python.exe -m unittest models.tests.test_yolov5_gray_model -v`

Expected: PASS; the original SiLU-oriented source remains compatible.
