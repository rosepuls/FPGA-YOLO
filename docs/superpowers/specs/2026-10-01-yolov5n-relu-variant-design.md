# YOLOv5n ReLU Variant Design

## Goal

Create an isolated YOLOv5n-ReLU training and deployment source tree for the AC550 industrial-defect detector, while preserving the existing SiLU baseline unchanged.

## Scope

- Copy `third_party/yolov5` to `third_party/yolov5-relu`, excluding Python cache directories.
- Add one fixed-deployment model configuration: `models/yolov5n_relu_4cls_gray.yaml`.
- Add a project-level contract test that builds the copied model and checks its observable deployment interface.
- Do not alter `third_party/yolov5`, the trained SiLU checkpoint, data splits, or model topology.

## Architecture

The copied tree retains the project's existing grayscale dataloader and checkpoint-loading fixes. The ReLU variant is selected by the top-level YAML value `activation: nn.ReLU()`, which the vendored `parse_model()` already applies to all `Conv` modules. Standard YOLOv5n uses `C3` and `SPPF`, whose convolutions inherit that setting.

`Detect` remains unchanged: its three final layers are bare `nn.Conv2d` modules and therefore emit unactivated logits. For four classes and three anchors, each head has `3 * (5 + 4) = 27` channels. At `256 x 256` input, the three training-mode heads are `32 x 32`, `16 x 16`, and `8 x 8`.

## AC550 Resource Constraints

- Target device is the Xiaomeige AC550 Cyclone V SoC platform.
- Keep YOLOv5n depth and width multipliers at `0.33` and `0.25`; do not enlarge the network.
- Use one grayscale input channel and four classes: `ca_shang`, `zang_wu`, `zhe_zhou`, `zhen_kong`.
- Use ReLU only in ordinary `Conv` blocks, so FPGA deployment needs no SiLU approximation hardware.
- Keep SPPF MaxPool and graph topology intact; the ARM/FPGA partition remains defined by the AC550 deployment plan.

## Acceptance Criteria

- `third_party/yolov5-relu` exists and the original `third_party/yolov5` has no modifications.
- The ReLU YAML declares `ch: 1`, `nc: 4`, and `activation: nn.ReLU()`.
- The contract test confirms a model has a one-channel first convolution, four classes, ReLU ordinary convolution activations, three linear 27-channel detect heads, and the expected three head shapes for a `1 x 1 x 256 x 256` input.
