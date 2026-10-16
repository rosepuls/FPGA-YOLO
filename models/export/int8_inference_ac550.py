"""Run a reproducible AC550 INT8-vs-FP32 comparison on one input image.

The runner consumes the exported INT8 package rather than re-quantizing weights
from the checkpoint. Convolution nodes execute with package INT8 weights and
INT32 bias; YOLO routing, concatenation, residual adds and nearest upsample use
the frozen ReLU topology. The report deliberately records the measured error
instead of silently declaring deployment acceptance.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable
from collections import Counter

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]
if str(THIS_FILE.parent) not in sys.path:
    sys.path.insert(0, str(THIS_FILE.parent))

from export_yolov5n_relu_ac550 import _load_frozen_model


HEAD_NAMES = ("P3", "P4", "P5")
CONV_EXECUTORS = {"FPGA_CONV_RELU", "FPGA_CONV_LINEAR"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read JSON package file: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"package JSON must contain an object: {path}")
    return value


def _resolve_module(root: nn.Module, dotted_name: str) -> tuple[nn.Module, str]:
    parts = dotted_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


@dataclass(frozen=True)
class ConvPackageSpec:
    source_module: str
    weight: torch.Tensor
    bias: torch.Tensor
    weight_scales: torch.Tensor
    input_scale: float
    output_scale: float
    stride: tuple[int, int]
    padding: tuple[int, int]
    dilation: tuple[int, int]
    groups: int


@dataclass
class ComparisonContext:
    package: Path
    relu_root: Path
    manifest: dict[str, Any]
    graph: dict[str, Any]
    reference_model: nn.Module
    quantized_model: nn.Module
    node_coverage: dict[str, Any]
    boundary_quantizer: "IntegerBoundaryQuantizer"
    reference_node_capture: "_NodeOutputCapture"
    quantized_node_capture: "_NodeOutputCapture"


def _read_binary_tensor(path: Path, *, offset: int, size: int, dtype: np.dtype[Any], shape: list[int]) -> torch.Tensor:
    with path.open("rb") as stream:
        stream.seek(offset)
        raw = stream.read(size)
    expected = int(np.prod(shape)) * np.dtype(dtype).itemsize
    if len(raw) != expected:
        raise ValueError(f"binary tensor size mismatch in {path.name}: expected {expected}, got {len(raw)}")
    return torch.from_numpy(np.frombuffer(raw, dtype=dtype).copy().reshape(shape))


def _fake_quantize_dequantize(inputs: torch.Tensor, scale: float) -> torch.Tensor:
    """Apply signed symmetric INT8 rounding/saturation and return a float simulation tensor."""
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"quantization scale must be positive and finite, got {scale}")
    return torch.round(inputs / scale).clamp(-127, 127) * scale


def _map_tensors(value: Any, function: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return function(value)
    if isinstance(value, list):
        return [_map_tensors(item, function) for item in value]
    if isinstance(value, tuple):
        return tuple(_map_tensors(item, function) for item in value)
    return value


class IntegerBoundaryQuantizer:
    """Fake-quantize non-Conv module boundaries using exported Golden scales."""

    def __init__(
        self,
        model: nn.Module,
        vectors: dict[str, Any],
        node_scales: dict[str, Any] | None = None,
    ) -> None:
        self._handles: list[Any] = []
        self._entries: dict[str, list[tuple[float, float]]] = {}
        self._occurrences: Counter[str] = Counter()
        self._active: dict[str, tuple[float, float]] = {}
        self._unresolved: list[str] = []
        node_scales = node_scales or {}
        for entry in vectors.get("vectors", []):
            name = str(entry["source_module"])
            base = name.split("#", maxsplit=1)[0]
            if base.endswith(".conv"):
                continue
            exported = node_scales.get(base, {})
            input_scale = float(
                exported.get("input_quantization", {}).get("scale", entry["input"]["quantization"]["scale"])
            )
            output_scale = float(
                exported.get("output_quantization", {}).get("scale", entry["output"]["quantization"]["scale"])
            )
            self._entries.setdefault(base, []).append((input_scale, output_scale))
        hooked = 0
        for base in self._entries:
            try:
                parent, attribute = _resolve_module(model, base)
                module = getattr(parent, attribute)
            except (AttributeError, TypeError):
                self._unresolved.append(base)
                continue
            if isinstance(module, nn.Conv2d) or type(module).__name__ == "QuantizedConv2d":
                continue
            self._handles.append(module.register_forward_pre_hook(self._pre_hook(base)))
            self._handles.append(module.register_forward_hook(self._post_hook(base)))
            hooked += len(self._entries[base])
        self.coverage = {
            "golden_non_conv_boundaries": sum(len(entries) for entries in self._entries.values()),
            "hooked_non_conv_boundaries": hooked,
            "hooked_modules": sorted(self._entries.keys()),
            "unresolved_non_conv_modules": sorted(self._unresolved),
        }

    def begin_forward(self) -> None:
        self._occurrences.clear()
        self._active.clear()

    def _next_spec(self, base: str) -> tuple[float, float]:
        index = self._occurrences[base]
        self._occurrences[base] += 1
        entries = self._entries[base]
        if index >= len(entries):
            raise ValueError(f"module executed more times than Golden vectors: {base}")
        spec = entries[index]
        self._active[base] = spec
        return spec

    def _pre_hook(self, base: str):
        def quantize_inputs(_module: nn.Module, inputs: tuple[Any, ...]) -> tuple[Any, ...]:
            input_scale, _ = self._next_spec(base)
            # Concat's input list can contain branches with different producer scales;
            # its consumer Conv will requantize the concatenated tensor. Preserve the
            # list here and quantize the operation output instead.
            if type(_module).__name__ in {"Concat", "Ac550QatConcat"}:
                return inputs
            return _map_tensors(inputs, lambda tensor: _fake_quantize_dequantize(tensor, input_scale))

        return quantize_inputs

    def _post_hook(self, base: str):
        def quantize_output(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> Any:
            _, output_scale = self._active[base]
            return _map_tensors(output, lambda tensor: _fake_quantize_dequantize(tensor, output_scale))

        return quantize_output

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def _load_conv_specs(package: Path, quantization: dict[str, Any], model: nn.Module) -> dict[str, ConvPackageSpec]:
    specs: dict[str, ConvPackageSpec] = {}
    for entry in quantization.get("convolutions", []):
        source = entry["source_module"]
        parent, attribute = _resolve_module(model, source)
        original = getattr(parent, attribute)
        original_conv = getattr(original, "conv", original)
        if not isinstance(original_conv, nn.Conv2d):
            raise ValueError(f"package source is not a Conv2d in checkpoint: {source}")
        weight_entry = entry["weight"]
        bias_entry = entry["bias"]
        weight_path = package / weight_entry["file"]
        bias_path = package / bias_entry["file"]
        weight = _read_binary_tensor(
            weight_path,
            offset=int(weight_entry["offset"]),
            size=int(weight_entry["bytes"]),
            dtype=np.dtype("int8"),
            shape=[int(value) for value in weight_entry["shape"]],
        )
        bias = _read_binary_tensor(
            bias_path,
            offset=int(bias_entry["offset"]),
            size=int(bias_entry["bytes"]),
            dtype=np.dtype("<i4"),
            shape=[int(value) for value in bias_entry["shape"]],
        )
        scales = torch.tensor(weight_entry["quantization"]["scales"], dtype=torch.float32)
        specs[source] = ConvPackageSpec(
            source_module=source,
            weight=weight,
            bias=bias,
            weight_scales=scales,
            input_scale=float(entry["input_quantization"]["scale"]),
            output_scale=float(entry["output_quantization"]["scale"]),
            stride=tuple(int(value) for value in original_conv.stride),
            padding=tuple(int(value) for value in original_conv.padding),
            dilation=tuple(int(value) for value in original_conv.dilation),
            groups=int(original_conv.groups),
        )
    if not specs:
        raise ValueError("quantization package contains no convolution entries")
    return specs


class QuantizedConv2d(nn.Module):
    """A deterministic signed-INT8 convolution followed by dequantization."""

    def __init__(self, spec: ConvPackageSpec) -> None:
        super().__init__()
        self.register_buffer("weight_i8", spec.weight)
        self.register_buffer("bias_i32", spec.bias)
        self.register_buffer("weight_scales", spec.weight_scales)
        self.input_scale = spec.input_scale
        self.output_scale = spec.output_scale
        self.stride = spec.stride
        self.padding = spec.padding
        self.dilation = spec.dilation
        self.groups = spec.groups

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        quantized_input = torch.round(inputs / self.input_scale).clamp(-127, 127)
        accumulator = F.conv2d(
            quantized_input.float(),
            self.weight_i8.float(),
            self.bias_i32.float(),
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        multiplier = (self.input_scale * self.weight_scales / self.output_scale).view(1, -1, 1, 1)
        quantized_output = torch.round(accumulator * multiplier).clamp(-127, 127)
        return quantized_output * self.output_scale


def _patch_model_with_package(model: nn.Module, specs: dict[str, ConvPackageSpec]) -> nn.Module:
    for source, spec in specs.items():
        parent, attribute = _resolve_module(model, source)
        setattr(parent, attribute, QuantizedConv2d(spec))
    return model


def _node_coverage(
    graph: dict[str, Any], specs: dict[str, ConvPackageSpec], boundary: IntegerBoundaryQuantizer
) -> dict[str, Any]:
    nodes = graph.get("nodes", [])
    executor_counts: dict[str, int] = {}
    missing_conv_sources: list[str] = []
    missing_non_conv_nodes: list[str] = []
    missing_non_conv_executors: list[str] = []
    for node in nodes:
        executor = str(node.get("executor", "UNKNOWN"))
        executor_counts[executor] = executor_counts.get(executor, 0) + 1
        source_module = str(node.get("source_module", node.get("id", "<unknown>")))
        package_source = source_module if source_module in specs else f"{source_module}.conv"
        if executor in CONV_EXECUTORS and package_source not in specs:
            missing_conv_sources.append(source_module)
    hooked_modules = set(boundary.coverage["hooked_modules"])
    supported_non_conv: set[str] = set()
    for node in nodes:
        executor = str(node.get("executor", "UNKNOWN"))
        source_module = str(node.get("source_module", node.get("id", "<unknown>")))
        if executor not in CONV_EXECUTORS and source_module in hooked_modules:
            supported_non_conv.add(executor)
        elif executor not in CONV_EXECUTORS:
            missing_non_conv_nodes.append(str(node.get("id", source_module)))
            missing_non_conv_executors.append(executor)
    unquantized_executors = sorted(set(missing_non_conv_executors))
    return {
        "graph_node_count": len(nodes),
        "package_conv_nodes": len(specs),
        "measured_conv_nodes": len(specs) - len(missing_conv_sources),
        "missing_conv_sources": missing_conv_sources,
        "executor_counts": executor_counts,
        "unquantized_executors": unquantized_executors,
        "boundary_simulated_executors": sorted(supported_non_conv),
        "missing_non_conv_nodes": missing_non_conv_nodes,
        "integer_simulation_complete": not missing_conv_sources and not missing_non_conv_nodes,
        "boundary": boundary.coverage,
    }


class _HeadCapture:
    def __init__(self, model: nn.Module) -> None:
        self.outputs: dict[int, torch.Tensor] = {}
        self.handles = [
            module.register_forward_hook(self._hook(index))
            for index, module in enumerate(model.model[-1].m)
        ]

    def _hook(self, index: int):
        def capture(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            if not isinstance(output, torch.Tensor):
                raise ValueError(f"Detect head {index} did not produce a tensor")
            self.outputs[index] = output.detach().cpu()

        return capture

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def _load_input(package: Path, image: Path | None) -> tuple[torch.Tensor, str]:
    if image is not None:
        with Image.open(image) as source:
            source.load()
            if source.mode != "L" or source.size != (256, 256):
                raise ValueError(f"input image must be L-mode 256x256: {image}")
            pixels = np.asarray(source, dtype=np.uint8).copy()
        origin = str(image)
    else:
        path = package / "golden_input_u8.bin"
        raw = path.read_bytes()
        if len(raw) != 256 * 256:
            raise ValueError(f"golden_input_u8.bin must contain 65536 bytes, got {len(raw)}")
        pixels = np.frombuffer(raw, dtype=np.uint8).copy().reshape(256, 256)
        origin = str(path)
    return torch.from_numpy(pixels.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0), origin


def _run_with_heads(
    model: nn.Module,
    inputs: torch.Tensor,
    boundary_quantizer: IntegerBoundaryQuantizer | None = None,
    node_capture: _NodeOutputCapture | None = None,
) -> tuple[torch.Tensor, dict[int, torch.Tensor], dict[str, torch.Tensor]]:
    capture = _HeadCapture(model)
    if boundary_quantizer is not None:
        boundary_quantizer.begin_forward()
    if node_capture is not None:
        node_capture.begin_forward()
    try:
        with torch.no_grad():
            output = model(inputs)
    finally:
        capture.close()
    prediction = output[0] if isinstance(output, (tuple, list)) else output
    return prediction.detach().cpu(), capture.outputs, node_capture.outputs if node_capture is not None else {}


def _head_report(reference: torch.Tensor, quantized: torch.Tensor) -> dict[str, Any]:
    if reference.shape != quantized.shape:
        raise ValueError(f"head shape mismatch: {tuple(reference.shape)} vs {tuple(quantized.shape)}")
    difference = (reference - quantized).abs()
    reference_float = reference.float()
    return {
        "shape": list(reference.shape),
        "max_abs_diff": float(difference.max()),
        "mean_abs_diff": float(difference.mean()),
        "rmse": float(torch.sqrt(torch.mean((reference - quantized) ** 2))),
        "reference_min": float(reference_float.min()),
        "reference_max": float(reference_float.max()),
        "quantized_min": float(quantized.min()),
        "quantized_max": float(quantized.max()),
    }


def _node_report(name: str, reference: torch.Tensor, quantized: torch.Tensor) -> dict[str, Any]:
    if reference.shape != quantized.shape:
        raise ValueError(f"node {name} shape mismatch: {tuple(reference.shape)} vs {tuple(quantized.shape)}")
    difference = (reference - quantized).abs()
    return {
        "shape": list(reference.shape),
        "max_abs_diff": float(difference.max()),
        "mean_abs_diff": float(difference.mean()),
        "rmse": float(torch.sqrt(torch.mean((reference - quantized) ** 2))),
        "reference_min": float(reference.min()),
        "reference_max": float(reference.max()),
        "quantized_min": float(quantized.min()),
        "quantized_max": float(quantized.max()),
    }


class _NodeOutputCapture:
    def __init__(self, model: nn.Module, names: Iterable[str]) -> None:
        self.outputs: dict[str, torch.Tensor] = {}
        self._handles: list[Any] = []
        self._names = sorted(set(names))
        self._occurrences: Counter[str] = Counter()
        for name in self._names:
            base = name.split("#", maxsplit=1)[0]
            try:
                parent, attribute = _resolve_module(model, base)
                module = getattr(parent, attribute)
            except (AttributeError, TypeError):
                continue
            self._handles.append(module.register_forward_hook(self._hook(base)))

    def begin_forward(self) -> None:
        self.outputs.clear()
        self._occurrences.clear()

    def _hook(self, base: str):
        def capture(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            if not isinstance(output, torch.Tensor):
                return
            occurrence = self._occurrences[base]
            self._occurrences[base] += 1
            name = base if occurrence == 0 else f"{base}#{occurrence + 1}"
            self.outputs[name] = output.detach().cpu()

        return capture

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def _box_iou(left: Iterable[float], right: Iterable[float]) -> float:
    x1, y1, x2, y2 = [float(value) for value in left]
    a1, b1, a2, b2 = [float(value) for value in right]
    intersection = max(0.0, min(x2, a2) - max(x1, a1)) * max(0.0, min(y2, b2) - max(y1, b1))
    area_left = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_right = max(0.0, a2 - a1) * max(0.0, b2 - b1)
    union = area_left + area_right - intersection
    return intersection / union if union else 0.0


def compare_detections(
    reference: list[list[float]],
    quantized: list[list[float]],
    *,
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """Greedily match detections by geometry, retaining class and confidence differences."""
    used: set[int] = set()
    matches: list[dict[str, Any]] = []
    unmatched_reference: list[int] = []
    for reference_index, reference_detection in enumerate(reference):
        candidates = [
            (index, _box_iou(reference_detection[:4], detection[:4]))
            for index, detection in enumerate(quantized)
            if index not in used
        ]
        if not candidates:
            unmatched_reference.append(reference_index)
            continue
        quantized_index, iou = max(candidates, key=lambda item: item[1])
        if iou < iou_threshold:
            unmatched_reference.append(reference_index)
            continue
        used.add(quantized_index)
        quantized_detection = quantized[quantized_index]
        matches.append(
            {
                "reference_index": reference_index,
                "quantized_index": quantized_index,
                "box_iou": iou,
                "box_abs_max_diff": max(
                    abs(float(reference_detection[offset]) - float(quantized_detection[offset]))
                    for offset in range(4)
                ),
                "confidence_abs_diff": abs(float(reference_detection[4]) - float(quantized_detection[4])),
                "reference_class": int(reference_detection[5]),
                "quantized_class": int(quantized_detection[5]),
                "class_match": int(reference_detection[5]) == int(quantized_detection[5]),
            }
        )
    return {
        "reference_count": len(reference),
        "quantized_count": len(quantized),
        "matched": len(matches),
        "matches": matches,
        "unmatched_reference": unmatched_reference,
        "unmatched_quantized": [index for index in range(len(quantized)) if index not in used],
    }


def _detections(prediction: torch.Tensor, relu_root: Path, conf_threshold: float, iou_threshold: float) -> list[list[float]]:
    if str(relu_root) not in sys.path:
        sys.path.insert(0, str(relu_root))
    from utils.general import non_max_suppression

    result = non_max_suppression(
        prediction,
        conf_thres=conf_threshold,
        iou_thres=iou_threshold,
        multi_label=False,
        max_det=300,
    )[0]
    return [[float(value) for value in row.tolist()] for row in result]


def load_comparison_context(*, package: Path, weights: Path, relu_root: Path) -> ComparisonContext:
    package = package.resolve()
    weights = weights.resolve()
    relu_root = relu_root.resolve()
    manifest = _load_json(package / "manifest.json")
    graph = _load_json(package / "graph.json")
    quantization = _load_json(package / "quant_params.json")
    if manifest.get("status") != "PRE_G2_STATIC_QUANTIZATION":
        raise ValueError(f"unsupported package status: {manifest.get('status')}")
    reference_model = _load_frozen_model(weights, relu_root)
    quantized_model = _load_frozen_model(weights, relu_root)
    specs = _load_conv_specs(package, quantization, quantized_model)
    _patch_model_with_package(quantized_model, specs)
    vectors = _load_json(package / "golden_vectors.json")
    boundary_quantizer = IntegerBoundaryQuantizer(quantized_model, vectors, quantization.get("nodes"))
    node_names = list(specs) + list(boundary_quantizer.coverage["hooked_modules"])
    return ComparisonContext(
        package=package,
        relu_root=relu_root,
        manifest=manifest,
        graph=graph,
        reference_model=reference_model,
        quantized_model=quantized_model,
        node_coverage=_node_coverage(graph, specs, boundary_quantizer),
        boundary_quantizer=boundary_quantizer,
        reference_node_capture=_NodeOutputCapture(reference_model, node_names),
        quantized_node_capture=_NodeOutputCapture(quantized_model, node_names),
    )


def compare_context(
    context: ComparisonContext,
    *,
    inputs: torch.Tensor,
    input_origin: str,
    conf_threshold: float,
    iou_threshold: float,
) -> dict[str, Any]:
    reference_prediction, reference_heads, reference_nodes = _run_with_heads(
        context.reference_model,
        inputs,
        node_capture=context.reference_node_capture,
    )
    quantized_prediction, quantized_heads, quantized_nodes = _run_with_heads(
        context.quantized_model,
        inputs,
        context.boundary_quantizer,
        context.quantized_node_capture,
    )
    heads: dict[str, Any] = {}
    for index, name in enumerate(HEAD_NAMES):
        heads[name] = _head_report(reference_heads[index], quantized_heads[index])
    nodes: dict[str, Any] = {}
    for name in sorted(set(reference_nodes) & set(quantized_nodes)):
        nodes[name] = _node_report(name, reference_nodes[name], quantized_nodes[name])

    reference_detections = _detections(reference_prediction, context.relu_root, conf_threshold, iou_threshold)
    quantized_detections = _detections(quantized_prediction, context.relu_root, conf_threshold, iou_threshold)
    detection_comparison = compare_detections(
        reference_detections,
        quantized_detections,
        iou_threshold=iou_threshold,
    )
    return {
        "schema_version": 1,
        "package": str(context.package),
        "package_manifest_sha256": _sha256(context.package / "manifest.json"),
        "package_status": context.manifest["status"],
        "input": {"source": input_origin, "shape": list(inputs.shape)},
        "graph": {
            "node_count": len(context.graph.get("nodes", [])),
            "node_coverage": context.node_coverage,
        },
        "heads": heads,
        "nodes": nodes,
        "detections": {
            "reference": reference_detections,
            "quantized": quantized_detections,
            "comparison": detection_comparison,
        },
        "thresholds": {"confidence": conf_threshold, "nms_iou": iou_threshold},
        "notes": [
            "Convolution nodes use package INT8 weights, INT32 bias and calibrated scales.",
            "Non-convolution graph nodes retain the frozen topology and are not yet integer-executed by this simulator.",
            "This report does not approve QAT, FPGA timing, resource constraints, or manifest freeze.",
        ],
    }


def compare_package(
    *,
    package: Path,
    weights: Path,
    relu_root: Path,
    image: Path | None = None,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
) -> dict[str, Any]:
    context = load_comparison_context(package=package, weights=weights, relu_root=relu_root)
    inputs, input_origin = _load_input(package, image)
    return compare_context(
        context,
        inputs=inputs,
        input_origin=input_origin,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare AC550 package INT8 inference with FP32 ReLU inference.")
    parser.add_argument("--package", type=Path, required=True, help="runs/ac550_yolov5n_relu_int8")
    parser.add_argument("--weights", type=Path, required=True, help="frozen FP32 ReLU best.pt")
    parser.add_argument("--relu-root", type=Path, required=True, help="third_party/yolov5-relu")
    parser.add_argument("--image", type=Path, help="optional L-mode 256x256 image; defaults to package golden input")
    parser.add_argument("--conf-thres", type=float, default=0.25)
    parser.add_argument("--iou-thres", type=float, default=0.45)
    parser.add_argument("--output", type=Path, help="JSON report; defaults to <package>/int8_compare.json")
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    report = compare_package(
        package=arguments.package.resolve(),
        weights=arguments.weights.resolve(),
        relu_root=arguments.relu_root.resolve(),
        image=arguments.image.resolve() if arguments.image else None,
        conf_threshold=arguments.conf_thres,
        iou_threshold=arguments.iou_thres,
    )
    output = (arguments.output or arguments.package / "int8_compare.json").resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"G2 comparison report: {output}")
    print(f"P3 mean_abs_diff: {report['heads']['P3']['mean_abs_diff']:.6f}")
    print(f"P4 mean_abs_diff: {report['heads']['P4']['mean_abs_diff']:.6f}")
    print(f"P5 mean_abs_diff: {report['heads']['P5']['mean_abs_diff']:.6f}")
    print(f"FP32 detections: {len(report['detections']['reference'])}")
    print(f"INT8 detections: {len(report['detections']['quantized'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
