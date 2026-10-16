"""Export the frozen YOLOv5n-ReLU checkpoint as an AC550 INT8 model package.

The package is intentionally self-describing: FPGA receives only supported
operators and binary tensors, while HPS can read the graph and quantization
metadata without relying on Python or a PyTorch checkpoint at runtime.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

import numpy as np
from PIL import Image
import torch
import torch.nn as nn

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]
if str(THIS_FILE.parent) not in sys.path:
    sys.path.insert(0, str(THIS_FILE.parent))

from quantize_yolov5n_relu_ac550 import (
    QuantizationError,
    build_execution_graph,
    quantize_per_output_channel,
    quantize_symmetric,
)


CLASS_NAMES = ["ca_shang", "zang_wu", "zhe_zhou", "zhen_kong"]
INPUT_SHAPE = [1, 1, 256, 256]


@dataclass(frozen=True)
class CapturedTensor:
    name: str
    input: torch.Tensor
    output: torch.Tensor


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _tensor_from_image(path: Path) -> tuple[np.ndarray, torch.Tensor]:
    with Image.open(path) as image:
        image.load()
        if image.mode != "L" or image.size != (256, 256):
            raise QuantizationError(f"calibration image must be L-mode 256x256: {path}")
        pixels = np.asarray(image, dtype=np.uint8).copy()
    model_input = torch.from_numpy(pixels.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
    return pixels, model_input


def _calibration_images(directory: Path, maximum: int) -> list[Path]:
    if maximum <= 0:
        raise QuantizationError("calibration-images must be positive")
    images = sorted(path for path in directory.rglob("*") if path.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if not images:
        raise QuantizationError(f"no calibration images found: {directory}")
    return images[:maximum]


def _first_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            try:
                return _first_tensor(item)
            except QuantizationError:
                pass
    raise QuantizationError("captured operation did not contain a tensor")


def _is_capture_target(module: nn.Module) -> bool:
    return isinstance(module, (nn.Conv2d, nn.MaxPool2d, nn.Upsample)) or type(module).__name__ in {
        "Concat",
        "Bottleneck",
        "C3",
        "SPPF",
        "Ac550QatConv",
        "Ac550QatDetectConv",
        "Ac550QatConcat",
        "Ac550QatBottleneck",
        "Ac550QatC3",
        "Ac550QatSPPF",
        "Ac550QatUpsample",
    }


class ActivationCapture:
    """Capture concrete float boundaries once, then serialize their INT8 counterparts."""

    def __init__(self, model: nn.Module) -> None:
        self._records: list[CapturedTensor] = []
        self._handles: list[Any] = []
        self._forward_occurrences: Counter[str] = Counter()
        self._capture_golden = False
        self._statistics: dict[str, dict[str, float]] = {}
        for name, module in model.named_modules():
            if _is_capture_target(module):
                class_name = type(module).__name__
                if class_name == "Ac550QatConv":
                    name = f"{name}.conv"
                self._handles.append(module.register_forward_hook(self._hook(name)))

    @property
    def records(self) -> list[CapturedTensor]:
        return self._records

    @property
    def statistics(self) -> dict[str, dict[str, float]]:
        return self._statistics

    def begin_forward(self, *, capture_golden: bool) -> None:
        self._forward_occurrences = Counter()
        self._capture_golden = capture_golden

    def _hook(self, name: str):
        def capture(_module: nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
            self._forward_occurrences[name] += 1
            suffix = "" if self._forward_occurrences[name] == 1 else f"#{self._forward_occurrences[name]}"
            record_name = f"{name}{suffix}"
            input_tensor = _first_tensor(inputs).detach()
            output_tensor = _first_tensor(output).detach()
            statistics = self._statistics.setdefault(record_name, {"input_max_abs": 0.0, "output_max_abs": 0.0})
            statistics["input_max_abs"] = max(statistics["input_max_abs"], float(input_tensor.abs().max().cpu()))
            statistics["output_max_abs"] = max(statistics["output_max_abs"], float(output_tensor.abs().max().cpu()))
            if self._capture_golden:
                self._records.append(
                    CapturedTensor(
                        name=record_name,
                        input=input_tensor.cpu(),
                        output=output_tensor.cpu(),
                    )
                )

        return capture

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def _load_frozen_model(weights: Path, relu_root: Path) -> nn.Module:
    if not weights.is_file():
        raise QuantizationError(f"checkpoint does not exist: {weights}")
    if not (relu_root / "models" / "yolo.py").is_file():
        raise QuantizationError(f"ReLU YOLOv5 source tree is incomplete: {relu_root}")
    sys.path.insert(0, str(relu_root))
    checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
    model = checkpoint.get("model") if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(model, nn.Module):
        raise QuantizationError("checkpoint does not contain a PyTorch model")
    model.float().eval().fuse()
    # QAT checkpoints retain serializable wrappers. Disable their fake-quant
    # forward path for export calibration and fold each wrapper's BN exactly
    # once, matching the fused FP32 exporter contract.
    from utils.torch_utils import fuse_conv_and_bn

    for module in model.modules():
        if type(module).__name__ == "Ac550FakeQuant":
            module.set_enabled(False)
        if type(module).__name__ == "Ac550QatConv" and isinstance(module.bn, nn.BatchNorm2d):
            module.conv = fuse_conv_and_bn(module.conv, module.bn)
            module.bn = nn.Identity()
    detect = model.model[-1]
    detect_channels = [getattr(getattr(head, "conv", head), "out_channels", None) for head in detect.m]
    if getattr(detect, "nc", None) != 4 or detect_channels != [27, 27, 27]:
        raise QuantizationError("checkpoint does not satisfy the four-class, 27-channel Detect contract")
    return model


def _capture_calibration(
    model: nn.Module, image_paths: list[Path]
) -> tuple[np.ndarray, list[CapturedTensor], dict[str, dict[str, float]]]:
    first_pixels: np.ndarray | None = None
    capture = ActivationCapture(model)
    try:
        with torch.no_grad():
            for index, image_path in enumerate(image_paths):
                pixels, model_input = _tensor_from_image(image_path)
                if first_pixels is None:
                    first_pixels = pixels
                capture.begin_forward(capture_golden=index == 0)
                _ = model(model_input)
    finally:
        capture.close()
    if first_pixels is None:
        raise QuantizationError("calibration did not produce a model input")
    return first_pixels, capture.records, capture.statistics


def _record_by_name(records: list[CapturedTensor]) -> dict[str, CapturedTensor]:
    result: dict[str, CapturedTensor] = {}
    for record in records:
        result.setdefault(record.name, record)
    return result


def _module_scales(
    statistics: dict[str, dict[str, float]], field: str
) -> dict[str, float]:
    scales: dict[str, float] = {}
    for name, values in statistics.items():
        base_name = name.split("#", maxsplit=1)[0]
        maximum = float(values[field])
        scale = maximum / 127.0 if maximum else 1.0
        scales[base_name] = max(scales.get(base_name, 0.0), scale)
    return scales


def _integer_multiplier(real_multiplier: float) -> tuple[int, int]:
    if real_multiplier < 0 or not np.isfinite(real_multiplier):
        raise QuantizationError(f"invalid requantization multiplier: {real_multiplier}")
    if real_multiplier == 0:
        return 0, 0
    significand, exponent = np.frexp(real_multiplier)
    multiplier = int(round(significand * (1 << 31)))
    if multiplier == 1 << 31:
        multiplier //= 2
        exponent += 1
    return int(multiplier), int(31 - exponent)


def _conv_modules(model: nn.Module) -> list[tuple[str, nn.Conv2d]]:
    modules: list[tuple[str, nn.Conv2d]] = []
    for name, module in model.named_modules():
        if type(module).__name__ in {"Conv", "Ac550QatConv"}:
            modules.append((f"{name}.conv", module.conv))
        elif type(module).__name__ == "Detect":
            modules.extend(
                (f"{name}.m.{index}", getattr(conv, "conv", conv))
                for index, conv in enumerate(module.m)
            )
    if not modules:
        raise QuantizationError("fused model has no convolution modules")
    return modules


def _pack_weights_and_biases(
    model: nn.Module,
    calibration_statistics: dict[str, dict[str, float]],
    output_dir: Path,
) -> dict[str, Any]:
    input_scales = _module_scales(calibration_statistics, "input_max_abs")
    output_scales = _module_scales(calibration_statistics, "output_max_abs")
    for name, module in model.named_modules():
        if isinstance(module, nn.MaxPool2d) and name in input_scales:
            # MaxPool is a byte-preserving signed max operation; it has no output requantizer.
            output_scales[name] = input_scales[name]
    node_scales: dict[str, Any] = {}
    for name in calibration_statistics:
        base_name = name.split("#", maxsplit=1)[0]
        node_scales[base_name] = {
            "input_quantization": {"scale": input_scales[base_name], "zero_point": 0},
            "output_quantization": {"scale": output_scales[base_name], "zero_point": 0},
        }
    weights_path = output_dir / "weights_i8.bin"
    bias_path = output_dir / "bias_i32.bin"
    entries: list[dict[str, Any]] = []
    weight_offset = 0
    bias_offset = 0
    with weights_path.open("wb") as weights_stream, bias_path.open("wb") as bias_stream:
        for name, conv in _conv_modules(model):
            quantized_weights, weight_params = quantize_per_output_channel(conv.weight)
            weight_bytes = quantized_weights.numpy().tobytes(order="C")
            weights_stream.write(weight_bytes)
            input_scale = input_scales.get(name, 1.0 / 127.0)
            output_scale = output_scales.get(name, 1.0 / 127.0)
            weight_scales = np.asarray(weight_params["scales"], dtype=np.float64)
            bias = conv.bias.detach().cpu().numpy() if conv.bias is not None else np.zeros(conv.out_channels, dtype=np.float32)
            bias_i32 = np.rint(bias / (input_scale * weight_scales)).clip(-(1 << 31), (1 << 31) - 1).astype("<i4")
            bias_bytes = bias_i32.tobytes(order="C")
            bias_stream.write(bias_bytes)
            real_multipliers = [input_scale * scale / output_scale for scale in weight_params["scales"]]
            integer_multipliers, right_shifts = zip(*(_integer_multiplier(value) for value in real_multipliers))
            entries.append(
                {
                    "source_module": name,
                    "weight": {
                        "file": weights_path.name,
                        "offset": weight_offset,
                        "bytes": len(weight_bytes),
                        "shape": list(conv.weight.shape),
                        "quantization": weight_params,
                    },
                    "bias": {
                        "file": bias_path.name,
                        "offset": bias_offset,
                        "bytes": len(bias_bytes),
                        "shape": [conv.out_channels],
                        "storage_dtype": "int32",
                    },
                    "input_quantization": {"scale": input_scale, "zero_point": 0},
                    "output_quantization": {"scale": output_scale, "zero_point": 0},
                    "requantization": {
                        "integer_multiplier": list(integer_multipliers),
                        "right_shift": list(right_shifts),
                        "rounding": "ROUND_HALF_TO_EVEN",
                        "saturation": [-127, 127],
                    },
                }
            )
            weight_offset += len(weight_bytes)
            bias_offset += len(bias_bytes)
    return {"schema_version": 2, "nodes": node_scales, "convolutions": entries}


def _safe_vector_name(name: str) -> str:
    return name.replace(".", "_").replace("#", "_call_")


def _quantize_with_scale(tensor: torch.Tensor, scale: float) -> tuple[torch.Tensor, dict[str, Any]]:
    """Quantize with the same fixed per-node scale used by packed Conv parameters."""
    if not np.isfinite(scale) or scale <= 0:
        raise QuantizationError(f"invalid activation scale: {scale}")
    if not torch.isfinite(tensor).all():
        raise QuantizationError("cannot quantize non-finite tensor values")
    quantized = torch.clamp(torch.round(tensor.detach().cpu() / scale), -127, 127).to(torch.int8)
    return quantized, {
        "scheme": "SYMMETRIC_PER_TENSOR",
        "storage_dtype": "int8",
        "scale": scale,
        "zero_point": 0,
        "rounding": "ROUND_HALF_TO_EVEN",
        "saturation": [-127, 127],
    }


def _integer_conv_output(
    inputs: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
    multipliers: np.ndarray,
    shifts: np.ndarray,
    *,
    stride: int,
    padding: int,
    linear: bool,
) -> np.ndarray:
    """Execute one packed OIHW Conv exactly, including INT32 wrap and nearest-even requant."""
    if inputs.ndim != 4 or inputs.shape[0] != 1 or weights.ndim != 4:
        raise QuantizationError("integer Golden Conv expects one NCHW input and OIHW weights")
    _, input_channels, input_height, input_width = inputs.shape
    output_channels, weight_channels, kernel_height, kernel_width = weights.shape
    if weight_channels != input_channels or kernel_height != kernel_width or stride <= 0 or padding < 0:
        raise QuantizationError("unsupported integer Golden Conv shape or parameters")
    if bias.size != output_channels or multipliers.size != output_channels or shifts.size != output_channels:
        raise QuantizationError("integer Golden Conv parameter count differs from output channels")
    output_height = (input_height + 2 * padding - kernel_height) // stride + 1
    output_width = (input_width + 2 * padding - kernel_width) // stride + 1
    if output_height <= 0 or output_width <= 0:
        raise QuantizationError("integer Golden Conv output shape is empty")

    padded = np.pad(
        inputs[0].astype(np.int64),
        ((0, 0), (padding, padding), (padding, padding)),
        mode="constant",
        constant_values=0,
    )
    windows = np.lib.stride_tricks.sliding_window_view(
        padded, (kernel_height, kernel_width), axis=(1, 2)
    )[:, ::stride, ::stride, :, :]
    patches = windows.transpose(1, 2, 0, 3, 4).reshape(output_height * output_width, -1)
    output_weights = weights.astype(np.int64).reshape(output_channels, -1)
    accumulators = (patches @ output_weights.T).T.reshape(output_channels, output_height, output_width)
    accumulators += bias.astype(np.int64)[:, None, None]
    accumulators = np.remainder(accumulators + (1 << 31), 1 << 32) - (1 << 31)
    products = accumulators * multipliers.astype(np.int64)[:, None, None]

    quantized = np.empty_like(accumulators)
    for channel in range(output_channels):
        shift = int(shifts[channel])
        if shift > 63:
            raise QuantizationError(f"integer Golden Conv shift exceeds 63: {shift}")
        quotient = np.right_shift(products[channel], shift)
        if shift:
            remainder = np.bitwise_and(products[channel], (1 << shift) - 1)
            half = 1 << (shift - 1)
            round_up = (remainder > half) | ((remainder == half) & ((quotient & 1) != 0))
            quotient = quotient + round_up.astype(np.int64)
        if not linear:
            quotient = np.maximum(quotient, 0)
        quantized[channel] = np.clip(quotient, -127, 127)
    return quantized.astype(np.int8)[None, ...]


def _integer_maxpool_output(
    inputs: np.ndarray, *, kernel: int, stride: int, padding: int
) -> np.ndarray:
    """Execute signed INT8 MaxPool with the contract's -127 padding value."""
    if inputs.ndim != 4 or inputs.shape[0] != 1 or kernel <= 0 or stride <= 0 or padding < 0:
        raise QuantizationError("unsupported integer Golden MaxPool shape or parameters")
    _, channels, input_height, input_width = inputs.shape
    output_height = (input_height + 2 * padding - kernel) // stride + 1
    output_width = (input_width + 2 * padding - kernel) // stride + 1
    padded = np.pad(
        inputs[0],
        ((0, 0), (padding, padding), (padding, padding)),
        mode="constant",
        constant_values=-127,
    )
    windows = np.lib.stride_tricks.sliding_window_view(padded, (kernel, kernel), axis=(1, 2))
    pooled = windows[:, ::stride, ::stride, :, :].max(axis=(-2, -1))
    if pooled.shape != (channels, output_height, output_width):
        raise QuantizationError("integer Golden MaxPool output shape mismatch")
    return pooled.astype(np.int8)[None, ...]


def _write_golden_vectors(
    records: list[CapturedTensor],
    output_dir: Path,
    node_scales: dict[str, Any],
    convolutions: list[dict[str, Any]],
    conv_modules: dict[str, nn.Conv2d],
    maxpool_modules: dict[str, nn.MaxPool2d],
) -> dict[str, Any]:
    vectors_dir = output_dir / "vectors"
    vectors_dir.mkdir()
    entries: list[dict[str, Any]] = []
    heads: dict[int, Path] = {}
    convolution_by_name = {entry["source_module"]: entry for entry in convolutions}
    for record in records:
        base_name = record.name.split("#", maxsplit=1)[0]
        try:
            scales = node_scales[base_name]
            input_scale = float(scales["input_quantization"]["scale"])
            output_scale = float(scales["output_quantization"]["scale"])
        except KeyError as error:
            raise QuantizationError(f"missing static activation scales for Golden node {base_name}") from error
        input_i8, input_quantization = _quantize_with_scale(record.input, input_scale)
        if base_name in convolution_by_name:
            entry = convolution_by_name[base_name]
            conv = conv_modules[base_name]
            weight_entry = entry["weight"]
            weight_path = output_dir / weight_entry["file"]
            with weight_path.open("rb") as stream:
                stream.seek(int(weight_entry["offset"]))
                weight_bytes = stream.read(int(weight_entry["bytes"]))
            weights = np.frombuffer(weight_bytes, dtype=np.int8).reshape(weight_entry["shape"])
            bias_entry = entry["bias"]
            with (output_dir / bias_entry["file"]).open("rb") as stream:
                stream.seek(int(bias_entry["offset"]))
                bias_bytes = stream.read(int(bias_entry["bytes"]))
            bias = np.frombuffer(bias_bytes, dtype="<i4")
            requantization = entry["requantization"]
            output_array = _integer_conv_output(
                input_i8.numpy()[None, ...] if input_i8.ndim == 3 else input_i8.numpy(),
                weights,
                bias,
                np.asarray(requantization["integer_multiplier"], dtype=np.int32),
                np.asarray(requantization["right_shift"], dtype=np.uint8),
                stride=int(conv.stride[0]),
                padding=int(conv.padding[0]),
                linear=base_name.startswith("model.24.m."),
            )
            output_i8 = torch.from_numpy(output_array.copy())
            output_quantization = {
                "scheme": "SYMMETRIC_PER_TENSOR",
                "storage_dtype": "int8",
                "scale": output_scale,
                "zero_point": 0,
                "rounding": "ROUND_HALF_TO_EVEN",
                "saturation": [-127, 127],
            }
        elif base_name in maxpool_modules:
            maxpool = maxpool_modules[base_name]
            output_array = _integer_maxpool_output(
                input_i8.numpy(),
                kernel=int(maxpool.kernel_size if isinstance(maxpool.kernel_size, int) else maxpool.kernel_size[0]),
                stride=int(maxpool.stride if isinstance(maxpool.stride, int) else maxpool.stride[0]),
                padding=int(maxpool.padding if isinstance(maxpool.padding, int) else maxpool.padding[0]),
            )
            output_i8 = torch.from_numpy(output_array.copy())
            output_quantization = {
                "scheme": "SYMMETRIC_PER_TENSOR",
                "storage_dtype": "int8",
                "scale": output_scale,
                "zero_point": 0,
                "rounding": "ROUND_HALF_TO_EVEN",
                "saturation": [-127, 127],
            }
        else:
            output_i8, output_quantization = _quantize_with_scale(record.output, output_scale)
        stem = _safe_vector_name(record.name)
        input_path = vectors_dir / f"{stem}_input_i8.bin"
        output_path = vectors_dir / f"{stem}_output_i8.bin"
        input_path.write_bytes(input_i8.numpy().tobytes(order="C"))
        output_path.write_bytes(output_i8.numpy().tobytes(order="C"))
        entries.append(
            {
                "source_module": record.name,
                "input": {
                    "file": input_path.relative_to(output_dir).as_posix(),
                    "shape": list(record.input.shape),
                    "quantization": input_quantization,
                },
                "output": {
                    "file": output_path.relative_to(output_dir).as_posix(),
                    "shape": list(record.output.shape),
                    "quantization": output_quantization,
                },
            }
        )
        if record.name.startswith("model.24.m.") and "#" not in record.name:
            heads[int(record.name.rsplit(".", maxsplit=1)[1])] = output_path

    if sorted(heads) != [0, 1, 2]:
        raise QuantizationError("did not capture all P3/P4/P5 Detect heads")
    for index, head_path in heads.items():
        (output_dir / f"head_p{index + 3}_i8.bin").write_bytes(head_path.read_bytes())
    return {"schema_version": 1, "vectors": entries}


def _anchors(model: nn.Module) -> dict[str, Any]:
    detect = model.model[-1]
    anchors = (detect.anchors * detect.stride.view(-1, 1, 1)).detach().cpu().tolist()
    return {"schema_version": 1, "strides": [int(value) for value in detect.stride.tolist()], "anchors_pixels": anchors}


def export_package(
    *,
    weights: Path,
    relu_root: Path,
    calibration_dir: Path,
    calibration_images: int,
    output: Path,
) -> Path:
    """Write a new, immutable AC550 package and return its directory."""
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing package: {output}")
    image_paths = _calibration_images(calibration_dir.resolve(), calibration_images)
    model = _load_frozen_model(weights.resolve(), relu_root.resolve())
    graph = build_execution_graph(model)
    pixels, capture_records, calibration_statistics = _capture_calibration(model, image_paths)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f"{output.name}.tmp-", dir=output.parent))
    try:
        (temporary / "golden_input_u8.bin").write_bytes(pixels.tobytes(order="C"))
        quantization = _pack_weights_and_biases(model, calibration_statistics, temporary)
        conv_modules = dict(_conv_modules(model))
        maxpool_modules = {
            name: module for name, module in model.named_modules() if isinstance(module, nn.MaxPool2d)
        }
        vectors = _write_golden_vectors(
            capture_records,
            temporary,
            quantization["nodes"],
            quantization["convolutions"],
            conv_modules,
            maxpool_modules,
        )
        _write_json(temporary / "graph.json", graph)
        _write_json(temporary / "quant_params.json", quantization)
        _write_json(temporary / "anchors.json", _anchors(model))
        vector_path = temporary / "golden_vectors.json"
        _write_json(vector_path, vectors)
        artifact_names = [
            "golden_input_u8.bin",
            "weights_i8.bin",
            "bias_i32.bin",
            "graph.json",
            "quant_params.json",
            "anchors.json",
            "golden_vectors.json",
            "head_p3_i8.bin",
            "head_p4_i8.bin",
            "head_p5_i8.bin",
        ]
        manifest = {
            "schema_version": 1,
            "status": "PRE_G2_STATIC_QUANTIZATION",
            "target_platform": "Xiaomeige AC550 Cyclone V SoC",
            "model": {
                "architecture": "YOLOv5n-ReLU",
                "checkpoint_sha256": _sha256(weights.resolve()),
                "classes": CLASS_NAMES,
            },
            "input": {
                "shape": INPUT_SHAPE,
                "storage_dtype": "uint8",
                "layout": "NCHW",
                "normalization": "float = uint8 / 255.0",
                "golden_file": "golden_input_u8.bin",
            },
            "calibration": {"images": [str(path) for path in image_paths]},
            "graph": {"file": "graph.json", "nodes": len(graph["nodes"]), "node_count": len(graph["nodes"])},
            "artifacts": {name: {"sha256": _sha256(temporary / name), "bytes": (temporary / name).stat().st_size} for name in artifact_names},
        }
        _write_json(temporary / "manifest.json", manifest)
        temporary.replace(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a frozen YOLOv5n-ReLU checkpoint for the AC550 INT8 runtime.")
    parser.add_argument("--weights", type=Path, required=True, help="frozen YOLOv5n-ReLU best.pt")
    parser.add_argument("--relu-root", type=Path, default=PROJECT_ROOT / "third_party" / "yolov5-relu")
    parser.add_argument("--calibration-dir", type=Path, required=True, help="directory containing L-mode 256x256 images")
    parser.add_argument("--calibration-images", type=int, default=64, help="number of deterministic, sorted calibration images")
    parser.add_argument("--output", type=Path, required=True, help="new output directory; it must not already exist")
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    package = export_package(
        weights=arguments.weights,
        relu_root=arguments.relu_root,
        calibration_dir=arguments.calibration_dir,
        calibration_images=arguments.calibration_images,
        output=arguments.output,
    )
    print(f"AC550 INT8 package: {package}")
    print(f"manifest_sha256: {_sha256(package / 'manifest.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
