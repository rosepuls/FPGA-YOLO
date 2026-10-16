"""INT8 packing primitives and AC550 execution-graph discovery for YOLOv5n-ReLU."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import torch
import torch.nn as nn


SUPPORTED_EXECUTORS = frozenset(
    {
        "FPGA_CONV_RELU",
        "FPGA_CONV_LINEAR",
        "FPGA_MAXPOOL",
        "ARM_ADD",
        "ARM_CONCAT",
        "ARM_UPSAMPLE",
    }
)


class QuantizationError(ValueError):
    """Raised when a tensor cannot be represented by the AC550 INT8 package."""


def _require_finite(tensor: torch.Tensor) -> None:
    if not torch.is_floating_point(tensor):
        return
    if not torch.isfinite(tensor).all():
        raise QuantizationError("cannot quantize non-finite tensor values")


def _symmetric_scale(tensor: torch.Tensor) -> float:
    _require_finite(tensor)
    if tensor.numel() == 0:
        raise QuantizationError("cannot quantize an empty tensor")
    maximum = float(tensor.detach().abs().max().cpu())
    return maximum / 127.0 if maximum else 1.0


def quantize_symmetric(tensor: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
    """Return a signed-INT8 tensor and its explicit, hardware-facing quantization rule."""
    scale = _symmetric_scale(tensor)
    quantized = torch.clamp(torch.round(tensor.detach().cpu() / scale), -127, 127).to(torch.int8)
    return quantized, {
        "scheme": "SYMMETRIC_PER_TENSOR",
        "storage_dtype": "int8",
        "scale": scale,
        "zero_point": 0,
        "rounding": "ROUND_HALF_TO_EVEN",
        "saturation": [-127, 127],
    }


def quantize_per_output_channel(weights: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
    """Quantize convolution weights per output channel, retaining one scale per output channel."""
    if weights.ndim < 1 or weights.shape[0] == 0:
        raise QuantizationError("weights must have at least one non-empty output-channel dimension")
    _require_finite(weights)
    flattened = weights.detach().cpu().reshape(weights.shape[0], -1)
    maxima = flattened.abs().amax(dim=1)
    scales = torch.where(maxima == 0, torch.ones_like(maxima), maxima / 127.0)
    reshaped_scales = scales.reshape((weights.shape[0],) + (1,) * (weights.ndim - 1))
    quantized = torch.clamp(torch.round(weights.detach().cpu() / reshaped_scales), -127, 127).to(torch.int8)
    return quantized, {
        "scheme": "SYMMETRIC_PER_OUTPUT_CHANNEL",
        "storage_dtype": "int8",
        "axis": 0,
        "scales": [float(scale) for scale in scales],
        "zero_point": 0,
        "rounding": "ROUND_HALF_TO_EVEN",
        "saturation": [-127, 127],
    }


def _conv_node(name: str, module: nn.Module) -> dict[str, Any]:
    conv = module.conv
    activation = type(module.act).__name__
    if activation != "ReLU":
        raise QuantizationError(f"ordinary convolution is not ReLU: {name} ({activation})")
    return {
        "id": name,
        "executor": "FPGA_CONV_RELU",
        "kind": "CONV_RELU",
        "source_module": name,
        "input_channels": conv.in_channels,
        "output_channels": conv.out_channels,
        "kernel": list(conv.kernel_size),
        "stride": list(conv.stride),
        "padding": list(conv.padding),
        "groups": conv.groups,
    }


def _detect_nodes(name: str, module: nn.Module) -> Iterator[dict[str, Any]]:
    for index, conv in enumerate(module.m):
        conv = getattr(conv, "conv", conv)
        yield {
            "id": f"{name}.m.{index}",
            "executor": "FPGA_CONV_LINEAR",
            "kind": "DETECT_CONV",
            "source_module": f"{name}.m.{index}",
            "input_channels": conv.in_channels,
            "output_channels": conv.out_channels,
            "kernel": list(conv.kernel_size),
            "stride": list(conv.stride),
            "padding": list(conv.padding),
            "groups": conv.groups,
        }


def build_execution_graph(model: nn.Module) -> dict[str, Any]:
    """Discover all executable AC550 operations without treating a YOLO block as a black box."""
    nodes: list[dict[str, Any]] = []
    for name, module in model.named_modules():
        class_name = type(module).__name__
        if class_name in {"Conv", "Ac550QatConv"}:
            nodes.append(_conv_node(name, module))
        elif class_name == "Detect":
            nodes.extend(_detect_nodes(name, module))
        elif isinstance(module, nn.Upsample) or class_name == "Ac550QatUpsample":
            upsample = getattr(module, "upsample", module)
            nodes.append(
                {
                    "id": name,
                    "executor": "ARM_UPSAMPLE",
                    "kind": "UPSAMPLE_NEAREST",
                    "source_module": name,
                    "scale_factor": upsample.scale_factor,
                }
            )
        elif class_name in {"Concat", "Ac550QatConcat"}:
            dimension = getattr(module, "d", getattr(module, "dimension", 1))
            nodes.append(
                {
                    "id": name,
                    "executor": "ARM_CONCAT",
                    "kind": "CONCAT",
                    "source_module": name,
                    "dimension": dimension,
                }
            )
        elif class_name in {"C3", "Ac550QatC3"}:
            nodes.append(
                {
                    "id": f"{name}.concat",
                    "executor": "ARM_CONCAT",
                    "kind": "CONCAT",
                    "source_module": name,
                    "dimension": 1,
                }
            )
        elif class_name in {"Bottleneck", "Ac550QatBottleneck"} and module.add:
            nodes.append(
                {
                    "id": f"{name}.add",
                    "executor": "ARM_ADD",
                    "kind": "RESIDUAL_ADD",
                    "source_module": name,
                }
            )
        elif class_name in {"SPPF", "Ac550QatSPPF"}:
            pooling = getattr(module, "m", None)
            if pooling is None:
                raise QuantizationError(f"SPPF node has no pooling operator: {name}")
            for occurrence in range(1, 4):
                nodes.append(
                    {
                        "id": f"{name}.maxpool.{occurrence}",
                        "executor": "FPGA_MAXPOOL",
                        "kind": "MAXPOOL",
                        "source_module": f"{name}.m",
                        "kernel": list(pooling.kernel_size if isinstance(pooling.kernel_size, tuple) else (pooling.kernel_size,) * 2),
                        "stride": list(pooling.stride if isinstance(pooling.stride, tuple) else (pooling.stride,) * 2),
                        "padding": list(pooling.padding if isinstance(pooling.padding, tuple) else (pooling.padding,) * 2),
                    }
                )
            nodes.append(
                {
                    "id": f"{name}.concat",
                    "executor": "ARM_CONCAT",
                    "kind": "CONCAT",
                    "source_module": name,
                    "dimension": 1,
                }
            )

    if not nodes:
        raise QuantizationError("no AC550 execution nodes discovered")
    unsupported = {node["executor"] for node in nodes} - SUPPORTED_EXECUTORS
    if unsupported:
        raise QuantizationError(f"unsupported execution nodes: {sorted(unsupported)}")
    return {"schema_version": 1, "nodes": nodes}
