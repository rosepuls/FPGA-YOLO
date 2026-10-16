"""Serializable AC550 QAT wrappers for the ReLU YOLOv5 graph.

The wrappers deliberately retain familiar YOLO attributes (``conv``, ``bn``,
``act``, ``cv1`` and so on).  This keeps state-dict paths predictable while
making each integer-storage boundary explicit for QAT and later export.
"""

from __future__ import annotations

from typing import Any, Literal

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from .ac550_fake_quant import Ac550FakeQuant, Ac550Observer


QatPhase = Literal["collect", "weight", "train", "freeze"]


def _activation_fake_quant(momentum: float) -> Ac550FakeQuant:
    return Ac550FakeQuant(Ac550Observer(per_channel=False, channel_axis=None, momentum=momentum))


def _weight_fake_quant(momentum: float, channels: int) -> Ac550FakeQuant:
    return Ac550FakeQuant(
        Ac550Observer(per_channel=True, channel_axis=0, momentum=momentum, channels=channels),
        quantization_kind="weight",
    )


def _copy_yolo_metadata(source: nn.Module, destination: nn.Module) -> None:
    """Retain attributes used by YOLO's routed ``BaseModel`` forward method."""
    for attribute in ("i", "f", "type", "np"):
        if hasattr(source, attribute):
            setattr(destination, attribute, getattr(source, attribute))


class Ac550QatConv(nn.Module):
    """QAT replacement for YOLO's Conv+BN+ReLU block."""

    def __init__(self, source: nn.Module, observer_momentum: float = 0.95) -> None:
        super().__init__()
        self.conv = source.conv
        # ``Model.fuse()`` removes the BatchNorm attribute.  Supporting that
        # form makes graph inspection safe; QAT training itself loads an
        # unfused checkpoint and therefore retains trainable BatchNorm.
        self.bn = getattr(source, "bn", nn.Identity())
        self.act = source.act
        self.input_fake_quant = _activation_fake_quant(observer_momentum)
        self.weight_fake_quant = _weight_fake_quant(observer_momentum, self.conv.out_channels)
        self.output_fake_quant = _activation_fake_quant(observer_momentum)
        self.quantize_input = True
        self.to(device=self.conv.weight.device)
        _copy_yolo_metadata(source, self)

    def forward(self, value: Tensor) -> Tensor:
        if self.quantize_input:
            value = self.input_fake_quant(value)
        if isinstance(self.bn, nn.BatchNorm2d):
            # Quantize the inference-time Conv+BN folded weight.  The
            # following one-pass approximation follows PyTorch's fused QAT
            # ConvBn2d formulation while retaining BN batch-stat updates.
            running_std = torch.sqrt(self.bn.running_var + self.bn.eps)
            scale_factor = self.bn.weight / running_std
            weight_shape = [-1] + [1] * (self.conv.weight.ndim - 1)
            channel_shape = [1, -1] + [1] * (self.conv.weight.ndim - 2)
            folded_weight = self.conv.weight * scale_factor.reshape(weight_shape)
            quantized_weight = self.weight_fake_quant(folded_weight)
            zero_bias = torch.zeros_like(self.conv.bias) if self.conv.bias is not None else None
            value = functional.conv2d(
                value,
                quantized_weight,
                zero_bias,
                self.conv.stride,
                self.conv.padding,
                self.conv.dilation,
                self.conv.groups,
            )
            value = value / scale_factor.reshape(channel_shape)
            if self.conv.bias is not None:
                value = value + self.conv.bias.reshape(channel_shape)
            value = self.bn(value)
        else:
            if isinstance(self.conv, nn.Conv2d):
                weight = self.weight_fake_quant(self.conv.weight)
                value = functional.conv2d(
                    value,
                    weight,
                    self.conv.bias,
                    self.conv.stride,
                    self.conv.padding,
                    self.conv.dilation,
                    self.conv.groups,
                )
            else:
                # The Python INT8 verifier replaces ``conv`` with a module
                # that consumes package weights directly.
                value = self.conv(value)
            value = self.bn(value)
        return self.output_fake_quant(self.act(value))

    forward_fuse = forward


class Ac550QatDetectConv(nn.Module):
    """QAT replacement for one linear Detect 1x1 convolution (no activation)."""

    def __init__(self, source: nn.Conv2d, observer_momentum: float = 0.95) -> None:
        super().__init__()
        self.conv = source
        self.input_fake_quant = _activation_fake_quant(observer_momentum)
        self.weight_fake_quant = _weight_fake_quant(observer_momentum, self.conv.out_channels)
        self.output_fake_quant = _activation_fake_quant(observer_momentum)
        self.to(device=self.conv.weight.device)

    def forward(self, value: Tensor) -> Tensor:
        if isinstance(self.conv, nn.Conv2d):
            weight = self.weight_fake_quant(self.conv.weight)
            value = functional.conv2d(
                value,
                weight,
                self.conv.bias,
                self.conv.stride,
                self.conv.padding,
                self.conv.dilation,
                self.conv.groups,
            )
        else:
            value = self.conv(value)
        return self.output_fake_quant(value)


class Ac550QatBottleneck(nn.Module):
    """Explicit quantized residual-add boundary."""

    def __init__(self, source: nn.Module, observer_momentum: float = 0.95) -> None:
        super().__init__()
        self.cv1 = source.cv1
        self.cv2 = source.cv2
        self.add = source.add
        self.residual_fake_quant = _activation_fake_quant(observer_momentum)
        self.main_fake_quant = _activation_fake_quant(observer_momentum)
        self.add_output_fake_quant = _activation_fake_quant(observer_momentum)
        _copy_yolo_metadata(source, self)

    def forward(self, value: Tensor) -> Tensor:
        main = self.cv2(self.cv1(value))
        if not self.add:
            return main
        return self.add_output_fake_quant(value + main)


class Ac550QatC3(nn.Module):
    """C3 wrapper with a declared shared concat quantization boundary."""

    def __init__(self, source: nn.Module, observer_momentum: float = 0.95) -> None:
        super().__init__()
        self.cv1 = source.cv1
        self.cv2 = source.cv2
        self.cv3 = source.cv3
        self.m = source.m
        self.left_branch_fake_quant = _activation_fake_quant(observer_momentum)
        self.right_branch_fake_quant = _activation_fake_quant(observer_momentum)
        self.concat_output_fake_quant = _activation_fake_quant(observer_momentum)
        _copy_yolo_metadata(source, self)

    def forward(self, value: Tensor) -> Tensor:
        left = self.m(self.cv1(value))
        right = self.cv2(value)
        return self.cv3(self.concat_output_fake_quant(torch.cat((left, right), 1)))


class Ac550QatSPPF(nn.Module):
    """SPPF wrapper with three separately visible max-pool boundaries."""

    def __init__(self, source: nn.Module, observer_momentum: float = 0.95) -> None:
        super().__init__()
        self.cv1 = source.cv1
        self.cv2 = source.cv2
        self.m = source.m
        self.pool1_fake_quant = _activation_fake_quant(observer_momentum)
        self.pool2_fake_quant = _activation_fake_quant(observer_momentum)
        self.pool3_fake_quant = _activation_fake_quant(observer_momentum)
        self.concat_output_fake_quant = _activation_fake_quant(observer_momentum)
        _copy_yolo_metadata(source, self)

    def forward(self, value: Tensor) -> Tensor:
        value = self.cv1(value)
        first = self.m(value)
        second = self.m(first)
        third = self.m(second)
        return self.cv2(self.concat_output_fake_quant(torch.cat((value, first, second, third), 1)))


class Ac550QatConcat(nn.Module):
    """External route concat with per-branch and shared-output fake quantizers."""

    def __init__(self, source: nn.Module, observer_momentum: float = 0.95) -> None:
        super().__init__()
        self.dimension = source.d
        self.left_branch_fake_quant = _activation_fake_quant(observer_momentum)
        self.right_branch_fake_quant = _activation_fake_quant(observer_momentum)
        self.concat_output_fake_quant = _activation_fake_quant(observer_momentum)
        _copy_yolo_metadata(source, self)

    def forward(self, values: list[Tensor]) -> Tensor:
        if len(values) != 2:
            raise ValueError(f"AC550 YOLOv5 concat expects two branches, received {len(values)}")
        return self.concat_output_fake_quant(torch.cat(values, self.dimension))


class Ac550QatUpsample(nn.Module):
    """Upsample is scale preserving but records an explicit output boundary."""

    def __init__(self, source: nn.Upsample, observer_momentum: float = 0.95) -> None:
        super().__init__()
        self.upsample = source
        self.input_fake_quant = _activation_fake_quant(observer_momentum)
        self.output_fake_quant = _activation_fake_quant(observer_momentum)
        _copy_yolo_metadata(source, self)

    def forward(self, value: Tensor) -> Tensor:
        return self.upsample(value)


def _replace_module(root: nn.Module, name: str, replacement: nn.Module) -> None:
    parent_name, _, child_name = name.rpartition(".")
    parent = root.get_submodule(parent_name) if parent_name else root
    parent._modules[child_name] = replacement


def _parent(root: nn.Module, name: str) -> nn.Module | None:
    parent_name, separator, _ = name.rpartition(".")
    return root.get_submodule(parent_name) if separator else None


def _is_detect_child(root: nn.Module, name: str) -> bool:
    parent_name, separator, _ = name.rpartition(".")
    if not separator or not parent_name.endswith(".m"):
        return False
    detect_name = parent_name.rpartition(".")[0]
    return type(root.get_submodule(detect_name)).__name__ == "Detect"


def prepare_ac550_qat(model: nn.Module, observer_momentum: float = 0.95) -> nn.Module:
    """Replace supported ReLU YOLOv5 blocks with serializable AC550 QAT wrappers."""
    if getattr(model, "_ac550_qat_prepared", False):
        return model
    if not 0.0 <= observer_momentum < 1.0:
        raise ValueError("observer_momentum must be in [0, 1)")

    modules = sorted(list(model.named_modules()), key=lambda item: item[0].count("."), reverse=True)
    for name, module in modules:
        if not name:
            continue
        module_name = type(module).__name__
        if module_name == "Conv":
            _replace_module(model, name, Ac550QatConv(module, observer_momentum))
        elif module_name == "Bottleneck":
            _replace_module(model, name, Ac550QatBottleneck(module, observer_momentum))
        elif module_name == "C3":
            _replace_module(model, name, Ac550QatC3(module, observer_momentum))
        elif module_name == "SPPF":
            _replace_module(model, name, Ac550QatSPPF(module, observer_momentum))
        elif module_name == "Concat":
            _replace_module(model, name, Ac550QatConcat(module, observer_momentum))
        elif isinstance(module, nn.Upsample):
            _replace_module(model, name, Ac550QatUpsample(module, observer_momentum))
        elif isinstance(module, nn.Conv2d) and _is_detect_child(model, name):
            _replace_module(model, name, Ac550QatDetectConv(module, observer_momentum))

    # Edges are quantized by their producer.  The image is the only model
    # input without a preceding producer, so only model.0 may quantize input.
    for module in model.modules():
        if isinstance(module, Ac550QatConv):
            module.quantize_input = False
    first_layer = getattr(model, "model", [None])[0]
    if not isinstance(first_layer, Ac550QatConv):
        raise ValueError("AC550 QAT model must begin with a normal Conv layer")
    first_layer.quantize_input = True

    # Training creates the base model on ``device`` before firing the QAT
    # preparation hook.  New observer modules begin on CPU, so relocate the
    # entire rewritten graph back to the original parameter device.
    first_parameter = next(model.parameters(), None)
    if first_parameter is not None:
        model.to(device=first_parameter.device)
    model._ac550_qat_prepared = True
    model._ac550_qat_phase = "collect"
    set_qat_phase(model, "collect")
    return model


def set_qat_phase(model: nn.Module, phase: QatPhase) -> None:
    """Switch every QAT edge atomically between range collection and fake quantization."""
    if phase not in {"collect", "weight", "train", "freeze"}:
        raise ValueError("QAT phase must be one of: collect, weight, train, freeze")
    for module in model.modules():
        if isinstance(module, Ac550FakeQuant):
            kind = getattr(module, "quantization_kind", "activation")
            module.set_enabled(phase in {"train", "freeze"} or (phase == "weight" and kind == "weight"))
            module.set_observer_enabled(phase != "freeze")
    model._ac550_qat_phase = phase


def qat_metadata(model: nn.Module) -> dict[str, Any]:
    """Return a JSON-safe description of the active AC550 QAT contract."""
    if not getattr(model, "_ac550_qat_prepared", False):
        raise ValueError("model was not prepared with prepare_ac550_qat")
    modules = list(model.modules())
    return {
        "quantization_method": "QAT_AC550_SYMMETRIC_INT8",
        "qrange": [-127, 127],
        "zero_point": 0,
        "rounding": "ROUND_HALF_TO_EVEN",
        "phase": getattr(model, "_ac550_qat_phase", None),
        "normal_conv_weight_quantization": "per_output_channel_symmetric_int8",
        "activation_quantization": "per_tensor_symmetric_int8",
        "counts": {
            "normal_convs": sum(isinstance(module, Ac550QatConv) for module in modules),
            "detect_convs": sum(isinstance(module, Ac550QatDetectConv) for module in modules),
            "residual_adds": sum(isinstance(module, Ac550QatBottleneck) and module.add for module in modules),
            "external_concats": sum(isinstance(module, Ac550QatConcat) for module in modules),
            "sppf": sum(isinstance(module, Ac550QatSPPF) for module in modules),
            "upsamples": sum(isinstance(module, Ac550QatUpsample) for module in modules),
        },
    }


__all__ = [
    "Ac550QatBottleneck",
    "Ac550QatC3",
    "Ac550QatConcat",
    "Ac550QatConv",
    "Ac550QatDetectConv",
    "Ac550QatSPPF",
    "Ac550QatUpsample",
    "prepare_ac550_qat",
    "qat_metadata",
    "set_qat_phase",
]
