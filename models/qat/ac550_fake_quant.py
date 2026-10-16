"""Signed symmetric INT8 fake quantization primitives for the AC550 contract."""

from __future__ import annotations

from numbers import Real

import torch
from torch import Tensor, nn


AC550_QMIN = -127
AC550_QMAX = 127
_MIN_SCALE = torch.finfo(torch.float32).eps


def _as_valid_scale(scale: Tensor | Real, *, device: torch.device) -> Tensor:
    """Return an FP32 scale tensor, rejecting values that cannot define quantization."""
    scale_tensor = torch.as_tensor(scale, dtype=torch.float32, device=device)
    if scale_tensor.numel() == 0 or not torch.isfinite(scale_tensor).all() or not (scale_tensor > 0).all():
        raise ValueError("AC550 quantization scales must be finite and strictly positive")
    return scale_tensor


def _quantize_dequantize(value: Tensor, scale: Tensor | Real) -> Tensor:
    scale_tensor = _as_valid_scale(scale, device=value.device)
    quantized = torch.round(value.float() / scale_tensor).clamp(AC550_QMIN, AC550_QMAX)
    return quantized * scale_tensor


def requantize_i8(value: Tensor, source_scale: Tensor | Real, target_scale: Tensor | Real) -> Tensor:
    """Emulate AC550's source INT8 storage followed by target-scale requantization.

    ``torch.round`` provides the required Round-Half-To-Even behavior.  Values are
    returned dequantized so this function can compose with FP32 QAT training.
    """
    source = _as_valid_scale(source_scale, device=value.device)
    target = _as_valid_scale(target_scale, device=value.device)
    source_value = _quantize_dequantize(value, source)
    return _quantize_dequantize(source_value, target)


class Ac550Observer(nn.Module):
    """FP32 EMA absolute-maximum observer for symmetric AC550 quantization."""

    def __init__(
        self,
        per_channel: bool,
        channel_axis: int | None,
        momentum: float = 0.95,
        channels: int | None = None,
    ) -> None:
        super().__init__()
        if per_channel and channel_axis is None:
            raise ValueError("per-channel observer requires channel_axis")
        if channels is not None and (not per_channel or channels <= 0):
            raise ValueError("channels is only valid as a positive per-channel observer size")
        if not 0.0 <= momentum < 1.0:
            raise ValueError("observer momentum must be in [0, 1)")
        self.per_channel = per_channel
        self.channel_axis = channel_axis
        self.momentum = float(momentum)
        initial_maximum = torch.zeros(channels if channels is not None else 1, dtype=torch.float32)
        self.register_buffer("max_abs", initial_maximum)
        self.register_buffer("initialized", torch.tensor(False, dtype=torch.bool))

    @torch.no_grad()
    def update(self, value: Tensor) -> None:
        detached = value.detach().float()
        if self.per_channel:
            axis = self.channel_axis
            assert axis is not None
            axis %= detached.ndim
            reduction_dims = tuple(index for index in range(detached.ndim) if index != axis)
            observed = detached.abs().amax(dim=reduction_dims) if reduction_dims else detached.abs()
        else:
            observed = detached.abs().amax().reshape(1)

        if not bool(self.initialized.item()) or self.max_abs.shape != observed.shape:
            self.max_abs = observed.clone()
            self.initialized.fill_(True)
        else:
            self.max_abs.mul_(self.momentum).add_(observed, alpha=1.0 - self.momentum)

    def scale(self) -> Tensor:
        return (self.max_abs / AC550_QMAX).clamp_min(_MIN_SCALE)

    def forward(self, value: Tensor) -> Tensor:
        self.update(value)
        return value


class Ac550FakeQuant(nn.Module):
    """AC550 symmetric fake quantizer with a straight-through estimator."""

    def __init__(
        self,
        observer: Ac550Observer,
        enabled: bool = True,
        observer_enabled: bool = True,
        quantization_kind: str = "activation",
    ) -> None:
        super().__init__()
        if quantization_kind not in {"activation", "weight"}:
            raise ValueError("quantization_kind must be 'activation' or 'weight'")
        self.observer = observer
        self.quantization_kind = quantization_kind
        self.register_buffer("enabled", torch.tensor(bool(enabled), dtype=torch.bool))
        self.register_buffer("observer_enabled", torch.tensor(bool(observer_enabled), dtype=torch.bool))

    def set_enabled(self, enabled: bool) -> None:
        self.enabled.fill_(enabled)

    def set_observer_enabled(self, enabled: bool) -> None:
        self.observer_enabled.fill_(enabled)

    def _broadcast_scale(self, value: Tensor) -> Tensor:
        scale = self.observer.scale().to(device=value.device, dtype=torch.float32)
        if not self.observer.per_channel:
            return scale
        axis = self.observer.channel_axis
        assert axis is not None
        axis %= value.ndim
        shape = [1] * value.ndim
        shape[axis] = scale.numel()
        return scale.reshape(shape)

    def forward(self, value: Tensor) -> Tensor:
        if bool(self.observer_enabled.item()):
            self.observer.update(value)
        if not bool(self.enabled.item()):
            return value
        dequantized = _quantize_dequantize(value, self._broadcast_scale(value)).to(dtype=value.dtype)
        return value + (dequantized - value).detach()
