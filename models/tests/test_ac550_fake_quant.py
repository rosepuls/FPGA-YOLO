from __future__ import annotations

import unittest

import torch


class Ac550FakeQuantTests(unittest.TestCase):
    def test_requantize_clamps_to_signed_ac550_range(self) -> None:
        """AC550 must store signed INT8 values in the closed [-127, 127] range."""
        from models.qat.ac550_fake_quant import requantize_i8

        actual = requantize_i8(torch.tensor([-100.0, 100.0]), 0.5, 0.5)

        self.assertTrue(torch.equal(actual, torch.tensor([-63.5, 63.5])))

    def test_requantize_uses_round_half_to_even(self) -> None:
        """Ties must use the same banker's rounding rule as the deployment contract."""
        from models.qat.ac550_fake_quant import requantize_i8

        actual = requantize_i8(torch.tensor([-1.5, -0.5, 0.5, 1.5]), 1.0, 1.0)

        self.assertTrue(torch.equal(actual, torch.tensor([-2.0, 0.0, 0.0, 2.0])))

    def test_per_output_channel_observer_has_one_scale_per_channel(self) -> None:
        """Conv weights require a separate symmetric scale for each output channel."""
        from models.qat.ac550_fake_quant import Ac550Observer

        observer = Ac550Observer(per_channel=True, channel_axis=0, momentum=0.9)
        observer(torch.tensor([[[[1.0]]], [[[2.0]]], [[[127.0]]]]))

        self.assertTrue(torch.allclose(observer.scale(), torch.tensor([1.0 / 127.0, 2.0 / 127.0, 1.0])))

    def test_preallocated_weight_observer_keeps_ema_compatible_shape(self) -> None:
        """Per-channel weight ranges have a fixed state shape before the first optimizer step."""
        from models.qat.ac550_fake_quant import Ac550Observer

        observer = Ac550Observer(per_channel=True, channel_axis=0, channels=3)

        self.assertEqual(tuple(observer.max_abs.shape), (3,))
        observer(torch.ones(3, 1, 1, 1))
        self.assertEqual(tuple(observer.max_abs.shape), (3,))

    def test_disabled_fake_quant_is_identity(self) -> None:
        """Collection-only epochs must not perturb the FP32 forward value."""
        from models.qat.ac550_fake_quant import Ac550FakeQuant, Ac550Observer

        fake_quant = Ac550FakeQuant(Ac550Observer(per_channel=False, channel_axis=None), enabled=False)
        value = torch.tensor([-1.2, 0.25, 2.7])

        self.assertTrue(torch.equal(fake_quant(value), value))

    def test_fake_quant_uses_ste_gradient(self) -> None:
        """Quantization error is detached so trainable in-range inputs retain unit gradient."""
        from models.qat.ac550_fake_quant import Ac550FakeQuant, Ac550Observer

        observer = Ac550Observer(per_channel=False, channel_axis=None)
        observer(torch.tensor([-1.0, 1.0]))
        fake_quant = Ac550FakeQuant(observer, enabled=True, observer_enabled=False)
        value = torch.tensor([-0.25, 0.25], requires_grad=True)

        fake_quant(value).sum().backward()

        self.assertTrue(torch.equal(value.grad, torch.ones_like(value)))

    def test_invalid_scale_is_rejected(self) -> None:
        """Zero, negative and non-finite scales cannot silently form an integer package."""
        from models.qat.ac550_fake_quant import requantize_i8

        with self.assertRaises(ValueError):
            requantize_i8(torch.tensor([1.0]), 0.0, 1.0)


if __name__ == "__main__":
    unittest.main()
