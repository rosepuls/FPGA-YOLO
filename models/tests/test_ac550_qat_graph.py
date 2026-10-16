from __future__ import annotations

from pathlib import Path
import sys
import unittest

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = PROJECT_ROOT / "runs" / "yolov5n_gray_relu_256" / "weights" / "best.pt"
RELU_ROOT = PROJECT_ROOT / "third_party" / "yolov5-relu"


@unittest.skipUnless(WEIGHTS.is_file(), "requires the trained ReLU checkpoint")
class Ac550QatGraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import models

        vendor_models = str(RELU_ROOT / "models")
        if vendor_models not in models.__path__:
            models.__path__.append(vendor_models)
        if str(RELU_ROOT) not in sys.path:
            sys.path.insert(0, str(RELU_ROOT))
        from models.export.export_yolov5n_relu_ac550 import _load_frozen_model

        cls.model = _load_frozen_model(WEIGHTS, RELU_ROOT)

    def test_prepare_replaces_conv_and_boundary_operators_with_qat_modules(self) -> None:
        """The model graph must expose fake-quant boundaries beyond only Conv weights."""
        from models.qat.ac550_qat_graph import (
            Ac550QatBottleneck,
            Ac550QatConcat,
            Ac550QatConv,
            Ac550QatDetectConv,
            Ac550QatSPPF,
            Ac550QatUpsample,
            prepare_ac550_qat,
        )

        qat_model = prepare_ac550_qat(self.model)
        modules = list(qat_model.modules())

        convs = [module for module in modules if isinstance(module, Ac550QatConv)]
        self.assertGreater(len(convs), 0)
        self.assertTrue(all(module.weight_fake_quant.observer.per_channel for module in convs))
        self.assertTrue(all(not module.output_fake_quant.observer.per_channel for module in convs))
        self.assertEqual(len([module for module in modules if isinstance(module, Ac550QatDetectConv)]), 3)
        self.assertTrue(any(isinstance(module, Ac550QatBottleneck) for module in modules))
        self.assertTrue(any(isinstance(module, Ac550QatConcat) for module in modules))
        self.assertTrue(any(isinstance(module, Ac550QatSPPF) for module in modules))
        self.assertEqual(len([module for module in modules if isinstance(module, Ac550QatUpsample)]), 2)

    def test_qat_phase_changes_fake_quant_and_observer_state(self) -> None:
        """Weight-only warmup must not quantize activation edges before full QAT."""
        from models.qat.ac550_qat_graph import prepare_ac550_qat, set_qat_phase
        from models.qat.ac550_fake_quant import Ac550FakeQuant

        qat_model = prepare_ac550_qat(self.model)
        quantizers = [module for module in qat_model.modules() if isinstance(module, Ac550FakeQuant)]
        set_qat_phase(qat_model, "collect")
        self.assertTrue(all(not bool(module.enabled) and bool(module.observer_enabled) for module in quantizers))
        set_qat_phase(qat_model, "weight")
        self.assertTrue(
            all(
                bool(module.enabled) == (module.quantization_kind == "weight")
                and bool(module.observer_enabled)
                for module in quantizers
            )
        )
        set_qat_phase(qat_model, "train")
        self.assertTrue(all(bool(module.enabled) and bool(module.observer_enabled) for module in quantizers))
        set_qat_phase(qat_model, "freeze")
        self.assertTrue(all(bool(module.enabled) and not bool(module.observer_enabled) for module in quantizers))

    def test_prepared_model_supports_one_batch_forward_backward(self) -> None:
        """The QAT wrappers must remain differentiable through all three Detect heads."""
        from models.qat.ac550_qat_graph import prepare_ac550_qat, set_qat_phase

        qat_model = prepare_ac550_qat(self.model)
        qat_model.train()
        set_qat_phase(qat_model, "train")
        outputs = qat_model(torch.rand(1, 1, 256, 256))
        torch.stack([output.float().mean() for output in outputs]).sum().backward()

        gradients = [parameter.grad for parameter in qat_model.parameters() if parameter.requires_grad]
        self.assertTrue(all(gradient is None or torch.isfinite(gradient).all() for gradient in gradients))
        self.assertTrue(any(gradient is not None for gradient in gradients))

    def test_vendor_ema_copies_qat_phase_buffers(self) -> None:
        """EMA validation must enter the same fake-quant phase as the train model."""
        from models.qat.ac550_fake_quant import Ac550FakeQuant, Ac550Observer
        from models.train.train_yolov5n_relu_ac550_qat import _import_vendor_training
        from torch import nn

        _import_vendor_training()
        from utils.torch_utils import ModelEMA

        model = nn.Sequential(Ac550FakeQuant(Ac550Observer(per_channel=False, channel_axis=None)))
        ema = ModelEMA(model)
        model[0].set_enabled(False)
        model[0].set_observer_enabled(False)
        ema.update(model)

        self.assertFalse(bool(ema.ema[0].enabled))
        self.assertFalse(bool(ema.ema[0].observer_enabled))

    def test_qat_wrapper_keeps_new_observer_buffers_on_source_device(self) -> None:
        """Preparing an already-GPU model must not introduce CPU-only QAT buffers."""
        from models.qat.ac550_qat_graph import Ac550QatConv
        from torch import nn

        class SourceConv(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = nn.Conv2d(1, 2, 1, bias=False)
                self.bn = nn.BatchNorm2d(2)
                self.act = nn.ReLU()

        source = SourceConv().to("meta")
        qat_conv = Ac550QatConv(source)

        self.assertEqual(qat_conv.conv.weight.device, qat_conv.weight_fake_quant.observer.max_abs.device)
        self.assertEqual(qat_conv.conv.weight.device, qat_conv.output_fake_quant.observer.max_abs.device)

    def test_qat_conv_observes_batch_norm_folded_weight(self) -> None:
        """QAT must quantize the Conv+BN folded weight used by AC550 inference."""
        from models.qat.ac550_qat_graph import Ac550QatConv
        from torch import nn

        class SourceConv(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = nn.Conv2d(1, 1, 1, bias=False)
                self.bn = nn.BatchNorm2d(1)
                self.act = nn.ReLU()

        source = SourceConv().eval()
        source.conv.weight.data.fill_(1.0)
        source.bn.weight.data.fill_(4.0)
        source.bn.running_var.data.fill_(4.0)
        qat_conv = Ac550QatConv(source)
        qat_conv(torch.ones(1, 1, 1, 1))

        expected = torch.tensor([4.0 / (4.0 + source.bn.eps) ** 0.5])
        self.assertTrue(torch.allclose(qat_conv.weight_fake_quant.observer.max_abs, expected))

    def test_fused_qat_conv_matches_source_when_fake_quant_is_disabled(self) -> None:
        """Collection mode must retain the original Conv+BN+ReLU float result."""
        from models.qat.ac550_qat_graph import Ac550QatConv
        from torch import nn

        class SourceConv(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = nn.Conv2d(1, 1, 1, bias=False)
                self.bn = nn.BatchNorm2d(1)
                self.act = nn.ReLU()

            def forward(self, value):
                return self.act(self.bn(self.conv(value)))

        source = SourceConv().eval()
        source.conv.weight.data.fill_(0.75)
        source.bn.weight.data.fill_(1.5)
        source.bn.bias.data.fill_(-0.2)
        source.bn.running_mean.data.fill_(0.3)
        source.bn.running_var.data.fill_(2.0)
        qat_conv = Ac550QatConv(source)
        for fake_quant in (
            qat_conv.input_fake_quant,
            qat_conv.weight_fake_quant,
            qat_conv.output_fake_quant,
        ):
            fake_quant.set_enabled(False)
        value = torch.tensor([[[[1.0, -2.0]]]])

        self.assertTrue(torch.allclose(qat_conv(value), source(value), atol=1e-6, rtol=1e-6))

    def test_only_first_conv_quantizes_model_input(self) -> None:
        """A sequential consumer must use its producer's INT8 edge without a second quantization."""
        from models.qat.ac550_qat_graph import Ac550QatConv, prepare_ac550_qat

        qat_model = prepare_ac550_qat(self.model)
        qat_model(torch.rand(1, 1, 256, 256))
        convs = [module for module in qat_model.modules() if isinstance(module, Ac550QatConv)]

        self.assertTrue(bool(convs[0].input_fake_quant.observer.initialized))
        self.assertTrue(all(not bool(module.input_fake_quant.observer.initialized) for module in convs[1:]))

    def test_concat_quantizes_only_the_merged_edge(self) -> None:
        """Concat branches are already stored INT8 edges and must not be quantized again."""
        from models.qat.ac550_qat_graph import Ac550QatConcat
        from torch import nn

        class SourceConcat(nn.Module):
            d = 1

        qat_concat = Ac550QatConcat(SourceConcat())
        qat_concat([torch.ones(1, 1, 1, 1), torch.ones(1, 1, 1, 1)])

        self.assertFalse(bool(qat_concat.left_branch_fake_quant.observer.initialized))
        self.assertFalse(bool(qat_concat.right_branch_fake_quant.observer.initialized))
        self.assertTrue(bool(qat_concat.concat_output_fake_quant.observer.initialized))

    def test_residual_add_quantizes_only_the_result_edge(self) -> None:
        """Residual and main tensors are producer edges; only their sum gets a new scale."""
        from models.qat.ac550_qat_graph import Ac550QatBottleneck
        from torch import nn

        class SourceBottleneck(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.cv1 = nn.Identity()
                self.cv2 = nn.Identity()
                self.add = True

        qat_bottleneck = Ac550QatBottleneck(SourceBottleneck())
        qat_bottleneck(torch.ones(1, 1, 1, 1))

        self.assertFalse(bool(qat_bottleneck.residual_fake_quant.observer.initialized))
        self.assertFalse(bool(qat_bottleneck.main_fake_quant.observer.initialized))
        self.assertTrue(bool(qat_bottleneck.add_output_fake_quant.observer.initialized))


if __name__ == "__main__":
    unittest.main()
