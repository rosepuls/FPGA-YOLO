from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from PIL import Image
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RELU_ROOT = PROJECT_ROOT / "third_party" / "yolov5-relu"
RELU_WEIGHTS = PROJECT_ROOT / "runs" / "yolov5n_gray_relu_256" / "weights" / "best.pt"
from models.export.quantize_yolov5n_relu_ac550 import (
    QuantizationError,
    build_execution_graph,
    quantize_per_output_channel,
    quantize_symmetric,
)


class Ac550QuantizationTests(unittest.TestCase):
    def test_symmetric_quantization_uses_signed_int8_and_nearest_even_rounding(self) -> None:
        """Catch a quantizer that emits unsigned values or changes the declared rounding rule."""
        quantized, parameters = quantize_symmetric(torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0]))

        self.assertEqual(quantized.dtype, torch.int8)
        self.assertEqual(quantized.tolist(), [-127, -64, 0, 64, 127])
        self.assertEqual(parameters["zero_point"], 0)
        self.assertEqual(parameters["rounding"], "ROUND_HALF_TO_EVEN")
        self.assertAlmostEqual(parameters["scale"], 1 / 127)

    def test_quantization_rejects_non_finite_tensor_values(self) -> None:
        """Catch an exporter that silently creates a corrupt hardware package from NaN or infinity."""
        with self.assertRaisesRegex(QuantizationError, "non-finite"):
            quantize_symmetric(torch.tensor([0.0, float("nan")]))

    def test_per_output_channel_weights_keep_each_channel_scale(self) -> None:
        """Catch a weight packer that incorrectly applies one scale to all convolution output channels."""
        weights = torch.tensor([[[[1.0]]], [[[0.25]]]])

        quantized, parameters = quantize_per_output_channel(weights)

        self.assertEqual(quantized.dtype, torch.int8)
        self.assertEqual(quantized.reshape(2).tolist(), [127, 127])
        self.assertEqual(parameters["axis"], 0)
        self.assertAlmostEqual(parameters["scales"][0], 1 / 127)
        self.assertAlmostEqual(parameters["scales"][1], 0.25 / 127)

    def test_execution_graph_assigns_only_supported_ac550_executors(self) -> None:
        """Catch a graph exporter that sends a Conv to ARM or accidentally applies ReLU to Detect."""
        probe = f"""
import importlib.util
import json
import sys

exporter = {str(PROJECT_ROOT / 'models' / 'export' / 'quantize_yolov5n_relu_ac550.py')!r}
spec = importlib.util.spec_from_file_location('ac550_quantizer', exporter)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
sys.path.insert(0, {str(RELU_ROOT)!r})
from models.yolo import Model

model = Model({str(RELU_ROOT / 'models' / 'yolov5n_relu_4cls_gray.yaml')!r}).eval().fuse()
print('GRAPH=' + json.dumps(module.build_execution_graph(model)))
"""
        result = subprocess.run(
            [sys.executable, "-c", probe],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        marker = next(line for line in result.stdout.splitlines() if line.startswith("GRAPH="))
        graph = json.loads(marker.removeprefix("GRAPH="))
        executors = {node["executor"] for node in graph["nodes"]}
        detect_nodes = [node for node in graph["nodes"] if node["kind"] == "DETECT_CONV"]

        self.assertTrue(executors <= {
            "FPGA_CONV_RELU",
            "FPGA_CONV_LINEAR",
            "FPGA_MAXPOOL",
            "ARM_ADD",
            "ARM_CONCAT",
            "ARM_UPSAMPLE",
        })
        self.assertEqual(len(detect_nodes), 3)
        self.assertEqual({node["executor"] for node in detect_nodes}, {"FPGA_CONV_LINEAR"})
        self.assertEqual({node["output_channels"] for node in detect_nodes}, {27})
        self.assertIn("ARM_ADD", executors)
        self.assertIn("ARM_CONCAT", executors)
        self.assertIn("FPGA_MAXPOOL", executors)

    def test_export_cli_writes_ac550_model_package_and_head_golden_vectors(self) -> None:
        """Catch an exporter that reports success without writing the package required by FPGA and HPS."""
        temporary = Path(tempfile.mkdtemp(prefix="ac550-export-", dir=PROJECT_ROOT / "models" / "tests"))
        calibration = temporary / "calibration"
        output = temporary / "package"
        calibration.mkdir()
        Image.new("L", (256, 256), 96).save(calibration / "sample.png")
        Image.new("L", (256, 256), 160).save(calibration / "sample-2.png")
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "models" / "export" / "export_yolov5n_relu_ac550.py"),
                    "--weights",
                    str(RELU_WEIGHTS),
                    "--relu-root",
                    str(RELU_ROOT),
                    "--calibration-dir",
                    str(calibration),
                    "--calibration-images",
                    "2",
                    "--output",
                    str(output),
                ],
                cwd=PROJECT_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual((output / "golden_input_u8.bin").stat().st_size, 256 * 256)
            self.assertTrue((output / "manifest.json").is_file())
            self.assertTrue((output / "graph.json").is_file())
            self.assertTrue((output / "weights_i8.bin").is_file())
            self.assertTrue((output / "bias_i32.bin").is_file())
            self.assertTrue((output / "quant_params.json").is_file())
            for name in ("head_p3_i8.bin", "head_p4_i8.bin", "head_p5_i8.bin"):
                self.assertTrue((output / name).is_file(), name)

            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "PRE_G2_STATIC_QUANTIZATION")
            self.assertEqual(manifest["input"]["shape"], [1, 1, 256, 256])
            self.assertEqual(manifest["model"]["classes"], ["ca_shang", "zang_wu", "zhe_zhou", "zhen_kong"])
            self.assertEqual(manifest["graph"]["nodes"], manifest["graph"]["node_count"])
            self.assertEqual(len(manifest["calibration"]["images"]), 2)
            vectors = json.loads((output / "golden_vectors.json").read_text(encoding="utf-8"))
            self.assertFalse(any(vector["source_module"].startswith("model.0.conv#") for vector in vectors["vectors"]))
            quant_params = json.loads((output / "quant_params.json").read_text(encoding="utf-8"))
            self.assertIn("nodes", quant_params)
            self.assertIn("model.9.m", quant_params["nodes"])
            self.assertGreater(quant_params["nodes"]["model.9.m"]["output_quantization"]["scale"], 0.0)
        finally:
            shutil.rmtree(temporary)


if __name__ == "__main__":
    unittest.main()
