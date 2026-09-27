from __future__ import annotations

import os
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
YOLOV5_ROOT = PROJECT_ROOT / "third_party" / "yolov5"
TEST_CACHE_ROOT = Path(tempfile.gettempdir())
os.environ.setdefault("YOLOV5_CONFIG_DIR", str(TEST_CACHE_ROOT / "fpga-yolo-ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(TEST_CACHE_ROOT / "fpga-yolo-matplotlib"))
if str(YOLOV5_ROOT) not in sys.path:
    sys.path.insert(0, str(YOLOV5_ROOT))

from utils.torch_utils import adapt_input_conv


class YoloV5GrayModelTests(unittest.TestCase):
    def run_vendor_python(self, code: str) -> subprocess.CompletedProcess[str]:
        environment = os.environ.copy()
        environment["YOLOV5_CONFIG_DIR"] = str(TEST_CACHE_ROOT / "fpga-yolo-ultralytics")
        environment["MPLCONFIGDIR"] = str(TEST_CACHE_ROOT / "fpga-yolo-matplotlib")
        return subprocess.run(
            [sys.executable, "-c", code],
            cwd=PROJECT_ROOT,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_adapt_input_conv_averages_rgb_channels(self) -> None:
        rgb = torch.arange(2 * 3 * 2 * 2, dtype=torch.float32).reshape(2, 3, 2, 2)
        source = {"model.0.conv.weight": rgb.clone()}
        target = {"model.0.conv.weight": torch.empty(2, 1, 2, 2)}

        adapted, key = adapt_input_conv(source, target, input_channels=1)

        self.assertEqual(key, "model.0.conv.weight")
        torch.testing.assert_close(adapted[key], rgb.mean(dim=1, keepdim=True))

    def test_adapt_input_conv_leaves_rgb_checkpoint_unchanged(self) -> None:
        rgb = torch.ones(2, 3, 2, 2)
        source = {"model.0.conv.weight": rgb}
        target = {"model.0.conv.weight": torch.empty_like(rgb)}

        adapted, key = adapt_input_conv(source, target, input_channels=3)

        self.assertIsNone(key)
        self.assertIs(adapted["model.0.conv.weight"], rgb)

    def test_adapt_input_conv_accepts_already_grayscale_checkpoint(self) -> None:
        gray = torch.ones(2, 1, 2, 2)
        source = {"model.0.conv.weight": gray}
        target = {"model.0.conv.weight": torch.empty_like(gray)}

        adapted, key = adapt_input_conv(source, target, input_channels=1)

        self.assertIsNone(key)
        self.assertIs(adapted["model.0.conv.weight"], gray)

    def test_single_channel_model_has_expected_first_conv_and_heads(self) -> None:
        code = f"""
import json
import sys
sys.path.insert(0, {str(YOLOV5_ROOT)!r})
from models.yolo import Model
model = Model({str(YOLOV5_ROOT / 'models' / 'yolov5n.yaml')!r}, ch=1, nc=10)
print('CONTRACT=' + json.dumps({{
    'first_conv': model.model[0].conv.in_channels,
    'heads': [layer.out_channels for layer in model.model[-1].m],
}}))
"""
        result = self.run_vendor_python(code)
        self.assertEqual(result.returncode, 0, result.stderr)
        marker = next(line for line in result.stdout.splitlines() if line.startswith("CONTRACT="))
        contract = json.loads(marker.removeprefix("CONTRACT="))

        self.assertEqual(contract, {"first_conv": 1, "heads": [45, 45, 45]})

    def test_training_cli_exposes_channels(self) -> None:
        result = subprocess.run(
            [sys.executable, str(YOLOV5_ROOT / "train.py"), "--help"],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--channels", result.stdout)

    def test_amp_check_accepts_channel_count(self) -> None:
        code = f"""
import sys
import torch.nn as nn
sys.path.insert(0, {str(YOLOV5_ROOT)!r})
from utils.general import check_amp
print(check_amp(nn.Conv2d(1, 1, 1), image_channels=1))
"""
        result = self.run_vendor_python(code)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip().splitlines()[-1], "False")


if __name__ == "__main__":
    unittest.main()
