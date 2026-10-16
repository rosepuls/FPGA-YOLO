from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RELU_ROOT = PROJECT_ROOT / "third_party" / "yolov5-relu"
RELU_CONFIG = RELU_ROOT / "models" / "yolov5n_relu_4cls_gray.yaml"


def model_probe() -> str:
    """Return a standalone probe so vendor imports cannot leak between tests."""
    return f"""
import json
import sys
from pathlib import Path
import torch
import torch.nn as nn

relu_root = Path({str(RELU_ROOT)!r})
sys.path.insert(0, str(relu_root))
from models.common import Conv
from models.yolo import Model

model = Model(str(relu_root / 'models' / 'yolov5n_relu_4cls_gray.yaml'))
detect = model.model[-1]
with torch.no_grad():
    heads = model(torch.zeros(1, 1, 256, 256))

contract = {{
    'first_conv_in_channels': model.model[0].conv.in_channels,
    'detect_classes': detect.nc,
    'detect_output_channels': [head.out_channels for head in detect.m],
    'detect_head_types': [type(head).__name__ for head in detect.m],
    'conv_activation_types': sorted({{type(module.act).__name__ for module in model.modules() if isinstance(module, Conv)}}),
    'silu_module_count': sum(isinstance(module, nn.SiLU) for module in model.modules()),
    'head_shapes': [list(head.shape) for head in heads],
}}
print('CONTRACT=' + json.dumps(contract))
"""


class YoloV5ReluVariantTests(unittest.TestCase):
    def test_relu_variant_has_ac550_deployment_contract(self) -> None:
        """Catch a wrong activation, shape, class count, or non-linear Detect head."""
        result = subprocess.run(
            [sys.executable, "-c", model_probe()],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        marker = next(line for line in result.stdout.splitlines() if line.startswith("CONTRACT="))
        contract = json.loads(marker.removeprefix("CONTRACT="))

        self.assertEqual(contract["first_conv_in_channels"], 1)
        self.assertEqual(contract["detect_classes"], 4)
        self.assertEqual(contract["detect_output_channels"], [27, 27, 27])
        self.assertEqual(contract["detect_head_types"], ["Conv2d", "Conv2d", "Conv2d"])
        self.assertEqual(contract["conv_activation_types"], ["ReLU"])
        self.assertEqual(contract["silu_module_count"], 0)
        self.assertEqual(
            contract["head_shapes"],
            [[1, 3, 32, 32, 9], [1, 3, 16, 16, 9], [1, 3, 8, 8, 9]],
        )


if __name__ == "__main__":
    unittest.main()
