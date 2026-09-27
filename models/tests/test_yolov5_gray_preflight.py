from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import shutil
import subprocess
import sys
import unittest
import uuid

from PIL import Image
import torch
import yaml

from tools.train_yolov5n_silu import load_run_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_CONFIG = PROJECT_ROOT / "configs" / "yolov5n_gray_silu_baseline.yaml"
TRAIN_SCRIPT = PROJECT_ROOT / "tools" / "train_yolov5n_silu.py"


@contextmanager
def fixture_preflight_config():
    root = Path(__file__).parent / f"yolov5-preflight-fixture-{uuid.uuid4().hex}"
    image_dir = root / "dataset" / "images" / "train"
    label_dir = root / "dataset" / "labels" / "train"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    try:
        for index, value in enumerate((32, 64, 96, 128)):
            stem = f"sample-{index}"
            Image.new("L", (256, 256), value).save(image_dir / f"{stem}.png")
            (label_dir / f"{stem}.txt").write_text(
                "0 0.500000 0.500000 0.250000 0.250000\n",
                encoding="utf-8",
            )

        relative_root = root.relative_to(PROJECT_ROOT).as_posix()
        data_yaml = root / "fixture_data.yaml"
        data_yaml.write_text(
            yaml.safe_dump({
                "path": f"{relative_root}/dataset",
                "train": "images/train",
                "val": "images/train",
                "test": "images/train",
                "nc": 10,
                "names": {index: f"class-{index}" for index in range(10)},
            }, sort_keys=False),
            encoding="utf-8",
        )

        checkpoint = root / "rgb-first-conv.pt"
        rgb = torch.arange(16 * 3 * 6 * 6, dtype=torch.float32).reshape(16, 3, 6, 6)
        torch.save({"state_dict": {"model.0.conv.weight": rgb}}, checkpoint)

        run_config = root / "run.yaml"
        run_config.write_text(
            yaml.safe_dump({
                "data": data_yaml.relative_to(PROJECT_ROOT).as_posix(),
                "cfg": "configs/yolov5n_gray_256.yaml",
                "weights": checkpoint.relative_to(PROJECT_ROOT).as_posix(),
                "hyp": "third_party/yolov5/data/hyps/hyp.scratch-low.yaml",
                "imgsz": 256,
                "channels": 1,
                "epochs": 300,
                "batch_size": 32,
                "seed": 20260927,
                "device": "0",
                "workers": 4,
                "cache": None,
                "project": "runs",
                "name": "fixture",
                "exist_ok": False,
            }, sort_keys=False),
            encoding="utf-8",
        )
        yield run_config
    finally:
        shutil.rmtree(root)


class YoloV5GrayPreflightTests(unittest.TestCase):
    def test_run_config_locks_baseline_contract(self) -> None:
        config = load_run_config(RUN_CONFIG)

        self.assertEqual(config["data"], "gc10_yolo.yaml")
        self.assertEqual(config["cfg"], "configs/yolov5n_gray_256.yaml")
        self.assertEqual(config["weights"], "yolov5n.pt")
        self.assertEqual(config["imgsz"], 256)
        self.assertEqual(config["channels"], 1)
        self.assertEqual(config["epochs"], 300)
        self.assertEqual(config["batch_size"], 32)
        self.assertEqual(config["seed"], 20260927)
        self.assertIsNone(config["cache"])

    def test_preflight_proves_grayscale_training_contract(self) -> None:
        with fixture_preflight_config() as run_config:
            result = subprocess.run(
                [sys.executable, str(TRAIN_SCRIPT), "--config", str(run_config), "--preflight-only"],
                cwd=PROJECT_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout.strip().splitlines()[-1])
        self.assertEqual(report["batch_shape"][1:], [1, 256, 256])
        self.assertEqual(report["first_conv_in_channels"], 1)
        self.assertEqual(report["head_out_channels"], [45, 45, 45])
        self.assertTrue(report["pretrained_first_conv_verified"])
        self.assertEqual(report["status"], "ready")


if __name__ == "__main__":
    unittest.main()
