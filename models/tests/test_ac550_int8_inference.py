from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE = PROJECT_ROOT / "runs" / "ac550_yolov5n_relu_int8_qat_staged_v1"
RELU_ROOT = PROJECT_ROOT / "third_party" / "yolov5-relu"
WEIGHTS = PROJECT_ROOT / "runs" / "yolov5n_gray_relu_256_qat_staged_v1" / "weights" / "best_qat.pt"


class Ac550Int8InferenceTests(unittest.TestCase):
    def test_fake_quantize_dequantize_uses_symmetric_int8_saturation(self) -> None:
        """Non-Conv boundaries must clamp to signed INT8 before returning to float simulation."""
        from models.export.int8_inference_ac550 import _fake_quantize_dequantize
        import torch

        actual = _fake_quantize_dequantize(torch.tensor([-100.0, -0.5, 0.0, 0.5, 100.0]), 0.5)
        expected = torch.tensor([-63.5, -0.5, 0.0, 0.5, 63.5])
        self.assertTrue(torch.equal(actual, expected))

    def test_node_report_contains_shape_and_error_metrics(self) -> None:
        """Per-node evidence must retain enough information to locate quantization drift."""
        from models.export.int8_inference_ac550 import _node_report
        import torch

        report = _node_report("model.9.m", torch.tensor([[[[0.0, 1.0]]]]), torch.tensor([[[[0.0, 0.5]]]]))
        self.assertEqual(report["shape"], [1, 1, 1, 2])
        self.assertAlmostEqual(report["max_abs_diff"], 0.5)
        self.assertAlmostEqual(report["mean_abs_diff"], 0.25)

    def test_select_images_is_deterministic_and_filters_non_images(self) -> None:
        """Acceptance must use a stable image order and ignore unrelated files."""
        from models.export.ac550_g2_acceptance import select_images

        temporary = Path(tempfile.mkdtemp(prefix="ac550-images-", dir=PROJECT_ROOT / "models" / "tests"))
        try:
            (temporary / "b.png").write_bytes(b"png")
            (temporary / "a.jpg").write_bytes(b"jpg")
            (temporary / "notes.txt").write_text("ignore", encoding="utf-8")
            selected = select_images(temporary, limit=2)
            self.assertEqual([path.name for path in selected], ["a.jpg", "b.png"])
        finally:
            shutil.rmtree(temporary)

    def test_aggregate_reports_fails_on_unmatched_detection(self) -> None:
        """G2 cannot pass when a quantized image loses a reference detection."""
        from models.export.ac550_g2_acceptance import aggregate_reports

        report = {
            "input": {"source": "sample.png"},
            "package_manifest_sha256": "ABC",
            "package_status": "PRE_G2_STATIC_QUANTIZATION",
            "graph": {"node_count": 85, "node_coverage": {"integer_simulation_complete": False}},
            "heads": {
                "P3": {"mean_abs_diff": 0.1, "rmse": 0.2},
                "P4": {"mean_abs_diff": 0.1, "rmse": 0.2},
                "P5": {"mean_abs_diff": 0.1, "rmse": 0.2},
            },
            "detections": {
                "comparison": {
                    "reference_count": 1,
                    "quantized_count": 0,
                    "matched": 0,
                    "matches": [],
                    "unmatched_reference": [0],
                    "unmatched_quantized": [],
                }
            },
        }
        aggregate = aggregate_reports(
            [report],
            {
                "max_head_mean_abs_diff": 0.25,
                "max_head_rmse": 0.35,
                "min_box_iou": 0.90,
                "max_confidence_diff": 0.05,
                "min_images": 1,
            },
        )
        self.assertFalse(aggregate["passed"])
        self.assertTrue(any("unmatched" in reason for reason in aggregate["failure_reasons"]))

    def test_detection_matching_reports_box_class_and_confidence_differences(self) -> None:
        """Catch a comparator that ignores class or confidence while claiming a box matched."""
        from models.export.int8_inference_ac550 import compare_detections

        reference = [[10.0, 10.0, 30.0, 30.0, 0.80, 2.0]]
        quantized = [[11.0, 10.0, 30.0, 30.0, 0.75, 2.0]]

        report = compare_detections(reference, quantized, iou_threshold=0.5)

        self.assertEqual(report["matched"], 1)
        self.assertAlmostEqual(report["matches"][0]["confidence_abs_diff"], 0.05, places=6)
        self.assertEqual(report["matches"][0]["reference_class"], 2)
        self.assertEqual(report["matches"][0]["quantized_class"], 2)

    def test_g2_cli_writes_head_and_detection_comparison_report(self) -> None:
        """Catch a CLI that runs but does not persist reproducible FP32-vs-INT8 evidence."""
        temporary = Path(tempfile.mkdtemp(prefix="ac550-g2-", dir=PROJECT_ROOT / "models" / "tests"))
        try:
            output = temporary / "int8_compare.json"
            result = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "models" / "export" / "int8_inference_ac550.py"),
                    "--package",
                    str(PACKAGE),
                    "--weights",
                    str(WEIGHTS),
                    "--relu-root",
                    str(RELU_ROOT),
                    "--output",
                    str(output),
                ],
                cwd=PROJECT_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["package_status"], "PRE_G2_STATIC_QUANTIZATION")
            self.assertEqual(set(report["heads"]), {"P3", "P4", "P5"})
            self.assertIn("detections", report)
            self.assertIn("reference", report["detections"])
            self.assertIn("quantized", report["detections"])
            self.assertIn("comparison", report["detections"])
        finally:
            shutil.rmtree(temporary)

    def test_g2_acceptance_cli_records_detection_threshold_failure(self) -> None:
        """The batch gate must persist an explicit failure when a strict detection threshold is missed."""
        temporary = Path(tempfile.mkdtemp(prefix="ac550-g2-acceptance-", dir=PROJECT_ROOT / "models" / "tests"))
        try:
            output = temporary / "g2_acceptance.json"
            result = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "models" / "export" / "ac550_g2_acceptance.py"),
                    "--package",
                    str(PACKAGE),
                    "--weights",
                    str(WEIGHTS),
                    "--relu-root",
                    str(RELU_ROOT),
                    "--image-dir",
                    str(PROJECT_ROOT / "data" / "LV-YOLO-ROI256" / "images" / "train"),
                    "--max-images",
                    "1",
                    "--min-box-iou",
                    "1.0",
                    "--output",
                    str(output),
                ],
                cwd=PROJECT_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertFalse(report["passed"])
            self.assertTrue(any("minimum box IoU" in reason for reason in report["failure_reasons"]))
            self.assertEqual(report["aggregate"]["image_count"], 1)
            self.assertEqual(report["aggregate"]["node_coverage"]["package_conv_nodes"], 60)
            self.assertEqual(report["aggregate"]["node_coverage"]["measured_conv_nodes"], 60)
            self.assertEqual(report["aggregate"]["node_coverage"]["missing_conv_sources"], [])
            self.assertTrue(report["aggregate"]["node_coverage"]["integer_simulation_complete"])
        finally:
            shutil.rmtree(temporary)


if __name__ == "__main__":
    unittest.main()
