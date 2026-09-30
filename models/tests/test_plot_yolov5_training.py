from __future__ import annotations

import csv
from pathlib import Path
import shutil
import sys
import unittest
import uuid

from tools.plot_yolov5_training import build_validation_command, create_training_plots


PROJECT_ROOT = Path(__file__).resolve().parents[2]
YOLOV5_ROOT = PROJECT_ROOT / "third_party" / "yolov5"
if str(YOLOV5_ROOT) not in sys.path:
    sys.path.insert(0, str(YOLOV5_ROOT))

from utils.metrics import background_background_annotation, confusion_annotation_style


CSV_COLUMNS = [
    "epoch",
    "train/box_loss",
    "train/obj_loss",
    "train/cls_loss",
    "metrics/precision",
    "metrics/recall",
    "metrics/mAP_0.5",
    "metrics/mAP_0.5:0.95",
    "val/box_loss",
    "val/obj_loss",
    "val/cls_loss",
]


class PlotYoloV5TrainingTests(unittest.TestCase):
    def write_results_csv(self, directory: Path) -> Path:
        csv_path = directory / "results.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            writer.writerow(
                {
                    "epoch": "0",
                    "train/box_loss": "0.12",
                    "train/obj_loss": "0.02",
                    "train/cls_loss": "0.04",
                    "metrics/precision": "0.30",
                    "metrics/recall": "0.20",
                    "metrics/mAP_0.5": "0.10",
                    "metrics/mAP_0.5:0.95": "0.03",
                    "val/box_loss": "0.10",
                    "val/obj_loss": "0.01",
                    "val/cls_loss": "0.03",
                }
            )
            writer.writerow(
                {
                    "epoch": "1",
                    "train/box_loss": "0.08",
                    "train/obj_loss": "0.01",
                    "train/cls_loss": "0.02",
                    "metrics/precision": "0.80",
                    "metrics/recall": "0.70",
                    "metrics/mAP_0.5": "0.75",
                    "metrics/mAP_0.5:0.95": "0.40",
                    "val/box_loss": "0.07",
                    "val/obj_loss": "0.01",
                    "val/cls_loss": "0.02",
                }
            )
        return csv_path

    def test_create_training_plots_writes_metrics_and_loss_pngs(self) -> None:
        root = Path(__file__).parent / f"yolo-plot-fixture-{uuid.uuid4().hex}"
        root.mkdir()
        try:
            outputs = create_training_plots(self.write_results_csv(root), root / "analysis")

            self.assertEqual(set(outputs), {"metrics", "losses"})
            for output in outputs.values():
                self.assertTrue(output.is_file())
                self.assertGreater(output.stat().st_size, 1_000)
        finally:
            shutil.rmtree(root)

    def test_build_validation_command_uses_grayscale_test_configuration(self) -> None:
        command = build_validation_command(
            python="python",
            yolo_root=Path("vendor/yolov5"),
            weights=Path("run/weights/best.pt"),
            data=Path("lv_yolo.yaml"),
            output_root=Path("run/analysis"),
            device="0",
            batch_size=32,
            workers=4,
        )

        self.assertEqual(command[:2], ["python", str(Path("vendor/yolov5/val.py"))])
        self.assertEqual(command[command.index("--channels") + 1], "1")
        self.assertEqual(command[command.index("--task") + 1], "test")
        self.assertEqual(command[command.index("--imgsz") + 1], "256")
        self.assertEqual(command[command.index("--weights") + 1], str(Path("run/weights/best.pt")))

    def test_confusion_annotations_use_high_contrast_on_light_and_dark_cells(self) -> None:
        self.assertEqual(confusion_annotation_style(0.02), ("#111827", "white"))
        self.assertEqual(confusion_annotation_style(0.98), ("white", "#111827"))

    def test_background_to_background_cell_is_explicitly_marked_not_applicable(self) -> None:
        self.assertEqual(background_background_annotation(), "N/A")


if __name__ == "__main__":
    unittest.main()
