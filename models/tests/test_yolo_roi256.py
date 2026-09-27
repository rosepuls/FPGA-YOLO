from __future__ import annotations

from contextlib import contextmanager, redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
import shutil
import unittest
import uuid

from PIL import Image

from models.convert_coco_to_yolo_roi256 import (
    ConversionConfig,
    coco_bbox_to_yolo,
    convert_dataset,
    main,
)


@contextmanager
def fixture_coco_dataset():
    root = Path(__file__).parent / f"yolo-roi256-fixture-{uuid.uuid4().hex}"
    root.mkdir()
    try:
        source = root / "GC10-DET-COCO"
        for split_name in ("train", "val", "test"):
            (source / "images" / split_name).mkdir(parents=True)
        (source / "annotations").mkdir()

        Image.new("L", (200, 100), 128).save(source / "images" / "train" / "positive.png")
        Image.new("L", (2048, 1000), 96).save(source / "images" / "train" / "edge.png")
        Image.new("L", (2048, 1000), 80).save(source / "images" / "train" / "edge-round-up.png")
        Image.new("L", (50, 50), 64).save(source / "images" / "val" / "negative.png")
        Image.new("L", (80, 40), 32).save(source / "images" / "test" / "test.png")

        payloads = {
            "train": {
                "images": [
                    {"id": 1, "file_name": "positive.png", "width": 200, "height": 100},
                    {"id": 4, "file_name": "edge.png", "width": 2048, "height": 1000},
                    {"id": 5, "file_name": "edge-round-up.png", "width": 2048, "height": 1000},
                ],
                "annotations": [
                    {"id": 1, "image_id": 1, "category_id": 1, "bbox": [20, 10, 40, 20]},
                    {"id": 3, "image_id": 4, "category_id": 1, "bbox": [190, 76, 1858, 883]},
                    {"id": 4, "image_id": 5, "category_id": 1, "bbox": [1753, 62, 295, 896]},
                ],
            },
            "val": {
                "images": [{"id": 2, "file_name": "negative.png", "width": 50, "height": 50}],
                "annotations": [],
            },
            "test": {
                "images": [{"id": 3, "file_name": "test.png", "width": 80, "height": 40}],
                "annotations": [{"id": 2, "image_id": 3, "category_id": 10, "bbox": [0, 0, 40, 20]}],
            },
        }
        for split_name, payload in payloads.items():
            (source / "annotations" / f"instances_{split_name}.json").write_text(
                json.dumps(payload),
                encoding="utf-8",
            )
        yield source
    finally:
        shutil.rmtree(root)


class YoloRoi256Tests(unittest.TestCase):
    def test_coco_bbox_to_yolo_scales_independent_axes(self) -> None:
        self.assertEqual(
            coco_bbox_to_yolo([20, 10, 40, 20], image_width=200, image_height=100),
            (0.2, 0.2, 0.2, 0.2),
        )

    def test_convert_dataset_writes_256_grayscale_images_and_yolo_labels(self) -> None:
        with fixture_coco_dataset() as source:
            output = source.parent / "output"
            report = convert_dataset(ConversionConfig(source, output))

            self.assertTrue(report["validation"]["valid"])
            with Image.open(output / "images" / "train" / "positive.png") as image:
                self.assertEqual((image.mode, image.size), ("L", (256, 256)))
            self.assertEqual(
                (output / "labels" / "train" / "positive.txt").read_text(encoding="utf-8").strip(),
                "0 0.200000 0.200000 0.200000 0.200000",
            )
            self.assertEqual(
                (output / "labels" / "val" / "negative.txt").read_text(encoding="utf-8"),
                "",
            )
            self.assertEqual(
                (output / "labels" / "test" / "test.txt").read_text(encoding="utf-8").strip(),
                "9 0.250000 0.250000 0.500000 0.500000",
            )

    def test_convert_dataset_keeps_six_decimal_edge_box_inside_yolo_bounds(self) -> None:
        with fixture_coco_dataset() as source:
            output = source.parent / "output"
            report = convert_dataset(ConversionConfig(source, output))

            self.assertTrue(report["validation"]["valid"])
            for filename in ("edge.txt", "edge-round-up.txt"):
                fields = (output / "labels" / "train" / filename).read_text(encoding="utf-8").split()
                self.assertEqual(len(fields), 5)
                self.assertTrue(all(len(value.partition(".")[2]) == 6 for value in fields[1:]))
                xc, yc, width, height = (float(value) for value in fields[1:])
                self.assertLessEqual(xc + width / 2, 1.0)
                self.assertLessEqual(yc + height / 2, 1.0)

    def test_cli_requires_overwrite_before_replacing_dataset(self) -> None:
        with fixture_coco_dataset() as source:
            output = source.parent / "output"
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                self.assertEqual(main(["--source", str(source), "--output", str(output)]), 0)
                self.assertEqual(main(["--source", str(source), "--output", str(output)]), 2)
                self.assertEqual(
                    main(["--source", str(source), "--output", str(output), "--overwrite"]),
                    0,
                )

    def test_cli_validates_without_writing(self) -> None:
        with fixture_coco_dataset() as source:
            output = source.parent / "output"
            with redirect_stdout(StringIO()):
                self.assertEqual(main(["--source", str(source), "--output", str(output)]), 0)
                self.assertEqual(main(["--validate", str(output)]), 0)


if __name__ == "__main__":
    unittest.main()

