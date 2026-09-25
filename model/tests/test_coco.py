from contextlib import contextmanager
import json
from pathlib import Path
import shutil
import unittest
import uuid

import numpy as np
from PIL import Image

from model.gc10_coco.coco import ConversionConfig, coco_bbox, validate_coco, write_dataset
from model.gc10_coco.source import Box, SourceRecord


@contextmanager
def test_directory():
    path = Path(__file__).parent / f"coco-fixture-{uuid.uuid4().hex}"
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path)


def make_record(root: Path, index: int, group: str, category: int | None) -> SourceRecord:
    image_path = root / f"{index}.jpg"
    y, x = np.indices((64, 64))
    array = ((x * 7 + y * 11 + index) % 256).astype(np.uint8)
    Image.fromarray(array).save(image_path)
    boxes = () if category is None else (Box(category, f"class-{category}", 24, 24, 40, 40),)
    return SourceRecord(
        xml_path=root / f"{index}.xml",
        image_path=image_path,
        output_stem=f"1__img_01_{group}_{index:05d}",
        material_group=group,
        width=64,
        height=64,
        boxes=boxes,
    )


class CocoTests(unittest.TestCase):
    def test_pascal_voc_box_converts_to_coco_width_and_height(self) -> None:
        box = Box(1, "1_chongkong", 10, 20, 31, 55)
        self.assertEqual(coco_bbox(box), [10, 20, 21, 35])

    def test_writer_preserves_augmented_boxes_and_creates_empty_negatives(self) -> None:
        with test_directory() as root:
            source = root / "source"
            source.mkdir()
            splits = {
                "train": [
                    make_record(source, 1, "TRAIN-A", 1),
                    make_record(source, 2, "TRAIN-B", 8),
                    make_record(source, 3, "TRAIN-C", None),
                ],
                "val": [make_record(source, 4, "VAL-A", 1)],
                "test": [make_record(source, 5, "TEST-A", 8)],
            }
            output = root / "coco"
            config = ConversionConfig(
                source_dir=source,
                output_dir=output,
                seed=17,
                negative_ratio=1.0,
                negative_crop_size=16,
                negative_margin=2,
                negative_attempts=100,
                negative_min_std=1.0,
                minority_target=2,
                max_augmentations_per_image=2,
            )

            report = write_dataset(config, splits, quarantine=[], unused=[])

            self.assertTrue(report["validation"]["valid"])
            train = json.loads((output / "annotations" / "instances_train.json").read_text(encoding="utf-8"))
            manifest = json.loads((output / "reports" / "augmentation_manifest.json").read_text(encoding="utf-8"))
            split_manifest = json.loads((output / "reports" / "split_manifest.json").read_text(encoding="utf-8"))
            image_by_name = {item["file_name"]: item for item in train["images"]}
            boxes_by_image: dict[int, list[list[int]]] = {}
            for annotation in train["annotations"]:
                boxes_by_image.setdefault(annotation["image_id"], []).append(annotation["bbox"])

            minority_entries = [item for item in manifest if item["kind"] == "minority"]
            self.assertGreaterEqual(len(minority_entries), 1)
            for item in minority_entries:
                source_image = image_by_name[item["source_file"]]
                augmented_image = image_by_name[item["output_file"]]
                self.assertEqual(boxes_by_image[source_image["id"]], boxes_by_image[augmented_image["id"]])
                self.assertEqual((source_image["width"], source_image["height"]), (augmented_image["width"], augmented_image["height"]))

            negative_files = {item["file_name"] for item in split_manifest if item["kind"] == "negative"}
            self.assertGreaterEqual(len(negative_files), 1)
            for filename in negative_files:
                image_id = image_by_name[filename]["id"]
                self.assertNotIn(image_id, boxes_by_image)

            self.assertFalse(any(item["kind"] != "original" for item in split_manifest if item["split"] != "train"))
            for split_name in ("train", "val", "test"):
                for image_info in json.loads(
                    (output / "annotations" / f"instances_{split_name}.json").read_text(encoding="utf-8")
                )["images"]:
                    with Image.open(output / "images" / split_name / image_info["file_name"]) as image:
                        self.assertEqual(image.mode, "L")

            independent = validate_coco(output)
            self.assertTrue(independent["valid"], independent["errors"])


if __name__ == "__main__":
    unittest.main()
