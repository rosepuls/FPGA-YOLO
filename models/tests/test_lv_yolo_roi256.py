from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from models.convert_lv_coco_to_yolo_roi256 import (
    ConversionConfig,
    assign_splits,
    convert_dataset,
    format_yolo_label,
    gray_u8,
    load_source_records,
)


CATEGORIES = [
    {"id": 1, "name": "ca_shang"},
    {"id": 2, "name": "zang_wu"},
    {"id": 3, "name": "zhe_zhou"},
    {"id": 4, "name": "zhen_kong"},
]


def temporary_workspace_directory() -> tempfile.TemporaryDirectory[str]:
    return tempfile.TemporaryDirectory(
        dir=Path(__file__).parent,
        prefix="lv-yolo-fixture-",
    )


def make_lv_source(
    parent: Path,
    split_records: dict[str, list[tuple[str, int | None, str]]],
    *,
    create_images: bool = True,
) -> Path:
    source = parent / "LV-COCO"
    annotation_dir = source / "annotations"
    annotation_dir.mkdir(parents=True)

    for split_name in ("train", "val"):
        images = []
        annotations = []
        image_dir = source / f"{split_name}2017"
        image_dir.mkdir()
        for image_id, (filename, category_id, mode) in enumerate(
            split_records.get(split_name, []), start=1
        ):
            images.append(
                {"id": image_id, "file_name": filename, "width": 640, "height": 480}
            )
            if create_images:
                color = (1, 2, 3) if mode == "RGB" else 17
                Image.new(mode, (640, 480), color).save(image_dir / filename)
            if category_id is not None:
                annotations.append(
                    {
                        "id": image_id,
                        "image_id": image_id,
                        "category_id": category_id,
                        "bbox": [0, 0, 64, 48],
                    }
                )

        payload = {"images": images, "annotations": annotations, "categories": CATEGORIES}
        (annotation_dir / f"instances_{split_name}2017.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
    return source


def make_complete_four_class_source(parent: Path) -> Path:
    records = {
        split_name: [
            (f"class_{category_id}_{index}.jpg", category_id, "RGB" if index % 2 else "L")
            for category_id in range(1, 5)
            for index in range(start, start + 3)
        ]
        for split_name, start in (("train", 0), ("val", 3))
    }
    return make_lv_source(parent, records)


class LvYoloRoi256Tests(unittest.TestCase):
    def test_load_source_records_merges_train_val_and_groups_augmented_files(self) -> None:
        with temporary_workspace_directory() as temporary:
            source = make_lv_source(
                Path(temporary),
                {
                    "train": [("part.jpg", 1, "RGB"), ("part_aug_7.jpg", 2, "L")],
                    "val": [("other.jpg", 3, "RGB")],
                },
            )

            records, categories = load_source_records(source)

        self.assertEqual(len(records), 3)
        self.assertEqual({record.group_key for record in records}, {"part", "other"})
        self.assertEqual(categories, {1: "ca_shang", 2: "zang_wu", 3: "zhe_zhou", 4: "zhen_kong"})

    def test_load_source_records_rejects_missing_referenced_image(self) -> None:
        with temporary_workspace_directory() as temporary:
            source = make_lv_source(
                Path(temporary),
                {"train": [("missing.jpg", 1, "RGB")], "val": []},
                create_images=False,
            )

            with self.assertRaisesRegex(ValueError, "missing source image"):
                load_source_records(source)

    def test_load_source_records_merges_duplicate_source_image_records(self) -> None:
        with temporary_workspace_directory() as temporary:
            source = make_lv_source(
                Path(temporary),
                {
                    "train": [("same.jpg", 1, "RGB"), ("same.jpg", 2, "RGB")],
                    "val": [("other.jpg", 3, "L")],
                },
            )

            records, _ = load_source_records(source)

        merged = next(record for record in records if record.file_name == "same.jpg")
        self.assertEqual(len(records), 2)
        self.assertEqual({annotation.category_id for annotation in merged.annotations}, {1, 2})

    def test_assign_splits_is_repeatable_and_keeps_augmented_group_together(self) -> None:
        with temporary_workspace_directory() as temporary:
            source = make_complete_four_class_source(Path(temporary))
            extra_path = source / "train2017" / "class_1_0_aug_1.jpg"
            Image.new("RGB", (640, 480), (1, 2, 3)).save(extra_path)
            data_path = source / "annotations" / "instances_train2017.json"
            payload = json.loads(data_path.read_text(encoding="utf-8"))
            payload["images"].append(
                {"id": 99, "file_name": extra_path.name, "width": 640, "height": 480}
            )
            payload["annotations"].append(
                {"id": 99, "image_id": 99, "category_id": 1, "bbox": [0, 0, 64, 48]}
            )
            data_path.write_text(json.dumps(payload), encoding="utf-8")
            records, _ = load_source_records(source)

        first = assign_splits(records, seed=20260928)
        second = assign_splits(records, seed=20260928)
        self.assertEqual(
            {split: [record.file_name for record in values] for split, values in first.items()},
            {split: [record.file_name for record in values] for split, values in second.items()},
        )
        group_splits = {}
        for split_name, values in first.items():
            for record in values:
                group_splits.setdefault(record.group_key, set()).add(split_name)
        self.assertTrue(all(len(splits) == 1 for splits in group_splits.values()))
        for values in first.values():
            self.assertEqual(
                {annotation.category_id for record in values for annotation in record.annotations},
                {1, 2, 3, 4},
            )

    def test_gray_u8_uses_the_project_fixed_point_rgb_formula(self) -> None:
        image = Image.new("RGB", (1, 1), (1, 2, 3))

        self.assertEqual(
            list(gray_u8(image).get_flattened_data()),
            [(77 + 300 + 87 + 128) >> 8],
        )

    def test_format_yolo_label_maps_coco_one_to_yolo_zero(self) -> None:
        self.assertEqual(
            format_yolo_label(1, (0, 0, 64, 48), 640, 480),
            "0 0.050000 0.050000 0.100000 0.100000\n",
        )

    def test_convert_dataset_writes_valid_grayscale_pngs_labels_and_report(self) -> None:
        with temporary_workspace_directory() as temporary:
            root = Path(temporary)
            output = root / "LV-YOLO-ROI256"
            report = convert_dataset(
                ConversionConfig(make_complete_four_class_source(root), output, seed=20260928)
            )

            self.assertTrue(report["validation"]["valid"])
            self.assertTrue((output / "reports" / "conversion_report.json").is_file())
            for image_path in (output / "images").glob("*/*.png"):
                with Image.open(image_path) as image:
                    self.assertEqual((image.mode, image.size), ("L", (256, 256)))
                self.assertTrue(
                    (output / "labels" / image_path.parent.name / f"{image_path.stem}.txt").is_file()
                )

    def test_convert_dataset_refuses_to_replace_existing_output_without_overwrite(self) -> None:
        with temporary_workspace_directory() as temporary:
            root = Path(temporary)
            output = root / "LV-YOLO-ROI256"
            config = ConversionConfig(make_complete_four_class_source(root), output, seed=20260928)
            convert_dataset(config)

            with self.assertRaises(FileExistsError):
                convert_dataset(config)

    def test_convert_dataset_disambiguates_same_filename_from_source_splits(self) -> None:
        with temporary_workspace_directory() as temporary:
            root = Path(temporary)
            source = make_complete_four_class_source(root)
            val_annotation_path = source / "annotations" / "instances_val2017.json"
            payload = json.loads(val_annotation_path.read_text(encoding="utf-8"))
            payload["images"][0]["file_name"] = "class_1_0.jpg"
            Image.new("RGB", (640, 480), (3, 2, 1)).save(
                source / "val2017" / "class_1_0.jpg"
            )
            val_annotation_path.write_text(json.dumps(payload), encoding="utf-8")

            output = root / "LV-YOLO-ROI256"
            convert_dataset(ConversionConfig(source, output, seed=20260928))

            output_images = list((output / "images").glob("*/*.png"))
            self.assertEqual(len(output_images), 24)
            self.assertEqual(len({image_path.stem for image_path in output_images}), 24)


if __name__ == "__main__":
    unittest.main()
