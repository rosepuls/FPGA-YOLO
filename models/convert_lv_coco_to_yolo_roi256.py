from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import json
import math
from pathlib import Path
import random
import re
import shutil
import sys
from typing import Iterable, Sequence
import uuid

from PIL import Image


SPLITS = ("train", "val", "test")
SOURCE_SPLITS = ("train", "val")
CATEGORY_IDS = (1, 2, 3, 4)
SERIALIZATION_SCALE = 1_000_000
DEFAULT_SEED = 20260928
GROUP_AUGMENTATION_SUFFIX = re.compile(r"_aug_\d+$")


@dataclass(frozen=True)
class Annotation:
    category_id: int
    bbox: tuple[float, float, float, float]


@dataclass(frozen=True)
class SourceRecord:
    source_path: Path
    source_split: str
    file_name: str
    width: int
    height: int
    annotations: tuple[Annotation, ...]
    group_key: str

    @property
    def output_stem(self) -> str:
        return f"{self.source_split}__{Path(self.file_name).stem}"


@dataclass(frozen=True)
class ConversionConfig:
    source_dir: Path
    output_dir: Path
    seed: int = DEFAULT_SEED
    overwrite: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_dir", Path(self.source_dir))
        object.__setattr__(self, "output_dir", Path(self.output_dir))


def group_key(file_name: str) -> str:
    path = Path(file_name)
    if path.name != file_name or not path.stem:
        raise ValueError(f"invalid source filename: {file_name!r}")
    return GROUP_AUGMENTATION_SUFFIX.sub("", path.stem)


def _read_categories(payload: dict, annotation_path: Path) -> dict[int, str]:
    categories = payload.get("categories")
    if not isinstance(categories, list):
        raise ValueError(f"missing categories: {annotation_path}")

    result: dict[int, str] = {}
    for category in categories:
        if not isinstance(category, dict):
            raise ValueError(f"invalid category: {annotation_path}")
        category_id = category.get("id")
        name = category.get("name")
        if not isinstance(category_id, int) or not isinstance(name, str) or not name:
            raise ValueError(f"invalid category: {annotation_path}")
        if category_id in result:
            raise ValueError(f"duplicate category id: {annotation_path}")
        result[category_id] = name

    if tuple(sorted(result)) != CATEGORY_IDS:
        raise ValueError(f"categories must use IDs {CATEGORY_IDS}: {annotation_path}")
    return result


def _valid_bbox(bbox: object, width: int, height: int) -> tuple[float, float, float, float]:
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ValueError("invalid COCO bbox")
    values = tuple(float(value) for value in bbox)
    x, y, box_width, box_height = values
    if (
        not all(math.isfinite(value) for value in values)
        or box_width <= 0
        or box_height <= 0
        or x < 0
        or y < 0
        or x + box_width > width
        or y + box_height > height
    ):
        raise ValueError("invalid COCO bbox")
    return values


def _read_valid_split(source_dir: Path, split_name: str) -> tuple[list[SourceRecord], dict[int, str]]:
    annotation_path = source_dir / "annotations" / f"instances_{split_name}2017.json"
    try:
        payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"missing COCO annotations: {annotation_path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid COCO JSON: {annotation_path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"invalid COCO structure: {annotation_path}")

    categories = _read_categories(payload, annotation_path)
    images = payload.get("images")
    annotations = payload.get("annotations")
    if not isinstance(images, list) or not isinstance(annotations, list):
        raise ValueError(f"invalid COCO structure: {annotation_path}")

    image_info: dict[int, dict] = {}
    for item in images:
        if not isinstance(item, dict):
            raise ValueError(f"invalid image record: {annotation_path}")
        image_id = item.get("id")
        file_name = item.get("file_name")
        width = item.get("width")
        height = item.get("height")
        if (
            not isinstance(image_id, int)
            or image_id in image_info
            or not isinstance(file_name, str)
            or Path(file_name).name != file_name
            or not isinstance(width, int)
            or not isinstance(height, int)
            or width <= 0
            or height <= 0
        ):
            raise ValueError(f"invalid image record: {annotation_path}")
        source_path = source_dir / f"{split_name}2017" / file_name
        if not source_path.is_file():
            raise ValueError(f"missing source image: {source_path}")
        image_info[image_id] = item

    annotations_by_image: dict[int, list[Annotation]] = defaultdict(list)
    annotation_ids: set[int] = set()
    for item in annotations:
        if not isinstance(item, dict):
            raise ValueError(f"invalid annotation: {annotation_path}")
        annotation_id = item.get("id")
        image_id = item.get("image_id")
        category_id = item.get("category_id")
        if (
            not isinstance(annotation_id, int)
            or annotation_id in annotation_ids
            or image_id not in image_info
            or category_id not in CATEGORY_IDS
        ):
            raise ValueError(f"invalid annotation: {annotation_path}")
        annotation_ids.add(annotation_id)
        image = image_info[image_id]
        annotations_by_image[image_id].append(
            Annotation(category_id, _valid_bbox(item.get("bbox"), image["width"], image["height"]))
        )

    raw_records = [
        SourceRecord(
            source_path=source_dir / f"{split_name}2017" / item["file_name"],
            source_split=split_name,
            file_name=item["file_name"],
            width=item["width"],
            height=item["height"],
            annotations=tuple(annotations_by_image[image_id]),
            group_key=group_key(item["file_name"]),
        )
        for image_id, item in image_info.items()
    ]
    records_by_path: dict[Path, SourceRecord] = {}
    for record in raw_records:
        existing = records_by_path.get(record.source_path)
        if existing is None:
            records_by_path[record.source_path] = record
            continue
        if (existing.width, existing.height, existing.group_key) != (
            record.width,
            record.height,
            record.group_key,
        ):
            raise ValueError(f"inconsistent duplicate image record: {record.source_path}")
        records_by_path[record.source_path] = SourceRecord(
            source_path=existing.source_path,
            source_split=existing.source_split,
            file_name=existing.file_name,
            width=existing.width,
            height=existing.height,
            annotations=existing.annotations + record.annotations,
            group_key=existing.group_key,
        )
    return list(records_by_path.values()), categories


def load_source_records(source_dir: Path) -> tuple[list[SourceRecord], dict[int, str]]:
    source_dir = Path(source_dir)
    if not source_dir.is_dir():
        raise ValueError(f"source directory does not exist: {source_dir}")

    records: list[SourceRecord] = []
    categories: dict[int, str] | None = None
    for split_name in SOURCE_SPLITS:
        split_records, split_categories = _read_valid_split(source_dir, split_name)
        if categories is None:
            categories = split_categories
        elif categories != split_categories:
            raise ValueError("train and val categories differ")
        records.extend(split_records)

    if categories is None or not records:
        raise ValueError("source dataset is empty")

    source_paths = [record.source_path for record in records]
    output_stems = [record.output_stem for record in records]
    if len(set(source_paths)) != len(source_paths) or len(set(output_stems)) != len(output_stems):
        raise ValueError("source filenames would collide in output")
    return records, categories


def _group_records(records: Iterable[SourceRecord]) -> dict[str, list[SourceRecord]]:
    groups: dict[str, list[SourceRecord]] = defaultdict(list)
    for record in records:
        groups[record.group_key].append(record)
    return dict(groups)


def _class_counts(records: Iterable[SourceRecord]) -> Counter[int]:
    return Counter(
        annotation.category_id
        for record in records
        for annotation in record.annotations
    )


def _assign_group(
    assigned: dict[str, list[SourceRecord]],
    split_name: str,
    group_name: str,
    groups: dict[str, list[SourceRecord]],
    remaining: set[str],
) -> None:
    assigned[split_name].extend(groups[group_name])
    remaining.remove(group_name)


def assign_splits(records: list[SourceRecord], seed: int) -> dict[str, list[SourceRecord]]:
    groups = _group_records(records)
    if len(groups) < len(SPLITS):
        raise ValueError("fewer source groups than output splits")

    generator = random.Random(seed)
    group_names = sorted(groups)
    generator.shuffle(group_names)
    order = {group_name: index for index, group_name in enumerate(group_names)}
    remaining = set(group_names)
    assigned = {split_name: [] for split_name in SPLITS}

    for split_name in ("val", "test"):
        for category_id in CATEGORY_IDS:
            if _class_counts(assigned[split_name])[category_id]:
                continue
            candidates = [
                group_name
                for group_name in remaining
                if _class_counts(groups[group_name])[category_id] > 0
            ]
            if not candidates:
                raise ValueError(f"cannot provide class {category_id} to {split_name}")
            selected = min(
                candidates,
                key=lambda group_name: (len(groups[group_name]), order[group_name]),
            )
            _assign_group(assigned, split_name, selected, groups, remaining)

    total_images = len(records)
    total_classes = _class_counts(records)
    ratios = {"train": 0.8, "val": 0.1, "test": 0.1}
    target_images = {split_name: total_images * ratio for split_name, ratio in ratios.items()}
    target_classes = {
        split_name: {category_id: total_classes[category_id] * ratio for category_id in CATEGORY_IDS}
        for split_name, ratio in ratios.items()
    }

    for group_name in sorted(remaining, key=order.get):
        group_records = groups[group_name]
        group_classes = _class_counts(group_records)

        def score(split_name: str) -> tuple[float, float, str]:
            current_classes = _class_counts(assigned[split_name])
            class_gain = sum(
                min(
                    group_classes[category_id],
                    max(0.0, target_classes[split_name][category_id] - current_classes[category_id]),
                )
                / max(1.0, target_classes[split_name][category_id])
                for category_id in CATEGORY_IDS
            )
            image_gain = min(
                len(group_records),
                max(0.0, target_images[split_name] - len(assigned[split_name])),
            ) / max(1.0, target_images[split_name])
            return (class_gain + image_gain * 0.25, image_gain, split_name)

        selected_split = max(SPLITS, key=score)
        _assign_group(assigned, selected_split, group_name, groups, remaining)

    split_groups = {
        split_name: {record.group_key for record in split_records}
        for split_name, split_records in assigned.items()
    }
    if any(split_groups[first] & split_groups[second] for first, second in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("augmentation group split leakage")
    for split_name, split_records in assigned.items():
        missing = set(CATEGORY_IDS) - set(_class_counts(split_records))
        if missing:
            raise ValueError(f"split {split_name} is missing classes: {sorted(missing)}")
    return assigned


def gray_u8(image: Image.Image) -> Image.Image:
    if image.mode == "L":
        return image.copy()
    if image.mode != "RGB":
        raise ValueError(f"unsupported source image mode: {image.mode}")
    return image.convert("L", (77 / 256, 150 / 256, 29 / 256, 0))


def _fit_center_units(center_units: int, size_units: int) -> int:
    minimum = (size_units + 1) // 2
    maximum = (2 * SERIALIZATION_SCALE - size_units) // 2
    return min(max(center_units, minimum), maximum)


def format_yolo_label(
    category_id: int,
    bbox: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
) -> str:
    if category_id not in CATEGORY_IDS or image_width <= 0 or image_height <= 0:
        raise ValueError("invalid label inputs")
    x, y, width, height = _valid_bbox(list(bbox), image_width, image_height)
    values = (
        (x + width / 2) / image_width,
        (y + height / 2) / image_height,
        width / image_width,
        height / image_height,
    )
    center_x, center_y, box_width, box_height = (
        round(value * SERIALIZATION_SCALE) for value in values
    )
    if not 0 < box_width <= SERIALIZATION_SCALE or not 0 < box_height <= SERIALIZATION_SCALE:
        raise ValueError("invalid normalized bbox")
    center_x = _fit_center_units(center_x, box_width)
    center_y = _fit_center_units(center_y, box_height)
    return (
        f"{category_id - 1} {center_x / SERIALIZATION_SCALE:.6f} "
        f"{center_y / SERIALIZATION_SCALE:.6f} {box_width / SERIALIZATION_SCALE:.6f} "
        f"{box_height / SERIALIZATION_SCALE:.6f}\n"
    )


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _validate_output_location(source: Path, output: Path) -> None:
    source = source.resolve()
    output = output.resolve()
    if source == output or source.is_relative_to(output) or output.is_relative_to(source):
        raise ValueError("output directory must be separate from source directory")


def _parse_label_line(line: str) -> tuple[int, Decimal, Decimal, Decimal, Decimal]:
    fields = line.split()
    if len(fields) != 5:
        raise ValueError("label must have five fields")
    try:
        class_id = int(fields[0])
        coordinates = tuple(Decimal(value) for value in fields[1:])
    except (InvalidOperation, ValueError) as error:
        raise ValueError("invalid label value") from error
    if class_id not in range(4) or not all(value.is_finite() for value in coordinates):
        raise ValueError("invalid label value")
    center_x, center_y, width, height = coordinates
    if (
        width <= 0
        or height <= 0
        or center_x - width / 2 < 0
        or center_y - height / 2 < 0
        or center_x + width / 2 > 1
        or center_y + height / 2 > 1
    ):
        raise ValueError("invalid label geometry")
    return (class_id, center_x, center_y, width, height)


def validate_dataset(root: Path) -> dict[str, object]:
    root = Path(root)
    errors: list[str] = []
    images_total = 0
    labels_total = 0
    annotations_total = 0
    class_counts: dict[str, dict[str, int]] = {}

    for split_name in SPLITS:
        image_dir = root / "images" / split_name
        label_dir = root / "labels" / split_name
        images = sorted(image_dir.glob("*.png")) if image_dir.is_dir() else []
        labels = sorted(label_dir.glob("*.txt")) if label_dir.is_dir() else []
        if not image_dir.is_dir() or not label_dir.is_dir():
            errors.append(f"missing output split directories: {split_name}")
            continue
        if {path.stem for path in images} != {path.stem for path in labels}:
            errors.append(f"unpaired output images and labels: {split_name}")

        split_classes = Counter()
        for image_path in images:
            images_total += 1
            try:
                with Image.open(image_path) as image:
                    image.load()
                    if image.mode != "L" or image.size != (256, 256):
                        errors.append(f"invalid output image: {image_path.name}")
            except OSError:
                errors.append(f"unreadable output image: {image_path.name}")

        for label_path in labels:
            labels_total += 1
            try:
                lines = label_path.read_text(encoding="utf-8").splitlines()
                for line in lines:
                    class_id, *_ = _parse_label_line(line)
                    split_classes[class_id] += 1
                    annotations_total += 1
            except (OSError, ValueError) as error:
                errors.append(f"invalid output label {label_path.name}: {error}")
        if set(split_classes) != set(range(4)):
            errors.append(f"output split missing classes: {split_name}")
        class_counts[split_name] = {str(class_id): split_classes[class_id] for class_id in range(4)}

    report_path = root / "reports" / "conversion_report.json"
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        groups = report.get("group_assignments")
        if not isinstance(groups, dict) or any(split_name not in SPLITS for split_name in groups.values()):
            errors.append("invalid group assignment report")
    except (OSError, json.JSONDecodeError):
        errors.append("missing conversion report")

    return {
        "valid": not errors,
        "errors": errors,
        "images": images_total,
        "labels": labels_total,
        "annotations": annotations_total,
        "class_counts": class_counts,
    }


def convert_dataset(config: ConversionConfig) -> dict[str, object]:
    source = config.source_dir
    output = config.output_dir
    _validate_output_location(source, output)
    if output.exists() and not config.overwrite:
        raise FileExistsError(f"output directory already exists: {output}")

    records, categories = load_source_records(source)
    assignments = assign_splits(records, config.seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.name}.tmp-{uuid.uuid4().hex}")
    temporary.mkdir()

    try:
        for split_name, split_records in assignments.items():
            image_dir = temporary / "images" / split_name
            label_dir = temporary / "labels" / split_name
            image_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)
            for record in sorted(split_records, key=lambda item: item.file_name):
                image_path = image_dir / f"{record.output_stem}.png"
                label_path = label_dir / f"{record.output_stem}.txt"
                try:
                    with Image.open(record.source_path) as source_image:
                        source_image.load()
                        if source_image.size != (record.width, record.height):
                            raise ValueError(f"source image size mismatch: {record.source_path.name}")
                        grayscale = gray_u8(source_image)
                        grayscale.resize((256, 256), Image.Resampling.BILINEAR).save(
                            image_path, format="PNG"
                        )
                except OSError as error:
                    raise ValueError(f"unreadable source image: {record.source_path.name}") from error
                labels = [
                    format_yolo_label(
                        annotation.category_id,
                        annotation.bbox,
                        record.width,
                        record.height,
                    )
                    for annotation in record.annotations
                ]
                label_path.write_text("".join(labels), encoding="utf-8")

        report = {
            "source": str(source.resolve()),
            "source_splits": list(SOURCE_SPLITS),
            "excluded_source_split": "test2017 (incomplete image references)",
            "seed": config.seed,
            "categories": {str(category_id - 1): categories[category_id] for category_id in CATEGORY_IDS},
            "group_assignments": {
                group_name: split_name
                for split_name, split_records in assignments.items()
                for group_name in sorted({record.group_key for record in split_records})
            },
            "split_summary": {
                split_name: {
                    "images": len(split_records),
                    "annotations": sum(len(record.annotations) for record in split_records),
                    "class_counts": {
                        str(category_id - 1): _class_counts(split_records)[category_id]
                        for category_id in CATEGORY_IDS
                    },
                }
                for split_name, split_records in assignments.items()
            },
        }
        report_path = temporary / "reports" / "conversion_report.json"
        _write_json(report_path, report)
        validation = validate_dataset(temporary)
        if not validation["valid"]:
            raise ValueError("generated dataset validation failed: " + "; ".join(validation["errors"]))
        report["validation"] = validation
        _write_json(report_path, report)

        backup: Path | None = None
        if output.exists():
            backup = output.with_name(f"{output.name}.backup-{uuid.uuid4().hex}")
            output.rename(backup)
        try:
            temporary.rename(output)
        except Exception:
            if backup is not None and not output.exists():
                backup.rename(output)
            raise
        if backup is not None:
            shutil.rmtree(backup)
        return report
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert LV-COCO into a grouped-split 256x256 grayscale YOLO dataset."
    )
    parser.add_argument("--source", type=Path, required=False, default=Path("data/LV-COCO"))
    parser.add_argument("--output", type=Path, default=Path("data/LV-YOLO-ROI256"))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        if arguments.validate is not None:
            print(json.dumps(validate_dataset(arguments.validate), ensure_ascii=False, indent=2))
            return 0 if validate_dataset(arguments.validate)["valid"] else 1
        report = convert_dataset(
            ConversionConfig(
                source_dir=arguments.source,
                output_dir=arguments.output,
                seed=arguments.seed,
                overwrite=arguments.overwrite,
            )
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except Exception as error:
        print(json.dumps({"error": type(error).__name__, "message": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
