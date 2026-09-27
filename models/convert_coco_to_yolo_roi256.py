from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
import shutil
import uuid

from PIL import Image


SPLIT_NAMES = ("train", "val", "test")


@dataclass(frozen=True)
class ConversionConfig:
    source_dir: Path
    output_dir: Path
    overwrite: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_dir", Path(self.source_dir))
        object.__setattr__(self, "output_dir", Path(self.output_dir))


def coco_bbox_to_yolo(
    bbox: list[float], image_width: int, image_height: int
) -> tuple[float, float, float, float]:
    if image_width <= 0 or image_height <= 0 or len(bbox) != 4:
        raise ValueError("invalid COCO bbox")
    x, y, width, height = bbox
    if (
        not all(isinstance(value, (int, float)) and math.isfinite(value) for value in bbox)
        or width <= 0
        or height <= 0
        or x < 0
        or y < 0
        or x + width > image_width
        or y + height > image_height
    ):
        raise ValueError("invalid COCO bbox")
    return (
        (x + width / 2) / image_width,
        (y + height / 2) / image_height,
        width / image_width,
        height / image_height,
    )


def _read_split(source: Path, split_name: str) -> tuple[list[dict], dict[int, list[dict]]]:
    annotation_path = source / "annotations" / f"instances_{split_name}.json"
    try:
        dataset = json.loads(annotation_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"missing COCO annotations: {annotation_path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid COCO annotations: {annotation_path}") from error

    images = dataset.get("images")
    annotations = dataset.get("annotations")
    if not isinstance(images, list) or not isinstance(annotations, list):
        raise ValueError(f"invalid COCO structure: {annotation_path}")

    image_ids: set[int] = set()
    for image_info in images:
        image_id = image_info.get("id") if isinstance(image_info, dict) else None
        if not isinstance(image_id, int) or image_id in image_ids:
            raise ValueError(f"invalid image id in {annotation_path}")
        image_ids.add(image_id)

    annotations_by_image: dict[int, list[dict]] = defaultdict(list)
    for annotation in annotations:
        if not isinstance(annotation, dict):
            raise ValueError(f"invalid annotation in {annotation_path}")
        image_id = annotation.get("image_id")
        category_id = annotation.get("category_id")
        if image_id not in image_ids:
            raise ValueError(f"annotation references unknown image in {annotation_path}")
        if not isinstance(category_id, int) or category_id not in range(1, 11):
            raise ValueError(f"invalid category id in {annotation_path}")
        annotations_by_image[image_id].append(annotation)
    return images, annotations_by_image


def _validate_output_location(source: Path, output: Path) -> None:
    source_resolved = source.resolve()
    output_resolved = output.resolve()
    if source_resolved == output_resolved:
        raise ValueError("output directory must differ from source directory")
    if source_resolved.is_relative_to(output_resolved):
        raise ValueError("output directory cannot contain source directory")
    if output_resolved.is_relative_to(source_resolved):
        raise ValueError("output directory cannot be inside source directory")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def convert_dataset(config: ConversionConfig) -> dict:
    source = config.source_dir
    output = config.output_dir
    _validate_output_location(source, output)
    if output.exists() and not config.overwrite:
        raise FileExistsError(f"output already exists: {output}")
    if not source.is_dir():
        raise ValueError(f"source directory does not exist: {source}")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.name}.tmp-{uuid.uuid4().hex}")
    temporary.mkdir()
    try:
        image_count = 0
        annotation_count = 0
        for split_name in SPLIT_NAMES:
            images, annotations_by_image = _read_split(source, split_name)
            image_directory = temporary / "images" / split_name
            label_directory = temporary / "labels" / split_name
            image_directory.mkdir(parents=True)
            label_directory.mkdir(parents=True)

            for image_info in images:
                filename = image_info.get("file_name")
                image_id = image_info.get("id")
                width = image_info.get("width")
                height = image_info.get("height")
                if (
                    not isinstance(filename, str)
                    or Path(filename).name != filename
                    or not isinstance(width, int)
                    or not isinstance(height, int)
                    or width <= 0
                    or height <= 0
                ):
                    raise ValueError(f"invalid image metadata in instances_{split_name}.json")
                source_image_path = source / "images" / split_name / filename
                destination_image_path = image_directory / filename
                try:
                    with Image.open(source_image_path) as image:
                        image.load()
                        if image.mode != "L":
                            raise ValueError(f"source image is not grayscale: {source_image_path}")
                        if image.size != (width, height):
                            raise ValueError(f"source image size mismatch: {source_image_path}")
                        image.resize((256, 256), Image.Resampling.BILINEAR).save(
                            destination_image_path, format="PNG"
                        )
                except OSError as error:
                    raise ValueError(f"unreadable source image: {source_image_path}") from error

                labels: list[str] = []
                for annotation in annotations_by_image[image_id]:
                    bbox = annotation.get("bbox")
                    if not isinstance(bbox, list):
                        raise ValueError(f"invalid COCO bbox in instances_{split_name}.json")
                    xc, yc, box_width, box_height = coco_bbox_to_yolo(bbox, width, height)
                    labels.append(
                        f"{annotation['category_id'] - 1} {xc:.6f} {yc:.6f} {box_width:.6f} {box_height:.6f}\n"
                    )
                (label_directory / f"{Path(filename).stem}.txt").write_text("".join(labels), encoding="utf-8")
                image_count += 1
                annotation_count += len(labels)

        validation = validate_dataset(temporary)
        if not validation["valid"]:
            raise ValueError("generated YOLO validation failed: " + "; ".join(validation["errors"]))
        report = {
            "source": str(source.resolve()),
            "images": image_count,
            "annotations": annotation_count,
            "validation": validation,
        }
        _write_json(temporary / "reports" / "conversion_report.json", report)

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


def validate_dataset(root: Path) -> dict:
    root = Path(root)
    errors: list[str] = []
    image_count = 0
    label_count = 0
    annotation_count = 0

    for split_name in SPLIT_NAMES:
        image_directory = root / "images" / split_name
        label_directory = root / "labels" / split_name
        if not image_directory.is_dir():
            errors.append(f"missing image directory: {image_directory}")
            image_paths: list[Path] = []
        else:
            image_paths = sorted(path for path in image_directory.iterdir() if path.is_file())
        if not label_directory.is_dir():
            errors.append(f"missing label directory: {label_directory}")
            label_paths: list[Path] = []
        else:
            label_paths = sorted(path for path in label_directory.iterdir() if path.is_file())

        image_stems = {path.stem for path in image_paths}
        label_stems = {path.stem for path in label_paths if path.suffix == ".txt"}
        if image_stems != label_stems:
            errors.append(f"unpaired image and label stems in {split_name}")
        for label_path in label_paths:
            if label_path.suffix != ".txt":
                errors.append(f"label is not a text file: {label_path}")

        for image_path in image_paths:
            image_count += 1
            try:
                with Image.open(image_path) as image:
                    image.load()
                    if image.mode != "L" or image.size != (256, 256):
                        errors.append(f"invalid output image: {image_path}")
            except OSError as error:
                errors.append(f"unreadable output image {image_path}: {error}")

        for label_path in label_paths:
            if label_path.suffix != ".txt":
                continue
            label_count += 1
            try:
                lines = label_path.read_text(encoding="utf-8").splitlines()
            except OSError as error:
                errors.append(f"unreadable label file {label_path}: {error}")
                continue
            for line_number, line in enumerate(lines, start=1):
                fields = line.split()
                if len(fields) != 5:
                    errors.append(f"invalid label syntax: {label_path}:{line_number}")
                    continue
                try:
                    class_id = int(fields[0])
                    xc, yc, width, height = (float(value) for value in fields[1:])
                except ValueError:
                    errors.append(f"invalid label values: {label_path}:{line_number}")
                    continue
                if class_id not in range(10):
                    errors.append(f"invalid class id: {label_path}:{line_number}")
                    continue
                geometry = (xc, yc, width, height)
                if (
                    not all(math.isfinite(value) for value in geometry)
                    or width <= 0
                    or height <= 0
                    or xc < 0
                    or yc < 0
                    or xc > 1
                    or yc > 1
                    or xc - width / 2 < 0
                    or yc - height / 2 < 0
                    or xc + width / 2 > 1
                    or yc + height / 2 > 1
                ):
                    errors.append(f"invalid normalized geometry: {label_path}:{line_number}")
                    continue
                annotation_count += 1

    return {
        "valid": not errors,
        "errors": errors,
        "images": image_count,
        "labels": label_count,
        "annotations": annotation_count,
    }
