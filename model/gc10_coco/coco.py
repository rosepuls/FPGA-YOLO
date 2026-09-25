from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
import shutil
import statistics
import uuid

from PIL import Image

from .augment import find_safe_crop, photometric_augment
from .source import Box, CATEGORY_NAMES, QuarantineEntry, SourceRecord, parse_source
from .split import SPLIT_NAMES, split_records, summarize_splits


def coco_bbox(box: Box) -> list[int]:
    return [box.xmin, box.ymin, box.xmax - box.xmin, box.ymax - box.ymin]


@dataclass(frozen=True)
class ConversionConfig:
    source_dir: Path
    output_dir: Path
    seed: int = 20260925
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15)
    overwrite: bool = False
    negative_ratio: float = 0.25
    negative_crop_size: int = 512
    negative_margin: int = 24
    negative_attempts: int = 100
    negative_min_std: float = 3.0
    minority_target: int | None = None
    max_augmentations_per_image: int = 3
    enable_negative_augmentation: bool = True
    enable_minority_augmentation: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_dir", Path(self.source_dir))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if self.negative_ratio < 0:
            raise ValueError("negative_ratio must be non-negative")
        if self.minority_target is not None and self.minority_target <= 0:
            raise ValueError("minority_target must be positive")
        if self.max_augmentations_per_image < 0:
            raise ValueError("max_augmentations_per_image must be non-negative")


def _categories() -> list[dict]:
    return [
        {"id": category_id, "name": name, "supercategory": "surface_defect"}
        for category_id, name in enumerate(CATEGORY_NAMES, start=1)
    ]


def _new_coco() -> dict:
    return {
        "info": {"description": "Cleaned GC10-DET", "version": "1.0"},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": _categories(),
    }


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


def _instance_counts(records: list[SourceRecord]) -> Counter[int]:
    counts: Counter[int] = Counter()
    for record in records:
        counts.update(box.category_id for box in record.boxes)
    return counts


def _validate_output_location(source: Path, output: Path) -> None:
    source_resolved = source.resolve()
    output_resolved = output.resolve()
    if source_resolved == output_resolved:
        raise ValueError("output directory must differ from source directory")
    if source_resolved.is_relative_to(output_resolved):
        raise ValueError("output directory cannot contain the source directory")
    if output_resolved.is_relative_to(source_resolved):
        raise ValueError("output directory cannot be inside the source directory")


def write_dataset(
    config: ConversionConfig,
    splits: dict[str, list[SourceRecord]],
    quarantine: list[QuarantineEntry],
    unused: list[Path],
) -> dict:
    source = config.source_dir
    output = config.output_dir
    _validate_output_location(source, output)
    if output.exists() and not config.overwrite:
        raise FileExistsError(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.name}.tmp-{uuid.uuid4().hex}")
    temporary.mkdir()

    try:
        for split_name in SPLIT_NAMES:
            (temporary / "images" / split_name).mkdir(parents=True)
        (temporary / "annotations").mkdir()
        (temporary / "reports").mkdir()

        coco = {split_name: _new_coco() for split_name in SPLIT_NAMES}
        split_manifest: list[dict] = []
        augmentation_manifest: list[dict] = []
        original_info: dict[str, dict] = {}
        next_image_id = 1
        next_annotation_id = 1

        def add_image(
            split_name: str,
            filename: str,
            image: Image.Image,
            boxes: tuple[Box, ...],
            *,
            source_file: str,
            material: str,
            kind: str,
        ) -> tuple[int, list[list[int]]]:
            nonlocal next_image_id, next_annotation_id
            destination = temporary / "images" / split_name / filename
            grayscale = image.convert("L")
            grayscale.save(destination, format="PNG", optimize=False)
            width, height = grayscale.size
            image_id = next_image_id
            next_image_id += 1
            coco[split_name]["images"].append(
                {"id": image_id, "file_name": filename, "width": width, "height": height}
            )
            written_boxes: list[list[int]] = []
            for box in boxes:
                bbox = coco_bbox(box)
                written_boxes.append(bbox)
                coco[split_name]["annotations"].append(
                    {
                        "id": next_annotation_id,
                        "image_id": image_id,
                        "category_id": box.category_id,
                        "bbox": bbox,
                        "area": bbox[2] * bbox[3],
                        "iscrowd": 0,
                    }
                )
                next_annotation_id += 1
            split_manifest.append(
                {
                    "file_name": filename,
                    "split": split_name,
                    "source_file": source_file,
                    "material_group": material,
                    "kind": kind,
                }
            )
            return image_id, written_boxes

        for split_name in SPLIT_NAMES:
            for record in sorted(splits.get(split_name, []), key=lambda item: item.output_stem):
                filename = f"{record.output_stem}.png"
                with Image.open(record.image_path) as source_image:
                    image_id, boxes = add_image(
                        split_name,
                        filename,
                        source_image,
                        record.boxes,
                        source_file=_relative_or_absolute(record.image_path, source),
                        material=record.material_group,
                        kind="original",
                    )
                original_info[record.output_stem] = {
                    "filename": filename,
                    "image_id": image_id,
                    "boxes": boxes,
                    "record": record,
                    "split": split_name,
                }

        rng = random.Random(config.seed)
        train_records = list(splits.get("train", []))
        minority_generated = 0
        if config.enable_minority_augmentation and config.max_augmentations_per_image > 0:
            counts = _instance_counts(train_records)
            present_counts = [counts[category] for category in sorted(counts) if counts[category] > 0]
            target = config.minority_target
            if target is None and present_counts:
                target = max(1, math.ceil(statistics.median(present_counts)))
            usage: Counter[str] = Counter()
            while target is not None:
                deficits = {category: target - count for category, count in counts.items() if count < target}
                if not deficits:
                    break
                candidates: list[tuple[int, str, SourceRecord]] = []
                for record in train_records:
                    if not record.boxes or usage[record.output_stem] >= config.max_augmentations_per_image:
                        continue
                    contribution = sum(
                        1 for box in record.boxes if deficits.get(box.category_id, 0) > 0
                    )
                    if contribution:
                        candidates.append((-contribution, record.output_stem, record))
                if not candidates:
                    break
                best_score = min(item[0] for item in candidates)
                top = [item[2] for item in candidates if item[0] == best_score]
                record = rng.choice(sorted(top, key=lambda item: item.output_stem))
                usage[record.output_stem] += 1
                with Image.open(record.image_path) as source_image:
                    augmented, transform = photometric_augment(source_image, rng, "minority")
                filename = f"{record.output_stem}__minority_{usage[record.output_stem]:02d}.png"
                _, boxes = add_image(
                    "train",
                    filename,
                    augmented,
                    record.boxes,
                    source_file=original_info[record.output_stem]["filename"],
                    material=record.material_group,
                    kind="minority",
                )
                counts.update(box.category_id for box in record.boxes)
                augmentation_manifest.append(
                    {
                        "kind": "minority",
                        "source_file": original_info[record.output_stem]["filename"],
                        "output_file": filename,
                        "material_group": record.material_group,
                        "transform": transform,
                        "boxes": boxes,
                    }
                )
                minority_generated += 1

        positive_records = [record for record in train_records if record.boxes]
        trusted_negatives = sum(not record.boxes for record in train_records)
        target_negatives = math.ceil(len(positive_records) * config.negative_ratio)
        negatives_needed = max(0, target_negatives - trusted_negatives)
        negative_generated = 0
        candidates = list(positive_records)
        rng.shuffle(candidates)
        while negative_generated < negatives_needed and candidates:
            produced = False
            for record in candidates:
                with Image.open(record.image_path) as source_image:
                    grayscale = source_image.convert("L")
                    crop_box = find_safe_crop(
                        grayscale,
                        record.boxes,
                        rng,
                        config.negative_crop_size,
                        config.negative_margin,
                        config.negative_attempts,
                        config.negative_min_std,
                    )
                    if crop_box is None:
                        continue
                    negative = grayscale.crop(crop_box)
                    if config.enable_negative_augmentation:
                        negative, transform = photometric_augment(negative, rng, "normal")
                    else:
                        transform = {"operation": "none"}
                negative_generated += 1
                filename = f"negative__{record.output_stem}__{negative_generated:05d}.png"
                add_image(
                    "train",
                    filename,
                    negative,
                    (),
                    source_file=original_info[record.output_stem]["filename"],
                    material=record.material_group,
                    kind="negative",
                )
                augmentation_manifest.append(
                    {
                        "kind": "negative",
                        "source_file": original_info[record.output_stem]["filename"],
                        "output_file": filename,
                        "material_group": record.material_group,
                        "crop": list(crop_box),
                        "transform": transform,
                        "boxes": [],
                    }
                )
                produced = True
                if negative_generated >= negatives_needed:
                    break
            if not produced:
                break

        for split_name in SPLIT_NAMES:
            _write_json(temporary / "annotations" / f"instances_{split_name}.json", coco[split_name])
        _write_json(temporary / "reports" / "split_manifest.json", split_manifest)
        _write_json(temporary / "reports" / "augmentation_manifest.json", augmentation_manifest)
        _write_json(
            temporary / "reports" / "quarantine.json",
            [
                {
                    "xml_path": _relative_or_absolute(item.xml_path, source),
                    "filename": item.filename,
                    "reason": item.reason,
                    "detail": item.detail,
                }
                for item in quarantine
            ],
        )
        _write_json(
            temporary / "reports" / "unused_images.json",
            [_relative_or_absolute(path, source) for path in unused],
        )

        validation = validate_coco(temporary)
        if not validation["valid"]:
            raise ValueError("generated COCO validation failed: " + "; ".join(validation["errors"]))
        report = {
            "source_xml": sum(len(items) for items in splits.values()) + len(quarantine),
            "accepted_source_images": sum(len(items) for items in splits.values()),
            "quarantined_records": len(quarantine),
            "unused_physical_images": len(unused),
            "split_summary": summarize_splits(splits),
            "minority_images_generated": minority_generated,
            "negative_images_generated": negative_generated,
            "validation": validation,
        }
        _write_json(temporary / "reports" / "audit_report.json", report)

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


def build_dataset(config: ConversionConfig) -> dict:
    records, quarantine, unused = parse_source(config.source_dir)
    splits = split_records(records, config.ratios, config.seed)
    return write_dataset(config, splits, quarantine, unused)


def validate_coco(dataset_root: str | Path) -> dict:
    root = Path(dataset_root)
    errors: list[str] = []
    image_ids: set[int] = set()
    annotation_ids: set[int] = set()
    image_names: set[str] = set()
    annotations_by_image: Counter[int] = Counter()
    image_lookup: dict[tuple[str, str], dict] = {}
    total_annotations = 0

    for split_name in SPLIT_NAMES:
        annotation_path = root / "annotations" / f"instances_{split_name}.json"
        if not annotation_path.is_file():
            errors.append(f"missing annotation file: {annotation_path}")
            continue
        try:
            dataset = json.loads(annotation_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            errors.append(f"unreadable annotation file {annotation_path}: {error}")
            continue
        category_ids = {item.get("id") for item in dataset.get("categories", [])}
        if category_ids != set(range(1, 11)):
            errors.append(f"invalid categories in {split_name}")
        local_image_ids: set[int] = set()
        for image_info in dataset.get("images", []):
            image_id = image_info.get("id")
            filename = image_info.get("file_name")
            if image_id in image_ids:
                errors.append(f"duplicate image id: {image_id}")
            if filename in image_names:
                errors.append(f"duplicate image filename: {filename}")
            image_ids.add(image_id)
            local_image_ids.add(image_id)
            image_names.add(filename)
            image_lookup[(split_name, filename)] = image_info
            image_path = root / "images" / split_name / filename
            if not image_path.is_file():
                errors.append(f"missing image: {image_path}")
                continue
            try:
                with Image.open(image_path) as image:
                    image.load()
                    if image.mode != "L":
                        errors.append(f"image is not grayscale: {image_path}")
                    if image.size != (image_info.get("width"), image_info.get("height")):
                        errors.append(f"image size mismatch: {image_path}")
            except OSError as error:
                errors.append(f"unreadable image {image_path}: {error}")
        for annotation in dataset.get("annotations", []):
            annotation_id = annotation.get("id")
            image_id = annotation.get("image_id")
            category_id = annotation.get("category_id")
            bbox = annotation.get("bbox")
            if annotation_id in annotation_ids:
                errors.append(f"duplicate annotation id: {annotation_id}")
            annotation_ids.add(annotation_id)
            if image_id not in local_image_ids:
                errors.append(f"annotation references unknown image: {annotation_id}")
                continue
            if category_id not in range(1, 11):
                errors.append(f"invalid category id: {category_id}")
            if not isinstance(bbox, list) or len(bbox) != 4 or bbox[2] <= 0 or bbox[3] <= 0:
                errors.append(f"invalid bbox: {annotation_id}")
                continue
            image_info = next(item for item in dataset["images"] if item["id"] == image_id)
            if bbox[0] < 0 or bbox[1] < 0 or bbox[0] + bbox[2] > image_info["width"] or bbox[1] + bbox[3] > image_info["height"]:
                errors.append(f"bbox out of bounds: {annotation_id}")
            annotations_by_image[image_id] += 1
            total_annotations += 1

    manifest_path = root / "reports" / "split_manifest.json"
    if not manifest_path.is_file():
        errors.append("missing split manifest")
        manifest: list[dict] = []
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    group_owners: dict[str, str] = {}
    for item in manifest:
        split_name = item.get("split")
        filename = item.get("file_name")
        kind = item.get("kind")
        group = item.get("material_group")
        if (split_name, filename) not in image_lookup:
            errors.append(f"manifest references unknown image: {split_name}/{filename}")
            continue
        previous = group_owners.setdefault(group, split_name)
        if previous != split_name:
            errors.append(f"material group crosses splits: {group}")
        if kind != "original" and split_name != "train":
            errors.append(f"augmentation outside train split: {filename}")
        if kind == "negative":
            image_id = image_lookup[(split_name, filename)]["id"]
            if annotations_by_image[image_id] != 0:
                errors.append(f"negative image has annotations: {filename}")

    return {
        "valid": not errors,
        "errors": errors,
        "images": len(image_ids),
        "annotations": total_annotations,
        "manifest_entries": len(manifest),
    }
